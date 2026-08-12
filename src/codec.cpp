#include "codec.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
	// A normalised step, not a raw gradient scale: every weight moves by about this much per
	// update regardless of layer, so it is far smaller than a plain SGD rate would be.
	constexpr float learning_rate = 0.002f;

	double psnr_from_rmse(const double rmse)
	{
		return rmse == 0.0 ? std::numeric_limits<double>::infinity() : 20.0 * std::log10(255.0 / rmse);
	}

	// Coarser steps for higher spatial frequencies, applied identically by both ends.
	float frequency_scale(const int u, const int v)
	{
		return 1.0f + 0.12f * static_cast<float>(u + v);
	}

	// Importance in [32,255] mapped onto a bounded gain. Bounded matters: an unbounded
	// preference for face blocks starves every other block that needs repair.
	double importance_gain(const double average_importance)
	{
		return 1.0 + 7.0 * std::clamp((average_importance - 32.0) / 223.0, 0.0, 1.0);
	}

	// Packets a unit with visible error may be deferred before it is promoted ahead of every
	// unit that is still inside its deadline. Importance decides order, never whether a unit
	// is served at all, which is what stops a moving face from leaving copies of itself behind.
	constexpr uint16_t starvation_deadline = 8;

	constexpr int max_motion = 16;

	// Squared pixel error the transform codec will accept to save one byte, in units of the
	// block's own quantiser squared. This is the Lagrange multiplier of the usual
	// rate-distortion formulation, and it is what lets a well predicted block send nothing.
	constexpr double rd_lambda = 2.0;

	// Three-step search seeded with the previous block's vector, which is nearly free because
	// neighbouring blocks of one moving head share a vector. 32 comparisons, not 1,089.
	point_i search_motion(const image_u8& source, const image_u8& reference,
		const int block_x, const int block_y, const point_i seed)
	{
		const auto cost = [&](const int dx, const int dy)
		{
			int total = 0;
			for (int y = 0; y < transform_codec::block_size; ++y)
				for (int x = 0; x < transform_codec::block_size; ++x)
				{
					const int sx = std::min(block_x + x, source.width() - 1);
					const int sy = std::min(block_y + y, source.height() - 1);
					const int rx = std::clamp(block_x + x + dx, 0, reference.width() - 1);
					const int ry = std::clamp(block_y + y + dy, 0, reference.height() - 1);
					total += std::abs(static_cast<int>(source(sx, sy)) - reference(rx, ry));
				}
			return total;
		};

		point_i best{0, 0};
		int best_cost = cost(0, 0);
		if (seed.x != 0 || seed.y != 0)
		{
			const int seeded = cost(seed.x, seed.y);
			if (seeded < best_cost)
			{
				best_cost = seeded;
				best = seed;
			}
		}
		for (int step = 8; step >= 1; step /= 2)
		{
			const point_i centre = best;
			for (int dy = -1; dy <= 1; ++dy)
				for (int dx = -1; dx <= 1; ++dx)
				{
					if (dx == 0 && dy == 0) continue;
					const point_i trial{centre.x + dx * step, centre.y + dy * step};
					if (std::abs(trial.x) > max_motion || std::abs(trial.y) > max_motion) continue;
					const int trial_cost = cost(trial.x, trial.y);
					if (trial_cost < best_cost)
					{
						best_cost = trial_cost;
						best = trial;
					}
				}
		}
		return best;
	}

	void append_u16(std::vector<uint8_t>& bytes, const int value)
	{
		bytes.push_back(static_cast<uint8_t>(value & 0xff));
		bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
	}

	bool read_u16(const std::vector<uint8_t>& bytes, size_t& offset, int& value)
	{
		if (offset + 2 > bytes.size()) return false;
		value = bytes[offset] | static_cast<int>(bytes[offset + 1]) << 8;
		offset += 2;
		return true;
	}

	void append_varuint(std::vector<uint8_t>& bytes, uint32_t value)
	{
		do
		{
			uint8_t byte = static_cast<uint8_t>(value & 0x7f);
			value >>= 7;
			if (value != 0) byte |= 0x80;
			bytes.push_back(byte);
		}
		while (value != 0);
	}

	bool read_varuint(const std::vector<uint8_t>& bytes, size_t& offset, uint32_t& value)
	{
		value = 0;
		for (int shift = 0; shift <= 28; shift += 7)
		{
			if (offset >= bytes.size()) return false;
			const uint8_t byte = bytes[offset++];
			value |= static_cast<uint32_t>(byte & 0x7f) << shift;
			if ((byte & 0x80) == 0) return true;
		}
		return false;
	}

	uint32_t zigzag_encode(const int value)
	{
		return static_cast<uint32_t>(value >= 0 ? value * 2 : -value * 2 - 1);
	}

	int zigzag_decode(const uint32_t value)
	{
		return (value & 1) == 0 ? static_cast<int>(value / 2) : -static_cast<int>((value + 1) / 2);
	}

	void append_packet_header(std::vector<uint8_t>& bytes, const uint8_t kind, const uint8_t version,
		const bool keyframe, const int width, const int height, const int unit_count)
	{
		bytes = {'L', kind, version, static_cast<uint8_t>(keyframe ? 1 : 0)};
		append_u16(bytes, width);
		append_u16(bytes, height);
		append_u16(bytes, unit_count);
	}

	size_t varuint_size(uint32_t value)
	{
		size_t bytes = 1;
		while (value >= 0x80)
		{
			value >>= 7;
			++bytes;
		}
		return bytes;
	}

	// Separable orthonormal 8x8 DCT basis, built once and shared by both directions.
	struct dct_table
	{
		float basis[transform_codec::block_size][transform_codec::block_size];

		dct_table()
		{
			constexpr float pi = 3.14159265358979323846f;
			for (int frequency = 0; frequency < transform_codec::block_size; ++frequency)
			{
				const float scale = frequency == 0 ? std::sqrt(1.0f / transform_codec::block_size)
					: std::sqrt(2.0f / transform_codec::block_size);
				for (int position = 0; position < transform_codec::block_size; ++position)
					basis[frequency][position] = scale
						* std::cos(pi * (2 * position + 1) * frequency / (2 * transform_codec::block_size));
			}
		}
	};

	const dct_table& dct()
	{
		static const dct_table table;
		return table;
	}

	// Low frequencies first, so the zero runs between surviving coefficients are long.
	constexpr uint8_t scan_order[transform_codec::block_coefficients] = {
		 0,  1,  8, 16,  9,  2,  3, 10,
		17, 24, 32, 25, 18, 11,  4,  5,
		12, 19, 26, 33, 40, 48, 41, 34,
		27, 20, 13,  6,  7, 14, 21, 28,
		35, 42, 49, 56, 57, 50, 43, 36,
		29, 22, 15, 23, 30, 37, 44, 51,
		58, 59, 52, 45, 38, 31, 39, 46,
		53, 60, 61, 54, 47, 55, 62, 63};

	// One byte carries a zero run up to 15 and a level in -7..7, which is what almost every
	// motion-compensated residual coefficient turns out to be. Zero escapes to full varints.
	void append_coefficient(std::vector<uint8_t>& bytes, const int run, const int level)
	{
		if (run <= 15 && level != 0 && level >= -7 && level <= 7)
		{
			const int code = 2 * std::abs(level) - (level > 0 ? 1 : 0);
			bytes.push_back(static_cast<uint8_t>(run << 4 | code));
			return;
		}
		bytes.push_back(0);
		append_varuint(bytes, static_cast<uint32_t>(run));
		append_varuint(bytes, zigzag_encode(level));
	}

	bool read_coefficient(const std::vector<uint8_t>& bytes, size_t& offset, int& run, int& level)
	{
		if (offset >= bytes.size()) return false;
		const uint8_t token = bytes[offset++];
		if (token != 0)
		{
			const int code = token & 0x0f;
			if (code == 0 || code == 15) return false;
			run = token >> 4;
			level = (code + 1) / 2 * ((code & 1) != 0 ? 1 : -1);
			return true;
		}
		uint32_t encoded_run = 0;
		uint32_t encoded_level = 0;
		if (!read_varuint(bytes, offset, encoded_run) || !read_varuint(bytes, offset, encoded_level))
			return false;
		if (encoded_run >= transform_codec::block_coefficients) return false;
		run = static_cast<int>(encoded_run);
		level = zigzag_decode(encoded_level);
		return level != 0;
	}
}

reconstruction_quality measure_quality(const image_u8& source, const image_u8& reconstructed,
	                                    const image_u8* importance)
{
	reconstruction_quality quality;
	if (source.width() != reconstructed.width() || source.height() != reconstructed.height() || source.empty())
		return quality;

	double squared_error = 0.0;
	double weighted_error = 0.0;
	double total_weight = 0.0;
	const bool weighted = importance != nullptr && importance->width() == source.width()
		&& importance->height() == source.height();

	for (size_t i = 0; i < source.size(); ++i)
	{
		const double error = static_cast<double>(source.data()[i]) - reconstructed.data()[i];
		const double weight = weighted ? std::max(1.0, static_cast<double>(importance->data()[i])) : 1.0;
		squared_error += error * error;
		weighted_error += weight * error * error;
		total_weight += weight;
	}

	quality.rmse = std::sqrt(squared_error / source.size());
	quality.psnr = psnr_from_rmse(quality.rmse);
	quality.weighted_rmse = std::sqrt(weighted_error / total_weight);
	quality.weighted_psnr = psnr_from_rmse(quality.weighted_rmse);
	return quality;
}

void add_face_importance(image_u8& importance, const rect_i& face, const std::vector<point_i>& landmarks)
{
	if (importance.empty() || face.empty()) return;
	const int left = std::clamp(face.left, 0, importance.width() - 1);
	const int right = std::clamp(face.right, 0, importance.width() - 1);
	const int top = std::clamp(face.top, 0, importance.height() - 1);
	const int bottom = std::clamp(face.bottom, 0, importance.height() - 1);
	for (int y = top; y <= bottom; ++y)
		for (int x = left; x <= right; ++x) importance(x, y) = std::max<uint8_t>(importance(x, y), 96);

	const int radius = std::max(3, face.width() / 20);
	for (size_t i = 17; i < landmarks.size(); ++i)
	{
		const point_i centre = landmarks[i];
		for (int y = std::max(0, centre.y - radius); y <= std::min(importance.height() - 1, centre.y + radius); ++y)
			for (int x = std::max(0, centre.x - radius); x <= std::min(importance.width() - 1, centre.x + radius); ++x)
			{
				const int dx = x - centre.x;
				const int dy = y - centre.y;
				if (dx * dx + dy * dy <= radius * radius)
					importance(x, y) = std::max<uint8_t>(importance(x, y), 255);
			}
	}
}

void transform_codec::reconstruct(const image_u8& source, const image_u8* importance,
	                              image_u8& reconstructed) const
{
	std::vector<int16_t> coefficients;
	std::vector<uint8_t> quantizers;
	encode_coefficients(source, importance, coefficients, quantizers);
	reconstruct_coefficients(coefficients, quantizers, source.width(), source.height(), reconstructed);
}

void transform_codec::encode_coefficients(const image_u8& source, const image_u8* importance,
	                                      std::vector<int16_t>& coefficients,
	                                      std::vector<uint8_t>& quantizers) const
{
	const int cols = (source.width() + block_size - 1) / block_size;
	const int rows = (source.height() + block_size - 1) / block_size;
	coefficients.assign(static_cast<size_t>(cols) * rows * block_coefficients, 0);
	quantizers.assign(static_cast<size_t>(cols) * rows, 0);
	if (source.empty()) return;

	const auto& basis = dct().basis;

	for (int block_y = 0; block_y < source.height(); block_y += block_size)
	{
		for (int block_x = 0; block_x < source.width(); block_x += block_size)
		{
			float samples[block_size][block_size];
			float horizontal[block_size][block_size] = {};
			float transformed[block_size][block_size] = {};

			int importance_sum = 0;
			for (int y = 0; y < block_size; ++y)
				for (int x = 0; x < block_size; ++x)
				{
					const int sx = std::min(block_x + x, source.width() - 1);
					const int sy = std::min(block_y + y, source.height() - 1);
					samples[y][x] = static_cast<float>(source(sx, sy)) - 128.0f;
					importance_sum += importance != nullptr && importance->contains(sx, sy) ? (*importance)(sx, sy) : 32;
				}

			// Snapped to a ladder: without it a landmark disc drifting by a pixel re-quantises
			// blocks whose pixels never moved, and those blocks then have to be re-sent.
			const float average_importance = std::round(
				importance_sum / static_cast<float>(block_size * block_size) / 32.0f) * 32.0f;
			const int quantizer = static_cast<int>(std::lround(
				34.0f - 26.0f * std::clamp(average_importance / 255.0f, 0.0f, 1.0f)));
			const int block_index = block_y / block_size * cols + block_x / block_size;
			quantizers[block_index] = static_cast<uint8_t>(quantizer);

			for (int y = 0; y < block_size; ++y)
				for (int u = 0; u < block_size; ++u)
					for (int x = 0; x < block_size; ++x) horizontal[y][u] += samples[y][x] * basis[u][x];

			for (int v = 0; v < block_size; ++v)
				for (int u = 0; u < block_size; ++u)
				{
					for (int y = 0; y < block_size; ++y) transformed[v][u] += basis[v][y] * horizontal[y][u];
					const int value = static_cast<int>(std::lround(
						transformed[v][u] / (quantizer * frequency_scale(u, v))));
					coefficients[static_cast<size_t>(block_index) * block_coefficients + v * block_size + u]
						= static_cast<int16_t>(std::clamp(value, -32767, 32767));
				}
		}
	}
}

bool transform_codec::reconstruct_coefficients(const std::vector<int16_t>& coefficients,
	                                           const std::vector<uint8_t>& quantizers,
	                                           const int width, const int height,
	                                           image_u8& reconstructed) const
{
	const int cols = (width + block_size - 1) / block_size;
	const int rows = (height + block_size - 1) / block_size;
	if (width <= 0 || height <= 0 || quantizers.size() != static_cast<size_t>(cols) * rows
		|| coefficients.size() != quantizers.size() * block_coefficients) return false;

	const auto& basis = dct().basis;

	reconstructed.resize(width, height);
	for (int block_y = 0; block_y < height; block_y += block_size)
		for (int block_x = 0; block_x < width; block_x += block_size)
		{
			const int block_index = block_y / block_size * cols + block_x / block_size;
			float transformed[block_size][block_size];
			for (int v = 0; v < block_size; ++v)
				for (int u = 0; u < block_size; ++u)
					transformed[v][u] = coefficients[static_cast<size_t>(block_index) * block_coefficients
						+ v * block_size + u] * quantizers[block_index] * frequency_scale(u, v);
			float vertical[block_size][block_size] = {};
			for (int y = 0; y < block_size; ++y)
				for (int u = 0; u < block_size; ++u)
					for (int v = 0; v < block_size; ++v) vertical[y][u] += basis[v][y] * transformed[v][u];
			for (int y = 0; y < block_size && block_y + y < height; ++y)
				for (int x = 0; x < block_size && block_x + x < width; ++x)
				{
					float value = 128.0f;
					for (int u = 0; u < block_size; ++u) value += basis[u][x] * vertical[y][u];
					reconstructed(block_x + x, block_y + y) = static_cast<uint8_t>(
						std::clamp(static_cast<int>(std::lround(value)), 0, 255));
				}
		}
	return true;
}

void temporal_transform_stream::reset()
{
	encoder_coefficients.clear();
	decoder_coefficients.clear();
	encoder_quantizers.clear();
	decoder_quantizers.clear();
	encoder_age.clear();
	encoder_reference = {};
	decoder_reference = {};
	frame_width = 0;
	frame_height = 0;
}

transform_packet temporal_transform_stream::encode(const transform_codec& codec, const image_u8& source,
	                                                const image_u8* importance, bool force_keyframe,
	                                                const size_t max_packet_bytes)
{
	transform_packet packet;
	std::vector<int16_t> current_coefficients;
	std::vector<uint8_t> current_quantizers;
	codec.encode_coefficients(source, importance, current_coefficients, current_quantizers);
	force_keyframe = force_keyframe || source.width() != frame_width || source.height() != frame_height
		|| encoder_coefficients.size() != current_coefficients.size();
	const int cols = (source.width() + transform_codec::block_size - 1) / transform_codec::block_size;
	const int block_count = static_cast<int>(current_quantizers.size());
	if (force_keyframe || encoder_age.size() != current_quantizers.size())
		encoder_age.assign(current_quantizers.size(), 0);

	const bool can_predict = !force_keyframe && encoder_reference.width() == source.width()
		&& encoder_reference.height() == source.height() && !encoder_reference.empty();

	// Cost of one residual token, near enough for a decision that only has to be about right.
	const auto token_bytes = [](const int delta)
	{
		return delta >= -7 && delta <= 7 ? 1.0 : 3.0;
	};

	struct coded_block
	{
		bool motion = false;
		bool send_quantizer = false;
		point_i vector{0, 0};
		int count = 0;
		std::vector<uint8_t> tokens;
		double distortion = 0.0; // squared pixel error still left after this block is applied
		int16_t coefficients[transform_codec::block_coefficients] = {};

		// Assumes the vector has to be spelled out. Inheriting it at serialization time can
		// only shrink the record, so a budget reserved from this figure always holds.
		size_t bytes() const
		{
			return 1 + (send_quantizer ? 1u : 0u)
				+ (motion ? varuint_size(zigzag_encode(vector.x)) + varuint_size(zigzag_encode(vector.y)) : 0u)
				+ (count > 0 ? varuint_size(static_cast<uint32_t>(count)) + tokens.size() : 0u);
		}
	};

	// One block coded against a predictor. Each residual coefficient has to earn its byte,
	// which is what turns a well predicted block into a two byte "it just moved" record
	// instead of a page of requantisation noise.
	const auto code_against = [&](const int block, const int16_t* predictor, const bool send_quantizer,
		const point_i vector, const bool motion, const double lambda)
	{
		const size_t base = static_cast<size_t>(block) * transform_codec::block_coefficients;
		const double quantizer = current_quantizers[block];
		coded_block coded;
		coded.motion = motion;
		coded.send_quantizer = send_quantizer;
		coded.vector = vector;

		int run = 0;
		for (const uint8_t coefficient : scan_order)
		{
			const int delta = current_coefficients[base + coefficient] - predictor[coefficient];
			const double scale = quantizer * frequency_scale(coefficient % transform_codec::block_size,
				coefficient / transform_codec::block_size);
			const double gain = delta * scale * (delta * scale);
			if (delta == 0 || gain <= lambda * token_bytes(delta))
			{
				coded.coefficients[coefficient] = predictor[coefficient];
				coded.distortion += gain;
				++run;
				continue;
			}
			coded.coefficients[coefficient] = current_coefficients[base + coefficient];
			append_coefficient(coded.tokens, run, delta);
			run = 0;
			++coded.count;
		}
		return coded;
	};

	struct candidate
	{
		int index = 0;
		bool urgent = false;
		double priority = 0.0;
		size_t cost = 0;
		coded_block coded;
	};
	std::vector<candidate> candidates;
	int16_t predicted[transform_codec::block_coefficients];
	point_i seed_vector{0, 0};

	for (int block = 0; block < block_count; ++block)
	{
		const size_t base = static_cast<size_t>(block) * transform_codec::block_coefficients;
		const int sent_quantizer = force_keyframe ? 0 : encoder_quantizers[block];
		const int block_x = block % cols * transform_codec::block_size;
		const int block_y = block / cols * transform_codec::block_size;

		bool identical = current_quantizers[block] == sent_quantizer;
		double distortion = 0.0;
		for (int coefficient = 0; coefficient < transform_codec::block_coefficients; ++coefficient)
		{
			const int now = current_coefficients[base + coefficient];
			const int sent = force_keyframe ? 0 : encoder_coefficients[base + coefficient];

			// The DCT here is orthonormal, so this sum of squared dequantised coefficient
			// errors is exactly the squared pixel error the decoder is currently showing.
			const double error = (now * static_cast<double>(current_quantizers[block])
				- sent * static_cast<double>(sent_quantizer))
				* frequency_scale(coefficient % transform_codec::block_size,
					coefficient / transform_codec::block_size);
			distortion += error * error;
			if (now != sent) identical = false;
		}
		if (identical) continue;

		// Squared pixel error the codec is willing to accept to save one byte. Tied to the
		// block's own quantiser, so a coarsely coded background is allowed to be sloppier
		// about its residual than a face is.
		const double lambda = force_keyframe ? 0.0
			: rd_lambda * current_quantizers[block] * current_quantizers[block];
		const bool send_quantizer = current_quantizers[block] != sent_quantizer;
		coded_block coded;
		if (force_keyframe)
		{
			std::fill_n(predicted, transform_codec::block_coefficients, static_cast<int16_t>(0));
			coded = code_against(block, predicted, true, {0, 0}, false, 0.0);
		}
		else
		{
			coded = code_against(block, encoder_coefficients.data() + base, send_quantizer, {0, 0}, false, lambda);
			if (can_predict)
			{
				const point_i motion = search_motion(source, encoder_reference, block_x, block_y, seed_vector);
				codec.predict_block(encoder_reference, block_x, block_y, motion.x, motion.y,
					current_quantizers[block], predicted);
				coded_block moved = code_against(block, predicted, true, motion, true, lambda);

				// Mode decision by the same rule as everything else: the cheaper way to reach
				// an acceptable picture wins. A translating head then costs a vector and a
				// handful of residuals instead of a fresh copy of every block it moves through.
				if (moved.distortion + lambda * moved.bytes() < coded.distortion + lambda * coded.bytes())
				{
					coded = std::move(moved);
					seed_vector = motion;
				}
			}
		}

		// A keyframe carries every block even when it has nothing to say about one, so the
		// decoder ends up with a quantiser for all of them. Otherwise a flat mid-grey block
		// has no error to remove and would silently be left out of the "complete" scene.
		const bool changes_decoder = coded.motion || coded.send_quantizer || coded.count > 0;
		const double improvement = distortion - coded.distortion;
		if (!force_keyframe && (!changes_decoder || improvement <= 0.0)) continue;

		int importance_sum = 0;
		int counted = 0;
		for (int y = 0; y < transform_codec::block_size && block_y + y < source.height(); ++y)
			for (int x = 0; x < transform_codec::block_size && block_x + x < source.width(); ++x, ++counted)
				importance_sum += importance != nullptr && importance->contains(block_x + x, block_y + y)
					? (*importance)(block_x + x, block_y + y) : 32;

		// Distortion actually removed per byte spent, tilted towards faces. A block the face
		// has moved off carries a large error at a low importance, so only a bounded
		// importance gain - and the deadline below - ever let it outbid the face itself.
		const double average = counted > 0 ? importance_sum / static_cast<double>(counted) : 32.0;
		candidate entry;
		entry.index = block;
		entry.cost = varuint_size(static_cast<uint32_t>(block)) + coded.bytes();
		entry.urgent = encoder_age[block] >= starvation_deadline;
		entry.priority = importance_gain(average) * std::max(0.0, improvement) / entry.cost;
		entry.coded = std::move(coded);
		candidates.push_back(std::move(entry));
	}

	// Two stages, because the two questions are different: which blocks are worth their bytes,
	// then what order puts them on the wire most compactly.
	std::sort(candidates.begin(), candidates.end(), [](const candidate& a, const candidate& b)
	{
		if (a.urgent != b.urgent) return a.urgent;
		return a.priority != b.priority ? a.priority > b.priority : a.index < b.index;
	});

	std::vector<const candidate*> selected;
	size_t used = 10;
	for (const candidate& entry : candidates)
	{
		if (max_packet_bytes > 0 && used + entry.cost > max_packet_bytes)
		{
			if (encoder_age[entry.index] < 0xffff) ++encoder_age[entry.index];
			continue;
		}
		used += entry.cost;
		selected.push_back(&entry);
	}
	std::sort(selected.begin(), selected.end(), [](const candidate* a, const candidate* b)
	{
		return a->index < b->index;
	});

	append_packet_header(packet.bytes, 'T', 2, force_keyframe, source.width(), source.height(), 0);
	if (force_keyframe)
	{
		encoder_coefficients.assign(current_coefficients.size(), 0);
		encoder_quantizers.assign(current_quantizers.size(), 0);
	}

	// A gap never needs more bytes than the absolute index it replaces, and an inherited
	// vector never more than a spelled out one, so the budget reserved above always holds.
	int previous_index = -1;
	point_i previous_vector{0, 0};
	for (const candidate* entry : selected)
	{
		const coded_block& coded = entry->coded;
		const bool inherit = coded.motion && coded.vector.x == previous_vector.x
			&& coded.vector.y == previous_vector.y;
		append_varuint(packet.bytes, static_cast<uint32_t>(entry->index - previous_index - 1));
		packet.bytes.push_back(static_cast<uint8_t>((coded.motion ? 1 : 0) | (coded.send_quantizer ? 2 : 0)
			| (inherit ? 4 : 0) | (coded.count == 0 ? 8 : 0)));
		if (coded.send_quantizer) packet.bytes.push_back(current_quantizers[entry->index]);
		if (coded.motion && !inherit)
		{
			append_varuint(packet.bytes, zigzag_encode(coded.vector.x));
			append_varuint(packet.bytes, zigzag_encode(coded.vector.y));
		}
		if (coded.count > 0)
		{
			append_varuint(packet.bytes, static_cast<uint32_t>(coded.count));
			packet.bytes.insert(packet.bytes.end(), coded.tokens.begin(), coded.tokens.end());
		}
		previous_index = entry->index;
		if (coded.motion) previous_vector = coded.vector;

		// What the decoder will hold, which is not the same as the source once a residual
		// coefficient has been judged not worth its byte.
		const size_t base = static_cast<size_t>(entry->index) * transform_codec::block_coefficients;
		for (int coefficient = 0; coefficient < transform_codec::block_coefficients; ++coefficient)
			encoder_coefficients[base + coefficient] = coded.coefficients[coefficient];
		encoder_quantizers[entry->index] = current_quantizers[entry->index];
		encoder_age[entry->index] = 0;
		++packet.updated_blocks;
	}
	if (packet.updated_blocks == 0)
	{
		packet.bytes.clear();
		return packet;
	}
	packet.bytes[8] = static_cast<uint8_t>(packet.updated_blocks & 0xff);
	packet.bytes[9] = static_cast<uint8_t>((packet.updated_blocks >> 8) & 0xff);
	packet.keyframe = force_keyframe;
	frame_width = source.width();
	frame_height = source.height();

	// What the decoder will be holding once this packet lands, and therefore what the next
	// packet's motion vectors are allowed to point into.
	codec.reconstruct_coefficients(encoder_coefficients, encoder_quantizers, frame_width, frame_height,
		encoder_reference);
	return packet;
}

bool temporal_transform_stream::decode(const transform_codec& codec, const transform_packet& packet,
	                                   const int width, const int height, image_u8& reconstructed)
{
	if (packet.bytes.size() < 10 || packet.bytes[0] != 'L' || packet.bytes[1] != 'T'
		|| packet.bytes[2] != 2) return false;
	size_t offset = 4;
	int encoded_width = 0, encoded_height = 0, count = 0;
	if (!read_u16(packet.bytes, offset, encoded_width) || !read_u16(packet.bytes, offset, encoded_height)
		|| !read_u16(packet.bytes, offset, count) || encoded_width != width || encoded_height != height)
		return false;
	const int cols = (width + transform_codec::block_size - 1) / transform_codec::block_size;
	const int rows = (height + transform_codec::block_size - 1) / transform_codec::block_size;
	const int block_count = cols * rows;
	const bool keyframe = (packet.bytes[3] & 1) != 0;
	auto next_coefficients = keyframe ? std::vector<int16_t>(static_cast<size_t>(block_count)
		* transform_codec::block_coefficients, 0) : decoder_coefficients;
	auto next_quantizers = keyframe ? std::vector<uint8_t>(block_count, 0) : decoder_quantizers;
	if (next_coefficients.size() != static_cast<size_t>(block_count) * transform_codec::block_coefficients
		|| next_quantizers.size() != static_cast<size_t>(block_count)) return false;
	const bool can_predict = !keyframe && decoder_reference.width() == width
		&& decoder_reference.height() == height && !decoder_reference.empty();

	int16_t predicted[transform_codec::block_coefficients];
	int64_t block = -1;
	point_i previous_vector{0, 0};
	for (int change = 0; change < count; ++change)
	{
		uint32_t gap = 0;
		if (!read_varuint(packet.bytes, offset, gap)) return false;
		block += static_cast<int64_t>(gap) + 1;
		if (block >= block_count) return false;
		if (offset >= packet.bytes.size()) return false;

		const uint8_t flags = packet.bytes[offset++];
		if ((flags & ~0x0fu) != 0) return false;
		if ((flags & 4) != 0 && (flags & 1) == 0) return false;
		if ((flags & 2) != 0)
		{
			if (offset >= packet.bytes.size()) return false;
			next_quantizers[block] = packet.bytes[offset++];
		}
		if (next_quantizers[block] == 0) return false;

		const size_t base = static_cast<size_t>(block) * transform_codec::block_coefficients;
		if ((flags & 1) != 0)
		{
			if (!can_predict) return false;
			if ((flags & 4) == 0)
			{
				uint32_t encoded_x = 0, encoded_y = 0;
				if (!read_varuint(packet.bytes, offset, encoded_x)
					|| !read_varuint(packet.bytes, offset, encoded_y)) return false;
				previous_vector = {zigzag_decode(encoded_x), zigzag_decode(encoded_y)};
				if (std::abs(previous_vector.x) > max_motion || std::abs(previous_vector.y) > max_motion)
					return false;
			}
			codec.predict_block(decoder_reference, static_cast<int>(block) % cols * transform_codec::block_size,
				static_cast<int>(block) / cols * transform_codec::block_size,
				previous_vector.x, previous_vector.y, next_quantizers[block], predicted);
			for (int coefficient = 0; coefficient < transform_codec::block_coefficients; ++coefficient)
				next_coefficients[base + coefficient] = predicted[coefficient];
		}

		uint32_t changed = 0;
		if ((flags & 8) == 0 && (!read_varuint(packet.bytes, offset, changed)
			|| changed == 0 || changed > transform_codec::block_coefficients)) return false;
		int position = 0;
		for (uint32_t token = 0; token < changed; ++token)
		{
			int run = 0, level = 0;
			if (!read_coefficient(packet.bytes, offset, run, level)) return false;
			position += run;
			if (position >= transform_codec::block_coefficients) return false;
			const size_t coefficient_offset = base + scan_order[position];
			const int value = next_coefficients[coefficient_offset] + level;
			if (value < -32767 || value > 32767) return false;
			next_coefficients[coefficient_offset] = static_cast<int16_t>(value);
			++position;
		}
	}
	if (offset != packet.bytes.size()) return false;
	if (!codec.reconstruct_coefficients(next_coefficients, next_quantizers, width, height, reconstructed)) return false;
	decoder_coefficients = std::move(next_coefficients);
	decoder_quantizers = std::move(next_quantizers);
	decoder_reference = reconstructed;
	return true;
}

void transform_codec::predict_block(const image_u8& reference, const int block_x, const int block_y,
	                                const int dx, const int dy, const int quantizer,
	                                int16_t* coefficients) const
{
	const auto& basis = dct().basis;
	float samples[block_size][block_size];
	float horizontal[block_size][block_size] = {};
	float transformed[block_size][block_size] = {};

	for (int y = 0; y < block_size; ++y)
		for (int x = 0; x < block_size; ++x)
		{
			const int sx = std::clamp(block_x + x + dx, 0, reference.width() - 1);
			const int sy = std::clamp(block_y + y + dy, 0, reference.height() - 1);
			samples[y][x] = static_cast<float>(reference(sx, sy)) - 128.0f;
		}

	for (int y = 0; y < block_size; ++y)
		for (int u = 0; u < block_size; ++u)
			for (int x = 0; x < block_size; ++x) horizontal[y][u] += samples[y][x] * basis[u][x];

	for (int v = 0; v < block_size; ++v)
		for (int u = 0; u < block_size; ++u)
		{
			for (int y = 0; y < block_size; ++y) transformed[v][u] += basis[v][y] * horizontal[y][u];
			const int value = static_cast<int>(std::lround(
				transformed[v][u] / (quantizer * frequency_scale(u, v))));
			coefficients[v * block_size + u] = static_cast<int16_t>(std::clamp(value, -32767, 32767));
		}
}

void patch_autoencoder::dense::reset(const int layer_inputs, const int layer_outputs, std::mt19937& rng)
{
	inputs = layer_inputs;
	outputs = layer_outputs;
	weights.assign(static_cast<size_t>(inputs) * outputs, 0.0f);
	bias.assign(outputs, 0.0f);
	weight_rate.assign(weights.size(), 0.0f);
	bias_rate.assign(outputs, 0.0f);

	// Symmetry breaking, scaled by fan-in so the initial activations stay in range.
	const float limit = 1.0f / std::sqrt(static_cast<float>(inputs));
	std::uniform_real_distribution<float> distribution(-limit, limit);
	for (float& weight : weights) weight = distribution(rng);
}

void patch_autoencoder::dense::forward(const float* input, float* output, const bool activate) const
{
	for (int o = 0; o < outputs; ++o)
	{
		const float* w = weights.data() + static_cast<size_t>(o) * inputs;
		float sum = bias[o];
		for (int i = 0; i < inputs; ++i) sum += w[i] * input[i];
		output[o] = activate ? std::tanh(sum) : sum;
	}
}

// Per-weight step normalised by a running mean square gradient. Plain SGD needs a learning
// rate that suits the worst-scaled layer; this lets every layer move at its own pace, which
// is what makes the reset-and-watch-it-converge demo work in seconds rather than minutes.
void patch_autoencoder::dense::backward(const float* input, const float* gradient, float* input_gradient)
{
	if (input_gradient != nullptr)
		for (int i = 0; i < inputs; ++i) input_gradient[i] = 0.0f;

	for (int o = 0; o < outputs; ++o)
	{
		float* w = weights.data() + static_cast<size_t>(o) * inputs;
		float* rate = weight_rate.data() + static_cast<size_t>(o) * inputs;
		const float g = gradient[o];
		for (int i = 0; i < inputs; ++i)
		{
			if (input_gradient != nullptr) input_gradient[i] += w[i] * g;
			const float step = g * input[i];
			rate[i] = 0.9f * rate[i] + 0.1f * step * step;
			w[i] -= learning_rate * step / (std::sqrt(rate[i]) + 1e-8f);
		}
		bias_rate[o] = 0.9f * bias_rate[o] + 0.1f * g * g;
		bias[o] -= learning_rate * g / (std::sqrt(bias_rate[o]) + 1e-8f);
	}
}

patch_autoencoder::patch_autoencoder() : rng(12345)
{
	reset();
}

void patch_autoencoder::reset()
{
	training_steps = 0;
	recent_loss = 0.0f;
	rng.seed(12345);
	analysis.reset(patch_pixels, code_size, rng);
	synthesis_hidden.reset(code_size, hidden_size, rng);
	synthesis_pixels.reset(hidden_size, patch_pixels, rng);
}

int patch_autoencoder::normalise(const image_u8& src, const int px, const int py, float* input) const
{
	int total = 0;
	for (int y = 0; y < patch_size; ++y)
	{
		const uint8_t* row = src.row(py + y) + px;
		for (int x = 0; x < patch_size; ++x) total += row[x];
	}

	const int mean = (total + patch_pixels / 2) / patch_pixels;
	for (int y = 0; y < patch_size; ++y)
	{
		const uint8_t* row = src.row(py + y) + px;
		for (int x = 0; x < patch_size; ++x)
			input[y * patch_size + x] = (static_cast<int>(row[x]) - mean) / 128.0f;
	}
	return mean;
}

void patch_autoencoder::encode(const float* input, float* code) const
{
	analysis.forward(input, code, true);
}

void patch_autoencoder::decode(const float* code, float* hidden, float* output) const
{
	synthesis_hidden.forward(code, hidden, true);
	synthesis_pixels.forward(hidden, output, false);
}

float patch_autoencoder::train(const image_u8& luma, const int batch, const image_u8* importance)
{
	const int cols = luma.width() / patch_size;
	const int rows = luma.height() / patch_size;
	if (cols <= 0 || rows <= 0 || batch <= 0) return recent_loss;

	std::uniform_int_distribution<int> pick_col(0, cols - 1);
	std::uniform_int_distribution<int> pick_row(0, rows - 1);

	float input[patch_pixels];
	float code[code_size];
	float hidden[hidden_size];
	float output[patch_pixels];
	float output_grad[patch_pixels];
	float hidden_grad[hidden_size];
	float code_grad[code_size];

	double total_error = 0.0;
	double total_weight = 0.0;

	for (int n = 0; n < batch; ++n)
	{
		const int px = pick_col(rng) * patch_size;
		const int py = pick_row(rng) * patch_size;

		normalise(luma, px, py, input);
		encode(input, code);
		decode(code, hidden, output);

		// The output layer is linear, so this is already dL/dz for it.
		for (int i = 0; i < patch_pixels; ++i)
		{
			const float error = output[i] - input[i];
			const int x = px + i % patch_size;
			const int y = py + i / patch_size;
			const float weight = importance != nullptr && importance->contains(x, y)
				? std::max(1.0f, (*importance)(x, y) / 32.0f) : 1.0f;
			total_error += weight * static_cast<double>(error) * error;
			total_weight += weight;
			output_grad[i] = weight * error / patch_pixels;
		}

		synthesis_pixels.backward(hidden, output_grad, hidden_grad);
		for (int i = 0; i < hidden_size; ++i) hidden_grad[i] *= 1.0f - hidden[i] * hidden[i];
		synthesis_hidden.backward(code, hidden_grad, code_grad);
		for (int j = 0; j < code_size; ++j) code_grad[j] *= 1.0f - code[j] * code[j];
		analysis.backward(input, code_grad, nullptr);
	}

	++training_steps;
	recent_loss = static_cast<float>(total_error / total_weight);
	return recent_loss;
}

void patch_autoencoder::reconstruct(const image_u8& src, image_u8& dst) const
{
	std::vector<int8_t> codes;
	encode_codes(src, codes);
	reconstruct_codes(codes, src.width(), src.height(), dst);
}

void patch_autoencoder::encode_codes(const image_u8& src, std::vector<int8_t>& codes) const
{
	const int cols = src.width() / patch_size;
	const int rows = src.height() / patch_size;
	codes.resize(static_cast<size_t>(cols) * rows * bytes_per_patch);
	float input[patch_pixels];
	float code[code_size];

	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			const int mean = normalise(src, col * patch_size, row * patch_size, input);
			encode(input, code);
			const size_t offset = (static_cast<size_t>(row) * cols + col) * bytes_per_patch;
			codes[offset] = static_cast<int8_t>(mean - 128);
			for (int i = 0; i < code_size; ++i)
				codes[offset + 1 + i] = static_cast<int8_t>(
					std::round(std::clamp(code[i], -1.0f, 1.0f) * 127.0f));
		}
	}
}

bool patch_autoencoder::reconstruct_codes(const std::vector<int8_t>& codes, const int width,
	                                      const int height, image_u8& dst) const
{
	const int cols = width / patch_size;
	const int rows = height / patch_size;
	if (width <= 0 || height <= 0 || codes.size() != static_cast<size_t>(cols) * rows * bytes_per_patch)
		return false;

	dst.resize(width, height);
	dst.fill(0);
	float code[code_size];
	float hidden[hidden_size];
	float output[patch_pixels];
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			const size_t offset = (static_cast<size_t>(row) * cols + col) * bytes_per_patch;
			const int mean = codes[offset] + 128;
			for (int i = 0; i < code_size; ++i) code[i] = codes[offset + 1 + i] / 127.0f;
			decode(code, hidden, output);

			for (int y = 0; y < patch_size; ++y)
			{
				uint8_t* out = dst.row(row * patch_size + y) + col * patch_size;
				for (int x = 0; x < patch_size; ++x)
					out[x] = static_cast<uint8_t>(std::clamp(
						static_cast<int>(std::lround(mean + output[y * patch_size + x] * 128.0f)), 0, 255));
			}
		}
	}
	return true;
}

void temporal_patch_stream::reset()
{
	encoder_codes.clear();
	decoder_codes.clear();
	encoder_age.clear();
	frame_width = 0;
	frame_height = 0;
}

latent_packet temporal_patch_stream::encode(const patch_autoencoder& model, const image_u8& source,
	                                        const image_u8* importance, bool force_keyframe,
	                                        const size_t max_packet_bytes)
{
	latent_packet packet;
	std::vector<int8_t> current;
	model.encode_codes(source, current);
	const int cols = source.width() / patch_autoencoder::patch_size;
	const int rows = source.height() / patch_autoencoder::patch_size;
	const int patch_count = cols * rows;
	force_keyframe = force_keyframe || source.width() != frame_width || source.height() != frame_height
		|| encoder_codes.size() != current.size();

	if (force_keyframe)
	{
		append_packet_header(packet.bytes, 'B', 2, true, source.width(), source.height(), patch_count);
		for (const int8_t code : current) packet.bytes.push_back(static_cast<uint8_t>(static_cast<int>(code) + 128));
		packet.keyframe = true;
		packet.updated_patches = patch_count;
		encoder_codes = std::move(current);
		encoder_age.assign(patch_count, 0);
		frame_width = source.width();
		frame_height = source.height();
		return packet;
	}
	if (encoder_age.size() != static_cast<size_t>(patch_count)) encoder_age.assign(patch_count, 0);
	if (max_packet_bytes < 10) return packet;

	struct candidate
	{
		int index = 0;
		bool urgent = false;
		double priority = 0.0;
		size_t cost = 0;
		std::vector<uint8_t> body;
	};
	std::vector<candidate> candidates;
	for (int index = 0; index < patch_count; ++index)
	{
		candidate entry;
		double distortion = 0.0;
		for (int code = 0; code < patch_autoencoder::bytes_per_patch; ++code)
		{
			const size_t offset = static_cast<size_t>(index) * patch_autoencoder::bytes_per_patch + code;
			const int delta = static_cast<int>(current[offset]) - encoder_codes[offset];

			// A mean step and a code step are both worth roughly one grey level at the
			// output, so they can be summed without weighting.
			distortion += static_cast<double>(delta) * delta;
			append_varuint(entry.body, zigzag_encode(delta));
		}
		if (distortion == 0.0) continue;

		int importance_sum = 0;
		const int patch_x = index % cols * patch_autoencoder::patch_size;
		const int patch_y = index / cols * patch_autoencoder::patch_size;
		for (int y = 0; y < patch_autoencoder::patch_size; ++y)
			for (int x = 0; x < patch_autoencoder::patch_size; ++x)
				importance_sum += importance != nullptr && importance->contains(patch_x + x, patch_y + y)
					? (*importance)(patch_x + x, patch_y + y) : 32;

		// Same rule as the transform stream: latent error removed per byte, with face
		// importance as a bounded multiplier rather than an overriding one.
		const double average = importance_sum / static_cast<double>(patch_autoencoder::patch_pixels);
		entry.index = index;
		entry.cost = varuint_size(static_cast<uint32_t>(index)) + entry.body.size();
		entry.urgent = encoder_age[index] >= starvation_deadline;
		entry.priority = importance_gain(average) * distortion / entry.cost;
		candidates.push_back(std::move(entry));
	}
	std::sort(candidates.begin(), candidates.end(), [](const candidate& a, const candidate& b)
	{
		if (a.urgent != b.urgent) return a.urgent;
		return a.priority != b.priority ? a.priority > b.priority : a.index < b.index;
	});

	std::vector<const candidate*> selected;
	size_t used = 10;
	for (const candidate& entry : candidates)
	{
		if (used + entry.cost > max_packet_bytes)
		{
			if (encoder_age[entry.index] < 0xffff) ++encoder_age[entry.index];
			continue;
		}
		used += entry.cost;
		selected.push_back(&entry);
	}
	std::sort(selected.begin(), selected.end(), [](const candidate* a, const candidate* b)
	{
		return a->index < b->index;
	});

	append_packet_header(packet.bytes, 'B', 2, false, source.width(), source.height(), 0);
	int previous_index = -1;
	for (const candidate* change : selected)
	{
		append_varuint(packet.bytes, static_cast<uint32_t>(change->index - previous_index - 1));
		packet.bytes.insert(packet.bytes.end(), change->body.begin(), change->body.end());
		previous_index = change->index;
		for (int code = 0; code < patch_autoencoder::bytes_per_patch; ++code)
		{
			const size_t offset = static_cast<size_t>(change->index) * patch_autoencoder::bytes_per_patch + code;
			encoder_codes[offset] = current[offset];
		}
		encoder_age[change->index] = 0;
		++packet.updated_patches;
	}

	if (packet.updated_patches == 0)
	{
		packet.bytes.clear();
		return packet;
	}
	packet.bytes[8] = static_cast<uint8_t>(packet.updated_patches & 0xff);
	packet.bytes[9] = static_cast<uint8_t>((packet.updated_patches >> 8) & 0xff);
	return packet;
}

bool temporal_patch_stream::decode(const patch_autoencoder& model, const latent_packet& packet,
	                               const int width, const int height, image_u8& reconstructed)
{
	if (packet.bytes.size() < 10 || packet.bytes[0] != 'L' || packet.bytes[1] != 'B'
		|| packet.bytes[2] != 2) return false;
	size_t offset = 4;
	int encoded_width = 0;
	int encoded_height = 0;
	int count = 0;
	if (!read_u16(packet.bytes, offset, encoded_width) || !read_u16(packet.bytes, offset, encoded_height)
		|| !read_u16(packet.bytes, offset, count) || encoded_width != width || encoded_height != height)
		return false;
	const int patch_count = width / patch_autoencoder::patch_size * (height / patch_autoencoder::patch_size);
	const size_t stride = patch_autoencoder::bytes_per_patch;
	std::vector<int8_t> next = decoder_codes;
	const bool keyframe = (packet.bytes[3] & 1) != 0;
	if (keyframe)
	{
		if (count != patch_count || packet.bytes.size() != offset + static_cast<size_t>(count) * stride)
			return false;
		next.resize(static_cast<size_t>(count) * stride);
		for (size_t index = 0; index < static_cast<size_t>(count) * stride; ++index)
			next[index] = static_cast<int8_t>(static_cast<int>(packet.bytes[offset++]) - 128);
	}
	else
	{
		if (next.size() != static_cast<size_t>(patch_count) * stride) return false;
		int64_t index = -1;
		for (int change = 0; change < count; ++change)
		{
			uint32_t gap = 0;
			if (!read_varuint(packet.bytes, offset, gap)) return false;
			index += static_cast<int64_t>(gap) + 1;
			if (index >= patch_count) return false;
			for (int code = 0; code < patch_autoencoder::bytes_per_patch; ++code)
			{
				uint32_t encoded_delta = 0;
				if (!read_varuint(packet.bytes, offset, encoded_delta)) return false;
				const size_t code_offset = static_cast<size_t>(index) * stride + code;
				const int value = static_cast<int>(next[code_offset]) + zigzag_decode(encoded_delta);
				if (value < -128 || value > 127) return false;
				next[code_offset] = static_cast<int8_t>(value);
			}
		}
		if (offset != packet.bytes.size()) return false;
	}

	if (!model.reconstruct_codes(next, width, height, reconstructed)) return false;
	decoder_codes = std::move(next);
	return true;
}
