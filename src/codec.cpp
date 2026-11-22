#include "codec.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include <emmintrin.h>

#include "entropy.h"
#include "warp.h"

// Every model the wire adapts. Both ends start from these defaults at a keyframe and make the
// same decisions against them in the same order, so their copies never differ.
struct wire_models
{
	// Residual coefficients, by set - 0 intra, 1 inter - and scan position.
	bit_model significant[2][64];
	bit_model last[2][64];
	// Levels, by class - the set, twice, for the DC position and the rest - and context.
	bit_model greater_one[4][5];
	bit_model remainder[4][8];

	// Macroblock records.
	bit_model records;    // whether a delta packet updates any macroblock at all
	bit_model updated[6]; // each macroblock in raster order, by its updated neighbours and last packet
	bit_model skip[3];    // by how many of its left and upper neighbours were skipped
	bit_model intra[3];   // by how many of them are intra
	bit_model split;      // a vector for each block rather than one for the macroblock
	signed_models vector[2];
	bit_model coded[2][3]; // a block's residual, by set and how many of its left and upper neighbours have one
	signed_models step;
	bit_model probable_mode; // an intra block takes the direction its neighbours suggest
	bit_model mode[7];       // or names one of the other eight, as a three-level tree

	// Landmarks.
	bit_model layout;
	bit_model faces[6];
	bit_model points[6];
	signed_models first[2];
	signed_models along[2];
	bit_model face_moved;
	signed_models shift[2];
	signed_models residual[5][2];
};

namespace
{
	constexpr int block = transform_block;
	constexpr int block_area = block * block;

	// Quantisers double every six steps from 4 at step 0, as H.264's do. The session picks the
	// background's step to suit the rate, and importance makes a face's macroblocks finer by up
	// to face_steps: a step a 32nd of the importance map, so the map drifting by a pixel with
	// the face rarely changes a quantiser.
	constexpr int ladder_steps = quantizer_steps;
	constexpr int first_step = 18; // what step deltas in a packet start from
	constexpr int face_steps = 9;

	float quantizer(const int step)
	{
		return 4.0f * std::exp2(static_cast<float>(step) / 6.0f);
	}

	constexpr uint8_t wire_version = 5;
	constexpr uint8_t flag_keyframe = 1;
	constexpr uint8_t flag_geometry = 2;
	constexpr uint8_t flag_warp = 4;
	constexpr int max_dimension = 4096;
	constexpr int max_faces = 32;
	constexpr int max_points = 64;
	constexpr int coordinate_bound = 1 << 16;

	// Vectors are in quarter pixels, so this is sixteen pixels either way.
	constexpr int max_motion = 64;

	// Packets a macroblock with work to do may be passed over before it is promoted ahead of
	// every macroblock still inside its deadline: about a second at 15 Hz. Importance decides
	// the order macroblocks are served in, never whether one is served at all, which is what
	// stops a moving face leaving copies of itself behind.
	constexpr uint16_t starvation_deadline = 16;

	// Squared pixel error the encoder accepts to save one bit, in units of the macroblock's own
	// quantiser squared: the Lagrange multiplier of the usual rate-distortion formulation. The
	// vector search weighs bits against absolute error, so its multiplier is the square root.
	constexpr double rd_lambda = 0.15;

	// Dead-zone rounding: a coefficient rounds up only past this fraction of a step, which
	// spends fewer bits on levels that barely exceed a boundary.
	constexpr float intra_rounding = 1.0f / 3.0f;
	constexpr float inter_rounding = 0.3f;

	// What marking a macroblock as updated costs, before the packet's other updates are known.
	constexpr double address_bits = 2.0;

	// Of an intra block's nine directions, how many the Hadamard ranking passes on to be
	// quantised and costed in full, besides the probable one. More measured no better.
	constexpr int intra_candidates = 3;

	// How far, in quarter pixels, the first step of the search for a block's own vector reaches
	// from the best of its seeds, the macroblock's vector among them: the search refines, it
	// does not start again.
	constexpr int split_reach = 8;

	// H.264's nine directions for an 8x8 intra block, in its numbering.
	constexpr int mode_vertical = 0;
	constexpr int mode_horizontal = 1;
	constexpr int mode_dc = 2;
	constexpr int mode_down_left = 3;
	constexpr int mode_down_right = 4;
	constexpr int mode_vertical_right = 5;
	constexpr int mode_horizontal_down = 6;
	constexpr int mode_vertical_left = 7;
	constexpr int mode_horizontal_up = 8;
	constexpr int intra_modes = 9;

	// What a packet does to a macroblock.
	constexpr uint8_t mb_untouched = 0;
	constexpr uint8_t mb_skip = 1;  // moved along the vector its neighbours predict, and nothing more
	constexpr uint8_t mb_inter = 2; // moved by vectors it sends, with residuals where they pay
	constexpr uint8_t mb_intra = 3; // predicted from the picture around it

	// Low frequencies first, so the coefficients that survive quantisation come early.
	// clang-format off
	constexpr uint8_t scan_order[block_area] = {
		 0,  1,  8, 16,  9,  2,  3, 10,
		17, 24, 32, 25, 18, 11,  4,  5,
		12, 19, 26, 33, 40, 48, 41, 34,
		27, 20, 13,  6,  7, 14, 21, 28,
		35, 42, 49, 56, 57, 50, 43, 36,
		29, 22, 15, 23, 30, 37, 44, 51,
		58, 59, 52, 45, 38, 31, 39, 46,
		53, 60, 61, 54, 47, 55, 62, 63};
	// clang-format on

	double psnr_from_rmse(const double rmse)
	{
		return rmse == 0.0 ? std::numeric_limits<double>::infinity() : 20.0 * std::log10(255.0 / rmse);
	}

	// The orthonormal 8x8 DCT basis, and every quantiser's step for every coefficient: coarser
	// for higher spatial frequencies, identically at both ends.
	struct transform_tables
	{
		float basis[block][block] = {};      // by frequency, then position
		float transposed[block][block] = {}; // by position, then frequency
		float step[ladder_steps][block_area] = {};

		transform_tables()
		{
			constexpr double pi = 3.14159265358979323846;
			for (int frequency = 0; frequency < block; ++frequency)
			{
				const double scale = std::sqrt((frequency == 0 ? 1.0 : 2.0) / block);
				for (int position = 0; position < block; ++position)
				{
					basis[frequency][position] = static_cast<float>(
						scale * std::cos(pi * (2 * position + 1) * frequency / (2 * block)));
					transposed[position][frequency] = basis[frequency][position];
				}
			}
			for (int s = 0; s < ladder_steps; ++s)
				for (int v = 0; v < block; ++v)
					for (int u = 0; u < block; ++u)
						step[s][v * block + u] = quantizer(s) * (1.0f + 0.12f * static_cast<float>(u + v));
		}
	};

	const transform_tables& tables()
	{
		static const transform_tables table;
		return table;
	}

	// out[i] = the sum over k of weights[k] * rows[k][i], for a row of eight, accumulated in k
	// order exactly as a scalar loop would, in two SSE registers. Every pass of both transforms
	// is eight of these; the decoder runs the same inverse, so the two ends round identically.
	void combine_rows(const float* weights, const float* rows, float* out)
	{
		__m128 low = _mm_setzero_ps();
		__m128 high = _mm_setzero_ps();
		for (int k = 0; k < block; ++k)
		{
			const __m128 weight = _mm_set1_ps(weights[k]);
			low = _mm_add_ps(low, _mm_mul_ps(weight, _mm_loadu_ps(rows + k * block)));
			high = _mm_add_ps(high, _mm_mul_ps(weight, _mm_loadu_ps(rows + k * block + 4)));
		}
		_mm_storeu_ps(out, low);
		_mm_storeu_ps(out + 4, high);
	}

	void forward_dct(const float* samples, float* coefficients)
	{
		const transform_tables& table = tables();
		float horizontal[block_area];
		for (int y = 0; y < block; ++y)
			combine_rows(samples + y * block, &table.transposed[0][0], horizontal + y * block);
		for (int v = 0; v < block; ++v) combine_rows(table.basis[v], horizontal, coefficients + v * block);
	}

	void inverse_dct(const float* coefficients, float* samples)
	{
		const transform_tables& table = tables();
		float vertical[block_area];
		for (int y = 0; y < block; ++y) combine_rows(table.transposed[y], coefficients, vertical + y * block);
		for (int y = 0; y < block; ++y) combine_rows(vertical + y * block, &table.basis[0][0], samples + y * block);
	}

	// A block of an image, the edge repeated where the block overhangs it.
	void read_block(const image_u8& image, const int block_x, const int block_y, uint8_t* out)
	{
		for (int y = 0; y < block; ++y)
		{
			const uint8_t* row = image.row(std::min(block_y + y, image.height() - 1));
			for (int x = 0; x < block; ++x) out[y * block + x] = row[std::min(block_x + x, image.width() - 1)];
		}
	}

	void write_block(image_u8& image, const int block_x, const int block_y, const uint8_t* in)
	{
		for (int y = 0; y < block && block_y + y < image.height(); ++y)
			for (int x = 0; x < block && block_x + x < image.width(); ++x)
				image(block_x + x, block_y + y) = in[y * block + x];
	}

	// Blocks are 64 contiguous bytes, four SSE2 registers; low-ban builds for x64 only, where
	// SSE2 is always present.
	double squared_error(const uint8_t* a, const uint8_t* b)
	{
		const __m128i zero = _mm_setzero_si128();
		__m128i total = zero;
		for (int i = 0; i < block_area; i += 16)
		{
			const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
			const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
			const __m128i low = _mm_sub_epi16(_mm_unpacklo_epi8(x, zero), _mm_unpacklo_epi8(y, zero));
			const __m128i high = _mm_sub_epi16(_mm_unpackhi_epi8(x, zero), _mm_unpackhi_epi8(y, zero));
			total = _mm_add_epi32(total, _mm_add_epi32(_mm_madd_epi16(low, low), _mm_madd_epi16(high, high)));
		}
		total = _mm_add_epi32(total, _mm_shuffle_epi32(total, _MM_SHUFFLE(1, 0, 3, 2)));
		total = _mm_add_epi32(total, _mm_shuffle_epi32(total, _MM_SHUFFLE(2, 3, 0, 1)));
		return _mm_cvtsi128_si32(total);
	}

	int horizontal_sum(const __m128i sad)
	{
		return _mm_cvtsi128_si32(sad) + _mm_cvtsi128_si32(_mm_srli_si128(sad, 8));
	}

	int absolute_error(const uint8_t* a, const uint8_t* b)
	{
		__m128i total = _mm_setzero_si128();
		for (int i = 0; i < block_area; i += 16)
			total = _mm_add_epi64(total, _mm_sad_epu8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)),
			                                          _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i))));
		return horizontal_sum(total);
	}

	// Two rows of eight, `stride` apart, as one register.
	__m128i load_rows(const uint8_t* at, const ptrdiff_t stride)
	{
		return _mm_unpacklo_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(at)),
		                          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(at + stride)));
	}

	point_i difference(const point_i a, const point_i b)
	{
		return {a.x - b.x, a.y - b.y};
	}

	// The reference sampled at every half-pixel position by H.264's six-tap filter, (1, -5, 20,
	// 20, -5, 1) / 32, which keeps detail a two-tap average would smear: a block that follows a
	// head for many packets is resampled each time, and an average blurs a little more every
	// time. Plane hy * 2 + hx holds the samples half a pixel right of and below each pixel by
	// those flags. Built once per packet over the picture both ends hold, edges repeated.
	struct half_pel_planes
	{
		image_u8 planes[4];

		explicit half_pel_planes(const image_u8& reference)
		{
			const int width = reference.width();
			const int height = reference.height();
			planes[0] = reference;
			for (int p = 1; p < 4; ++p) planes[p].resize(width, height);
			const auto tap = [](const int a, const int b, const int c, const int d, const int e, const int f)
			{
				return a - 5 * b + 20 * c + 20 * d - 5 * e + f;
			};
			const auto clip = [](const int value)
			{
				return static_cast<uint8_t>(std::clamp(value, 0, 255));
			};

			// Unrounded horizontal sums, which the diagonal plane filters again vertically.
			std::vector<int> across(static_cast<size_t>(width) * height);
			for (int y = 0; y < height; ++y)
			{
				const uint8_t* row = reference.row(y);
				const auto at = [&](const int x)
				{
					return static_cast<int>(row[std::clamp(x, 0, width - 1)]);
				};
				for (int x = 0; x < width; ++x)
				{
					const int sum = tap(at(x - 2), at(x - 1), at(x), at(x + 1), at(x + 2), at(x + 3));
					across[static_cast<size_t>(y) * width + x] = sum;
					planes[1](x, y) = clip((sum + 16) >> 5);
				}
			}
			for (int y = 0; y < height; ++y)
			{
				const auto row_at = [&](const int dy)
				{
					return std::clamp(y + dy, 0, height - 1);
				};
				const int r[6] = {row_at(-2), row_at(-1), row_at(0), row_at(1), row_at(2), row_at(3)};
				for (int x = 0; x < width; ++x)
				{
					planes[2](x, y) = clip((tap(reference(x, r[0]), reference(x, r[1]), reference(x, r[2]),
					                            reference(x, r[3]), reference(x, r[4]), reference(x, r[5])) +
					                        16) >>
					                       5);
					const auto h = [&](const int k)
					{
						return across[static_cast<size_t>(r[k]) * width + x];
					};
					planes[3](x, y) = clip((tap(h(0), h(1), h(2), h(3), h(4), h(5)) + 512) >> 10);
				}
			}
		}
	};

	// A quarter-pixel sample is the rounded mean of two samples of the half-pixel planes, the
	// second a pixel to the right or below where that one is nearer: H.264's luma
	// interpolation. A position on the half-pixel lattice names its one sample twice.
	struct quarter_tap
	{
		int plane;
		int dx;
		int dy;
	};

	// By the vector's quarter-pixel fraction, vertical then horizontal.
	// clang-format off
	constexpr quarter_tap quarter_samples[4][4][2] = {
		{{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {1, 0, 0}}, {{1, 0, 0}, {1, 0, 0}}, {{0, 1, 0}, {1, 0, 0}}},
		{{{0, 0, 0}, {2, 0, 0}}, {{1, 0, 0}, {2, 0, 0}}, {{1, 0, 0}, {3, 0, 0}}, {{1, 0, 0}, {2, 1, 0}}},
		{{{2, 0, 0}, {2, 0, 0}}, {{2, 0, 0}, {3, 0, 0}}, {{3, 0, 0}, {3, 0, 0}}, {{3, 0, 0}, {2, 1, 0}}},
		{{{0, 0, 1}, {2, 0, 0}}, {{2, 0, 0}, {1, 0, 1}}, {{3, 0, 0}, {1, 0, 1}}, {{2, 1, 0}, {1, 0, 1}}}};
	// clang-format on

	// Where a displaced block's two quarter-pixel sources start, when every sample it reads lies
	// inside the picture; false where the edge has to be repeated.
	struct displaced_block
	{
		const uint8_t* first = nullptr;
		const uint8_t* second = nullptr;
		ptrdiff_t stride = 0;
	};

	bool locate(const half_pel_planes& reference, const int block_x, const int block_y, const point_i vector,
	            displaced_block& out)
	{
		const int fraction_x = vector.x & 3;
		const int fraction_y = vector.y & 3;
		const quarter_tap* taps = quarter_samples[fraction_y][fraction_x];
		const image_u8& first = reference.planes[taps[0].plane];
		const int left = block_x + (vector.x - fraction_x) / 4;
		const int top = block_y + (vector.y - fraction_y) / 4;
		if (left < 0 || top < 0 || left + block >= first.width() || top + block >= first.height()) return false;
		out.first = first.row(top + taps[0].dy) + left + taps[0].dx;
		out.second = reference.planes[taps[1].plane].row(top + taps[1].dy) + left + taps[1].dx;
		out.stride = first.width();
		return true;
	}

	// A block of the reference displaced by a vector in quarter pixels, the edge repeated beyond
	// it. SSE2's byte average rounds exactly as the interpolation does.
	void predict_motion(const half_pel_planes& reference, const int block_x, const int block_y, const point_i vector,
	                    uint8_t* out)
	{
		displaced_block at;
		if (locate(reference, block_x, block_y, vector, at))
		{
			for (int y = 0; y < block; y += 2)
				_mm_storeu_si128(reinterpret_cast<__m128i*>(out + y * block),
				                 _mm_avg_epu8(load_rows(at.first + y * at.stride, at.stride),
				                              load_rows(at.second + y * at.stride, at.stride)));
			return;
		}

		const int fraction_x = vector.x & 3;
		const int fraction_y = vector.y & 3;
		const quarter_tap* taps = quarter_samples[fraction_y][fraction_x];
		const image_u8& first = reference.planes[taps[0].plane];
		const image_u8& second = reference.planes[taps[1].plane];
		const int left = block_x + (vector.x - fraction_x) / 4;
		const int top = block_y + (vector.y - fraction_y) / 4;
		const int last_x = first.width() - 1;
		const int last_y = first.height() - 1;
		for (int y = 0; y < block; ++y)
		{
			const uint8_t* a = first.row(std::clamp(top + y + taps[0].dy, 0, last_y));
			const uint8_t* b = second.row(std::clamp(top + y + taps[1].dy, 0, last_y));
			for (int x = 0; x < block; ++x)
				out[y * block + x] = static_cast<uint8_t>((a[std::clamp(left + x + taps[0].dx, 0, last_x)] +
				                                           b[std::clamp(left + x + taps[1].dx, 0, last_x)] + 1) >>
				                                          1);
		}
	}

	// The sum of absolute differences between a block of the source and the reference displaced
	// by a vector, without forming the prediction: the vector search's inner loop.
	int motion_error(const half_pel_planes& reference, const int block_x, const int block_y, const point_i vector,
	                 const uint8_t* source)
	{
		displaced_block at;
		if (!locate(reference, block_x, block_y, vector, at))
		{
			uint8_t window[block_area];
			predict_motion(reference, block_x, block_y, vector, window);
			return absolute_error(source, window);
		}
		__m128i total = _mm_setzero_si128();
		for (int y = 0; y < block; y += 2)
		{
			const __m128i predicted = _mm_avg_epu8(load_rows(at.first + y * at.stride, at.stride),
			                                       load_rows(at.second + y * at.stride, at.stride));
			total = _mm_add_epi64(total,
			                      _mm_sad_epu8(predicted,
			                                   _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + y * block))));
		}
		return horizontal_sum(total);
	}

	// The picture as the wire divides it: 8x8 blocks, which carry residuals, quantiser steps
	// and vectors, four to a 16x16 macroblock in Z order, which carries how they are
	// predicted. A picture that is not a whole number of macroblocks ends in partial ones,
	// holding only the blocks it reaches.
	struct block_grid
	{
		int width = 0;
		int height = 0;
		int cols = 0; // in blocks
		int rows = 0;
		int mb_cols = 0;
		int mb_rows = 0;

		block_grid(const int picture_width, const int picture_height) :
			width(picture_width), height(picture_height), cols((picture_width + block - 1) / block),
			rows((picture_height + block - 1) / block), mb_cols((cols + 1) / 2), mb_rows((rows + 1) / 2)
		{
		}

		int blocks() const { return cols * rows; }
		int macroblocks() const { return mb_cols * mb_rows; }
		int macroblock(const int mx, const int my) const { return my * mb_cols + mx; }

		// The block at (bx, by), or -1 off the picture.
		int at(const int bx, const int by) const
		{
			return bx >= 0 && by >= 0 && bx < cols && by < rows ? by * cols + bx : -1;
		}

		// Block `which`, in Z order, of macroblock (mx, my), or -1 where the picture ends first.
		int part(const int mx, const int my, const int which) const
		{
			return at(2 * mx + (which & 1), 2 * my + (which >> 1));
		}
	};

	// What a packet has done so far, in the order it is decoded: what vectors are predicted
	// from, what contexts read and, once the packet's records are all in, what the loop filter
	// reads. Both ends fill it identically.
	struct packet_state
	{
		std::vector<uint8_t> macroblocks; // each macroblock: untouched, skip, inter or intra
		std::vector<uint8_t> moving;      // each block: carries a vector, which may be zero
		std::vector<point_i> vectors;
		std::vector<uint8_t> modes; // each block's intra direction, DC for a block without one
		std::vector<uint8_t> coded; // each block: carries a residual
		point_i last_vector{0, 0};

		void reset(const block_grid& grid)
		{
			const auto blocks = static_cast<size_t>(grid.blocks());
			macroblocks.assign(static_cast<size_t>(grid.macroblocks()), mb_untouched);
			moving.assign(blocks, 0);
			vectors.assign(blocks, point_i{});
			modes.assign(blocks, static_cast<uint8_t>(mode_dc));
			coded.assign(blocks, 0);
			last_vector = {};
		}

		// Forgets what was tried on one macroblock.
		void clear(const block_grid& grid, const int mx, const int my)
		{
			macroblocks[grid.macroblock(mx, my)] = mb_untouched;
			for (int which = 0; which < 4; ++which)
			{
				const int index = grid.part(mx, my, which);
				if (index < 0) continue;
				moving[index] = 0;
				vectors[index] = {};
				modes[index] = static_cast<uint8_t>(mode_dc);
				coded[index] = 0;
			}
		}
	};

	// The one reconstruction both ends run: the predictor plus the dequantised residual,
	// rounded and clamped.
	void reconstruct_block(const uint8_t* predictor, const int16_t* levels, const int step, uint8_t* out)
	{
		const float* steps = tables().step[step];
		float coefficients[block_area];
		bool any = false;
		for (int i = 0; i < block_area; ++i)
		{
			coefficients[i] = static_cast<float>(levels[i]) * steps[i];
			any = any || levels[i] != 0;
		}
		if (!any)
		{
			std::copy_n(predictor, block_area, out);
			return;
		}
		float samples[block_area];
		inverse_dct(coefficients, samples);
		for (int i = 0; i < block_area; ++i)
			out[i] = static_cast<uint8_t>(
				std::clamp(static_cast<int>(predictor[i]) + static_cast<int>(std::lround(samples[i])), 0, 255));
	}

	struct residual
	{
		int16_t levels[block_area] = {}; // natural order
		bool coded = false;
		double distortion = 0.0; // squared error left, which the orthonormal DCT makes the pixels'
		double bits = 0.0;
	};

	void transform_residual(const uint8_t* source, const uint8_t* predictor, float* coefficients)
	{
		float samples[block_area];
		for (int i = 0; i < block_area; ++i) samples[i] = static_cast<float>(source[i]) - predictor[i];
		forward_dct(samples, coefficients);
	}

	// What coding these levels costs against the models as they stand.
	double residual_bits(const wire_models& models, const int set, const int16_t* levels)
	{
		int last = -1;
		for (int i = 0; i < block_area; ++i)
			if (levels[scan_order[i]] != 0) last = i;
		if (last < 0) return 0.0;

		double bits = 0.0;
		for (int i = 0; i <= last && i < block_area - 1; ++i)
		{
			const int nonzero = levels[scan_order[i]] != 0 ? 1 : 0;
			bits += bit_cost(models.significant[set][i], nonzero);
			if (nonzero != 0) bits += bit_cost(models.last[set][i], i == last ? 1 : 0);
		}
		int ones = 0;
		int greater = 0;
		for (int i = last; i >= 0; --i)
		{
			const int magnitude = std::abs(levels[scan_order[i]]);
			if (magnitude == 0) continue;
			const int level_class = set * 2 + (i == 0 ? 1 : 0);
			const int context = greater != 0 ? 0 : std::min(4, 1 + ones);
			bits += 1.0 + bit_cost(models.greater_one[level_class][context], magnitude > 1 ? 1 : 0);
			if (magnitude > 1)
			{
				bits += golomb_cost(models.remainder[level_class], 8, static_cast<uint32_t>(magnitude - 2));
				++greater;
			}
			else ++ones;
		}
		return bits;
	}

	void encode_residual(range_encoder& out, wire_models& models, const int set, const int16_t* levels)
	{
		int last = -1;
		for (int i = 0; i < block_area; ++i)
			if (levels[scan_order[i]] != 0) last = i;
		for (int i = 0; i <= last && i < block_area - 1; ++i)
		{
			const int nonzero = levels[scan_order[i]] != 0 ? 1 : 0;
			out.encode(models.significant[set][i], nonzero);
			if (nonzero != 0) out.encode(models.last[set][i], i == last ? 1 : 0);
		}
		int ones = 0;
		int greater = 0;
		for (int i = last; i >= 0; --i)
		{
			const int level = levels[scan_order[i]];
			if (level == 0) continue;
			const int magnitude = std::abs(level);
			const int level_class = set * 2 + (i == 0 ? 1 : 0);
			const int context = greater != 0 ? 0 : std::min(4, 1 + ones);
			out.encode(models.greater_one[level_class][context], magnitude > 1 ? 1 : 0);
			if (magnitude > 1)
			{
				encode_golomb(out, models.remainder[level_class], 8, static_cast<uint32_t>(magnitude - 2));
				++greater;
			}
			else ++ones;
			out.encode_bypass(level < 0 ? 1 : 0, 1);
		}
	}

	// The significance map ends at the last coefficient that is not zero, so a block with a
	// residual always has one; the final scan position is significant by implication.
	bool decode_residual(range_decoder& in, wire_models& models, const int set, int16_t* levels)
	{
		bool significant[block_area] = {};
		int last = block_area - 1;
		for (int i = 0; i < block_area - 1; ++i)
		{
			if (in.decode(models.significant[set][i]) != 0)
			{
				significant[i] = true;
				if (in.decode(models.last[set][i]) != 0)
				{
					last = i;
					break;
				}
			}
			if (!in.ok()) return false;
		}
		if (last == block_area - 1) significant[last] = true;

		int ones = 0;
		int greater = 0;
		for (int i = last; i >= 0 && in.ok(); --i)
		{
			if (!significant[i]) continue;
			const int level_class = set * 2 + (i == 0 ? 1 : 0);
			const int context = greater != 0 ? 0 : std::min(4, 1 + ones);
			uint32_t magnitude = 1;
			if (in.decode(models.greater_one[level_class][context]) != 0)
			{
				magnitude = 2 + decode_golomb(in, models.remainder[level_class], 8);
				++greater;
			}
			else ++ones;
			if (magnitude > 32767)
			{
				in.fail();
				return false;
			}
			const bool negative = in.decode_bypass(1) != 0;
			levels[scan_order[i]] = static_cast<int16_t>(negative ? -static_cast<int>(magnitude)
			                                                      : static_cast<int>(magnitude));
		}
		return in.ok();
	}

	// Dead-zone quantisation, then, when lambda is not zero, every level of one or two from the
	// highest frequency down is lowered by one if the bits that saves are worth more than the
	// error it adds. `bits` is what the levels left cost.
	void quantise(const float* coefficients, const int step, const float rounding, const double lambda,
	              const wire_models& models, const int set, residual& out)
	{
		const float* steps = tables().step[step];
		out.distortion = 0.0;
		for (int i = 0; i < block_area; ++i)
		{
			// Truncation is the floor of a value that is never negative, without a library call.
			const auto magnitude = static_cast<int>(std::fabs(coefficients[i]) / steps[i] + rounding);
			const int level = std::min(magnitude, 32767);
			out.levels[i] = static_cast<int16_t>(coefficients[i] < 0.0f ? -level : level);
			const double error = coefficients[i] - static_cast<double>(out.levels[i]) * steps[i];
			out.distortion += error * error;
		}

		double bits = residual_bits(models, set, out.levels);
		if (lambda > 0.0)
			for (int i = block_area - 1; i >= 0; --i)
			{
				const int position = scan_order[i];
				const int level = out.levels[position];
				if (level == 0 || std::abs(level) > 2) continue;
				const int lowered = level > 0 ? level - 1 : level + 1;
				const double before = coefficients[position] - static_cast<double>(level) * steps[position];
				const double after = coefficients[position] - static_cast<double>(lowered) * steps[position];
				out.levels[position] = static_cast<int16_t>(lowered);
				const double trial = residual_bits(models, set, out.levels);
				if (after * after - before * before < lambda * (bits - trial))
				{
					out.distortion += after * after - before * before;
					bits = trial;
				}
				else out.levels[position] = static_cast<int16_t>(level);
			}

		out.coded = std::any_of(std::begin(out.levels), std::end(out.levels), [](const int16_t l) { return l != 0; });
		out.bits = out.coded ? bits : 0.0;
	}

	// The samples an intra block is predicted from, smoothed by [1, 2, 1] as H.264's 8x8 intra
	// prediction smooths them: sixteen above - the eight past the block's right edge repeating
	// the eighth where the block above right is not decoded yet - eight to the left, and the
	// corner, which both arrays hold at index 0.
	struct intra_references
	{
		bool top = false;
		bool left = false;
		int above[17] = {};
		int beside[9] = {};
	};

	// A block's neighbours are decoded before it unless they lie to the right of it in its own
	// macroblock row: the block above right of a macroblock's fourth block is the next
	// macroblock's first.
	intra_references gather_references(const image_u8& picture, const block_grid& grid, const int bx, const int by)
	{
		intra_references r;
		r.top = by > 0;
		r.left = bx > 0;
		const bool corner = r.top && r.left;
		const bool top_right = r.top && bx + 1 < grid.cols && !((bx & 1) != 0 && (by & 1) != 0);
		const int x0 = bx * block;
		const int y0 = by * block;
		const int last_x = picture.width() - 1;
		const int last_y = picture.height() - 1;
		int top[16] = {};
		int left[8] = {};
		int top_left = 0;
		if (r.top)
		{
			const uint8_t* row = picture.row(y0 - 1);
			for (int x = 0; x < 16; ++x)
				top[x] = x < block || top_right ? row[std::min(x0 + x, last_x)] : top[block - 1];
		}
		if (r.left)
			for (int y = 0; y < block; ++y) left[y] = picture(x0 - 1, std::min(y0 + y, last_y));
		if (corner) top_left = picture(x0 - 1, y0 - 1);

		if (r.top)
		{
			r.above[1] = corner ? (top_left + 2 * top[0] + top[1] + 2) >> 2 : (3 * top[0] + top[1] + 2) >> 2;
			for (int x = 1; x < 15; ++x) r.above[1 + x] = (top[x - 1] + 2 * top[x] + top[x + 1] + 2) >> 2;
			r.above[16] = (top[14] + 3 * top[15] + 2) >> 2;
		}
		if (r.left)
		{
			r.beside[1] = corner ? (top_left + 2 * left[0] + left[1] + 2) >> 2 : (3 * left[0] + left[1] + 2) >> 2;
			for (int y = 1; y < 7; ++y) r.beside[1 + y] = (left[y - 1] + 2 * left[y] + left[y + 1] + 2) >> 2;
			r.beside[8] = (left[6] + 3 * left[7] + 2) >> 2;
		}
		if (corner)
		{
			r.above[0] = (top[0] + 2 * top_left + left[0] + 2) >> 2;
			r.beside[0] = r.above[0];
		}
		return r;
	}

	// Which directions a block can take: those whose samples exist.
	bool mode_allowed(const bool top, const bool left, const int mode)
	{
		switch (mode)
		{
		case mode_dc: return true;
		case mode_vertical:
		case mode_down_left:
		case mode_vertical_left: return top;
		case mode_horizontal:
		case mode_horizontal_up: return left;
		default: return top && left;
		}
	}

	// H.264's 8x8 intra prediction along one of its nine directions, each direction its own loop.
	void predict_intra(const intra_references& r, const int mode, uint8_t* out)
	{
		const auto t = [&](const int i)
		{
			return r.above[i + 1];
		};
		const auto l = [&](const int j)
		{
			return r.beside[j + 1];
		};
		const auto fill = [&](const auto& value)
		{
			for (int y = 0; y < block; ++y)
				for (int x = 0; x < block; ++x) out[y * block + x] = static_cast<uint8_t>(value(x, y));
		};
		switch (mode)
		{
		case mode_vertical: fill([&](const int x, int) { return t(x); }); return;
		case mode_horizontal: fill([&](int, const int y) { return l(y); }); return;
		case mode_down_left:
			fill([&](const int x, const int y) {
				return x == 7 && y == 7 ? (t(14) + 3 * t(15) + 2) >> 2
				                        : (t(x + y) + 2 * t(x + y + 1) + t(x + y + 2) + 2) >> 2;
			});
			return;
		case mode_down_right:
			fill([&](const int x, const int y)
			{
				if (x > y) return (t(x - y - 2) + 2 * t(x - y - 1) + t(x - y) + 2) >> 2;
				if (x < y) return (l(y - x - 2) + 2 * l(y - x - 1) + l(y - x) + 2) >> 2;
				return (t(0) + 2 * t(-1) + l(0) + 2) >> 2;
			});
			return;
		case mode_vertical_right:
			fill([&](const int x, const int y)
			{
				const int z = 2 * x - y;
				const int i = x - (y >> 1);
				if (z >= 0 && (z & 1) == 0) return (t(i - 1) + t(i) + 1) >> 1;
				if (z >= 0) return (t(i - 2) + 2 * t(i - 1) + t(i) + 2) >> 2;
				if (z == -1) return (l(0) + 2 * l(-1) + t(0) + 2) >> 2;
				return (l(y - 2 * x - 1) + 2 * l(y - 2 * x - 2) + l(y - 2 * x - 3) + 2) >> 2;
			});
			return;
		case mode_horizontal_down:
			fill([&](const int x, const int y)
			{
				const int z = 2 * y - x;
				const int j = y - (x >> 1);
				if (z >= 0 && (z & 1) == 0) return (l(j - 1) + l(j) + 1) >> 1;
				if (z >= 0) return (l(j - 2) + 2 * l(j - 1) + l(j) + 2) >> 2;
				if (z == -1) return (l(0) + 2 * l(-1) + t(0) + 2) >> 2;
				return (t(x - 2 * y - 1) + 2 * t(x - 2 * y - 2) + t(x - 2 * y - 3) + 2) >> 2;
			});
			return;
		case mode_vertical_left:
			fill([&](const int x, const int y)
			{
				const int i = x + (y >> 1);
				return (y & 1) == 0 ? (t(i) + t(i + 1) + 1) >> 1 : (t(i) + 2 * t(i + 1) + t(i + 2) + 2) >> 2;
			});
			return;
		case mode_horizontal_up:
			fill([&](const int x, const int y)
			{
				const int z = x + 2 * y;
				const int j = y + (x >> 1);
				if (z > 13) return l(7);
				if (z == 13) return (l(6) + 3 * l(7) + 2) >> 2;
				if ((z & 1) == 0) return (l(j) + l(j + 1) + 1) >> 1;
				return (l(j) + 2 * l(j + 1) + l(j + 2) + 2) >> 2;
			});
			return;
		default:
		{
			int sum = 0;
			for (int i = 0; i < block; ++i) sum += (r.top ? t(i) : 0) + (r.left ? l(i) : 0);
			const int dc = r.top && r.left ? (sum + 8) >> 4 : r.top || r.left ? (sum + 4) >> 3 : 128;
			std::fill_n(out, block_area, static_cast<uint8_t>(dc));
			return;
		}
		}
	}

	// The direction an intra block most probably takes: the lesser of its left and upper
	// neighbours', counting DC for a neighbour that is not intra, and DC outright at the
	// picture's edge.
	int probable_mode(const block_grid& grid, const packet_state& state, const int bx, const int by)
	{
		if (bx == 0 || by == 0) return mode_dc;
		return std::min(state.modes[grid.at(bx - 1, by)], state.modes[grid.at(bx, by - 1)]);
	}

	// A direction is the probable one, or names one of the eight others in three decisions.
	void encode_mode(range_encoder& out, wire_models& models, const int mode, const int probable)
	{
		out.encode(models.probable_mode, mode == probable ? 1 : 0);
		if (mode == probable) return;
		const int rest = mode < probable ? mode : mode - 1;
		const int high = rest >> 2;
		const int middle = (rest >> 1) & 1;
		out.encode(models.mode[0], high);
		out.encode(models.mode[1 + high], middle);
		out.encode(models.mode[3 + high * 2 + middle], rest & 1);
	}

	double mode_bits(const wire_models& models, const int mode, const int probable)
	{
		if (mode == probable) return bit_cost(models.probable_mode, 1);
		const int rest = mode < probable ? mode : mode - 1;
		const int high = rest >> 2;
		const int middle = (rest >> 1) & 1;
		return bit_cost(models.probable_mode, 0) + bit_cost(models.mode[0], high) +
		       bit_cost(models.mode[1 + high], middle) + bit_cost(models.mode[3 + high * 2 + middle], rest & 1);
	}

	int decode_mode(range_decoder& in, wire_models& models, const int probable)
	{
		if (in.decode(models.probable_mode) != 0) return probable;
		const int high = in.decode(models.mode[0]);
		const int middle = in.decode(models.mode[1 + high]);
		const int rest = high * 4 + middle * 2 + in.decode(models.mode[3 + high * 2 + middle]);
		return rest < probable ? rest : rest + 1;
	}

	// A block's residual flag is coded against how many of its left and upper neighbours
	// carry one this packet: residuals cluster where the picture changed.
	int coded_context(const block_grid& grid, const packet_state& state, const int bx, const int by)
	{
		const int left = bx > 0 ? state.coded[grid.at(bx - 1, by)] : 0;
		const int above = by > 0 ? state.coded[grid.at(bx, by - 1)] : 0;
		return left + above;
	}

	// How many of a macroblock's left and upper neighbours this packet coded as `mode`.
	int neighbour_count(const block_grid& grid, const packet_state& state, const int mx, const int my,
	                    const uint8_t mode)
	{
		const int left = mx > 0 && state.macroblocks[grid.macroblock(mx - 1, my)] == mode ? 1 : 0;
		const int above = my > 0 && state.macroblocks[grid.macroblock(mx, my - 1)] == mode ? 1 : 0;
		return left + above;
	}

	// Whether a macroblock is updated is coded in raster order against whether its left and
	// upper neighbours were, and whether it was in the previous packet: updates cluster where
	// the face is, in space and in time.
	int updated_context(const block_grid& grid, const packet_state& state, const std::vector<uint8_t>& recent,
	                    const int mx, const int my)
	{
		const int left = mx > 0 && state.macroblocks[grid.macroblock(mx - 1, my)] != mb_untouched ? 1 : 0;
		const int above = my > 0 && state.macroblocks[grid.macroblock(mx, my - 1)] != mb_untouched ? 1 : 0;
		return left + above + (recent[grid.macroblock(mx, my)] != 0 ? 3 : 0);
	}

	struct block_weight
	{
		int step = 0;
		double importance = 32.0;
	};

	// A macroblock's importance and the quantiser step it selects: `quality` for the
	// background's 32, finer in whole steps towards the expression prior's 255. The importance
	// is its most important block's mean, so a face that only clips a macroblock still gets
	// the face's quantiser there.
	block_weight weigh_macroblock(const image_u8* importance, const block_grid& grid, const int mx, const int my,
	                              const int quality)
	{
		double strongest = 0.0;
		for (int which = 0; which < 4; ++which)
		{
			if (grid.part(mx, my, which) < 0) continue;
			const int left = (2 * mx + (which & 1)) * block;
			const int top = (2 * my + (which >> 1)) * block;
			int block_sum = 0;
			int block_count = 0;
			for (int y = top; y < std::min(top + block, grid.height); ++y)
				for (int x = left; x < std::min(left + block, grid.width); ++x, ++block_count)
					block_sum += importance != nullptr && importance->contains(x, y) ? (*importance)(x, y) : 32;
			strongest = std::max(strongest, block_count > 0 ? static_cast<double>(block_sum) / block_count : 32.0);
		}
		block_weight weight;
		weight.importance = strongest;
		const int finer = static_cast<int>(
			std::lround(face_steps * std::clamp((weight.importance - 32.0) / 223.0, 0.0, 1.0)));
		weight.step = std::clamp(quality - finer, 0, ladder_steps - 1);
		return weight;
	}

	std::vector<block_weight> weigh_macroblocks(const image_u8* importance, const block_grid& grid, const int quality)
	{
		std::vector<block_weight> weights(static_cast<size_t>(grid.macroblocks()));
		for (int my = 0; my < grid.mb_rows; ++my)
			for (int mx = 0; mx < grid.mb_cols; ++mx)
				weights[grid.macroblock(mx, my)] = weigh_macroblock(importance, grid, mx, my, quality);
		return weights;
	}

	// Importance in [32,255] mapped onto a bounded gain. Bounded matters: an unbounded
	// preference for face macroblocks starves every other macroblock that needs repair.
	double importance_gain(const double average_importance)
	{
		return 1.0 + 7.0 * std::clamp((average_importance - 32.0) / 223.0, 0.0, 1.0);
	}

	// What sending a vector difference costs, component by component, against the models as
	// the packet starts.
	class vector_costs
	{
	public:
		explicit vector_costs(const wire_models& models)
		{
			for (int axis = 0; axis < 2; ++axis)
			{
				double least = std::numeric_limits<double>::infinity();
				for (int d = -span; d <= span; ++d)
				{
					bits[axis][d + span] = signed_cost(models.vector[axis], d);
					least = std::min(least, bits[axis][d + span]);
				}
				cheapest += least;
			}
		}

		double operator()(const point_i d) const
		{
			return bits[0][std::clamp(d.x, -span, span) + span] + bits[1][std::clamp(d.y, -span, span) + span];
		}

		// The least any vector difference costs.
		double floor() const { return cheapest; }

	private:
		static constexpr int span = 2 * max_motion;
		double bits[2][2 * span + 1] = {};
		double cheapest = 0.0;
	};

	// The blocks one vector moves: a whole macroblock's, or one block.
	struct partition
	{
		int count = 0;
		point_i origin[4];              // each block's top-left pixel
		const uint8_t* samples[4] = {}; // each block of the source
	};

	int partition_error(const half_pel_planes& reference, const partition& blocks, const point_i vector)
	{
		int total = 0;
		for (int i = 0; i < blocks.count; ++i)
			total += motion_error(reference, blocks.origin[i].x, blocks.origin[i].y, vector, blocks.samples[i]);
		return total;
	}

	// The vector that minimises a partition's sum of absolute differences plus `lambda` times
	// what the vector costs to send against its predictor: seeded with the predictor and the
	// vectors around it, then a square pattern from `widest` quarter pixels down to one.
	point_i search_motion(const half_pel_planes& reference, const partition& blocks, const point_i predictor,
	                      const point_i* seeds, const int seed_count, const int widest, const double lambda,
	                      const vector_costs& costs)
	{
		const auto inside = [](const point_i vector)
		{
			return std::abs(vector.x) <= max_motion && std::abs(vector.y) <= max_motion;
		};
		const auto cost = [&](const point_i vector)
		{
			return partition_error(reference, blocks, vector) + lambda * costs(difference(vector, predictor));
		};

		point_i best = inside(predictor) ? predictor : point_i{};
		double best_cost = cost(best);
		for (int i = 0; i < seed_count; ++i)
		{
			if (seeds[i] == best || !inside(seeds[i])) continue;
			const double trial = cost(seeds[i]);
			if (trial < best_cost)
			{
				best_cost = trial;
				best = seeds[i];
			}
		}
		for (int step = widest; step >= 1; step /= 2)
		{
			const point_i centre = best;
			for (int dy = -1; dy <= 1; ++dy)
				for (int dx = -1; dx <= 1; ++dx)
				{
					const point_i trial{centre.x + dx * step, centre.y + dy * step};
					if ((dx == 0 && dy == 0) || !inside(trial)) continue;
					const double trial_cost = cost(trial);
					if (trial_cost < best_cost)
					{
						best_cost = trial_cost;
						best = trial;
					}
				}
		}
		return best;
	}

	bool same_layout(const face_geometry& a, const face_geometry& b)
	{
		if (a.size() != b.size()) return false;
		for (size_t i = 0; i < a.size(); ++i)
			if (a[i].size() != b[i].size()) return false;
		return true;
	}

	int median(std::vector<int> values)
	{
		if (values.empty()) return 0;
		std::nth_element(values.begin(), values.begin() + static_cast<ptrdiff_t>(values.size() / 2), values.end());
		return values[values.size() / 2];
	}

	// Landmarks travel as an absolute layout when the faces' number or shape changes, and
	// otherwise as each face's rigid shift - the median of its points' motion, what the head
	// did - plus each point's residual from it, what the expression did.
	void encode_geometry(range_encoder& out, wire_models& models, const face_geometry& previous,
	                     const face_geometry& next, const bool absolute)
	{
		const bool layout = absolute || !same_layout(previous, next);
		if (!absolute) out.encode(models.layout, layout ? 1 : 0);
		if (layout)
		{
			encode_golomb(out, models.faces, 6, static_cast<uint32_t>(next.size()));
			for (const auto& face : next)
			{
				encode_golomb(out, models.points, 6, static_cast<uint32_t>(face.size()));
				point_i at{0, 0};
				for (size_t i = 0; i < face.size(); ++i)
				{
					signed_models* axes = i == 0 ? models.first : models.along;
					encode_signed(out, axes[0], face[i].x - at.x);
					encode_signed(out, axes[1], face[i].y - at.y);
					at = face[i];
				}
			}
			return;
		}

		for (size_t f = 0; f < next.size(); ++f)
		{
			const auto& before = previous[f];
			const auto& after = next[f];
			std::vector<int> moved_x;
			std::vector<int> moved_y;
			for (size_t i = 0; i < after.size(); ++i)
			{
				moved_x.push_back(after[i].x - before[i].x);
				moved_y.push_back(after[i].y - before[i].y);
			}
			const bool moved = std::any_of(moved_x.begin(), moved_x.end(), [](const int d) { return d != 0; }) ||
			                   std::any_of(moved_y.begin(), moved_y.end(), [](const int d) { return d != 0; });
			out.encode(models.face_moved, moved ? 1 : 0);
			if (!moved) continue;
			const point_i shift{median(moved_x), median(moved_y)};
			encode_signed(out, models.shift[0], shift.x);
			encode_signed(out, models.shift[1], shift.y);
			for (size_t i = 0; i < after.size(); ++i)
			{
				auto& point_models = models.residual[std::min<size_t>(i, 4)];
				encode_signed(out, point_models[0], moved_x[i] - shift.x);
				encode_signed(out, point_models[1], moved_y[i] - shift.y);
			}
		}
	}

	bool decode_geometry(range_decoder& in, wire_models& models, const face_geometry& previous, const bool absolute,
	                     face_geometry& next)
	{
		const auto in_bounds = [](const int64_t value)
		{
			return value >= -coordinate_bound && value <= coordinate_bound;
		};
		const bool layout = absolute || in.decode(models.layout) != 0;
		if (layout)
		{
			const uint32_t face_count = decode_golomb(in, models.faces, 6);
			if (!in.ok() || face_count > max_faces) return false;
			next.assign(face_count, {});
			for (auto& face : next)
			{
				const uint32_t point_count = decode_golomb(in, models.points, 6);
				if (!in.ok() || point_count > max_points) return false;
				face.resize(point_count);
				point_i at{0, 0};
				for (size_t i = 0; i < face.size(); ++i)
				{
					signed_models* axes = i == 0 ? models.first : models.along;
					const int64_t x = static_cast<int64_t>(at.x) + decode_signed(in, axes[0]);
					const int64_t y = static_cast<int64_t>(at.y) + decode_signed(in, axes[1]);
					if (!in.ok() || !in_bounds(x) || !in_bounds(y)) return false;
					face[i] = {static_cast<int>(x), static_cast<int>(y)};
					at = face[i];
				}
			}
			return in.ok();
		}

		next = previous;
		for (auto& face : next)
		{
			if (in.decode(models.face_moved) == 0) continue;
			const int shift_x = decode_signed(in, models.shift[0]);
			const int shift_y = decode_signed(in, models.shift[1]);
			for (size_t i = 0; i < face.size(); ++i)
			{
				auto& point_models = models.residual[std::min<size_t>(i, 4)];
				const int64_t x = static_cast<int64_t>(face[i].x) + shift_x + decode_signed(in, point_models[0]);
				const int64_t y = static_cast<int64_t>(face[i].y) + shift_y + decode_signed(in, point_models[1]);
				if (!in.ok() || !in_bounds(x) || !in_bounds(y)) return false;
				face[i] = {static_cast<int>(x), static_cast<int>(y)};
			}
		}
		return in.ok();
	}

	face_geometry bounded_faces(const face_geometry& faces)
	{
		face_geometry result(faces.begin(),
		                     faces.begin() + static_cast<ptrdiff_t>(std::min<size_t>(faces.size(), max_faces)));
		for (auto& face : result)
		{
			if (face.size() > max_points) face.resize(max_points);
			for (auto& point : face)
			{
				point.x = std::clamp(point.x, -coordinate_bound, coordinate_bound);
				point.y = std::clamp(point.y, -coordinate_bound, coordinate_bound);
			}
		}
		return result;
	}

	// What a vector is predicted from, at both ends: the median of the vectors on the left,
	// above and above right - above left where above right is not decoded yet - when this
	// packet moved all three, else the first of them it moved, else the last vector it sent.
	// One moving head moves its blocks alike. (x, y) is the partition's top-left block and
	// `span` its width in blocks.
	point_i predict_vector(const block_grid& grid, const packet_state& state, const int x, const int y, const int span)
	{
		point_i found[3];
		int count = 0;
		const auto take = [&](const int bx, const int by)
		{
			const int index = grid.at(bx, by);
			if (index < 0 || state.moving[index] == 0 || count == 3) return false;
			found[count++] = state.vectors[index];
			return true;
		};
		take(x - 1, y);
		take(x, y - 1);
		if (!take(x + span, y - 1)) take(x - 1, y - 1);
		if (count == 0) return state.last_vector;
		if (count < 3) return found[0];
		const auto middle = [](const int a, const int b, const int c)
		{
			return std::max(std::min(a, b), std::min(std::max(a, b), c));
		};
		return {middle(found[0].x, found[1].x, found[2].x), middle(found[0].y, found[1].y, found[2].y)};
	}

	// H.264's loop filter thresholds and limits, by an index that is a quantiser step plus 16:
	// where the two ladders meet, H.264's quantiser parameter 16 being a step of 4.
	// clang-format off
	constexpr uint8_t filter_alpha[52] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28, 32, 36, 40, 45,
		50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255};
	constexpr uint8_t filter_beta[52] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
		11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18};
	constexpr uint8_t filter_clip[52][3] = {
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
		{0, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 1}, {1, 1, 1},
		{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 2}, {1, 1, 2}, {1, 1, 2}, {1, 1, 2}, {1, 2, 3},
		{1, 2, 3}, {2, 2, 3}, {2, 2, 4}, {2, 3, 4}, {2, 3, 4}, {3, 3, 5}, {3, 4, 6}, {3, 4, 6},
		{4, 5, 7}, {4, 5, 8}, {4, 6, 9}, {5, 7, 10}, {6, 8, 11}, {6, 8, 13}, {7, 10, 14}, {8, 11, 16},
		{9, 12, 18}, {10, 13, 20}, {11, 15, 23}, {13, 17, 25}};
	// clang-format on

	// One edge position: p3..p0 before it and q0..q3 after, `stride` apart. A step across the
	// edge smaller than the quantiser explains, between two sides that are each nearly flat,
	// is the quantiser's doing. Strength 4, an intra macroblock's edge, blends flat sides
	// across three pixels each way; weaker edges move the two pixels nearest by a clipped
	// amount.
	void filter_samples(uint8_t* q0_at, const ptrdiff_t stride, const int strength, const int index)
	{
		const int alpha = filter_alpha[index];
		const int beta = filter_beta[index];
		const int p0 = q0_at[-stride];
		const int q0 = q0_at[0];
		const int p1 = q0_at[-2 * stride];
		const int q1 = q0_at[stride];
		if (std::abs(p0 - q0) >= alpha || std::abs(p1 - p0) >= beta || std::abs(q1 - q0) >= beta) return;
		const int p2 = q0_at[-3 * stride];
		const int q2 = q0_at[2 * stride];
		const bool flat_p = std::abs(p2 - p0) < beta;
		const bool flat_q = std::abs(q2 - q0) < beta;
		const auto put = [&](const ptrdiff_t offset, const int value)
		{
			q0_at[offset * stride] = static_cast<uint8_t>(std::clamp(value, 0, 255));
		};

		if (strength < 4)
		{
			const int limit = filter_clip[index][strength - 1];
			const int clip = limit + (flat_p ? 1 : 0) + (flat_q ? 1 : 0);
			const int delta = std::clamp(((q0 - p0) * 4 + (p1 - q1) + 4) >> 3, -clip, clip);
			put(-1, p0 + delta);
			put(0, q0 - delta);
			if (flat_p) put(-2, p1 + std::clamp((p2 + ((p0 + q0 + 1) >> 1) - p1 * 2) >> 1, -limit, limit));
			if (flat_q) put(1, q1 + std::clamp((q2 + ((p0 + q0 + 1) >> 1) - q1 * 2) >> 1, -limit, limit));
			return;
		}

		const int p3 = q0_at[-4 * stride];
		const int q3 = q0_at[3 * stride];
		const bool small = std::abs(p0 - q0) < (alpha >> 2) + 2;
		if (flat_p && small)
		{
			put(-1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
			put(-2, (p2 + p1 + p0 + q0 + 2) >> 2);
			put(-3, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
		}
		else put(-1, (2 * p1 + p0 + q1 + 2) >> 2);
		if (flat_q && small)
		{
			put(0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
			put(1, (p0 + q0 + q1 + q2 + 2) >> 2);
			put(2, (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
		}
		else put(0, (2 * q1 + q0 + p1 + 2) >> 2);
	}

	// The loop filter, which both ends run once a packet's records are in, so what it leaves
	// is what the next packet predicts from. It visits every edge touching a macroblock the
	// packet updated, macroblock by macroblock in raster order, the vertical edges of each
	// before its horizontal ones, as H.264 does. An edge's strength is 4 where an intra
	// macroblock meets another and 3 inside one, 2 where either side carries a residual, 1
	// where the two sides moved a pixel or more apart, an untouched block holding still, and
	// otherwise 0: nothing there is the codec's doing.
	void loop_filter(image_u8& picture, const block_grid& grid, const packet_state& state,
	                 const std::vector<uint8_t>& steps)
	{
		const int width = picture.width();
		const int height = picture.height();
		const auto touched = [&](const int mx, const int my)
		{
			return state.macroblocks[grid.macroblock(mx, my)] != mb_untouched;
		};
		const auto intra = [&](const int bx, const int by)
		{
			return state.macroblocks[grid.macroblock(bx / 2, by / 2)] == mb_intra;
		};
		const auto motion = [&](const int index)
		{
			return state.moving[index] != 0 ? state.vectors[index] : point_i{};
		};
		const auto edge = [&](const int pbx, const int pby, const int qbx, const int qby, const bool outer,
		                      uint8_t* q0_at, const ptrdiff_t across, const ptrdiff_t along, const int length)
		{
			const int p = grid.at(pbx, pby);
			const int q = grid.at(qbx, qby);
			int strength = 0;
			if (intra(pbx, pby) || intra(qbx, qby)) strength = outer ? 4 : 3;
			else if (state.coded[p] != 0 || state.coded[q] != 0) strength = 2;
			else
			{
				const point_i a = motion(p);
				const point_i b = motion(q);
				strength = std::abs(a.x - b.x) >= 4 || std::abs(a.y - b.y) >= 4 ? 1 : 0;
			}
			if (strength == 0) return;
			const int index = std::clamp((steps[p] + steps[q] + 1) / 2 + 16, 0, 51);
			for (int i = 0; i < length; ++i) filter_samples(q0_at + i * along, across, strength, index);
		};

		for (int my = 0; my < grid.mb_rows; ++my)
			for (int mx = 0; mx < grid.mb_cols; ++mx)
			{
				const bool here = touched(mx, my);
				for (int column = 0; column < 2; ++column)
				{
					const int bx = 2 * mx + column;
					const bool outer = column == 0;
					if (bx == 0 || bx >= grid.cols || bx * block + 4 > width) continue;
					if (!here && !(outer && touched(mx - 1, my))) continue;
					for (int row = 0; row < 2; ++row)
					{
						const int by = 2 * my + row;
						if (by >= grid.rows) continue;
						const int y = by * block;
						edge(bx - 1, by, bx, by, outer, picture.row(y) + bx * block, 1, width,
						     std::min(block, height - y));
					}
				}
				for (int row = 0; row < 2; ++row)
				{
					const int by = 2 * my + row;
					const bool outer = row == 0;
					if (by == 0 || by >= grid.rows || by * block + 4 > height) continue;
					if (!here && !(outer && touched(mx, my - 1))) continue;
					for (int column = 0; column < 2; ++column)
					{
						const int bx = 2 * mx + column;
						if (bx >= grid.cols) continue;
						const int x = bx * block;
						edge(bx, by - 1, bx, by, outer, picture.row(by * block) + x, width, 1,
						     std::min(block, width - x));
					}
				}
			}
	}

	struct intra_macroblock
	{
		uint8_t modes[4] = {};
		bool coded[4] = {};
		int16_t levels[4][block_area] = {};
		uint8_t pictures[4][block_area] = {}; // each block as reconstructed
		double distortion = 0.0;
		double bits = 0.0; // directions, flags and residuals, not the step
	};

	// Three butterfly stages across eight registers: one 8-point Walsh-Hadamard transform for
	// every lane at once.
	void hadamard_columns(__m128i* rows)
	{
		for (int span = block / 2; span >= 1; span /= 2)
			for (int i = 0; i < block; i += 2 * span)
				for (int j = i; j < i + span; ++j)
				{
					const __m128i a = rows[j];
					const __m128i b = rows[j + span];
					rows[j] = _mm_add_epi16(a, b);
					rows[j + span] = _mm_sub_epi16(a, b);
				}
	}

	void transpose_8x8(__m128i* rows)
	{
		const __m128i a0 = _mm_unpacklo_epi16(rows[0], rows[1]);
		const __m128i a1 = _mm_unpackhi_epi16(rows[0], rows[1]);
		const __m128i a2 = _mm_unpacklo_epi16(rows[2], rows[3]);
		const __m128i a3 = _mm_unpackhi_epi16(rows[2], rows[3]);
		const __m128i a4 = _mm_unpacklo_epi16(rows[4], rows[5]);
		const __m128i a5 = _mm_unpackhi_epi16(rows[4], rows[5]);
		const __m128i a6 = _mm_unpacklo_epi16(rows[6], rows[7]);
		const __m128i a7 = _mm_unpackhi_epi16(rows[6], rows[7]);
		const __m128i b0 = _mm_unpacklo_epi32(a0, a2);
		const __m128i b1 = _mm_unpackhi_epi32(a0, a2);
		const __m128i b2 = _mm_unpacklo_epi32(a1, a3);
		const __m128i b3 = _mm_unpackhi_epi32(a1, a3);
		const __m128i b4 = _mm_unpacklo_epi32(a4, a6);
		const __m128i b5 = _mm_unpackhi_epi32(a4, a6);
		const __m128i b6 = _mm_unpacklo_epi32(a5, a7);
		const __m128i b7 = _mm_unpackhi_epi32(a5, a7);
		rows[0] = _mm_unpacklo_epi64(b0, b4);
		rows[1] = _mm_unpackhi_epi64(b0, b4);
		rows[2] = _mm_unpacklo_epi64(b1, b5);
		rows[3] = _mm_unpackhi_epi64(b1, b5);
		rows[4] = _mm_unpacklo_epi64(b2, b6);
		rows[5] = _mm_unpackhi_epi64(b2, b6);
		rows[6] = _mm_unpacklo_epi64(b3, b7);
		rows[7] = _mm_unpackhi_epi64(b3, b7);
	}

	// The prediction error's 8x8 Walsh-Hadamard transform summed in magnitude, scaled as if the
	// transform were orthonormal: a cheap stand-in for what the error would cost to code. The
	// columns are transformed, then the rows by way of a transpose, which leaves the
	// magnitudes as they are; no sum leaves sixteen bits.
	double hadamard_error(const uint8_t* source, const uint8_t* predicted)
	{
		const __m128i zero = _mm_setzero_si128();
		__m128i rows[block];
		for (int y = 0; y < block; y += 2)
		{
			const __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + y * block));
			const __m128i p = _mm_loadu_si128(reinterpret_cast<const __m128i*>(predicted + y * block));
			rows[y] = _mm_sub_epi16(_mm_unpacklo_epi8(s, zero), _mm_unpacklo_epi8(p, zero));
			rows[y + 1] = _mm_sub_epi16(_mm_unpackhi_epi8(s, zero), _mm_unpackhi_epi8(p, zero));
		}
		hadamard_columns(rows);
		transpose_8x8(rows);
		hadamard_columns(rows);
		const __m128i ones = _mm_set1_epi16(1);
		__m128i total = zero;
		for (const __m128i& row : rows)
			total = _mm_add_epi32(total, _mm_madd_epi16(_mm_max_epi16(row, _mm_sub_epi16(zero, row)), ones));
		total = _mm_add_epi32(total, _mm_shuffle_epi32(total, _MM_SHUFFLE(1, 0, 3, 2)));
		total = _mm_add_epi32(total, _mm_shuffle_epi32(total, _MM_SHUFFLE(2, 3, 0, 1)));
		return _mm_cvtsi128_si32(total) / 8.0;
	}

	// Chooses an intra macroblock block by block in Z order: each block's direction by
	// distortion plus lambda times bits, predicted from the picture as the blocks before it
	// left it. Writes each block into `picture` and `state` as it goes. `trim` lets the
	// quantiser trade small levels for bits; a keyframe keeps every level.
	void choose_intra(const image_u8& source, image_u8& picture, const block_grid& grid, packet_state& state,
	                  const int mx, const int my, const int step, const double lambda, const bool trim,
	                  const wire_models& models, intra_macroblock& out)
	{
		const double rank_lambda = std::sqrt(lambda);
		uint8_t samples[block_area] = {};
		float coefficients[block_area] = {};
		for (int which = 0; which < 4; ++which)
		{
			const int index = grid.part(mx, my, which);
			if (index < 0) continue;
			const int bx = 2 * mx + (which & 1);
			const int by = 2 * my + (which >> 1);
			read_block(source, bx * block, by * block, samples);
			const intra_references references = gather_references(picture, grid, bx, by);
			const int probable = probable_mode(grid, state, bx, by);
			const bit_model& flag = models.coded[0][coded_context(grid, state, bx, by)];

			// Every direction is ranked by the Hadamard transform of its prediction error plus
			// what naming it costs. Only the most promising few, and the probable direction,
			// the cheapest to name, are quantised and costed in full.
			struct ranked
			{
				double cost;
				int mode;
			};
			ranked ranking[intra_modes] = {};
			int ranked_count = 0;
			uint8_t predictions[intra_modes][block_area];
			for (int mode = 0; mode < intra_modes; ++mode)
			{
				if (!mode_allowed(references.top, references.left, mode)) continue;
				predict_intra(references, mode, predictions[mode]);
				ranking[ranked_count++] = {hadamard_error(samples, predictions[mode]) +
				                               rank_lambda * mode_bits(models, mode, probable),
				                           mode};
			}
			const int kept = std::min(ranked_count, intra_candidates);
			std::partial_sort(ranking, ranking + kept, ranking + ranked_count, [](const ranked& a, const ranked& b)
			{ return a.cost != b.cost ? a.cost < b.cost : a.mode < b.mode; });
			bool considered[intra_modes] = {};
			considered[probable] = true;
			for (int i = 0; i < kept; ++i) considered[ranking[i].mode] = true;

			double best_cost = std::numeric_limits<double>::infinity();
			double best_bits = 0.0;
			int best_mode = mode_dc;
			residual best;
			for (int mode = 0; mode < intra_modes; ++mode)
			{
				if (!considered[mode]) continue;
				transform_residual(samples, predictions[mode], coefficients);
				residual trial;
				quantise(coefficients, step, intra_rounding, trim ? lambda : 0.0, models, 0, trial);
				const double bits = mode_bits(models, mode, probable) + bit_cost(flag, trial.coded ? 1 : 0) +
				                    trial.bits;
				const double cost = trial.distortion + lambda * bits;
				if (cost < best_cost)
				{
					best_cost = cost;
					best_bits = bits;
					best_mode = mode;
					best = trial;
				}
			}
			reconstruct_block(predictions[best_mode], best.levels, step, out.pictures[which]);
			write_block(picture, bx * block, by * block, out.pictures[which]);
			out.modes[which] = static_cast<uint8_t>(best_mode);
			out.coded[which] = best.coded;
			std::copy_n(best.levels, block_area, out.levels[which]);
			out.distortion += squared_error(samples, out.pictures[which]);
			out.bits += best_bits;
			state.modes[index] = static_cast<uint8_t>(best_mode);
			state.coded[index] = best.coded ? 1 : 0;
		}
	}

	// An intra macroblock on the wire: each block's direction, then which blocks carry a
	// residual, then the step - always in a keyframe, else only when a residual needs one -
	// then the residuals.
	void encode_intra(range_encoder& out, wire_models& models, const block_grid& grid, const packet_state& state,
	                  const int mx, const int my, const intra_macroblock& choice, const int step, const bool keyframe,
	                  int& previous_step, std::vector<uint8_t>& steps)
	{
		bool any = false;
		for (int which = 0; which < 4; ++which)
			if (grid.part(mx, my, which) >= 0)
				encode_mode(out, models, choice.modes[which],
				            probable_mode(grid, state, 2 * mx + (which & 1), 2 * my + (which >> 1)));
		for (int which = 0; which < 4; ++which)
		{
			if (grid.part(mx, my, which) < 0) continue;
			out.encode(models.coded[0][coded_context(grid, state, 2 * mx + (which & 1), 2 * my + (which >> 1))],
			           choice.coded[which] ? 1 : 0);
			any = any || choice.coded[which];
		}
		if (keyframe || any)
		{
			encode_signed(out, models.step, step - previous_step);
			previous_step = step;
			for (int which = 0; which < 4; ++which)
			{
				const int index = grid.part(mx, my, which);
				if (index >= 0) steps[index] = static_cast<uint8_t>(step);
			}
		}
		for (int which = 0; which < 4; ++which)
			if (grid.part(mx, my, which) >= 0 && choice.coded[which])
				encode_residual(out, models, 0, choice.levels[which]);
	}

	bool decode_intra(range_decoder& in, wire_models& models, const block_grid& grid, packet_state& state,
	                  image_u8& picture, const int mx, const int my, const bool keyframe, int& previous_step,
	                  std::vector<uint8_t>& steps)
	{
		int modes[4] = {mode_dc, mode_dc, mode_dc, mode_dc};
		bool coded[4] = {};
		bool any = false;
		for (int which = 0; which < 4; ++which)
		{
			const int index = grid.part(mx, my, which);
			if (index < 0) continue;
			const int bx = 2 * mx + (which & 1);
			const int by = 2 * my + (which >> 1);
			modes[which] = decode_mode(in, models, probable_mode(grid, state, bx, by));
			if (!in.ok() || !mode_allowed(by > 0, bx > 0, modes[which])) return false;
			state.modes[index] = static_cast<uint8_t>(modes[which]);
		}
		for (int which = 0; which < 4; ++which)
		{
			const int index = grid.part(mx, my, which);
			if (index < 0) continue;
			coded[which] = in.decode(models.coded[0][coded_context(grid, state, 2 * mx + (which & 1),
			                                                       2 * my + (which >> 1))]) != 0;
			state.coded[index] = coded[which] ? 1 : 0;
			any = any || coded[which];
		}
		int step = previous_step;
		if (keyframe || any)
		{
			step = previous_step + decode_signed(in, models.step);
			if (!in.ok() || step < 0 || step >= ladder_steps) return false;
			previous_step = step;
		}
		int16_t levels[4][block_area] = {};
		for (int which = 0; which < 4; ++which)
			if (coded[which] && !decode_residual(in, models, 0, levels[which])) return false;

		uint8_t predicted[block_area] = {};
		uint8_t reconstructed[block_area] = {};
		for (int which = 0; which < 4; ++which)
		{
			const int index = grid.part(mx, my, which);
			if (index < 0) continue;
			const int bx = 2 * mx + (which & 1);
			const int by = 2 * my + (which >> 1);
			predict_intra(gather_references(picture, grid, bx, by), modes[which], predicted);
			reconstruct_block(predicted, levels[which], step, reconstructed);
			write_block(picture, bx * block, by * block, reconstructed);
			if (keyframe || any) steps[index] = static_cast<uint8_t>(step);
		}
		return in.ok();
	}

	// A keyframe's picture: every macroblock intra at its own step, then the loop filter over
	// every edge. intra_picture runs exactly this and throws the bytes away.
	void encode_keyframe(range_encoder& out, wire_models& models, const image_u8& source, const block_grid& grid,
	                     const std::vector<block_weight>& weights, image_u8& picture, std::vector<uint8_t>& steps)
	{
		picture = image_u8(grid.width, grid.height);
		steps.assign(static_cast<size_t>(grid.blocks()), 0);
		packet_state state;
		state.reset(grid);
		int previous_step = first_step;
		for (int my = 0; my < grid.mb_rows; ++my)
			for (int mx = 0; mx < grid.mb_cols; ++mx)
			{
				const int step = weights[grid.macroblock(mx, my)].step;
				const double quantizer_value = quantizer(step);
				intra_macroblock choice;
				choose_intra(source, picture, grid, state, mx, my, step, rd_lambda * quantizer_value * quantizer_value,
				             false, models, choice);
				state.macroblocks[grid.macroblock(mx, my)] = mb_intra;
				encode_intra(out, models, grid, state, mx, my, choice, step, true, previous_step, steps);
			}
		loop_filter(picture, grid, state, steps);
	}

	bool decode_keyframe(range_decoder& in, wire_models& models, const block_grid& grid, image_u8& picture,
	                     std::vector<uint8_t>& steps)
	{
		picture = image_u8(grid.width, grid.height);
		steps.assign(static_cast<size_t>(grid.blocks()), 0);
		packet_state state;
		state.reset(grid);
		int previous_step = first_step;
		for (int my = 0; my < grid.mb_rows; ++my)
			for (int mx = 0; mx < grid.mb_cols; ++mx)
			{
				state.macroblocks[grid.macroblock(mx, my)] = mb_intra;
				if (!decode_intra(in, models, grid, state, picture, mx, my, true, previous_step, steps)) return false;
			}
		loop_filter(picture, grid, state, steps);
		return true;
	}

	// One macroblock record of a delta packet, as the encoder means to send it.
	struct mb_record
	{
		int index = 0; // the macroblock, raster order
		uint8_t kind = mb_skip;
		bool split = false; // a vector for each block
		int step = 0;
		point_i vectors[4] = {};
		uint8_t modes[4] = {};
		bool coded[4] = {};
		int16_t levels[4][block_area] = {};
		double bits = 0.0; // estimated, for the ranking and the budget
		double priority = 0.0;
		bool urgent = false;
	};

	// Every macroblock worth its bits as a candidate record: the best of skipping it along the
	// vector its neighbours predict, moving it by one vector or one for each block with
	// residuals where they pay, or coding it intra - whichever minimises distortion plus lambda
	// times bits - and what it is worth per bit. Each macroblock is judged as if every
	// candidate before it is sent, which is what most are.
	std::vector<mb_record> find_candidates(const image_u8& source, const half_pel_planes& planes,
	                                       const block_grid& grid, const std::vector<block_weight>& weights,
	                                       const wire_models& models, std::vector<uint16_t>& ages)
	{
		const image_u8& base = planes.planes[0];
		const vector_costs costs(models);
		const double step_bits = signed_cost(models.step, 0);
		const auto least = [](const bit_model* flags, const int count)
		{
			double cheapest = std::numeric_limits<double>::infinity();
			for (int i = 0; i < count; ++i)
				cheapest = std::min({cheapest, bit_cost(flags[i], 0), bit_cost(flags[i], 1)});
			return cheapest;
		};
		const double least_inter_flag = least(models.coded[1], 3);
		const double least_intra_block = least(models.coded[0], 3) + least(&models.probable_mode, 1);
		packet_state estimate;
		estimate.reset(grid);
		image_u8 work = base; // what the receiver holds if every candidate so far is sent
		std::vector<mb_record> candidates;

		mb_record trial;
		mb_record best;
		uint8_t samples[4][block_area] = {};
		uint8_t held[4][block_area] = {};
		uint8_t trial_pictures[4][block_area] = {};
		uint8_t best_pictures[4][block_area] = {};
		uint8_t predicted[block_area] = {};
		float coefficients[block_area] = {};

		for (int my = 0; my < grid.mb_rows; ++my)
			for (int mx = 0; mx < grid.mb_cols; ++mx)
			{
				const int macroblock = grid.macroblock(mx, my);
				const int step = weights[macroblock].step;
				const double quantizer_value = quantizer(step);
				const double lambda = rd_lambda * quantizer_value * quantizer_value;
				const double motion_lambda = std::sqrt(rd_lambda) * quantizer_value;

				partition whole;
				double held_error = 0.0;
				for (int which = 0; which < 4; ++which)
				{
					if (grid.part(mx, my, which) < 0) continue;
					const point_i origin{(2 * mx + (which & 1)) * block, (2 * my + (which >> 1)) * block};
					read_block(source, origin.x, origin.y, samples[which]);
					read_block(base, origin.x, origin.y, held[which]);
					held_error += squared_error(samples[which], held[which]);
					whole.origin[whole.count] = origin;
					whole.samples[whole.count] = samples[which];
					++whole.count;
				}
				if (held_error <= lambda * address_bits)
				{
					ages[macroblock] = 0;
					continue;
				}

				const int skip_context = neighbour_count(grid, estimate, mx, my, mb_skip);
				const int intra_context = neighbour_count(grid, estimate, mx, my, mb_intra);
				const double unskipped_bits = address_bits + bit_cost(models.skip[skip_context], 0);
				const double inter_bits = unskipped_bits + bit_cost(models.intra[intra_context], 0);
				const point_i predicted_whole = predict_vector(grid, estimate, 2 * mx, 2 * my, 2);

				double best_cost = held_error;
				double best_distortion = held_error;
				bool found = false;
				const auto start = [&](const uint8_t kind, const bool split)
				{
					estimate.clear(grid, mx, my);
					trial = mb_record{};
					trial.index = macroblock;
					trial.step = step;
					trial.kind = kind;
					trial.split = split;
				};
				const auto offer = [&](const double distortion, const double bits)
				{
					const double cost = distortion + lambda * bits;
					if (cost >= best_cost) return;
					best_cost = cost;
					best_distortion = distortion;
					trial.bits = bits;
					best = trial;
					std::copy_n(&trial_pictures[0][0], 4 * block_area, &best_pictures[0][0]);
					found = true;
				};
				// Moved along `vectors`, each block's residual sent only where it is worth its
				// bits. Returns the bits beyond the vectors.
				const auto code_inter = [&](const point_i* vectors, double& distortion)
				{
					double bits = 0.0;
					bool any = false;
					distortion = 0.0;
					for (int which = 0; which < 4; ++which)
					{
						const int index = grid.part(mx, my, which);
						if (index < 0) continue;
						const int bx = 2 * mx + (which & 1);
						const int by = 2 * my + (which >> 1);
						predict_motion(planes, bx * block, by * block, vectors[which], predicted);
						const double plain = squared_error(samples[which], predicted);
						transform_residual(samples[which], predicted, coefficients);
						residual corrected;
						quantise(coefficients, step, inter_rounding, lambda, models, 1, corrected);
						const bit_model& flag = models.coded[1][coded_context(grid, estimate, bx, by)];
						const bool send = corrected.coded &&
						                  corrected.distortion + lambda * (bit_cost(flag, 1) + corrected.bits) <
						                      plain + lambda * bit_cost(flag, 0);
						trial.vectors[which] = vectors[which];
						trial.coded[which] = send;
						estimate.coded[index] = send ? 1 : 0;
						if (send)
						{
							std::copy_n(corrected.levels, block_area, trial.levels[which]);
							reconstruct_block(predicted, corrected.levels, step, trial_pictures[which]);
							distortion += corrected.distortion;
							bits += bit_cost(flag, 1) + corrected.bits;
							any = true;
						}
						else
						{
							std::copy_n(predicted, block_area, trial_pictures[which]);
							distortion += plain;
							bits += bit_cost(flag, 0);
						}
					}
					return any ? bits + step_bits : bits;
				};

				// Skipped: moved along the predicted vector, and nothing more.
				{
					start(mb_skip, false);
					double distortion = 0.0;
					for (int which = 0; which < 4; ++which)
					{
						if (grid.part(mx, my, which) < 0) continue;
						predict_motion(planes, (2 * mx + (which & 1)) * block, (2 * my + (which >> 1)) * block,
						               predicted_whole, trial_pictures[which]);
						distortion += squared_error(samples[which], trial_pictures[which]);
						trial.vectors[which] = predicted_whole;
					}
					offer(distortion, address_bits + bit_cost(models.skip[skip_context], 1));
				}

				// One vector: the search's, the predicted one, and none. The search's first step is
				// half the vector range, so its steps together reach all of it.
				const int above = grid.at(2 * mx, 2 * my - 1);
				const point_i seeds[3] = {point_i{}, estimate.last_vector,
				                          above >= 0 && estimate.moving[above] != 0 ? estimate.vectors[above]
				                                                                    : point_i{}};
				const point_i searched = search_motion(planes, whole, predicted_whole, seeds, 3, max_motion / 2,
				                                       motion_lambda, costs);
				const point_i options[3] = {searched, predicted_whole, point_i{}};
				for (int option = 0; option < 3; ++option)
				{
					const point_i vector = options[option];
					if ((option > 0 && vector == options[0]) || (option > 1 && vector == options[1])) continue;
					start(mb_inter, false);
					const point_i vectors[4] = {vector, vector, vector, vector};
					double distortion = 0.0;
					const double bits = code_inter(vectors, distortion);
					offer(distortion,
					      inter_bits + bit_cost(models.split, 0) + costs(difference(vector, predicted_whole)) + bits);
				}

				// A vector for each block, each searched near the macroblock's against what the
				// blocks before it predict. Not tried where even its cheapest conceivable
				// signalling, at no distortion at all, loses to what is already found.
				const double split_floor = lambda * (inter_bits + bit_cost(models.split, 1) +
				                                     whole.count * (costs.floor() + least_inter_flag));
				if (whole.count > 1 && split_floor < best_cost)
				{
					start(mb_inter, true);
					const point_i last_vector = estimate.last_vector;
					point_i vectors[4] = {};
					double vector_bits = 0.0;
					for (int which = 0; which < 4; ++which)
					{
						const int index = grid.part(mx, my, which);
						if (index < 0) continue;
						const int bx = 2 * mx + (which & 1);
						const int by = 2 * my + (which >> 1);
						const point_i predictor = predict_vector(grid, estimate, bx, by, 1);
						partition single;
						single.count = 1;
						single.origin[0] = {bx * block, by * block};
						single.samples[0] = samples[which];
						const point_i part_seeds[3] = {searched, predicted_whole, point_i{}};
						vectors[which] = search_motion(planes, single, predictor, part_seeds, 3, split_reach,
						                               motion_lambda, costs);
						vector_bits += costs(difference(vectors[which], predictor));
						estimate.moving[index] = 1;
						estimate.vectors[index] = vectors[which];
						estimate.last_vector = vectors[which];
					}
					estimate.last_vector = last_vector;
					double distortion = 0.0;
					const double bits = code_inter(vectors, distortion);
					offer(distortion, inter_bits + bit_cost(models.split, 1) + vector_bits + bits);
				}

				// Intra, predicted from the picture as the candidates before it leave it, and
				// likewise not tried where its cheapest signalling already loses.
				const double intra_floor = lambda * (unskipped_bits + bit_cost(models.intra[intra_context], 1) +
				                                     whole.count * least_intra_block);
				if (intra_floor < best_cost)
				{
					start(mb_intra, false);
					intra_macroblock choice;
					choose_intra(source, work, grid, estimate, mx, my, step, lambda, true, models, choice);
					bool any = false;
					for (int which = 0; which < 4; ++which)
					{
						trial.modes[which] = choice.modes[which];
						trial.coded[which] = choice.coded[which];
						any = any || choice.coded[which];
					}
					std::copy_n(&choice.pictures[0][0], 4 * block_area, &trial_pictures[0][0]);
					offer(choice.distortion, unskipped_bits + bit_cost(models.intra[intra_context], 1) + choice.bits +
					                             (any ? step_bits : 0.0));
				}

				estimate.clear(grid, mx, my);
				for (int which = 0; which < 4; ++which)
					if (grid.part(mx, my, which) >= 0)
						write_block(work, (2 * mx + (which & 1)) * block, (2 * my + (which >> 1)) * block,
						            found ? best_pictures[which] : held[which]);
				if (!found)
				{
					ages[macroblock] = 0;
					continue;
				}
				estimate.macroblocks[macroblock] = best.kind;
				for (int which = 0; which < 4; ++which)
				{
					const int index = grid.part(mx, my, which);
					if (index < 0) continue;
					estimate.coded[index] = best.coded[which] ? 1 : 0;
					if (best.kind == mb_intra) estimate.modes[index] = best.modes[which];
					else
					{
						estimate.moving[index] = 1;
						estimate.vectors[index] = best.vectors[which];
						estimate.last_vector = best.vectors[which];
					}
				}

				// Distortion actually removed per bit, tilted towards faces. A macroblock the face
				// has moved off carries a large error at a low importance, so only a bounded gain -
				// and the deadline - ever let it outbid the face itself.
				best.urgent = ages[macroblock] >= starvation_deadline;
				best.priority = importance_gain(weights[macroblock].importance) * (held_error - best_distortion) /
				                best.bits;
				candidates.push_back(best);
			}
		return candidates;
	}

	// Codes a delta packet's records into `out` and `picture`, which holds the base on entry,
	// filling `state` and `steps` as the decoder will. `chosen` holds a record or nothing for
	// each macroblock. Inter records are sent as the candidate pass chose them, as a skip
	// wherever the vector turns out to be the predicted one; an intra record is chosen afresh
	// from the picture as this pass has reconstructed it, which is what its receiver predicts
	// it from.
	void encode_records(range_encoder& out, wire_models& models, const block_grid& grid, const half_pel_planes& planes,
	                    const image_u8& source, const std::vector<const mb_record*>& chosen,
	                    const std::vector<uint8_t>& recent, image_u8& picture, std::vector<uint8_t>& steps,
	                    packet_state& state)
	{
		state.reset(grid);
		const bool any_record = std::any_of(chosen.begin(), chosen.end(),
		                                    [](const mb_record* entry) { return entry != nullptr; });
		out.encode(models.records, any_record ? 1 : 0);
		if (!any_record) return;

		int previous_step = first_step;
		uint8_t predicted[block_area] = {};
		uint8_t reconstructed[block_area] = {};
		for (int my = 0; my < grid.mb_rows; ++my)
			for (int mx = 0; mx < grid.mb_cols; ++mx)
			{
				const int macroblock = grid.macroblock(mx, my);
				const mb_record* entry = chosen[macroblock];
				out.encode(models.updated[updated_context(grid, state, recent, mx, my)], entry != nullptr ? 1 : 0);
				if (entry == nullptr) continue;
				const int skip_context = neighbour_count(grid, state, mx, my, mb_skip);
				const int intra_context = neighbour_count(grid, state, mx, my, mb_intra);

				if (entry->kind == mb_intra)
				{
					out.encode(models.skip[skip_context], 0);
					out.encode(models.intra[intra_context], 1);
					state.macroblocks[macroblock] = mb_intra;
					const double quantizer_value = quantizer(entry->step);
					intra_macroblock choice;
					choose_intra(source, picture, grid, state, mx, my, entry->step,
					             rd_lambda * quantizer_value * quantizer_value, true, models, choice);
					encode_intra(out, models, grid, state, mx, my, choice, entry->step, false, previous_step, steps);
					continue;
				}

				bool any = false;
				bool split = false;
				for (int which = 0; which < 4; ++which)
				{
					if (grid.part(mx, my, which) < 0) continue;
					any = any || entry->coded[which];
					split = split || (entry->split && entry->vectors[which] != entry->vectors[0]);
				}
				const point_i predicted_whole = predict_vector(grid, state, 2 * mx, 2 * my, 2);
				const bool skip = !split && !any && entry->vectors[0] == predicted_whole;
				out.encode(models.skip[skip_context], skip ? 1 : 0);
				if (!skip)
				{
					out.encode(models.intra[intra_context], 0);
					out.encode(models.split, split ? 1 : 0);
				}
				state.macroblocks[macroblock] = skip ? mb_skip : mb_inter;
				for (int which = 0; which < 4; ++which)
				{
					const int index = grid.part(mx, my, which);
					if (index < 0) continue;
					if (!skip && (split || which == 0))
					{
						const point_i predictor = split ? predict_vector(grid, state, 2 * mx + (which & 1),
						                                                 2 * my + (which >> 1), 1)
						                                : predicted_whole;
						encode_signed(out, models.vector[0], entry->vectors[which].x - predictor.x);
						encode_signed(out, models.vector[1], entry->vectors[which].y - predictor.y);
					}
					state.moving[index] = 1;
					state.vectors[index] = entry->vectors[which];
					state.last_vector = entry->vectors[which];
				}
				if (!skip)
				{
					for (int which = 0; which < 4; ++which)
					{
						const int index = grid.part(mx, my, which);
						if (index < 0) continue;
						out.encode(
							models.coded[1][coded_context(grid, state, 2 * mx + (which & 1), 2 * my + (which >> 1))],
							entry->coded[which] ? 1 : 0);
						state.coded[index] = entry->coded[which] ? 1 : 0;
					}
					if (any)
					{
						encode_signed(out, models.step, entry->step - previous_step);
						previous_step = entry->step;
					}
					for (int which = 0; which < 4; ++which)
						if (grid.part(mx, my, which) >= 0 && entry->coded[which])
							encode_residual(out, models, 1, entry->levels[which]);
				}
				for (int which = 0; which < 4; ++which)
				{
					const int index = grid.part(mx, my, which);
					if (index < 0) continue;
					const int x = (2 * mx + (which & 1)) * block;
					const int y = (2 * my + (which >> 1)) * block;
					predict_motion(planes, x, y, entry->vectors[which], predicted);
					reconstruct_block(predicted, entry->levels[which], entry->step, reconstructed);
					write_block(picture, x, y, reconstructed);
					if (any) steps[index] = static_cast<uint8_t>(entry->step);
				}
			}
	}

	// The records of a delta packet that says it has some, into `picture`, which holds the
	// base on entry. `updated` marks the macroblocks they touch.
	bool decode_records(range_decoder& in, wire_models& models, const block_grid& grid, const image_u8& base,
	                    const std::vector<uint8_t>& recent, image_u8& picture, std::vector<uint8_t>& steps,
	                    std::vector<uint8_t>& updated)
	{
		const half_pel_planes planes(base);
		packet_state state;
		state.reset(grid);
		int previous_step = first_step;
		bool any_record = false;
		uint8_t predicted[block_area] = {};
		uint8_t reconstructed[block_area] = {};
		for (int my = 0; my < grid.mb_rows; ++my)
			for (int mx = 0; mx < grid.mb_cols; ++mx)
			{
				const int macroblock = grid.macroblock(mx, my);
				if (in.decode(models.updated[updated_context(grid, state, recent, mx, my)]) == 0)
				{
					if (!in.ok()) return false;
					continue;
				}
				any_record = true;
				updated[macroblock] = 1;
				const int skip_context = neighbour_count(grid, state, mx, my, mb_skip);
				const int intra_context = neighbour_count(grid, state, mx, my, mb_intra);
				const bool skip = in.decode(models.skip[skip_context]) != 0;
				if (!skip && in.decode(models.intra[intra_context]) != 0)
				{
					state.macroblocks[macroblock] = mb_intra;
					if (!decode_intra(in, models, grid, state, picture, mx, my, false, previous_step, steps))
						return false;
					continue;
				}
				const bool split = !skip && in.decode(models.split) != 0;
				state.macroblocks[macroblock] = skip ? mb_skip : mb_inter;

				const point_i predicted_whole = predict_vector(grid, state, 2 * mx, 2 * my, 2);
				point_i vectors[4] = {};
				for (int which = 0; which < 4; ++which)
				{
					const int index = grid.part(mx, my, which);
					if (index < 0) continue;
					vectors[which] = which == 0 || split ? predicted_whole : vectors[0];
					if (!skip && (split || which == 0))
					{
						const point_i predictor = split ? predict_vector(grid, state, 2 * mx + (which & 1),
						                                                 2 * my + (which >> 1), 1)
						                                : predicted_whole;
						const int dx = decode_signed(in, models.vector[0]);
						const int dy = decode_signed(in, models.vector[1]);
						vectors[which] = {predictor.x + dx, predictor.y + dy};
						if (!in.ok() || std::abs(vectors[which].x) > max_motion ||
						    std::abs(vectors[which].y) > max_motion)
							return false;
					}
					state.moving[index] = 1;
					state.vectors[index] = vectors[which];
					state.last_vector = vectors[which];
				}

				bool coded[4] = {};
				bool any = false;
				if (!skip)
					for (int which = 0; which < 4; ++which)
					{
						const int index = grid.part(mx, my, which);
						if (index < 0) continue;
						coded[which] = in.decode(models.coded[1][coded_context(grid, state, 2 * mx + (which & 1),
						                                                       2 * my + (which >> 1))]) != 0;
						state.coded[index] = coded[which] ? 1 : 0;
						any = any || coded[which];
					}
				int step = previous_step;
				if (any)
				{
					step = previous_step + decode_signed(in, models.step);
					if (!in.ok() || step < 0 || step >= ladder_steps) return false;
					previous_step = step;
				}
				int16_t levels[4][block_area] = {};
				for (int which = 0; which < 4; ++which)
					if (coded[which] && !decode_residual(in, models, 1, levels[which])) return false;
				for (int which = 0; which < 4; ++which)
				{
					const int index = grid.part(mx, my, which);
					if (index < 0) continue;
					const int x = (2 * mx + (which & 1)) * block;
					const int y = (2 * my + (which >> 1)) * block;
					predict_motion(planes, x, y, vectors[which], predicted);
					reconstruct_block(predicted, levels[which], step, reconstructed);
					write_block(picture, x, y, reconstructed);
					if (any) steps[index] = static_cast<uint8_t>(step);
				}
				if (!in.ok()) return false;
			}
		// A packet that says it has records has at least one.
		if (!any_record) return false;
		loop_filter(picture, grid, state, steps);
		return true;
	}

	void write_u16(std::vector<uint8_t>& bytes, const int value)
	{
		bytes.push_back(static_cast<uint8_t>(value & 0xff));
		bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
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
	const bool weighted = importance != nullptr && importance->width() == source.width() &&
	                      importance->height() == source.height();

	for (size_t i = 0; i < source.size(); ++i)
	{
		const double error = static_cast<double>(source.data()[i]) - reconstructed.data()[i];
		const double weight = weighted ? std::max(1.0, static_cast<double>(importance->data()[i])) : 1.0;
		squared_error += error * error;
		weighted_error += weight * error * error;
		total_weight += weight;
	}

	quality.rmse = std::sqrt(squared_error / static_cast<double>(source.size()));
	quality.psnr = psnr_from_rmse(quality.rmse);
	quality.weighted_rmse = std::sqrt(weighted_error / total_weight);
	quality.weighted_psnr = psnr_from_rmse(quality.weighted_rmse);
	return quality;
}

void intra_picture(const image_u8& source, const image_u8* importance, const int quality, image_u8& reconstructed)
{
	if (source.empty())
	{
		reconstructed = {};
		return;
	}
	const block_grid grid(source.width(), source.height());
	range_encoder unused;
	wire_models models;
	std::vector<uint8_t> steps;
	encode_keyframe(unused, models, source, grid,
	                weigh_macroblocks(importance, grid, std::clamp(quality, 0, ladder_steps - 1)), reconstructed,
	                steps);
}

void deblock_intra(image_u8& picture, const std::vector<uint8_t>& steps)
{
	const block_grid grid(picture.width(), picture.height());
	if (picture.empty() || steps.size() != static_cast<size_t>(grid.blocks())) return;
	packet_state state;
	state.reset(grid);
	std::fill(state.macroblocks.begin(), state.macroblocks.end(), mb_intra);
	loop_filter(picture, grid, state, steps);
}

void lowban_wire::reset()
{
	encoder = {};
	decoder = {};
	encoder_age.clear();
}

lowban_packet lowban_wire::encode(const image_u8& source, const image_u8* importance, const face_geometry& faces_in,
                                  const bool force_keyframe, const size_t max_packet_bytes, const int quality_in)
{
	lowban_packet packet;
	const int width = source.width();
	const int height = source.height();
	if (width <= 0 || height <= 0 || width > max_dimension || height > max_dimension) return packet;
	const block_grid grid(width, height);
	const auto macroblock_count = static_cast<size_t>(grid.macroblocks());
	const bool keyframe = force_keyframe || !encoder.models || encoder.picture.width() != width ||
	                      encoder.picture.height() != height;
	const face_geometry faces = bounded_faces(faces_in);
	const std::vector<block_weight> weights = weigh_macroblocks(importance, grid,
	                                                            std::clamp(quality_in, 0, ladder_steps - 1));
	if (keyframe || encoder_age.size() != macroblock_count) encoder_age.assign(macroblock_count, 0);

	// Nothing below touches the encoder's own state until the packet is known to be sent.
	auto models = std::make_shared<wire_models>(keyframe ? wire_models{} : *encoder.models);
	side next;
	next.geometry = faces;
	range_encoder out;
	std::vector<uint8_t> header;

	if (keyframe)
	{
		header.push_back(static_cast<uint8_t>(wire_version << 4 | flag_keyframe));
		write_u16(header, width);
		write_u16(header, height);
		encode_geometry(out, *models, {}, faces, true);
		packet.geometry_bytes = out.size_if_finished() - 4;
		encode_keyframe(out, *models, source, grid, weights, next.picture, next.steps);
		next.recent.assign(macroblock_count, 0);
		packet.keyframe = true;
		packet.geometry = true;
		packet.updated_blocks = grid.blocks();
	}
	else
	{
		// New landmarks are always sent. The warp onto them is committed only when it brings
		// the blocks it moves closer to the source than leaving them alone would: a hand over
		// the face or a lost track moves the mesh without moving the face.
		const bool geometry = faces != encoder.geometry;
		image_u8 warped;
		bool warp = false;
		if (geometry && same_layout(encoder.geometry, faces) &&
		    warp_faces(encoder.picture, encoder.geometry, faces, warped))
		{
			double still_error = 0.0;
			double warped_error = 0.0;
			uint8_t samples[block_area];
			uint8_t held[block_area];
			uint8_t moved[block_area];
			for (int index = 0; index < grid.blocks(); ++index)
			{
				const int block_x = index % grid.cols * block;
				const int block_y = index / grid.cols * block;
				read_block(encoder.picture, block_x, block_y, held);
				read_block(warped, block_x, block_y, moved);
				if (std::equal(std::begin(held), std::end(held), std::begin(moved))) continue;
				read_block(source, block_x, block_y, samples);
				still_error += squared_error(samples, held);
				warped_error += squared_error(samples, moved);
			}
			warp = warped_error < still_error;
		}
		const image_u8& base = warp ? warped : encoder.picture;
		const half_pel_planes planes(base);

		header.push_back(
			static_cast<uint8_t>(wire_version << 4 | (geometry ? flag_geometry : 0) | (warp ? flag_warp : 0)));
		if (geometry)
		{
			encode_geometry(out, *models, encoder.geometry, faces, false);
			packet.geometry_bytes = out.size_if_finished() - 4;
		}

		std::vector<mb_record> candidates = find_candidates(source, planes, grid, weights, *models, encoder_age);
		std::sort(candidates.begin(), candidates.end(), [](const mb_record& a, const mb_record& b)
		{
			if (a.urgent != b.urgent) return a.urgent;
			return a.priority != b.priority ? a.priority > b.priority : a.index < b.index;
		});

		// Choose on estimated bits, then code in macroblock order and, while the real packet is
		// over its cap, drop the least deserving choices and code again. The range coder knows
		// its exact length at every point, so what is sent never exceeds the cap.
		const size_t fixed_bytes = header.size() + out.size_if_finished();
		const double available_bits = max_packet_bytes == 0 ? std::numeric_limits<double>::infinity()
		                              : max_packet_bytes > fixed_bytes
		                                  ? static_cast<double>(max_packet_bytes - fixed_bytes) * 8.0 - 8.0
		                                  : 0.0;
		std::vector<bool> selected(candidates.size(), false);
		double used_bits = 0.0;
		for (size_t i = 0; i < candidates.size(); ++i)
		{
			if (used_bits + candidates[i].bits > available_bits) continue;
			used_bits += candidates[i].bits;
			selected[i] = true;
		}

		const range_encoder after_geometry = out;
		const wire_models models_after_geometry = *models;
		std::vector<const mb_record*> chosen(macroblock_count, nullptr);
		packet_state state;
		bool any_record = false;
		for (;;)
		{
			std::fill(chosen.begin(), chosen.end(), nullptr);
			any_record = false;
			for (size_t i = 0; i < candidates.size(); ++i)
				if (selected[i])
				{
					chosen[static_cast<size_t>(candidates[i].index)] = &candidates[i];
					any_record = true;
				}
			out = after_geometry;
			*models = models_after_geometry;
			next.picture = base;
			next.steps = encoder.steps;
			encode_records(out, *models, grid, planes, source, chosen, encoder.recent, next.picture, next.steps, state);
			const size_t size = header.size() + out.size_if_finished();
			if (max_packet_bytes == 0 || size <= max_packet_bytes || !any_record) break;

			double overflow_bits = static_cast<double>(size - max_packet_bytes) * 8.0;
			for (size_t i = candidates.size(); i-- > 0 && overflow_bits > 0.0;)
			{
				if (!selected[i]) continue;
				selected[i] = false;
				overflow_bits -= candidates[i].bits;
			}
		}
		for (size_t i = 0; i < candidates.size(); ++i)
		{
			auto& age = encoder_age[static_cast<size_t>(candidates[i].index)];
			if (selected[i]) age = 0;
			else if (age < 0xffff) ++age;
		}
		packet.limited = std::find(selected.begin(), selected.end(), false) != selected.end();

		// A packet with neither landmarks nor records is never sent, and commits nothing.
		if (!geometry && !any_record) return packet;

		next.recent.assign(macroblock_count, 0);
		for (int my = 0; my < grid.mb_rows; ++my)
			for (int mx = 0; mx < grid.mb_cols; ++mx)
			{
				const int macroblock = grid.macroblock(mx, my);
				if (state.macroblocks[macroblock] == mb_untouched) continue;
				next.recent[static_cast<size_t>(macroblock)] = 1;
				for (int which = 0; which < 4; ++which)
					if (grid.part(mx, my, which) >= 0) ++packet.updated_blocks;
			}
		if (any_record) loop_filter(next.picture, grid, state, next.steps);
		packet.geometry = geometry;
		packet.warped = warp;
	}

	packet.bytes = std::move(header);
	const std::vector<uint8_t> payload = out.finish();
	packet.bytes.insert(packet.bytes.end(), payload.begin(), payload.end());
	next.models = std::move(models);
	encoder = std::move(next);
	return packet;
}

bool lowban_wire::decode(const lowban_packet& packet, image_u8& picture)
{
	const std::vector<uint8_t>& bytes = packet.bytes;
	if (bytes.empty() || bytes[0] >> 4 != wire_version) return false;
	const int flags = bytes[0] & 0x0f;
	const bool keyframe = (flags & flag_keyframe) != 0;
	const bool geometry = (flags & flag_geometry) != 0;
	const bool warp = (flags & flag_warp) != 0;
	if ((flags & ~(flag_keyframe | flag_geometry | flag_warp)) != 0 || (keyframe && (geometry || warp)) ||
	    (warp && !geometry))
		return false;

	size_t offset = 1;
	int width = decoder.picture.width();
	int height = decoder.picture.height();
	if (keyframe)
	{
		if (bytes.size() < 5) return false;
		width = bytes[1] | bytes[2] << 8;
		height = bytes[3] | bytes[4] << 8;
		if (width <= 0 || height <= 0 || width > max_dimension || height > max_dimension) return false;
		offset = 5;
	}
	else if (!decoder.models || decoder.picture.empty()) return false;
	const block_grid grid(width, height);
	const auto macroblock_count = static_cast<size_t>(grid.macroblocks());

	auto models = std::make_shared<wire_models>(keyframe ? wire_models{} : *decoder.models);
	range_decoder in(bytes.data() + offset, bytes.size() - offset);
	side next;
	next.geometry = decoder.geometry;
	if ((keyframe || geometry) &&
	    !decode_geometry(in, *models, keyframe ? face_geometry{} : decoder.geometry, keyframe, next.geometry))
		return false;

	next.recent.assign(macroblock_count, 0);
	if (keyframe)
	{
		if (!decode_keyframe(in, *models, grid, next.picture, next.steps)) return false;
	}
	else
	{
		image_u8 warped;
		if (warp && (!same_layout(decoder.geometry, next.geometry) ||
		             !warp_faces(decoder.picture, decoder.geometry, next.geometry, warped)))
			return false;
		const image_u8& base = warp ? warped : decoder.picture;
		next.picture = base;
		next.steps = decoder.steps;
		if (in.decode(models->records) != 0 &&
		    !decode_records(in, *models, grid, base, decoder.recent, next.picture, next.steps, next.recent))
			return false;
	}
	if (!in.finished()) return false;

	next.models = std::move(models);
	decoder = std::move(next);
	picture = decoder.picture;
	return true;
}

bool lowban_wire::in_sync() const
{
	return encoder.picture.width() == decoder.picture.width() && encoder.picture.height() == decoder.picture.height() &&
	       std::equal(encoder.picture.data(), encoder.picture.data() + encoder.picture.size(),
	                  decoder.picture.data()) &&
	       encoder.steps == decoder.steps && encoder.recent == decoder.recent && encoder.geometry == decoder.geometry;
}

bool lowban_wire::show(image_u8& shown) const
{
	if (decoder.picture.empty()) return false;
	shown = decoder.picture;
	return true;
}
