#include "landmarks.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace
{
	// Reader for the model file's binary encoding.
	//
	// Integers are stored as a control byte followed by little-endian magnitude bytes: the
	// low four bits of the control byte hold the byte count and 0x80 marks a negative value.
	// Floating point values are stored as a (mantissa, exponent) integer pair so that the
	// file is independent of the writer's floating point format.
	class model_reader
	{
		const uint8_t* cursor;
		const uint8_t* end;

	public:
		model_reader(const uint8_t* data, const size_t len) : cursor(data), end(data + len)
		{
		}

		int64_t read_int()
		{
			if (cursor >= end) throw std::runtime_error("unexpected end of model file");
			const uint8_t control = *cursor++;
			const bool negative = (control & 0x80) != 0;
			const uint8_t count = control & 0x0F;
			if (count == 0 || count > 8) throw std::runtime_error("corrupt integer in model file");
			if (end - cursor < count) throw std::runtime_error("unexpected end of model file");

			uint64_t magnitude = 0;
			for (uint8_t i = 0; i < count; ++i)
			{
				magnitude |= static_cast<uint64_t>(*cursor++) << (8 * i);
			}

			return negative ? -static_cast<int64_t>(magnitude) : static_cast<int64_t>(magnitude);
		}

		uint32_t read_count()
		{
			const int64_t v = read_int();
			if (v < 0 || v > 0x7FFFFFFF) throw std::runtime_error("corrupt count in model file");
			return static_cast<uint32_t>(v);
		}

		float read_float()
		{
			const int64_t mantissa = read_int();
			const int64_t exponent = read_int();

			switch (exponent)
			{
			case 32000: return std::numeric_limits<float>::infinity();
			case 32001: return -std::numeric_limits<float>::infinity();
			case 32002: return std::numeric_limits<float>::quiet_NaN();
			default: break;
			}

			return static_cast<float>(std::ldexp(static_cast<double>(mantissa), static_cast<int>(exponent)));
		}

		// A vector-shaped matrix is stored as (-rows, -columns) followed by its elements.
		void read_column_vector(std::vector<float>& out)
		{
			const int64_t nr = read_int();
			const int64_t nc = read_int();
			const int64_t rows = nr < 0 ? -nr : nr;
			const int64_t cols = nc < 0 ? -nc : nc;
			const int64_t total = rows * cols;
			if (total < 0 || total > (1 << 20)) throw std::runtime_error("implausible matrix size in model file");

			out.resize(static_cast<size_t>(total));
			for (auto& v : out) v = read_float();
		}

		// Appends a vector-shaped matrix onto an existing flat buffer.
		void append_column_vector(std::vector<float>& out, const size_t expected)
		{
			const int64_t nr = read_int();
			const int64_t nc = read_int();
			const size_t total = static_cast<size_t>(std::llabs(nr)) * static_cast<size_t>(std::llabs(nc));
			if (total != expected) throw std::runtime_error("unexpected leaf size in model file");

			for (size_t i = 0; i < total; ++i) out.push_back(read_float());
		}
	};

	bool read_whole_file(const std::wstring& path, std::vector<uint8_t>& out, std::string& error)
	{
		FILE* file = nullptr;
		if (_wfopen_s(&file, path.c_str(), L"rb") != 0 || file == nullptr)
		{
			error = "cannot open model file";
			return false;
		}

		fseek(file, 0, SEEK_END);
		const long long len = _ftelli64(file);
		fseek(file, 0, SEEK_SET);

		if (len <= 0)
		{
			fclose(file);
			error = "model file is empty";
			return false;
		}

		out.resize(static_cast<size_t>(len));
		const size_t got = fread(out.data(), 1, out.size(), file);
		fclose(file);

		if (got != out.size())
		{
			error = "model file could not be read";
			return false;
		}

		return true;
	}

	// The 2x2 linear part of the least squares similarity transform mapping `from` onto `to`.
	// Translation is not needed because the sampler works in offsets from an anchor point.
	struct linear2x2
	{
		float a = 1.0f, b = 0.0f, c = 0.0f, d = 1.0f;

		point_f apply(const point_f p) const { return {a * p.x + b * p.y, c * p.x + d * p.y}; }
	};

	linear2x2 similarity_between(const std::vector<float>& from, const std::vector<float>& to)
	{
		const size_t n = from.size() / 2;
		if (n < 2) return {};

		double mean_from_x = 0, mean_from_y = 0, mean_to_x = 0, mean_to_y = 0;
		for (size_t i = 0; i < n; ++i)
		{
			mean_from_x += from[i * 2];
			mean_from_y += from[i * 2 + 1];
			mean_to_x += to[i * 2];
			mean_to_y += to[i * 2 + 1];
		}
		mean_from_x /= n;
		mean_from_y /= n;
		mean_to_x /= n;
		mean_to_y /= n;

		// Optimal scaled rotation [[k,-s],[s,k]] in closed form.
		double dot = 0, cross = 0, norm = 0;
		for (size_t i = 0; i < n; ++i)
		{
			const double px = from[i * 2] - mean_from_x;
			const double py = from[i * 2 + 1] - mean_from_y;
			const double qx = to[i * 2] - mean_to_x;
			const double qy = to[i * 2 + 1] - mean_to_y;

			dot += px * qx + py * qy;
			cross += px * qy - py * qx;
			norm += px * px + py * py;
		}

		if (norm <= 0) return {};

		const auto k = static_cast<float>(dot / norm);
		const auto s = static_cast<float>(cross / norm);
		return {k, -s, s, k};
	}
}

void shape_predictor::clear()
{
	initial_shape.clear();
	forests.clear();
	anchor_idx.clear();
	deltas.clear();
}

bool shape_predictor::load(const std::wstring& path, std::string& error)
{
	clear();

	std::vector<uint8_t> bytes;
	if (!read_whole_file(path, bytes, error)) return false;

	try
	{
		model_reader reader(bytes.data(), bytes.size());

		if (reader.read_int() != 1)
		{
			error = "unsupported model file version";
			return false;
		}

		reader.read_column_vector(initial_shape);
		if (initial_shape.empty() || initial_shape.size() % 2 != 0)
		{
			error = "model file has no usable initial shape";
			return false;
		}
		const size_t shape_size = initial_shape.size();

		const uint32_t cascade_count = reader.read_count();
		forests.resize(cascade_count);
		for (auto& forest : forests)
		{
			const uint32_t tree_count = reader.read_count();
			forest.resize(tree_count);

			for (auto& t : forest)
			{
				const uint32_t split_count = reader.read_count();
				t.splits.resize(split_count);
				for (auto& s : t.splits)
				{
					s.idx1 = reader.read_count();
					s.idx2 = reader.read_count();
					s.thresh = reader.read_float();
				}

				const uint32_t leaf_count = reader.read_count();
				if (leaf_count != split_count + 1)
				{
					error = "model file has a malformed regression tree";
					return false;
				}

				t.leaves.reserve(static_cast<size_t>(leaf_count) * shape_size);
				for (uint32_t i = 0; i < leaf_count; ++i) reader.append_column_vector(t.leaves, shape_size);
			}
		}

		const uint32_t anchor_groups = reader.read_count();
		anchor_idx.resize(anchor_groups);
		for (auto& group : anchor_idx)
		{
			group.resize(reader.read_count());
			for (auto& v : group) v = reader.read_count();
		}

		const uint32_t delta_groups = reader.read_count();
		deltas.resize(delta_groups);
		for (auto& group : deltas)
		{
			group.resize(reader.read_count());
			for (auto& v : group)
			{
				v.x = reader.read_float();
				v.y = reader.read_float();
			}
		}

		if (anchor_idx.size() != forests.size() || deltas.size() != forests.size())
		{
			error = "model file cascade counts disagree";
			return false;
		}

		for (size_t i = 0; i < forests.size(); ++i)
		{
			if (anchor_idx[i].size() != deltas[i].size())
			{
				error = "model file feature tables disagree";
				return false;
			}
			for (const auto a : anchor_idx[i])
			{
				if (a >= shape_size / 2)
				{
					error = "model file anchor index out of range";
					return false;
				}
			}
			for (const auto& t : forests[i])
			{
				for (const auto& s : t.splits)
				{
					if (s.idx1 >= anchor_idx[i].size() || s.idx2 >= anchor_idx[i].size())
					{
						error = "model file split index out of range";
						return false;
					}
				}
			}
		}
	}
	catch (const std::exception& e)
	{
		clear();
		error = e.what();
		return false;
	}

	return true;
}

point_f shape_predictor::mean_part(const size_t index) const
{
	if (index >= num_parts()) return {};
	return {initial_shape[index * 2], initial_shape[index * 2 + 1]};
}

std::vector<point_i> shape_predictor::predict(const image_u8& img, const rect_i& rect) const
{
	std::vector<point_i> parts;
	if (!loaded() || rect.empty() || img.empty()) return parts;

	const size_t shape_size = initial_shape.size();
	std::vector<float> shape = initial_shape;
	std::vector<uint8_t> features;

	// Maps the unit box onto the detection rectangle.
	const float origin_x = static_cast<float>(rect.left);
	const float origin_y = static_cast<float>(rect.top);
	const float span_x = static_cast<float>(rect.right - rect.left);
	const float span_y = static_cast<float>(rect.bottom - rect.top);

	for (size_t cascade = 0; cascade < forests.size(); ++cascade)
	{
		const auto& anchors = anchor_idx[cascade];
		const auto& offsets = deltas[cascade];
		const linear2x2 tform = similarity_between(initial_shape, shape);

		features.resize(anchors.size());
		for (size_t i = 0; i < anchors.size(); ++i)
		{
			const point_f offset = tform.apply(offsets[i]);
			const float nx = shape[anchors[i] * 2] + offset.x;
			const float ny = shape[anchors[i] * 2 + 1] + offset.y;

			const int px = static_cast<int>(std::floor(origin_x + nx * span_x + 0.5f));
			const int py = static_cast<int>(std::floor(origin_y + ny * span_y + 0.5f));
			features[i] = img.sample_or_zero(px, py);
		}

		for (const auto& t : forests[cascade])
		{
			size_t node = 0;
			while (node < t.splits.size())
			{
				const auto& s = t.splits[node];
				const int diff = static_cast<int>(features[s.idx1]) - static_cast<int>(features[s.idx2]);
				node = static_cast<float>(diff) > s.thresh ? 2 * node + 1 : 2 * node + 2;
			}

			const float* leaf = t.leaves.data() + (node - t.splits.size()) * shape_size;
			for (size_t i = 0; i < shape_size; ++i) shape[i] += leaf[i];
		}
	}

	parts.resize(shape_size / 2);
	for (size_t i = 0; i < parts.size(); ++i)
	{
		parts[i].x = static_cast<int>(std::floor(origin_x + shape[i * 2] * span_x + 0.5f));
		parts[i].y = static_cast<int>(std::floor(origin_y + shape[i * 2 + 1] * span_y + 0.5f));
	}

	return parts;
}

rect_i shape_predictor::box_from_parts(const std::vector<point_i>& parts) const
{
	if (!loaded() || parts.size() != num_parts()) return {};

	// Least squares fit of image = origin + span * model, independently per axis.
	const auto fit = [&](const int axis, float& origin, float& span)
	{
		double mean_model = 0, mean_image = 0;
		for (size_t i = 0; i < parts.size(); ++i)
		{
			mean_model += initial_shape[i * 2 + axis];
			mean_image += axis == 0 ? parts[i].x : parts[i].y;
		}
		mean_model /= parts.size();
		mean_image /= parts.size();

		double covariance = 0, variance = 0;
		for (size_t i = 0; i < parts.size(); ++i)
		{
			const double m = initial_shape[i * 2 + axis] - mean_model;
			const double v = (axis == 0 ? parts[i].x : parts[i].y) - mean_image;
			covariance += m * v;
			variance += m * m;
		}

		if (variance <= 0)
		{
			origin = 0.0f;
			span = 0.0f;
			return false;
		}

		span = static_cast<float>(covariance / variance);
		origin = static_cast<float>(mean_image - span * mean_model);
		return true;
	};

	float origin_x = 0, span_x = 0, origin_y = 0, span_y = 0;
	if (!fit(0, origin_x, span_x) || !fit(1, origin_y, span_y)) return {};
	if (span_x <= 1.0f || span_y <= 1.0f) return {};

	rect_i box;
	box.left = static_cast<int>(std::lround(origin_x));
	box.top = static_cast<int>(std::lround(origin_y));
	box.right = static_cast<int>(std::lround(origin_x + span_x));
	box.bottom = static_cast<int>(std::lround(origin_y + span_y));
	return box;
}

namespace
{
	void wire_append_u16(std::vector<uint8_t>& bytes, const size_t value)
	{
		bytes.push_back(static_cast<uint8_t>(value & 0xff));
		bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
	}

	bool wire_read_u16(const std::vector<uint8_t>& bytes, size_t& offset, size_t& value)
	{
		if (offset + 2 > bytes.size()) return false;
		value = bytes[offset] | static_cast<size_t>(bytes[offset + 1]) << 8;
		offset += 2;
		return true;
	}

	uint32_t wire_zigzag_encode(const int value)
	{
		return static_cast<uint32_t>(value >= 0 ? value * 2 : -value * 2 - 1);
	}

	int wire_zigzag_decode(const uint32_t value)
	{
		return (value & 1) == 0 ? static_cast<int>(value / 2) : -static_cast<int>((value + 1) / 2);
	}

	void wire_append_varuint(std::vector<uint8_t>& bytes, uint32_t value)
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

	bool wire_read_varuint(const std::vector<uint8_t>& bytes, size_t& offset, uint32_t& value)
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

	std::vector<std::vector<point_i>> quantize_points(const std::vector<std::vector<point_i>>& faces)
	{
		auto result = faces;
		for (auto& face : result)
			for (auto& point : face)
			{
				point.x = static_cast<int>(std::lround(point.x / 2.0));
				point.y = static_cast<int>(std::lround(point.y / 2.0));
			}
		return result;
	}

	bool same_shape(const std::vector<std::vector<point_i>>& a, const std::vector<std::vector<point_i>>& b)
	{
		if (a.size() != b.size()) return false;
		for (size_t i = 0; i < a.size(); ++i)
			if (a[i].size() != b[i].size()) return false;
		return true;
	}

	bool same_points(const std::vector<std::vector<point_i>>& a, const std::vector<std::vector<point_i>>& b)
	{
		if (!same_shape(a, b)) return false;
		for (size_t face = 0; face < a.size(); ++face)
			for (size_t point = 0; point < a[face].size(); ++point)
				if (a[face][point].x != b[face][point].x || a[face][point].y != b[face][point].y)
					return false;
		return true;
	}
}

void temporal_landmark_stream::reset()
{
	encoder_points.clear();
	decoder_points.clear();
}

landmark_packet temporal_landmark_stream::encode(const std::vector<std::vector<point_i>>& faces,
	                                             bool force_keyframe)
{
	landmark_packet packet;
	const auto quantized = quantize_points(faces);
	force_keyframe = force_keyframe || !same_shape(quantized, encoder_points);
	if (!force_keyframe && same_points(quantized, encoder_points)) return packet;
	packet.bytes = {'L', 'M', 1, static_cast<uint8_t>(force_keyframe ? 1 : 0),
		static_cast<uint8_t>(std::min<size_t>(quantized.size(), 255))};
	packet.keyframe = force_keyframe;

	if (force_keyframe)
	{
		for (const auto& face : quantized)
		{
			packet.bytes.push_back(static_cast<uint8_t>(std::min<size_t>(face.size(), 255)));
			for (const point_i point : face)
			{
				wire_append_varuint(packet.bytes, wire_zigzag_encode(point.x));
				wire_append_varuint(packet.bytes, wire_zigzag_encode(point.y));
			}
		}
		encoder_points = quantized;
		return packet;
	}

	for (size_t face_index = 0; face_index < quantized.size(); ++face_index)
	{
		const auto& face = quantized[face_index];
		std::vector<uint8_t> nibbles;
		std::vector<uint8_t> escapes;
		nibbles.reserve(face.size() * 2);
		for (size_t point_index = 0; point_index < face.size(); ++point_index)
		{
			const int deltas[2] = {face[point_index].x - encoder_points[face_index][point_index].x,
				face[point_index].y - encoder_points[face_index][point_index].y};
			for (const int delta : deltas)
			{
				if (delta >= -7 && delta <= 7) nibbles.push_back(static_cast<uint8_t>(delta + 7));
				else
				{
					nibbles.push_back(15);
					wire_append_varuint(escapes, wire_zigzag_encode(delta));
				}
			}
		}

		packet.bytes.push_back(static_cast<uint8_t>(face.size()));
		wire_append_u16(packet.bytes, escapes.size());
		for (size_t i = 0; i < nibbles.size(); i += 2)
		{
			const uint8_t high = i + 1 < nibbles.size() ? nibbles[i + 1] : 0;
			packet.bytes.push_back(static_cast<uint8_t>(nibbles[i] | high << 4));
		}
		packet.bytes.insert(packet.bytes.end(), escapes.begin(), escapes.end());
	}
	encoder_points = quantized;
	return packet;
}

bool temporal_landmark_stream::decode(const landmark_packet& packet,
	                                  std::vector<std::vector<point_i>>& faces)
{
	if (packet.bytes.size() < 5 || packet.bytes[0] != 'L' || packet.bytes[1] != 'M'
		|| packet.bytes[2] != 1) return false;
	const bool keyframe = (packet.bytes[3] & 1) != 0;
	const size_t face_count = packet.bytes[4];
	size_t offset = 5;
	std::vector<std::vector<point_i>> next;

	if (keyframe)
	{
		next.resize(face_count);
		for (auto& face : next)
		{
			if (offset >= packet.bytes.size()) return false;
			face.resize(packet.bytes[offset++]);
			for (auto& point : face)
			{
				uint32_t x = 0;
				uint32_t y = 0;
				if (!wire_read_varuint(packet.bytes, offset, x) || !wire_read_varuint(packet.bytes, offset, y))
					return false;
				point = {wire_zigzag_decode(x), wire_zigzag_decode(y)};
			}
		}
	}
	else
	{
		if (decoder_points.size() != face_count) return false;
		next = decoder_points;
		for (size_t face_index = 0; face_index < face_count; ++face_index)
		{
			if (offset >= packet.bytes.size()) return false;
			const size_t point_count = packet.bytes[offset++];
			size_t escape_size = 0;
			if (point_count != next[face_index].size() || !wire_read_u16(packet.bytes, offset, escape_size))
				return false;
			const size_t packed_size = (point_count * 2 + 1) / 2;
			if (offset + packed_size + escape_size > packet.bytes.size()) return false;
			const size_t packed_offset = offset;
			size_t escape_offset = offset + packed_size;
			const size_t escape_end = escape_offset + escape_size;
			offset = escape_end;

			for (size_t coordinate = 0; coordinate < point_count * 2; ++coordinate)
			{
				const uint8_t packed = packet.bytes[packed_offset + coordinate / 2];
				const uint8_t nibble = coordinate % 2 == 0 ? packed & 0x0f : packed >> 4;
				int delta = 0;
				if (nibble < 15) delta = static_cast<int>(nibble) - 7;
				else
				{
					uint32_t escaped = 0;
					if (!wire_read_varuint(packet.bytes, escape_offset, escaped) || escape_offset > escape_end)
						return false;
					delta = wire_zigzag_decode(escaped);
				}
				point_i& point = next[face_index][coordinate / 2];
				if (coordinate % 2 == 0) point.x += delta;
				else point.y += delta;
			}
			if (escape_offset != escape_end) return false;
		}
	}
	if (offset != packet.bytes.size()) return false;

	decoder_points = next;
	faces = next;
	for (auto& face : faces)
		for (auto& point : face)
		{
			point.x *= 2;
			point.y *= 2;
		}
	return true;
}
