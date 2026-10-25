// Face detection, adapted from libfacedetection.
//
// Original work: Copyright (c) 2018-2023, Shiqi Yu, all rights reserved.
// https://github.com/ShiqiYu/libfacedetection
//
//                   License Agreement For libfacedetection
//                        (3-clause BSD License)
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * Neither the names of the copyright holders nor the names of the contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
#include "face.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{
	// Detection runs at half the capture resolution. A face in a webcam frame is hundreds of
	// pixels across, so this costs nothing in recall and a quarter of the arithmetic.
	constexpr int detect_step = 2;

	// The backbone pools three times, so both blob axes must stay even down to stride 32.
	constexpr int pad_divisor = 32;

	// The reference implementation's default. Anything stricter risks dropping a real face,
	// and the landmark stage is tolerant of the occasional false positive.
	constexpr float confidence_threshold = 0.3f;
	constexpr float overlap_threshold = 0.45f;
	constexpr size_t max_faces = 8;

	// The trained boxes span hairline to chin, wider than the square the landmark model
	// wants. The engine refits the box from the first prediction, so this only has to be
	// close.
	constexpr float box_top_fraction = 0.20f; // of the detected height
	constexpr float box_side_fraction = 1.00f; // of the detected width

	struct blob
	{
		int rows = 0;
		int cols = 0;
		int channels = 0;
		std::vector<float> values;

		blob() = default;

		blob(const int r, const int c, const int ch) : rows(r), cols(c), channels(ch),
		                                               values(static_cast<size_t>(r) * c * ch, 0.0f)
		{
		}

		float* at(const int r, const int c)
		{
			return values.data() + (static_cast<size_t>(r) * cols + c) * channels;
		}

		const float* at(const int r, const int c) const
		{
			return values.data() + (static_cast<size_t>(r) * cols + c) * channels;
		}

		bool empty() const { return values.empty(); }
	};

	struct candidate
	{
		float score;
		float xmin, ymin, xmax, ymax;
	};

	float dot(const float* a, const float* b, const int n)
	{
		float sum = 0.0f;
		for (int i = 0; i < n; ++i) sum += a[i] * b[i];
		return sum;
	}

	void apply_relu(blob& data)
	{
		for (auto& v : data.values) v = v > 0.0f ? v : 0.0f;
	}

	// 1x1 convolution: weights are num_filters consecutive vectors of `channels` values.
	blob pointwise(const blob& input, const face_conv_layer& filter)
	{
		if (input.channels != filter.channels) return {};

		blob output(input.rows, input.cols, filter.num_filters);

		for (int r = 0; r < output.rows; ++r)
		{
			for (int c = 0; c < output.cols; ++c)
			{
				const float* in = input.at(r, c);
				float* out = output.at(r, c);
				for (int f = 0; f < filter.num_filters; ++f)
				{
					out[f] = dot(in, filter.weights + static_cast<size_t>(f) * filter.channels, filter.channels)
						+ filter.biases[f];
				}
			}
		}

		return output;
	}

	// 3x3 depthwise convolution: weights are 9 consecutive vectors of `channels` values, one
	// per tap position, and each channel is convolved independently.
	blob depthwise(const blob& input, const face_conv_layer& filter)
	{
		if (input.channels != filter.channels || filter.num_filters != filter.channels) return {};

		blob output(input.rows, input.cols, filter.num_filters);

		for (int r = 0; r < output.rows; ++r)
		{
			const int row_begin = std::max(0, r - 1);
			const int row_end = std::min(r + 2, input.rows);

			for (int c = 0; c < output.cols; ++c)
			{
				const int col_begin = std::max(0, c - 1);
				const int col_end = std::min(c + 2, input.cols);
				float* out = output.at(r, c);

				for (int sr = row_begin; sr < row_end; ++sr)
				{
					for (int sc = col_begin; sc < col_end; ++sc)
					{
						const int tap = (sr - r + 1) * 3 + (sc - c + 1);
						const float* in = input.at(sr, sc);
						const float* weights = filter.weights + static_cast<size_t>(tap) * filter.channels;
						for (int ch = 0; ch < filter.num_filters; ++ch) out[ch] += in[ch] * weights[ch];
					}
				}

				for (int ch = 0; ch < filter.num_filters; ++ch) out[ch] += filter.biases[ch];
			}
		}

		return output;
	}

	blob convolution(const blob& input, const face_conv_layer& filter, const bool with_relu = true)
	{
		blob output = filter.is_depthwise ? depthwise(input, filter) : pointwise(input, filter);
		if (with_relu) apply_relu(output);
		return output;
	}

	blob convolution_dp(const blob& input, const face_conv_layer& point, const face_conv_layer& depth,
	                    const bool with_relu = true)
	{
		return convolution(convolution(input, point, false), depth, with_relu);
	}

	blob convolution_unit(const blob& input, const face_conv_layer& point1, const face_conv_layer& depth1,
	                      const face_conv_layer& point2, const face_conv_layer& depth2)
	{
		return convolution_dp(convolution_dp(input, point1, depth1), point2, depth2);
	}

	blob max_pool_2x2(const blob& input)
	{
		blob output((input.rows - 2) / 2 + 1, (input.cols - 2) / 2 + 1, input.channels);

		for (int r = 0; r < output.rows; ++r)
		{
			const int row_end = std::min(r * 2 + 2, input.rows);

			for (int c = 0; c < output.cols; ++c)
			{
				const int col_end = std::min(c * 2 + 2, input.cols);
				float* out = output.at(r, c);
				std::memcpy(out, input.at(r * 2, c * 2), sizeof(float) * output.channels);

				for (int sr = r * 2; sr < row_end; ++sr)
				{
					for (int sc = c * 2; sc < col_end; ++sc)
					{
						const float* in = input.at(sr, sc);
						for (int ch = 0; ch < output.channels; ++ch) out[ch] = std::max(out[ch], in[ch]);
					}
				}
			}
		}

		return output;
	}

	// Nearest neighbour upsample, then accumulate into the matching pyramid level.
	void add_upsampled(const blob& coarse, blob& fine)
	{
		if (coarse.channels != fine.channels) return;

		for (int r = 0; r < fine.rows; ++r)
		{
			for (int c = 0; c < fine.cols; ++c)
			{
				const float* in = coarse.at(std::min(r / 2, coarse.rows - 1), std::min(c / 2, coarse.cols - 1));
				float* out = fine.at(r, c);
				for (int ch = 0; ch < fine.channels; ++ch) out[ch] += in[ch];
			}
		}
	}

	// The first layer is a 3x3 stride 2 convolution, expressed as a gather so that every
	// later layer can be 1x1 or depthwise. Each output cell holds the 3x3 BGR neighbourhood.
	blob gather_input(const std::vector<uint8_t>& bgr, const int width, const int height)
	{
		const int rows = ((height - 1) / pad_divisor + 1) * pad_divisor / 2;
		const int cols = ((width - 1) / pad_divisor + 1) * pad_divisor / 2;
		blob output(rows, cols, 32);

		for (int r = 0; r < rows; ++r)
		{
			for (int c = 0; c < cols; ++c)
			{
				float* out = output.at(r, c);

				for (int fy = -1; fy <= 1; ++fy)
				{
					const int sy = r * 2 + fy;
					if (sy < 0 || sy >= height) continue;

					for (int fx = -1; fx <= 1; ++fx)
					{
						const int sx = c * 2 + fx;
						if (sx < 0 || sx >= width) continue;

						const uint8_t* pixel = bgr.data() + (static_cast<size_t>(sy) * width + sx) * 3;
						const int tap = ((fy + 1) * 3 + fx + 1) * 3;
						out[tap + 0] = pixel[0];
						out[tap + 1] = pixel[1];
						out[tap + 2] = pixel[2];
					}
				}
			}
		}

		return output;
	}

	uint8_t clamp_u8(const int v)
	{
		return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
	}

	void to_bgr(const video_frame& frame, const int step, std::vector<uint8_t>& bgr, int& width, int& height)
	{
		width = frame.width() / step;
		height = frame.height() / step;
		bgr.resize(static_cast<size_t>(width) * height * 3);

		for (int y = 0; y < height; ++y)
		{
			const int sy = y * step;
			const uint8_t* luma = frame.luma.row(sy);
			const uint8_t* cb = frame.cb.row(sy);
			const uint8_t* cr = frame.cr.row(sy);
			uint8_t* dst = bgr.data() + static_cast<size_t>(y) * width * 3;

			for (int x = 0; x < width; ++x)
			{
				const int sx = x * step;
				const int c = luma[sx] - 16;
				const int d = cb[sx / 2] - 128;
				const int e = cr[sx / 2] - 128;

				dst[x * 3 + 0] = clamp_u8((298 * c + 516 * d + 128) >> 8);
				dst[x * 3 + 1] = clamp_u8((298 * c - 100 * d - 208 * e + 128) >> 8);
				dst[x * 3 + 2] = clamp_u8((298 * c + 409 * e + 128) >> 8);
			}
		}
	}

	float sigmoid(const float x)
	{
		const float clamped = std::min(std::max(x, -88.0f), 88.0f);
		return 1.0f / (1.0f + std::exp(-clamped));
	}

	// Turns one pyramid level into scored boxes in detection-image pixels.
	void collect(const blob& cls, const blob& obj, const blob& reg, const int stride,
	             std::vector<candidate>& out, float& peak)
	{
		if (cls.empty() || obj.empty() || reg.empty() || reg.channels < 4) return;

		for (int r = 0; r < cls.rows; ++r)
		{
			for (int c = 0; c < cls.cols; ++c)
			{
				const float score = std::sqrt(sigmoid(cls.at(r, c)[0]) * sigmoid(obj.at(r, c)[0]));
				peak = std::max(peak, score);
				if (score < confidence_threshold) continue;

				const float* box = reg.at(r, c);
				const auto fstride = static_cast<float>(stride);
				const float cx = box[0] * fstride + static_cast<float>(c * stride);
				const float cy = box[1] * fstride + static_cast<float>(r * stride);
				const float w = std::exp(box[2]) * fstride;
				const float h = std::exp(box[3]) * fstride;

				out.push_back({score, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2});
			}
		}
	}

	float overlap(const candidate& a, const candidate& b)
	{
		const float w = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
		const float h = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);
		if (w <= 0.0f || h <= 0.0f) return 0.0f;

		const float intersection = w * h;
		const float total = (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin);
		return intersection / (total - intersection);
	}

	void suppress(std::vector<candidate>& boxes)
	{
		std::stable_sort(boxes.begin(), boxes.end(),
		                 [](const candidate& a, const candidate& b) { return a.score > b.score; });

		std::vector<candidate> kept;
		for (const auto& box : boxes)
		{
			bool keep = true;
			for (const auto& other : kept)
			{
				if (overlap(box, other) > overlap_threshold)
				{
					keep = false;
					break;
				}
			}
			if (keep) kept.push_back(box);
			if (kept.size() >= max_faces) break;
		}

		boxes = std::move(kept);
	}
}

face_detection face_detector::detect(const video_frame& frame) const
{
	face_detection result;
	if (frame.empty() || frame.width() < pad_divisor * detect_step) return result;

	std::vector<uint8_t> bgr;
	int width = 0;
	int height = 0;
	to_bgr(frame, detect_step, bgr, width, height);

	const auto& layer = face_model_layers;

	blob fx = gather_input(bgr, width, height);
	fx = convolution(fx, layer[0]);
	fx = convolution_dp(fx, layer[1], layer[2]);
	fx = max_pool_2x2(fx);

	fx = convolution_unit(fx, layer[3], layer[4], layer[5], layer[6]);
	fx = convolution_unit(fx, layer[7], layer[8], layer[9], layer[10]);

	fx = max_pool_2x2(fx);
	blob level3 = convolution_unit(fx, layer[11], layer[12], layer[13], layer[14]);

	fx = max_pool_2x2(level3);
	blob level4 = convolution_unit(fx, layer[15], layer[16], layer[17], layer[18]);

	fx = max_pool_2x2(level4);
	blob level5 = convolution_unit(fx, layer[19], layer[20], layer[21], layer[22]);

	std::vector<candidate> boxes;

	level5 = convolution_dp(level5, layer[27], layer[28]);
	collect(convolution_dp(level5, layer[33], layer[34], false),
	        convolution_dp(level5, layer[45], layer[46], false),
	        convolution_dp(level5, layer[39], layer[40], false), 32, boxes, result.peak_confidence);

	add_upsampled(level5, level4);
	level4 = convolution_dp(level4, layer[25], layer[26]);
	collect(convolution_dp(level4, layer[31], layer[32], false),
	        convolution_dp(level4, layer[43], layer[44], false),
	        convolution_dp(level4, layer[37], layer[38], false), 16, boxes, result.peak_confidence);

	add_upsampled(level4, level3);
	level3 = convolution_dp(level3, layer[23], layer[24]);
	collect(convolution_dp(level3, layer[29], layer[30], false),
	        convolution_dp(level3, layer[41], layer[42], false),
	        convolution_dp(level3, layer[35], layer[36], false), 8, boxes, result.peak_confidence);

	suppress(boxes);

	for (const auto& box : boxes)
	{
		const float scale = static_cast<float>(detect_step);
		const float x = box.xmin * scale;
		const float y = box.ymin * scale;
		const float w = (box.xmax - box.xmin) * scale;
		const float h = (box.ymax - box.ymin) * scale;

		const float side = w * box_side_fraction;
		const float left = x + w / 2 - side / 2;
		const float top = y + h * box_top_fraction;

		result.faces.push_back(rect_i::from_size(static_cast<int>(std::lround(left)),
		                                         static_cast<int>(std::lround(top)),
		                                         static_cast<int>(std::lround(side)),
		                                         static_cast<int>(std::lround(side))));
	}

	return result;
}
