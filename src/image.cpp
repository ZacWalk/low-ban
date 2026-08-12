#include "image.h"

#include <cmath>
#include <cstring>

namespace
{
	uint8_t clamp_u8(const int v)
	{
		return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
	}
}

bool decode_yuy2(video_frame& out, const uint8_t* buffer, const size_t buffer_len, const int width, const int height,
                 const int stride)
{
	const int abs_stride = std::abs(stride);
	if (buffer == nullptr || width <= 1 || height <= 0 || (width & 1) != 0) return false;
	if (abs_stride < width * 2) return false;
	if (buffer_len < static_cast<size_t>(abs_stride) * static_cast<size_t>(height)) return false;

	const int chroma_width = width / 2;
	out.luma.resize(width, height);
	out.cb.resize(chroma_width, height);
	out.cr.resize(chroma_width, height);

	for (int y = 0; y < height; ++y)
	{
		const int src_y = stride < 0 ? height - 1 - y : y;
		const uint8_t* src = buffer + static_cast<size_t>(src_y) * abs_stride;
		uint8_t* luma = out.luma.row(y);
		uint8_t* cb = out.cb.row(y);
		uint8_t* cr = out.cr.row(y);

		for (int x = 0; x < chroma_width; ++x)
		{
			luma[x * 2 + 0] = src[x * 4 + 0];
			cb[x] = src[x * 4 + 1];
			luma[x * 2 + 1] = src[x * 4 + 2];
			cr[x] = src[x * 4 + 3];
		}
	}

	return true;
}

void yuv_to_bgra(const video_frame& frame, std::vector<uint8_t>& bgra)
{
	const int w = frame.width();
	const int h = frame.height();
	bgra.resize(static_cast<size_t>(w) * h * 4);

	for (int y = 0; y < h; ++y)
	{
		const uint8_t* luma = frame.luma.row(y);
		const uint8_t* cb = frame.cb.row(y);
		const uint8_t* cr = frame.cr.row(y);
		uint8_t* dst = bgra.data() + static_cast<size_t>(y) * w * 4;

		for (int x = 0; x < w; ++x)
		{
			const int c = luma[x] - 16;
			const int d = cb[x / 2] - 128;
			const int e = cr[x / 2] - 128;

			dst[x * 4 + 0] = clamp_u8((298 * c + 516 * d + 128) >> 8);
			dst[x * 4 + 1] = clamp_u8((298 * c - 100 * d - 208 * e + 128) >> 8);
			dst[x * 4 + 2] = clamp_u8((298 * c + 409 * e + 128) >> 8);
			dst[x * 4 + 3] = 255;
		}
	}
}

void resize_bilinear(const image_u8& src, image_u8& dst)
{
	const int sw = src.width();
	const int sh = src.height();
	const int dw = dst.width();
	const int dh = dst.height();
	if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;

	// Half-pixel centred mapping in 16.16 fixed point.
	const int64_t x_ratio = (static_cast<int64_t>(sw) << 16) / dw;
	const int64_t y_ratio = (static_cast<int64_t>(sh) << 16) / dh;

	for (int y = 0; y < dh; ++y)
	{
		const int64_t sy = (y * y_ratio + (y_ratio >> 1)) - (1 << 15);
		const int y0 = std::clamp(static_cast<int>(sy >> 16), 0, sh - 1);
		const int y1 = std::min(y0 + 1, sh - 1);
		const int fy = static_cast<int>(sy < 0 ? 0 : (sy & 0xFFFF));

		const uint8_t* r0 = src.row(y0);
		const uint8_t* r1 = src.row(y1);
		uint8_t* out = dst.row(y);

		for (int x = 0; x < dw; ++x)
		{
			const int64_t sx = (x * x_ratio + (x_ratio >> 1)) - (1 << 15);
			const int x0 = std::clamp(static_cast<int>(sx >> 16), 0, sw - 1);
			const int x1 = std::min(x0 + 1, sw - 1);
			const int fx = static_cast<int>(sx < 0 ? 0 : (sx & 0xFFFF));

			const int top = r0[x0] + (((r0[x1] - r0[x0]) * fx) >> 16);
			const int bottom = r1[x0] + (((r1[x1] - r1[x0]) * fx) >> 16);
			out[x] = clamp_u8(top + (((bottom - top) * fy) >> 16));
		}
	}
}

void gaussian_blur(const image_u8& src, image_u8& dst)
{
	const int w = src.width();
	const int h = src.height();
	dst.resize(w, h);
	if (w < 5 || h < 5)
	{
		std::memcpy(dst.data(), src.data(), src.size());
		return;
	}

	// Separable [1 4 6 4 1] / 16, edges clamped.
	static constexpr int kernel[5] = {1, 4, 6, 4, 1};
	image<uint16_t> tmp(w, h);

	for (int y = 0; y < h; ++y)
	{
		const uint8_t* in = src.row(y);
		uint16_t* out = tmp.row(y);
		for (int x = 0; x < w; ++x)
		{
			int sum = 0;
			for (int k = -2; k <= 2; ++k)
				sum += kernel[k + 2] * in[std::clamp(x + k, 0, w - 1)];
			out[x] = static_cast<uint16_t>(sum);
		}
	}

	for (int y = 0; y < h; ++y)
	{
		uint8_t* out = dst.row(y);
		for (int x = 0; x < w; ++x)
		{
			int sum = 0;
			for (int k = -2; k <= 2; ++k)
				sum += kernel[k + 2] * tmp.row(std::clamp(y + k, 0, h - 1))[x];
			out[x] = clamp_u8(sum / 256);
		}
	}
}

void sobel_edges(const image_u8& src, image_u8& dst)
{
	const int w = src.width();
	const int h = src.height();
	dst.resize(w, h);
	dst.fill(0);
	if (w < 3 || h < 3) return;

	image<int16_t> gx(w, h);
	image<int16_t> gy(w, h);

	for (int y = 1; y < h - 1; ++y)
	{
		const uint8_t* r0 = src.row(y - 1);
		const uint8_t* r1 = src.row(y);
		const uint8_t* r2 = src.row(y + 1);

		for (int x = 1; x < w - 1; ++x)
		{
			gx(x, y) = static_cast<int16_t>((r0[x + 1] + 2 * r1[x + 1] + r2[x + 1])
				- (r0[x - 1] + 2 * r1[x - 1] + r2[x - 1]));
			gy(x, y) = static_cast<int16_t>((r2[x - 1] + 2 * r2[x] + r2[x + 1])
				- (r0[x - 1] + 2 * r0[x] + r0[x + 1]));
		}
	}

	// Non-maximum suppression along the gradient direction, quantised to 4 orientations.
	for (int y = 1; y < h - 1; ++y)
	{
		for (int x = 1; x < w - 1; ++x)
		{
			const int dx = gx(x, y);
			const int dy = gy(x, y);
			const int magnitude = std::abs(dx) + std::abs(dy);
			if (magnitude == 0) continue;

			int step_x = 0;
			int step_y = 0;
			const int adx = std::abs(dx);
			const int ady = std::abs(dy);

			if (adx > 2 * ady) step_x = 1;
			else if (ady > 2 * adx) step_y = 1;
			else
			{
				step_x = 1;
				step_y = (dx > 0) == (dy > 0) ? 1 : -1;
			}

			const auto neighbour = [&](const int ox, const int oy)
			{
				return std::abs(static_cast<int>(gx(x + ox, y + oy))) + std::abs(static_cast<int>(gy(x + ox, y + oy)));
			};

			if (magnitude >= neighbour(step_x, step_y) && magnitude > neighbour(-step_x, -step_y))
			{
				dst(x, y) = clamp_u8(magnitude / 4);
			}
		}
	}
}

void contrast_stretch(image_u8& img)
{
	if (img.empty()) return;

	uint8_t lo = 255;
	uint8_t hi = 0;
	for (size_t i = 0; i < img.size(); ++i)
	{
		lo = std::min(lo, img.data()[i]);
		hi = std::max(hi, img.data()[i]);
	}

	const int range = hi - lo;
	if (range <= 0) return;

	for (size_t i = 0; i < img.size(); ++i)
	{
		img.data()[i] = clamp_u8((img.data()[i] - lo) * 255 / range);
	}
}
