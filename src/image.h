// Basic image containers and pixel operations.
// Everything downstream works on these types, never on raw capture buffers.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

struct point_i
{
	int x = 0;
	int y = 0;
};

struct point_f
{
	float x = 0.0f;
	float y = 0.0f;
};

// Inclusive on all four edges, matching the convention used by the landmark model file.
struct rect_i
{
	int left = 0;
	int top = 0;
	int right = -1;
	int bottom = -1;

	int width() const { return right - left + 1; }
	int height() const { return bottom - top + 1; }
	bool empty() const { return right < left || bottom < top; }

	static rect_i from_size(const int x, const int y, const int w, const int h)
	{
		return {x, y, x + w - 1, y + h - 1};
	}
};

template <typename T>
class image
{
	int frame_width = 0;
	int frame_height = 0;
	std::vector<T> frame_pixels;

public:
	image() = default;

	image(const int w, const int h) : frame_width(w), frame_height(h),
	                                  frame_pixels(static_cast<size_t>(w) * static_cast<size_t>(h))
	{
	}

	int width() const { return frame_width; }
	int height() const { return frame_height; }
	bool empty() const { return frame_pixels.empty(); }
	size_t size() const { return frame_pixels.size(); }

	T* data() { return frame_pixels.data(); }
	const T* data() const { return frame_pixels.data(); }

	T* row(const int y) { return frame_pixels.data() + static_cast<size_t>(y) * frame_width; }
	const T* row(const int y) const { return frame_pixels.data() + static_cast<size_t>(y) * frame_width; }

	T& operator()(const int x, const int y) { return row(y)[x]; }
	const T& operator()(const int x, const int y) const { return row(y)[x]; }

	bool contains(const int x, const int y) const
	{
		return x >= 0 && y >= 0 && x < frame_width && y < frame_height;
	}

	// Zero outside the image; the landmark sampler relies on this.
	T sample_or_zero(const int x, const int y) const
	{
		return contains(x, y) ? row(y)[x] : T{};
	}

	void resize(const int w, const int h)
	{
		if (w == frame_width && h == frame_height) return;
		frame_width = w;
		frame_height = h;
		frame_pixels.assign(static_cast<size_t>(w) * static_cast<size_t>(h), T{});
	}

	void fill(const T v) { std::fill(frame_pixels.begin(), frame_pixels.end(), v); }

	rect_i bounds() const { return {0, 0, frame_width - 1, frame_height - 1}; }
};

using image_u8 = image<uint8_t>;
using image_f32 = image<float>;

// A captured frame in planar form. Chroma is half width (YUY2 subsampling) and full height.
struct video_frame
{
	image_u8 luma;
	image_u8 cb;
	image_u8 cr;

	int width() const { return luma.width(); }
	int height() const { return luma.height(); }
	bool empty() const { return luma.empty(); }
};

// Decodes a packed YUY2 buffer (Y0 Cb Y1 Cr) into planes. A negative stride means the
// source rows are stored bottom-up, which is normalised away here.
bool decode_yuy2(video_frame& out, const uint8_t* buffer, size_t buffer_len, int width, int height, int stride);

void yuv_to_bgra(const video_frame& frame, std::vector<uint8_t>& bgra);

// The destination must already be at the wanted size; that is what selects the scale.
void resize_bilinear(const image_u8& src, image_u8& dst);
void gaussian_blur(const image_u8& src, image_u8& dst);
void sobel_edges(const image_u8& src, image_u8& dst);
void contrast_stretch(image_u8& img);
