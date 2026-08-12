#include "draw.h"

#include <cmath>

namespace
{
	void blend(image_u8& img, const int x, const int y, const uint8_t colour, const float alpha)
	{
		if (alpha <= 0.0f || !img.contains(x, y)) return;
		const float a = alpha > 1.0f ? 1.0f : alpha;
		const float v = img(x, y) * (1.0f - a) + colour * a;
		img(x, y) = static_cast<uint8_t>(v + 0.5f);
	}

	// The 68 point layout: contiguous runs, some of which close back on themselves.
	struct feature_run
	{
		int first;
		int last;
		bool closed;
	};

	constexpr feature_run face_runs[] = {
		{0, 16, false}, // jaw
		{17, 21, false}, // left brow
		{22, 26, false}, // right brow
		{27, 30, false}, // nose bridge
		{31, 35, false}, // nostrils
		{36, 41, true}, // left eye
		{42, 47, true}, // right eye
		{48, 59, true}, // outer lip
		{60, 67, true}, // inner lip
	};
}

void draw_line(image_u8& img, float x0, float y0, float x1, float y1, const uint8_t colour)
{
	// Xiaolin Wu's algorithm, without the endpoint special cases.
	const bool steep = std::fabs(y1 - y0) > std::fabs(x1 - x0);
	if (steep)
	{
		std::swap(x0, y0);
		std::swap(x1, y1);
	}
	if (x0 > x1)
	{
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	const float dx = x1 - x0;
	const float gradient = dx == 0.0f ? 0.0f : (y1 - y0) / dx;

	const int first = static_cast<int>(std::lround(x0));
	const int last = static_cast<int>(std::lround(x1));

	for (int x = first; x <= last; ++x)
	{
		const float y = y0 + gradient * (static_cast<float>(x) - x0);
		const float base = std::floor(y);
		const float frac = y - base;
		const int iy = static_cast<int>(base);

		if (steep)
		{
			blend(img, iy, x, colour, 1.0f - frac);
			blend(img, iy + 1, x, colour, frac);
		}
		else
		{
			blend(img, x, iy, colour, 1.0f - frac);
			blend(img, x, iy + 1, colour, frac);
		}
	}
}

void draw_rect(image_u8& img, const rect_i& r, const uint8_t colour)
{
	if (r.empty()) return;

	const auto l = static_cast<float>(r.left);
	const auto t = static_cast<float>(r.top);
	const auto rr = static_cast<float>(r.right);
	const auto b = static_cast<float>(r.bottom);

	draw_line(img, l, t, rr, t, colour);
	draw_line(img, rr, t, rr, b, colour);
	draw_line(img, rr, b, l, b, colour);
	draw_line(img, l, b, l, t, colour);
}

void draw_face(image_u8& img, const std::vector<point_i>& parts, const uint8_t colour)
{
	if (parts.size() != 68) return;

	const auto segment = [&](const int a, const int b)
	{
		draw_line(img, static_cast<float>(parts[a].x), static_cast<float>(parts[a].y),
		          static_cast<float>(parts[b].x), static_cast<float>(parts[b].y), colour);
	};

	for (const auto& run : face_runs)
	{
		for (int i = run.first; i < run.last; ++i) segment(i, i + 1);
		if (run.closed) segment(run.last, run.first);
	}
}
