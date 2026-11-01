#include "warp.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace
{
	constexpr size_t landmark_count = 68;

	// Mesh coordinates are clamped to this range and the enclosing triangle is kept within a
	// few spans of the points, so every predicate below fits comfortably in 64 bits.
	constexpr int coordinate_limit = 2048;

	int64_t floor_div(const int64_t value, const int64_t divisor)
	{
		const int64_t quotient = value / divisor;
		return quotient * divisor > value ? quotient - 1 : quotient;
	}

	int64_t round_div(const int64_t value, const int64_t divisor)
	{
		return floor_div(value + divisor / 2, divisor);
	}

	// Twice the signed area of oab. Positive means b is to the left of o->a in a y-up frame.
	int64_t cross(const point_i& o, const point_i& a, const point_i& b)
	{
		return static_cast<int64_t>(a.x - o.x) * (b.y - o.y) - static_cast<int64_t>(a.y - o.y) * (b.x - o.x);
	}

	// True when d lies strictly inside the circumcircle of abc, which must have positive area.
	bool in_circumcircle(const point_i& a, const point_i& b, const point_i& c, const point_i& d)
	{
		const int64_t adx = a.x - d.x;
		const int64_t ady = a.y - d.y;
		const int64_t bdx = b.x - d.x;
		const int64_t bdy = b.y - d.y;
		const int64_t cdx = c.x - d.x;
		const int64_t cdy = c.y - d.y;
		const int64_t alift = adx * adx + ady * ady;
		const int64_t blift = bdx * bdx + bdy * bdy;
		const int64_t clift = cdx * cdx + cdy * cdy;
		return adx * (bdy * clift - cdy * blift) - ady * (bdx * clift - cdx * blift)
			+ alift * (bdx * cdy - cdx * bdy) > 0;
	}

	point_i clamp_point(const point_i p)
	{
		return {std::clamp(p.x, -coordinate_limit, coordinate_limit),
			std::clamp(p.y, -coordinate_limit, coordinate_limit)};
	}

	// Bilinear sample at a position in 1/256 pixel units, edges clamped.
	uint8_t sample(const image_u8& img, const int64_t fx, const int64_t fy)
	{
		const int64_t x0 = floor_div(fx, 256);
		const int64_t y0 = floor_div(fy, 256);
		const int ax = static_cast<int>(fx - x0 * 256);
		const int ay = static_cast<int>(fy - y0 * 256);
		const auto at = [&](const int64_t x, const int64_t y)
		{
			return static_cast<int>(img(static_cast<int>(std::clamp<int64_t>(x, 0, img.width() - 1)),
				static_cast<int>(std::clamp<int64_t>(y, 0, img.height() - 1))));
		};
		const int top = at(x0, y0) * (256 - ax) + at(x0 + 1, y0) * ax;
		const int bottom = at(x0, y0 + 1) * (256 - ax) + at(x0 + 1, y0 + 1) * ax;
		return static_cast<uint8_t>((top * (256 - ay) + bottom * ay + 32768) >> 16);
	}

	// Fills the pixels of triangle d0 d1 d2 by sampling the matching point of s0 s1 s2. The
	// first triangle to reach a pixel keeps it, so shared edges are written exactly once.
	void warp_triangle(const image_u8& reference, point_i s0, point_i s1, point_i s2,
		const point_i d0, point_i d1, point_i d2, image_u8& covered, image_u8& out)
	{
		int64_t area = cross(d0, d1, d2);
		if (area == 0) return;
		if (area < 0)
		{
			std::swap(d1, d2);
			std::swap(s1, s2);
			area = -area;
		}

		const int left = std::max(0, std::min({d0.x, d1.x, d2.x}));
		const int right = std::min(out.width() - 1, std::max({d0.x, d1.x, d2.x}));
		const int top = std::max(0, std::min({d0.y, d1.y, d2.y}));
		const int bottom = std::min(out.height() - 1, std::max({d0.y, d1.y, d2.y}));

		for (int y = top; y <= bottom; ++y)
			for (int x = left; x <= right; ++x)
			{
				const point_i p{x, y};

				// Barycentric weights of d1 and d2, scaled by the area.
				const int64_t w1 = cross(d0, p, d2);
				const int64_t w2 = cross(d0, d1, p);
				if (w1 < 0 || w2 < 0 || w1 + w2 > area || covered(x, y) != 0) continue;
				covered(x, y) = 1;

				const int64_t sx = floor_div((s0.x * area + w1 * (s1.x - s0.x) + w2 * (s2.x - s0.x)) * 256, area);
				const int64_t sy = floor_div((s0.y * area + w1 * (s1.y - s0.y) + w2 * (s2.y - s0.y)) * 256, area);
				out(x, y) = sample(reference, sx, sy);
			}
	}

	bool same_points(const std::vector<point_i>& a, const std::vector<point_i>& b)
	{
		if (a.size() != b.size()) return false;
		for (size_t i = 0; i < a.size(); ++i)
			if (a[i].x != b[i].x || a[i].y != b[i].y) return false;
		return true;
	}
}

std::vector<point_i> face_mesh_points(const std::vector<point_i>& landmarks)
{
	std::vector<point_i> mesh;
	mesh.reserve(landmarks.size() + 10);
	for (const point_i p : landmarks) mesh.push_back(clamp_point(p));
	if (landmarks.size() != landmark_count) return mesh;

	// Up the face, from the chin to the centre of the eyes, then far enough again above each
	// brow point to reach roughly the hairline. Twelve eye points are summed, so the vector
	// is kept twelve times too long until the final division.
	int64_t up_x = 0;
	int64_t up_y = 0;
	for (int i = 36; i < 48; ++i)
	{
		up_x += landmarks[i].x - landmarks[8].x;
		up_y += landmarks[i].y - landmarks[8].y;
	}
	for (int i = 17; i <= 26; ++i)
		mesh.push_back(clamp_point({landmarks[i].x + static_cast<int>(round_div(up_x * 9, 12 * 20)),
			landmarks[i].y + static_cast<int>(round_div(up_y * 9, 12 * 20))}));
	return mesh;
}

std::vector<mesh_triangle> delaunay_triangles(const std::vector<point_i>& input)
{
	std::vector<point_i> points;
	std::vector<int> original;
	for (size_t i = 0; i < input.size(); ++i)
	{
		const point_i p = clamp_point(input[i]);
		bool repeated = false;
		for (const point_i& q : points)
			if (q.x == p.x && q.y == p.y) repeated = true;
		if (repeated) continue;
		points.push_back(p);
		original.push_back(static_cast<int>(i));
	}
	const int count = static_cast<int>(points.size());
	if (count < 3) return {};

	int left = points[0].x, right = points[0].x, top = points[0].y, bottom = points[0].y;
	for (const point_i& p : points)
	{
		left = std::min(left, p.x);
		right = std::max(right, p.x);
		top = std::min(top, p.y);
		bottom = std::max(bottom, p.y);
	}

	// An enclosing triangle with a generous margin around the bounding box. Triangles that
	// still touch it at the end are dropped, which can leave a sliver of the convex hull
	// uncovered; uncovered pixels are simply not warped.
	const int span = std::max(right - left, bottom - top) + 16;
	const int centre_x = (left + right) / 2;
	const int centre_y = (top + bottom) / 2;
	points.push_back({centre_x - 2 * span, centre_y - span});
	points.push_back({centre_x, centre_y + 2 * span});
	points.push_back({centre_x + 2 * span, centre_y - span});

	struct triangle
	{
		int a, b, c;
	};
	std::vector<triangle> triangles{{count, count + 1, count + 2}};
	if (cross(points[count], points[count + 1], points[count + 2]) < 0) std::swap(triangles[0].b, triangles[0].c);

	std::vector<triangle> kept;
	std::vector<std::pair<int, int>> edges;
	for (int p = 0; p < count; ++p)
	{
		kept.clear();
		edges.clear();
		for (const triangle& t : triangles)
		{
			if (in_circumcircle(points[t.a], points[t.b], points[t.c], points[p]))
			{
				edges.emplace_back(t.a, t.b);
				edges.emplace_back(t.b, t.c);
				edges.emplace_back(t.c, t.a);
			}
			else kept.push_back(t);
		}

		// An edge shared by two removed triangles appears once in each direction; the rest
		// form the boundary of the cavity, and each is joined to the new point.
		for (const auto& edge : edges)
		{
			bool shared = false;
			for (const auto& other : edges)
				if (other.first == edge.second && other.second == edge.first) shared = true;
			if (!shared) kept.push_back({edge.first, edge.second, p});
		}
		triangles.swap(kept);
	}

	std::vector<mesh_triangle> result;
	for (const triangle& t : triangles)
	{
		if (t.a >= count || t.b >= count || t.c >= count) continue;
		if (cross(points[t.a], points[t.b], points[t.c]) <= 0) continue;
		result.push_back({original[t.a], original[t.b], original[t.c]});
	}
	return result;
}

bool warp_faces(const image_u8& reference, const face_geometry& from, const face_geometry& to, image_u8& out)
{
	out = reference;
	if (reference.empty()) return false;

	image_u8 covered;
	bool moved = false;
	const size_t faces = std::min(from.size(), to.size());
	for (size_t face = 0; face < faces; ++face)
	{
		if (from[face].size() != landmark_count || to[face].size() != landmark_count) continue;
		if (same_points(from[face], to[face])) continue;
		if (covered.empty())
		{
			covered.resize(reference.width(), reference.height());
			covered.fill(0);
		}

		const std::vector<point_i> source = face_mesh_points(from[face]);
		const std::vector<point_i> target = face_mesh_points(to[face]);
		for (const mesh_triangle& t : delaunay_triangles(target))
			warp_triangle(reference, source[t.a], source[t.b], source[t.c],
				target[t.a], target[t.b], target[t.c], covered, out);
		moved = true;
	}
	return moved;
}

face_geometry scale_geometry(const face_geometry& geometry, const int divisor)
{
	face_geometry scaled = geometry;
	if (divisor <= 1) return scaled;
	for (auto& face : scaled)
		for (auto& point : face)
		{
			point.x = static_cast<int>(floor_div(point.x, divisor));
			point.y = static_cast<int>(floor_div(point.y, divisor));
		}
	return scaled;
}
