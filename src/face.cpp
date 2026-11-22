// Faces through libmug, and the importance the expression prior gives them.
#include "face.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "libmug/detect.h"

namespace
{
	// Detection reads the luma reduced to this long edge. A face in a webcam frame is a hundred pixels or more
	// across, so half of a 640-pixel frame costs nothing in recall and three quarters of the arithmetic.
	constexpr int detect_edge = 320;

	// Below libmug's default of 70, which suits photographs: a webcam face in poor light scores lower, and a face
	// that drops out for a frame costs a geometry keyframe and its importance.
	constexpr int detect_confidence = 50;

	constexpr size_t max_faces = 8;

	int round_to_int(const double v)
	{
		return static_cast<int>(std::lround(v));
	}

	// The prior at a point of the template square, bilinear between cell centres; negative past the grid.
	double prior_at(const double x, const double y)
	{
		constexpr double cell = static_cast<double>(libmug::crop_extent) / expression_prior_cells;
		constexpr int last = expression_prior_cells - 1;
		const double gx = x / cell - 0.5;
		const double gy = y / cell - 0.5;
		if (gx < -0.5 || gy < -0.5 || gx > last + 0.5 || gy > last + 0.5) return -1.0;

		const double fx = std::clamp(gx, 0.0, static_cast<double>(last));
		const double fy = std::clamp(gy, 0.0, static_cast<double>(last));
		const int x0 = std::min(static_cast<int>(fx), last - 1);
		const int y0 = std::min(static_cast<int>(fy), last - 1);
		const double ax = fx - x0;
		const double ay = fy - y0;
		const auto at = [](const int cx, const int cy)
		{
			return static_cast<double>(expression_prior[cy * expression_prior_cells + cx]);
		};
		const double top = at(x0, y0) * (1.0 - ax) + at(x0 + 1, y0) * ax;
		const double bottom = at(x0, y0 + 1) * (1.0 - ax) + at(x0 + 1, y0 + 1) * ax;
		return top * (1.0 - ay) + bottom * ay;
	}
}

face_detector::face_detector() : network(std::make_unique<libmug::face_detector>())
{
}

face_detector::~face_detector() = default;

std::vector<detected_face> face_detector::detect(const image_u8& luma)
{
	std::vector<detected_face> faces;
	if (luma.empty()) return faces;

	libmug::image_view view;
	view.format = libmug::pixel_format::gray8;
	view.width = luma.width();
	view.height = luma.height();
	view.pixels = luma.data();
	view.stride = luma.width();
	view.limited_range = true;

	libmug::detect_options options;
	options.confidence = detect_confidence;
	options.edge = detect_edge;
	libmug::face_set found = network->detect(view, options);

	if (found.faces.size() > max_faces)
	{
		std::stable_sort(found.faces.begin(), found.faces.end(),
		                 [](const libmug::face& a, const libmug::face& b) { return a.confidence > b.confidence; });
		found.faces.resize(max_faces);
		libmug::order_faces(found.faces);
	}

	faces.reserve(found.faces.size());
	for (const libmug::face& found_face : found.faces)
	{
		detected_face face;
		face.box = rect_i::from_size(round_to_int(found_face.bounds.x), round_to_int(found_face.bounds.y),
		                             std::max(1, round_to_int(found_face.bounds.width)),
		                             std::max(1, round_to_int(found_face.bounds.height)));
		for (const libmug::point& mark : found_face.marks)
			face.points.push_back({round_to_int(mark.x), round_to_int(mark.y)});
		face.confidence = found_face.confidence;
		faces.push_back(std::move(face));
	}
	return faces;
}

void add_face_importance(image_u8& importance, const rect_i& box, const std::vector<point_i>& points)
{
	if (importance.empty()) return;

	if (!box.empty())
	{
		const int left = std::clamp(box.left, 0, importance.width() - 1);
		const int right = std::clamp(box.right, 0, importance.width() - 1);
		const int top = std::clamp(box.top, 0, importance.height() - 1);
		const int bottom = std::clamp(box.bottom, 0, importance.height() - 1);
		for (int y = top; y <= bottom; ++y)
			for (int x = left; x <= right; ++x) importance(x, y) = std::max(importance(x, y), face_importance);
	}

	if (points.size() != face_point_count) return;
	libmug::landmarks marks;
	for (size_t i = 0; i < marks.size(); ++i)
		marks[i] = {static_cast<double>(points[i].x), static_cast<double>(points[i].y)};
	const auto to_template = libmug::template_fit(marks);
	if (!to_template) return;
	const auto from_template = to_template->invert();
	if (!from_template) return;

	// The frame rectangle the template square lands in.
	double left = std::numeric_limits<double>::max();
	double top = std::numeric_limits<double>::max();
	double right = std::numeric_limits<double>::lowest();
	double bottom = std::numeric_limits<double>::lowest();
	constexpr auto extent = static_cast<double>(libmug::crop_extent);
	for (const libmug::point corner :
	     {libmug::point{0, 0}, libmug::point{extent, 0}, libmug::point{0, extent}, libmug::point{extent, extent}})
	{
		const libmug::point p = from_template->apply(corner);
		left = std::min(left, p.x);
		top = std::min(top, p.y);
		right = std::max(right, p.x);
		bottom = std::max(bottom, p.y);
	}
	if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right) || !std::isfinite(bottom)) return;

	const int x0 = std::clamp(static_cast<int>(std::floor(left)), 0, importance.width() - 1);
	const int x1 = std::clamp(static_cast<int>(std::ceil(right)), 0, importance.width() - 1);
	const int y0 = std::clamp(static_cast<int>(std::floor(top)), 0, importance.height() - 1);
	const int y1 = std::clamp(static_cast<int>(std::ceil(bottom)), 0, importance.height() - 1);
	constexpr double span = 255.0 - face_importance;
	for (int y = y0; y <= y1; ++y)
		for (int x = x0; x <= x1; ++x)
		{
			const libmug::point q = to_template->apply({static_cast<double>(x), static_cast<double>(y)});
			const double prior = prior_at(q.x, q.y);
			if (prior < 0.0) continue;
			const auto value = static_cast<uint8_t>(face_importance + std::lround(span * prior / 255.0));
			importance(x, y) = std::max(importance(x, y), value);
		}
}
