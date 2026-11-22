#include "landmarks.h"

#include <cmath>
#include <cstdlib>

namespace
{
	// One euro filter: cut-offs in hertz, and the speed term in hertz per pixel per second. A
	// still landmark is held to 1.5 Hz; one moving at 200 pixels a second is followed at 7.5 Hz.
	constexpr double filter_min_cutoff = 1.5;
	constexpr double filter_speed_gain = 0.03;
	constexpr double filter_speed_cutoff = 1.0;

	double smoothing(const double cutoff, const double seconds)
	{
		const double time_constant = 1.0 / (2.0 * 3.14159265358979323846 * cutoff);
		return 1.0 / (1.0 + time_constant / seconds);
	}
}

void landmark_filter::reset()
{
	state.clear();
	previous_seconds = 0.0;
}

face_geometry landmark_filter::apply(const face_geometry& faces, const double seconds)
{
	bool fresh = state.size() != faces.size();
	for (size_t face = 0; !fresh && face < faces.size(); ++face) fresh = state[face].size() != faces[face].size() * 2;

	face_geometry smoothed = faces;
	const double elapsed = seconds - previous_seconds;

	// Faces are paired by index, so a change in count, or a long gap, starts over from the
	// raw landmarks rather than dragging one face's history onto another.
	if (fresh || elapsed > 1.0 || elapsed < 0.0)
	{
		state.assign(faces.size(), {});
		for (size_t face = 0; face < faces.size(); ++face)
			for (const point_i point : faces[face])
			{
				state[face].push_back({static_cast<double>(point.x), 0.0});
				state[face].push_back({static_cast<double>(point.y), 0.0});
			}
		previous_seconds = seconds;
		return smoothed;
	}

	for (size_t face = 0; face < faces.size(); ++face)
		for (size_t point = 0; point < faces[face].size(); ++point)
			for (int axis_index = 0; axis_index < 2; ++axis_index)
			{
				axis& filtered = state[face][point * 2 + axis_index];
				if (elapsed > 0.0)
				{
					const double raw = axis_index == 0 ? faces[face][point].x : faces[face][point].y;
					const double speed = (raw - filtered.value) / elapsed;
					filtered.speed += smoothing(filter_speed_cutoff, elapsed) * (speed - filtered.speed);
					const double cutoff = filter_min_cutoff + filter_speed_gain * std::abs(filtered.speed);
					filtered.value += smoothing(cutoff, elapsed) * (raw - filtered.value);
				}
				int& out = axis_index == 0 ? smoothed[face][point].x : smoothed[face][point].y;
				out = static_cast<int>(std::lround(filtered.value));
			}
	previous_seconds = seconds;
	return smoothed;
}

bool compare_landmarks(const std::vector<point_i>& reference, const std::vector<point_i>& measured,
                       landmark_error& error)
{
	error = {};
	constexpr size_t points = 5;
	if (reference.size() != points || measured.size() != points) return false;
	const double ocular = std::hypot(static_cast<double>(reference[1].x - reference[0].x),
	                                 static_cast<double>(reference[1].y - reference[0].y));
	if (ocular < 1.0) return false;

	double overall = 0.0;
	double mouth = 0.0;
	for (size_t i = 0; i < points; ++i)
	{
		const double distance = std::hypot(static_cast<double>(measured[i].x - reference[i].x),
		                                   static_cast<double>(measured[i].y - reference[i].y));
		overall += distance;
		if (i >= 3) mouth += distance;
	}
	error.overall = overall / static_cast<double>(points) / ocular;
	error.mouth = mouth / 2.0 / ocular;
	return true;
}
