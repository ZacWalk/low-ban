// The sender-side jitter filter for landmarks, and landmark error.
//
// Landmarks are the five points libmug's detector finds on each face - the eye centres, the tip of the nose and the
// corners of the mouth - in that order. They travel inside low-ban's packets (codec.h), so the receiver can move the
// face it holds when the face moves, and the same five points found again in a received picture measure whether the
// face survived the wire.
#pragma once

#include <vector>

#include "image.h"

// Suppresses landmark jitter before transmission: the "one euro" filter of Casiez, Roussel and
// Vogel (2012), whose cut-off rises with speed. A still face otherwise flickers by a pixel every
// frame, which costs landmark bits and makes the receiver's warped face shimmer, while a fast
// cut-off during real motion keeps the mouth from lagging.
class landmark_filter
{
public:
	void reset();
	face_geometry apply(const face_geometry& faces, double seconds);

private:
	struct axis
	{
		double value = 0.0;
		double speed = 0.0;
	};

	std::vector<std::vector<axis>> state; // per face, x and y of every point
	double previous_seconds = 0.0;
};

// Normalised mean error: the average distance between corresponding points, as a fraction of
// the reference's distance between the eye centres. Computed on the landmarks found in a received
// picture against those found in the source, it measures whether the face survived the wire.
struct landmark_error
{
	double overall = 0.0;
	double mouth = 0.0; // the two mouth corners alone, which a smile or speech moves most
};

bool compare_landmarks(const std::vector<point_i>& reference, const std::vector<point_i>& measured,
                       landmark_error& error);
