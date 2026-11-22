// Faces: where they are, their five landmarks, and how much each part of a face matters.
//
// Detection is libmug's (https://github.com/ZacWalk/libmug): YuNet - libfacedetection's network and weights, under
// its 3-clause BSD licence - run on libmug's SIMD kernels over the frame's luma, read where it lies at video range.
// Each face comes with a box and five landmarks, the same five every frame, so they are what the landmark wire
// carries and what the receiver's warp moves.
//
// Importance is where the transform codec spends its bits. Five points cannot say whether a mouth is open, a brow
// raised or an eye shut, so the receiver's warp cannot reproduce those; they have to come through the picture. The
// expression prior is a map, learned offline by tools/learn_expression_prior.py, of how far each part of a face
// moves in ways the five points do not predict, in the frame libmug's template fit aligns every face to.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "image.h"

namespace libmug
{
	class face_detector;
}

// The five landmarks, in this order on every face. Left and right are the picture's.
enum face_point : size_t
{
	eye_left,
	eye_right,
	nose_tip,
	mouth_left,
	mouth_right,
	face_point_count
};

struct detected_face
{
	rect_i box;                  // the network's box, hairline to chin
	std::vector<point_i> points; // face_point_count landmarks
	int confidence = 0;          // 0 to 100
};

class face_detector
{
public:
	face_detector();
	~face_detector();

	face_detector(const face_detector&) = delete;
	face_detector& operator=(const face_detector&) = delete;

	// The faces in a frame's luma at video range, left to right then top to bottom: at most eight, the most
	// confident. The detector reuses its buffers, so it serves one thread at a time.
	std::vector<detected_face> detect(const image_u8& luma);

private:
	std::unique_ptr<libmug::face_detector> network;
};

// The expression prior: a square grid of cells over the 112-pixel square that libmug::template_fit aligns faces to,
// 0 where a face's appearance is fully predicted by its five landmarks and 255 where it varies most beyond them.
// Generated into expression-prior.cpp.
constexpr int expression_prior_cells = 28;
extern const uint8_t expression_prior[expression_prior_cells * expression_prior_cells];

// The importance a face adds to the map, never lowering what is there: the face's box at the face's own weight,
// raised towards the most important by the expression prior where the five landmarks place it.
void add_face_importance(image_u8& importance, const rect_i& box, const std::vector<point_i>& points);

// The background's and a face box's importance, on the 0 to 255 scale the codec reads; the prior raises a face from
// its box's weight towards 255.
constexpr uint8_t background_importance = 32;
constexpr uint8_t face_importance = 96;
