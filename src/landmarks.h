// 68-point face landmarking.
//
// This is a self-contained reimplementation of the "one millisecond face alignment with an
// ensemble of regression trees" predictor (Kazemi & Sullivan, 2014) plus a reader for the
// model file that the reference implementation publishes
// (shape_predictor_68_face_landmarks.dat). Only the inference path is implemented; there is
// no training code and no third party dependency.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "image.h"

class shape_predictor
{
public:
	struct split
	{
		uint32_t idx1 = 0;
		uint32_t idx2 = 0;
		float thresh = 0.0f;
	};

	// A fully balanced binary tree: leaf_count == splits.size() + 1.
	struct tree
	{
		std::vector<split> splits;
		std::vector<float> leaves; // leaf_count blocks of shape_size floats
	};

	bool load(const std::wstring& path, std::string& error);
	bool loaded() const { return !forests.empty(); }

	size_t num_parts() const { return initial_shape.size() / 2; }
	size_t num_cascades() const { return forests.size(); }

	// The model's mean shape, in unit box coordinates.
	point_f mean_part(size_t index) const;

	// rect is in image pixels and follows the training convention: a square box that spans
	// the face from brow to chin.
	std::vector<point_i> predict(const image_u8& img, const rect_i& rect) const;

	// Inverts the model's own box-to-image mapping to recover the detection box that a
	// predicted shape implies. Feeding this back as the next frame's box lets a coarse
	// detector converge onto the convention the model was trained with.
	rect_i box_from_parts(const std::vector<point_i>& parts) const;

private:
	std::vector<float> initial_shape; // interleaved x,y in [0,1] box space
	std::vector<std::vector<tree>> forests;
	std::vector<std::vector<uint32_t>> anchor_idx;
	std::vector<std::vector<point_f>> deltas;

	void clear();
};

struct landmark_packet
{
	std::vector<uint8_t> bytes;
	bool keyframe = false;
};

// Serializes landmark geometry independently of image pixels, on a two pixel grid. Keyframes
// store absolute coordinates. Delta packets describe each face as one rigid shift plus small
// per-point residuals, listing only the points whose residual is not zero when that is
// shorter, so a still or rigidly moving face costs a few bytes rather than one per point.
class temporal_landmark_stream
{
public:
	void reset();
	landmark_packet encode(const face_geometry& faces, bool force_keyframe);
	bool decode(const landmark_packet& packet, face_geometry& faces);

	// What the decoder holds once every packet from `encode` has arrived, in the same
	// coordinates `decode` returns. The encoder side of anything that predicts from the
	// geometry has to use this, not the landmarks it was given.
	face_geometry transmitted() const;

private:
	face_geometry encoder_points;
	face_geometry decoder_points;
};

// Suppresses landmark jitter before transmission: the "one euro" filter of Casiez, Roussel and
// Vogel (2012), whose cut-off rises with speed. A still face otherwise flickers by a grid step
// every frame, which costs geometry bytes and makes the receiver's warped face shimmer, while
// a fast cut-off during real motion keeps lips and brows from lagging.
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
// the reference's outer eye corner distance. Computed on landmarks found in a received picture
// against those found in the source, it measures whether expression survived the wire.
struct landmark_error
{
	double overall = 0.0;
	double mouth = 0.0; // the 20 lip points alone, where speech and expression move most
};

bool compare_landmarks(const std::vector<point_i>& reference, const std::vector<point_i>& measured,
	landmark_error& error);
