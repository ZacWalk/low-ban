// Face detection.
//
// A CNN face detector: a small depthwise-separable backbone with a three level feature
// pyramid and per-cell objectness, classification and box regression heads.
//
// The network design and the trained weights in face-model.cpp come from libfacedetection by
// Shiqi Yu (https://github.com/ShiqiYu/libfacedetection), used under its 3-clause BSD
// licence; the full notice is reproduced in face.cpp and face-model.cpp. The inference code
// here was rewritten for this application: no SIMD or OpenMP variants, no aligned blob
// allocator, no C result-buffer API, and the five point keypoint branch is not evaluated
// because landmarks come from the 68 point model instead.
#pragma once

#include <vector>

#include "image.h"

// Descriptor for one convolution in the generated weight table.
struct face_conv_layer
{
	int channels;
	int num_filters;
	bool is_depthwise;
	bool is_pointwise;
	bool with_relu;
	float* weights;
	float* biases;
};

constexpr int face_model_layer_count = 53;
extern face_conv_layer face_model_layers[face_model_layer_count];

struct face_detection
{
	std::vector<rect_i> faces;

	// Best score the network produced anywhere in the frame, before thresholding. A high
	// peak with no faces means the threshold or the suppression rejected them; a peak near
	// zero means the network never fired.
	float peak_confidence = 0.0f;
};

class face_detector
{
public:
	// Boxes follow the landmark model's convention: square, brow to chin.
	face_detection detect(const video_frame& frame) const;
};
