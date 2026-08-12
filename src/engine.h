// The processing pipeline.
//
// Three participants, each with one job:
//   capture thread  - pulls frames from the camera
//   analysis thread - detects faces, landmarks them, and encodes the low bandwidth pane
//   UI thread       - only ever blits the most recent finished result
// The threads exchange whole immutable frames through `engine::latest`, so no partially
// updated state can ever reach the screen.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "codec.h"
#include "face.h"
#include "image.h"
#include "landmarks.h"

struct render_state
{
	std::vector<uint8_t> source_bgra; // panel 1, top-down BGRA
	image_u8 traditional; // panel 2, prioritised DCT reconstruction
	image_u8 neural; // panel 3, prioritised autoencoder reconstruction
	std::vector<rect_i> faces; // boxes the landmarker was given, in frame coordinates
	int width = 0;
	int height = 0;
	int face_count = 0;
	int training_steps = 0;
	float peak_confidence = 0.0f; // best detector score in the frame, before thresholding
	int64_t traditional_bits = 0;
	double traditional_kbit_per_second = 0.0;
	int traditional_updated_blocks = 0;
	bool traditional_keyframe = false;
	int64_t neural_bits = 0; // bytes emitted for this packet, including landmark accounting
	double neural_kbit_per_second = 0.0;
	int neural_updated_patches = 0;
	bool neural_keyframe = false;
	reconstruction_quality traditional_quality;
	reconstruction_quality neural_quality;
};

class engine
{
public:
	engine() = default;
	~engine();

	engine(const engine&) = delete;
	engine& operator=(const engine&) = delete;

	// Starts the capture and analysis threads. Failures (no camera, no landmark model) are
	// reported through `status` rather than preventing the window from running.
	void start(int width, int height);
	void stop();

	void set_show_faces(const bool v) { show_faces_.store(v); }
	bool show_faces() const { return show_faces_.load(); }

	void set_show_landmarks(const bool v) { show_landmarks_.store(v); }
	bool show_landmarks() const { return show_landmarks_.load(); }

	// Whether the received geometry is drawn over panels 2 and 3. Landmarking still runs and
	// is still charged to both wires when this is off; it is a reference overlay, not a codec
	// input, so hiding it shows the reconstructions as a viewer would actually receive them.
	void set_show_overlay(const bool v) { show_overlay_.store(v); }
	bool show_overlay() const { return show_overlay_.load(); }

	void reset_training();

	std::shared_ptr<const render_state> latest() const;
	uint64_t generation() const { return generation_.load(); }

	double source_fps() const { return source_fps_.load(); }
	double processed_fps() const { return processed_fps_.load(); }

	std::string status() const;

private:
	void capture_loop(int width, int height);
	void analysis_loop();

	std::atomic<bool> show_faces_{true};
	std::atomic<bool> show_landmarks_{true};
	std::atomic<bool> show_overlay_{true};
	std::atomic<bool> stop_requested{false};
	std::atomic<bool> reset_requested{false};

	std::atomic<double> source_fps_{0.0};
	std::atomic<double> processed_fps_{0.0};
	std::atomic<uint64_t> generation_{0};

	mutable std::mutex source_mutex;
	std::shared_ptr<const video_frame> source_frame;

	mutable std::mutex output_mutex;
	std::shared_ptr<const render_state> output;

	mutable std::mutex status_mutex;
	std::string status_text;

	face_detector detector;
	shape_predictor landmarks;
	transform_codec traditional_codec;
	temporal_transform_stream traditional_stream;
	temporal_landmark_stream traditional_landmarks;
	patch_autoencoder codec;
	temporal_patch_stream neural_stream;
	temporal_landmark_stream neural_landmarks;

	std::thread capture_thread;
	std::thread analysis_thread;
};

// Resolves a file that ships next to the executable.
std::wstring asset_path(const wchar_t* name);
