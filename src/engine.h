// The processing pipeline.
//
// Three participants, each with one job:
//   capture thread  - pulls frames from the camera, and records them to a clip on request
//   analysis thread - finds faces and their landmarks, and runs both wires of the call session
//   UI thread       - only ever blits the most recent finished result
// The threads exchange whole immutable frames through `engine::latest`, so no partially
// updated state can ever reach the screen.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "face.h"
#include "image.h"
#include "session.h"

class engine
{
public:
	engine() = default;
	~engine();

	engine(const engine&) = delete;
	engine& operator=(const engine&) = delete;

	// Starts the capture and analysis threads. Failures (no camera) are reported through
	// `status` rather than preventing the window from running.
	void start(int width, int height);
	void stop();

	void set_show_faces(const bool v) { show_faces_.store(v); }
	bool show_faces() const { return show_faces_.load(); }

	// Records the camera to samples\clip-<time>.mp4 until switched off again.
	void set_recording(const bool v) { recording_.store(v); }
	bool recording() const { return recording_.load(); }

	std::shared_ptr<const render_state> latest() const;
	// Changes whenever there is something new to paint: a published frame or a new status.
	// The window repaints on a change, so a status reported while no frames arrive - the
	// camera unavailable - still reaches the screen.
	uint64_t generation() const { return generation_.load(); }

	double source_fps() const { return source_fps_.load(); }
	double processed_fps() const { return processed_fps_.load(); }

	std::string status() const;

private:
	void capture_loop(int width, int height);
	void analysis_loop();
	void set_status(std::string text);

	std::atomic<bool> show_faces_{true};
	std::atomic<bool> recording_{false};
	std::atomic<bool> stop_requested{false};

	std::atomic<double> source_fps_{0.0};
	std::atomic<double> processed_fps_{0.0};
	std::atomic<uint64_t> generation_{0};

	mutable std::mutex source_mutex;
	std::shared_ptr<const video_frame> source_frame;

	mutable std::mutex output_mutex;
	std::shared_ptr<const render_state> output;

	mutable std::mutex status_mutex;
	std::string status_text;

	// Used only by the analysis thread: the source's faces, then the receivers' to measure them.
	face_detector detector;

	std::thread capture_thread;
	std::thread analysis_thread;
};

// Resolves a file that ships next to the executable.
std::wstring asset_path(const wchar_t* name);
