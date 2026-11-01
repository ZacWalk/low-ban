#include "engine.h"

#include <algorithm>
#include <chrono>

#include <windows.h>

#include "capture.h"

namespace
{
	// Exponential moving average over frame intervals.
	class frame_rate
	{
		std::chrono::steady_clock::time_point previous;
		bool primed = false;
		double interval_ms = 0.0;

	public:
		double tick()
		{
			const auto now = std::chrono::steady_clock::now();
			if (!primed)
			{
				previous = now;
				primed = true;
				return 0.0;
			}

			const double delta = std::chrono::duration<double, std::milli>(now - previous).count();
			previous = now;
			if (delta <= 0.0) return interval_ms > 0.0 ? 1000.0 / interval_ms : 0.0;

			interval_ms = interval_ms == 0.0 ? delta : interval_ms * 0.9 + delta * 0.1;
			return 1000.0 / interval_ms;
		}
	};

	double seconds_since(const std::chrono::steady_clock::time_point start)
	{
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}

	// The status line only ever shows paths this program chose, which are plain ASCII.
	std::string narrow(const std::wstring& text)
	{
		std::string result;
		for (const wchar_t c : text) result.push_back(c < 128 ? static_cast<char>(c) : '?');
		return result;
	}

	// A new samples\clip-<local time>.mp4, beside the PNGs `evaluate` scores.
	std::wstring new_clip_path(std::wstring& name)
	{
		const std::wstring folder = asset_path(L"..\\samples");
		CreateDirectoryW(folder.c_str(), nullptr);

		SYSTEMTIME now = {};
		GetLocalTime(&now);
		wchar_t file[64] = {};
		swprintf_s(file, L"clip-%04u%02u%02u-%02u%02u%02u.mp4", now.wYear, now.wMonth, now.wDay, now.wHour,
			now.wMinute, now.wSecond);
		name = file;
		return folder + L"\\" + name;
	}
}

std::wstring asset_path(const wchar_t* name)
{
	wchar_t module[MAX_PATH] = {};
	const DWORD len = GetModuleFileNameW(nullptr, module, MAX_PATH);
	if (len == 0 || len >= MAX_PATH) return name;

	std::wstring path(module, len);
	const size_t slash = path.find_last_of(L'\\');
	if (slash == std::wstring::npos) return name;

	path.resize(slash + 1);
	path += name;
	return path;
}

engine::~engine()
{
	stop();
}

std::string engine::status() const
{
	std::lock_guard<std::mutex> guard(status_mutex);
	return status_text;
}

void engine::set_status(std::string text)
{
	{
		std::lock_guard<std::mutex> guard(status_mutex);
		status_text = std::move(text);
	}
	generation_.fetch_add(1);
}

std::shared_ptr<const render_state> engine::latest() const
{
	std::lock_guard<std::mutex> guard(output_mutex);
	return output;
}

void engine::start(const int width, const int height)
{
	analysis_thread = std::thread(&engine::analysis_loop, this);
	capture_thread = std::thread(&engine::capture_loop, this, width, height);
}

void engine::stop()
{
	stop_requested.store(true);
	if (capture_thread.joinable()) capture_thread.join();
	if (analysis_thread.joinable()) analysis_thread.join();
}

void engine::capture_loop(const int width, const int height)
{
	// Media Foundation's synchronous reader expects an initialised multi-threaded apartment
	// on the thread that drives it.
	const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	const bool com_ready = SUCCEEDED(com) || com == RPC_E_CHANGED_MODE;

	{
		webcam camera;
		clip_writer clip;
		std::wstring clip_name;
		double next_clip_frame = 0.0;
		std::string error;
		const auto started = std::chrono::steady_clock::now();

		const auto finish_clip = [&]
		{
			if (!clip.is_open()) return;
			const int frames = clip.frames();
			std::string clip_error;
			if (clip.close(clip_error))
				set_status("saved samples\\" + narrow(clip_name) + " (" + std::to_string(frames) + " frames)");
			else set_status("clip not saved: " + clip_error);
		};

		if (camera.open(width, height, error))
		{
			frame_rate rate;

			while (!stop_requested.load())
			{
				auto frame = std::make_shared<video_frame>();
				if (!camera.read(*frame, error))
				{
					if (!error.empty()) break;
					std::this_thread::sleep_for(std::chrono::milliseconds(5));
					continue;
				}

				source_fps_.store(rate.tick());

				const double seconds = seconds_since(started);
				if (recording_.load() && !clip.is_open())
				{
					std::string clip_error;
					if (clip.open(new_clip_path(clip_name), frame->width(), frame->height(), geometry_hz, clip_error))
					{
						next_clip_frame = seconds;
						set_status("recording samples\\" + narrow(clip_name) + " - File > Record clip to stop");
					}
					else
					{
						recording_.store(false);
						set_status("could not record a clip: " + clip_error);
					}
				}
				else if (!recording_.load())
				{
					finish_clip();
				}

				// Written on an even cadence, so the clip replays at exactly its frame rate.
				if (clip.is_open() && seconds + 0.008 >= next_clip_frame)
				{
					std::string clip_error;
					if (!clip.write(*frame, clip_error))
					{
						std::string ignored;
						clip.close(ignored);
						recording_.store(false);
						set_status("recording stopped: " + clip_error);
					}
					next_clip_frame += 1.0 / geometry_hz;
					if (next_clip_frame <= seconds) next_clip_frame = seconds + 1.0 / geometry_hz;
				}

				std::lock_guard<std::mutex> guard(source_mutex);
				source_frame = std::move(frame);
			}
		}

		// A clip that is never finished cannot be played back, so a camera that stops
		// mid-recording still leaves a usable file.
		finish_clip();
		recording_.store(false);

		if (!error.empty())
		{
			source_fps_.store(0.0);
			set_status("camera unavailable: " + error);
		}
	}

	if (com_ready) CoUninitialize();
}

void engine::analysis_loop()
{
	// The H.264 baseline's transforms are COM objects, created and used on this thread only.
	const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	const bool com_ready = SUCCEEDED(com) || com == RPC_E_CHANGED_MODE;

	{
		// The model is tens of megabytes, so it is loaded here rather than making the window
		// wait for it. Only this thread ever touches the predictor.
		std::string error;
		if (!landmarks.load(asset_path(L"shape_predictor_68_face_landmarks.dat"), error))
		{
			set_status("landmark model unavailable: " + error);
		}

		frame_rate rate;
		face_tracker tracker;
		call_session session;
		std::shared_ptr<const video_frame> previous;
		std::string baseline_status;
		const auto started = std::chrono::steady_clock::now();

		while (!stop_requested.load())
		{
			std::shared_ptr<const video_frame> frame;
			{
				std::lock_guard<std::mutex> guard(source_mutex);
				frame = source_frame;
			}

			if (!frame || frame == previous)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(2));
				continue;
			}
			previous = frame;

			const double seconds = seconds_since(started);
			face_analysis faces;
			if (show_faces()) faces = tracker.analyse(detector, show_landmarks() ? &landmarks : nullptr, *frame);
			else tracker.reset();

			auto state = std::make_shared<render_state>();
			yuv_to_bgra(*frame, state->source_bgra);
			session_options options;
			options.landmark_overlay = show_overlay();
			session.process(*frame, seconds, faces, landmarks.loaded() ? &landmarks : nullptr, options, *state);

			// A missing baseline is worth one status line, not one per frame.
			if (state->baseline_status != baseline_status)
			{
				baseline_status = state->baseline_status;
				if (!baseline_status.empty()) set_status(baseline_status);
			}

			processed_fps_.store(rate.tick());

			{
				std::lock_guard<std::mutex> guard(output_mutex);
				output = std::move(state);
			}
			generation_.fetch_add(1);
		}
	}

	if (com_ready) CoUninitialize();
}
