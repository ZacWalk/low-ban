#include "engine.h"

#include <algorithm>
#include <chrono>
#include <deque>

#include <windows.h>

#include "capture.h"
#include "draw.h"

namespace
{
	constexpr uint8_t overlay_colour = 0;
	constexpr auto wire_send_interval = std::chrono::milliseconds(200);
	constexpr auto traditional_rate_window = std::chrono::seconds(10);
	constexpr size_t traditional_delta_budget = 1200;
	constexpr auto neural_keyframe_interval = std::chrono::seconds(3);
	constexpr size_t neural_delta_budget = 1500;

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

	class payload_rate
	{
		using clock = std::chrono::steady_clock;
		std::deque<std::pair<clock::time_point, int64_t>> packets;
		clock::duration window;

	public:
		explicit payload_rate(const clock::duration duration) : window(duration) {}
		void reset() { packets.clear(); }

		void add(const clock::time_point now, const int64_t bits)
		{
			if (bits > 0) packets.emplace_back(now, bits);
		}

		double bits_per_second(const clock::time_point now)
		{
			while (!packets.empty() && now - packets.front().first > window)
				packets.pop_front();
			if (packets.empty()) return 0.0;
			int64_t total_bits = 0;
			for (const auto& packet : packets) total_bits += packet.second;
			const double seconds = std::max(0.2,
				std::chrono::duration<double>(now - packets.front().first).count() + 0.2);
			return total_bits / seconds;
		}
	};

	int overlap_area(const rect_i& a, const rect_i& b)
	{
		const int w = std::min(a.right, b.right) - std::max(a.left, b.left) + 1;
		const int h = std::min(a.bottom, b.bottom) - std::max(a.top, b.top) + 1;
		return w > 0 && h > 0 ? w * h : 0;
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

std::shared_ptr<const render_state> engine::latest() const
{
	std::lock_guard<std::mutex> guard(output_mutex);
	return output;
}

void engine::reset_training()
{
	reset_requested.store(true);
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
		std::string error;

		if (camera.open(width, height, error))
		{
			frame_rate rate;

			while (!stop_requested.load())
			{
				auto frame = std::make_shared<video_frame>();
				if (!camera.read(*frame))
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(5));
					continue;
				}

				source_fps_.store(rate.tick());

				std::lock_guard<std::mutex> guard(source_mutex);
				source_frame = std::move(frame);
			}
		}
		else
		{
			std::lock_guard<std::mutex> guard(status_mutex);
			status_text = "camera unavailable: " + error;
		}
	}

	if (com_ready) CoUninitialize();
}

void engine::analysis_loop()
{
	// The model is tens of megabytes, so it is loaded here rather than making the window
	// wait for it. Only this thread ever touches the predictor.
	std::string error;
	if (!landmarks.load(asset_path(L"shape_predictor_68_face_landmarks.dat"), error))
	{
		std::lock_guard<std::mutex> guard(status_mutex);
		status_text = "landmark model unavailable: " + error;
	}

	frame_rate rate;
	payload_rate traditional_rate(traditional_rate_window);
	payload_rate neural_rate(neural_keyframe_interval);
	std::shared_ptr<const video_frame> previous;
	std::vector<rect_i> tracked;
	std::vector<std::vector<point_i>> transmitted_traditional_parts;
	std::vector<std::vector<point_i>> transmitted_face_parts;
	image_u8 traditional_reconstruction;
	image_u8 neural_reconstruction;
	int last_updated_blocks = 0;
	bool last_traditional_was_keyframe = false;
	int last_updated_patches = 0;
	bool last_was_keyframe = false;
	auto next_traditional_send = std::chrono::steady_clock::time_point::min();
	auto next_neural_send = std::chrono::steady_clock::time_point::min();
	auto last_neural_keyframe = std::chrono::steady_clock::time_point::min();

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

		if (reset_requested.exchange(false))
		{
			codec.reset();
			traditional_stream.reset();
			traditional_landmarks.reset();
			traditional_rate.reset();
			neural_stream.reset();
			neural_landmarks.reset();
			neural_rate.reset();
			traditional_reconstruction = {};
			neural_reconstruction = {};
			tracked.clear();
			transmitted_traditional_parts.clear();
			transmitted_face_parts.clear();
			last_updated_blocks = 0;
			last_traditional_was_keyframe = false;
			last_updated_patches = 0;
			last_was_keyframe = false;
			next_traditional_send = std::chrono::steady_clock::time_point::min();
			next_neural_send = std::chrono::steady_clock::time_point::min();
			last_neural_keyframe = std::chrono::steady_clock::time_point::min();
		}

		auto state = std::make_shared<render_state>();
		state->width = frame->width();
		state->height = frame->height();
		yuv_to_bgra(*frame, state->source_bgra);

		image_u8 importance(frame->width(), frame->height());
		importance.fill(32);
		std::vector<std::vector<point_i>> face_parts;

		if (show_faces())
		{
			face_detection detected = detector.detect(*frame);
			std::vector<rect_i> boxes = std::move(detected.faces);
			state->peak_confidence = detected.peak_confidence;

			// Prefer a box refined from the previous frame's landmarks: the detector is
			// coarse, but the model tells us exactly where it wants the box to be.
			for (auto& box : boxes)
			{
				for (const auto& t : tracked)
				{
					if (overlap_area(box, t) * 2 > box.width() * box.height())
					{
						box = t;
						break;
					}
				}
			}

			tracked.clear();
			state->face_count = static_cast<int>(boxes.size());
			state->faces = boxes;

			if (show_landmarks() && landmarks.loaded())
			{
				for (const auto& box : boxes)
				{
					const std::vector<point_i> parts = landmarks.predict(frame->luma, box);
					if (parts.empty()) continue;
					add_face_importance(importance, box, parts);
					face_parts.push_back(parts);

					// Only trust the refinement if it stayed near the box it came from, so a bad
					// prediction cannot drag the tracker away frame after frame.
					const rect_i refined = landmarks.box_from_parts(parts);
					if (!refined.empty() && refined.width() * 2 > box.width() && refined.width() < box.width() * 2 &&
						overlap_area(refined, box) * 2 > box.width() * box.height())
					{
						tracked.push_back(refined);
					}
				}

			}
		}

		codec.train(frame->luma, 256, &importance);
		const auto now = std::chrono::steady_clock::now();
		if (traditional_reconstruction.empty() || now >= next_traditional_send)
		{
			const bool force_keyframe = traditional_reconstruction.empty();
			const transform_packet packet = traditional_stream.encode(traditional_codec, frame->luma,
				&importance, force_keyframe, force_keyframe ? 0 : traditional_delta_budget);
			const landmark_packet geometry = traditional_landmarks.encode(face_parts, force_keyframe);
			const bool image_decoded = packet.bytes.empty() || traditional_stream.decode(traditional_codec,
				packet, frame->width(), frame->height(), traditional_reconstruction);
			std::vector<std::vector<point_i>> decoded_parts = transmitted_traditional_parts;
			const bool geometry_decoded = geometry.bytes.empty()
				|| traditional_landmarks.decode(geometry, decoded_parts);
			if (image_decoded && geometry_decoded && (!packet.bytes.empty() || !geometry.bytes.empty()))
			{
				transmitted_traditional_parts = std::move(decoded_parts);
				state->traditional_bits = static_cast<int64_t>(packet.bytes.size() + geometry.bytes.size()) * 8;
				last_updated_blocks = packet.updated_blocks;
				last_traditional_was_keyframe = packet.keyframe || geometry.keyframe;
				traditional_rate.add(now, state->traditional_bits);
			}
			next_traditional_send = now + wire_send_interval;
		}
		state->traditional = traditional_reconstruction;
		state->traditional_kbit_per_second = traditional_rate.bits_per_second(now) / 1000.0;
		state->traditional_updated_blocks = last_updated_blocks;
		state->traditional_keyframe = last_traditional_was_keyframe;

		const bool should_send = neural_reconstruction.empty() || now >= next_neural_send;
		if (should_send)
		{
			const bool force_keyframe = neural_reconstruction.empty()
				|| now - last_neural_keyframe >= neural_keyframe_interval;
			const latent_packet packet = neural_stream.encode(codec, frame->luma, &importance,
				force_keyframe, neural_delta_budget);
			const landmark_packet geometry = neural_landmarks.encode(face_parts, force_keyframe);
			const bool image_decoded = packet.bytes.empty() || neural_stream.decode(codec, packet,
				frame->width(), frame->height(), neural_reconstruction);
			std::vector<std::vector<point_i>> decoded_parts = transmitted_face_parts;
			const bool geometry_decoded = geometry.bytes.empty() || neural_landmarks.decode(geometry, decoded_parts);
			if (image_decoded && geometry_decoded && (!packet.bytes.empty() || !geometry.bytes.empty()))
			{
				transmitted_face_parts = std::move(decoded_parts);
				state->neural_bits = static_cast<int64_t>(packet.bytes.size() + geometry.bytes.size()) * 8;
				last_updated_patches = packet.updated_patches;
				last_was_keyframe = packet.keyframe || geometry.keyframe;
				neural_rate.add(now, state->neural_bits);
				if (packet.keyframe) last_neural_keyframe = now;
			}
			next_neural_send = now + wire_send_interval;
		}
		state->neural = neural_reconstruction;
		state->neural_kbit_per_second = neural_rate.bits_per_second(now) / 1000.0;
		state->neural_updated_patches = last_updated_patches;
		state->neural_keyframe = last_was_keyframe;
		state->training_steps = codec.steps();
		state->traditional_quality = measure_quality(frame->luma, state->traditional, &importance);
		state->neural_quality = measure_quality(frame->luma, state->neural, &importance);

		if (show_overlay())
		{
			for (const auto& parts : transmitted_traditional_parts)
				draw_face(state->traditional, parts, overlay_colour);
			for (const auto& parts : transmitted_face_parts) draw_face(state->neural, parts, overlay_colour);
		}

		processed_fps_.store(rate.tick());

		{
			std::lock_guard<std::mutex> guard(output_mutex);
			output = std::move(state);
		}
		generation_.fetch_add(1);
	}
}
