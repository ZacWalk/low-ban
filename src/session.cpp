#include "session.h"

#include <algorithm>
#include <cmath>

namespace
{
	constexpr double rate_window_seconds = 10.0;

	// Frames rarely land exactly on a tick; this much slack keeps a 30 fps camera on an even
	// 15 Hz instead of slipping to every third frame.
	constexpr double clock_slack_seconds = 0.008;

	constexpr double tick_seconds = 1.0 / packet_hz;
	constexpr double bucket_bytes_per_tick = call_kbit_per_second * 1000.0 / 8.0 / packet_hz;

	int floor_div(const int value, const int divisor)
	{
		const int quotient = value / divisor;
		return quotient * divisor > value ? quotient - 1 : quotient;
	}

	point_f eye_centre(const std::vector<point_i>& points)
	{
		return {(points[eye_left].x + points[eye_right].x) / 2.0f, (points[eye_left].y + points[eye_right].y) / 2.0f};
	}
}

double audio_link_kbit_per_second()
{
	return speech_kbit_per_second + packet_header_bytes * 8.0 / speech_packet_seconds / 1000.0;
}

rect_i coded_box(const rect_i& box)
{
	return {floor_div(box.left, coded_scale), floor_div(box.top, coded_scale), floor_div(box.right, coded_scale),
	        floor_div(box.bottom, coded_scale)};
}

face_geometry coded_geometry(const face_geometry& faces)
{
	face_geometry result = faces;
	for (auto& face : result)
		for (auto& point : face)
		{
			point.x = static_cast<int>(std::lround(point.x / static_cast<double>(coded_scale)));
			point.y = static_cast<int>(std::lround(point.y / static_cast<double>(coded_scale)));
		}
	return result;
}

face_analysis analyse_faces(face_detector& detector, const image_u8& luma)
{
	face_analysis result;
	for (detected_face& face : detector.detect(luma))
	{
		result.confidence = std::max(result.confidence, face.confidence);
		result.boxes.push_back(face.box);
		result.points.push_back(std::move(face.points));
	}
	return result;
}

bool measure_expression(face_detector& detector, const image_u8& picture, const face_analysis& faces,
                        landmark_error& error)
{
	if (picture.empty() || faces.points.empty()) return false;
	const std::vector<detected_face> found = detector.detect(picture);
	landmark_error total;
	int measured = 0;
	for (const auto& source : faces.points)
	{
		if (source.size() != face_point_count) continue;
		const double spacing = std::hypot(static_cast<double>(source[eye_right].x - source[eye_left].x),
		                                  static_cast<double>(source[eye_right].y - source[eye_left].y));
		if (spacing < min_scored_eye_spacing) continue;

		// The received face whose eyes are nearest the source's, within half their spacing.
		const point_f centre = eye_centre(source);
		const detected_face* nearest = nullptr;
		double nearest_distance = spacing / 2.0;
		for (const detected_face& face : found)
		{
			if (face.points.size() != face_point_count) continue;
			const point_f other = eye_centre(face.points);
			const double distance = std::hypot(other.x - centre.x, other.y - centre.y);
			if (distance <= nearest_distance)
			{
				nearest_distance = distance;
				nearest = &face;
			}
		}

		landmark_error face_error{1.0, 1.0};
		if (nearest != nullptr) compare_landmarks(source, nearest->points, face_error);
		total.overall += face_error.overall;
		total.mouth += face_error.mouth;
		++measured;
	}
	if (measured == 0) return false;
	error.overall = total.overall / measured;
	error.mouth = total.mouth / measured;
	return true;
}

void call_session::rolling_rate::add(const double seconds, const int64_t bits)
{
	if (bits > 0) entries.emplace_back(seconds, bits);
}

double call_session::rolling_rate::bits_per_second(const double seconds, const double tick)
{
	while (!entries.empty() && seconds - entries.front().first > rate_window_seconds) entries.pop_front();
	if (entries.empty()) return 0.0;
	int64_t total = 0;
	for (const auto& entry : entries) total += entry.second;
	return static_cast<double>(total) / std::max(tick, seconds - entries.front().first + tick);
}

bool call_session::due(clock& schedule, const double seconds, const double interval)
{
	if (!schedule.started)
	{
		schedule.started = true;
		schedule.next = seconds + interval;
		return true;
	}
	if (seconds + clock_slack_seconds < schedule.next) return false;

	// After a stall, start again from now rather than bursting through the missed ticks.
	schedule.next += interval;
	if (schedule.next <= seconds) schedule.next = seconds + interval;
	return true;
}

void call_session::reset()
{
	tick_clock = {};
	smoother.reset();

	lowban.reset();
	lowban_started = false;
	lowban_tokens = 0.0;
	lowban_quality = startup_quality;
	lowban_shown = {};
	lowban_updated_blocks = 0;
	lowban_keyframe = false;
	lowban_warped = false;
	lowban_rate.reset();
	lowban_geometry_rate.reset();
	lowban_header_rate.reset();

	baseline.close();
	baseline_tried = false;
	baseline_status.clear();
	baseline_decoded = {};
	baseline_shown = {};
	baseline_rate.reset();
	baseline_header_rate.reset();

	sums = {};
}

void call_session::send_lowban(const image_u8& coded, const image_u8& coded_importance, const face_geometry& faces,
                               const double seconds)
{
	// The startup keyframe is the whole scene, whatever it costs. After it, every tick adds
	// the call rate's share to a bucket that can save up one capped packet, and a packet
	// spends what the bucket holds: a still scene sends little or nothing and banks the rest
	// for the next movement.
	const bool startup = !lowban_started;
	if (!startup)
		lowban_tokens = std::min(lowban_tokens + bucket_bytes_per_tick, static_cast<double>(packet_cap_bytes));
	const size_t budget = startup ? 0 : std::max<size_t>(1, static_cast<size_t>(std::max(0.0, lowban_tokens)));
	const lowban_packet packet = lowban.encode(coded, &coded_importance, faces, startup, budget, lowban_quality);

	// The quality follows the bucket a step a tick: coarser while macroblocks worth sending are
	// left for want of bytes and the bucket is all but dry, finer while nothing is left and it fills.
	const auto spent = static_cast<double>(packet.bytes.size());
	if (!startup)
	{
		if (packet.limited && lowban_tokens - spent < bucket_bytes_per_tick)
			lowban_quality = std::min(coarsest_quality, lowban_quality + 1);
		else if (!packet.limited && lowban_tokens - spent > packet_cap_bytes / 2.0)
			lowban_quality = std::max(finest_quality, lowban_quality - 1);
	}
	if (packet.bytes.empty()) return;

	image_u8 decoded;
	if (!lowban.decode(packet, decoded)) return;
	lowban_started = true;
	lowban_updated_blocks = packet.updated_blocks;
	lowban_keyframe = packet.keyframe;
	lowban_warped = packet.warped;

	image_u8 coded_shown;
	if (lowban.show(coded_shown))
	{
		lowban_shown.resize(frame_width, frame_height);
		resize_bicubic(coded_shown, lowban_shown);
	}

	const auto size = static_cast<int64_t>(packet.bytes.size());
	lowban_rate.add(seconds, size * 8);
	lowban_geometry_rate.add(seconds, static_cast<int64_t>(packet.geometry_bytes) * 8);
	lowban_header_rate.add(seconds, packet_header_bytes * 8);
	if (startup)
	{
		sums.lowban_startup_bytes = size;
		return;
	}
	lowban_tokens -= static_cast<double>(size);
	sums.lowban_bytes += size;
	sums.lowban_geometry_bytes += static_cast<int64_t>(packet.geometry_bytes);
	sums.lowban_largest_packet = std::max(sums.lowban_largest_packet, packet.bytes.size());
	++sums.lowban_packets;
	if (packet.warped) ++sums.lowban_warped_packets;
}

void call_session::send_baseline(const image_u8& coded, const face_analysis& faces, const double seconds,
                                 const session_options& options)
{
	if (!baseline_tried)
	{
		baseline_tried = true;
		h264_settings settings;
		settings.width = coded.width();
		settings.height = coded.height();
		settings.frames_per_second = packet_hz;
		settings.kbit_per_second = call_kbit_per_second;
		settings.face_offset = options.baseline_face_offset ? baseline_face_offset : 0;
		std::string error;
		if (!baseline.open(settings, error)) baseline_status = "H.264 baseline unavailable: " + error;
	}
	if (!baseline.is_open()) return;

	// The same faces the importance map is built from, at the coded size.
	std::vector<rect_i> boxes;
	boxes.reserve(faces.boxes.size());
	for (const rect_i& box : faces.boxes) boxes.push_back(coded_box(box));

	std::vector<uint8_t> bytes;
	std::string error;
	if (!baseline.encode(coded, boxes, bytes, error) || !baseline.decode(bytes, baseline_decoded, error))
	{
		baseline_status = "H.264 baseline stopped: " + error;
		baseline.close();
		return;
	}
	if (bytes.empty()) return;

	const auto size = static_cast<int64_t>(bytes.size());
	baseline_rate.add(seconds, size * 8);
	baseline_header_rate.add(seconds, packet_header_bytes * 8);
	if (sums.baseline_frames == 0) sums.baseline_first_frame_bytes = size;
	else sums.baseline_bytes += size;
	++sums.baseline_frames;

	if (!baseline_decoded.empty())
	{
		baseline_shown.resize(frame_width, frame_height);
		resize_bicubic(baseline_decoded, baseline_shown);
	}
}

void call_session::process(const video_frame& frame, const double seconds, const face_analysis& faces,
                           face_detector* measure, const session_options& options, render_state& state)
{
	const int width = frame.width();
	const int height = frame.height();
	if (width != frame_width || height != frame_height || options.baseline_face_offset != face_offset_option)
	{
		reset();
		frame_width = width;
		frame_height = height;
		face_offset_option = options.baseline_face_offset;
	}

	// The sender's view: face importance from the landmarks, then everything at the coded size.
	image_u8 importance(width, height);
	importance.fill(background_importance);
	for (size_t i = 0; i < faces.boxes.size() && i < faces.points.size(); ++i)
		add_face_importance(importance, faces.boxes[i], faces.points[i]);
	image_u8 coded(width / coded_scale, height / coded_scale);
	resize_bilinear(frame.luma, coded);
	image_u8 coded_importance(coded.width(), coded.height());
	resize_bilinear(importance, coded_importance);
	const face_geometry smoothed = smoother.apply(faces.points, seconds);

	if (due(tick_clock, seconds, tick_seconds))
	{
		++sums.ticks;
		send_lowban(coded, coded_importance, coded_geometry(smoothed), seconds);
		send_baseline(coded, faces, seconds, options);
	}

	state.width = width;
	state.height = height;
	state.faces = faces.boxes;
	state.face_count = static_cast<int>(faces.boxes.size());
	state.confidence = faces.confidence;

	const auto receive = [&](receiver_view& view, const image_u8& shown)
	{
		view.picture = shown;
		view.quality = measure_quality(frame.luma, shown, &importance);
		view.expression_measured = measure != nullptr && measure_expression(*measure, shown, faces, view.expression);
	};
	receive(state.lowban, lowban_shown);
	receive(state.baseline, baseline_shown);

	const double audio = audio_link_kbit_per_second();
	state.lowban.payload_kbit_per_second = lowban_rate.bits_per_second(seconds, tick_seconds) / 1000.0;
	state.lowban.link_kbit_per_second = state.lowban.payload_kbit_per_second +
	                                    lowban_header_rate.bits_per_second(seconds, tick_seconds) / 1000.0 + audio;
	state.lowban_geometry_kbit_per_second = lowban_geometry_rate.bits_per_second(seconds, tick_seconds) / 1000.0;
	state.lowban_updated_blocks = lowban_updated_blocks;
	state.lowban_quality = lowban_quality;
	state.lowban_keyframe = lowban_keyframe;
	state.lowban_warped = lowban_warped;

	state.baseline.payload_kbit_per_second = baseline_rate.bits_per_second(seconds, tick_seconds) / 1000.0;
	state.baseline.link_kbit_per_second = state.baseline.payload_kbit_per_second +
	                                      baseline_header_rate.bits_per_second(seconds, tick_seconds) / 1000.0 + audio;
	state.baseline_face_offset = baseline.face_offset_active();
	state.baseline_status = baseline_status;
}
