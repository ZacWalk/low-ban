#include "session.h"

#include <algorithm>

#include "draw.h"
#include "warp.h"

namespace
{
	constexpr double rate_window_seconds = 10.0;

	// Frames rarely land exactly on a tick; this much slack keeps a 30 fps camera on an even
	// 15 Hz instead of slipping to every third frame.
	constexpr double clock_slack_seconds = 0.008;

	constexpr uint8_t overlay_colour = 0;

	int overlap_area(const rect_i& a, const rect_i& b)
	{
		const int w = std::min(a.right, b.right) - std::max(a.left, b.left) + 1;
		const int h = std::min(a.bottom, b.bottom) - std::max(a.top, b.top) + 1;
		return w > 0 && h > 0 ? w * h : 0;
	}

	int floor_div(const int value, const int divisor)
	{
		const int quotient = value / divisor;
		return quotient * divisor > value ? quotient - 1 : quotient;
	}

	// Landmarks found in a received picture, from the same boxes the source was landmarked
	// with, against the source's own landmarks. Averaged over the faces.
	bool measure_expression(const shape_predictor& predictor, const image_u8& picture, const face_analysis& faces,
		landmark_error& error)
	{
		if (picture.empty()) return false;
		landmark_error total;
		int measured = 0;
		for (size_t i = 0; i < faces.landmarks.size(); ++i)
		{
			landmark_error face_error;
			if (!compare_landmarks(faces.landmarks[i], predictor.predict(picture, faces.landmarked[i]), face_error))
				continue;
			total.overall += face_error.overall;
			total.mouth += face_error.mouth;
			++measured;
		}
		if (measured == 0) return false;
		error.overall = total.overall / measured;
		error.mouth = total.mouth / measured;
		return true;
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

void face_tracker::reset()
{
	tracked.clear();
}

face_analysis face_tracker::analyse(const face_detector& detector, const shape_predictor* predictor,
	const video_frame& frame)
{
	face_analysis result;
	face_detection detected = detector.detect(frame);
	result.peak_confidence = detected.peak_confidence;
	result.boxes = std::move(detected.faces);

	// Prefer a box refined from the previous frame's landmarks: the detector is coarse, but
	// the model tells us exactly where it wants the box to be.
	for (auto& box : result.boxes)
		for (const auto& t : tracked)
			if (overlap_area(box, t) * 2 > box.width() * box.height())
			{
				box = t;
				break;
			}
	tracked.clear();

	if (predictor == nullptr || !predictor->loaded()) return result;
	for (const auto& box : result.boxes)
	{
		std::vector<point_i> parts = predictor->predict(frame.luma, box);
		if (parts.size() != 68) continue;

		// Only trust the refinement if it stayed near the box it came from, so a bad
		// prediction cannot drag the tracker away frame after frame.
		const rect_i refined = predictor->box_from_parts(parts);
		if (!refined.empty() && refined.width() * 2 > box.width() && refined.width() < box.width() * 2
			&& overlap_area(refined, box) * 2 > box.width() * box.height())
			tracked.push_back(refined);

		result.landmarked.push_back(box);
		result.landmarks.push_back(std::move(parts));
	}
	return result;
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
	geometry_clock = {};
	picture_clock = {};
	smoother.reset();

	geometry_wire.reset();
	picture_wire.reset();
	received_geometry.clear();
	picture_started = false;
	animation_stale = false;
	lowban_shown = {};
	lowban_updated_blocks = 0;
	lowban_keyframe = false;
	lowban_warped = false;
	lowban_picture_rate.reset();
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

void call_session::send_geometry(const face_geometry& smoothed, const double seconds)
{
	// The first packet is a keyframe; after that the stream sends one itself whenever the
	// number of faces changes.
	const landmark_packet packet = geometry_wire.encode(smoothed, sums.lowban_geometry_packets == 0);
	if (packet.bytes.empty()) return;

	face_geometry decoded = received_geometry;
	if (!geometry_wire.decode(packet, decoded)) return;
	received_geometry = std::move(decoded);
	animation_stale = true;

	lowban_geometry_rate.add(seconds, static_cast<int64_t>(packet.bytes.size()) * 8);
	lowban_header_rate.add(seconds, packet_header_bytes * 8);
	sums.lowban_geometry_bytes += static_cast<int64_t>(packet.bytes.size());
	++sums.lowban_geometry_packets;
}

void call_session::send_picture(const image_u8& coded, const image_u8& coded_importance, const double seconds)
{
	// The encoder has to predict from exactly the geometry the receiver holds: what the
	// geometry stream has sent, not what the tracker found.
	const bool startup = !picture_started;
	const transform_packet packet = picture_wire.encode(codec, coded, &coded_importance, startup,
		startup ? 0 : picture_packet_budget, scale_geometry(geometry_wire.transmitted(), coded_scale));
	if (packet.bytes.empty()) return;

	image_u8 decoded;
	if (!picture_wire.decode(codec, packet, coded.width(), coded.height(), decoded,
		scale_geometry(received_geometry, coded_scale))) return;
	picture_started = true;
	animation_stale = true;
	lowban_updated_blocks = packet.updated_blocks;
	lowban_keyframe = packet.keyframe;
	lowban_warped = packet.warped;

	const auto size = static_cast<int64_t>(packet.bytes.size());
	lowban_picture_rate.add(seconds, size * 8);
	lowban_header_rate.add(seconds, packet_header_bytes * 8);
	if (packet.keyframe) sums.lowban_keyframe_bytes += size;
	else
	{
		sums.lowban_picture_bytes += size;
		sums.lowban_largest_picture_delta = std::max(sums.lowban_largest_picture_delta, packet.bytes.size());
	}
	++sums.lowban_picture_packets;
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
		settings.frames_per_second = geometry_hz;
		settings.kbit_per_second = baseline_kbit_per_second;
		settings.face_offset = options.baseline_face_offset ? baseline_face_offset : 0;
		std::string error;
		if (!baseline.open(settings, error)) baseline_status = "H.264 baseline unavailable: " + error;
	}
	if (!baseline.is_open()) return;

	// The same faces the importance map is built from, at the coded size.
	std::vector<rect_i> boxes;
	for (const rect_i& box : faces.landmarked) boxes.push_back(coded_box(box));

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
		resize_bilinear(baseline_decoded, baseline_shown);
	}
}

void call_session::process(const video_frame& frame, const double seconds, const face_analysis& faces,
	const shape_predictor* predictor, const session_options& options, render_state& state)
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
	importance.fill(32);
	for (size_t i = 0; i < faces.landmarks.size(); ++i)
		add_face_importance(importance, faces.landmarked[i], faces.landmarks[i]);
	image_u8 coded(width / coded_scale, height / coded_scale);
	resize_bilinear(frame.luma, coded);
	image_u8 coded_importance(coded.width(), coded.height());
	resize_bilinear(importance, coded_importance);
	const face_geometry smoothed = smoother.apply(faces.landmarks, seconds);

	// Geometry first, so a picture packet on the same tick predicts from the newest landmarks.
	if (due(geometry_clock, seconds, 1.0 / geometry_hz))
	{
		send_geometry(smoothed, seconds);
		send_baseline(coded, faces, seconds, options);
	}
	if (due(picture_clock, seconds, picture_interval_seconds)) send_picture(coded, coded_importance, seconds);

	// The low-ban receiver shows its last picture with the face warped onto the newest
	// geometry, so expression moves at the geometry rate rather than the picture rate.
	if (animation_stale)
	{
		image_u8 coded_shown;
		if (picture_wire.animate(scale_geometry(received_geometry, coded_scale), coded_shown))
		{
			lowban_shown.resize(width, height);
			resize_bilinear(coded_shown, lowban_shown);
		}
		animation_stale = false;
	}

	state.width = width;
	state.height = height;
	state.faces = faces.boxes;
	state.face_count = static_cast<int>(faces.boxes.size());
	state.peak_confidence = faces.peak_confidence;

	const bool can_measure_expression = predictor != nullptr && predictor->loaded();
	const auto receive = [&](receiver_view& view, const image_u8& shown)
	{
		view.picture = shown;
		view.quality = measure_quality(frame.luma, shown, &importance);
		view.expression_measured = can_measure_expression
			&& measure_expression(*predictor, shown, faces, view.expression);
	};
	receive(state.lowban, lowban_shown);
	receive(state.baseline, baseline_shown);

	const double geometry_tick = 1.0 / geometry_hz;
	const double audio = audio_link_kbit_per_second();
	const double picture_bits = lowban_picture_rate.bits_per_second(seconds, picture_interval_seconds);
	const double geometry_bits = lowban_geometry_rate.bits_per_second(seconds, geometry_tick);
	state.lowban.payload_kbit_per_second = (picture_bits + geometry_bits) / 1000.0;
	state.lowban.link_kbit_per_second = state.lowban.payload_kbit_per_second
		+ lowban_header_rate.bits_per_second(seconds, geometry_tick) / 1000.0 + audio;
	state.lowban_geometry_kbit_per_second = geometry_bits / 1000.0;
	state.lowban_updated_blocks = lowban_updated_blocks;
	state.lowban_keyframe = lowban_keyframe;
	state.lowban_warped = lowban_warped;

	state.baseline.payload_kbit_per_second = baseline_rate.bits_per_second(seconds, geometry_tick) / 1000.0;
	state.baseline.link_kbit_per_second = state.baseline.payload_kbit_per_second
		+ baseline_header_rate.bits_per_second(seconds, geometry_tick) / 1000.0 + audio;
	state.baseline_face_offset = baseline.face_offset_active();
	state.baseline_status = baseline_status;

	// Drawn after measuring, so the overlay never counts as picture quality. Only geometry
	// this receiver actually decoded is drawn; the baseline receives none.
	if (options.landmark_overlay && !state.lowban.picture.empty())
		for (const auto& parts : received_geometry) draw_face(state.lowban.picture, parts, overlay_colour);
}
