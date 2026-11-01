// One simulated call. The sender analyses each captured frame and two independent wires carry it
// to two receivers: low-ban, with landmark geometry at 15 Hz and warped transform pictures at
// 5 Hz, and an H.264 baseline at the same bitrate. Everything runs on explicit timestamps, so the
// live engine and the offline evaluation run exactly the same code.
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "baseline.h"
#include "codec.h"
#include "face.h"
#include "image.h"
#include "landmarks.h"

// Pictures are coded at this fraction of the capture size and scaled back up for display.
// Detection and landmarking keep the full capture, where the features are.
constexpr int coded_scale = 2;

// The low-ban picture clock, and the cap on every picture packet after the startup keyframe.
constexpr double picture_interval_seconds = 0.2;
constexpr size_t picture_packet_budget = 1200;

// The geometry clock. It is also the low-ban receiver's animation rate, the H.264 frame rate
// and the frame rate recorded clips are written at.
constexpr int geometry_hz = 15;

// The H.264 baseline's constant bit rate: the low-ban picture cap plus a typical single-face
// geometry stream, so the two wires spend about the same.
constexpr int baseline_kbit_per_second = 52;

// The quantiser offset H.264 applies inside the face boxes; six steps halve the quantiser.
constexpr int baseline_face_offset = -6;

// What a real call also carries, for the link estimate: low-rate speech in 60 ms packets, and
// IPv4, UDP and RTP headers on every packet of every stream.
constexpr double speech_kbit_per_second = 6.0;
constexpr double speech_packet_seconds = 0.06;
constexpr int packet_header_bytes = 40;

double audio_link_kbit_per_second();

// A box in capture coordinates, at the coded picture's scale.
rect_i coded_box(const rect_i& box);

struct face_analysis
{
	std::vector<rect_i> boxes;      // every detection, refined by the previous frame's landmarks
	std::vector<rect_i> landmarked; // the boxes that produced 68 points, matching `landmarks`
	face_geometry landmarks;        // raw predictions on the source frame
	float peak_confidence = 0.0f;
};

// Detection, landmarking and box refinement for the sender.
class face_tracker
{
public:
	void reset();

	// `predictor` may be null, or unloaded, to detect without landmarking.
	face_analysis analyse(const face_detector& detector, const shape_predictor* predictor, const video_frame& frame);

private:
	std::vector<rect_i> tracked;
};

struct receiver_view
{
	image_u8 picture; // what this receiver shows, at the capture size
	double payload_kbit_per_second = 0.0;
	double link_kbit_per_second = 0.0;
	reconstruction_quality quality;
	landmark_error expression; // landmarks found in `picture` against those found in the source
	bool expression_measured = false;
};

struct render_state
{
	std::vector<uint8_t> source_bgra; // panel 1, top-down BGRA
	std::vector<rect_i> faces;        // boxes the landmarker was given, in frame coordinates
	int width = 0;
	int height = 0;
	int face_count = 0;
	float peak_confidence = 0.0f;

	receiver_view lowban;
	double lowban_geometry_kbit_per_second = 0.0;
	int lowban_updated_blocks = 0;
	bool lowban_keyframe = false;
	bool lowban_warped = false;

	receiver_view baseline;
	bool baseline_face_offset = false;
	std::string baseline_status; // why the baseline is missing, when it is
};

struct session_options
{
	bool landmark_overlay = true;
	bool baseline_face_offset = true;
};

// Totals since the last reset, for the offline evaluation.
struct session_totals
{
	int64_t lowban_keyframe_bytes = 0;
	int64_t lowban_picture_bytes = 0; // after the startup keyframe
	int64_t lowban_geometry_bytes = 0;
	size_t lowban_largest_picture_delta = 0;
	int lowban_picture_packets = 0;
	int lowban_geometry_packets = 0;
	int lowban_warped_packets = 0;
	int64_t baseline_first_frame_bytes = 0;
	int64_t baseline_bytes = 0; // after the first frame
	int baseline_frames = 0;
};

class call_session
{
public:
	// Both wires start over from their startup keyframes.
	void reset();

	// Runs every clock that is due at `seconds`, then fills `state` with what both receivers
	// show. `predictor` measures expression on the received pictures; null skips that.
	void process(const video_frame& frame, double seconds, const face_analysis& faces,
		const shape_predictor* predictor, const session_options& options, render_state& state);

	const session_totals& totals() const { return sums; }

private:
	struct clock
	{
		double next = 0.0;
		bool started = false;
	};

	// Bits over a trailing window. `tick` is the stream's own interval, the shortest span a
	// rate can sensibly be averaged over.
	class rolling_rate
	{
	public:
		void reset() { entries.clear(); }
		void add(double seconds, int64_t bits);
		double bits_per_second(double seconds, double tick);

	private:
		std::deque<std::pair<double, int64_t>> entries;
	};

	static bool due(clock& schedule, double seconds, double interval);
	void send_geometry(const face_geometry& smoothed, double seconds);
	void send_picture(const image_u8& coded, const image_u8& coded_importance, double seconds);
	void send_baseline(const image_u8& coded, const face_analysis& faces, double seconds,
		const session_options& options);

	int frame_width = 0;
	int frame_height = 0;
	bool face_offset_option = true;

	clock geometry_clock;
	clock picture_clock;
	landmark_filter smoother;

	// Low-ban: one stream object holds both ends, as in-process encoder and decoder.
	transform_codec codec;
	temporal_landmark_stream geometry_wire;
	temporal_transform_stream picture_wire;
	face_geometry received_geometry;
	bool picture_started = false;
	bool animation_stale = false;
	image_u8 lowban_shown;
	int lowban_updated_blocks = 0;
	bool lowban_keyframe = false;
	bool lowban_warped = false;
	rolling_rate lowban_picture_rate;
	rolling_rate lowban_geometry_rate;
	rolling_rate lowban_header_rate;

	h264_wire baseline;
	bool baseline_tried = false;
	std::string baseline_status;
	image_u8 baseline_decoded;
	image_u8 baseline_shown;
	rolling_rate baseline_rate;
	rolling_rate baseline_header_rate;

	session_totals sums;
};
