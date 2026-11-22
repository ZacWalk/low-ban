// One simulated call. The sender analyses each captured frame and two independent wires carry it
// to two receivers: low-ban, whose packets carry the faces' landmarks and the picture together,
// and an H.264 baseline held to the same payload rate. Both run on one 15 Hz clock. Everything
// runs on explicit timestamps, so the live engine and the offline evaluation run exactly the
// same code.
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
// Detection keeps the full capture, where the features are.
constexpr int coded_scale = 2;

// The call's clock: at most one low-ban packet and one H.264 frame each tick. Recorded clips are
// written at this rate too.
constexpr int packet_hz = 15;

// The payload rate both wires are held to: H.264's constant bit rate, and the rate low-ban's
// packets draw from a token bucket at.
constexpr int call_kbit_per_second = 20;

// No low-ban packet after the startup keyframe is larger than this, which is also as much as
// its token bucket can save up: one packet that never needs fragmenting.
constexpr size_t packet_cap_bytes = 1200;

// Low-ban's quality, the background's quantiser step (codec.h): the startup keyframe codes at
// the first, and after it the session moves it a step a tick between the other two, coarser
// while macroblocks worth sending go unsent for want of bytes, finer while bytes go unspent.
constexpr int startup_quality = 12;
constexpr int finest_quality = 9;
constexpr int coarsest_quality = 36;

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

// Landmarks in capture coordinates, rounded to the coded picture's pixels: what low-ban sends.
face_geometry coded_geometry(const face_geometry& faces);

struct face_analysis
{
	std::vector<rect_i> boxes; // one per face, left to right
	face_geometry points;      // each face's five landmarks, in the same order
	int confidence = 0;        // the most confident face's, 0 to 100
};

// The sender's analysis of a frame: the faces in its luma and their landmarks.
face_analysis analyse_faces(face_detector& detector, const image_u8& luma);

// The five landmarks found again in a received picture against the source's, averaged over the
// faces. A face the detector cannot find where it was counts as a whole eye spacing out. Faces
// whose eyes are closer than min_scored_eye_spacing are left out: too small to carry an
// expression at the coded size, and the detector finds them only intermittently - a poster on
// the wall behind a speaker. False when there is nothing to measure.
constexpr double min_scored_eye_spacing = 24.0;
bool measure_expression(face_detector& detector, const image_u8& picture, const face_analysis& faces,
                        landmark_error& error);

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
	std::vector<rect_i> faces;        // detected boxes, in frame coordinates
	int width = 0;
	int height = 0;
	int face_count = 0;
	int confidence = 0;

	receiver_view lowban;
	double lowban_geometry_kbit_per_second = 0.0; // the landmarks' share of the payload
	int lowban_updated_blocks = 0;
	int lowban_quality = 0; // the background's quantiser step
	bool lowban_keyframe = false;
	bool lowban_warped = false;

	receiver_view baseline;
	bool baseline_face_offset = false;
	std::string baseline_status; // why the baseline is missing, when it is
};

struct session_options
{
	bool baseline_face_offset = true;
};

// Totals since the last reset, for the offline evaluation.
struct session_totals
{
	int64_t lowban_startup_bytes = 0;
	int64_t lowban_bytes = 0;          // after the startup keyframe, landmarks included
	int64_t lowban_geometry_bytes = 0; // the landmarks' share of lowban_bytes
	size_t lowban_largest_packet = 0;  // after the startup keyframe
	int lowban_packets = 0;            // after the startup keyframe
	int lowban_warped_packets = 0;
	int ticks = 0;
	int64_t baseline_first_frame_bytes = 0;
	int64_t baseline_bytes = 0; // after the first frame
	int baseline_frames = 0;
};

class call_session
{
public:
	// Both wires start over from their startup keyframes.
	void reset();

	// Runs the clock if it is due at `seconds`, then fills `state` with what both receivers
	// show. `measure` finds the landmarks again in the received pictures; null skips that.
	void process(const video_frame& frame, double seconds, const face_analysis& faces, face_detector* measure,
	             const session_options& options, render_state& state);

	const session_totals& totals() const { return sums; }

private:
	struct clock
	{
		double next = 0.0;
		bool started = false;
	};

	// Bits over a trailing window. `tick` is the clock's interval, the shortest span a rate
	// can sensibly be averaged over.
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
	void send_lowban(const image_u8& coded, const image_u8& coded_importance, const face_geometry& faces,
	                 double seconds);
	void send_baseline(const image_u8& coded, const face_analysis& faces, double seconds,
	                   const session_options& options);

	int frame_width = 0;
	int frame_height = 0;
	bool face_offset_option = true;

	clock tick_clock;
	landmark_filter smoother;

	// Low-ban: one wire object holds both ends, as in-process encoder and decoder.
	lowban_wire lowban;
	bool lowban_started = false;
	double lowban_tokens = 0.0; // bytes the bucket holds
	int lowban_quality = startup_quality;
	image_u8 lowban_shown;
	int lowban_updated_blocks = 0;
	bool lowban_keyframe = false;
	bool lowban_warped = false;
	rolling_rate lowban_rate;
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
