// Offline evaluation, with no camera. The PNG stills in ../samples are scored as startup
// pictures against H.264 at the same byte count, and any clips there - recorded ones
// (../samples/*.mp4) or YUV4MPEG2 test sequences (../samples/*.y4m) - are played through the
// full call session, both wires, exactly as the window runs them.
#include <SDKDDKVer.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <objbase.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "baseline.h"
#include "capture.h"
#include "codec.h"
#include "engine.h"
#include "face.h"
#include "landmarks.h"
#include "session.h"

namespace
{
	constexpr int frame_width = 640;
	constexpr int frame_height = 480;

	std::wstring samples_folder()
	{
		return asset_path(L"..\\samples\\");
	}

	std::vector<std::wstring> sample_files(const wchar_t* pattern)
	{
		std::vector<std::wstring> names;
		WIN32_FIND_DATAW found = {};
		const HANDLE search = FindFirstFileW((samples_folder() + pattern).c_str(), &found);
		if (search == INVALID_HANDLE_VALUE) return names;
		do names.emplace_back(found.cFileName);
		while (FindNextFileW(search, &found));
		FindClose(search);
		std::sort(names.begin(), names.end());
		return names;
	}

	image_u8 face_importance_map(const video_frame& frame, const face_analysis& faces)
	{
		image_u8 importance(frame.width(), frame.height());
		importance.fill(background_importance);
		for (size_t i = 0; i < faces.boxes.size() && i < faces.points.size(); ++i)
			add_face_importance(importance, faces.boxes[i], faces.points[i]);
		return importance;
	}

	std::vector<rect_i> coded_boxes(const face_analysis& faces)
	{
		std::vector<rect_i> boxes;
		boxes.reserve(faces.boxes.size());
		for (const rect_i& box : faces.boxes) boxes.push_back(coded_box(box));
		return boxes;
	}

	// Sums of what a receiver showed, scored the way the window scores it.
	struct tally
	{
		double psnr = 0.0;
		double face_psnr = 0.0;
		double expression = 0.0;
		double mouth = 0.0;
		int pictures = 0;
		int expressions = 0;

		void add(const reconstruction_quality& quality, const landmark_error* error)
		{
			psnr += quality.psnr;
			face_psnr += quality.weighted_psnr;
			++pictures;
			if (error == nullptr) return;
			expression += error->overall;
			mouth += error->mouth;
			++expressions;
		}

		void add(const receiver_view& view)
		{
			if (!view.picture.empty()) add(view.quality, view.expression_measured ? &view.expression : nullptr);
		}

		void add(const tally& other)
		{
			psnr += other.psnr;
			face_psnr += other.face_psnr;
			expression += other.expression;
			mouth += other.mouth;
			pictures += other.pictures;
			expressions += other.expressions;
		}

		double mean_psnr() const { return pictures > 0 ? psnr / pictures : 0.0; }
		double mean_face_psnr() const { return pictures > 0 ? face_psnr / pictures : 0.0; }

		// Percentages of the eye spacing; -1 when nothing could be measured.
		double mean_expression() const { return expressions > 0 ? expression / expressions * 100.0 : -1.0; }
		double mean_mouth() const { return expressions > 0 ? mouth / expressions * 100.0 : -1.0; }
	};

	void print_scores(const tally& scores)
	{
		std::printf(",%.2f,%.2f,%.2f", scores.mean_psnr(), scores.mean_face_psnr(), scores.mean_expression());
	}

	// Scores a picture decoded at the coded size: upscaled as the window shows it, then PSNR,
	// face-weighted PSNR and the error of the landmarks found in it.
	tally score_still(const image_u8& coded_picture, const video_frame& frame, const face_analysis& faces,
	                  const image_u8& importance, face_detector& detector)
	{
		image_u8 shown(frame.width(), frame.height());
		resize_bicubic(coded_picture, shown);

		landmark_error error;
		const bool measured = measure_expression(detector, shown, faces, error);
		tally scores;
		scores.add(measure_quality(frame.luma, shown, &importance), measured ? &error : nullptr);
		return scores;
	}

	// The finest H.264 picture that fits `budget` bytes, by binary search over the fixed
	// quantiser. When even the coarsest overshoots, that one is returned as it is.
	bool h264_still(const image_u8& coded, const std::vector<rect_i>& faces, const int face_offset, const size_t budget,
	                std::vector<uint8_t>& best_bytes, image_u8& best_picture, std::string& error)
	{
		const auto encode = [&](const int quantizer, std::vector<uint8_t>& bytes, image_u8& picture)
		{
			h264_settings settings;
			settings.width = coded.width();
			settings.height = coded.height();
			settings.frames_per_second = packet_hz;
			settings.quantizer = quantizer;
			settings.face_offset = face_offset;
			h264_wire wire;
			if (!wire.open(settings, error) || !wire.encode(coded, faces, bytes, error) ||
			    !wire.decode(bytes, picture, error))
				return false;
			if (picture.empty()) error = "the H.264 decoder returned no picture for a single frame";
			return !picture.empty();
		};

		bool found = false;
		int low = 0;
		int high = 51;
		while (low <= high)
		{
			const int quantizer = (low + high) / 2;
			std::vector<uint8_t> bytes;
			image_u8 picture;
			if (!encode(quantizer, bytes, picture)) return false;
			if (bytes.size() <= budget)
			{
				found = true;
				best_bytes = std::move(bytes);
				best_picture = std::move(picture);
				high = quantizer - 1;
			}
			else low = quantizer + 1;
		}
		return found || encode(51, best_bytes, best_picture);
	}

	int evaluate_stills(face_detector& detector)
	{
		const std::vector<std::wstring> names = sample_files(L"*.png");
		if (names.empty())
		{
			std::printf("no PNG samples found\n");
			return 1;
		}

		std::printf("still,faces,lowban_kbit,lowban_psnr,lowban_face_psnr,lowban_landmark_error_pct,"
		            "h264_kbit,h264_psnr,h264_face_psnr,h264_landmark_error_pct,"
		            "h264_face_offset_kbit,h264_face_offset_psnr,h264_face_offset_face_psnr,"
		            "h264_face_offset_landmark_error_pct\n");

		tally lowban_total;
		tally h264_total;
		tally offset_total;
		double lowban_kbit = 0.0;
		double h264_kbit = 0.0;
		double offset_kbit = 0.0;
		int scored = 0;
		std::string h264_error;

		for (const std::wstring& name : names)
		{
			video_frame frame;
			if (!load_still(samples_folder() + name, frame_width, frame_height, frame))
			{
				std::printf("could not decode %ls\n", name.c_str());
				continue;
			}

			const face_analysis faces = analyse_faces(detector, frame.luma);
			const image_u8 importance = face_importance_map(frame, faces);
			image_u8 coded(frame.width() / coded_scale, frame.height() / coded_scale);
			resize_bilinear(frame.luma, coded);
			image_u8 coded_importance(coded.width(), coded.height());
			resize_bilinear(importance, coded_importance);

			// Low-ban's startup: one keyframe packet, the landmarks and the complete picture.
			lowban_wire wire;
			const lowban_packet packet = wire.encode(coded, &coded_importance, coded_geometry(faces.points), true, 0,
			                                         startup_quality);
			image_u8 lowban_decoded;
			image_u8 lowban_picture;
			if (!wire.decode(packet, lowban_decoded) || !wire.show(lowban_picture))
			{
				std::printf("could not decode the low-ban startup for %ls\n", name.c_str());
				continue;
			}
			const size_t lowban_bytes = packet.bytes.size();
			const tally lowban_scores = score_still(lowban_picture, frame, faces, importance, detector);

			// H.264 gets the same total, landmarks included, since it has no landmarks to send.
			tally h264_scores;
			tally offset_scores;
			std::vector<uint8_t> h264_bytes;
			std::vector<uint8_t> offset_bytes;
			image_u8 h264_picture;
			image_u8 offset_picture;
			if (h264_error.empty() &&
			    h264_still(coded, coded_boxes(faces), 0, lowban_bytes, h264_bytes, h264_picture, h264_error) &&
			    h264_still(coded, coded_boxes(faces), baseline_face_offset, lowban_bytes, offset_bytes, offset_picture,
			               h264_error))
			{
				h264_scores = score_still(h264_picture, frame, faces, importance, detector);
				offset_scores = score_still(offset_picture, frame, faces, importance, detector);
			}

			std::printf("%ls,%zu,%.1f", name.c_str(), faces.points.size(), lowban_bytes * 8.0 / 1000.0);
			print_scores(lowban_scores);
			std::printf(",%.1f", h264_bytes.size() * 8.0 / 1000.0);
			print_scores(h264_scores);
			std::printf(",%.1f", offset_bytes.size() * 8.0 / 1000.0);
			print_scores(offset_scores);
			std::printf("\n");

			lowban_kbit += lowban_bytes * 8.0 / 1000.0;
			h264_kbit += h264_bytes.size() * 8.0 / 1000.0;
			offset_kbit += offset_bytes.size() * 8.0 / 1000.0;
			lowban_total.add(lowban_scores);
			h264_total.add(h264_scores);
			offset_total.add(offset_scores);
			++scored;
		}

		if (scored == 0) return 1;
		if (!h264_error.empty()) std::printf("h264 unavailable: %s\n", h264_error.c_str());
		std::printf("mean,%d,%.1f", scored, lowban_kbit / scored);
		print_scores(lowban_total);
		std::printf(",%.1f", h264_kbit / scored);
		print_scores(h264_total);
		std::printf(",%.1f", offset_kbit / scored);
		print_scores(offset_total);
		std::printf("\n");
		return 0;
	}

	// Rate over a clip, excluding the startup picture, which is reported on its own.
	double kbit_per_second(const int64_t bytes, const double seconds)
	{
		return seconds > 0.0 ? bytes * 8.0 / seconds / 1000.0 : 0.0;
	}

	std::vector<std::wstring> clip_files()
	{
		std::vector<std::wstring> names = sample_files(L"*.mp4");
		const std::vector<std::wstring> streams = sample_files(L"*.y4m");
		names.insert(names.end(), streams.begin(), streams.end());
		return names;
	}

	void evaluate_clips(face_detector& detector)
	{
		const std::vector<std::wstring> names = clip_files();
		if (names.empty())
		{
			std::printf("no clips in samples/ - record one with the Record clip button, or add a .y4m sequence, "
			            "to measure motion\n");
			return;
		}

		std::printf("clip,frames,seconds,"
		            "lowban_startup_kbit,lowban_kbit_per_second,lowban_landmark_kbit_per_second,"
		            "lowban_packets_per_second,lowban_link_kbit_per_second,lowban_psnr,lowban_face_psnr,"
		            "lowban_landmark_error_pct,lowban_mouth_error_pct,lowban_warped_packets,"
		            "h264_startup_kbit,h264_kbit_per_second,h264_link_kbit_per_second,h264_psnr,h264_face_psnr,"
		            "h264_landmark_error_pct,h264_mouth_error_pct,"
		            "h264_uniform_kbit_per_second,h264_uniform_psnr,h264_uniform_face_psnr,"
		            "h264_uniform_landmark_error_pct,h264_uniform_mouth_error_pct\n");

		for (const std::wstring& name : names)
		{
			clip_reader reader;
			std::string error;
			if (!reader.open(samples_folder() + name, frame_width, frame_height, error))
			{
				std::printf("could not open %ls: %s\n", name.c_str(), error.c_str());
				continue;
			}

			// Two sessions see the same frames and the same analysis; they differ only in
			// whether H.264 is given the face boxes. Low-ban runs identically in both.
			call_session with_offset;
			call_session uniform;
			const session_options offset_options;
			session_options uniform_options;
			uniform_options.baseline_face_offset = false;

			tally lowban_scores;
			tally h264_scores;
			tally uniform_scores;
			video_frame frame;
			double seconds = 0.0;
			double first = -1.0;
			double last = 0.0;
			int frames = 0;
			while (reader.read(frame, seconds, error))
			{
				if (first < 0.0) first = seconds;
				last = seconds - first;
				const face_analysis faces = analyse_faces(detector, frame.luma);
				render_state offset_state;
				render_state uniform_state;
				with_offset.process(frame, last, faces, &detector, offset_options, offset_state);
				uniform.process(frame, last, faces, &detector, uniform_options, uniform_state);
				lowban_scores.add(offset_state.lowban);
				h264_scores.add(offset_state.baseline);
				uniform_scores.add(uniform_state.baseline);
				++frames;
			}
			if (!error.empty()) std::printf("%ls stopped early: %s\n", name.c_str(), error.c_str());
			if (frames == 0) continue;

			// Steady rates leave out each wire's startup picture, which is reported on its own.
			const double duration = last + 1.0 / packet_hz;
			const session_totals& sums = with_offset.totals();
			const session_totals& uniform_sums = uniform.totals();
			const double audio = audio_link_kbit_per_second();
			const double lowban_headers = kbit_per_second(static_cast<int64_t>(sums.lowban_packets) *
			                                                  packet_header_bytes,
			                                              duration);
			const double h264_headers = kbit_per_second(static_cast<int64_t>(std::max(0, sums.baseline_frames - 1)) *
			                                                packet_header_bytes,
			                                            duration);

			std::printf("%ls,%d,%.2f", name.c_str(), frames, duration);
			std::printf(",%.1f,%.1f,%.1f,%.1f,%.1f", sums.lowban_startup_bytes * 8.0 / 1000.0,
			            kbit_per_second(sums.lowban_bytes, duration),
			            kbit_per_second(sums.lowban_geometry_bytes, duration), sums.lowban_packets / duration,
			            kbit_per_second(sums.lowban_bytes, duration) + lowban_headers + audio);
			print_scores(lowban_scores);
			std::printf(",%.2f,%d", lowban_scores.mean_mouth(), sums.lowban_warped_packets);
			std::printf(",%.1f,%.1f,%.1f", sums.baseline_first_frame_bytes * 8.0 / 1000.0,
			            kbit_per_second(sums.baseline_bytes, duration),
			            kbit_per_second(sums.baseline_bytes, duration) + h264_headers + audio);
			print_scores(h264_scores);
			std::printf(",%.2f", h264_scores.mean_mouth());
			std::printf(",%.1f", kbit_per_second(uniform_sums.baseline_bytes, duration));
			print_scores(uniform_scores);
			std::printf(",%.2f\n", uniform_scores.mean_mouth());
		}
	}
}
int run_sample_evaluation()
{
	const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	const bool uninitialise = SUCCEEDED(com);
	if (FAILED(com) && com != RPC_E_CHANGED_MODE)
	{
		std::printf("could not initialise COM (0x%08lx)\n", static_cast<unsigned long>(com));
		return 1;
	}

	int result = 1;
	{
		const media_foundation media;
		face_detector detector;

		result = evaluate_stills(detector);
		evaluate_clips(detector);

		std::printf("packet_hz,%d\n", packet_hz);
		std::printf("packet_cap_bytes,%zu\n", packet_cap_bytes);
		std::printf("call_kbit_per_second,%d\n", call_kbit_per_second);

		// An encoder that refuses region of interest offsets codes the face-offset columns
		// exactly like the uniform ones; this says which happened.
		h264_settings probe;
		probe.width = frame_width / coded_scale;
		probe.height = frame_height / coded_scale;
		probe.frames_per_second = packet_hz;
		probe.kbit_per_second = call_kbit_per_second;
		probe.face_offset = baseline_face_offset;
		h264_wire wire;
		std::string probe_error;
		std::printf("h264_face_offset_active,%d\n", wire.open(probe, probe_error) && wire.face_offset_active() ? 1 : 0);
		std::printf("audio_link_allowance_kbit_per_second,%.1f\n", audio_link_kbit_per_second());
	}

	if (uninitialise) CoUninitialize();
	return result;
}
