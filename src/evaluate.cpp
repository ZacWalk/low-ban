// Offline evaluation, with no camera. The PNG stills in ../samples are scored as startup
// pictures against H.264 at the same byte count, and any recorded clips (../samples/*.mp4) are
// played through the full call session, both wires, exactly as the window runs them.
#include <SDKDDKVer.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

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
#include "warp.h"

namespace
{
	using Microsoft::WRL::ComPtr;

	constexpr int frame_width = 640;
	constexpr int frame_height = 480;

	std::wstring samples_folder()
	{
		wchar_t module[MAX_PATH] = {};
		const DWORD length = GetModuleFileNameW(nullptr, module, MAX_PATH);
		if (length == 0 || length >= MAX_PATH) return L"samples\\";
		std::wstring path(module, length);
		const size_t slash = path.find_last_of(L'\\');
		path.resize(slash == std::wstring::npos ? 0 : slash + 1);
		return path + L"..\\samples\\";
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

	uint8_t clamp_byte(const int value)
	{
		return static_cast<uint8_t>(std::clamp(value, 0, 255));
	}

	bool load_png(IWICImagingFactory* factory, const std::wstring& path, video_frame& result)
	{
		ComPtr<IWICBitmapDecoder> decoder;
		ComPtr<IWICBitmapFrameDecode> source;
		ComPtr<IWICBitmapScaler> scaler;
		ComPtr<IWICFormatConverter> converter;
		if (FAILED(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
			WICDecodeMetadataCacheOnDemand, &decoder))) return false;
		if (FAILED(decoder->GetFrame(0, &source))) return false;
		if (FAILED(factory->CreateBitmapScaler(&scaler))) return false;
		if (FAILED(scaler->Initialize(source.Get(), frame_width, frame_height, WICBitmapInterpolationModeFant)))
			return false;
		if (FAILED(factory->CreateFormatConverter(&converter))) return false;
		if (FAILED(converter->Initialize(scaler.Get(), GUID_WICPixelFormat32bppBGRA,
			WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) return false;

		std::vector<uint8_t> pixels(static_cast<size_t>(frame_width) * frame_height * 4);
		if (FAILED(converter->CopyPixels(nullptr, frame_width * 4, static_cast<UINT>(pixels.size()), pixels.data())))
			return false;

		result.luma.resize(frame_width, frame_height);
		result.cb.resize(frame_width / 2, frame_height);
		result.cr.resize(frame_width / 2, frame_height);
		for (int y = 0; y < frame_height; ++y)
		{
			for (int x = 0; x < frame_width; x += 2)
			{
				int cb_sum = 0;
				int cr_sum = 0;
				for (int offset = 0; offset < 2; ++offset)
				{
					const uint8_t* pixel = pixels.data() + (static_cast<size_t>(y) * frame_width + x + offset) * 4;
					const int blue = pixel[0];
					const int green = pixel[1];
					const int red = pixel[2];
					result.luma(x + offset, y) = clamp_byte(16 + ((66 * red + 129 * green + 25 * blue + 128) >> 8));
					cb_sum += 128 + ((-38 * red - 74 * green + 112 * blue + 128) >> 8);
					cr_sum += 128 + ((112 * red - 94 * green - 18 * blue + 128) >> 8);
				}
				result.cb(x / 2, y) = clamp_byte(cb_sum / 2);
				result.cr(x / 2, y) = clamp_byte(cr_sum / 2);
			}
		}
		return true;
	}

	image_u8 face_importance(const video_frame& frame, const face_analysis& faces)
	{
		image_u8 importance(frame.width(), frame.height());
		importance.fill(32);
		for (size_t i = 0; i < faces.landmarks.size(); ++i)
			add_face_importance(importance, faces.landmarked[i], faces.landmarks[i]);
		return importance;
	}

	std::vector<rect_i> coded_boxes(const face_analysis& faces)
	{
		std::vector<rect_i> boxes;
		for (const rect_i& box : faces.landmarked) boxes.push_back(coded_box(box));
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

		// Percentages of the eye-corner distance; -1 when nothing could be measured.
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
		const image_u8& importance, const shape_predictor& predictor)
	{
		image_u8 shown(frame.width(), frame.height());
		resize_bilinear(coded_picture, shown);

		landmark_error total;
		int measured = 0;
		for (size_t i = 0; i < faces.landmarks.size(); ++i)
		{
			landmark_error error;
			if (!compare_landmarks(faces.landmarks[i], predictor.predict(shown, faces.landmarked[i]), error)) continue;
			total.overall += error.overall;
			total.mouth += error.mouth;
			++measured;
		}
		if (measured > 0)
		{
			total.overall /= measured;
			total.mouth /= measured;
		}

		tally scores;
		scores.add(measure_quality(frame.luma, shown, &importance), measured > 0 ? &total : nullptr);
		return scores;
	}

	// The finest H.264 picture that fits `budget` bytes, by binary search over the fixed
	// quantiser. When even the coarsest overshoots, that one is returned as it is.
	bool h264_still(const image_u8& coded, const std::vector<rect_i>& faces, const int face_offset,
		const size_t budget, std::vector<uint8_t>& best_bytes, image_u8& best_picture, std::string& error)
	{
		const auto encode = [&](const int quantizer, std::vector<uint8_t>& bytes, image_u8& picture)
		{
			h264_settings settings;
			settings.width = coded.width();
			settings.height = coded.height();
			settings.frames_per_second = geometry_hz;
			settings.quantizer = quantizer;
			settings.face_offset = face_offset;
			h264_wire wire;
			if (!wire.open(settings, error) || !wire.encode(coded, faces, bytes, error)
				|| !wire.decode(bytes, picture, error)) return false;
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

	int evaluate_stills(IWICImagingFactory* factory, const face_detector& detector, const shape_predictor& predictor)
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

		const transform_codec codec;
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
			if (!load_png(factory, samples_folder() + name, frame))
			{
				std::printf("could not decode %ls\n", name.c_str());
				continue;
			}

			face_tracker tracker;
			const face_analysis faces = tracker.analyse(detector, &predictor, frame);
			const image_u8 importance = face_importance(frame, faces);
			image_u8 coded(frame.width() / coded_scale, frame.height() / coded_scale);
			resize_bilinear(frame.luma, coded);
			image_u8 coded_importance(coded.width(), coded.height());
			resize_bilinear(importance, coded_importance);

			// Low-ban's startup: the geometry keyframe, then one complete picture keyframe.
			temporal_landmark_stream geometry;
			const landmark_packet points = geometry.encode(faces.landmarks, true);
			temporal_transform_stream wire;
			const transform_packet packet = wire.encode(codec, coded, &coded_importance, true, 0,
				scale_geometry(geometry.transmitted(), coded_scale));
			face_geometry received;
			image_u8 lowban_picture;
			if (!geometry.decode(points, received) || !wire.decode(codec, packet, coded.width(), coded.height(),
				lowban_picture, scale_geometry(received, coded_scale)))
			{
				std::printf("could not decode the low-ban startup for %ls\n", name.c_str());
				continue;
			}
			const size_t lowban_bytes = packet.bytes.size() + points.bytes.size();
			const tally lowban_scores = score_still(lowban_picture, frame, faces, importance, predictor);

			// H.264 gets the same total, geometry included, since it has no geometry to send.
			tally h264_scores;
			tally offset_scores;
			std::vector<uint8_t> h264_bytes;
			std::vector<uint8_t> offset_bytes;
			image_u8 h264_picture;
			image_u8 offset_picture;
			if (h264_error.empty()
				&& h264_still(coded, coded_boxes(faces), 0, lowban_bytes, h264_bytes, h264_picture, h264_error)
				&& h264_still(coded, coded_boxes(faces), baseline_face_offset, lowban_bytes, offset_bytes,
					offset_picture, h264_error))
			{
				h264_scores = score_still(h264_picture, frame, faces, importance, predictor);
				offset_scores = score_still(offset_picture, frame, faces, importance, predictor);
			}

			std::printf("%ls,%zu,%.1f", name.c_str(), faces.landmarks.size(), lowban_bytes * 8.0 / 1000.0);
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

	void evaluate_clips(const face_detector& detector, const shape_predictor& predictor)
	{
		const std::vector<std::wstring> names = sample_files(L"*.mp4");
		if (names.empty())
		{
			std::printf("no clips in samples/ - record one with File > Record clip to measure motion\n");
			return;
		}

		std::printf("clip,frames,seconds,"
			"lowban_startup_kbit,lowban_picture_kbit_per_second,lowban_geometry_kbit_per_second,"
			"lowban_link_kbit_per_second,lowban_psnr,lowban_face_psnr,lowban_landmark_error_pct,"
			"lowban_mouth_error_pct,lowban_warped_packets,"
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
			face_tracker tracker;
			call_session with_offset;
			call_session uniform;
			session_options offset_options;
			offset_options.landmark_overlay = false;
			session_options uniform_options = offset_options;
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
				const face_analysis faces = tracker.analyse(detector, &predictor, frame);
				render_state offset_state;
				render_state uniform_state;
				with_offset.process(frame, last, faces, &predictor, offset_options, offset_state);
				uniform.process(frame, last, faces, &predictor, uniform_options, uniform_state);
				lowban_scores.add(offset_state.lowban);
				h264_scores.add(offset_state.baseline);
				uniform_scores.add(uniform_state.baseline);
				++frames;
			}
			if (!error.empty()) std::printf("%ls stopped early: %s\n", name.c_str(), error.c_str());
			if (frames == 0) continue;

			const double duration = last + 1.0 / geometry_hz;
			const session_totals& sums = with_offset.totals();
			const session_totals& uniform_sums = uniform.totals();
			const double audio = audio_link_kbit_per_second();
			const double lowban_headers = kbit_per_second(
				static_cast<int64_t>(sums.lowban_picture_packets + sums.lowban_geometry_packets) * packet_header_bytes,
				duration);
			const double h264_headers = kbit_per_second(
				static_cast<int64_t>(sums.baseline_frames) * packet_header_bytes, duration);

			std::printf("%ls,%d,%.2f", name.c_str(), frames, duration);
			std::printf(",%.1f,%.1f,%.1f,%.1f", sums.lowban_keyframe_bytes * 8.0 / 1000.0,
				kbit_per_second(sums.lowban_picture_bytes, duration),
				kbit_per_second(sums.lowban_geometry_bytes, duration),
				kbit_per_second(sums.lowban_picture_bytes + sums.lowban_geometry_bytes, duration) + lowban_headers + audio);
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
		ComPtr<IWICImagingFactory> factory;
		if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))))
		{
			std::printf("could not create Windows image decoder\n");
		}
		else
		{
			shape_predictor predictor;
			std::string model_error;
			if (!predictor.load(asset_path(L"shape_predictor_68_face_landmarks.dat"), model_error))
				std::printf("landmark model unavailable: %s\n", model_error.c_str());
			const face_detector detector;

			result = evaluate_stills(factory.Get(), detector, predictor);
			evaluate_clips(detector, predictor);

			std::printf("picture_hz,%.0f\n", 1.0 / picture_interval_seconds);
			std::printf("picture_packet_cap_bytes,%zu\n", picture_packet_budget);
			std::printf("geometry_hz,%d\n", geometry_hz);
			std::printf("h264_constant_kbit_per_second,%d\n", baseline_kbit_per_second);

			// An encoder that refuses region of interest offsets codes the face-offset columns
			// exactly like the uniform ones; this says which happened.
			h264_settings probe;
			probe.width = frame_width / coded_scale;
			probe.height = frame_height / coded_scale;
			probe.frames_per_second = geometry_hz;
			probe.kbit_per_second = baseline_kbit_per_second;
			probe.face_offset = baseline_face_offset;
			h264_wire wire;
			std::string probe_error;
			std::printf("h264_face_offset_active,%d\n", wire.open(probe, probe_error) && wire.face_offset_active() ? 1 : 0);
			std::printf("audio_link_allowance_kbit_per_second,%.1f\n", audio_link_kbit_per_second());
		}
	}

	if (uninitialise) CoUninitialize();
	return result;
}
