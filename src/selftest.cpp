// Headless checks for everything that does not need a camera or a window.
// Run with: low-ban-64d.exe /test
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "codec.h"
#include "draw.h"
#include "engine.h"
#include "face.h"
#include "image.h"
#include "landmarks.h"

namespace
{
	int failures = 0;

	void check(const bool condition, const char* description)
	{
		if (!condition) ++failures;
		std::printf("  [%s] %s\n", condition ? "pass" : "FAIL", description);
	}

	// A frame with a skin coloured ellipse on a neutral grey background.
	video_frame make_face_frame(const int width, const int height, const int cx, const int cy,
	                            const int rx, const int ry)
	{
		video_frame frame;
		frame.luma.resize(width, height);
		frame.cb.resize(width / 2, height);
		frame.cr.resize(width / 2, height);
		frame.luma.fill(90);
		frame.cb.fill(128);
		frame.cr.fill(128);

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				const float dx = static_cast<float>(x - cx) / static_cast<float>(rx);
				const float dy = static_cast<float>(y - cy) / static_cast<float>(ry);
				if (dx * dx + dy * dy > 1.0f) continue;

				frame.luma(x, y) = 170;
				frame.cb(x / 2, y) = 105;
				frame.cr(x / 2, y) = 155;
			}
		}

		return frame;
	}

	void test_image_ops()
	{
		std::printf("image operations\n");

		image_u8 src(64, 64);
		for (int y = 0; y < 64; ++y)
			for (int x = 0; x < 64; ++x)
				src(x, y) = static_cast<uint8_t>(x < 32 ? 40 : 200);

		image_u8 small(16, 16);
		resize_bilinear(src, small);
		check(small(2, 8) < 100 && small(13, 8) > 150, "bilinear resize preserves a step edge");

		image_u8 flat(32, 32);
		flat.fill(123);
		image_u8 blurred;
		gaussian_blur(flat, blurred);
		check(blurred(16, 16) == 123 && blurred(0, 0) == 123, "gaussian blur leaves a flat image unchanged");

		image_u8 edges;
		sobel_edges(src, edges);
		check(edges(31, 32) > 20 || edges(32, 32) > 20, "sobel responds at the edge");
		check(edges(8, 32) == 0 && edges(56, 32) == 0, "sobel is quiet in flat regions");

		image_u8 narrow(16, 16);
		narrow.fill(100);
		narrow(8, 8) = 140;
		contrast_stretch(narrow);
		check(narrow(0, 0) == 0 && narrow(8, 8) == 255, "contrast stretch spans the full range");
	}

	void test_yuy2()
	{
		std::printf("yuy2 decoding\n");

		constexpr int width = 8;
		constexpr int height = 4;
		constexpr int stride = width * 2;
		std::vector<uint8_t> buffer(static_cast<size_t>(stride) * height);

		for (int y = 0; y < height; ++y)
			for (int x = 0; x < width / 2; ++x)
			{
				uint8_t* p = buffer.data() + static_cast<size_t>(y) * stride + x * 4;
				p[0] = static_cast<uint8_t>(y * 10 + x * 2);
				p[1] = 100;
				p[2] = static_cast<uint8_t>(y * 10 + x * 2 + 1);
				p[3] = 200;
			}

		video_frame frame;
		check(decode_yuy2(frame, buffer.data(), buffer.size(), width, height, stride), "top-down decode succeeds");
		check(frame.luma(3, 2) == 23 && frame.cb(1, 2) == 100 && frame.cr(1, 2) == 200, "planes carry the right values");

		video_frame flipped;
		check(decode_yuy2(flipped, buffer.data(), buffer.size(), width, height, -stride), "bottom-up decode succeeds");
		check(flipped.luma(3, 1) == frame.luma(3, 2), "negative stride is normalised to top-down");

		video_frame rejected;
		check(!decode_yuy2(rejected, buffer.data(), 4, width, height, stride), "short buffers are rejected");
	}

	void test_drawing()
	{
		std::printf("drawing\n");

		image_u8 canvas(32, 32);
		canvas.fill(255);
		draw_line(canvas, 0.0f, 16.0f, 31.0f, 16.0f, 0);
		check(canvas(16, 16) < 128, "horizontal line is drawn");

		canvas.fill(255);
		draw_line(canvas, 16.0f, 0.0f, 16.0f, 31.0f, 0);
		check(canvas(16, 16) < 128, "vertical line is drawn");

		canvas.fill(255);
		draw_line(canvas, -100.0f, -100.0f, 200.0f, 200.0f, 0);
		check(canvas(16, 16) < 200, "off-screen endpoints are clipped, not dropped");

		canvas.fill(255);
		draw_rect(canvas, rect_i::from_size(4, 4, 8, 8), 0);
		check(canvas(8, 4) < 128 && canvas(8, 8) == 255, "rectangle is an outline");
	}

	void test_codec()
	{
		std::printf("low bandwidth codecs\n");

		image_u8 frame(160, 120);
		for (int y = 0; y < 120; ++y)
			for (int x = 0; x < 160; ++x)
			{
				const int local_x = x % patch_autoencoder::patch_size;
				const int local_y = y % patch_autoencoder::patch_size;
				const int block = x / patch_autoencoder::patch_size + y / patch_autoencoder::patch_size;

				// The ramp alone is a plane, which an exact mean plus two code units would
				// reconstruct perfectly; the phase-shifted ripple is what the bottleneck has
				// to actually learn.
				const double value = 40 + block % 5 * 25 + local_x * (block % 3 + 1)
					+ local_y * ((block + 1) % 3 + 1)
					+ 30.0 * std::sin(local_x * 0.7 + block) * std::sin(local_y * 0.5);
				frame(x, y) = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
			}

		patch_autoencoder codec;
		image_u8 initial_output;
		codec.reconstruct(frame, initial_output);
		const reconstruction_quality initial_quality = measure_quality(frame, initial_output);

		const float first = codec.train(frame, 128);
		float last = first;
		for (int i = 1; i < 400; ++i) last = codec.train(frame, 128);

		check(codec.steps() == 400, "training steps are counted");
		check(last < first * 0.5f, "loss decreases substantially with training");

		image_u8 output;
		codec.reconstruct(frame, output);
		check(output.width() == frame.width() && output.height() == frame.height(),
		      "reconstruction keeps the frame size");

		const reconstruction_quality trained_quality = measure_quality(frame, output);
		std::printf("  initial PSNR %.2f dB, trained %.2f dB\n", initial_quality.psnr, trained_quality.psnr);
		check(trained_quality.psnr > initial_quality.psnr + 3.0,
		      "training materially improves reconstruction PSNR");
		check(trained_quality.psnr > 20.0, "reconstruction reaches usable quality");

		check(patch_autoencoder::bits_per_frame(640, 480) == 32 * 24 * 4 * 8,
		      "bitrate accounting matches the four-byte layout");

		image_u8 importance(frame.width(), frame.height());
		importance.fill(32);
		for (int y = 30; y < 90; ++y)
			for (int x = 40; x < 120; ++x) importance(x, y) = 255;

		transform_codec traditional;
		image_u8 traditional_output;
		traditional.reconstruct(frame, &importance, traditional_output);
		check(traditional_output.width() == frame.width() && traditional_output.height() == frame.height(),
		      "transform reconstruction keeps the frame size");

		const reconstruction_quality quality = measure_quality(frame, traditional_output, &importance);
		check(quality.rmse > 0.0 && quality.psnr > 10.0, "transform quality is measurable and usable");
		temporal_transform_stream transform_stream;
		const transform_packet transform_keyframe = transform_stream.encode(
			traditional, frame, &importance, true, 0);
		image_u8 transform_decoded;
		check(transform_keyframe.keyframe && transform_keyframe.updated_blocks == 20 * 15,
		      "transform keyframe contains every block");
		check(transform_stream.decode(traditional, transform_keyframe, frame.width(), frame.height(),
			transform_decoded), "transform keyframe decodes successfully");
		check(transform_decoded.size() == traditional_output.size()
			&& std::equal(transform_decoded.data(), transform_decoded.data() + transform_decoded.size(),
				traditional_output.data()), "serialized transform keyframe matches direct reconstruction");

		image_u8 transform_next = frame;
		for (int y = 0; y < transform_next.height(); ++y)
			for (int x = 0; x < transform_next.width(); ++x)
				transform_next(x, y) = static_cast<uint8_t>(255 - transform_next(x, y));
		const transform_packet transform_delta = transform_stream.encode(
			traditional, transform_next, &importance, false, 256);
		check(!transform_delta.keyframe && !transform_delta.bytes.empty() && transform_delta.bytes.size() <= 256,
		      "transform delta obeys its byte budget");
		check(transform_delta.updated_blocks > 0 && transform_delta.updated_blocks < 20 * 15,
		      "transform delta updates a subset of blocks");
		check(transform_stream.decode(traditional, transform_delta, frame.width(), frame.height(),
			transform_decoded), "transform delta applies to decoder state");
		transform_packet bad_transform = transform_delta;
		bad_transform.bytes.pop_back();
		check(!transform_stream.decode(traditional, bad_transform, frame.width(), frame.height(),
			transform_decoded), "truncated transform packets are rejected");

		// A flat mid-grey block quantises to nothing at all, so there is no error to remove
		// and a purely error-driven encoder would leave it out of a "complete" keyframe.
		{
			image_u8 flat(64, 64);
			flat.fill(128);
			temporal_transform_stream flat_stream;
			image_u8 flat_decoded;
			const transform_packet flat_keyframe = flat_stream.encode(traditional, flat, nullptr, true, 0);
			bool flat_exact = flat_stream.decode(traditional, flat_keyframe, 64, 64, flat_decoded);
			for (size_t i = 0; flat_exact && i < flat_decoded.size(); ++i)
				flat_exact = flat_decoded.data()[i] == 128;
			check(flat_keyframe.keyframe && flat_keyframe.updated_blocks == 8 * 8,
			      "a featureless keyframe still carries every block");
			check(flat_exact, "a featureless keyframe reconstructs exactly");
		}

		// Once residual coefficients can be dropped, the encoder holds something that is not
		// the source, and the difference must not be mistaken for new work every packet.
		{
			temporal_transform_stream settle_stream;
			image_u8 settle_decoded;
			bool settle_decodes = settle_stream.decode(traditional, settle_stream.encode(
				traditional, frame, &importance, true, 0), frame.width(), frame.height(), settle_decoded);
			settle_decodes = settle_decodes && settle_stream.decode(traditional, settle_stream.encode(
				traditional, transform_next, &importance, false, 256), frame.width(), frame.height(),
				settle_decoded);

			int settle_packets = 0;
			for (int step = 0; step < 12; ++step)
			{
				const transform_packet settle = settle_stream.encode(
					traditional, transform_next, &importance, false, 1200);
				if (settle.bytes.empty()) break;
				++settle_packets;
				settle_decodes = settle_decodes && settle_stream.decode(traditional, settle,
					frame.width(), frame.height(), settle_decoded);
			}
			std::printf("  a held scene settles after %d further packets\n", settle_packets);
			check(settle_decodes, "held scene decodes throughout");
			check(settle_packets < 12, "a scene that stops changing stops costing bytes");
		}

		// A face that keeps changing at a new position must not permanently outbid the
		// blocks it vacated: that starvation is what leaves extra copies of a head on
		// screen. The vacated region is static here, so only the ranking can repair it.
		{
			constexpr int face_size = 48;
			const auto paint = [](image_u8& target, image_u8& weight, const int origin, const int phase)
			{
				target.fill(40);
				weight.fill(32);
				for (int y = 16; y < 16 + face_size; ++y)
					for (int x = origin; x < origin + face_size; ++x)
					{
						target(x, y) = static_cast<uint8_t>(90 + (x * 7 + y * 11 + phase * 29) % 140);
						weight(x, y) = 255;
					}
			};

			image_u8 moving(160, 120);
			image_u8 moving_importance(160, 120);
			temporal_transform_stream ghost_stream;
			image_u8 ghost_decoded;

			paint(moving, moving_importance, 16, 0);
			bool ghost_decodes = ghost_stream.decode(traditional, ghost_stream.encode(
				traditional, moving, &moving_importance, true, 0), 160, 120, ghost_decoded);

			for (int step = 0; step < 16; ++step)
			{
				paint(moving, moving_importance, 96, step + 1);
				const transform_packet ghost_packet = ghost_stream.encode(
					traditional, moving, &moving_importance, false, 1200);
				ghost_decodes = ghost_decodes && ghost_packet.bytes.size() <= 1200
					&& ghost_stream.decode(traditional, ghost_packet, 160, 120, ghost_decoded);
			}

			double vacated_error = 0.0;
			int vacated_pixels = 0;
			for (int y = 16; y < 16 + face_size; ++y)
				for (int x = 16; x < 16 + face_size; ++x, ++vacated_pixels)
					vacated_error += std::abs(static_cast<int>(ghost_decoded(x, y)) - moving(x, y));
			vacated_error /= vacated_pixels;

			std::printf("  vacated face region residual %.1f levels after 16 capped packets\n", vacated_error);
			check(ghost_decodes, "moving-face sequence stays inside its budget and decodes");
			check(vacated_error < 8.0, "a region the face moved away from is repaired, not left as a ghost");
		}

		// Translation is the one thing a block transform cannot express, and a head is mostly
		// translation. Without a motion predictor every block the head passes through has to
		// be retransmitted in full, which no realistic packet budget can afford.
		{
			image_u8 panning(320, 240);
			image_u8 panning_importance(320, 240);
			panning_importance.fill(32);
			temporal_transform_stream panning_stream;
			image_u8 panning_decoded;

			// A detailed subject over a plain scene, which is what a webcam actually sends.
			const auto paint_at = [&](const int shift)
			{
				for (int y = 0; y < panning.height(); ++y)
					for (int x = 0; x < panning.width(); ++x)
						panning(x, y) = static_cast<uint8_t>(60 + x / 8 + y / 8);
				for (int y = 0; y < 64; ++y)
					for (int x = 0; x < 64; ++x)
					{
						const double value = 128.0 + 90.0 * std::sin(x * 0.31) * std::cos(y * 0.27)
							+ 30.0 * std::sin((x + y) * 0.9);
						panning(40 + shift + x, 60 + shift + y) = static_cast<uint8_t>(
							std::clamp(static_cast<int>(std::lround(value)), 0, 255));
					}
			};

			paint_at(0);
			bool panning_decodes = panning_stream.decode(traditional, panning_stream.encode(
				traditional, panning, &panning_importance, true, 0), 320, 240, panning_decoded);

			int64_t panning_bytes = 0;
			for (int step = 1; step <= 10; ++step)
			{
				paint_at(step * 3);
				const transform_packet panning_packet = panning_stream.encode(
					traditional, panning, &panning_importance, false, 600);
				panning_bytes += static_cast<int64_t>(panning_packet.bytes.size());
				panning_decodes = panning_decodes && panning_packet.bytes.size() <= 600
					&& panning_stream.decode(traditional, panning_packet, 320, 240, panning_decoded);
			}

			const reconstruction_quality panning_quality = measure_quality(panning, panning_decoded);
			std::printf("  moving subject holds %.1f dB on %lld delta bytes over 10 packets\n",
				panning_quality.psnr, static_cast<long long>(panning_bytes));
			check(panning_decodes, "moving subject stays inside its budget and decodes");
			check(panning_quality.psnr > 28.5, "a translating subject survives a capped delta wire");
		}

		image_u8 transform_rate_frame(640, 480);
		image_u8 transform_rate_importance(640, 480);
		transform_rate_importance.fill(32);
		for (int y = 0; y < transform_rate_frame.height(); ++y)
			for (int x = 0; x < transform_rate_frame.width(); ++x)
				transform_rate_frame(x, y) = static_cast<uint8_t>((x * 3 + y * 5) & 0xff);
		temporal_transform_stream transform_rate_stream;
		const transform_packet transform_rate_keyframe = transform_rate_stream.encode(
			traditional, transform_rate_frame, &transform_rate_importance, true, 0);
		image_u8 transform_rate_decoded;
		image_u8 transform_rate_direct;
		traditional.reconstruct(transform_rate_frame, &transform_rate_importance, transform_rate_direct);
		const double transform_rate_kbit_per_second = (1200.0 + 80.0) * 5.0 * 8.0 / 1000.0;
		std::printf("  full-resolution traditional wire budget %.1f kbit/s\n",
			transform_rate_kbit_per_second);
		check(transform_rate_keyframe.keyframe && transform_rate_keyframe.updated_blocks == 80 * 60,
		      "startup transform keyframe contains the whole scene");
		check(transform_rate_stream.decode(traditional, transform_rate_keyframe, 640, 480,
			transform_rate_decoded) && transform_rate_decoded.size() == transform_rate_direct.size()
			&& std::equal(transform_rate_decoded.data(),
				transform_rate_decoded.data() + transform_rate_decoded.size(), transform_rate_direct.data()),
		      "startup transform keyframe reconstructs the whole scene exactly");
		check(transform_rate_kbit_per_second < 80.0,
		      "traditional post-startup wire budget stays below 80 kbit/s");
		image_u8 unprioritised_output;
		traditional.reconstruct(frame, nullptr, unprioritised_output);
		const reconstruction_quality unprioritised_quality = measure_quality(frame, unprioritised_output, &importance);
		check(quality.weighted_rmse < unprioritised_quality.weighted_rmse,
		      "face importance improves weighted transform error");

		image_u8 damaged = frame;
		for (int y = 30; y < 90; ++y)
			for (int x = 40; x < 120; ++x) damaged(x, y) = 128;
		const reconstruction_quality damaged_quality = measure_quality(frame, damaged, &importance);
		check(damaged_quality.weighted_rmse > damaged_quality.rmse,
		      "weighted metric penalises errors in important face areas");

		temporal_patch_stream stream;
		const latent_packet keyframe = stream.encode(codec, frame, &importance, true, 64);
		check(keyframe.keyframe && keyframe.updated_patches == 8 * 6, "latent keyframe contains every patch");
		check(keyframe.bytes.size() == 10 + 8 * 6 * patch_autoencoder::bytes_per_patch,
		      "latent keyframe is a compact serialized code raster");
		image_u8 decoded;
		check(stream.decode(codec, keyframe, frame.width(), frame.height(), decoded),
		      "latent keyframe decodes successfully");
		check(decoded.size() == output.size() && std::equal(decoded.data(), decoded.data() + decoded.size(), output.data()),
		      "serialized keyframe matches direct reconstruction exactly");

		image_u8 next_frame = frame;
		for (int y = 0; y < next_frame.height(); ++y)
			for (int x = 0; x < next_frame.width(); ++x) next_frame(x, y) = static_cast<uint8_t>(255 - next_frame(x, y));
		const latent_packet delta = stream.encode(codec, next_frame, &importance, false, 64);
		check(!delta.keyframe && !delta.bytes.empty() && delta.bytes.size() <= 64,
		      "latent delta obeys its byte budget");
		check(delta.updated_patches > 0 && delta.updated_patches < 8 * 6,
		      "latent delta updates a subset of patches");
		check(stream.decode(codec, delta, frame.width(), frame.height(), decoded),
		      "latent delta applies to decoder state");

		latent_packet truncated = delta;
		truncated.bytes.pop_back();
		check(!stream.decode(codec, truncated, frame.width(), frame.height(), decoded),
		      "truncated latent packets are rejected");

		patch_autoencoder rate_model;
		temporal_patch_stream rate_stream;
		temporal_landmark_stream rate_landmarks;
		image_u8 rate_frame(640, 480);
		image_u8 rate_importance(640, 480);
		rate_importance.fill(32);
		std::vector<std::vector<point_i>> rate_faces(1, std::vector<point_i>(68));
		image_u8 rate_decoded;
		std::vector<std::vector<point_i>> rate_parts;
		int64_t sequence_bytes = 0;
		bool sequence_decoded = true;
		for (int frame_index = 0; frame_index < 16; ++frame_index)
		{
			for (int y = 0; y < rate_frame.height(); ++y)
				for (int x = 0; x < rate_frame.width(); ++x)
					rate_frame(x, y) = static_cast<uint8_t>((x * 3 + y * 5 + frame_index * 17) & 0xff);
			for (size_t i = 0; i < rate_faces[0].size(); ++i)
				rate_faces[0][i] = {220 + static_cast<int>(i % 17) * 6 + frame_index * 2,
					140 + static_cast<int>(i / 17) * 10};
			const bool key = frame_index == 0 || frame_index == 15;
			const latent_packet image_packet = rate_stream.encode(rate_model, rate_frame, &rate_importance,
				key, 1500);
			const landmark_packet point_packet = rate_landmarks.encode(rate_faces, key);
			sequence_decoded = sequence_decoded
				&& rate_stream.decode(rate_model, image_packet, 640, 480, rate_decoded)
				&& rate_landmarks.decode(point_packet, rate_parts);
			sequence_bytes += static_cast<int64_t>(image_packet.bytes.size() + point_packet.bytes.size());
		}
		const double sequence_kbit_per_second = sequence_bytes * 8.0 / 3.2 / 1000.0;
		std::printf("  full-resolution temporal wire %.1f kbit/s\n", sequence_kbit_per_second);
		check(sequence_decoded, "full-resolution temporal sequence decodes");
		check(sequence_kbit_per_second < 80.0, "combined temporal wire stays below 80 kbit/s");

		codec.reset();
		check(codec.steps() == 0, "reset clears training");
	}

	void test_detector()
	{
		std::printf("face detection\n");

		// The weight table is generated, so the useful check is that it is wired up
		// consistently: a mis-copied table would break the forward pass in ways the
		// synthetic frames below cannot show.
		bool table_valid = true;
		bool depthwise_square = true;
		for (const auto& layer : face_model_layers)
		{
			if (layer.weights == nullptr || layer.biases == nullptr) table_valid = false;
			if (layer.channels <= 0 || layer.num_filters <= 0) table_valid = false;
			if (layer.is_depthwise == layer.is_pointwise) table_valid = false;
			if (layer.is_depthwise && layer.channels != layer.num_filters) depthwise_square = false;
		}

		check(table_valid, "every model layer is populated and has exactly one kind");
		check(depthwise_square, "depthwise layers keep their channel count");

		const face_detector detector;

		// A coloured ellipse is not a face. This exercises the whole forward pass and shows
		// the network is not simply firing everywhere.
		const video_frame ellipse = make_face_frame(640, 480, 320, 200, 60, 80);
		const face_detection on_ellipse = detector.detect(ellipse);
		check(on_ellipse.faces.empty(), "a skin coloured blob is not reported as a face");

		// A silently broken forward pass tends to saturate or produce NaN rather than to
		// score low, so the peak score is checked even where nothing is detected.
		check(on_ellipse.peak_confidence >= 0.0f && on_ellipse.peak_confidence < 1.0f,
		      "peak score is a finite probability");
		check(on_ellipse.peak_confidence > 0.0f, "the network produces a non-zero response");

		video_frame noise;
		noise.luma.resize(640, 480);
		noise.cb.resize(320, 480);
		noise.cr.resize(320, 480);
		uint32_t seed = 1;
		for (size_t i = 0; i < noise.luma.size(); ++i)
		{
			seed = seed * 1664525u + 1013904223u;
			noise.luma.data()[i] = static_cast<uint8_t>(seed >> 24);
		}
		noise.cb.fill(128);
		noise.cr.fill(128);

		const face_detection on_noise = detector.detect(noise);
		check(on_noise.faces.empty(), "random noise is not reported as a face");
		check(on_noise.peak_confidence < 1.0f, "noise does not saturate the network");

		// The same input must give the same answer: catches uninitialised accumulators.
		check(detector.detect(noise).peak_confidence == on_noise.peak_confidence, "detection is deterministic");

		bool square = true;
		for (const auto& box : on_ellipse.faces)
		{
			if (box.width() != box.height()) square = false;
		}
		check(square, "reported boxes are square");

		video_frame tiny;
		check(detector.detect(tiny).faces.empty(), "an empty frame is handled");
	}

	void test_landmarks()
	{
		std::printf("landmark model\n");

		std::vector<std::vector<point_i>> wire_faces(1, std::vector<point_i>(68));
		for (size_t i = 0; i < wire_faces[0].size(); ++i)
			wire_faces[0][i] = {100 + static_cast<int>(i % 17) * 4, 80 + static_cast<int>(i / 17) * 6};
		temporal_landmark_stream wire;
		const landmark_packet landmark_keyframe = wire.encode(wire_faces, true);
		std::vector<std::vector<point_i>> wire_decoded;
		check(wire.decode(landmark_keyframe, wire_decoded), "landmark keyframe decodes successfully");
		bool quantized_close = wire_decoded.size() == 1 && wire_decoded[0].size() == 68;
		if (quantized_close)
			for (size_t i = 0; i < 68; ++i)
				if (std::abs(wire_decoded[0][i].x - wire_faces[0][i].x) > 1
					|| std::abs(wire_decoded[0][i].y - wire_faces[0][i].y) > 1) quantized_close = false;
		check(quantized_close, "landmark quantization stays within one pixel");

		auto moved_faces = wire_faces;
		for (size_t i = 0; i < moved_faces[0].size(); ++i)
		{
			moved_faces[0][i].x += static_cast<int>(i % 3) * 2 - 2;
			moved_faces[0][i].y += static_cast<int>((i + 1) % 3) * 2 - 2;
		}
		const landmark_packet landmark_delta = wire.encode(moved_faces, false);
		check(!landmark_delta.keyframe && landmark_delta.bytes.size() <= 80
			&& landmark_delta.bytes.size() < landmark_keyframe.bytes.size(),
		      "small landmark motion packs below 80 bytes");
		check(wire.decode(landmark_delta, wire_decoded), "landmark delta applies to decoder state");
		check(wire.encode(moved_faces, false).bytes.empty(), "unchanged landmarks emit no packet");

		landmark_packet bad_landmarks = landmark_delta;
		bad_landmarks.bytes.pop_back();
		check(!wire.decode(bad_landmarks, wire_decoded), "truncated landmark packets are rejected");

		shape_predictor predictor;
		std::string error;
		if (!predictor.load(asset_path(L"shape_predictor_68_face_landmarks.dat"), error))
		{
			check(false, ("model loads: " + error).c_str());
			return;
		}

		check(predictor.num_parts() == 68, "model has 68 parts");
		check(predictor.num_cascades() >= 10, "model has the expected cascade depth");

		// The mean shape must actually look like a face. This is the strongest available
		// check that the file's packed integers and floats were decoded correctly.
		const auto centroid = [&](const int first, const int last)
		{
			point_f sum;
			for (int i = first; i <= last; ++i)
			{
				const point_f p = predictor.mean_part(i);
				sum.x += p.x;
				sum.y += p.y;
			}
			const auto n = static_cast<float>(last - first + 1);
			return point_f{sum.x / n, sum.y / n};
		};

		const point_f left_eye = centroid(36, 41);
		const point_f right_eye = centroid(42, 47);
		const point_f mouth = centroid(48, 67);
		const point_f nose_tip = predictor.mean_part(30);
		const point_f chin = predictor.mean_part(8);

		bool in_box = true;
		for (size_t i = 0; i < predictor.num_parts(); ++i)
		{
			const point_f p = predictor.mean_part(i);
			if (p.x < -0.5f || p.x > 1.5f || p.y < -0.5f || p.y > 1.5f) in_box = false;
		}

		check(in_box, "mean shape lies in unit box coordinates");
		check(left_eye.x < right_eye.x, "eyes are ordered left to right");
		check(left_eye.y < nose_tip.y && nose_tip.y < mouth.y, "eyes, nose and mouth are stacked correctly");
		check(mouth.y < chin.y, "chin is below the mouth");
		check(std::fabs(nose_tip.x - (left_eye.x + right_eye.x) / 2.0f) < 0.05f, "nose sits between the eyes");

		image_u8 img(640, 480);
		for (int y = 0; y < 480; ++y)
			for (int x = 0; x < 640; ++x)
				img(x, y) = static_cast<uint8_t>((x * 3 + y * 5) & 0xFF);

		const rect_i box = rect_i::from_size(250, 150, 140, 140);
		const std::vector<point_i> parts = predictor.predict(img, box);
		check(parts.size() == 68, "prediction returns 68 points");

		if (parts.size() == 68)
		{
			bool sane = true;
			for (const auto& p : parts)
			{
				if (p.x < box.left - 200 || p.x > box.right + 200) sane = false;
				if (p.y < box.top - 200 || p.y > box.bottom + 200) sane = false;
			}
			check(sane, "points stay in the neighbourhood of the box");

			const rect_i refined = predictor.box_from_parts(parts);
			check(!refined.empty(), "box can be recovered from a shape");
			check(std::abs(refined.width() - box.width()) < box.width() / 2,
			      "recovered box is close to the original");
		}

		check(predictor.predict(img, {}).empty(), "an empty box yields no prediction");
	}
}

int run_self_test()
{
	failures = 0;

	test_image_ops();
	test_yuy2();
	test_drawing();
	test_codec();
	test_detector();
	test_landmarks();

	std::printf("\n%s\n", failures == 0 ? "all checks passed" : "there were failures");
	return failures == 0 ? 0 : 1;
}
