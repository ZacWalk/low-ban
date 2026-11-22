// Headless checks for everything that does not need a camera or a window.
// Run with: low-ban-64d.exe /test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "baseline.h"
#include "capture.h"
#include "codec.h"
#include "engine.h"
#include "entropy.h"
#include "face.h"
#include "image.h"
#include "landmarks.h"
#include "session.h"
#include "warp.h"

namespace
{
	int failures = 0;
	int checks = 0;
	int skipped = 0;
	constexpr double pi = 3.14159265358979323846;

	void check(const bool condition, const char* description)
	{
		++checks;
		if (!condition) ++failures;
		std::printf("  [%s] %s\n", condition ? "pass" : "FAIL", description);
	}

	// For checks that need a platform component this machine may not have, such as Media
	// Foundation's H.264 transforms on a server without the media feature.
	void skip(const char* description, const std::string& reason)
	{
		++skipped;
		std::printf("  [skip] %s: %s\n", description, reason.c_str());
	}

	// A frame with a skin coloured ellipse on a neutral grey background.
	video_frame make_face_frame(const int width, const int height, const int cx, const int cy, const int rx,
	                            const int ry)
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

	// A plausible face's five landmarks - libmug's template face - centred on (cx, cy) and `size`
	// pixels across, smiling `smile` pixels: the mouth corners that much further apart and a third
	// of it higher.
	std::vector<point_i> synthetic_face(const int cx, const int cy, const int size, const int smile = 0)
	{
		// The template's landmarks about its centre, in 112ths of the face.
		constexpr double template_points[5][2] = {{-17.74, -20.20},
		                                          {17.50, -20.40},
		                                          {0.0, -0.16},
		                                          {-14.48, 20.47},
		                                          {14.70, 20.30}};
		const double s = size / 112.0;
		std::vector<point_i> p;
		for (size_t i = 0; i < face_point_count; ++i)
		{
			double x = template_points[i][0] * s;
			double y = template_points[i][1] * s;
			if (i == mouth_left || i == mouth_right)
			{
				x += (i == mouth_left ? -smile : smile) / 2.0;
				y -= smile / 3.0;
			}
			p.push_back({cx + static_cast<int>(std::lround(x)), cy + static_cast<int>(std::lround(y))});
		}
		return p;
	}

	// Detail everywhere, so a wrong prediction always costs bytes or decibels.
	image_u8 textured_scene(const int width, const int height)
	{
		image_u8 img(width, height);
		for (int y = 0; y < height; ++y)
			for (int x = 0; x < width; ++x)
			{
				const double value = 128.0 + 50.0 * std::sin(x * 0.21) * std::cos(y * 0.17) +
				                     30.0 * std::sin((x + 2 * y) * 0.07) + (x * 7 + y * 13) % 17 - 8;
				img(x, y) = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
			}
		return img;
	}

	video_frame grey_frame(const image_u8& luma)
	{
		video_frame frame;
		frame.luma = luma;
		frame.cb.resize(luma.width() / 2, luma.height());
		frame.cr.resize(luma.width() / 2, luma.height());
		frame.cb.fill(128);
		frame.cr.fill(128);
		return frame;
	}

	int64_t doubled_area(const point_i& a, const point_i& b, const point_i& c)
	{
		return static_cast<int64_t>(b.x - a.x) * (c.y - a.y) - static_cast<int64_t>(b.y - a.y) * (c.x - a.x);
	}

	bool same_image(const image_u8& a, const image_u8& b)
	{
		return a.width() == b.width() && a.height() == b.height() &&
		       std::equal(a.data(), a.data() + a.size(), b.data());
	}

	void test_image_ops()
	{
		std::printf("image operations\n");

		image_u8 src(64, 64);
		for (int y = 0; y < 64; ++y)
			for (int x = 0; x < 64; ++x) src(x, y) = static_cast<uint8_t>(x < 32 ? 40 : 200);

		image_u8 small(16, 16);
		resize_bilinear(src, small);
		check(small(2, 8) < 100 && small(13, 8) > 150, "bilinear resize preserves a step edge");

		image_u8 doubled(128, 128);
		resize_bicubic(src, doubled);
		image_u8 ramp(32, 32);
		image_u8 ramp_doubled(64, 64);
		for (int y = 0; y < 32; ++y)
			for (int x = 0; x < 32; ++x) ramp(x, y) = static_cast<uint8_t>(20 + x * 6);
		resize_bicubic(ramp, ramp_doubled);
		bool ramp_kept = true;
		for (int x = 4; x < 60; ++x)
			ramp_kept = ramp_kept && std::abs(ramp_doubled(x, 32) - (20.0 + ((x + 0.5) / 2.0 - 0.5) * 6.0)) <= 1.0;
		check(doubled(20, 64) == 40 && doubled(108, 64) == 200 && doubled(62, 64) < doubled(66, 64) && ramp_kept,
		      "bicubic enlarging keeps flat regions and ramps exact and edges in place");

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
		std::printf("yuy2 and nv12 decoding\n");

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
		check(frame.luma(3, 2) == 23 && frame.cb(1, 2) == 100 && frame.cr(1, 2) == 200,
		      "planes carry the right values");

		video_frame flipped;
		check(decode_yuy2(flipped, buffer.data(), buffer.size(), width, height, -stride), "bottom-up decode succeeds");
		check(flipped.luma(3, 1) == frame.luma(3, 2), "negative stride is normalised to top-down");

		video_frame rejected;
		check(!decode_yuy2(rejected, buffer.data(), 4, width, height, stride), "short buffers are rejected");

		std::vector<uint8_t> packed;
		encode_yuy2(frame, packed);
		check(packed == buffer, "packing planes back into YUY2 reproduces the camera buffer");

		// Two luma rows of padding before the chroma plane, as a decoder aligning its height
		// would leave.
		constexpr int nv12_stride = 10;
		constexpr int plane_rows = height + 2;
		std::vector<uint8_t> nv12(static_cast<size_t>(nv12_stride) * (plane_rows + height / 2), 0);
		for (int y = 0; y < height; ++y)
			for (int x = 0; x < width; ++x)
				nv12[static_cast<size_t>(y) * nv12_stride + x] = static_cast<uint8_t>(y * 10 + x);
		for (int y = 0; y < height / 2; ++y)
			for (int x = 0; x < width / 2; ++x)
			{
				nv12[static_cast<size_t>(plane_rows + y) * nv12_stride + x * 2] = static_cast<uint8_t>(60 + y);
				nv12[static_cast<size_t>(plane_rows + y) * nv12_stride + x * 2 + 1] = static_cast<uint8_t>(190 + y);
			}
		video_frame from_nv12;
		check(decode_nv12(from_nv12, nv12.data(), nv12.size(), width, height, nv12_stride, plane_rows) &&
		          from_nv12.luma(5, 3) == 35 && from_nv12.cb(1, 3) == 61 && from_nv12.cr(1, 0) == 190,
		      "NV12 decodes into the same planes, chroma rows repeated");
		check(!decode_nv12(from_nv12, nv12.data(), nv12.size() - 1, width, height, nv12_stride, plane_rows),
		      "short NV12 buffers are rejected");
	}

	void test_entropy()
	{
		std::printf("range coder\n");

		// A skewed source on two models, raw bits and integers, in a fixed pseudo-random order.
		uint32_t seed = 12345;
		const auto next = [&seed]
		{
			seed = seed * 1664525u + 1013904223u;
			return seed >> 8;
		};
		std::vector<int> bits;
		std::vector<uint32_t> raw;
		const std::vector<uint32_t> values = {0, 1, 2, 5, 17, 300, 70000};
		bit_model models[2];
		bit_model prefix[4];
		signed_models numbers;
		range_encoder out;
		for (int i = 0; i < 4000; ++i)
		{
			bits.push_back(next() % 10 == 0 ? 1 : 0);
			out.encode(models[i & 1], bits.back());
			if (i % 50 == 0)
			{
				raw.push_back(next() & 0xfff);
				out.encode_bypass(raw.back(), 12);
			}
		}
		for (const uint32_t value : values)
		{
			encode_golomb(out, prefix, 4, value);
			encode_signed(out, numbers, -static_cast<int>(value));
		}
		const size_t predicted = out.size_if_finished();
		const std::vector<uint8_t> bytes = out.finish();
		std::printf("  4,000 decisions at 10%% ones, %zu raw bits and %zu integers in %zu bytes\n", raw.size() * 12,
		            values.size() * 2, bytes.size());
		check(bytes.size() == predicted, "the coder knows its exact length before it finishes");
		check(bytes.size() < 4000 * 0.6 / 8 + raw.size() * 12 / 8 + 40, "a skewed decision costs well under a bit");

		const auto replay = [&](const std::vector<uint8_t>& input)
		{
			range_decoder in(input.data(), input.size());
			bit_model decoded_models[2];
			bit_model decoded_prefix[4];
			signed_models decoded_numbers;
			bool same = true;
			size_t raw_index = 0;
			for (int i = 0; i < 4000; ++i)
			{
				same = same && in.decode(decoded_models[i & 1]) == bits[static_cast<size_t>(i)];
				if (i % 50 == 0) same = same && in.decode_bypass(12) == raw[raw_index++];
			}
			for (const uint32_t value : values)
			{
				same = same && decode_golomb(in, decoded_prefix, 4) == value;
				same = same && decode_signed(in, decoded_numbers) == -static_cast<int>(value);
			}
			return same && in.finished();
		};
		check(replay(bytes), "decisions, raw bits and integers round trip, consuming every byte");
		std::vector<uint8_t> truncated(bytes.begin(), bytes.end() - 1);
		std::vector<uint8_t> padded = bytes;
		padded.push_back(0);
		check(!replay(truncated) && !replay(padded), "a truncated or padded stream is detected");
	}

	void test_codec()
	{
		std::printf("low-ban codec\n");

		image_u8 frame(160, 120);
		for (int y = 0; y < 120; ++y)
			for (int x = 0; x < 160; ++x)
			{
				const int local_x = x % 20;
				const int local_y = y % 20;
				const int block = x / 20 + y / 20;
				const double value = 40 + block % 5 * 25 + local_x * (block % 3 + 1) + local_y * ((block + 1) % 3 + 1) +
				                     30.0 * std::sin(local_x * 0.7 + block) * std::sin(local_y * 0.5);
				frame(x, y) = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
			}

		image_u8 importance(frame.width(), frame.height());
		importance.fill(32);
		for (int y = 30; y < 90; ++y)
			for (int x = 40; x < 120; ++x) importance(x, y) = 255;

		image_u8 direct;
		intra_picture(frame, &importance, startup_quality, direct);
		const reconstruction_quality quality = measure_quality(frame, direct, &importance);
		check(direct.width() == frame.width() && direct.height() == frame.height() && quality.psnr > 25.0,
		      "intra coding keeps the frame size and a usable picture");

		lowban_wire wire;
		const lowban_packet keyframe = wire.encode(frame, &importance, {}, true, 0, startup_quality);
		image_u8 decoded;
		check(keyframe.keyframe && keyframe.updated_blocks == 20 * 15, "a keyframe contains every block");
		check(wire.decode(keyframe, decoded) && same_image(decoded, direct),
		      "the serialized keyframe matches direct intra coding byte for byte");

		image_u8 inverted = frame;
		for (size_t i = 0; i < inverted.size(); ++i) inverted.data()[i] = static_cast<uint8_t>(255 - frame.data()[i]);
		const lowban_packet delta = wire.encode(inverted, &importance, {}, false, 256, startup_quality);
		check(!delta.keyframe && !delta.bytes.empty() && delta.bytes.size() <= 256, "a delta obeys its byte budget");
		check(delta.updated_blocks > 0 && delta.updated_blocks < 20 * 15, "a delta updates a subset of blocks");

		// Every damaged copy is turned away, and the decoder is left as it was: the real packet
		// still applies afterwards and gives the picture a clean decoder gives.
		lowban_packet truncated = delta;
		truncated.bytes.pop_back();
		lowban_packet padded = delta;
		padded.bytes.push_back(0);
		lowban_packet unknown_flag = delta;
		unknown_flag.bytes[0] = static_cast<uint8_t>(unknown_flag.bytes[0] | 8);
		lowban_packet foreign = delta;
		foreign.bytes[0] = static_cast<uint8_t>(foreign.bytes[0] ^ 0x10);
		lowban_packet warp_alone = delta;
		warp_alone.bytes[0] = static_cast<uint8_t>(warp_alone.bytes[0] | 4);
		check(!wire.decode(truncated, decoded) && !wire.decode(padded, decoded) &&
		          !wire.decode(unknown_flag, decoded) && !wire.decode(foreign, decoded) &&
		          !wire.decode(warp_alone, decoded),
		      "truncated, padded, unknown, foreign and contradictory packets are rejected");
		lowban_wire clean;
		image_u8 clean_decoded;
		clean.decode(clean.encode(frame, &importance, {}, true, 0, startup_quality), clean_decoded);
		check(wire.decode(delta, decoded) &&
		          clean.decode(clean.encode(inverted, &importance, {}, false, 256, startup_quality), clean_decoded) &&
		          same_image(decoded, clean_decoded),
		      "a rejected packet leaves the decoder untouched");

		// A flat mid-grey block quantises to nothing at all, and must still be in a keyframe.
		{
			image_u8 flat(64, 64);
			flat.fill(128);
			lowban_wire flat_wire;
			image_u8 flat_decoded;
			const lowban_packet flat_keyframe = flat_wire.encode(flat, nullptr, {}, true, 0, startup_quality);
			bool flat_exact = flat_wire.decode(flat_keyframe, flat_decoded);
			for (size_t i = 0; flat_exact && i < flat_decoded.size(); ++i) flat_exact = flat_decoded.data()[i] == 128;
			check(flat_keyframe.updated_blocks == 8 * 8 && flat_exact,
			      "a featureless keyframe carries every block and reconstructs exactly");
		}

		// The encoder holds what it sent, not the source, and must not mistake the difference
		// for new work every packet.
		{
			lowban_wire settle;
			image_u8 settle_decoded;
			bool settle_decodes = settle.decode(settle.encode(frame, &importance, {}, true, 0, startup_quality),
			                                    settle_decoded) &&
			                      settle.decode(settle.encode(inverted, &importance, {}, false, 256, startup_quality),
			                                    settle_decoded);
			int settle_packets = 0;
			for (int step = 0; step < 24; ++step)
			{
				const lowban_packet packet = settle.encode(inverted, &importance, {}, false, 1200, startup_quality);
				if (packet.bytes.empty()) break;
				++settle_packets;
				settle_decodes = settle_decodes && settle.decode(packet, settle_decoded);
			}
			std::printf("  a held scene settles after %d further packets\n", settle_packets);
			check(settle_decodes && settle_packets < 24, "a scene that stops changing stops costing bytes");
		}

		// A face that keeps changing at a new position must not permanently outbid the blocks it
		// vacated: that starvation is what leaves extra copies of a head on screen. The vacated
		// region is static here, so only the ranking can repair it.
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
			lowban_wire ghost;
			image_u8 ghost_decoded;
			paint(moving, moving_importance, 16, 0);
			bool ghost_decodes = ghost.decode(ghost.encode(moving, &moving_importance, {}, true, 0, startup_quality),
			                                  ghost_decoded);
			constexpr int packets = 40;
			for (int step = 0; step < packets; ++step)
			{
				paint(moving, moving_importance, 96, step + 1);
				const lowban_packet packet = ghost.encode(moving, &moving_importance, {}, false, 300, startup_quality);
				ghost_decodes = ghost_decodes && packet.bytes.size() <= 300 &&
				                (packet.bytes.empty() || ghost.decode(packet, ghost_decoded));
			}

			double vacated_error = 0.0;
			int vacated_pixels = 0;
			for (int y = 16; y < 16 + face_size; ++y)
				for (int x = 16; x < 16 + face_size; ++x, ++vacated_pixels)
					vacated_error += std::abs(static_cast<int>(ghost_decoded(x, y)) - moving(x, y));
			vacated_error /= vacated_pixels;

			std::printf("  vacated face region residual %.1f levels after %d 300-byte packets\n", vacated_error,
			            packets);
			check(ghost_decodes, "a moving-face sequence stays inside its budget and decodes");
			check(vacated_error < 8.0, "a region the face moved away from is repaired, not left as a ghost");
		}

		// Translation is the one thing a block transform cannot express, and a head is mostly
		// translation, often by part of a pixel.
		{
			image_u8 panning(320, 240);
			image_u8 panning_importance(320, 240);
			panning_importance.fill(32);
			lowban_wire panning_wire;
			image_u8 panning_decoded;

			// A detailed subject over a plain scene, drawn at a fractional offset.
			const auto paint_at = [&](const double shift)
			{
				for (int y = 0; y < panning.height(); ++y)
					for (int x = 0; x < panning.width(); ++x)
					{
						const double sx = x - 40 - shift;
						const double sy = y - 60 - shift;
						double value = 60 + x / 8 + y / 8;
						if (sx >= 0 && sx < 64 && sy >= 0 && sy < 64)
							value = 128.0 + 90.0 * std::sin(sx * 0.31) * std::cos(sy * 0.27) +
							        30.0 * std::sin((sx + sy) * 0.9);
						panning(x, y) = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
					}
			};

			paint_at(0);
			bool panning_decodes = panning_wire.decode(panning_wire.encode(panning, &panning_importance, {}, true, 0,
			                                                               startup_quality),
			                                           panning_decoded);
			int64_t panning_bytes = 0;
			for (int step = 1; step <= 10; ++step)
			{
				paint_at(step * 2.5);
				const lowban_packet packet = panning_wire.encode(panning, &panning_importance, {}, false, 600,
				                                                 startup_quality);
				panning_bytes += static_cast<int64_t>(packet.bytes.size());
				panning_decodes = panning_decodes && packet.bytes.size() <= 600 &&
				                  (packet.bytes.empty() || panning_wire.decode(packet, panning_decoded));
			}
			const reconstruction_quality panning_quality = measure_quality(panning, panning_decoded);
			image_u8 panning_intra;
			intra_picture(panning, &panning_importance, startup_quality, panning_intra);
			const double intra_psnr = measure_quality(panning, panning_intra).psnr;
			std::printf("  moving subject holds %.1f dB, intra coding %.1f dB, on %lld delta bytes over 10 packets\n",
			            panning_quality.psnr, intra_psnr, static_cast<long long>(panning_bytes));
			check(panning_decodes, "a moving subject stays inside its budget and decodes");
			check(panning_quality.psnr > intra_psnr - 1.5,
			      "a subject moving by fractions of a pixel stays near its intra quality on a capped wire");
		}

		// The live startup keyframe is the whole scene at the coded size.
		image_u8 startup_frame(640 / coded_scale, 480 / coded_scale);
		image_u8 startup_importance(startup_frame.width(), startup_frame.height());
		startup_importance.fill(32);
		for (int y = 0; y < startup_frame.height(); ++y)
			for (int x = 0; x < startup_frame.width(); ++x)
				startup_frame(x, y) = static_cast<uint8_t>((x * 3 + y * 5) & 0xff);
		lowban_wire startup_wire;
		const lowban_packet startup_keyframe = startup_wire.encode(startup_frame, &startup_importance, {}, true, 0,
		                                                           startup_quality);
		image_u8 startup_decoded;
		image_u8 startup_direct;
		intra_picture(startup_frame, &startup_importance, startup_quality, startup_direct);
		check(startup_keyframe.updated_blocks == 40 * 30 && startup_wire.decode(startup_keyframe, startup_decoded) &&
		          same_image(startup_decoded, startup_direct),
		      "the startup keyframe reconstructs the whole coded scene exactly");

		image_u8 unprioritised;
		intra_picture(frame, nullptr, startup_quality, unprioritised);
		check(quality.weighted_rmse < measure_quality(frame, unprioritised, &importance).weighted_rmse,
		      "face importance improves weighted error");

		image_u8 damaged = frame;
		for (int y = 30; y < 90; ++y)
			for (int x = 40; x < 120; ++x) damaged(x, y) = 128;
		const reconstruction_quality damaged_quality = measure_quality(frame, damaged, &importance);
		check(damaged_quality.weighted_rmse > damaged_quality.rmse,
		      "the weighted metric penalises errors in important face areas");
	}

	void test_loop_filter()
	{
		std::printf("loop filter\n");

		// A gentle ramp coded as one level per block: the steps between blocks are the codec's.
		image_u8 blocky(64, 16);
		image_u8 ramp(64, 16);
		for (int y = 0; y < 16; ++y)
			for (int x = 0; x < 64; ++x)
			{
				blocky(x, y) = static_cast<uint8_t>(100 + x / 8 * 6);
				ramp(x, y) = static_cast<uint8_t>(std::lround(97.5 + x * 0.75));
			}
		image_u8 smoothed = blocky;
		deblock_intra(smoothed, std::vector<uint8_t>(8 * 2, 18));
		check(std::abs(smoothed(32, 8) - smoothed(31, 8)) < 6 &&
		          measure_quality(ramp, smoothed).psnr > measure_quality(ramp, blocky).psnr,
		      "steps the quantiser made between blocks are smoothed towards the scene");

		image_u8 edge(32, 16);
		for (int y = 0; y < 16; ++y)
			for (int x = 0; x < 32; ++x) edge(x, y) = static_cast<uint8_t>(x < 16 ? 40 : 200);
		image_u8 kept = edge;
		deblock_intra(kept, std::vector<uint8_t>(4 * 2, 18));
		check(same_image(kept, edge), "a real edge on a block boundary is left alone");
	}

	void test_warp()
	{
		std::printf("landmark warp\n");

		const std::vector<point_i> square = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
		const std::vector<mesh_triangle> halves = delaunay_triangles(square);
		int64_t square_area = 0;
		bool positive = true;
		for (const mesh_triangle& t : halves)
		{
			const int64_t area = doubled_area(square[t.a], square[t.b], square[t.c]);
			positive = positive && area > 0;
			square_area += area;
		}
		check(halves.size() == 2 && positive && square_area == 200,
		      "a square triangulates into two positive triangles");

		std::vector<point_i> grid;
		for (int y = 0; y < 5; ++y)
			for (int x = 0; x < 5; ++x) grid.push_back({x * 10, y * 10});
		const std::vector<mesh_triangle> cells = delaunay_triangles(grid);
		int64_t grid_area = 0;
		for (const mesh_triangle& t : cells) grid_area += doubled_area(grid[t.a], grid[t.b], grid[t.c]);
		check(cells.size() == 32 && grid_area == 2 * 40 * 40, "a cocircular grid still tiles its hull exactly once");

		const std::vector<point_i> repeated = {{0, 0}, {10, 0}, {10, 0}, {0, 10}};
		check(delaunay_triangles(repeated).size() == 1, "repeated points do not make degenerate triangles");

		const std::vector<point_i> face = synthetic_face(160, 120, 100);
		const std::vector<point_i> mesh = face_mesh_points(face);
		check(mesh.size() == 17 && mesh[5].y < face[eye_left].y - 30 && mesh[16].y > face[mouth_left].y + 15,
		      "the mesh reaches from over the forehead to below the chin");

		const image_u8 reference = textured_scene(320, 240);
		image_u8 out;
		check(!warp_faces(reference, {face}, {face}, out) && same_image(out, reference),
		      "unchanged geometry copies the picture exactly");

		std::vector<point_i> moved = face;
		for (point_i& p : moved)
		{
			p.x += 6;
			p.y += 4;
		}
		check(warp_faces(reference, {face}, {moved}, out), "moved geometry warps the picture");
		bool translated = true;
		for (int y = moved[nose_tip].y - 6; y <= moved[nose_tip].y + 6; ++y)
			for (int x = moved[nose_tip].x - 6; x <= moved[nose_tip].x + 6; ++x)
				translated = translated && out(x, y) == reference(x - 6, y - 4);
		check(translated, "a whole-pixel translation moves the face exactly");
		check(out(5, 5) == reference(5, 5) && out(315, 235) == reference(315, 235),
		      "pixels outside the face mesh are untouched");

		image_u8 again;
		warp_faces(reference, {face}, {moved}, again);
		check(same_image(out, again), "the warp is deterministic");

		check(scale_geometry({{{-3, 5}, {4, 7}}}, 2) == face_geometry{{{-2, 2}, {2, 3}}},
		      "scaling geometry rounds towards negative infinity");
	}

	// Pictures generated by warping a textured scene with landmarks that move: the face drifts
	// and smiles while everything else stays still. The motion is made by the same warp the
	// codec predicts with, so this checks the mechanism, not how well a real face fits a
	// piecewise affine model.
	void test_warped_stream()
	{
		std::printf("low-ban warp prediction\n");

		const image_u8 scene = textured_scene(320, 240);
		const std::vector<point_i> rest = synthetic_face(160, 120, 120);
		const auto geometry_at = [&](const int step)
		{
			return face_geometry{synthetic_face(160 + step * 3, 120 + step, 120, step * 2)};
		};
		const auto frame_at = [&](const int step)
		{
			image_u8 frame;
			warp_faces(scene, {rest}, geometry_at(step), frame);
			return frame;
		};

		struct outcome
		{
			bool decodes = true;
			bool in_sync = true;
			int64_t bytes = 0;
			int warped = 0;
			double startup_psnr = 0.0;
			reconstruction_quality quality;
		};
		const auto run = [&](const bool with_geometry, const bool still_picture)
		{
			outcome result;
			lowban_wire wire;
			image_u8 decoded;
			const face_geometry start = with_geometry ? geometry_at(0) : face_geometry{};
			result.decodes = wire.decode(wire.encode(frame_at(0), nullptr, start, true, 0, startup_quality), decoded);
			// Settled first, so whatever follows is the motion's doing, not the startup's refinement.
			for (int settle = 0; settle < 24; ++settle)
			{
				const lowban_packet packet = wire.encode(frame_at(0), nullptr, start, false, 1200, startup_quality);
				if (packet.bytes.empty()) break;
				result.decodes = result.decodes && wire.decode(packet, decoded);
			}
			result.startup_psnr = measure_quality(frame_at(0), decoded).psnr;
			image_u8 source;
			for (int step = 1; step <= 8; ++step)
			{
				source = still_picture ? frame_at(0) : frame_at(step);
				const lowban_packet packet = wire.encode(source, nullptr,
				                                         with_geometry ? geometry_at(step) : face_geometry{}, false,
				                                         1200, startup_quality);
				result.bytes += static_cast<int64_t>(packet.bytes.size());
				if (packet.warped) ++result.warped;
				result.decodes = result.decodes && packet.bytes.size() <= 1200 &&
				                 (packet.bytes.empty() || wire.decode(packet, decoded));
				result.in_sync = result.in_sync && wire.in_sync();
			}
			result.quality = measure_quality(source, decoded);
			return result;
		};

		const outcome warped = run(true, false);
		const outcome plain = run(false, false);
		std::printf("  with landmarks %lld bytes at %.1f dB, without %lld bytes at %.1f dB\n",
		            static_cast<long long>(warped.bytes), warped.quality.psnr, static_cast<long long>(plain.bytes),
		            plain.quality.psnr);
		check(warped.decodes && plain.decodes, "both wires stay inside their budget and decode");
		check(warped.in_sync && plain.in_sync,
		      "after every packet the encoder holds exactly the picture, quantisers and landmarks the decoder does");
		check(warped.warped > 0, "moving landmarks are committed as a warp");
		check(warped.bytes < plain.bytes && warped.quality.psnr + 0.25 > plain.quality.psnr,
		      "the warp reaches the same picture for fewer bytes");

		// Landmarks that move over a still picture are wrong. The encoder refuses the warp, so
		// they cost only themselves, and the receiver keeps the picture it holds.
		const outcome misled = run(true, true);
		std::printf("  landmarks defying a still picture cost %lld bytes over 8 packets\n",
		            static_cast<long long>(misled.bytes));
		check(misled.decodes && misled.warped == 0 && misled.bytes <= 8 * 12 &&
		          misled.quality.psnr == misled.startup_psnr,
		      "landmarks that defy the picture are refused and cost only themselves");

		// A rejected packet must leave the decoder as it was: the next good packet then
		// decodes to exactly what a decoder that never saw the bad one produces.
		const auto delta_after_keyframe = [&](lowban_wire& target, image_u8& picture)
		{
			target.decode(target.encode(frame_at(0), nullptr, geometry_at(0), true, 0, startup_quality), picture);
			return target.encode(frame_at(2), nullptr, geometry_at(2), false, 1200, startup_quality);
		};
		lowban_wire damaged_wire;
		lowban_wire clean_wire;
		image_u8 damaged_picture;
		image_u8 clean_picture;
		const lowban_packet damaged_delta = delta_after_keyframe(damaged_wire, damaged_picture);
		const lowban_packet clean_delta = delta_after_keyframe(clean_wire, clean_picture);
		lowban_packet truncated = damaged_delta;
		truncated.bytes.pop_back();
		lowban_packet unknown_flag = damaged_delta;
		unknown_flag.bytes[0] = static_cast<uint8_t>(unknown_flag.bytes[0] | 8);
		lowban_packet no_landmarks = damaged_delta;
		no_landmarks.bytes[0] = static_cast<uint8_t>(no_landmarks.bytes[0] & ~2);
		check(damaged_delta.warped && !damaged_wire.decode(truncated, damaged_picture) &&
		          !damaged_wire.decode(unknown_flag, damaged_picture) &&
		          !damaged_wire.decode(no_landmarks, damaged_picture),
		      "truncated, unknown or contradictory warp packets are rejected");
		check(damaged_wire.decode(damaged_delta, damaged_picture) && clean_wire.decode(clean_delta, clean_picture) &&
		          same_image(damaged_picture, clean_picture) && damaged_wire.received() == geometry_at(2),
		      "a rejected warp leaves decoder state untouched");

		lowban_wire key_wire;
		lowban_packet warped_keyframe = key_wire.encode(frame_at(0), nullptr, geometry_at(0), true, 0, startup_quality);
		warped_keyframe.bytes[0] = static_cast<uint8_t>(warped_keyframe.bytes[0] | 6);
		image_u8 decoded;
		check(!key_wire.decode(warped_keyframe, decoded), "a keyframe claiming a warp is rejected");
	}

	void test_landmark_wire()
	{
		std::printf("landmarks on the wire\n");

		// A still picture, settled, so only the landmarks have anything to say.
		const image_u8 picture = textured_scene(320, 240);
		const std::vector<point_i> face = synthetic_face(100, 80, 60);
		lowban_wire wire;
		image_u8 decoded;
		bool decodes = wire.decode(wire.encode(picture, nullptr, {face}, true, 0, startup_quality), decoded);
		check(decodes && wire.received() == face_geometry{face} && wire.transmitted() == wire.received(),
		      "landmarks travel exactly in the keyframe");
		for (int step = 0; step < 24; ++step)
		{
			const lowban_packet settle = wire.encode(picture, nullptr, {face}, false, 1200, startup_quality);
			if (settle.bytes.empty()) break;
			decodes = decodes && wire.decode(settle, decoded);
		}
		check(decodes && wire.encode(picture, nullptr, {face}, false, 1200, startup_quality).bytes.empty(),
		      "a still scene with still landmarks sends nothing");

		const auto send = [&](const face_geometry& faces, size_t& bytes)
		{
			const lowban_packet packet = wire.encode(picture, nullptr, faces, false, 1200, startup_quality);
			bytes = packet.bytes.size();
			return packet.geometry && wire.decode(packet, decoded) && wire.received() == faces &&
			       wire.transmitted() == faces;
		};

		// A head that moves without changing expression is one shift and no residuals.
		std::vector<point_i> shifted = face;
		for (point_i& p : shifted)
		{
			p.x += 4;
			p.y -= 3;
		}
		size_t shift_bytes = 0;
		check(send({shifted}, shift_bytes) && shift_bytes <= 9,
		      "a rigid head movement costs a shift, not a point each");

		std::vector<point_i> smile = shifted;
		smile[mouth_left].x -= 2;
		smile[mouth_right].x += 2;
		size_t smile_bytes = 0;
		check(send({smile}, smile_bytes) && smile_bytes <= 10, "a smile costs its mouth corners");

		std::vector<point_i> jump = smile;
		jump[nose_tip].y += 40;
		size_t jump_bytes = 0;
		check(send({jump}, jump_bytes), "large motion still decodes exactly");

		const face_geometry two_faces = {jump, synthetic_face(220, 90, 50)};
		size_t pair_bytes = 0;
		size_t alone_bytes = 0;
		size_t none_bytes = 0;
		check(send(two_faces, pair_bytes) && send({two_faces[1]}, alone_bytes) && send({}, none_bytes),
		      "faces arriving and leaving re-send the layout exactly");
		std::printf("  shift %zu, smile %zu, jump %zu, two faces %zu bytes a packet\n", shift_bytes, smile_bytes,
		            jump_bytes, pair_bytes);
	}

	void test_landmark_filter()
	{
		std::printf("landmark smoothing\n");

		const face_geometry still = {synthetic_face(320, 240, 160)};
		landmark_filter filter;
		face_geometry out;
		for (int i = 0; i < 30; ++i) out = filter.apply(still, i / 30.0);
		check(out == still, "a still face passes through unchanged");

		filter.reset();
		int raw_span = 0;
		int smoothed_span = 0;
		int smoothed_low = 1 << 20;
		int smoothed_high = -(1 << 20);
		for (int i = 0; i < 60; ++i)
		{
			face_geometry jittered = still;
			for (point_i& p : jittered[0]) p.x += i % 2 == 0 ? 2 : -2;
			out = filter.apply(jittered, i / 30.0);
			if (i >= 30)
			{
				smoothed_low = std::min(smoothed_low, out[0][nose_tip].x);
				smoothed_high = std::max(smoothed_high, out[0][nose_tip].x);
			}
		}
		raw_span = 4;
		smoothed_span = smoothed_high - smoothed_low;
		std::printf("  jitter of %d pixels comes out as %d\n", raw_span, smoothed_span);
		check(smoothed_span < raw_span, "frame to frame jitter is suppressed");

		filter.reset();
		face_geometry stepped = still;
		for (point_i& p : stepped[0]) p.x += 40;
		for (int i = 0; i < 30; ++i) filter.apply(still, i / 30.0);
		for (int i = 30; i < 40; ++i) out = filter.apply(stepped, i / 30.0);
		check(std::abs(out[0][nose_tip].x - stepped[0][nose_tip].x) <= 2,
		      "real motion is followed within a third of a second");

		landmark_error error;
		check(compare_landmarks(still[0], still[0], error) && error.overall == 0.0 && error.mouth == 0.0,
		      "identical landmarks have no error");
		std::vector<point_i> offset = still[0];
		for (point_i& p : offset)
		{
			p.x += 3;
			p.y += 4;
		}
		const double ocular = std::hypot(static_cast<double>(still[0][eye_right].x - still[0][eye_left].x),
		                                 static_cast<double>(still[0][eye_right].y - still[0][eye_left].y));
		check(compare_landmarks(still[0], offset, error) && std::fabs(error.overall - 5.0 / ocular) < 1e-9 &&
		          std::fabs(error.mouth - 5.0 / ocular) < 1e-9,
		      "landmark error is normalised by the eye spacing");
		check(!compare_landmarks(still[0], {}, error), "a missing face cannot be scored");
	}

	void test_baseline()
	{
		std::printf("h264 baseline\n");

		h264_settings settings;
		settings.width = 320;
		settings.height = 240;
		settings.frames_per_second = packet_hz;
		settings.kbit_per_second = call_kbit_per_second;
		settings.face_offset = baseline_face_offset;
		h264_wire wire;
		std::string error;
		if (!wire.open(settings, error))
		{
			skip("H.264 baseline", error);
			return;
		}

		const image_u8 scene = textured_scene(320, 240);
		const std::vector<point_i> rest = synthetic_face(160, 120, 120);

		// The unit types in an Annex B stream, in order.
		const auto unit_types = [](const std::vector<uint8_t>& bytes)
		{
			std::vector<int> types;
			for (size_t i = 0; i + 3 < bytes.size(); ++i)
				if (bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1) types.push_back(bytes[i + 3] & 0x1f);
			return types;
		};

		bool round_trips = true;
		bool first_has_parameter_sets = false;
		bool only_needed_units = true;
		int64_t bytes_after_first = 0;
		image_u8 frame;
		image_u8 picture;
		for (int i = 0; i < 45; ++i)
		{
			warp_faces(scene, {rest}, {synthetic_face(160 + i, 120, 120, i % 10)}, frame);
			std::vector<uint8_t> bytes;
			round_trips = round_trips && wire.encode(frame, {rect_i::from_size(100 + i, 60, 120, 120)}, bytes, error) &&
			              !bytes.empty() && wire.decode(bytes, picture, error);
			const std::vector<int> types = unit_types(bytes);
			if (i == 0) first_has_parameter_sets = std::count(types.begin(), types.end(), 7) > 0;
			only_needed_units = only_needed_units && std::count(types.begin(), types.end(), 6) == 0 &&
			                    std::count(types.begin(), types.end(), 9) == 0;
			if (i > 0) bytes_after_first += static_cast<int64_t>(bytes.size());
		}
		const double rate = bytes_after_first * 8.0 / (44.0 / packet_hz) / 1000.0;
		const reconstruction_quality quality = measure_quality(frame, picture);
		std::printf("  %.1f kbit/s after the first frame, last frame %.1f dB\n", rate, quality.psnr);
		check(round_trips, "every frame crosses the H.264 wire and decodes");
		check(first_has_parameter_sets && only_needed_units,
		      "the H.264 wire carries the parameter sets but no delimiters or SEI");
		check(rate < call_kbit_per_second * 1.5, "the H.264 wire holds near its constant rate");
		check(quality.psnr > 24.0, "the H.264 receiver gets a usable picture");

		// At a fixed quantiser the face offset can only spend more bytes inside the box.
		const rect_i box = rect_i::from_size(100, 60, 120, 120);
		const auto box_error = [&](const image_u8& decoded)
		{
			double total = 0.0;
			for (int y = box.top; y <= box.bottom; ++y)
				for (int x = box.left; x <= box.right; ++x)
				{
					const double d = static_cast<double>(decoded(x, y)) - scene(x, y);
					total += d * d;
				}
			return total / (box.width() * box.height());
		};
		const auto still = [&](const int offset, image_u8& decoded, size_t& size)
		{
			h264_settings fixed = settings;
			fixed.kbit_per_second = 0;
			fixed.quantizer = 36;
			fixed.face_offset = offset;
			h264_wire one;
			std::vector<uint8_t> bytes;
			const bool ok = one.open(fixed, error) && one.encode(scene, {box}, bytes, error) &&
			                one.decode(bytes, decoded, error) && !decoded.empty();
			size = bytes.size();
			return ok && (offset == 0 || one.face_offset_active());
		};
		image_u8 plain_picture;
		image_u8 offset_picture;
		size_t plain_size = 0;
		size_t offset_size = 0;
		if (!still(0, plain_picture, plain_size) || !still(-12, offset_picture, offset_size))
		{
			skip("H.264 face offset", error.empty() ? "the encoder does not accept region of interest offsets" : error);
			return;
		}
		std::printf("  face box error %.1f without the offset, %.1f with it\n", box_error(plain_picture),
		            box_error(offset_picture));
		check(box_error(offset_picture) < box_error(plain_picture) && offset_size > plain_size,
		      "the face offset buys a finer face");
	}

	void test_clip()
	{
		std::printf("clip recording\n");

		const std::filesystem::path path = std::filesystem::temp_directory_path() / "low-ban-selftest.mp4";
		std::string error;
		std::vector<video_frame> written;
		{
			clip_writer writer;
			if (!writer.open(path.wstring(), 640, 480, packet_hz, error))
			{
				skip("clip recording", error);
				return;
			}
			bool writes = true;
			for (int i = 0; i < 10; ++i)
			{
				written.push_back(grey_frame(textured_scene(640, 480)));
				for (int y = 0; y < 480; ++y)
					for (int x = 0; x < 640; ++x)
						written.back().luma(x, y) = static_cast<uint8_t>(
							std::clamp(written.back().luma(x, y) + (x + i * 8 < 320 ? -30 : 30), 0, 255));
				writes = writes && writer.write(written.back(), error);
			}
			check(writes && writer.frames() == 10 && writer.close(error), "a clip records and finishes");
		}

		clip_reader reader;
		check(reader.open(path.wstring(), 640, 480, error), "a recorded clip opens for replay");
		video_frame frame;
		double seconds = 0.0;
		double previous = -1.0;
		int frames = 0;
		bool ordered = true;
		double worst = 1000.0;
		while (reader.read(frame, seconds, error))
		{
			ordered = ordered && seconds > previous;
			previous = seconds;
			if (frames < static_cast<int>(written.size()))
				worst = std::min(worst, measure_quality(written[frames].luma, frame.luma).psnr);
			++frames;
		}
		reader.close();
		std::filesystem::remove(path);
		std::printf("  %d frames replayed, worst %.1f dB\n", frames, worst);
		check(error.empty() && frames == 10 && ordered, "every frame replays in order");
		check(worst > 30.0, "a recorded clip is a faithful source");
	}

	// A wide YUV4MPEG2 stream is cropped about its centre to the capture's shape - the outer
	// bands here never reach the frame - and scaled, and its frames keep the stream's rate.
	void test_y4m()
	{
		std::printf("y4m replay\n");

		const std::filesystem::path path = std::filesystem::temp_directory_path() / "low-ban-selftest.y4m";
		{
			std::ofstream out(path, std::ios::binary);
			out << "YUV4MPEG2 W64 H36 F30:1 Ip A1:1 C420jpeg\n";
			for (int f = 0; f < 3; ++f)
			{
				out << "FRAME\n";
				for (int y = 0; y < 36; ++y)
					for (int x = 0; x < 64; ++x) out.put(static_cast<char>(x < 8 ? 16 : x >= 56 ? 235 : 100 + f * 20));
				for (int i = 0; i < 32 * 18; ++i) out.put(static_cast<char>(128));
				for (int i = 0; i < 32 * 18; ++i) out.put(static_cast<char>(90));
			}
		}

		clip_reader reader;
		std::string error;
		const bool opened = reader.open(path.wstring(), 32, 24, error);
		video_frame frame;
		double seconds = 0.0;
		int frames = 0;
		bool exact = opened;
		while (reader.read(frame, seconds, error))
		{
			const auto expected = static_cast<uint8_t>(100 + frames * 20);
			exact = exact && frame.width() == 32 && frame.height() == 24 && frame.luma(0, 0) == expected &&
			        frame.luma(31, 23) == expected && frame.luma(16, 12) == expected && frame.cb(8, 20) == 128 &&
			        frame.cr(15, 0) == 90 && std::fabs(seconds - frames / 30.0) < 1e-9;
			++frames;
		}
		reader.close();
		std::filesystem::remove(path);
		check(exact && frames == 3 && error.empty(),
		      "a y4m stream replays cropped to the capture's shape at its own frame rate");
	}

	// The call session on synthetic frames and landmarks at a camera's 30 frames a second.
	void test_session()
	{
		std::printf("call session\n");

		const image_u8 scene = textured_scene(640, 480);
		const std::vector<point_i> rest = synthetic_face(320, 240, 220);
		call_session session;
		const session_options options;
		render_state state;
		for (int i = 0; i < 90; ++i)
		{
			face_analysis faces;
			faces.points = {synthetic_face(320 + i, 240, 220, (i / 3) % 12)};
			faces.boxes = {rect_i::from_size(200 + i, 150, 240, 240)};
			image_u8 luma;
			warp_faces(scene, {rest}, faces.points, luma);
			session.process(grey_frame(luma), i / 30.0, faces, nullptr, options, state);
		}

		// The bucket never holds more than a tick's share times the ticks so far, so the packets
		// after startup can never have spent more than that.
		const session_totals& sums = session.totals();
		const double share = call_kbit_per_second * 1000.0 / 8.0 / packet_hz;
		std::printf(
			"  %d ticks, %d packets after startup (%d warped), largest %zu bytes, %lld of %.0f bytes, %.1f dB\n",
			sums.ticks, sums.lowban_packets, sums.lowban_warped_packets, sums.lowban_largest_packet,
			static_cast<long long>(sums.lowban_bytes), share * (sums.ticks - 1), state.lowban.quality.psnr);
		check(sums.ticks >= 44 && sums.ticks <= 46, "the call's clock runs at 15 Hz");
		check(sums.lowban_packets <= sums.ticks - 1, "low-ban sends at most one packet a tick");
		check(sums.lowban_largest_packet <= packet_cap_bytes, "every packet after startup fits its cap");
		check(static_cast<double>(sums.lowban_bytes) <= share * (sums.ticks - 1) + 16.0 * sums.lowban_packets,
		      "low-ban spends no more than the call rate, landmarks included");
		check(sums.lowban_warped_packets > 0, "moving landmarks warp the held face");
		check(state.lowban.quality.psnr > 24.0, "the low-ban receiver keeps a usable picture");
		check(state.lowban.link_kbit_per_second > state.lowban.payload_kbit_per_second + audio_link_kbit_per_second(),
		      "the link estimate adds audio and headers to the payload");

		if (!state.baseline_status.empty())
		{
			skip("H.264 session wire", state.baseline_status);
			return;
		}
		std::printf("  %d H.264 frames, %.1f dB\n", sums.baseline_frames, state.baseline.quality.psnr);
		check(sums.baseline_frames >= 40 && sums.baseline_frames <= 46, "H.264 runs at 15 frames a second");
		check(state.baseline.quality.psnr > 20.0, "the H.264 receiver keeps a usable picture");
	}

	void test_detector()
	{
		std::printf("face detection\n");

		face_detector detector;

		// A skin coloured ellipse is not a face, and neither is noise: the whole network runs on
		// both and fires on neither.
		const video_frame ellipse = make_face_frame(640, 480, 320, 200, 60, 80);
		check(detector.detect(ellipse.luma).empty(), "a skin coloured blob is not reported as a face");

		image_u8 noise(640, 480);
		uint32_t seed = 1;
		for (size_t i = 0; i < noise.size(); ++i)
		{
			seed = seed * 1664525u + 1013904223u;
			noise.data()[i] = static_cast<uint8_t>(seed >> 24);
		}
		check(detector.detect(noise).empty(), "random noise is not reported as a face");
		check(detector.detect(image_u8{}).empty(), "an empty frame is handled");

		// A real face, from the stills the evaluation scores, read as the camera delivers frames.
		video_frame sample;
		if (!load_still(asset_path(L"..\\samples\\samp01.png"), 640, 480, sample))
		{
			skip("a real face", "samples\\samp01.png could not be read");
			return;
		}
		const std::vector<detected_face> faces = detector.detect(sample.luma);
		std::printf("  %zu faces in samp01.png\n", faces.size());
		check(!faces.empty(), "the sample's face is found");
		if (faces.empty()) return;

		const auto largest = std::max_element(faces.begin(), faces.end(), [](const auto& a, const auto& b)
		{ return a.box.width() < b.box.width(); });
		const std::vector<point_i>& p = largest->points;
		check(p.size() == face_point_count && p[eye_left].x < p[eye_right].x && p[eye_left].y < p[nose_tip].y &&
		          p[eye_right].y < p[nose_tip].y && p[nose_tip].y < p[mouth_left].y && p[nose_tip].y < p[mouth_right].y,
		      "eyes, nose and mouth are found in their places");
		bool inside = true;
		for (const point_i& point : p)
			inside = inside && point.x >= largest->box.left && point.x <= largest->box.right &&
			         point.y >= largest->box.top && point.y <= largest->box.bottom;
		check(inside, "the landmarks lie inside the face's box");

		const std::vector<detected_face> again = detector.detect(sample.luma);
		bool same = again.size() == faces.size();
		for (size_t i = 0; same && i < faces.size(); ++i)
			same = again[i].points == faces[i].points && again[i].box.left == faces[i].box.left &&
			       again[i].box.top == faces[i].box.top && again[i].confidence == faces[i].confidence;
		check(same, "detection is deterministic");

		// The receivers are scored by finding the face again: in an unchanged picture it has not moved.
		face_analysis analysis = analyse_faces(detector, sample.luma);
		landmark_error unchanged;
		check(measure_expression(detector, sample.luma, analysis, unchanged) && unchanged.overall == 0.0,
		      "landmarks found again in the same picture have no error");
		image_u8 blank(640, 480);
		blank.fill(128);
		landmark_error lost;
		check(measure_expression(detector, blank, analysis, lost) && lost.overall == 1.0,
		      "a face lost from the received picture counts as a whole eye spacing out");
	}

	void test_importance()
	{
		std::printf("face importance\n");

		// The learned prior, read in the template square's own coordinates.
		const auto prior = [](const double x, const double y)
		{
			constexpr double cell = 112.0 / expression_prior_cells;
			return static_cast<int>(
				expression_prior[static_cast<int>(y / cell) * expression_prior_cells + static_cast<int>(x / cell)]);
		};
		check(*std::max_element(std::begin(expression_prior), std::end(expression_prior)) == 255,
		      "the prior reaches the most important");
		check(prior(56, 92) > prior(10, 72) && prior(38, 48) > prior(10, 72),
		      "the mouth and the eyes outrank the cheek");

		const std::vector<point_i> face = synthetic_face(160, 120, 112);
		image_u8 importance(320, 240);
		importance.fill(background_importance);
		add_face_importance(importance, rect_i::from_size(104, 60, 112, 130), face);
		const point_i mouth{(face[mouth_left].x + face[mouth_right].x) / 2,
		                    (face[mouth_left].y + face[mouth_right].y) / 2};
		check(importance(5, 5) == background_importance, "the background keeps its weight");
		check(importance(106, 185) == face_importance, "the box below the template square takes the face's weight");
		check(importance(mouth.x, mouth.y) > importance(face[nose_tip].x - 50, face[nose_tip].y),
		      "the mouth outranks the cheek");

		// Turned a quarter with the head, the importance turns with it.
		std::vector<point_i> turned;
		turned.reserve(face.size());
		for (const point_i& point : face) turned.push_back({160 - (point.y - 120), 120 + (point.x - 160)});
		image_u8 rotated(320, 240);
		rotated.fill(background_importance);
		add_face_importance(rotated, {}, turned);
		const point_i turned_mouth{160 - (mouth.y - 120), 120 + (mouth.x - 160)};
		check(rotated(turned_mouth.x, turned_mouth.y) == importance(mouth.x, mouth.y) ||
		          std::abs(rotated(turned_mouth.x, turned_mouth.y) - importance(mouth.x, mouth.y)) <= 2,
		      "the importance turns with the face");
	}
}
int run_self_test()
{
	failures = 0;
	checks = 0;
	skipped = 0;

	test_image_ops();
	test_yuy2();
	test_entropy();
	test_codec();
	test_loop_filter();
	test_warp();
	test_warped_stream();
	test_landmark_wire();
	test_landmark_filter();
	test_baseline();
	test_clip();
	test_y4m();
	test_session();
	test_detector();
	test_importance();

	std::printf("\n%d checks, %d failed, %d skipped\n", checks, failures, skipped);
	return failures == 0 ? 0 : 1;
}
