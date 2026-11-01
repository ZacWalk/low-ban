// Headless checks for everything that does not need a camera or a window.
// Run with: low-ban-64d.exe /test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "baseline.h"
#include "capture.h"
#include "codec.h"
#include "draw.h"
#include "engine.h"
#include "face.h"
#include "image.h"
#include "landmarks.h"
#include "session.h"
#include "warp.h"

namespace
{
	int failures = 0;
	constexpr double pi = 3.14159265358979323846;

	void check(const bool condition, const char* description)
	{
		if (!condition) ++failures;
		std::printf("  [%s] %s\n", condition ? "pass" : "FAIL", description);
	}

	// For checks that need a platform component this machine may not have, such as Media
	// Foundation's H.264 transforms on a server without the media feature.
	void skip(const char* description, const std::string& reason)
	{
		std::printf("  [skip] %s: %s\n", description, reason.c_str());
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

	// A plausible 68-point face in the usual layout, centred on (cx, cy) and `size` pixels
	// across, with the lips `open` pixels apart.
	std::vector<point_i> synthetic_face(const int cx, const int cy, const int size, const int open = 0)
	{
		std::vector<point_i> p(68);
		const double s = size / 100.0;
		const auto at = [&](const double x, const double y)
		{
			return point_i{cx + static_cast<int>(std::lround(x * s)), cy + static_cast<int>(std::lround(y * s))};
		};

		for (int i = 0; i <= 16; ++i)
		{
			const double a = pi * i / 16.0;
			p[i] = at(-45.0 * std::cos(a), -10.0 + 65.0 * std::sin(a));
		}
		for (int i = 0; i < 5; ++i)
		{
			const double lift = i == 2 ? 4.0 : (i == 1 || i == 3 ? 3.0 : 0.0);
			p[17 + i] = at(-38.0 + i * 8.0, -30.0 - lift);
			p[22 + i] = at(6.0 + i * 8.0, -30.0 - lift);
		}
		for (int i = 0; i < 4; ++i) p[27 + i] = at(0.0, -18.0 + i * 7.0);
		for (int i = 0; i < 5; ++i) p[31 + i] = at(-8.0 + i * 4.0, 6.0 + (i == 2 ? 2.0 : 0.0));
		const double eye[6][2] = {{-8, 0}, {-4, -3}, {4, -3}, {8, 0}, {4, 3}, {-4, 3}};
		for (int i = 0; i < 6; ++i)
		{
			p[36 + i] = at(-20.0 + eye[i][0], -15.0 + eye[i][1]);
			p[42 + i] = at(20.0 + eye[i][0], -15.0 + eye[i][1]);
		}

		// Upper lip points rise and lower ones fall as the mouth opens; corners stay put.
		const double gap = open / s / 2.0;
		const auto lip = [&](const double width, const double height, const double a)
		{
			const double up = std::sin(a);
			const double parted = up > 1e-9 ? -gap : (up < -1e-9 ? gap : 0.0);
			return at(-width * std::cos(a), 25.0 - height * up + parted);
		};
		for (int i = 0; i < 12; ++i) p[48 + i] = lip(18.0, 7.0, pi * i / 6.0);
		for (int i = 0; i < 8; ++i) p[60 + i] = lip(11.0, 2.0, pi * i / 4.0);
		return p;
	}

	// Detail everywhere, so a wrong prediction always costs bytes or decibels.
	image_u8 textured_scene(const int width, const int height)
	{
		image_u8 img(width, height);
		for (int y = 0; y < height; ++y)
			for (int x = 0; x < width; ++x)
			{
				const double value = 128.0 + 50.0 * std::sin(x * 0.21) * std::cos(y * 0.17)
					+ 30.0 * std::sin((x + 2 * y) * 0.07) + (x * 7 + y * 13) % 17 - 8;
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
		return a.width() == b.width() && a.height() == b.height() && std::equal(a.data(), a.data() + a.size(), b.data());
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
		check(frame.luma(3, 2) == 23 && frame.cb(1, 2) == 100 && frame.cr(1, 2) == 200, "planes carry the right values");

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
			for (int x = 0; x < width; ++x) nv12[static_cast<size_t>(y) * nv12_stride + x] = static_cast<uint8_t>(y * 10 + x);
		for (int y = 0; y < height / 2; ++y)
			for (int x = 0; x < width / 2; ++x)
			{
				nv12[static_cast<size_t>(plane_rows + y) * nv12_stride + x * 2] = static_cast<uint8_t>(60 + y);
				nv12[static_cast<size_t>(plane_rows + y) * nv12_stride + x * 2 + 1] = static_cast<uint8_t>(190 + y);
			}
		video_frame from_nv12;
		check(decode_nv12(from_nv12, nv12.data(), nv12.size(), width, height, nv12_stride, plane_rows)
			&& from_nv12.luma(5, 3) == 35 && from_nv12.cb(1, 3) == 61 && from_nv12.cr(1, 0) == 190,
		      "NV12 decodes into the same planes, chroma rows repeated");
		check(!decode_nv12(from_nv12, nv12.data(), nv12.size() - 1, width, height, nv12_stride, plane_rows),
		      "short NV12 buffers are rejected");
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
		std::printf("low-ban transform codec\n");

		image_u8 frame(160, 120);
		for (int y = 0; y < 120; ++y)
			for (int x = 0; x < 160; ++x)
			{
				const int local_x = x % 20;
				const int local_y = y % 20;
				const int block = x / 20 + y / 20;
				const double value = 40 + block % 5 * 25 + local_x * (block % 3 + 1)
					+ local_y * ((block + 1) % 3 + 1)
					+ 30.0 * std::sin(local_x * 0.7 + block) * std::sin(local_y * 0.5);
				frame(x, y) = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
			}

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
		check(same_image(transform_decoded, traditional_output), "serialized transform keyframe matches direct reconstruction");

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

		// The live startup keyframe is the whole scene at the coded size.
		image_u8 startup_frame(640 / coded_scale, 480 / coded_scale);
		image_u8 startup_importance(startup_frame.width(), startup_frame.height());
		startup_importance.fill(32);
		for (int y = 0; y < startup_frame.height(); ++y)
			for (int x = 0; x < startup_frame.width(); ++x)
				startup_frame(x, y) = static_cast<uint8_t>((x * 3 + y * 5) & 0xff);
		temporal_transform_stream startup_stream;
		const transform_packet startup_keyframe = startup_stream.encode(
			traditional, startup_frame, &startup_importance, true, 0);
		image_u8 startup_decoded;
		image_u8 startup_direct;
		traditional.reconstruct(startup_frame, &startup_importance, startup_direct);
		check(startup_keyframe.keyframe && startup_keyframe.updated_blocks == 40 * 30,
		      "startup keyframe contains the whole coded scene");
		check(startup_stream.decode(traditional, startup_keyframe, startup_frame.width(), startup_frame.height(),
			startup_decoded) && same_image(startup_decoded, startup_direct),
		      "startup keyframe reconstructs the whole scene exactly");

		const double budget_kbit = (picture_packet_budget / picture_interval_seconds + 80.0 * geometry_hz) * 8.0 / 1000.0;
		std::printf("  one-face low-ban budget %.1f kbit/s\n", budget_kbit);
		check(budget_kbit < 80.0, "low-ban's post-startup budget for one face stays below 80 kbit/s");

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
		check(halves.size() == 2 && positive && square_area == 200, "a square triangulates into two positive triangles");

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
		check(mesh.size() == 78 && mesh[70].y < face[19].y - 20, "the mesh reaches up over the forehead");

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
		for (int y = moved[30].y - 6; y <= moved[30].y + 6; ++y)
			for (int x = moved[30].x - 6; x <= moved[30].x + 6; ++x)
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

	// Pictures generated by warping a textured scene with landmarks that move: the face
	// drifts and the mouth opens while everything else stays still. The motion is made by the
	// same warp the codec predicts with, so this checks the mechanism, not how well a real
	// face fits a piecewise affine model.
	void test_warped_stream()
	{
		std::printf("low-ban warp prediction\n");

		const transform_codec codec;
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
			int64_t bytes = 0;
			int warped = 0;
			double startup_psnr = 0.0;
			reconstruction_quality quality;
		};
		const auto run = [&](const bool with_geometry, const bool still_picture)
		{
			outcome result;
			temporal_transform_stream stream;
			const face_geometry start = with_geometry ? geometry_at(0) : face_geometry{};
			image_u8 decoded;
			result.decodes = stream.decode(codec, stream.encode(codec, frame_at(0), nullptr, true, 0, start),
				320, 240, decoded, start);
			result.startup_psnr = measure_quality(frame_at(0), decoded).psnr;
			image_u8 source;
			face_geometry geometry;
			for (int step = 1; step <= 8; ++step)
			{
				geometry = with_geometry ? geometry_at(step) : face_geometry{};
				source = still_picture ? frame_at(0) : frame_at(step);
				const transform_packet packet = stream.encode(codec, source, nullptr, false, 1200, geometry);
				result.bytes += static_cast<int64_t>(packet.bytes.size());
				if (packet.warped) ++result.warped;
				result.decodes = result.decodes && packet.bytes.size() <= 1200
					&& (packet.bytes.empty() || stream.decode(codec, packet, 320, 240, decoded, geometry));
			}
			image_u8 shown;
			stream.animate(geometry, shown);
			result.quality = measure_quality(source, shown);
			return result;
		};

		const outcome warped = run(true, false);
		const outcome plain = run(false, false);
		std::printf("  with geometry %lld bytes at %.1f dB, without %lld bytes at %.1f dB\n",
			static_cast<long long>(warped.bytes), warped.quality.psnr, static_cast<long long>(plain.bytes),
			plain.quality.psnr);
		check(warped.decodes && plain.decodes, "both streams stay inside their budget and decode");
		check(warped.warped > 0, "moving geometry is committed as a warp");
		check(warped.bytes < plain.bytes && warped.quality.psnr + 0.25 > plain.quality.psnr,
		      "the warp reaches the same picture for fewer bytes");

		// Geometry that moves over a still picture is wrong. The encoder refuses it once, in a
		// bare header, and the receiver goes back to showing the picture it holds.
		const outcome misled = run(true, true);
		check(misled.decodes && misled.warped == 0 && misled.bytes <= 10
			&& misled.quality.psnr == misled.startup_psnr, "geometry that defies the picture is refused for a header");

		temporal_transform_stream stream;
		image_u8 decoded;
		stream.decode(codec, stream.encode(codec, frame_at(0), nullptr, true, 0, geometry_at(0)), 320, 240, decoded,
			geometry_at(0));
		image_u8 shown;
		int animated = 0;
		const bool animates = stream.animate(geometry_at(3), shown);
		for (int y = rest[30].y - 10; animates && y <= rest[30].y + 10; ++y)
			for (int x = rest[30].x - 10; x <= rest[30].x + 10; ++x)
				if (shown(x, y) != decoded(x, y)) ++animated;
		check(animates && animated > 200 && shown(5, 5) == decoded(5, 5),
		      "the receiver animates its picture onto newer geometry");

		// A rejected packet must leave the decoder as it was: the next good packet then
		// decodes to exactly what a decoder that never saw the bad one produces.
		const auto delta_after_keyframe = [&](temporal_transform_stream& target, image_u8& picture)
		{
			target.decode(codec, target.encode(codec, frame_at(0), nullptr, true, 0, geometry_at(0)), 320, 240,
				picture, geometry_at(0));
			return target.encode(codec, frame_at(2), nullptr, false, 1200, geometry_at(2));
		};
		temporal_transform_stream damaged_stream;
		temporal_transform_stream clean_stream;
		image_u8 damaged_picture;
		image_u8 clean_picture;
		const transform_packet damaged_delta = delta_after_keyframe(damaged_stream, damaged_picture);
		const transform_packet clean_delta = delta_after_keyframe(clean_stream, clean_picture);
		transform_packet truncated = damaged_delta;
		truncated.bytes.pop_back();
		transform_packet unknown_flag = damaged_delta;
		unknown_flag.bytes[3] = static_cast<uint8_t>(unknown_flag.bytes[3] | 8);
		transform_packet contradictory = damaged_delta;
		contradictory.bytes[3] = static_cast<uint8_t>(contradictory.bytes[3] | 4);
		check(damaged_delta.warped && !damaged_stream.decode(codec, truncated, 320, 240, damaged_picture, geometry_at(2))
			&& !damaged_stream.decode(codec, unknown_flag, 320, 240, damaged_picture, geometry_at(2))
			&& !damaged_stream.decode(codec, contradictory, 320, 240, damaged_picture, geometry_at(2)),
		      "truncated, unknown or contradictory warp packets are rejected");
		check(damaged_stream.decode(codec, damaged_delta, 320, 240, damaged_picture, geometry_at(2))
			&& clean_stream.decode(codec, clean_delta, 320, 240, clean_picture, geometry_at(2))
			&& same_image(damaged_picture, clean_picture), "a rejected warp leaves decoder state untouched");

		temporal_transform_stream key_stream;
		transform_packet warped_keyframe = key_stream.encode(codec, frame_at(0), nullptr, true, 0, geometry_at(0));
		warped_keyframe.bytes[3] = static_cast<uint8_t>(warped_keyframe.bytes[3] | 2);
		check(!key_stream.decode(codec, warped_keyframe, 320, 240, decoded, geometry_at(0)),
		      "a keyframe claiming a warp is rejected");
	}

	void test_geometry_wire()
	{
		std::printf("landmark wire\n");

		face_geometry wire_faces(1, std::vector<point_i>(68));
		for (size_t i = 0; i < wire_faces[0].size(); ++i)
			wire_faces[0][i] = {100 + static_cast<int>(i % 17) * 4, 80 + static_cast<int>(i / 17) * 6};
		temporal_landmark_stream wire;
		const landmark_packet landmark_keyframe = wire.encode(wire_faces, true);
		face_geometry wire_decoded;
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

		// A head that moves without changing expression is one shift and no residuals.
		auto shifted = moved_faces;
		for (point_i& p : shifted[0])
		{
			p.x += 8;
			p.y -= 6;
		}
		const landmark_packet shift_packet = wire.encode(shifted, false);
		check(shift_packet.bytes.size() <= 20 && wire.decode(shift_packet, wire_decoded) && wire_decoded == wire.transmitted(),
		      "a rigid head movement costs a shift, not a byte per point");

		auto twitch = shifted;
		twitch[0][48].x += 4;
		twitch[0][54].x -= 4;
		twitch[0][57].y += 6;
		const landmark_packet twitch_packet = wire.encode(twitch, false);
		check(twitch_packet.bytes.size() <= 24 && wire.decode(twitch_packet, wire_decoded),
		      "a few moving points list only themselves");

		auto jump = twitch;
		jump[0][8].y += 60;
		const landmark_packet jump_packet = wire.encode(jump, false);
		check(wire.decode(jump_packet, wire_decoded) && wire_decoded[0][8].y == jump[0][8].y / 2 * 2
			&& wire_decoded == wire.transmitted(), "large motion escapes and still decodes exactly");

		auto next = jump;
		next[0][0].x += 2;
		landmark_packet bad_layout = wire.encode(next, false);
		bad_layout.bytes[6] = static_cast<uint8_t>(bad_layout.bytes[6] | 4);
		check(!wire.decode(bad_layout, wire_decoded), "an unknown layout is rejected");

		face_geometry two_faces = {wire_faces[0], moved_faces[0]};
		temporal_landmark_stream pair;
		face_geometry pair_decoded;
		check(pair.decode(pair.encode(two_faces, true), pair_decoded) && pair_decoded == pair.transmitted()
			&& pair_decoded.size() == 2, "several faces round trip");
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
				smoothed_low = std::min(smoothed_low, out[0][30].x);
				smoothed_high = std::max(smoothed_high, out[0][30].x);
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
		check(std::abs(out[0][30].x - stepped[0][30].x) <= 2, "real motion is followed within a third of a second");

		landmark_error error;
		check(compare_landmarks(still[0], still[0], error) && error.overall == 0.0 && error.mouth == 0.0,
		      "identical landmarks have no error");
		std::vector<point_i> offset = still[0];
		for (point_i& p : offset)
		{
			p.x += 3;
			p.y += 4;
		}
		const double ocular = std::hypot(static_cast<double>(still[0][45].x - still[0][36].x),
			static_cast<double>(still[0][45].y - still[0][36].y));
		check(compare_landmarks(still[0], offset, error) && std::fabs(error.overall - 5.0 / ocular) < 1e-9
			&& std::fabs(error.mouth - 5.0 / ocular) < 1e-9, "landmark error is normalised by the eye corners");
		check(!compare_landmarks(still[0], {}, error), "a missing face cannot be scored");
	}

	void test_baseline()
	{
		std::printf("h264 baseline\n");

		h264_settings settings;
		settings.width = 320;
		settings.height = 240;
		settings.frames_per_second = geometry_hz;
		settings.kbit_per_second = baseline_kbit_per_second;
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
			round_trips = round_trips && wire.encode(frame, {rect_i::from_size(100 + i, 60, 120, 120)}, bytes, error)
				&& !bytes.empty() && wire.decode(bytes, picture, error);
			const std::vector<int> types = unit_types(bytes);
			if (i == 0) first_has_parameter_sets = std::count(types.begin(), types.end(), 7) > 0;
			only_needed_units = only_needed_units && std::count(types.begin(), types.end(), 6) == 0
				&& std::count(types.begin(), types.end(), 9) == 0;
			if (i > 0) bytes_after_first += static_cast<int64_t>(bytes.size());
		}
		const double rate = bytes_after_first * 8.0 / (44.0 / geometry_hz) / 1000.0;
		const reconstruction_quality quality = measure_quality(frame, picture);
		std::printf("  %.1f kbit/s after the first frame, last frame %.1f dB\n", rate, quality.psnr);
		check(round_trips, "every frame crosses the H.264 wire and decodes");
		check(first_has_parameter_sets && only_needed_units,
		      "the H.264 wire carries the parameter sets but no delimiters or SEI");
		check(rate < baseline_kbit_per_second * 1.5, "the H.264 wire holds near its constant rate");
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
			const bool ok = one.open(fixed, error) && one.encode(scene, {box}, bytes, error)
				&& one.decode(bytes, decoded, error) && !decoded.empty();
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
			if (!writer.open(path.wstring(), 640, 480, geometry_hz, error))
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
						written.back().luma(x, y) = static_cast<uint8_t>(std::clamp(
							written.back().luma(x, y) + (x + i * 8 < 320 ? -30 : 30), 0, 255));
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

	// The call session on synthetic frames and landmarks at a camera's 30 frames a second.
	void test_session()
	{
		std::printf("call session\n");

		const image_u8 scene = textured_scene(640, 480);
		const std::vector<point_i> rest = synthetic_face(320, 240, 220);
		call_session session;
		session_options options;
		options.landmark_overlay = false;
		render_state state;
		for (int i = 0; i < 90; ++i)
		{
			face_analysis faces;
			faces.landmarks = {synthetic_face(320 + i, 240, 220, (i / 3) % 12)};
			faces.landmarked = {rect_i::from_size(200 + i, 150, 240, 240)};
			faces.boxes = faces.landmarked;
			image_u8 luma;
			warp_faces(scene, {rest}, faces.landmarks, luma);
			session.process(grey_frame(luma), i / 30.0, faces, nullptr, options, state);
		}

		const session_totals& sums = session.totals();
		std::printf("  %d geometry packets, %d pictures (%d warped), largest delta %zu bytes, %.1f dB\n",
			sums.lowban_geometry_packets, sums.lowban_picture_packets, sums.lowban_warped_packets,
			sums.lowban_largest_picture_delta, state.lowban.quality.psnr);
		check(sums.lowban_geometry_packets >= 40 && sums.lowban_geometry_packets <= 46, "geometry runs at 15 Hz");
		check(sums.lowban_picture_packets >= 2 && sums.lowban_picture_packets <= 16, "pictures run at no more than 5 Hz");
		check(sums.lowban_largest_picture_delta <= picture_packet_budget, "every picture after startup fits its cap");
		check(sums.lowban_warped_packets > 0, "moving geometry warps the held face");
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

			// The receiver-side measurement: landmarks found in an unchanged picture from the
			// same box are the source's own landmarks.
			landmark_error unchanged;
			check(compare_landmarks(parts, predictor.predict(img, box), unchanged) && unchanged.overall == 0.0,
			      "landmarks found again in the same picture have no error");
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
	test_warp();
	test_warped_stream();
	test_geometry_wire();
	test_landmark_filter();
	test_baseline();
	test_clip();
	test_session();
	test_detector();
	test_landmarks();

	std::printf("\n%s\n", failures == 0 ? "all checks passed" : "there were failures");
	return failures == 0 ? 0 : 1;
}
