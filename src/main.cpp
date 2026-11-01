// low-ban - low bandwidth video with face landmarking.
//
// The window is deliberately thin: it owns the frame, menu, presentation and
// event loop, and does no image processing. Everything else lives in
// engine/session and the modules below.
//
// Command line:
//   /test         run the headless self test and exit with a non-zero code on failure
//   /evaluate     score samples/*.png and samples/*.mp4 offline and print CSV
//   /probe-h264   one H.264 round trip, run as a child to find out whether it crashes
#include "platform.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "baseline.h"
#include "capture.h"
#include "engine.h"

int run_self_test();
int run_sample_evaluation();

namespace
{
	constexpr int video_width = 640;
	constexpr int video_height = 480;
	constexpr int layout_padding = 12;
	constexpr int title_height = 48;
	// Each pane has three caption lines - rate, quality, packets - so no figure has to share a
	// line long enough to run into the next pane's caption.
	constexpr int label_height = 24;
	// Segoe UI reports a 23-pixel text line for the old positive 24-pixel cell
	// request. The layout remains 24 pixels, as it did with the original GDI font.
	constexpr int measured_label_height = 23;
	constexpr int caption_lines = 3;
	constexpr uint32_t refresh_timer = 1;

	// The title sits a third of the way down the space above the panes and the status line
	// three paddings above the bottom. This much space above the panes keeps the title clear
	// of them and the status line clear of the captions.
	constexpr int minimum_pane_y = std::max(title_height * 3 / 2,
	                                        layout_padding * 4 + label_height * caption_lines);

	constexpr int menu_about = 200;
	constexpr int menu_exit = 201;
	constexpr int menu_face_detect = 205;
	constexpr int menu_face_landmark = 206;
	constexpr int menu_landmark_overlay = 207;
	constexpr int menu_record = 208;

	enum class command_mode
	{
		gui,
		test,
		evaluate,
		probe_h264,
	};

	struct pane_layout
	{
		int left_x = 0;
		int lowban_x = 0;
		int baseline_x = 0;
		int pane_y = 0;
	};

	struct render_cache
	{
		std::shared_ptr<const render_state> state;
		pf::bitmap source;
		pf::bitmap lowban;
		pf::bitmap baseline;
	};

	struct render_snapshot
	{
		uint64_t generation = 0;
		std::shared_ptr<const render_state> state;
	};

	pf::window_frame_ptr main_frame;
	std::shared_ptr<pf::frame_reactor> main_reactor;
	std::unique_ptr<media_foundation> media;
	engine pipeline;
	std::unique_ptr<render_cache> cache;
	uint64_t cached_generation = 0;
	uint64_t painted_generation = 0;
	pf::font_handle title_font_handle = 0;
	pf::font_handle label_font_handle = 0;
	pf::font_metrics_data title_font_metrics;
	pf::font_metrics_data label_font_metrics;

	pf::isize minimum_client_size()
	{
		return {video_width * 3 + layout_padding * 4, video_height + minimum_pane_y * 2};
	}

	pane_layout compute_layout(const pf::isize extent)
	{
		const int panes_width = video_width * 3 + layout_padding * 2;
		const int left_x = (extent.cx - panes_width) / 2;
		const int lowban_x = left_x + video_width + layout_padding;
		return {left_x, lowban_x, lowban_x + video_width + layout_padding,
		        (extent.cy - video_height) / 2};
	}

	command_mode parse_mode(const std::span<const std::string_view> params)
	{
		bool test = false;
		bool evaluate = false;
		bool probe = false;
		for (const auto param : params)
		{
			if (pf::icmp(param, "/test") == 0) test = true;
			else if (pf::icmp(param, "/evaluate") == 0) evaluate = true;
			else if (pf::icmp(param, "/probe-h264") == 0) probe = true;
		}

		// Keep the old command-line precedence when a caller supplies more than one
		// mode: the probe was checked first, then /test, then /evaluate.
		if (probe) return command_mode::probe_h264;
		if (test) return command_mode::test;
		if (evaluate) return command_mode::evaluate;
		return command_mode::gui;
	}

	pf::bitmap make_bgra_bitmap(const int width, const int height, const std::vector<uint8_t>& bgra)
	{
		pf::bitmap result;
		result.width = std::max(0, width);
		result.height = std::max(0, height);
		result.pixels.assign(static_cast<size_t>(result.width) * result.height, 0xFF000000);
		const size_t bytes = std::min(bgra.size(), result.pixels.size() * sizeof(uint32_t));
		if (bytes != 0) std::memcpy(result.pixels.data(), bgra.data(), bytes);
		return result;
	}

	pf::bitmap make_grey_bitmap(const image_u8& image)
	{
		pf::bitmap result;
		result.width = image.width();
		result.height = image.height();
		if (image.empty()) return result;

		result.pixels.resize(image.size());
		for (size_t i = 0; i < image.size(); ++i)
		{
			const uint32_t grey = image.data()[i];
			result.pixels[i] = grey | grey << 8 | grey << 16 | 0xFF000000;
		}
		return result;
	}

	template <typename GenerationReader, typename StateReader>
	render_snapshot read_snapshot(GenerationReader&& read_generation,
	                              StateReader&& read_latest)
	{
		const uint64_t generation = read_generation();
		return {generation, read_latest()};
	}

	void refresh_cache()
	{
		const render_snapshot snapshot = read_snapshot(
			[] { return pipeline.generation(); },
			[] { return pipeline.latest(); });
		auto next = std::make_unique<render_cache>();
		next->state = snapshot.state;
		if (next->state)
		{
			next->source = make_bgra_bitmap(next->state->width, next->state->height,
			                                next->state->source_bgra);
			next->lowban = make_grey_bitmap(next->state->lowban.picture);
			next->baseline = make_grey_bitmap(next->state->baseline.picture);
		}
		cache = std::move(next);
		cached_generation = snapshot.generation;
	}

	void ensure_cache()
	{
		if (cached_generation != pipeline.generation()) refresh_cache();
	}

	template <typename EnsureCache>
	bool queue_cached_repaint(const uint64_t requested_generation, uint64_t& painted,
	                          uint64_t& cached, EnsureCache&& ensure)
	{
		if (requested_generation == painted && cached == painted) return false;
		ensure();
		if (cached == painted) return false;
		painted = cached;
		return true;
	}

	void destroy_fonts()
	{
		if (title_font_handle != 0) pf::delete_font_handle(title_font_handle);
		if (label_font_handle != 0) pf::delete_font_handle(label_font_handle);
		title_font_handle = 0;
		label_font_handle = 0;
		title_font_metrics = {};
		label_font_metrics = {};
	}

	bool create_fonts()
	{
		destroy_fonts();

		pf::font_def title;
		title.face = "Segoe UI";
		title.size = title_height;
		title.size_mode = pf::font_size_mode::cell;
		pf::font_def label;
		label.face = "Segoe UI";
		label.size = label_height;
		label.size_mode = pf::font_size_mode::cell;

		title_font_handle = pf::create_font_handle(title, &title_font_metrics);
		label_font_handle = pf::create_font_handle(label, &label_font_metrics);
		if (title_font_handle != 0 && label_font_handle != 0) return true;

		destroy_fonts();
		return false;
	}

	std::string expression_text(const receiver_view& view)
	{
		if (!view.expression_measured) return "-";
		char text[32] = {};
		std::snprintf(text, sizeof(text), "%0.1f%%", view.expression.overall * 100.0);
		return text;
	}

	void draw_centred(pf::draw_context& draw, const pf::irect& bounds, const int y,
	                  const std::string_view text, const pf::font_handle font,
	                  const pf::color_t colour)
	{
		const pf::isize extent = draw.measure_text_h(text, font);
		const int x = std::max(bounds.left, (bounds.right - extent.cx) / 2);
		const int bottom = std::max(y + 1, y + extent.cy);
		draw.fill_solid_rect({x, y, std::min(bounds.right, x + extent.cx), bottom}, {0, 0, 0});
		draw.draw_text_h(x, y, text, font, colour);
	}

	void draw_caption(pf::draw_context& draw, const int x, const int pane_y, const int line,
	                  const std::string_view text, const pf::font_handle font)
	{
		const int y = pane_y + video_height + layout_padding + line * label_height;
		draw.fill_solid_rect({x, y, x + video_width, y + label_height}, {0, 0, 0});
		draw.draw_text_h(x, y, text, font, {200, 200, 200});
	}

	void draw_face_box(pf::draw_context& draw, const int left, const int top, const rect_i& face)
	{
		const pf::color_t colour(0, 220, 0);
		const pf::ipoint a(left + face.left, top + face.top);
		const pf::ipoint b(left + face.right + 1, top + face.bottom + 1);
		draw.draw_solid_line({a.x, a.y}, {b.x, a.y}, colour, 2);
		draw.draw_solid_line({b.x, a.y}, {b.x, b.y}, colour, 2);
		draw.draw_solid_line({b.x, b.y}, {a.x, b.y}, colour, 2);
		draw.draw_solid_line({a.x, b.y}, {a.x, a.y}, colour, 2);
	}

	void draw_frame(pf::window_frame_ptr& frame, pf::draw_context& draw,
	                const std::string_view status)
	{
		const pf::irect bounds = frame->get_client_rect();
		draw.fill_solid_rect(draw.clip_rect(), {0, 0, 0});

		const auto layout = compute_layout({bounds.width(), bounds.height()});
		draw_centred(draw, bounds, std::max(layout_padding, layout.pane_y / 3),
		             "Low bandwidth video with face landmarking", title_font_handle, {200, 200, 200});

		ensure_cache();
		if (cache && cache->state)
		{
			const auto& state = *cache->state;
			draw.draw_bitmap(layout.left_x, layout.pane_y, cache->source);
			draw.draw_bitmap(layout.lowban_x, layout.pane_y, cache->lowban);
			draw.draw_bitmap(layout.baseline_x, layout.pane_y, cache->baseline);

			// Drawn over the source so detection can be judged separately from landmarking.
			for (const auto& face : state.faces) draw_face_box(draw, layout.left_x, layout.pane_y, face);

			char text[192] = {};
			std::snprintf(text, sizeof(text), "Original: %0.1f fps camera, %0.1f fps analysed",
			              pipeline.source_fps(), pipeline.processed_fps());
			draw_caption(draw, layout.left_x, layout.pane_y, 0, text, label_font_handle);
			std::snprintf(text, sizeof(text), "%d faces (peak detector score %0.2f)",
			              state.face_count, state.peak_confidence);
			draw_caption(draw, layout.left_x, layout.pane_y, 1, text, label_font_handle);

			const receiver_view& lowban = state.lowban;
			std::snprintf(text, sizeof(text), "low-ban: %0.1f kbit/s (%0.1f geometry), link %0.1f kbit/s",
			              lowban.payload_kbit_per_second, state.lowban_geometry_kbit_per_second,
			              lowban.link_kbit_per_second);
			draw_caption(draw, layout.lowban_x, layout.pane_y, 0, text, label_font_handle);
			const std::string lowban_expression = expression_text(lowban);
			std::snprintf(text, sizeof(text), "PSNR %0.1f dB, face %0.1f dB, landmark error %s",
			              lowban.quality.psnr, lowban.quality.weighted_psnr, lowban_expression.c_str());
			draw_caption(draw, layout.lowban_x, layout.pane_y, 1, text, label_font_handle);
			std::snprintf(text, sizeof(text), "last picture: %s, %d blocks",
			              state.lowban_keyframe ? "startup keyframe"
			              : state.lowban_warped ? "warped delta" : "delta",
			              state.lowban_updated_blocks);
			draw_caption(draw, layout.lowban_x, layout.pane_y, 2, text, label_font_handle);

			const receiver_view& baseline = state.baseline;
			if (baseline.picture.empty() && !state.baseline_status.empty())
			{
				draw_caption(draw, layout.baseline_x, layout.pane_y, 0, state.baseline_status,
				             label_font_handle);
			}
			else
			{
				std::snprintf(text, sizeof(text), "H.264 baseline: %0.1f kbit/s, link %0.1f kbit/s",
				              baseline.payload_kbit_per_second, baseline.link_kbit_per_second);
				draw_caption(draw, layout.baseline_x, layout.pane_y, 0, text, label_font_handle);
				const std::string baseline_expression = expression_text(baseline);
				std::snprintf(text, sizeof(text), "PSNR %0.1f dB, face %0.1f dB, landmark error %s",
				              baseline.quality.psnr, baseline.quality.weighted_psnr,
				              baseline_expression.c_str());
				draw_caption(draw, layout.baseline_x, layout.pane_y, 1, text, label_font_handle);
				std::snprintf(text, sizeof(text), "%d kbit/s constant rate, %d fps, face offset %s",
				              baseline_kbit_per_second, geometry_hz,
				              state.baseline_face_offset ? "on" : "off");
				draw_caption(draw, layout.baseline_x, layout.pane_y, 2, text, label_font_handle);
			}
		}

		if (!status.empty())
		{
			draw_centred(draw, bounds, bounds.bottom - 3 * layout_padding, status, label_font_handle,
			             {220, 120, 120});
		}
	}

	void invalidate_after_menu()
	{
		if (main_frame) main_frame->invalidate();
	}

	std::vector<pf::menu_command> build_menu()
	{
		using pf::menu_command;

		menu_command file;
		file.text = "&File";
		file.children.emplace_back("&Record clip", menu_record,
			[] { pipeline.set_recording(!pipeline.recording()); invalidate_after_menu(); },
			nullptr, [] { return pipeline.recording(); });
		file.children.emplace_back();
		file.children.emplace_back("E&xit", menu_exit, [] { if (main_frame) main_frame->close(); });

		menu_command view;
		view.text = "&View";
		view.children.emplace_back("Face &detect", menu_face_detect,
			[] { pipeline.set_show_faces(!pipeline.show_faces()); invalidate_after_menu(); },
			nullptr, [] { return pipeline.show_faces(); });
		view.children.emplace_back("Face &landmarking", menu_face_landmark,
			[] { pipeline.set_show_landmarks(!pipeline.show_landmarks()); invalidate_after_menu(); },
			nullptr, [] { return pipeline.show_landmarks(); });
		view.children.emplace_back("Landmark &overlay", menu_landmark_overlay,
			[] { pipeline.set_show_overlay(!pipeline.show_overlay()); invalidate_after_menu(); },
			nullptr, [] { return pipeline.show_overlay(); });

		menu_command help;
		help.text = "&Help";
		help.children.emplace_back("&About ...", menu_about,
			[]
			{
				if (main_frame)
					main_frame->message_box(
						"low-ban - low bandwidth video experiment\n\nLandmark-warped codec against H.264.",
						"About low-ban", pf::msg_box_style::ok);
			});

		return {std::move(file), std::move(view), std::move(help)};
	}

	void set_initial_window_bounds(const pf::window_frame_ptr& frame)
	{
		// These are the old content dimensions plus the same spare title/status
		// margin used by the Win32 window. The platform owns the non-client frame.
		constexpr int width = video_width * 3 + layout_padding * 8;
		constexpr int height = video_height + minimum_pane_y * 2 + 60;
		const auto screen = pf::platform_screen_size();
		const int left = std::max(0, (screen.cx - width) / 2);
		const int top = std::max(0, (screen.cy - height) / 2);
		frame->move_window({left, top, left + width, top + height});
	}

	class app_reactor final : public pf::frame_reactor
	{
	public:
		uint32_t handle_message(pf::window_frame_ptr window, const pf::message_type message,
		                        const pf::message_params& params) override
		{
			switch (message)
			{
			case pf::message_type::create:
				window->set_text("low-ban");
				set_initial_window_bounds(window);
				window->set_minimum_client_size(minimum_client_size());
				window->set_timer(refresh_timer, 1000 / 60);
				return 0;

			case pf::message_type::erase_background:
				return 1; // draw_frame paints every pixel that needs repainting

			case pf::message_type::timer:
				if (params.timer_id == refresh_timer)
				{
					const uint64_t requested_generation = pipeline.generation();
					if (queue_cached_repaint(requested_generation, painted_generation,
					                         cached_generation, [] { ensure_cache(); }))
					{
						window->invalidate();
					}
				}
				return 0;

			case pf::message_type::close:
				window->close();
				return 0;

			case pf::message_type::destroy:
				window->kill_timer(refresh_timer);
				return 0;

			default:
				return 0;
			}
		}

		uint32_t handle_keyboard(pf::window_frame_ptr, const pf::keyboard_message_type,
		                         const pf::keyboard_params&) override
		{
			// Menu accelerators are dispatched by libdesktop before a key reaches
			// the reactor. There are no application-specific shortcuts.
			return 0;
		}

		void handle_paint(pf::window_frame_ptr& frame, pf::draw_context& draw) override
		{
			const std::string status = pipeline.status();
			draw_frame(frame, draw, status);
		}

		void handle_size(pf::window_frame_ptr&, const pf::isize,
		                pf::measure_context&) override
		{
		}
	};

	struct recording_window final : pf::window_frame
	{
		pf::irect client{0, 0, 1968, 720};
		placement normal{{100, 100, 2100, 820}, false};
		pf::frame_reactor_ptr reactor;
		pf::isize minimum;
		int minimum_calls = 0;
		int move_calls = 0;
		int invalidates = 0;
		int present_calls = 0;
		uint32_t timer_id = 0;
		uint32_t timer_ms = 0;
		std::string title;

		void set_reactor(pf::frame_reactor_ptr value) override { reactor = std::move(value); }
		void notify_size() override {}
		pf::irect get_client_rect() const override { return client; }
		void invalidate() override { ++invalidates; }
		void invalidate_rect(const pf::irect&) override { ++invalidates; }
		void set_focus() override {}
		bool has_focus() const override { return true; }
		void set_capture() override {}
		void release_capture() override {}
		uint32_t set_timer(const uint32_t id, const uint32_t ms) override
		{
			timer_id = id;
			timer_ms = ms;
			return id;
		}
		void kill_timer(const uint32_t id) override
		{
			if (timer_id == id) timer_id = 0;
		}
		pf::ipoint screen_to_client(const pf::ipoint point) const override { return point; }
		pf::ipoint client_to_screen(const pf::ipoint point) const override { return point; }
		void set_cursor_shape(pf::cursor_shape) override {}
		void move_window(const pf::irect& bounds) override
		{
			++move_calls;
			// The platform's normal placement is deliberately not changed by this
			// recording fake, so a minimize test can detect an erroneous correction.
			(void)bounds;
		}
		void show(bool) override {}
		bool is_visible() const override { return true; }
		void set_text(const std::string_view value) override { title = value; }
		std::string text_from_clipboard() override { return {}; }
		bool text_to_clipboard(std::string_view) override { return true; }
		placement get_placement() const override { return normal; }
		void set_placement(const placement& value) override { normal = value; }
		void set_minimum_client_size(const pf::isize value) override
		{
			minimum = value;
			++minimum_calls;
		}
		void track_mouse_leave() override {}
		bool is_key_down(unsigned int) const override { return false; }
		bool is_key_down_async(unsigned int) const override { return false; }
		pf::window_frame_ptr create_child(std::string_view, uint32_t, pf::color_t) const & override
		{
			return nullptr;
		}
		void close() override {}
		int message_box(std::string_view, std::string_view, uint32_t) override { return 0; }
		void present_pixels(const uint32_t*, int, int) override { ++present_calls; }
		void set_menu(std::vector<pf::menu_command>) override {}
		std::unique_ptr<pf::measure_context> create_measure_context() const override { return nullptr; }
		void show_popup_menu(const std::vector<pf::menu_command>&, const pf::ipoint&) override {}
		double get_dpi_scale() const override { return 1.0; }
		void accept_drop_files(bool) override {}
		pf::toolbar_frame_ptr create_address_bar(const pf::address_bar_config&) override { return nullptr; }
	};

	struct recording_draw_context final : pf::draw_context
	{
		pf::irect clip{0, 0, 1968, 720};
		int fill_calls = 0;
		int bitmap_calls = 0;
		int line_calls = 0;
		std::vector<std::string> texts;
		std::vector<pf::font_handle> text_handles;

		pf::irect clip_rect() const override { return clip; }
		void fill_solid_rect(const pf::irect&, pf::color_t) override { ++fill_calls; }
		void fill_solid_rect(int, int, int, int, pf::color_t) override { ++fill_calls; }
		void draw_text(int, int, const pf::irect&, std::string_view text, const pf::font&,
		               pf::color_t, pf::color_t) override
		{
			texts.emplace_back(text);
			text_handles.push_back(0);
		}
		pf::isize measure_text(std::string_view text, const pf::font&) const override
		{
			return {static_cast<int>(text.size()) * 8, 16};
		}
		void draw_text_h(int, int, std::string_view text, pf::font_handle handle,
		                 pf::color_t) override
		{
			texts.emplace_back(text);
			text_handles.push_back(handle);
		}
		pf::isize measure_text_h(std::string_view text, pf::font_handle) const override
		{
			return {static_cast<int>(text.size()) * 8, 16};
		}
		void draw_lines(std::span<const pf::ipoint>, pf::color_t) override {}
		void draw_solid_line(pf::ipoint, pf::ipoint, pf::color_t, int) override { ++line_calls; }
		void draw_ellipse(int, int, int, int, pf::color_t, int) override {}
		void fill_ellipse(int, int, int, int, pf::color_t) override {}
		void set_clip_rect(const pf::irect&) override {}
		void clear_clip_rect() override {}
		void draw_bitmap(int, int, const pf::bitmap&) override { ++bitmap_calls; }
		void draw_bitmap(const pf::irect&, const pf::bitmap&) override { ++bitmap_calls; }
	};

	struct recording_measure_context final : pf::measure_context
	{
		pf::isize measure_text(std::string_view text, const pf::font&) const override
		{
			return {static_cast<int>(text.size()) * 8, 16};
		}
		pf::isize measure_char(const pf::font&) const override { return {8, 16}; }
	};

	std::shared_ptr<render_state> make_recorded_state()
	{
		auto state = std::make_shared<render_state>();
		state->width = 1;
		state->height = 1;
		state->source_bgra = {80, 80, 80, 255};
		state->lowban.picture.resize(1, 1);
		state->lowban.picture(0, 0) = 96;
		state->baseline_status = "H.264 unavailable";
		state->faces.push_back(rect_i::from_size(0, 0, 1, 1));
		state->face_count = 1;
		state->peak_confidence = 0.5f;
		return state;
	}

	void set_recorded_cache(const std::shared_ptr<const render_state>& state)
	{
		cache = std::make_unique<render_cache>();
		cache->state = state;
		cache->source = make_bgra_bitmap(state->width, state->height, state->source_bgra);
		cache->lowban = make_grey_bitmap(state->lowban.picture);
		cache->baseline = make_grey_bitmap(state->baseline.picture);
		cached_generation = pipeline.generation();
	}

	bool has_recorded_text(const recording_draw_context& draw, const std::string_view needle)
	{
		return std::any_of(draw.texts.begin(), draw.texts.end(),
		                   [needle](const std::string& text) { return text.find(needle) != std::string::npos; });
	}

	bool same_rect(const pf::irect& left, const pf::irect& right)
	{
		return left.left == right.left && left.top == right.top &&
		       left.right == right.right && left.bottom == right.bottom;
	}

	void gui_check(const bool condition, const char* description, int& failures)
	{
		if (!condition) ++failures;
		std::printf("  [%s] %s\n", condition ? "pass" : "FAIL", description);
	}

	int run_gui_self_test()
	{
		std::printf("GUI sizing and command modes\n");
		int failures = 0;
		const bool fonts_ready = create_fonts();
		gui_check(fonts_ready && title_font_metrics.height == title_height &&
		          label_font_metrics.height == measured_label_height,
		          "Segoe UI cell fonts retain the old 48/24 pixel cell requests and measured line heights",
		          failures);
		const pf::isize minimum = minimum_client_size();
		gui_check(minimum == pf::isize(1968, 720), "minimum client size keeps all three panes and captions",
		          failures);

		const pane_layout minimum_layout = compute_layout(minimum);
		gui_check(minimum_layout.left_x == layout_padding && minimum_layout.lowban_x == 664 &&
		          minimum_layout.baseline_x == 1316 && minimum_layout.pane_y == minimum_pane_y,
		          "minimum layout is centred with a full caption area", failures);

		const pane_layout larger_layout = compute_layout({2016, 780});
		gui_check(larger_layout.left_x >= layout_padding && larger_layout.pane_y >= minimum_pane_y,
		          "larger windows retain the pane margins", failures);

		auto second_state = make_recorded_state();
		uint64_t source_generation = 0;
		int generation_reads = 0;
		int latest_calls = 0;
		const render_snapshot snapshot = read_snapshot(
			[&]
			{
				++generation_reads;
				return ++source_generation;
			},
			[&]() -> std::shared_ptr<const render_state>
			{
				++latest_calls;
				return second_state;
			});
		uint64_t repaint_painted = 1;
		uint64_t repaint_cached = 1;
		const std::array<uint64_t, 2> cached_updates{2, 3};
		size_t cached_update_index = 0;
		std::vector<uint64_t> repaint_tokens;
		const auto ensure_repaint_cache = [&]
		{
			repaint_cached = cached_updates[cached_update_index++];
		};
		if (queue_cached_repaint(3, repaint_painted, repaint_cached, ensure_repaint_cache))
			repaint_tokens.push_back(repaint_painted);
		if (queue_cached_repaint(3, repaint_painted, repaint_cached, ensure_repaint_cache))
			repaint_tokens.push_back(repaint_painted);
		gui_check(snapshot.generation == 1 && snapshot.state == second_state &&
		          generation_reads == 1 && latest_calls == 1 &&
		          repaint_tokens == std::vector<uint64_t>{2, 3} && cached_update_index == 2,
		          "bounded snapshot freshness and cached-token repaint catch the final generation",
		          failures);

		auto window = std::make_shared<recording_window>();
		app_reactor reactor;
		reactor.handle_message(window, pf::message_type::create, {});
		gui_check(window->minimum == minimum && window->minimum_calls == 1,
		          "the reactor delegates the minimum client size to the platform", failures);
		const pf::irect normal_bounds = window->normal.normal_bounds;
		recording_measure_context measure;
		const int moves_after_create = window->move_calls;
		pf::window_frame_ptr sized_window = window;
		reactor.handle_size(sized_window, {0, 0}, measure);
		gui_check(window->move_calls == moves_after_create &&
		          same_rect(window->normal.normal_bounds, normal_bounds),
		          "minimizing does not rewrite the saved normal placement", failures);

		painted_generation = UINT64_MAX;
		cached_generation = pipeline.generation();
		cache.reset();
		window->invalidates = 0;
		window->present_calls = 0;
		pf::message_params timer;
		timer.timer_id = refresh_timer;
		reactor.handle_message(window, pf::message_type::timer, timer);
		gui_check(window->invalidates == 1 && window->present_calls == 0,
		          "a timer queues one paint and never presents a captionless frame", failures);

		set_recorded_cache(second_state);
		painted_generation = cached_generation;
		recording_draw_context draw;
		pf::window_frame_ptr paint_window = window;
		draw_frame(paint_window, draw, "camera unavailable");
		gui_check(draw.bitmap_calls == 3 && draw.line_calls >= 4 &&
		          has_recorded_text(draw, "low-ban:") &&
		          has_recorded_text(draw, "H.264 unavailable") &&
		          has_recorded_text(draw, "camera unavailable") &&
		          window->present_calls == 0,
		          "captions, face overlay and status all share the draw_context paint", failures);

		const std::array<std::string_view, 1> test_params{"/TeSt"};
		const std::array<std::string_view, 1> evaluate_params{"/EVALUATE"};
		const std::array<std::string_view, 1> probe_params{"/probe-h264"};
		const std::array<std::string_view, 1> gui_params{"camera"};
		gui_check(parse_mode(test_params) == command_mode::test, "mode parsing accepts /test case-insensitively",
		          failures);
		gui_check(parse_mode(evaluate_params) == command_mode::evaluate &&
		          parse_mode(probe_params) == command_mode::probe_h264 &&
		          parse_mode(gui_params) == command_mode::gui,
		          "mode parsing keeps evaluate, probe and GUI modes distinct", failures);

		const std::array<std::string_view, 2> mixed_params{"/evaluate", "/test"};
		gui_check(parse_mode(mixed_params) == command_mode::test,
		          "mode parsing preserves the existing /test precedence", failures);
		cache.reset();
		destroy_fonts();
		return failures == 0 ? 0 : 1;
	}

	int run_console_mode(const bool evaluate)
	{
		pf::attach_console();
		setvbuf(stdout, nullptr, _IONBF, 0);

		return run_on_mta([evaluate]
		{
			if (evaluate) return run_sample_evaluation();

			const bool mta = is_mta_thread();
			std::printf("  [%s] /test console runtime is an MTA\n", mta ? "pass" : "FAIL");
			const int gui_result = run_gui_self_test();
			const media_foundation media;
			const int self_test_result = run_self_test();
			return mta && gui_result == 0 && self_test_result == 0 ? 0 : 1;
		});
	}
}

app_init_result app_init(const pf::window_frame_ptr& frame,
                         const std::span<const std::string_view> params)
{
	switch (parse_mode(params))
	{
	case command_mode::probe_h264:
		return {false, run_on_mta([] { return run_h264_probe(); })};
	case command_mode::test:
		return {false, run_console_mode(false)};
	case command_mode::evaluate:
		return {false, run_console_mode(true)};
	case command_mode::gui:
		break;
	}

	pf::config_set_app_name("low-ban");
	if (!create_fonts())
	{
		pf::platform_show_error("Could not create the Segoe UI presentation fonts.", "low-ban");
		return {false, 1};
	}
	main_frame = frame;
	main_reactor = std::make_shared<app_reactor>();
	frame->set_reactor(main_reactor);
	frame->set_menu(build_menu());
	media = std::make_unique<media_foundation>();
	pipeline.start(video_width, video_height);
	return {};
}

void app_idle()
{
}

void app_destroy()
{
	pipeline.stop();
	media.reset();
	destroy_fonts();
	cache.reset();
	main_reactor.reset();
	main_frame.reset();
}
