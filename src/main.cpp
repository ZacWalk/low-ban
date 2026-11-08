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
#include "libdesktop/ui.h"
#include "libdesktop/ui/elements.h"
#include "libdesktop/ui/menu_bar.h"
#include "libdesktop/ui/test_support.h"
#include "libdesktop/render/software.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
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
	constexpr int caption_lines = 3;
	// The drawn menu consumes client pixels; its 20-pixel line and 8-pixel padding scale with the
	// opening display, unlike the physical pane dimensions below.
	constexpr int menu_bar_height_96 = 28;

	// The title sits a third of the way down the space above the panes and the status line
	// three paddings above the bottom. This much space above the panes keeps the title clear
	// of them and the status line clear of the captions.
	constexpr int minimum_pane_y = std::max(title_height * 3 / 2,
	                                        layout_padding * 4 + label_height * caption_lines);

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
		core::surface_ptr source;
		core::surface_ptr lowban;
		core::surface_ptr baseline;
	};

	struct render_snapshot
	{
		uint64_t generation = 0;
		std::shared_ptr<const render_state> state;
	};

	std::unique_ptr<media_foundation> media;
	engine pipeline;
	std::unique_ptr<render_cache> cache;
	uint64_t cached_generation = 0;
	uint64_t painted_generation = 0;
	render::font_face title_font = render::font_face::title;
	render::font_face label_font = render::font_face::dialog;
	render::font_metrics title_font_metrics;
	render::font_metrics label_font_metrics;
	bool fonts_ready = false;

	ui::frame_ptr main_frame;
	std::vector<ui::command_ptr> menus;

	int menu_bar_height_for_scale(const double scale)
	{
		return core::round(menu_bar_height_96 * scale);
	}

	core::sizei minimum_client_size(const double menu_scale = 1.0)
	{
		return {video_width * 3 + layout_padding * 4,
		        video_height + minimum_pane_y * 2 + menu_bar_height_for_scale(menu_scale)};
	}

	core::sizei initial_client_size()
	{
		// These are the old content dimensions plus the same spare title/status margin used by
		// the native window. The frame owns the non-client title bar.
		return {video_width * 3 + layout_padding * 8,
		        video_height + minimum_pane_y * 2 + 60};
	}

	pane_layout compute_layout(const core::sizei extent)
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
			if (core::icmp(param, "/test") == 0) test = true;
			else if (core::icmp(param, "/evaluate") == 0) evaluate = true;
			else if (core::icmp(param, "/probe-h264") == 0) probe = true;
		}

		// Keep the old command-line precedence when a caller supplies more than one
		// mode: the probe was checked first, then /test, then /evaluate.
		if (probe) return command_mode::probe_h264;
		if (test) return command_mode::test;
		if (evaluate) return command_mode::evaluate;
		return command_mode::gui;
	}

	core::surface_ptr make_bgra_surface(const int width, const int height,
	                                    const std::vector<uint8_t>& bgra)
	{
		if (width <= 0 || height <= 0) return {};

		auto result = std::make_shared<core::surface>();
		auto* const pixels = result->alloc(width, height, core::texture_format::ARGB);
		if (!pixels) return {};

		const size_t row_bytes = static_cast<size_t>(width) * 4;
		for (int y = 0; y < height; ++y)
		{
			auto* const dst = pixels + static_cast<size_t>(y) * result->stride();
			const size_t source_offset = static_cast<size_t>(y) * row_bytes;
			const size_t available = source_offset < bgra.size() ? bgra.size() - source_offset : 0;
			const size_t bytes = std::min(row_bytes, available);
			if (bytes != 0) std::memcpy(dst, bgra.data() + source_offset, bytes);
			if (bytes < row_bytes) std::memset(dst + bytes, 0, row_bytes - bytes);
		}
		return result;
	}

	core::surface_ptr make_grey_surface(const image_u8& image)
	{
		if (image.empty()) return {};

		auto result = std::make_shared<core::surface>();
		auto* const pixels = result->alloc(image.width(), image.height(), core::texture_format::ARGB);
		if (!pixels) return {};

		for (int y = 0; y < image.height(); ++y)
		{
			auto* const row = reinterpret_cast<uint32_t*>(pixels + static_cast<size_t>(y) * result->stride());
			for (int x = 0; x < image.width(); ++x)
			{
				const uint32_t grey = image(x, y);
				row[x] = core::rgba(grey, grey, grey);
			}
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
			next->source = make_bgra_surface(next->state->width, next->state->height,
			                                next->state->source_bgra);
			next->lowban = make_grey_surface(next->state->lowban.picture);
			next->baseline = make_grey_surface(next->state->baseline.picture);
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
		title_font = render::font_face::title;
		label_font = render::font_face::dialog;
		title_font_metrics = {};
		label_font_metrics = {};
		fonts_ready = false;
	}

	bool create_fonts()
	{
		render::font_desc title;
		title.family = "Segoe UI";
		title.size = title_height;
		title.cell_size = true;

		render::font_desc label;
		label.family = "Segoe UI";
		label.size = label_height;
		label.cell_size = true;

		title_font = render::make_font(title);
		label_font = render::make_font(label);
		fonts_ready = true;
		return true;
	}

	std::string expression_text(const receiver_view& view)
	{
		if (!view.expression_measured) return "-";
		char text[32] = {};
		std::snprintf(text, sizeof(text), "%0.1f%%", view.expression.overall * 100.0);
		return text;
	}

	void draw_centred(render::draw_context& draw, const core::recti& bounds, const int y,
	                  const std::string_view text, const render::font_face font,
	                  const core::color colour)
	{
		const core::sizei extent = draw.measure_text(text, font, render::text_style::single_line,
		                                             bounds.width());
		const int x = std::max(bounds.left, (bounds.right - extent.cx) / 2);
		const int bottom = std::max(y + 1, y + extent.cy);
		const core::recti text_bounds{x, y, std::min(bounds.right, x + extent.cx), bottom};
		draw.draw_rect(text_bounds, core::color(core::rgba(0, 0, 0)));
		draw.draw_text(text, text_bounds, font, render::text_style::single_line, colour, {});
	}

	void draw_caption(render::draw_context& draw, const int x, const int pane_y, const int line,
	                  const std::string_view text, const render::font_face font)
	{
		const int y = pane_y + video_height + layout_padding + line * label_height;
		const core::recti bounds{x, y, x + video_width, y + label_height};
		draw.draw_rect(bounds, core::color(core::rgba(0, 0, 0)));
		draw.draw_text(text, bounds, font, render::text_style::single_line,
		               core::color(core::rgba(200, 200, 200)), {});
	}

	void draw_face_box(render::draw_context& draw, const int left, const int top, const rect_i& face)
	{
		const core::color colour(core::rgba(0, 220, 0));
		const core::pointd a{static_cast<double>(left + face.left), static_cast<double>(top + face.top)};
		const core::pointd b{static_cast<double>(left + face.right + 1),
		                    static_cast<double>(top + face.bottom + 1)};
		draw.draw_line(a, {b.X, a.Y}, colour, 2.0);
		draw.draw_line({b.X, a.Y}, b, colour, 2.0);
		draw.draw_line(b, {a.X, b.Y}, colour, 2.0);
		draw.draw_line({a.X, b.Y}, a, colour, 2.0);
	}

	void draw_frame(render::draw_context& draw, const core::recti& bounds,
	                const std::string_view status)
	{
		draw.draw_rect(bounds, core::color(core::rgba(0, 0, 0)));

		const auto relative_layout = compute_layout(bounds.extent());
		const pane_layout layout{
			relative_layout.left_x + bounds.left,
			relative_layout.lowban_x + bounds.left,
			relative_layout.baseline_x + bounds.left,
			relative_layout.pane_y + bounds.top};
		draw_centred(draw, bounds, bounds.top + std::max(layout_padding, relative_layout.pane_y / 3),
		             "Low bandwidth video with face landmarking", title_font,
		             core::color(core::rgba(200, 200, 200)));

		ensure_cache();
		if (cache && cache->state)
		{
			const auto& state = *cache->state;
			render::draw_surface(draw, cache->source, {layout.left_x, layout.pane_y,
		                                             layout.left_x + video_width,
		                                             layout.pane_y + video_height});
			render::draw_surface(draw, cache->lowban, {layout.lowban_x, layout.pane_y,
		                                             layout.lowban_x + video_width,
		                                             layout.pane_y + video_height});
			render::draw_surface(draw, cache->baseline, {layout.baseline_x, layout.pane_y,
		                                             layout.baseline_x + video_width,
		                                             layout.pane_y + video_height});

			// Drawn over the source so detection can be judged separately from landmarking.
			for (const auto& face : state.faces) draw_face_box(draw, layout.left_x, layout.pane_y, face);

			char text[192] = {};
			std::snprintf(text, sizeof(text), "Original: %0.1f fps camera, %0.1f fps analysed",
			              pipeline.source_fps(), pipeline.processed_fps());
			draw_caption(draw, layout.left_x, layout.pane_y, 0, text, label_font);
			std::snprintf(text, sizeof(text), "%d faces (peak detector score %0.2f)",
			              state.face_count, state.peak_confidence);
			draw_caption(draw, layout.left_x, layout.pane_y, 1, text, label_font);

			const receiver_view& lowban = state.lowban;
			std::snprintf(text, sizeof(text), "low-ban: %0.1f kbit/s (%0.1f geometry), link %0.1f kbit/s",
			              lowban.payload_kbit_per_second, state.lowban_geometry_kbit_per_second,
			              lowban.link_kbit_per_second);
			draw_caption(draw, layout.lowban_x, layout.pane_y, 0, text, label_font);
			const std::string lowban_expression = expression_text(lowban);
			std::snprintf(text, sizeof(text), "PSNR %0.1f dB, face %0.1f dB, landmark error %s",
			              lowban.quality.psnr, lowban.quality.weighted_psnr, lowban_expression.c_str());
			draw_caption(draw, layout.lowban_x, layout.pane_y, 1, text, label_font);
			std::snprintf(text, sizeof(text), "last picture: %s, %d blocks",
			              state.lowban_keyframe ? "startup keyframe"
			              : state.lowban_warped ? "warped delta" : "delta",
			              state.lowban_updated_blocks);
			draw_caption(draw, layout.lowban_x, layout.pane_y, 2, text, label_font);

			const receiver_view& baseline = state.baseline;
			if (baseline.picture.empty() && !state.baseline_status.empty())
			{
				draw_caption(draw, layout.baseline_x, layout.pane_y, 0, state.baseline_status, label_font);
			}
			else
			{
				std::snprintf(text, sizeof(text), "H.264 baseline: %0.1f kbit/s, link %0.1f kbit/s",
				              baseline.payload_kbit_per_second, baseline.link_kbit_per_second);
				draw_caption(draw, layout.baseline_x, layout.pane_y, 0, text, label_font);
				const std::string baseline_expression = expression_text(baseline);
				std::snprintf(text, sizeof(text), "PSNR %0.1f dB, face %0.1f dB, landmark error %s",
				              baseline.quality.psnr, baseline.quality.weighted_psnr,
				              baseline_expression.c_str());
				draw_caption(draw, layout.baseline_x, layout.pane_y, 1, text, label_font);
				std::snprintf(text, sizeof(text), "%d kbit/s constant rate, %d fps, face offset %s",
				              baseline_kbit_per_second, geometry_hz,
				              state.baseline_face_offset ? "on" : "off");
				draw_caption(draw, layout.baseline_x, layout.pane_y, 2, text, label_font);
			}
		}

		if (!status.empty())
		{
			draw_centred(draw, bounds, bounds.bottom - 3 * layout_padding, status, label_font,
			             core::color(core::rgba(220, 120, 120)));
		}
	}

	void invalidate_after_menu()
	{
		if (main_frame) main_frame->invalidate();
	}

	ui::command_ptr make_toggle_command(const std::string_view text,
	                                    const std::function<bool()>& read,
	                                    const std::function<void()>& toggle)
	{
		auto command = std::make_shared<ui::command>();
		command->text = text;
		command->checkable = true;
		command->checked = read();
		command->invoke = [command, read, toggle]
		{
			toggle();
			command->checked = read();
			invalidate_after_menu();
		};
		return command;
	}

	std::vector<ui::command_ptr> build_menu()
	{
		auto file_record = make_toggle_command("&Record clip",
		                                      [] { return pipeline.recording(); },
		                                      [] { pipeline.set_recording(!pipeline.recording()); });
		auto file_exit = std::make_shared<ui::command>();
		file_exit->text = "E&xit";
		file_exit->invoke = []
		{
			if (main_frame) main_frame->close();
		};

		auto file_separator = std::make_shared<ui::command>();
		auto file = std::make_shared<ui::command>();
		file->text = "&File";
		file->menu = [file_record,
		              children = std::vector<ui::command_ptr>{file_record, file_separator, file_exit}]
		{
			// Capture can stop a clip after an I/O or camera error, so the check belongs to
			// menu construction rather than to the last UI click.
			file_record->checked = pipeline.recording();
			return children;
		};

		auto view_detect = make_toggle_command("Face &detect",
		                                       [] { return pipeline.show_faces(); },
		                                       [] { pipeline.set_show_faces(!pipeline.show_faces()); });
		auto view_landmark = make_toggle_command("Face &landmarking",
		                                         [] { return pipeline.show_landmarks(); },
		                                         [] { pipeline.set_show_landmarks(!pipeline.show_landmarks()); });
		auto view_overlay = make_toggle_command("Landmark &overlay",
		                                        [] { return pipeline.show_overlay(); },
		                                        [] { pipeline.set_show_overlay(!pipeline.show_overlay()); });
		auto view = std::make_shared<ui::command>();
		view->text = "&View";
		view->menu = [children = std::vector<ui::command_ptr>{view_detect, view_landmark, view_overlay}]
		{
			return children;
		};

		auto about = std::make_shared<ui::command>();
		about->text = "&About ...";
		about->invoke = []
		{
			platform::message_box(nullptr,
			                      "low-ban - low bandwidth video experiment\n\nLandmark-warped codec against H.264.",
			                      "About low-ban",
			                      {platform::message_box_buttons::ok, platform::message_box_icon::info});
		};
		auto help = std::make_shared<ui::command>();
		help->text = "&Help";
		help->menu = [children = std::vector<ui::command_ptr>{about}]
		{
			return children;
		};

		return {file, view, help};
	}

	class canvas_element final : public ui::element
	{
		std::function<void(render::draw_context&, core::recti)> _paint;

	public:
		canvas_element(std::function<void(render::draw_context&, core::recti)> paint,
		               const ui::element_options& options) : ui::element(options), _paint(std::move(paint))
		{
		}

		core::sizei measure(render::measure_context&, const int) const override
		{
			return {};
		}

		void render(render::draw_context& dc, const core::pointi offset) const override
		{
			if (_paint) _paint(dc, bounds.offset(offset));
		}
	};

	class low_ban_host final : public ui::element_host,
	                           public std::enable_shared_from_this<low_ban_host>
	{
		ui::elements_ptr _root;
		ui::frame_ptr _frame;

	public:
		explicit low_ban_host(const std::vector<ui::command_ptr>& commands)
		{
			_root = std::make_shared<ui::elements>();
			_root->flex_container.direction = ui::flex_direction::column;
			_root->flex_container.wrap = ui::flex_wrap::no_wrap;
			_root->flex_container.align_items = ui::flex_align::stretch;

			auto menu = ui::make_menu_bar(commands);
			menu->flex.shrink = 0.0f;

			auto canvas = std::make_shared<canvas_element>(
				[](render::draw_context& dc, const core::recti bounds)
				{
					draw_frame(dc, bounds, pipeline.status());
				},
				ui::element_options(ui::flex_item::grow | ui::flex_item::stretch));
			// The presentation owns every pixel in this region; element chrome would leave a
			// frame-coloured strip around the black scene.
			canvas->padding(0);
			canvas->margin(0);

			_root->add(menu);
			_root->add(canvas);
		}

		void set_frame(ui::frame_ptr frame)
		{
			_frame = std::move(frame);
		}

		void on_window_layout(render::measure_context& mc, const core::sizei extent,
		                     const bool is_minimized) override
		{
			_extent = extent;
			if (is_minimized) return;
			if (fonts_ready)
			{
				title_font_metrics = render::metrics(mc, title_font);
				label_font_metrics = render::metrics(mc, label_font);
			}
			_root->layout(mc, core::recti(extent));
		}

		void on_window_paint(render::draw_context& dc) override
		{
			dc.time_now = core::now();
			_root->render(dc, {});
		}

		void frame_tick()
		{
			const uint64_t requested_generation = pipeline.generation();
			if (queue_cached_repaint(requested_generation, painted_generation,
			                         cached_generation, [] { ensure_cache(); }) && _frame)
			{
				_frame->invalidate();
			}
		}

		void visit_elements(const std::function<void(const ui::element_ptr&, core::pointi)>& visit) override
		{
			visit(_root, {});
		}

		ui::controller_ptr controller_from_location(const core::pointi loc,
		                                             ui::hit_test_context& ctx) override
		{
			const auto self = std::static_pointer_cast<ui::element_host>(shared_from_this());
			return _root->controller_from_location(self, loc, {}, ctx);
		}

		void invoke(commands) override
		{
		}

		bool is_command_checked(commands) override
		{
			return false;
		}

		void track_menu(const core::recti bounds,
		                const std::vector<ui::command_ptr>& commands) override
		{
			if (_frame) _frame->track_menu(bounds, commands);
		}

		void invalidate_element(const ui::element_ptr& element) override
		{
			if (_frame) _frame->invalidate(element ? element->invalidate_bounds() : core::recti{});
		}

		void invalidate(const ui::host_invalid what) override
		{
			if (!_frame) return;
			if (what && ui::host_invalid::layout) _frame->layout();
			if (what && ui::host_invalid::redraw) _frame->invalidate();
		}

		void controller_changed() override
		{
			if (_frame) _frame->invalidate();
		}

		const ui::frame_ptr frame() const override
		{
			return _frame ? _frame : ui::no_frame();
		}

		const ui::frame_ptr owner() override
		{
			return _frame ? _frame : ui::no_frame();
		}

		void on_window_destroy() override
		{
			ui::element_host::on_window_destroy();
			_frame.reset();
		}
	};

	class recording_frame final : public ui::null_frame
	{
	public:
		int invalidates = 0;

		void invalidate(core::recti = {}, bool = false) override
		{
			++invalidates;
		}
	};

	class recording_texture final : public render::texture
	{
	public:
		render::texture_update_result update(const render::video_frame_ptr&) override
		{
			return render::texture_update_result::tex_updated;
		}

		render::texture_update_result update(const core::const_surface_ptr&) override
		{
			return render::texture_update_result::tex_updated;
		}

		render::texture_update_result update(core::sizei, core::texture_format, core::orientation,
		                                    const uint8_t*, size_t, size_t) override
		{
			return render::texture_update_result::tex_updated;
		}

		bool is_valid() const override
		{
			return true;
		}
	};

	class recording_draw_context final : public ui::test::fake_draw_context
	{
	public:
		int texture_calls = 0;

		explicit recording_draw_context(const core::recti clip) : ui::test::fake_draw_context(clip)
		{
		}

		render::texture_ptr create_texture() override
		{
			return std::make_shared<recording_texture>();
		}

		void draw_texture(const render::texture_ptr& texture, const core::recti dst,
		                  const float alpha, const render::texture_sampler sampler) override
		{
			++texture_calls;
			ui::test::fake_draw_context::draw_texture(texture, dst, alpha, sampler);
		}
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
		cache->source = make_bgra_surface(state->width, state->height, state->source_bgra);
		cache->lowban = make_grey_surface(state->lowban.picture);
		cache->baseline = make_grey_surface(state->baseline.picture);
		cached_generation = pipeline.generation();
	}

	bool has_recorded_text(const ui::test::fake_draw_context& draw, const std::string_view needle)
	{
		return std::any_of(draw.texts.begin(), draw.texts.end(),
		                   [needle](const ui::test::fake_draw_context::text_draw& text)
		                   {
			                   return text.text.find(needle) != std::string::npos;
		                   });
	}

	void gui_check(const bool condition, const char* description, int& failures)
	{
		if (!condition) ++failures;
		std::printf("  [%s] %s\n", condition ? "pass" : "FAIL", description);
	}

	int run_gui_self_test(const std::function<void()>& app_frame_tick,
	                     const std::function<int()>& invalidation_count)
	{
		std::printf("GUI sizing and command modes\n");
		int failures = 0;
		const bool presentation_fonts_ready = create_fonts();
		auto measure_surface = std::make_shared<core::surface>();
		measure_surface->alloc(1, 1, core::texture_format::ARGB);
		const auto measure_context = render::create_surface_context(measure_surface);
		if (presentation_fonts_ready && measure_context)
		{
			auto& mc = *measure_context;
			title_font_metrics = render::metrics(mc, title_font);
			label_font_metrics = render::metrics(mc, label_font);
		}
		// DirectWrite measures the line each cell request asks for, as the original GDI font laid it out.
		gui_check(presentation_fonts_ready && measure_context &&
		          title_font_metrics.height == title_height &&
		          label_font_metrics.height == label_height,
		          "Segoe UI cell fonts retain the old 48/24 pixel cell requests and measured line heights",
		          failures);

		const core::sizei minimum = minimum_client_size();
		gui_check(minimum == core::sizei(1968, 720 + menu_bar_height_for_scale(1.0)),
		          "minimum client size leaves room for the drawn menu and all captions at 100%", failures);
		const core::sizei scaled_minimum = minimum_client_size(1.5);
		gui_check(scaled_minimum == core::sizei(1968, 720 + menu_bar_height_for_scale(1.5)) &&
		          menu_bar_height_for_scale(1.5) == 42,
		          "minimum client size grows with a scaled menu bar", failures);

		const core::sizei content_minimum{minimum.cx, minimum.cy - menu_bar_height_for_scale(1.0)};
		const pane_layout minimum_layout = compute_layout(content_minimum);
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

		painted_generation = UINT64_MAX;
		cached_generation = pipeline.generation();
		cache.reset();
		const int invalidates_before_tick = invalidation_count();
		app_frame_tick();
		gui_check(invalidation_count() == invalidates_before_tick + 1 &&
		          cached_generation == pipeline.generation(),
		          "the application frame tick caches a generation and invalidates the frame", failures);

		set_recorded_cache(second_state);
		painted_generation = cached_generation;
		recording_draw_context draw({0, 0, 1968, 720});
		draw_frame(draw, draw.clip_bounds(), "camera unavailable");
		// The H.264 pane has no picture while its encoder is unavailable, so two panes draw one.
		gui_check(draw.texture_calls == 2 && draw.lines.size() >= 4 &&
		          has_recorded_text(draw, "low-ban:") &&
		          has_recorded_text(draw, "H.264 unavailable") &&
		          has_recorded_text(draw, "camera unavailable"),
		          "captions, face overlay and status share one render draw_context paint", failures);

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

	int run_console_mode(const command_mode mode, const std::function<int()>& gui_self_test)
	{
		platform::attach_console();
		setvbuf(stdout, nullptr, _IONBF, 0);

		// The GUI checks measure and draw, which belongs to the thread platform::run named; the media
		// checks need an MTA of their own.
		const int gui_result = gui_self_test ? gui_self_test() : 0;
		return run_on_mta([mode, gui_result]
		{
			if (mode == command_mode::evaluate) return run_sample_evaluation();
			if (mode == command_mode::probe_h264) return run_h264_probe();

			const bool mta = is_mta_thread();
			std::printf("  [%s] /test console runtime is an MTA\n", mta ? "pass" : "FAIL");
			const media_foundation media_lifetime;
			const int self_test_result = run_self_test();
			return mta && gui_result == 0 && self_test_result == 0 ? 0 : 1;
		});
	}

	class low_ban_app final : public ui::app,
	                          public std::enable_shared_from_this<low_ban_app>
	{
		command_mode _mode = command_mode::gui;
		platform::setting_file_ptr _settings;
		std::shared_ptr<low_ban_host> _host;
		bool _pipeline_started = false;
		bool _cleaned_up = false;

		void cleanup()
		{
			if (_cleaned_up) return;
			_cleaned_up = true;
			// The fast callback belongs to this app instance; leaving it armed would tick a
			// destroyed frame while platform::run is tearing down.
			platform::set_frame_delay(0);
			if (_pipeline_started)
			{
				pipeline.stop();
				_pipeline_started = false;
			}
			media.reset();
			cache.reset();
			menus.clear();
			_host.reset();
			main_frame.reset();
			destroy_fonts();
		}

	public:
		~low_ban_app() override
		{
			cleanup();
		}

		bool load_settings(const platform::setting_file_ptr& store) override
		{
			_settings = store;
			return true;
		}

		bool parse_command_line(const std::string_view command_line) override
		{
			const auto params = core::split(command_line, true);
			_mode = parse_mode(params);
			return _mode != command_mode::gui;
		}

		int run_console_mode() override
		{
			if (_mode != command_mode::test) return ::run_console_mode(_mode, {});
			const auto test_frame = std::make_shared<recording_frame>();
			_host = std::make_shared<low_ban_host>(std::vector<ui::command_ptr>{});
			_host->set_frame(test_frame);
			const int result = ::run_console_mode(_mode, [this, test_frame]
			{
				return run_gui_self_test([this] { frame_tick(); },
				                           [test_frame] { return test_frame->invalidates; });
			});
			_host.reset();
			return result;
		}

		void frame_tick() override
		{
			if (_host) _host->frame_tick();
		}

		bool wants_gpu() const override
		{
			return true;
		}

		bool key_down(const char32_t key, const platform::key_state keys) override
		{
			return ui::dispatch_accelerator(menus, key, keys);
		}

		bool text_input(std::string_view) override
		{
			return false;
		}

		ui::focus_mode focus_mode() const override
		{
			return ui::focus_mode::none;
		}

		void track_menu(const ui::frame_ptr& parent, const core::recti bounds,
		                const std::vector<ui::command_ptr>& commands) override
		{
			if (parent) parent->track_menu(bounds, commands);
		}

		bool init(std::string_view) override
		{
			if (_mode != command_mode::gui) return false;
			const bool placement_restored = platform::read_placement(
				_settings, platform::placement_setting_name()).has_value();
			if (!create_fonts())
			{
				platform::show_error("Could not create the Segoe UI presentation fonts.");
				return false;
			}

			menus = build_menu();
			_host = std::make_shared<low_ban_host>(menus);

			ui::frame_style style;
			style.hardware_accelerated = wants_gpu();
			style.drawn_caption = false;
			style.client_size = initial_client_size();
			const double menu_scale = static_cast<double>(platform::primary_screen_dpi()) / 96.0;
			style.minimum_client = minimum_client_size(menu_scale);
			style.resizable = true;

			main_frame = ui::create_app_frame(shared_from_this(), _settings, _host, style);
			if (!main_frame)
			{
				platform::show_error("Could not create the low-ban presentation window.");
				platform::quit(1);
				return false;
			}
			_host->set_frame(main_frame);
			main_frame->window_text("low-ban");

			if (!placement_restored)
			{
				const auto screen = platform::primary_screen_size();
				const auto desired = initial_client_size();
				if (!screen.is_empty())
				{
					const core::recti centered{
						(screen.cx - desired.cx) / 2, (screen.cy - desired.cy) / 2,
						(screen.cx + desired.cx) / 2, (screen.cy + desired.cy) / 2};
					main_frame->window_bounds(centered, false);
				}
			}

			platform::set_frame_delay(16);
			media = std::make_unique<media_foundation>();
			pipeline.start(video_width, video_height);
			_pipeline_started = true;
			return true;
		}

		bool can_exit() override
		{
			return true;
		}

		void exit() override
		{
			cleanup();
		}

		void final_exit() override
		{
			cleanup();
		}
	};
}

int main(int argc, char** argv)
{
	core::set_app_name("low-ban");
	for (int i = 1; i < argc; ++i)
	{
		if (core::icmp(std::string_view(argv[i]), "/probe-h264") == 0)
			return run_on_mta([] { return run_h264_probe(); });
	}
	auto app = std::make_shared<low_ban_app>();
	return platform::run(app, argc, argv);
}
