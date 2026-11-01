// low-ban : low bandwidth video with face landmarking.
//
// The window is deliberately thin: it owns the message loop and the three panes, and does no
// image processing of its own. Everything else lives in engine/session and the modules below.
//
// Command line:
//   /test         run the headless self test and exit with a non-zero code on failure
//   /evaluate     score samples/*.png and samples/*.mp4 offline and print CSV
//   /probe-h264   one H.264 round trip, run as a child to find out whether it crashes
#include <SDKDDKVer.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <objbase.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

#include "baseline.h"
#include "capture.h"
#include "engine.h"
#include "res.h"

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
	constexpr UINT_PTR refresh_timer = 1;

	// The title sits a third of the way down the space above the panes and the status line
	// three paddings above the bottom. This much space above the panes keeps the title clear
	// of them and the status line clear of the captions.
	constexpr int minimum_pane_y = std::max(title_height * 3 / 2, layout_padding * 4 + label_height * caption_lines);

	// The panes are fixed-size bitmaps, so a smaller window would cut them off and overlap the
	// title, captions and status line.
	POINT minimum_window_size(const HWND wnd)
	{
		RECT rect = {0, 0, video_width * 3 + layout_padding * 4, video_height + minimum_pane_y * 2};
		AdjustWindowRectExForDpi(&rect, WS_OVERLAPPEDWINDOW, TRUE, 0, GetDpiForWindow(wnd));
		return {rect.right - rect.left, rect.bottom - rect.top};
	}

	HINSTANCE instance = nullptr;
	engine pipeline;
	uint64_t painted_generation = 0;

	// 8 bit DIB rows are DWORD aligned, so odd widths need a padded copy.
	void blit_grey(const HDC hdc, const int x, const int y, const image_u8& img)
	{
		if (img.empty()) return;

		struct
		{
			BITMAPINFOHEADER header;
			RGBQUAD palette[256];
		} bmi = {};

		bmi.header.biSize = sizeof(bmi.header);
		bmi.header.biWidth = img.width();
		bmi.header.biHeight = -img.height();
		bmi.header.biPlanes = 1;
		bmi.header.biBitCount = 8;
		bmi.header.biCompression = BI_RGB;
		bmi.header.biClrUsed = 256;

		for (int i = 0; i < 256; ++i)
		{
			bmi.palette[i].rgbRed = static_cast<BYTE>(i);
			bmi.palette[i].rgbGreen = static_cast<BYTE>(i);
			bmi.palette[i].rgbBlue = static_cast<BYTE>(i);
		}

		const int stride = (img.width() + 3) & ~3;
		if (stride == img.width())
		{
			SetDIBitsToDevice(hdc, x, y, img.width(), img.height(), 0, 0, 0, img.height(), img.data(),
			                  reinterpret_cast<BITMAPINFO*>(&bmi), DIB_RGB_COLORS);
			return;
		}

		static std::vector<uint8_t> padded;
		padded.assign(static_cast<size_t>(stride) * img.height(), 0);
		for (int row = 0; row < img.height(); ++row)
		{
			memcpy(padded.data() + static_cast<size_t>(row) * stride, img.row(row), img.width());
		}

		SetDIBitsToDevice(hdc, x, y, img.width(), img.height(), 0, 0, 0, img.height(), padded.data(),
		                  reinterpret_cast<BITMAPINFO*>(&bmi), DIB_RGB_COLORS);
	}

	void blit_colour(const HDC hdc, const int x, const int y, const int width, const int height,
	                 const std::vector<uint8_t>& bgra)
	{
		if (bgra.size() < static_cast<size_t>(width) * height * 4) return;

		BITMAPINFO bmi = {};
		bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
		bmi.bmiHeader.biWidth = width;
		bmi.bmiHeader.biHeight = -height;
		bmi.bmiHeader.biPlanes = 1;
		bmi.bmiHeader.biBitCount = 32;
		bmi.bmiHeader.biCompression = BI_RGB;

		SetDIBitsToDevice(hdc, x, y, width, height, 0, 0, 0, height, bgra.data(), &bmi, DIB_RGB_COLORS);
	}

	void draw_centred(const HDC hdc, const RECT& bounds, const int y, const wchar_t* text)
	{
		SIZE extent = {};
		const int len = static_cast<int>(wcslen(text));
		GetTextExtentPoint32W(hdc, text, len, &extent);
		TextOutW(hdc, (bounds.right - extent.cx) / 2, y, text, len);
	}

	// One caption line under a pane, `line` lines below its bottom edge.
	void draw_caption(const HDC hdc, const int x, const int pane_y, const int line, const wchar_t* text)
	{
		TextOutW(hdc, x, pane_y + video_height + layout_padding + line * label_height, text,
		         static_cast<int>(wcslen(text)));
	}

	// Landmark error as a percentage of the eye-corner distance, or a dash before there is a
	// received picture and a landmarked face to measure.
	std::wstring expression_text(const receiver_view& view)
	{
		if (!view.expression_measured) return L"-";
		wchar_t text[32];
		swprintf_s(text, L"%0.1f%%", view.expression.overall * 100.0);
		return text;
	}

	INT_PTR CALLBACK about_proc(const HWND dialog, const UINT message, const WPARAM wparam, LPARAM)
	{
		switch (message)
		{
		case WM_INITDIALOG:
			return TRUE;
		case WM_COMMAND:
			if (LOWORD(wparam) == IDOK || LOWORD(wparam) == IDCANCEL)
			{
				EndDialog(dialog, LOWORD(wparam));
				return TRUE;
			}
			break;
		default:
			break;
		}
		return FALSE;
	}

	void on_paint(const HDC hdc, const RECT& bounds)
	{
		static const HFONT title_font = CreateFontW(title_height, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
		                                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
		                                            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
		static const HFONT label_font = CreateFontW(label_height, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
		                                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
		                                            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

		const int panes_width = video_width * 3 + layout_padding * 2;
		const int left_x = (bounds.right - panes_width) / 2;
		const int lowban_x = left_x + video_width + layout_padding;
		const int baseline_x = lowban_x + video_width + layout_padding;
		const int pane_y = (bounds.bottom - video_height) / 2;

		SetTextColor(hdc, RGB(200, 200, 200));
		SetBkMode(hdc, TRANSPARENT);

		const HGDIOBJ previous_font = SelectObject(hdc, title_font);
		draw_centred(hdc, bounds, std::max(layout_padding, pane_y / 3), L"Low bandwidth video with face landmarking");
		SelectObject(hdc, label_font);

		const auto state = pipeline.latest();

		if (state)
		{
			blit_colour(hdc, left_x, pane_y, state->width, state->height, state->source_bgra);
			blit_grey(hdc, lowban_x, pane_y, state->lowban.picture);
			blit_grey(hdc, baseline_x, pane_y, state->baseline.picture);

			// Drawn over the source so detection can be judged separately from landmarking.
			static const HPEN face_pen = CreatePen(PS_SOLID, 2, RGB(0, 220, 0));
			const HGDIOBJ previous_pen = SelectObject(hdc, face_pen);
			const HGDIOBJ previous_brush = SelectObject(hdc, GetStockObject(NULL_BRUSH));
			for (const auto& face : state->faces)
			{
				Rectangle(hdc, left_x + face.left, pane_y + face.top,
				          left_x + face.right + 1, pane_y + face.bottom + 1);
			}
			SelectObject(hdc, previous_brush);
			SelectObject(hdc, previous_pen);

			wchar_t text[160];
			swprintf_s(text, L"Original: %0.1f fps camera, %0.1f fps analysed", pipeline.source_fps(),
			           pipeline.processed_fps());
			draw_caption(hdc, left_x, pane_y, 0, text);
			swprintf_s(text, L"%d faces (peak detector score %0.2f)", state->face_count, state->peak_confidence);
			draw_caption(hdc, left_x, pane_y, 1, text);

			const receiver_view& lowban = state->lowban;
			swprintf_s(text, L"low-ban: %0.1f kbit/s (%0.1f geometry), link %0.1f kbit/s",
			           lowban.payload_kbit_per_second, state->lowban_geometry_kbit_per_second,
			           lowban.link_kbit_per_second);
			draw_caption(hdc, lowban_x, pane_y, 0, text);
			swprintf_s(text, L"PSNR %0.1f dB, face %0.1f dB, landmark error %s", lowban.quality.psnr,
			           lowban.quality.weighted_psnr, expression_text(lowban).c_str());
			draw_caption(hdc, lowban_x, pane_y, 1, text);
			swprintf_s(text, L"last picture: %s, %d blocks", state->lowban_keyframe ? L"startup keyframe"
			           : state->lowban_warped ? L"warped delta" : L"delta", state->lowban_updated_blocks);
			draw_caption(hdc, lowban_x, pane_y, 2, text);

			const receiver_view& baseline = state->baseline;
			if (baseline.picture.empty() && !state->baseline_status.empty())
			{
				const std::wstring status(state->baseline_status.begin(), state->baseline_status.end());
				draw_caption(hdc, baseline_x, pane_y, 0, status.c_str());
			}
			else
			{
				swprintf_s(text, L"H.264 baseline: %0.1f kbit/s, link %0.1f kbit/s",
				           baseline.payload_kbit_per_second, baseline.link_kbit_per_second);
				draw_caption(hdc, baseline_x, pane_y, 0, text);
				swprintf_s(text, L"PSNR %0.1f dB, face %0.1f dB, landmark error %s", baseline.quality.psnr,
				           baseline.quality.weighted_psnr, expression_text(baseline).c_str());
				draw_caption(hdc, baseline_x, pane_y, 1, text);
				swprintf_s(text, L"%d kbit/s constant rate, %d fps, face offset %s", baseline_kbit_per_second,
				           geometry_hz, state->baseline_face_offset ? L"on" : L"off");
				draw_caption(hdc, baseline_x, pane_y, 2, text);
			}
		}

		const std::string status = pipeline.status();
		if (!status.empty())
		{
			const std::wstring wide(status.begin(), status.end());
			SetTextColor(hdc, RGB(220, 120, 120));
			draw_centred(hdc, bounds, bounds.bottom - 3 * layout_padding, wide.c_str());
		}

		SelectObject(hdc, previous_font);
	}

	LRESULT CALLBACK wnd_proc(const HWND wnd, const UINT message, const WPARAM wparam, const LPARAM lparam)
	{
		static RECT client_bounds = {};
		static HBITMAP back_buffer = nullptr;

		switch (message)
		{
		case WM_COMMAND:
			switch (LOWORD(wparam))
			{
			case IDM_ABOUT:
				DialogBoxW(instance, MAKEINTRESOURCEW(IDD_ABOUTBOX), wnd, about_proc);
				break;
			case IDM_RECORD:
				pipeline.set_recording(!pipeline.recording());
				break;
			case IDM_FACEDETECT:
				pipeline.set_show_faces(!pipeline.show_faces());
				break;
			case IDM_FACELANDMARK:
				pipeline.set_show_landmarks(!pipeline.show_landmarks());
				break;
			case IDM_LANDMARKOVERLAY:
				pipeline.set_show_overlay(!pipeline.show_overlay());
				break;
			case IDM_EXIT:
				DestroyWindow(wnd);
				break;
			default:
				return DefWindowProcW(wnd, message, wparam, lparam);
			}
			InvalidateRect(wnd, nullptr, FALSE);
			break;

		case WM_INITMENUPOPUP:
			{
				const auto menu = reinterpret_cast<HMENU>(wparam);
				const auto check = [menu](const UINT id, const bool on)
				{
					MENUITEMINFOW info = {sizeof(MENUITEMINFOW), MIIM_STATE};
					info.fState = on ? MFS_CHECKED : MFS_UNCHECKED;
					SetMenuItemInfoW(menu, id, FALSE, &info);
				};

				check(IDM_RECORD, pipeline.recording());
				check(IDM_FACEDETECT, pipeline.show_faces());
				check(IDM_FACELANDMARK, pipeline.show_landmarks());
				check(IDM_LANDMARKOVERLAY, pipeline.show_overlay());
			}
			break;

		case WM_CREATE:
		case WM_SIZE:
			{
				GetClientRect(wnd, &client_bounds);
				if (back_buffer) DeleteObject(back_buffer);
				back_buffer = nullptr;

				if (client_bounds.right > 0 && client_bounds.bottom > 0)
				{
					const HDC hdc = GetDC(wnd);
					back_buffer = CreateCompatibleBitmap(hdc, client_bounds.right, client_bounds.bottom);
					ReleaseDC(wnd, hdc);
				}
			}
			break;

		case WM_ERASEBKGND:
			return 1;

		case WM_GETMINMAXINFO:
			{
				auto* limits = reinterpret_cast<MINMAXINFO*>(lparam);
				limits->ptMinTrackSize = minimum_window_size(wnd);
			}
			break;

		case WM_PAINT:
			{
				PAINTSTRUCT ps;
				const HDC hdc = BeginPaint(wnd, &ps);

				if (back_buffer)
				{
					const HDC buffer_dc = CreateCompatibleDC(hdc);
					const HGDIOBJ previous = SelectObject(buffer_dc, back_buffer);
					FillRect(buffer_dc, &client_bounds, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
					on_paint(buffer_dc, client_bounds);
					BitBlt(hdc, 0, 0, client_bounds.right, client_bounds.bottom, buffer_dc, 0, 0, SRCCOPY);
					SelectObject(buffer_dc, previous);
					DeleteDC(buffer_dc);
				}

				EndPaint(wnd, &ps);
			}
			break;

		case WM_TIMER:
			if (pipeline.generation() != painted_generation)
			{
				painted_generation = pipeline.generation();
				InvalidateRect(wnd, nullptr, FALSE);
			}
			break;

		case WM_DESTROY:
			KillTimer(wnd, refresh_timer);
			if (back_buffer)
			{
				DeleteObject(back_buffer);
				back_buffer = nullptr;
			}
			PostQuitMessage(0);
			break;

		default:
			return DefWindowProcW(wnd, message, wparam, lparam);
		}

		return 0;
	}

	// A pipe or file on stdout means the caller is capturing the text - CTest, `dd evaluate`,
	// `| Out-String` - so it has to stay there rather than be sent to a console it cannot read.
	bool stdout_is_redirected()
	{
		const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
		if (out == nullptr || out == INVALID_HANDLE_VALUE) return false;
		const DWORD type = GetFileType(out);
		return type == FILE_TYPE_PIPE || type == FILE_TYPE_DISK;
	}

	int run_console_mode(const bool evaluate)
	{
		if (!stdout_is_redirected() && (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()))
		{
			FILE* stream = nullptr;
			freopen_s(&stream, "CONOUT$", "w", stdout);
			freopen_s(&stream, "CONOUT$", "w", stderr);
		}

		// Unbuffered, so a run that dies part-way still leaves every check it reached in the log.
		setvbuf(stdout, nullptr, _IONBF, 0);

		if (evaluate) return run_sample_evaluation();

		// The H.264 and clip checks drive Media Foundation from this thread.
		const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
		int result = 1;
		{
			const media_foundation media;
			result = run_self_test();
		}
		if (SUCCEEDED(com)) CoUninitialize();
		return result;
	}
}

int APIENTRY wWinMain(_In_ const HINSTANCE hinstance, _In_opt_ HINSTANCE, _In_ const LPWSTR command_line,
                      _In_ const int show_command)
{
	if (command_line != nullptr && wcsstr(command_line, L"/probe-h264") != nullptr)
	{
		return run_h264_probe();
	}
	if (command_line != nullptr && wcsstr(command_line, L"/test") != nullptr)
	{
		return run_console_mode(false);
	}
	if (command_line != nullptr && wcsstr(command_line, L"/evaluate") != nullptr)
	{
		return run_console_mode(true);
	}

	instance = hinstance;

	const HRESULT com = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
	const bool com_ready = SUCCEEDED(com) || com == RPC_E_CHANGED_MODE;
	const media_foundation media;

	wchar_t title[128] = {};
	wchar_t class_name[128] = {};
	LoadStringW(hinstance, IDS_APP_TITLE, title, ARRAYSIZE(title));
	LoadStringW(hinstance, IDC_APP, class_name, ARRAYSIZE(class_name));

	WNDCLASSEXW wcex = {sizeof(WNDCLASSEXW)};
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = wnd_proc;
	wcex.hInstance = hinstance;
	wcex.hIcon = LoadIconW(hinstance, MAKEINTRESOURCEW(IDI_APP));
	wcex.hIconSm = wcex.hIcon;
	wcex.hCursor = LoadCursorW(nullptr, IDC_ARROW);
	wcex.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
	wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_APP);
	wcex.lpszClassName = class_name;
	RegisterClassExW(&wcex);

	const HWND wnd = CreateWindowW(class_name, title, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
	                               video_width * 3 + layout_padding * 8, video_height + minimum_pane_y * 2 + 60,
	                               nullptr, nullptr, hinstance, nullptr);
	if (wnd == nullptr)
	{
		if (com_ready) CoUninitialize();
		return 1;
	}

	pipeline.start(video_width, video_height);
	ShowWindow(wnd, show_command);
	UpdateWindow(wnd);
	SetTimer(wnd, refresh_timer, 1000 / 60, nullptr);

	MSG msg = {};
	while (GetMessageW(&msg, nullptr, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessageW(&msg);
	}

	pipeline.stop();

	if (com_ready) CoUninitialize();

	return static_cast<int>(msg.wParam);
}
