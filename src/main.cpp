// low-ban : low bandwidth video with face landmarking.
//
// The window is deliberately thin: it owns the message loop and the three panes, and does no
// image processing of its own. Everything else lives in engine/detect/landmarks/codec.
//
// Command line:
//   /test   run the headless self test and exit with a non-zero code on failure
#include <SDKDDKVer.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <objbase.h>

#include <cstdio>
#include <memory>
#include <string>

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
	constexpr UINT_PTR refresh_timer = 1;

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
		static const HFONT title_font = CreateFontW(48, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
		                                            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
		                                            DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
		static const HFONT label_font = CreateFontW(28, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
		                                            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
		                                            DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

		const int panes_width = video_width * 3 + layout_padding * 2;
		const int left_x = (bounds.right - panes_width) / 2;
		const int traditional_x = left_x + video_width + layout_padding;
		const int neural_x = traditional_x + video_width + layout_padding;
		const int pane_y = (bounds.bottom - video_height) / 2;

		SetTextColor(hdc, RGB(200, 200, 200));
		SetBkMode(hdc, TRANSPARENT);

		const HGDIOBJ previous_font = SelectObject(hdc, title_font);
		draw_centred(hdc, bounds, std::max(layout_padding, pane_y / 3), L"Low bandwidth video with face landmarking");
		SelectObject(hdc, label_font);

		const auto state = pipeline.latest();
		const double source_fps = pipeline.source_fps();

		if (state)
		{
			blit_colour(hdc, left_x, pane_y, state->width, state->height, state->source_bgra);
			blit_grey(hdc, traditional_x, pane_y, state->traditional);
			blit_grey(hdc, neural_x, pane_y, state->neural);

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

			// Reference figure for the raw camera feed under conventional compression.
			const auto source_kbit = static_cast<int>(
				state->width * state->height * 24.0 * source_fps * 0.05 / 1000.0);

			wchar_t text[128];
			swprintf_s(text, L"Original: %0.1f fps, %d kbit/s reference, %d faces (peak %0.2f)", source_fps,
			           source_kbit, state->face_count, state->peak_confidence);
			TextOutW(hdc, left_x, pane_y + video_height + layout_padding, text, static_cast<int>(wcslen(text)));

			swprintf_s(text, L"Traditional wire: %0.1f kbit/s, PSNR %0.1f dB, face %0.1f dB, %s %d blocks",
			           state->traditional_kbit_per_second, state->traditional_quality.psnr,
			           state->traditional_quality.weighted_psnr,
			           state->traditional_keyframe ? L"keyframe" : L"delta",
			           state->traditional_updated_blocks);
			TextOutW(hdc, traditional_x, pane_y + video_height + layout_padding, text,
			         static_cast<int>(wcslen(text)));

			swprintf_s(text, L"Neural wire: %0.1f kbit/s, PSNR %0.1f dB, face %0.1f dB, %s %d patches",
			           state->neural_kbit_per_second, state->neural_quality.psnr,
			           state->neural_quality.weighted_psnr, state->neural_keyframe ? L"keyframe" : L"delta",
			           state->neural_updated_patches);
			TextOutW(hdc, neural_x, pane_y + video_height + layout_padding, text,
			         static_cast<int>(wcslen(text)));
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
			case IDM_AUTOENCODER:
				pipeline.reset_training();
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

	int run_console_mode(const bool evaluate)
	{
		if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole())
		{
			FILE* stream = nullptr;
			freopen_s(&stream, "CONOUT$", "w", stdout);
			freopen_s(&stream, "CONOUT$", "w", stderr);
		}

		return evaluate ? run_sample_evaluation() : run_self_test();
	}
}

int APIENTRY wWinMain(_In_ const HINSTANCE hinstance, _In_opt_ HINSTANCE, _In_ const LPWSTR command_line,
                      _In_ const int show_command)
{
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
	                               video_width * 3 + layout_padding * 8, video_height + 220, nullptr, nullptr,
	                               hinstance, nullptr);
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
