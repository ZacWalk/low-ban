// low_ban.auto_encoder.cpp : Defines the entry point for the application.
//

#include <SDKDDKVer.h>

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>
#include <atlbase.h>

#include <Mfapi.h>
#include <Mfidl.h>
#include <Mfreadwrite.h>

#pragma comment(lib, "Mfplat.lib")
#pragma comment(lib, "Mfuuid.lib")
#pragma comment(lib, "Mf.lib")
#pragma comment(lib, "Mfreadwrite.lib")

#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "res.h"

#include "../dlib/dnn.h"
#include "../dlib/image_processing/frontal_face_detector.h"
#include "../dlib/image_processing/render_face_detections.h"
#include "../dlib/image_processing.h"



template <class T> void SafeRelease(T **ppT)
{
	if (*ppT)
	{
		(*ppT)->Release();
		*ppT = nullptr;
	}
}

const int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_2 = 0;
const int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_ = 0;
const int DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_10_0 = 0;

enum class frame_mode_t
{
	edgedetect,
	autoencoder,
	scale
};

const int MAX_LOADSTRING = 100;
frame_mode_t frame_mode = frame_mode_t::scale;
bool exit_app = false;
bool show_facedetect = true; // http://dlib.net/face_landmark_detection_ex.cpp.html
bool show_facelandmark = true;
bool invalidate = false;

// Global Variables:
HINSTANCE hInst;                                // current instance
HWND hWnd;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

INT_PTR CALLBACK about_proc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}

CComPtr<IMFSourceReader> pSourceReader;

struct frame_buffer
{
	byte *pixels;
	int width, height;
};

class yuv_frame
{
private:
	long frame_width = 0, frame_height = 0;
	int frame_stride = 0;

	byte *frame_buffer = nullptr;
	size_t frame_buffer_len = 0;

public:

	yuv_frame() = default;

	yuv_frame(int frame_width, int frame_height, int frame_stride, byte* frame_buffer, size_t frame_buffer_len)
		: frame_width(frame_width),
		frame_height(frame_height),
		frame_stride(frame_stride),
		frame_buffer_len(frame_buffer_len)
	{
		this->frame_buffer = new byte[frame_buffer_len];
		std::memcpy(this->frame_buffer, frame_buffer, frame_buffer_len);
	}

	~yuv_frame()
	{
		delete[] frame_buffer;
	}

	void populate_array2d(dlib::array2d<unsigned char>& img) const
	{
		auto video_width = std::min(img.nc(), frame_width);
		auto video_height = std::min(img.nr(), frame_height);

		for (int y = 0; y < video_height; y++)
		{
			for (int x = 0; x < video_width; x++)
			{
				img[y][x] = frame_buffer[y * frame_stride + x * 2];
			}
		}
	}

	void populate_matrix(dlib::matrix<float>& m) const
	{
		auto video_width = std::min(m.nc(), frame_width);
		auto video_height = std::min(m.nr(), frame_height);

		for (int y = 0; y < video_height; y++)
		{
			for (int x = 0; x < video_width; x++)
			{
				m(y, x) = frame_buffer[y * frame_stride + x * 2] / 255.0f;
			}
		}
	}

	void populate_edge_matrix(dlib::matrix<float>& m) const
	{
		auto video_width = std::min(m.nc(), frame_width);
		auto video_height = std::min(m.nr(), frame_height);

		dlib::array2d<unsigned char> img(video_height, video_width);
		populate_array2d(img);

		dlib::array2d<unsigned char> blurred_img;
		dlib::gaussian_blur(img, blurred_img);

		dlib::array2d<short> horz_gradient, vert_gradient;
		dlib::array2d<unsigned char> edge_image;
		dlib::sobel_edge_detector(blurred_img, horz_gradient, vert_gradient);

		dlib::suppress_non_maximum_edges(horz_gradient, vert_gradient, edge_image);

		for (int y = 0; y < video_height; y++)
		{
			for (int x = 0; x < video_width; x++)
			{
				m(y, x) = (255 - edge_image[y][x]) / 255.0f;
			}
		}
	}

	void populate_grayscale_frame(byte *frame, const long width, const long height) const
	{
		auto video_width = std::min(width, frame_width);
		auto video_height = std::min(height, frame_height);

		for (int y = 0; y < video_height; y++)
		{
			for (int x = 0; x < video_width; x++)
			{
				frame[y * width + x] = frame_buffer[y * frame_stride + x * 2];
			}
		}
	}

	void populate_rgb_frame(byte *frame, const long width, const long height) const
	{
		auto video_width = std::min(width, frame_width);
		auto video_height = std::min(height, frame_height);

		for (int y = 0; y < video_height; y++)
		{
			for (int x = 0; x < video_width; x += 2)
			{
				auto i = y * frame_stride + x * 2;
				auto j = (y * width * 4) + (x * 4);

				float y = frame_buffer[i];
				float u = frame_buffer[i + 1];
				float y2 = frame_buffer[i + 2];
				float v = frame_buffer[i + 3];

				frame[j + 2] = get_red(y, u, v);
				frame[j + 1] = get_green(y, u, v);
				frame[j + 0] = get_blue(y, u, v);

				frame[j + 6] = get_red(y2, u, v);
				frame[j + 5] = get_green(y2, u, v);
				frame[j + 4] = get_blue(y2, u, v);
			}
		}
	}

	static int get_red(float y, float u, float v)
	{
		float px = 1.164f*(y - 16.0f) + 1.596f*(v - 128.0f);
		if (px < 0) return 0;
		if (px > 255) return 255;
		return (int)px;
	}

	static int get_green(float y, float u, float v)
	{
		float px = 1.164f*(y - 16.0f) - 0.813f*(v - 128.0f) - 0.391f*(u - 128.0f);
		if (px < 0) return 0;
		if (px > 255) return 255;
		return (int)px;
	}

	static int get_blue(float y, float u, float v)
	{
		float px = 1.164f*(y - 16.0f) + 2.018*(u - 128.0f);
		if (px < 0) return 0;
		if (px > 255) return 255;
		return (int)px;
	}

};

class frame_rate
{
private:

	static const int MAXSAMPLES = 30;

	double tick_last = GetTickCount();
	int tickindex = 0;
	double ticksum = 0;
	double ticklist[MAXSAMPLES];
	double current_fps = 0;

public:

	frame_rate()
	{
		for (auto &t : ticklist)
		{
			t = 0.0;
		}
	}

	void tick()
	{
		double tick = GetTickCount();
		current_fps = 1000 / average_tick(tick - tick_last);
		tick_last = tick;
	}

	double val() const
	{
		return current_fps;
	}

private:

	double average_tick(double newtick)
	{
		ticksum -= ticklist[tickindex];  /* subtract value falling off */
		ticksum += newtick;              /* add new value */
		ticklist[tickindex] = newtick;   /* save new value so it can be subtracted later */

		if (++tickindex == MAXSAMPLES)    /* inc buffer index */
			tickindex = 0;

		return((double)ticksum / MAXSAMPLES);
	}
};

typedef struct tagBITMAPINFO2 {
	BITMAPINFOHEADER    bmiHeader;
	RGBQUAD             bmiColors[256];
} BITMAPINFO2;

typedef struct tagLOGPALETTE2 {
	WORD        palVersion;
	WORD        palNumEntries;
	PALETTEENTRY        palPalEntry[256];
} LOGPALETTE2;

const long video_width = 640;
const long video_height = 480;

const int ae_xy = 10;
const int ae_stride = 10;

frame_rate video_fps;
frame_rate frame_fps;

std::shared_ptr<yuv_frame> current_frame;

using ae_net_type = dlib::loss_mean_squared_per_pixel<
	dlib::cont<1, ae_xy, ae_xy, ae_stride, ae_stride,
	dlib::relu<dlib::con<16, ae_xy, ae_xy, ae_stride, ae_stride,
	dlib::input<dlib::matrix<float>>>>>>;

ae_net_type ae_net;

std::vector<dlib::rectangle> face_rects;
std::vector<dlib::full_object_detection> face_shapes;
std::shared_ptr<yuv_frame> face_frame;
dlib::array2d<unsigned char> face_pyramid;
std::vector<dlib::rectangle> face_dets;
std::shared_ptr<yuv_frame> face_det_frame;

float face_scal_x = 1;
float face_scal_y = 1;

std::mutex frame_mutex;
std::mutex face_mutex;

std::vector<dlib::matrix<float>> training_data;

void add_training_data(const dlib::matrix<float> &mm)
{
	std::lock_guard<std::mutex> guard(frame_mutex);
	if (training_data.size() < 16)
	{
		OutputDebugStringW(L"Add training data\n");
		training_data.emplace_back(mm);
	}
}

std::vector<dlib::matrix<float>> resize_data(long nr, long nc, const std::vector<dlib::matrix<float>> &input)
{
	std::vector<dlib::matrix<float>> results;

	for (const auto & m : input)
	{
		dlib::matrix<float> r(nr, nc);

		for (int y = 0; y < nr; y++)
		{
			for (int x = 0; x < nc; x++)
			{
				r(y, x) = m(y, x);
			}
		}

		results.emplace_back(r);
	}

	return results;
}

int training_step = 0;
const int max_training_step = 3;

void perform_training()
{
	ae_net_type net;

	while (!exit_app)
	{
		if (training_step < max_training_step && frame_mode == frame_mode_t::autoencoder)
		{
			if (training_step == 0)
			{
				ae_net_type empty;
				net = empty;
			}

			auto frame = current_frame;

			if (frame != nullptr)
			{
				training_step += 1;
				OutputDebugStringW(L"Start Training\n");

				std::vector<dlib::matrix<float>> data;
				dlib::matrix<float> m(video_height, video_width);

				frame->populate_edge_matrix(m);

				data.emplace_back(m);

				dlib::sgd defsolver(0, 0.9);
				dlib::dnn_trainer<ae_net_type> trainer(net, defsolver);
				trainer.set_learning_rate(0.025);
				trainer.set_max_num_epochs(100000);

				auto sample_result = net(data[0]);
				auto expected = resize_data(sample_result.nr(), sample_result.nc(), data);

				trainer.train(data, expected);

				{
					std::lock_guard<std::mutex> guard(frame_mutex);
					ae_net = net;
					invalidate = true;
				}

				OutputDebugStringW(L"End Training\n");


			}
		}
		else
		{
			Sleep(100);
		}

		Sleep(0);
	}
}

void perform_face_detect()
{
	//auto face_detector = dlib::get_frontal_face_detector();

	typedef dlib::pyramid_down<6> pyr_t;
	typedef dlib::object_detector<dlib::scan_fhog_pyramid<pyr_t>> frontal_face_detector;

	frontal_face_detector face_detector;

	{
		std::istringstream sin(dlib::get_serialized_frontal_faces());
		dlib::deserialize(face_detector, sin);
	}

	dlib::array2d<unsigned char> img(video_height, video_width);
	dlib::array2d<unsigned char> pyramid;

	// Compile Dlib in Release Mode with Optimizations turned on
	// https://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/

	while (!exit_app)
	{
		if (show_facedetect)
		{
			auto frame = current_frame;

			if (frame != nullptr)
			{
				frame->populate_array2d(img);

				pyr_t pyr;
				dlib::pyramid_up(img, pyramid, pyr);

				face_scal_x = video_width / (float)pyramid.nc();
				face_scal_y = video_height / (float)pyramid.nr();

				auto dets = face_detector(pyramid);
				//std::vector<dlib::rectangle> dets;

				if (show_facelandmark)
				{
					std::lock_guard<std::mutex> guard(face_mutex);
					std::swap(face_pyramid, pyramid);
					std::swap(face_dets, dets);
					std::swap(face_det_frame, frame);
				}
				else
				{
					frame_fps.tick();

					{
						std::lock_guard<std::mutex> guard(frame_mutex);
						std::swap(face_rects, dets);
						std::swap(face_frame, frame);
						face_shapes.clear();
					}

					invalidate = true;
				}
			}
		}
		else
		{
			Sleep(100);
		}

		Sleep(0);
	}
}

void perform_face_landmark()
{
	dlib::shape_predictor sp;

	// Training data here
	// http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	while (!exit_app)
	{
		if (show_facelandmark)
		{
			dlib::array2d<unsigned char> pyramid;
			std::vector<dlib::rectangle> dets;
			std::shared_ptr<yuv_frame> frame;

			{
				std::lock_guard<std::mutex> guard(face_mutex);
				std::swap(face_pyramid, pyramid);
				std::swap(face_dets, dets);
				std::swap(face_det_frame, frame);
			}

			if (frame != nullptr)
			{
				std::vector<dlib::full_object_detection> shapes;

				for (const auto & d : dets)
				{
					auto shape = sp(pyramid, d);
					shapes.emplace_back(shape);
				}

				frame_fps.tick();

				{
					std::lock_guard<std::mutex> guard(frame_mutex);
					std::swap(face_rects, dets);
					std::swap(face_shapes, shapes);
					std::swap(face_frame, frame);
					invalidate = true;
				}
			}
			else
			{
				Sleep(0);
			}
		}
		else
		{
			Sleep(100);
		}
	}
}

void plot(const frame_buffer &frame, int x, int y, double c)
{ //x coord, y coord, intensity c.
	if (x < 0 || x >= frame.width) return;
	if (y < 0 || y >= frame.height) return;

	auto bg0 = frame.pixels[y * frame.width + x + 1] / 255.0;
	auto cc0 = c + bg0 * (1.0 - c);
	frame.pixels[y * frame.width + x + 1] = cc0 * 255;

	auto bg1 = frame.pixels[y * frame.width + x] / 255.0;
	auto cc1 = ((1.0 - c) + bg1*c);
	frame.pixels[y * frame.width + x] = cc1 * 255;
}

void plot_steep(const frame_buffer &frame, int x, int y, double c)
{ //Similar to above function, swaps the order that the pixels are set.
	if (x < 0 || x >= frame.width) return;
	if (y < 0 || y >= frame.height) return;	

	auto bg0 = frame.pixels[y * frame.width + x + 1] / 255.0;
	auto cc0 = c + bg0 * (1.0 - c);
	frame.pixels[(y + 1) * frame.width + x] = cc0 * 255;

	auto bg1 = frame.pixels[y * frame.width + x] / 255.0;
	auto cc1 = ((1.0 - c) + bg1 * c);
	frame.pixels[y * frame.width + x] = cc1 * 255;
}

void draw_line(const frame_buffer &frame, int x0, int y0, int x1, int y1)
{
	/*Wu's Line drawing algorithm and pseudocode obtained from Wikipedia:
	  http://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm

	  We are drawing a line from lineStart to lineEnd with RGBColor color*/
	bool steep = (abs(y1 - y0) > abs(x1 - x0)); /*Determine whether the algorithm can be run normally,
	Or if it needs to be run with x and y swapped.*/

	if (steep)
	{
		std::swap(x0, y0);
		std::swap(x1, y1);
	}
	if (x0 > x1) //Determine whether to run the algorithm right to left, or left to right.
	{
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	double dx = x1 - x0; //Find the distance in the x-axis
	double dy = y1 - y0; //Find distance in the y-axis

	double slope = dy / dx;
	double yy = y0 + slope;

	for (int x = x0; x < x1; x++) //Main loop, plot every point along the line.
	{
		if (steep)
		{
			double y;
			double frac = modf(yy, &y);
			plot(frame, (int)y, x, frac);
		}
		else
		{
			double y;
			double frac = modf(yy, &y);
			plot_steep(frame, x, (int)y, frac);
		}
		yy += slope;
	}

}

void draw_line_segment(const frame_buffer &frame, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
	for (int i = start; i < end; ++i)
	{
		const auto  x1 = d.part(i).x() * face_scal_x;
		const auto  y1 = d.part(i).y() * face_scal_y;
		const auto  x2 = d.part(i + 1).x() * face_scal_x;
		const auto  y2 = d.part(i + 1).y() * face_scal_y;
		draw_line(frame, x1, y1, x2, y2);
	}

	if (isClosed)
	{
		const auto  x1 = d.part(end).x() * face_scal_x;
		const auto  y1 = d.part(end).y() * face_scal_y;
		const auto  x2 = d.part(start).x() * face_scal_x;
		const auto y2 = d.part(start).y() * face_scal_y;
		draw_line(frame, x1, y1, x2, y2);
	}
}

void render_face(const frame_buffer &frame, const dlib::full_object_detection& d)
{
	if (d.num_parts() == 68)
	{
		draw_line_segment(frame, d, 0, 16);           // Jaw line
		draw_line_segment(frame, d, 17, 21);          // Left eyebrow
		draw_line_segment(frame, d, 22, 26);          // Right eyebrow
		draw_line_segment(frame, d, 27, 30);          // Nose bridge
		draw_line_segment(frame, d, 30, 35, true);    // Lower nose
		draw_line_segment(frame, d, 36, 41, true);    // Left eye
		draw_line_segment(frame, d, 42, 47, true);    // Right Eye
		draw_line_segment(frame, d, 48, 59, true);    // Outer lip
		draw_line_segment(frame, d, 60, 67, true);    // Inner lip
	}
}


void render_rgb_frame(HDC hdc, int x, int y, int width, int height, const byte *frame)
{
	BITMAPINFO bm;
	bm.bmiHeader.biSize = sizeof(bm.bmiHeader);
	bm.bmiHeader.biWidth = width;
	bm.bmiHeader.biHeight = -(long)height;
	bm.bmiHeader.biPlanes = 1;
	bm.bmiHeader.biBitCount = 32;
	bm.bmiHeader.biCompression = BI_RGB;
	bm.bmiHeader.biSizeImage = width * height * 4; // buffCurrLen;
	bm.bmiHeader.biXPelsPerMeter = 0;
	bm.bmiHeader.biYPelsPerMeter = 0;
	bm.bmiHeader.biClrUsed = 0;
	bm.bmiHeader.biClrImportant = 0;

	SetDIBitsToDevice(hdc, x, y, width, height, 0, 0, 0, height, frame, &bm, DIB_RGB_COLORS);
}

void render_grayscale_frame(HDC hdc, int x, int y, int width, int height, const byte *frame)
{
	BITMAPINFO2 gbm;
	gbm.bmiHeader.biSize = sizeof(gbm.bmiHeader);
	gbm.bmiHeader.biWidth = width;
	gbm.bmiHeader.biHeight = -height;
	gbm.bmiHeader.biPlanes = 1;
	gbm.bmiHeader.biBitCount = 8;
	gbm.bmiHeader.biCompression = BI_RGB;
	gbm.bmiHeader.biSizeImage = width * height; // buffCurrLen;
	gbm.bmiHeader.biXPelsPerMeter = 0;
	gbm.bmiHeader.biYPelsPerMeter = 0;
	gbm.bmiHeader.biClrUsed = 256;
	gbm.bmiHeader.biClrImportant = 0;

	for (int i = 0; i < 256; i++)
	{
		gbm.bmiColors[i].rgbBlue = i;
		gbm.bmiColors[i].rgbGreen = i;
		gbm.bmiColors[i].rgbRed = i;
		gbm.bmiColors[i].rgbReserved = 0;
	}

	LOGPALETTE2 pal;
	pal.palVersion = 0x300;
	pal.palNumEntries = 256;

	for (int i = 0; i < 256; i++)
	{
		pal.palPalEntry[i].peBlue = i;
		pal.palPalEntry[i].peGreen = i;
		pal.palPalEntry[i].peRed = i;
		pal.palPalEntry[i].peFlags = 0;
	}

	auto hpal = CreatePalette((LPLOGPALETTE)&pal);
	auto old_hpal = SelectPalette(hdc, hpal, FALSE);
	RealizePalette(hdc);

	SetDIBitsToDevice(hdc, x, y, width, height, 0, 0, 0, height, frame, (BITMAPINFO*)&gbm, DIB_PAL_COLORS);

	SelectPalette(hdc, old_hpal, FALSE);
	DeleteObject(hpal);
}

void contrast_stretch_grayscale_frame(const frame_buffer &frame)
{
	const int OUT_MIN = 0;   // The desired min output luminosity 0   to stretch to entire spectrum
	const int OUT_MAX = 255; // The desired max output luminosity 255 to stretch to entire spectrum

	int cmin = OUT_MAX;
	int cmax = OUT_MIN;

	for (int y = 0; y < frame.height; y++)
	{
		for (int x = 0; x < frame.width; x++)
		{
			int c = frame.pixels[y * frame.width + x];
			cmin = cmin < c ? cmin : c;
			cmax = cmax > c ? cmax : c;
		}
	}

	for (int y = 0; y < frame.height; y++)
	{
		for (int x = 0; x < frame.width; x++)
		{
			int c = frame.pixels[y * frame.width + x];
			int cc = MulDiv(c - cmin, OUT_MAX - OUT_MIN, cmax - cmin) + OUT_MIN;
			cc = cc < OUT_MIN ? OUT_MIN : cc;
			cc = cc > OUT_MAX ? OUT_MAX : cc;
			frame.pixels[y * frame.width + x] = cc;
		}
	}
}

int calc_bitrate(int num_faces)
{
	int result = 10;

	if (frame_mode == frame_mode_t::autoencoder)
	{
		result = 10;
	}
	else if (frame_mode == frame_mode_t::edgedetect)
	{
		result = 100;
	}
	else
	{
		result = 10;
	}

	result += num_faces * 1;

	return result;
}

void on_paint(HDC hdc, const RECT &clientBounds)
{
	auto layout_padding = 10;
	auto layout_bottom = clientBounds.bottom;
	auto layout_y = (layout_bottom - video_height) / 2;
	auto layout_x1 = (clientBounds.left + clientBounds.right - video_width - video_width - layout_padding) / 2;
	auto layout_x2 = layout_x1 + video_width + layout_padding;

	//
	//
	// Draw the title
	//
	//

	static HFONT font = CreateFont(48, 0, 0, 0, FW_NORMAL,
		FALSE, FALSE, FALSE, DEFAULT_CHARSET,
		OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
		L"Segoe UI");

	static HFONT font_title = CreateFont(64, 0, 0, 0, FW_NORMAL,
		FALSE, FALSE, FALSE, DEFAULT_CHARSET,
		OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
		L"Segoe UI");

	SetTextColor(hdc, RGB(200, 200, 200));
	SetBkMode(hdc, TRANSPARENT);
	SelectObject(hdc, font_title);

	auto title1 = L"Low bandwidth video with face landmarking";	
	auto tcy = layout_y / 3;

	SIZE title_extent;
	GetTextExtentPoint32(hdc, title1, wcslen(title1), &title_extent);
	auto title_y = layout_padding + (tcy - title_extent.cy) / 2;
	TextOut(hdc, (clientBounds.right - title_extent.cx) / 2, layout_padding + (tcy - title_extent.cy) / 2, title1, wcslen(title1));

	SelectObject(hdc, font);

	//
	//
	// Draw Face detected frames
	//
	//

	auto frame1 = current_frame;

	if (frame1 != nullptr)
	{
		static byte rgb_frame[video_width * video_height * 4];
		frame1->populate_rgb_frame(rgb_frame, video_width, video_height);
		render_rgb_frame(hdc, layout_x1, layout_y, video_width, video_height, rgb_frame);
	}

	auto face_count = 0;
	auto frame2 = show_facedetect ? face_frame : current_frame;

	if (frame2 != nullptr)
	{
		dlib::matrix<float> m(video_height, video_width);

		if (frame_mode == frame_mode_t::autoencoder)
		{
			frame2->populate_edge_matrix(m);

			dlib::matrix<float> transformed;

			{
				std::lock_guard<std::mutex> guard(frame_mutex);
				transformed = ae_net(m);
			}

			auto decoder_width = std::min(transformed.nc(), video_width);
			auto decoder_height = std::min(transformed.nr(), video_height);

			for (int y = 0; y < decoder_height; y++)
			{
				for (int x = 0; x < decoder_width; x++)
				{
					m(y, x) = transformed(y, x);
				}
			}
		}
		else if (frame_mode == frame_mode_t::edgedetect)
		{
			frame2->populate_edge_matrix(m);
		}
		else
		{
			dlib::matrix<float> m_small(video_height / 10, video_width / 10);
			frame2->populate_matrix(m);

			dlib::resize_image(m, m_small);
			dlib::resize_image(m_small, m);

			//frame2->populate_grayscale_frame(grayscale_frame, video_width, video_height);
		}

		//dlib::equalize_histogram(m);

		static byte grayscale_frame[video_width * video_height];

		for (int y = 0; y < video_height; y++)
		{
			for (int x = 0; x < video_width; x++)
			{
				grayscale_frame[video_width * y + x] = (int)(m(y, x) * 255.0f);
			}
		}

		if (!show_facedetect)
		{
			frame_fps.tick();
		}

		frame_buffer frame = { grayscale_frame, video_width, video_height };
		contrast_stretch_grayscale_frame(frame);

		//
		//
		// Draw faces
		//
		//

		if (show_facedetect)
		{
			std::vector<dlib::rectangle> rects;
			std::vector<dlib::full_object_detection> shapes;

			{
				std::lock_guard<std::mutex> guard(frame_mutex);
				rects = face_rects;
				shapes = face_shapes;
			}

			if (show_facelandmark)
			{
				face_count = shapes.size();

				for (const auto &s : shapes)
				{
					render_face(frame, s);
				}
			}
			else
			{
				for (const auto &r : rects)
				{
					draw_line(frame, r.left() * face_scal_x, r.top() * face_scal_y, r.right() * face_scal_x, r.top() * face_scal_y);
					draw_line(frame, r.right() * face_scal_x, r.top() * face_scal_y, r.right() * face_scal_x, r.bottom() * face_scal_y);
					draw_line(frame, r.right() * face_scal_x, r.bottom() * face_scal_y, r.left() * face_scal_x, r.bottom() * face_scal_y);
					draw_line(frame, r.left() * face_scal_x, r.bottom() * face_scal_y, r.left() * face_scal_x, r.top() * face_scal_y);
				}
			}

		}

		render_grayscale_frame(hdc, layout_x2, layout_y, video_width, video_height, grayscale_frame);
	}



	//
	//
	// Draw labels
	//
	//

	SelectObject(hdc, font);

	wchar_t sz[64];
	swprintf_s(sz, L"%0.1f fps     500 kbit/s", video_fps.val());
	TextOut(hdc, layout_x1, layout_y + video_height + layout_padding, sz, wcslen(sz));

	swprintf_s(sz, L"%0.1f fps     %d kbit/s     training %d%%", frame_fps.val(), calc_bitrate(face_count), training_step * 10);
	TextOut(hdc, layout_x2, layout_y + video_height + layout_padding, sz, wcslen(sz));
}

void on_timer(HWND hwnd)
{
	if (pSourceReader)
	{
		CComPtr<IMFSample> videoSample;

		static UINT32 frame_width = 0, frame_height = 0;
		static LONG frame_stride = 0;

		DWORD streamIndex = 0, flags = 0;
		LONGLONG llVideoTimeStamp = 0;

		auto hr = pSourceReader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &streamIndex, &flags, &llVideoTimeStamp, &videoSample);

		if (SUCCEEDED(hr))
		{
			if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
			{
				OutputDebugStringW(L"\tEnd of stream\n");
			}
			if (flags & MF_SOURCE_READERF_NEWSTREAM)
			{
				OutputDebugStringW(L"\tNew stream\n");
			}
			if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)
			{
				OutputDebugStringW(L"\tNative type changed\n");
			}
			if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
			{
				OutputDebugStringW(L"\tCurrent type changed\n");
			}

			if (flags & MF_SOURCE_READERF_STREAMTICK)
			{
				OutputDebugStringW(L"\tStream tick\n");

				CComPtr<IMFMediaType> videoType;
				hr = pSourceReader->GetCurrentMediaType(streamIndex, &videoType);

				if (SUCCEEDED(hr))
				{
					MFGetAttributeSize(videoType, MF_MT_FRAME_SIZE, &frame_width, &frame_height);
					videoType->GetUINT32(MF_MT_DEFAULT_STRIDE, (UINT32*)&frame_stride);
				}
			}
		}

		if (SUCCEEDED(hr) && streamIndex == 0)
		{
			if (videoSample)
			{
				DWORD nCurrBufferCount = 0;
				CComPtr<IMFMediaBuffer> pMediaBuffer;
				DWORD nCurrLen = 0;

				if (SUCCEEDED(hr))
				{
					hr = videoSample->GetBufferCount(&nCurrBufferCount);
				}

				if (SUCCEEDED(hr))
				{
					//hr = videoSample->ConvertToContiguousBuffer(&pMediaBuffer);
					hr = videoSample->GetBufferByIndex(0, &pMediaBuffer);
				}

				if (SUCCEEDED(hr))
				{

					hr = pMediaBuffer->GetCurrentLength(&nCurrLen);
				}

				byte *frame_buffer = nullptr;
				DWORD frame_buffer_len = 0;
				//DWORD buffMaxLen = 0;

				if (SUCCEEDED(hr))
				{
					hr = pMediaBuffer->Lock(&frame_buffer, nullptr, &frame_buffer_len);
				}

				if (SUCCEEDED(hr))
				{
					if (frame_buffer != nullptr)
					{
						current_frame = std::make_shared<yuv_frame>(frame_width, frame_height, frame_stride, frame_buffer, frame_buffer_len);
						video_fps.tick();
						invalidate = true;
					}

					pMediaBuffer->Unlock();
				}
			}
		}
	}

	if (invalidate)
	{
		InvalidateRect(hwnd, nullptr, FALSE);
		invalidate = false;
	}
}

LRESULT CALLBACK wnd_proc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	static RECT clientBounds;
	static HBITMAP hBitmapBackBuffer = nullptr;

	switch (message)
	{
	case WM_COMMAND:
	{
		int wmId = LOWORD(wParam);
		// Parse the menu selections:
		switch (wmId)
		{
		case IDM_ABOUT:
			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, about_proc);
			break;
		case IDM_EDGEDETECT:
			frame_mode = frame_mode_t::edgedetect;
			InvalidateRect(hWnd, nullptr, FALSE);
			break;
		case IDM_AUTOENCODER:
			frame_mode = frame_mode_t::autoencoder;
			training_step = 0;
			InvalidateRect(hWnd, nullptr, FALSE);
			break;
		case IDM_SCALE:
			frame_mode = frame_mode_t::scale;
			InvalidateRect(hWnd, nullptr, FALSE);
			break;
		case IDM_FACEDETECT:
			show_facedetect = !show_facedetect;
			InvalidateRect(hWnd, nullptr, FALSE);
			break;
		case IDM_FACELANDMARK:
			show_facelandmark = !show_facelandmark;
			InvalidateRect(hWnd, nullptr, FALSE);
			break;
		case IDM_EXIT:
			DestroyWindow(hWnd);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
	break;
	case WM_INITMENUPOPUP:
	{
		auto hmenu = (HMENU)wParam;
		auto uPos = (UINT)LOWORD(lParam);
		auto fSystemMenu = (BOOL)HIWORD(lParam);

		MENUITEMINFO mii = { sizeof(MENUITEMINFO),MIIM_STATE,0,0,0,nullptr,nullptr,nullptr,0,nullptr,0 };
		mii.fState = frame_mode == frame_mode_t::edgedetect ? MFS_CHECKED : MFS_UNCHECKED;
		SetMenuItemInfo(hmenu, IDM_EDGEDETECT, FALSE, &mii);
		mii.fState = frame_mode == frame_mode_t::autoencoder ? MFS_CHECKED : MFS_UNCHECKED;
		SetMenuItemInfo(hmenu, IDM_AUTOENCODER, FALSE, &mii);
		mii.fState = frame_mode == frame_mode_t::scale ? MFS_CHECKED : MFS_UNCHECKED;
		SetMenuItemInfo(hmenu, IDM_SCALE, FALSE, &mii);
		mii.fState = show_facedetect ? MFS_CHECKED : MFS_UNCHECKED;
		SetMenuItemInfo(hmenu, IDM_FACEDETECT, FALSE, &mii);
		mii.fState = show_facelandmark ? MFS_CHECKED : MFS_UNCHECKED;
		SetMenuItemInfo(hmenu, IDM_FACELANDMARK, FALSE, &mii);
	}
	break;

	case WM_CREATE:
	{
		GetClientRect(hWnd, &clientBounds);
		auto hdc = GetDC(hWnd);
		hBitmapBackBuffer = CreateCompatibleBitmap(hdc, clientBounds.right, clientBounds.bottom);
		ReleaseDC(hWnd, hdc);
	}
	break;
	case WM_SIZE:
	{
		DeleteObject(hBitmapBackBuffer);
		GetClientRect(hWnd, &clientBounds);
		auto hdc = GetDC(hWnd);
		hBitmapBackBuffer = CreateCompatibleBitmap(hdc, clientBounds.right, clientBounds.bottom);
		ReleaseDC(hWnd, hdc);
		break;
	}

	case WM_ERASEBKGND:
		return 1;

	case WM_PAINT:
	{
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hWnd, &ps);

		auto hdcBackBuffer = CreateCompatibleDC(hdc);
		auto oldBitmap = SelectObject(hdcBackBuffer, hBitmapBackBuffer);
		FillRect(hdcBackBuffer, &ps.rcPaint, (HBRUSH)GetStockObject(BLACK_BRUSH));
		on_paint(hdcBackBuffer, clientBounds);
		BitBlt(hdc, 0, 0, clientBounds.right, clientBounds.bottom, hdcBackBuffer, 0, 0, SRCCOPY);
		SelectObject(hdcBackBuffer, oldBitmap);
		DeleteObject(hdcBackBuffer);
		EndPaint(hWnd, &ps);
	}
	break;
	case WM_DESTROY:
		DeleteObject(hBitmapBackBuffer);
		PostQuitMessage(0);
		break;
	case WM_TIMER:
		on_timer(hWnd);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}



ATOM register_wnd_class(HINSTANCE hInstance)
{
	WNDCLASSEXW wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = wnd_proc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APP));
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_APP);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_APP));

	return RegisterClassExW(&wcex);
}

BOOL init_instance(HINSTANCE hInstance, int nCmdShow)
{
	register_wnd_class(hInstance);

	hInst = hInstance; // Store instance handle in our global variable

	hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

	if (!hWnd)
	{
		return FALSE;
	}

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return TRUE;
}

std::map<std::wstring, CComPtr<IMFActivate>> list_cameras()
{
	std::map<std::wstring, CComPtr<IMFActivate>> results;

	HRESULT hr = S_OK;
	IMFActivate** ppDevices = NULL;
	UINT32 nCount = 0;
	WCHAR* pszFriendlyName = NULL;
	UINT32 cchName = 0;
	IMFAttributes* pAttributes = NULL;

	hr = MFCreateAttributes(&pAttributes, 1);
	if (SUCCEEDED(hr))
	{
		hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	}

	hr = MFEnumDeviceSources(pAttributes, &ppDevices, &nCount);

	if (SUCCEEDED(hr))
	{
		for (size_t i = 0; i < nCount; i++)
		{
			auto hr2 = ppDevices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &pszFriendlyName, &cchName);

			if (SUCCEEDED(hr2))
			{
				results[pszFriendlyName] = ppDevices[i];
				CoTaskMemFree(pszFriendlyName);
				pszFriendlyName = nullptr;
			}
		}
	}

	if (SUCCEEDED(hr))
	{
		for (size_t i = 0; i < nCount; i++)
		{
			SafeRelease(&(ppDevices[i]));
		}
		if (ppDevices)
		{
			CoTaskMemFree(ppDevices);
		}
	}

	return results;
}

CComPtr<IMFMediaSource> select_device_source()
{
	auto cameras = list_cameras();

	HRESULT hr = S_OK;
	CComPtr<IMFMediaSource> result;

	if (cameras.size() > 0)
	{
		auto found = cameras.find(L"Logitech Webcam C930e");
		if (found == cameras.end()) found = cameras.begin();
		hr = found->second->ActivateObject(__uuidof(IMFMediaSource), (void**)&result);
	}

	return FAILED(hr) ? nullptr : result;
}

//void OnDeviceLost(HWND hwnd, DEV_BROADCAST_HDR *pHdr)
//{
//	if (g_pPreview == NULL || pHdr == NULL)
//	{
//		return;
//	}
//
//	HRESULT hr = S_OK;
//	BOOL bDeviceLost = FALSE;
//
//	hr = g_pPreview->CheckDeviceLost(pHdr, &bDeviceLost);
//
//	if (FAILED(hr) || bDeviceLost)
//	{
//		g_pPreview->CloseDevice();
//
//		MessageBox(hwnd, L"Lost the capture device.", WINDOW_NAME, MB_OK);
//	}
//}


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
	_In_opt_ HINSTANCE hPrevInstance,
	_In_ LPWSTR    lpCmdLine,
	_In_ int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

	// Initialize global strings
	LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadStringW(hInstance, IDC_APP, szWindowClass, MAX_LOADSTRING);

	// Perform application initialization:
	if (!init_instance(hInstance, nCmdShow))
	{
		return FALSE;
	}

	HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
	hr = MFStartup(MF_VERSION);

	MSG msg;

	{
		auto source = select_device_source();


		CComPtr<IMFAttributes> pAttributes;
		hr = MFCreateAttributes(&pAttributes, 1);

		if (SUCCEEDED(hr))
		{
			//hr = pAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
			//hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
		}

		if (SUCCEEDED(hr))
		{
			hr = MFCreateSourceReaderFromMediaSource(source, nullptr, &pSourceReader);
		}

		std::thread t1(perform_training);
		std::thread t2(perform_face_detect);
		std::thread t3(perform_face_landmark);

		HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_APP));
		SetTimer(hWnd, 99, 1000 / 20, nullptr);

		// Main message loop:
		while (GetMessage(&msg, nullptr, 0, 0))
		{
			if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}

		exit_app = true;
		t1.join();
		t2.join();
		t3.join();

		pSourceReader.Release();
	}


	MFShutdown();
	CoUninitialize();

	return (int)msg.wParam;
}
