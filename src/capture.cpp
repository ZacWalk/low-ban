#include "capture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <windows.h>
#include <codecapi.h>
#include <objbase.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wincodec.h>
#include <wrl/client.h>

#include "baseline.h"

#pragma comment(lib, "Mf.lib")
#pragma comment(lib, "Mfplat.lib")
#pragma comment(lib, "Mfuuid.lib")
#pragma comment(lib, "Mfreadwrite.lib")

using Microsoft::WRL::ComPtr;

// The selector is declared as a signed enum but the API takes a DWORD.
constexpr DWORD first_video_stream = static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM);

struct webcam::impl
{
	ComPtr<IMFMediaSource> source;
	ComPtr<IMFSourceReader> reader;
	UINT32 frame_width = 0;
	UINT32 frame_height = 0;
	INT32 frame_stride = 0;
	GUID subtype = GUID_NULL;

	bool refresh_format()
	{
		ComPtr<IMFMediaType> type;
		if (FAILED(reader->GetCurrentMediaType(first_video_stream, &type))) return false;
		if (FAILED(MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &frame_width, &frame_height))) return false;

		if (FAILED(type->GetGUID(MF_MT_SUBTYPE, &subtype))) subtype = GUID_NULL;

		if (FAILED(type->GetUINT32(MF_MT_DEFAULT_STRIDE, reinterpret_cast<UINT32*>(&frame_stride))))
		{
			frame_stride = static_cast<INT32>(frame_width) * 2; // packed YUY2
		}

		return frame_width > 0 && frame_height > 0;
	}
};

namespace
{
	// The subtype GUID's first four bytes are the FOURCC for the packed video formats.
	std::string format_name(const GUID& subtype)
	{
		char fourcc[5] = {};
		memcpy(fourcc, &subtype.Data1, 4);
		for (char& c : fourcc)
		{
			if (c < 32 || c > 126) c = '?';
		}
		return fourcc;
	}

	std::string describe_read_failure(const HRESULT hr)
	{
		char code[16] = {};
		snprintf(code, sizeof(code), " (0x%08lX)", static_cast<unsigned long>(hr));
		switch (hr)
		{
		case MF_E_HW_MFT_FAILED_START_STREAMING:
		case MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED:
			return std::string("the camera is in use by another application") + code;
		case MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED: return std::string("the camera was disconnected") + code;
		default: return std::string("the camera stopped delivering frames") + code;
		}
	}

	ComPtr<IMFMediaSource> activate_first_camera()
	{
		ComPtr<IMFAttributes> attributes;
		if (FAILED(MFCreateAttributes(&attributes, 1))) return nullptr;
		if (FAILED(attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
		                               MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)))
			return nullptr;

		IMFActivate** devices = nullptr;
		UINT32 count = 0;
		if (FAILED(MFEnumDeviceSources(attributes.Get(), &devices, &count))) return nullptr;

		ComPtr<IMFMediaSource> source;
		for (UINT32 i = 0; i < count; ++i)
		{
			if (!source && FAILED(devices[i]->ActivateObject(IID_PPV_ARGS(&source))))
			{
				source.Reset();
			}
			devices[i]->Release();
		}
		CoTaskMemFree(devices);

		return source;
	}
}

media_foundation::media_foundation()
{
	started = SUCCEEDED(MFStartup(MF_VERSION));
}

int run_on_mta(const std::function<int()>& callback)
{
	int result = 1;
	std::exception_ptr callback_error;
	std::thread worker([&]
	{
		struct com_scope
		{
			const HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);

			~com_scope()
			{
				if (SUCCEEDED(result)) CoUninitialize();
			}
		} apartment;

		if (FAILED(apartment.result))
		{
			std::fprintf(stderr, "could not initialise the console MTA (0x%08lx)\n",
			             static_cast<unsigned long>(apartment.result));
			return;
		}

		try
		{
			result = callback();
		}
		catch (...)
		{
			callback_error = std::current_exception();
		}
	});
	worker.join();
	if (callback_error) std::rethrow_exception(callback_error);
	return result;
}

bool is_mta_thread()
{
	APTTYPE type = APTTYPE_CURRENT;
	APTTYPEQUALIFIER qualifier = APTTYPEQUALIFIER_NONE;
	return SUCCEEDED(CoGetApartmentType(&type, &qualifier)) && type == APTTYPE_MTA;
}

media_foundation::~media_foundation()
{
	if (started) MFShutdown();
}

webcam::~webcam()
{
	close();
}

bool webcam::is_open() const
{
	return state != nullptr && state->reader;
}

bool webcam::open(const int width, const int height, std::string& error)
{
	close();

	auto owned = std::make_unique<impl>();
	owned->source = activate_first_camera();
	if (!owned->source)
	{
		error = "no camera was found";
		return false;
	}

	ComPtr<IMFAttributes> attributes;
	if (FAILED(MFCreateAttributes(&attributes, 1)) ||
	    FAILED(attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE)))
	{
		error = "could not configure the source reader";
		return false;
	}

	if (FAILED(MFCreateSourceReaderFromMediaSource(owned->source.Get(), attributes.Get(), &owned->reader)))
	{
		error = "could not create a source reader for the camera";
		return false;
	}

	ComPtr<IMFMediaType> type;
	if (FAILED(MFCreateMediaType(&type)))
	{
		error = "could not describe the wanted video format";
		return false;
	}

	type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
	type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_YUY2);
	MFSetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, width, height);

	if (FAILED(owned->reader->SetCurrentMediaType(first_video_stream, nullptr, type.Get())))
	{
		error = "the camera would not provide YUY2 video";
		return false;
	}

	if (!owned->refresh_format())
	{
		error = "the camera did not report a usable video format";
		return false;
	}

	// Everything downstream unpacks YUY2 by hand, so a silent substitution would show as
	// noise rather than as a failure.
	if (owned->subtype != MFVideoFormat_YUY2)
	{
		error = "camera delivered " + format_name(owned->subtype) + " instead of YUY2";
		return false;
	}

	state = owned.release();
	return true;
}

void webcam::close()
{
	if (state == nullptr) return;

	state->reader.Reset();
	if (state->source) state->source->Shutdown();
	state->source.Reset();

	delete state;
	state = nullptr;
}

bool webcam::read(video_frame& frame, std::string& error)
{
	if (!is_open()) return false;

	DWORD stream_index = 0;
	DWORD flags = 0;
	LONGLONG timestamp = 0;
	ComPtr<IMFSample> sample;

	// A failed read, an error flag or the end of the stream all mean the reader is finished;
	// retrying would only spin on the same failure with nothing on screen to say why.
	const HRESULT hr = state->reader->ReadSample(first_video_stream, 0, &stream_index, &flags, &timestamp, &sample);
	if (FAILED(hr))
	{
		error = describe_read_failure(hr);
		return false;
	}
	if (flags & MF_SOURCE_READERF_ERROR)
	{
		error = "the camera reported a stream error";
		return false;
	}
	if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
	{
		error = "the camera stream ended";
		return false;
	}

	if (flags & (MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED | MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED))
	{
		state->refresh_format();
	}

	if (!sample) return false;

	ComPtr<IMFMediaBuffer> buffer;
	if (FAILED(sample->ConvertToContiguousBuffer(&buffer))) return false;

	BYTE* bytes = nullptr;
	DWORD length = 0;
	if (FAILED(buffer->Lock(&bytes, nullptr, &length))) return false;

	const bool decoded = decode_yuy2(frame, bytes, length, static_cast<int>(state->frame_width),
	                                 static_cast<int>(state->frame_height), state->frame_stride);
	buffer->Unlock();

	return decoded;
}

namespace
{
	// High enough that a clip is a faithful source for both wires: about 0.9 bits per pixel at
	// 640x480 and 15 frames a second.
	constexpr UINT32 clip_bits_per_second = 4000000;

	std::string clip_failure(const char* what, const HRESULT hr)
	{
		char text[192] = {};
		snprintf(text, sizeof(text), "%s (0x%08lX)", what, static_cast<unsigned long>(hr));
		return text;
	}

	HRESULT describe_video(IMFMediaType* type, const GUID& subtype, const int width, const int height,
	                       const int frames_per_second)
	{
		HRESULT hr = type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		if (SUCCEEDED(hr)) hr = type->SetGUID(MF_MT_SUBTYPE, subtype);
		if (SUCCEEDED(hr)) hr = MFSetAttributeSize(type, MF_MT_FRAME_SIZE, width, height);
		if (SUCCEEDED(hr) && frames_per_second > 0)
			hr = MFSetAttributeRatio(type, MF_MT_FRAME_RATE, frames_per_second, 1);
		if (SUCCEEDED(hr)) hr = MFSetAttributeRatio(type, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
		if (SUCCEEDED(hr)) hr = type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
		return hr;
	}
}

struct clip_writer::impl
{
	ComPtr<IMFSinkWriter> writer;
	DWORD stream = 0;
	int width = 0;
	int height = 0;
	LONGLONG duration = 0;
	int frames = 0;
	std::vector<uint8_t> packed;
};

clip_writer::~clip_writer()
{
	std::string ignored;
	close(ignored);
}

bool clip_writer::is_open() const
{
	return state != nullptr;
}

int clip_writer::frames() const
{
	return state != nullptr ? state->frames : 0;
}

bool clip_writer::open(const std::wstring& path, const int width, const int height, const int frames_per_second,
                       std::string& error)
{
	std::string ignored;
	close(ignored);
	if (width <= 0 || height <= 0 || width % 2 != 0 || frames_per_second <= 0)
	{
		error = "a clip needs an even frame width and a frame rate";
		return false;
	}

	// The sink writer drives the same H.264 encoder the baseline uses.
	if (!h264_safe_to_use(error)) return false;

	auto owned = std::make_unique<impl>();
	owned->width = width;
	owned->height = height;
	owned->duration = 10000000LL / frames_per_second;

	HRESULT hr = MFCreateSinkWriterFromURL(path.c_str(), nullptr, nullptr, &owned->writer);
	if (FAILED(hr))
	{
		error = clip_failure("could not create the clip file", hr);
		return false;
	}

	ComPtr<IMFMediaType> output;
	hr = MFCreateMediaType(&output);
	if (SUCCEEDED(hr)) hr = describe_video(output.Get(), MFVideoFormat_H264, width, height, frames_per_second);
	if (SUCCEEDED(hr)) hr = output->SetUINT32(MF_MT_AVG_BITRATE, clip_bits_per_second);
	if (SUCCEEDED(hr)) hr = output->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_High);
	if (SUCCEEDED(hr)) hr = owned->writer->AddStream(output.Get(), &owned->stream);

	ComPtr<IMFMediaType> input;
	if (SUCCEEDED(hr)) hr = MFCreateMediaType(&input);
	if (SUCCEEDED(hr)) hr = describe_video(input.Get(), MFVideoFormat_YUY2, width, height, frames_per_second);
	if (SUCCEEDED(hr)) hr = owned->writer->SetInputMediaType(owned->stream, input.Get(), nullptr);
	if (SUCCEEDED(hr)) hr = owned->writer->BeginWriting();
	if (FAILED(hr))
	{
		error = clip_failure("could not start an H.264 clip", hr);
		return false;
	}

	state = owned.release();
	return true;
}

bool clip_writer::write(const video_frame& frame, std::string& error)
{
	if (state == nullptr)
	{
		error = "no clip is open";
		return false;
	}
	if (frame.width() != state->width || frame.height() != state->height)
	{
		error = "the frame size changed while recording";
		return false;
	}

	encode_yuy2(frame, state->packed);
	const auto length = static_cast<DWORD>(state->packed.size());
	ComPtr<IMFMediaBuffer> buffer;
	HRESULT hr = MFCreateMemoryBuffer(length, &buffer);
	BYTE* data = nullptr;
	if (SUCCEEDED(hr)) hr = buffer->Lock(&data, nullptr, nullptr);
	if (SUCCEEDED(hr))
	{
		memcpy(data, state->packed.data(), length);
		buffer->Unlock();
		hr = buffer->SetCurrentLength(length);
	}

	ComPtr<IMFSample> sample;
	if (SUCCEEDED(hr)) hr = MFCreateSample(&sample);
	if (SUCCEEDED(hr)) hr = sample->AddBuffer(buffer.Get());
	if (SUCCEEDED(hr)) hr = sample->SetSampleTime(state->frames * state->duration);
	if (SUCCEEDED(hr)) hr = sample->SetSampleDuration(state->duration);
	if (SUCCEEDED(hr)) hr = state->writer->WriteSample(state->stream, sample.Get());
	if (FAILED(hr))
	{
		error = clip_failure("could not write a clip frame", hr);
		return false;
	}
	++state->frames;
	return true;
}

bool clip_writer::close(std::string& error)
{
	if (state == nullptr) return true;
	const HRESULT hr = state->frames > 0 ? state->writer->Finalize() : S_OK;
	delete state;
	state = nullptr;
	if (FAILED(hr))
	{
		error = clip_failure("could not finish the clip", hr);
		return false;
	}
	return true;
}

struct clip_reader::impl
{
	ComPtr<IMFSourceReader> reader;
	GUID subtype = GUID_NULL;
	UINT32 width = 0;
	UINT32 height = 0;
	LONG stride = 0;

	// A YUV4MPEG2 stream instead, read straight from the file.
	FILE* y4m = nullptr;
	int source_width = 0;
	int source_height = 0;
	int target_width = 0;
	int target_height = 0;
	double frame_seconds = 0.0;
	int frames = 0;
	std::vector<uint8_t> buffer;

	impl() = default;
	impl(const impl&) = delete;
	impl& operator=(const impl&) = delete;
	~impl()
	{
		if (y4m != nullptr) std::fclose(y4m);
	}
};

namespace
{
	bool is_y4m(const std::wstring& path)
	{
		return path.size() >= 4 && _wcsicmp(path.c_str() + path.size() - 4, L".y4m") == 0;
	}

	bool read_line(FILE* file, std::string& line)
	{
		line.clear();
		for (int c = std::fgetc(file); c != EOF; c = std::fgetc(file))
		{
			if (c == '\n') return true;
			if (line.size() >= 1024) return false;
			line.push_back(static_cast<char>(c));
		}
		return !line.empty();
	}

	// Where each output sample along one axis reads from: the footprint it covers when
	// shrinking, so nothing aliases, or its two nearest neighbours when enlarging.
	struct resample_taps
	{
		std::vector<int> first;
		std::vector<std::vector<float>> weights;
	};

	resample_taps make_resample_taps(const int source, const int target)
	{
		resample_taps taps;
		taps.first.resize(static_cast<size_t>(target));
		taps.weights.resize(static_cast<size_t>(target));
		const double ratio = static_cast<double>(source) / target;
		for (int i = 0; i < target; ++i)
		{
			auto& weights = taps.weights[static_cast<size_t>(i)];
			if (ratio > 1.0)
			{
				const double start = i * ratio;
				const double end = start + ratio;
				const int first = static_cast<int>(std::floor(start));
				taps.first[static_cast<size_t>(i)] = first;
				for (int s = first; s < end && s < source; ++s)
					weights.push_back(
						static_cast<float>((std::min<double>(s + 1, end) - std::max<double>(s, start)) / ratio));
			}
			else
			{
				const double centre = std::max(0.0, (i + 0.5) * ratio - 0.5);
				const int first = std::min(static_cast<int>(centre), source - 1);
				const auto t = static_cast<float>(centre - first);
				taps.first[static_cast<size_t>(i)] = first;
				weights = {1.0f - t, t};
			}
		}
		return taps;
	}

	// A window of a plane scaled to fill `out`.
	void resample_plane(const uint8_t* plane, const int stride, const int left, const int top, const int width,
	                    const int height, image_u8& out)
	{
		const resample_taps columns = make_resample_taps(width, out.width());
		const resample_taps rows = make_resample_taps(height, out.height());
		std::vector<float> horizontal(static_cast<size_t>(out.width()) * height);
		for (int y = 0; y < height; ++y)
		{
			const uint8_t* in = plane + static_cast<size_t>(top + y) * stride + left;
			for (int x = 0; x < out.width(); ++x)
			{
				const auto& weights = columns.weights[static_cast<size_t>(x)];
				const int first = columns.first[static_cast<size_t>(x)];
				float sum = 0.0f;
				for (size_t k = 0; k < weights.size(); ++k)
					sum += weights[k] * in[std::min(first + static_cast<int>(k), width - 1)];
				horizontal[static_cast<size_t>(y) * out.width() + x] = sum;
			}
		}
		for (int y = 0; y < out.height(); ++y)
		{
			const auto& weights = rows.weights[static_cast<size_t>(y)];
			const int first = rows.first[static_cast<size_t>(y)];
			for (int x = 0; x < out.width(); ++x)
			{
				float sum = 0.0f;
				for (size_t k = 0; k < weights.size(); ++k)
					sum += weights[k] *
					       horizontal[static_cast<size_t>(std::min(first + static_cast<int>(k), height - 1)) *
					                      out.width() +
					                  x];
				out(x, y) = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(sum)), 0, 255));
			}
		}
	}
}

// The stream's header: its size, its frame rate and 4:2:0 chroma, the only layout read here.
bool clip_reader::open_y4m(const std::wstring& path, const int width, const int height, std::string& error)
{
	auto owned = std::make_unique<impl>();
	if (_wfopen_s(&owned->y4m, path.c_str(), L"rb") != 0 || owned->y4m == nullptr)
	{
		error = "could not open the clip";
		return false;
	}

	std::string header;
	if (!read_line(owned->y4m, header) || header.rfind("YUV4MPEG2 ", 0) != 0)
	{
		error = "the clip is not a YUV4MPEG2 stream";
		return false;
	}
	long rate = 0;
	long scale = 0;
	bool four_two_zero = true;
	for (size_t start = 10; start < header.size();)
	{
		size_t end = header.find(' ', start);
		if (end == std::string::npos) end = header.size();
		const std::string field = header.substr(start, end - start);
		if (!field.empty())
		{
			const char* value = field.c_str() + 1;
			if (field[0] == 'W') owned->source_width = std::atoi(value);
			else if (field[0] == 'H') owned->source_height = std::atoi(value);
			else if (field[0] == 'F')
			{
				char* colon = nullptr;
				rate = std::strtol(value, &colon, 10);
				scale = colon != nullptr && *colon == ':' ? std::strtol(colon + 1, nullptr, 10) : 0;
			}
			else if (field[0] == 'C') four_two_zero = field.rfind("C420", 0) == 0;
		}
		start = end + 1;
	}
	if (owned->source_width <= 0 || owned->source_height <= 0 || owned->source_width % 2 != 0 ||
	    owned->source_height % 2 != 0 || owned->source_width > 8192 || owned->source_height > 8192 || rate <= 0 ||
	    scale <= 0 || !four_two_zero)
	{
		error = "the clip is not an even-sized 4:2:0 stream with a frame rate";
		return false;
	}

	owned->target_width = width;
	owned->target_height = height;
	owned->frame_seconds = static_cast<double>(scale) / static_cast<double>(rate);
	owned->buffer.resize(static_cast<size_t>(owned->source_width) * owned->source_height * 3 / 2);
	state = owned.release();
	return true;
}

bool clip_reader::read_y4m(video_frame& frame, double& seconds, std::string& error)
{
	std::string marker;
	if (!read_line(state->y4m, marker)) return false;
	if (marker.rfind("FRAME", 0) != 0)
	{
		error = "a frame of the clip has no FRAME marker";
		return false;
	}
	if (std::fread(state->buffer.data(), 1, state->buffer.size(), state->y4m) != state->buffer.size())
	{
		error = "the clip's last frame is cut short";
		return false;
	}

	// The middle of the frame at the capture's shape, as a camera of that shape would frame it.
	const int sw = state->source_width;
	const int sh = state->source_height;
	const int tw = state->target_width;
	const int th = state->target_height;
	int crop_width = sw;
	int crop_height = sh;
	if (static_cast<int64_t>(sw) * th > static_cast<int64_t>(sh) * tw)
		crop_width = static_cast<int>(static_cast<int64_t>(sh) * tw / th) & ~1;
	else crop_height = static_cast<int>(static_cast<int64_t>(sw) * th / tw) & ~1;
	const int left = (sw - crop_width) / 2 & ~1;
	const int top = (sh - crop_height) / 2 & ~1;

	const uint8_t* luma = state->buffer.data();
	const uint8_t* cb = luma + static_cast<size_t>(sw) * sh;
	const uint8_t* cr = cb + static_cast<size_t>(sw / 2) * (sh / 2);
	frame.luma.resize(tw, th);
	frame.cb.resize(tw / 2, th);
	frame.cr.resize(tw / 2, th);
	resample_plane(luma, sw, left, top, crop_width, crop_height, frame.luma);
	resample_plane(cb, sw / 2, left / 2, top / 2, crop_width / 2, crop_height / 2, frame.cb);
	resample_plane(cr, sw / 2, left / 2, top / 2, crop_width / 2, crop_height / 2, frame.cr);
	seconds = state->frames++ * state->frame_seconds;
	return true;
}

clip_reader::~clip_reader()
{
	close();
}

void clip_reader::close()
{
	delete state;
	state = nullptr;
}

bool clip_reader::open(const std::wstring& path, const int width, const int height, std::string& error)
{
	close();
	if (width <= 0 || height <= 0 || width % 2 != 0)
	{
		error = "a clip is replayed at an even width";
		return false;
	}
	if (is_y4m(path)) return open_y4m(path, width, height, error);
	auto owned = std::make_unique<impl>();

	// The advanced processor converts whatever the clip holds to the wanted size and format.
	ComPtr<IMFAttributes> attributes;
	HRESULT hr = MFCreateAttributes(&attributes, 1);
	if (SUCCEEDED(hr)) hr = attributes->SetUINT32(MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, TRUE);
	if (SUCCEEDED(hr)) hr = MFCreateSourceReaderFromURL(path.c_str(), attributes.Get(), &owned->reader);
	if (FAILED(hr))
	{
		error = clip_failure("could not open the clip", hr);
		return false;
	}
	owned->reader->SetStreamSelection(static_cast<DWORD>(MF_SOURCE_READER_ALL_STREAMS), FALSE);
	owned->reader->SetStreamSelection(first_video_stream, TRUE);

	bool chosen = false;
	for (const GUID& subtype : {MFVideoFormat_YUY2, MFVideoFormat_NV12})
	{
		ComPtr<IMFMediaType> type;
		if (FAILED(MFCreateMediaType(&type)) || FAILED(describe_video(type.Get(), subtype, width, height, 0))) continue;
		if (SUCCEEDED(owned->reader->SetCurrentMediaType(first_video_stream, nullptr, type.Get())))
		{
			chosen = true;
			break;
		}
	}

	ComPtr<IMFMediaType> current;
	if (!chosen || FAILED(owned->reader->GetCurrentMediaType(first_video_stream, &current)) ||
	    FAILED(current->GetGUID(MF_MT_SUBTYPE, &owned->subtype)) ||
	    FAILED(MFGetAttributeSize(current.Get(), MF_MT_FRAME_SIZE, &owned->width, &owned->height)) ||
	    owned->width != static_cast<UINT32>(width) || owned->height != static_cast<UINT32>(height))
	{
		error = "the clip cannot be converted to the capture format";
		return false;
	}

	UINT32 stride = 0;
	owned->stride = SUCCEEDED(current->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride))
	                    ? static_cast<LONG>(static_cast<INT32>(stride))
	                    : static_cast<LONG>(owned->subtype == MFVideoFormat_YUY2 ? width * 2 : width);

	state = owned.release();
	return true;
}

bool clip_reader::read(video_frame& frame, double& seconds, std::string& error)
{
	if (state == nullptr)
	{
		error = "no clip is open";
		return false;
	}
	if (state->y4m != nullptr) return read_y4m(frame, seconds, error);

	for (;;)
	{
		DWORD stream_index = 0;
		DWORD flags = 0;
		LONGLONG timestamp = 0;
		ComPtr<IMFSample> sample;
		const HRESULT hr = state->reader->ReadSample(first_video_stream, 0, &stream_index, &flags, &timestamp, &sample);
		if (FAILED(hr))
		{
			error = clip_failure("could not read the clip", hr);
			return false;
		}
		if (flags & MF_SOURCE_READERF_ERROR)
		{
			error = "the clip reported a stream error";
			return false;
		}
		if (flags & MF_SOURCE_READERF_ENDOFSTREAM) return false;
		if (!sample) continue;

		ComPtr<IMFMediaBuffer> buffer;
		BYTE* bytes = nullptr;
		DWORD length = 0;
		if (FAILED(sample->ConvertToContiguousBuffer(&buffer)) || FAILED(buffer->Lock(&bytes, nullptr, &length)))
		{
			error = "a clip frame could not be read";
			return false;
		}
		const int width = static_cast<int>(state->width);
		const int height = static_cast<int>(state->height);
		const bool decoded = state->subtype == MFVideoFormat_YUY2
		                         ? decode_yuy2(frame, bytes, length, width, height, state->stride)
		                         : decode_nv12(frame, bytes, length, width, height, std::abs(state->stride), height);
		buffer->Unlock();
		if (!decoded)
		{
			error = "a clip frame was not in the expected layout";
			return false;
		}
		seconds = static_cast<double>(timestamp) / 10000000.0;
		return true;
	}
}

bool load_still(const std::wstring& path, const int width, const int height, video_frame& frame)
{
	if (width <= 0 || height <= 0 || width % 2 != 0) return false;

	ComPtr<IWICImagingFactory> factory;
	ComPtr<IWICBitmapDecoder> decoder;
	ComPtr<IWICBitmapFrameDecode> source;
	ComPtr<IWICBitmapScaler> scaler;
	ComPtr<IWICFormatConverter> converter;
	if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))))
		return false;
	if (FAILED(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand,
	                                              &decoder)))
		return false;
	if (FAILED(decoder->GetFrame(0, &source))) return false;
	if (FAILED(factory->CreateBitmapScaler(&scaler))) return false;
	if (FAILED(scaler->Initialize(source.Get(), static_cast<UINT>(width), static_cast<UINT>(height),
	                              WICBitmapInterpolationModeFant)))
		return false;
	if (FAILED(factory->CreateFormatConverter(&converter))) return false;
	if (FAILED(converter->Initialize(scaler.Get(), GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0.0,
	                                 WICBitmapPaletteTypeCustom)))
		return false;

	std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
	if (FAILED(converter->CopyPixels(nullptr, static_cast<UINT>(width) * 4, static_cast<UINT>(pixels.size()),
	                                 pixels.data())))
		return false;

	const auto clamp_byte = [](const int value)
	{
		return static_cast<uint8_t>(std::clamp(value, 0, 255));
	};
	frame.luma.resize(width, height);
	frame.cb.resize(width / 2, height);
	frame.cr.resize(width / 2, height);
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; x += 2)
		{
			int cb_sum = 0;
			int cr_sum = 0;
			for (int offset = 0; offset < 2; ++offset)
			{
				const uint8_t* pixel = pixels.data() + (static_cast<size_t>(y) * width + x + offset) * 4;
				const int blue = pixel[0];
				const int green = pixel[1];
				const int red = pixel[2];
				frame.luma(x + offset, y) = clamp_byte(16 + ((66 * red + 129 * green + 25 * blue + 128) >> 8));
				cb_sum += 128 + ((-38 * red - 74 * green + 112 * blue + 128) >> 8);
				cr_sum += 128 + ((112 * red - 94 * green - 18 * blue + 128) >> 8);
			}
			frame.cb(x / 2, y) = clamp_byte(cb_sum / 2);
			frame.cr(x / 2, y) = clamp_byte(cr_sum / 2);
		}
	return true;
}
