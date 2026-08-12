#include "capture.h"

#include <memory>

#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

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

bool webcam::read(video_frame& frame)
{
	if (!is_open()) return false;

	DWORD stream_index = 0;
	DWORD flags = 0;
	LONGLONG timestamp = 0;
	ComPtr<IMFSample> sample;

	if (FAILED(state->reader->ReadSample(first_video_stream, 0, &stream_index, &flags,
	                                     &timestamp, &sample)))
		return false;

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
