#include "baseline.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>

#include <windows.h>
#include <codecapi.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>
#include <strmif.h>
#include <wmcodecdsp.h>
#include <wrl/client.h>

#pragma comment(lib, "Mfplat.lib")
#pragma comment(lib, "Mfuuid.lib")
#pragma comment(lib, "wmcodecdspuuid.lib")

using Microsoft::WRL::ComPtr;

namespace
{
	// One key frame at the start, like low-ban's startup keyframe, and never another.
	constexpr UINT32 group_of_pictures = 1000000;

	// A healthy round trip takes well under a second; this much is a hang.
	constexpr DWORD probe_timeout_ms = 30000;

	// Set in the probe process itself, which must not wait on a probe of its own.
	bool probing = false;

	std::string failure(const char* what, const HRESULT hr)
	{
		char text[192] = {};
		snprintf(text, sizeof(text), "%s (0x%08lX)", what, static_cast<unsigned long>(hr));
		return text;
	}

	HRESULT set_property(ICodecAPI* api, const GUID& property, const VARTYPE type, const ULONGLONG value)
	{
		VARIANT variant;
		VariantInit(&variant);
		variant.vt = type;
		if (type == VT_BOOL) variant.boolVal = value != 0 ? VARIANT_TRUE : VARIANT_FALSE;
		else if (type == VT_UI8) variant.ullVal = value;
		else variant.ulVal = static_cast<ULONG>(value);
		return api->SetValue(&property, &variant);
	}

	HRESULT video_type(const GUID& subtype, const h264_settings& settings, ComPtr<IMFMediaType>& type)
	{
		HRESULT hr = MFCreateMediaType(&type);
		if (SUCCEEDED(hr)) hr = type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		if (SUCCEEDED(hr)) hr = type->SetGUID(MF_MT_SUBTYPE, subtype);
		if (SUCCEEDED(hr)) hr = MFSetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, settings.width, settings.height);
		if (SUCCEEDED(hr)) hr = MFSetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, settings.frames_per_second, 1);
		if (SUCCEEDED(hr)) hr = MFSetAttributeRatio(type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
		if (SUCCEEDED(hr)) hr = type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
		return hr;
	}

	HRESULT make_sample(const uint8_t* bytes, const size_t length, const LONGLONG time, const LONGLONG duration,
		ComPtr<IMFSample>& sample)
	{
		ComPtr<IMFMediaBuffer> buffer;
		HRESULT hr = MFCreateMemoryBuffer(static_cast<DWORD>(length), &buffer);
		BYTE* data = nullptr;
		if (SUCCEEDED(hr)) hr = buffer->Lock(&data, nullptr, nullptr);
		if (SUCCEEDED(hr))
		{
			std::memcpy(data, bytes, length);
			buffer->Unlock();
			hr = buffer->SetCurrentLength(static_cast<DWORD>(length));
		}
		if (SUCCEEDED(hr)) hr = MFCreateSample(&sample);
		if (SUCCEEDED(hr)) hr = sample->AddBuffer(buffer.Get());
		if (SUCCEEDED(hr)) hr = sample->SetSampleTime(time);
		if (SUCCEEDED(hr)) hr = sample->SetSampleDuration(duration);
		return hr;
	}

	// One output from a synchronous transform: S_OK with a sample, MF_E_TRANSFORM_NEED_MORE_INPUT
	// when nothing is ready, or MF_E_TRANSFORM_STREAM_CHANGE when the output type must be set again.
	HRESULT pull_output(IMFTransform* transform, const DWORD minimum_size, ComPtr<IMFSample>& sample)
	{
		MFT_OUTPUT_STREAM_INFO info = {};
		HRESULT hr = transform->GetOutputStreamInfo(0, &info);
		if (FAILED(hr)) return hr;

		const bool allocates = (info.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES
			| MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES)) != 0;
		ComPtr<IMFSample> provided;
		if (!allocates)
		{
			ComPtr<IMFMediaBuffer> buffer;
			hr = MFCreateMemoryBuffer(std::max(info.cbSize, minimum_size), &buffer);
			if (SUCCEEDED(hr)) hr = MFCreateSample(&provided);
			if (SUCCEEDED(hr)) hr = provided->AddBuffer(buffer.Get());
			if (FAILED(hr)) return hr;
		}

		MFT_OUTPUT_DATA_BUFFER output = {};
		output.pSample = provided.Get();
		DWORD status = 0;
		hr = transform->ProcessOutput(0, 1, &output, &status);
		if (output.pEvents != nullptr) output.pEvents->Release();
		if (allocates)
		{
			if (output.pSample != nullptr) sample.Attach(output.pSample);
		}
		else if (SUCCEEDED(hr))
		{
			sample = provided;
		}
		return hr;
	}

	HRESULT append_bytes(IMFSample* sample, std::vector<uint8_t>& bytes)
	{
		ComPtr<IMFMediaBuffer> buffer;
		HRESULT hr = sample->ConvertToContiguousBuffer(&buffer);
		BYTE* data = nullptr;
		DWORD length = 0;
		if (SUCCEEDED(hr)) hr = buffer->Lock(&data, nullptr, &length);
		if (SUCCEEDED(hr))
		{
			bytes.insert(bytes.end(), data, data + length);
			buffer->Unlock();
		}
		return hr;
	}

	// Whether an Annex B stream carries a sequence parameter set.
	bool carries_parameter_sets(const std::vector<uint8_t>& bytes)
	{
		for (size_t i = 0; i + 3 < bytes.size(); ++i)
			if (bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 && (bytes[i + 3] & 0x1f) == 7) return true;
		return false;
	}

	// Drops access unit delimiters and SEI messages from an Annex B stream. The decoder needs
	// neither, and charging them would handicap the baseline: this encoder opens every frame
	// with a delimiter and its first frame with about 300 bytes of SEI. Every other unit keeps
	// its own start code.
	std::vector<uint8_t> essential_units(const std::vector<uint8_t>& stream)
	{
		std::vector<size_t> codes;
		for (size_t i = 0; i + 2 < stream.size(); ++i)
			if (stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1)
			{
				codes.push_back(i);
				i += 2;
			}
		if (codes.empty()) return stream;

		std::vector<uint8_t> kept;
		for (size_t n = 0; n < codes.size(); ++n)
		{
			// A unit never ends in a zero byte, so a zero just before a start code is the
			// leading byte of a four-byte start code.
			const size_t begin = codes[n] > 0 && stream[codes[n] - 1] == 0 ? codes[n] - 1 : codes[n];
			const size_t body = codes[n] + 3;
			size_t end = n + 1 < codes.size() ? codes[n + 1] : stream.size();
			if (n + 1 < codes.size() && end > body && stream[end - 1] == 0) --end;
			if (body >= end) continue;
			const int type = stream[body] & 0x1f;
			if (type == 6 || type == 9) continue;
			kept.insert(kept.end(), stream.begin() + static_cast<ptrdiff_t>(begin),
				stream.begin() + static_cast<ptrdiff_t>(end));
		}
		return kept;
	}
}

struct h264_wire::impl
{
	h264_settings settings;
	ComPtr<IMFTransform> encoder;
	ComPtr<IMFTransform> decoder;
	bool face_offset = false;
	bool parameter_sets_checked = false;
	std::vector<uint8_t> sequence_header;
	LONGLONG frame_duration = 0;
	LONGLONG encoded_frames = 0;
	LONGLONG decoded_frames = 0;
	UINT32 decoded_width = 0;
	UINT32 decoded_height = 0;
	LONG decoded_stride = 0;

	bool open_encoder(std::string& error);
	bool open_decoder(std::string& error);
	bool choose_decoder_output(std::string& error);
	HRESULT drain_encoder(std::vector<uint8_t>& bytes);
	HRESULT drain_decoder(image_u8& picture, std::string& error);
	HRESULT copy_luma(IMFSample* sample, image_u8& picture) const;
};

bool h264_wire::impl::open_encoder(std::string& error)
{
	HRESULT hr = CoCreateInstance(CLSID_MSH264EncoderMFT, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&encoder));
	if (FAILED(hr))
	{
		error = failure("Windows has no H.264 encoder", hr);
		return false;
	}

	ComPtr<ICodecAPI> api;
	hr = encoder.As(&api);
	if (FAILED(hr))
	{
		error = failure("the H.264 encoder has no settings interface", hr);
		return false;
	}

	// The encoder reads these when its output type is set, so they come first.
	const bool constant_rate = settings.kbit_per_second > 0;
	hr = set_property(api.Get(), CODECAPI_AVEncCommonRateControlMode, VT_UI4,
		constant_rate ? eAVEncCommonRateControlMode_CBR : eAVEncCommonRateControlMode_Quality);
	if (SUCCEEDED(hr))
		hr = constant_rate
			? set_property(api.Get(), CODECAPI_AVEncCommonMeanBitRate, VT_UI4, settings.kbit_per_second * 1000ull)
			: set_property(api.Get(), CODECAPI_AVEncVideoEncodeQP, VT_UI8, static_cast<ULONGLONG>(settings.quantizer));
	if (FAILED(hr))
	{
		error = failure("the H.264 encoder rejected its rate control settings", hr);
		return false;
	}

	// Best effort: without any of these the stream is still valid, only less efficient or
	// later to arrive. Main profile, because CABAC is worth a quarter of the bytes here, and
	// not High, whose streams from this encoder in low latency mode did not survive the
	// in-box decoder past the first few macroblocks of each slice.
	set_property(api.Get(), CODECAPI_AVLowLatencyMode, VT_BOOL, 1);
	set_property(api.Get(), CODECAPI_AVEncMPVDefaultBPictureCount, VT_UI4, 0);
	set_property(api.Get(), CODECAPI_AVEncMPVGOPSize, VT_UI4, group_of_pictures);
	set_property(api.Get(), CODECAPI_AVEncCommonQualityVsSpeed, VT_UI4, 100);
	set_property(api.Get(), CODECAPI_AVEncH264CABACEnable, VT_BOOL, 1);
	face_offset = settings.face_offset != 0
		&& SUCCEEDED(set_property(api.Get(), CODECAPI_AVEncVideoROIEnabled, VT_UI4, 1));

	ComPtr<IMFMediaType> output;
	hr = video_type(MFVideoFormat_H264, settings, output);
	if (SUCCEEDED(hr))
		hr = output->SetUINT32(MF_MT_AVG_BITRATE, constant_rate ? settings.kbit_per_second * 1000u : 1000000u);
	if (SUCCEEDED(hr)) hr = output->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Main);
	if (SUCCEEDED(hr)) hr = encoder->SetOutputType(0, output.Get(), 0);
	if (FAILED(hr))
	{
		error = failure("the H.264 encoder rejected the output format", hr);
		return false;
	}

	ComPtr<IMFMediaType> input;
	hr = video_type(MFVideoFormat_NV12, settings, input);
	if (SUCCEEDED(hr)) hr = encoder->SetInputType(0, input.Get(), 0);
	if (FAILED(hr))
	{
		error = failure("the H.264 encoder rejected NV12 input", hr);
		return false;
	}

	ComPtr<IMFMediaType> current;
	UINT32 header_size = 0;
	if (SUCCEEDED(encoder->GetOutputCurrentType(0, &current))
		&& SUCCEEDED(current->GetBlobSize(MF_MT_MPEG_SEQUENCE_HEADER, &header_size)) && header_size > 0)
	{
		sequence_header.resize(header_size);
		if (FAILED(current->GetBlob(MF_MT_MPEG_SEQUENCE_HEADER, sequence_header.data(), header_size, nullptr)))
			sequence_header.clear();
	}

	hr = encoder->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
	if (SUCCEEDED(hr)) hr = encoder->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
	if (FAILED(hr))
	{
		error = failure("the H.264 encoder would not start", hr);
		return false;
	}
	return true;
}

bool h264_wire::impl::open_decoder(std::string& error)
{
	HRESULT hr = CoCreateInstance(CLSID_CMSH264DecoderMFT, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&decoder));
	if (FAILED(hr))
	{
		error = failure("Windows has no H.264 decoder", hr);
		return false;
	}

	// Otherwise the decoder holds frames back for a reordering this stream never uses. This
	// decoder reads the setting from its attribute store; its codec interface rejects it.
	ComPtr<IMFAttributes> attributes;
	if (SUCCEEDED(decoder->GetAttributes(&attributes))) attributes->SetUINT32(CODECAPI_AVLowLatencyMode, TRUE);

	ComPtr<IMFMediaType> input;
	hr = video_type(MFVideoFormat_H264, settings, input);
	if (SUCCEEDED(hr)) hr = decoder->SetInputType(0, input.Get(), 0);
	if (FAILED(hr))
	{
		error = failure("the H.264 decoder rejected the stream format", hr);
		return false;
	}
	if (!choose_decoder_output(error)) return false;

	hr = decoder->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
	if (SUCCEEDED(hr)) hr = decoder->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
	if (FAILED(hr))
	{
		error = failure("the H.264 decoder would not start", hr);
		return false;
	}
	return true;
}

bool h264_wire::impl::choose_decoder_output(std::string& error)
{
	for (DWORD index = 0;; ++index)
	{
		ComPtr<IMFMediaType> type;
		HRESULT hr = decoder->GetOutputAvailableType(0, index, &type);
		if (FAILED(hr))
		{
			error = failure("the H.264 decoder offers no NV12 output", hr);
			return false;
		}

		GUID subtype = GUID_NULL;
		if (FAILED(type->GetGUID(MF_MT_SUBTYPE, &subtype)) || subtype != MFVideoFormat_NV12) continue;
		hr = decoder->SetOutputType(0, type.Get(), 0);
		if (FAILED(hr))
		{
			error = failure("the H.264 decoder rejected NV12 output", hr);
			return false;
		}

		decoded_width = 0;
		decoded_height = 0;
		MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &decoded_width, &decoded_height);
		UINT32 stride = 0;
		decoded_stride = SUCCEEDED(type->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride))
			? static_cast<LONG>(stride) : static_cast<LONG>(decoded_width);
		return true;
	}
}

HRESULT h264_wire::impl::drain_encoder(std::vector<uint8_t>& bytes)
{
	for (;;)
	{
		ComPtr<IMFSample> sample;
		HRESULT hr = pull_output(encoder.Get(), static_cast<DWORD>(settings.width * settings.height * 2), sample);
		if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) return S_OK;
		if (hr == MF_E_TRANSFORM_STREAM_CHANGE)
		{
			ComPtr<IMFMediaType> type;
			hr = encoder->GetOutputAvailableType(0, 0, &type);
			if (SUCCEEDED(hr)) hr = encoder->SetOutputType(0, type.Get(), 0);
			if (FAILED(hr)) return hr;
			continue;
		}
		if (FAILED(hr)) return hr;
		if (sample)
		{
			hr = append_bytes(sample.Get(), bytes);
			if (FAILED(hr)) return hr;
		}
	}
}

HRESULT h264_wire::impl::drain_decoder(image_u8& picture, std::string& error)
{
	for (;;)
	{
		ComPtr<IMFSample> sample;
		HRESULT hr = pull_output(decoder.Get(), static_cast<DWORD>(settings.width * settings.height * 3 / 2), sample);
		if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) return S_OK;
		if (hr == MF_E_TRANSFORM_STREAM_CHANGE)
		{
			if (!choose_decoder_output(error)) return E_FAIL;
			continue;
		}
		if (FAILED(hr)) return hr;
		if (sample)
		{
			hr = copy_luma(sample.Get(), picture);
			if (FAILED(hr)) return hr;
		}
	}
}

HRESULT h264_wire::impl::copy_luma(IMFSample* sample, image_u8& picture) const
{
	ComPtr<IMFMediaBuffer> buffer;
	HRESULT hr = sample->ConvertToContiguousBuffer(&buffer);
	if (FAILED(hr)) return hr;

	const int width = std::min(settings.width, static_cast<int>(decoded_width));
	const int height = std::min(settings.height, static_cast<int>(decoded_height));
	if (width <= 0 || height <= 0) return MF_E_INVALIDMEDIATYPE;

	image_u8 luma(settings.width, settings.height);
	luma.fill(128);
	ComPtr<IMF2DBuffer> planes;
	BYTE* data = nullptr;
	LONG pitch = 0;
	if (SUCCEEDED(buffer.As(&planes)) && SUCCEEDED(planes->Lock2D(&data, &pitch)))
	{
		for (int y = 0; y < height; ++y) std::memcpy(luma.row(y), data + static_cast<ptrdiff_t>(y) * pitch, width);
		planes->Unlock2D();
	}
	else
	{
		DWORD length = 0;
		hr = buffer->Lock(&data, nullptr, &length);
		if (FAILED(hr)) return hr;
		pitch = decoded_stride > 0 ? decoded_stride : static_cast<LONG>(decoded_width);
		if (static_cast<size_t>(length) < static_cast<size_t>(pitch) * height)
		{
			buffer->Unlock();
			return MF_E_BUFFERTOOSMALL;
		}
		for (int y = 0; y < height; ++y) std::memcpy(luma.row(y), data + static_cast<ptrdiff_t>(y) * pitch, width);
		buffer->Unlock();
	}
	picture = std::move(luma);
	return S_OK;
}

h264_wire::~h264_wire()
{
	close();
}

bool h264_wire::open(const h264_settings& settings, std::string& error)
{
	close();
	if (settings.width <= 0 || settings.height <= 0 || settings.width % 2 != 0 || settings.height % 2 != 0
		|| settings.frames_per_second <= 0)
	{
		error = "the H.264 wire needs an even frame size and a frame rate";
		return false;
	}
	if (!h264_safe_to_use(error)) return false;

	auto owned = std::make_unique<impl>();
	owned->settings = settings;
	owned->frame_duration = 10000000LL / settings.frames_per_second;
	if (!owned->open_encoder(error) || !owned->open_decoder(error)) return false;
	state = owned.release();
	return true;
}

void h264_wire::close()
{
	delete state;
	state = nullptr;
}

bool h264_wire::is_open() const
{
	return state != nullptr;
}

bool h264_wire::face_offset_active() const
{
	return state != nullptr && state->face_offset;
}

bool h264_wire::encode(const image_u8& luma, const std::vector<rect_i>& faces, std::vector<uint8_t>& bytes,
	std::string& error)
{
	bytes.clear();
	if (!is_open())
	{
		error = "the H.264 wire is not open";
		return false;
	}

	impl& wire = *state;
	const int width = wire.settings.width;
	const int height = wire.settings.height;
	if (luma.width() != width || luma.height() != height)
	{
		error = "the frame does not match the H.264 wire's size";
		return false;
	}

	std::vector<uint8_t> nv12(static_cast<size_t>(width) * height * 3 / 2, 128);
	for (int y = 0; y < height; ++y) std::memcpy(nv12.data() + static_cast<size_t>(y) * width, luma.row(y), width);

	ComPtr<IMFSample> sample;
	HRESULT hr = make_sample(nv12.data(), nv12.size(), wire.encoded_frames * wire.frame_duration,
		wire.frame_duration, sample);
	if (SUCCEEDED(hr) && wire.face_offset && !faces.empty())
	{
		// The encoder takes one rectangle per frame, so several faces share their union.
		rect_i area = faces.front();
		for (const rect_i& face : faces)
		{
			area.left = std::min(area.left, face.left);
			area.top = std::min(area.top, face.top);
			area.right = std::max(area.right, face.right);
			area.bottom = std::max(area.bottom, face.bottom);
		}
		area.left = std::max(area.left, 0);
		area.top = std::max(area.top, 0);
		area.right = std::min(area.right, width - 1);
		area.bottom = std::min(area.bottom, height - 1);
		if (!area.empty())
		{
			ROI_AREA roi = {};
			roi.rect = {area.left, area.top, area.right + 1, area.bottom + 1};
			roi.QPDelta = wire.settings.face_offset;
			hr = sample->SetBlob(MFSampleExtension_ROIRectangle, reinterpret_cast<const UINT8*>(&roi), sizeof(roi));
		}
	}
	if (FAILED(hr))
	{
		error = failure("could not prepare an H.264 input frame", hr);
		return false;
	}
	++wire.encoded_frames;

	hr = wire.encoder->ProcessInput(0, sample.Get(), 0);
	if (hr == MF_E_NOTACCEPTING)
	{
		hr = wire.drain_encoder(bytes);
		if (SUCCEEDED(hr)) hr = wire.encoder->ProcessInput(0, sample.Get(), 0);
	}
	if (SUCCEEDED(hr)) hr = wire.drain_encoder(bytes);
	if (FAILED(hr))
	{
		error = failure("the H.264 encoder failed", hr);
		return false;
	}
	bytes = essential_units(bytes);

	// The decoder cannot start without the parameter sets. If the encoder kept them out of
	// band they still have to cross the wire, so they are charged to the first frame.
	if (!wire.parameter_sets_checked && !bytes.empty())
	{
		wire.parameter_sets_checked = true;
		if (!carries_parameter_sets(bytes))
			bytes.insert(bytes.begin(), wire.sequence_header.begin(), wire.sequence_header.end());
	}
	return true;
}

bool h264_wire::decode(const std::vector<uint8_t>& bytes, image_u8& picture, std::string& error)
{
	if (!is_open())
	{
		error = "the H.264 wire is not open";
		return false;
	}
	if (bytes.empty()) return true;

	impl& wire = *state;
	ComPtr<IMFSample> sample;
	HRESULT hr = make_sample(bytes.data(), bytes.size(), wire.decoded_frames * wire.frame_duration,
		wire.frame_duration, sample);
	++wire.decoded_frames;
	if (SUCCEEDED(hr)) hr = wire.decoder->ProcessInput(0, sample.Get(), 0);
	if (hr == MF_E_NOTACCEPTING)
	{
		hr = wire.drain_decoder(picture, error);
		if (SUCCEEDED(hr)) hr = wire.decoder->ProcessInput(0, sample.Get(), 0);
	}
	if (SUCCEEDED(hr)) hr = wire.drain_decoder(picture, error);
	if (FAILED(hr))
	{
		if (error.empty()) error = failure("the H.264 decoder failed", hr);
		return false;
	}
	return true;
}

bool h264_safe_to_use(std::string& reason)
{
	if (probing) return true;

	static std::once_flag once;
	static bool safe = false;
	static std::string why;
	std::call_once(once, []
	{
		wchar_t path[MAX_PATH] = {};
		const DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
		if (length == 0 || length >= MAX_PATH)
		{
			why = "could not find this program to test Windows' H.264 encoder";
			return;
		}

		std::wstring command = L"\"" + std::wstring(path) + L"\" /probe-h264";
		STARTUPINFOW startup = {};
		startup.cb = sizeof(startup);
		PROCESS_INFORMATION process = {};
		if (!CreateProcessW(path, command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup, &process))
		{
			why = failure("could not start a process to test Windows' H.264 encoder",
				HRESULT_FROM_WIN32(GetLastError()));
			return;
		}
		CloseHandle(process.hThread);

		DWORD code = 0;
		const DWORD waited = WaitForSingleObject(process.hProcess, probe_timeout_ms);
		if (waited != WAIT_OBJECT_0) TerminateProcess(process.hProcess, 1);
		else GetExitCodeProcess(process.hProcess, &code);
		CloseHandle(process.hProcess);

		if (waited != WAIT_OBJECT_0) why = "Windows' H.264 encoder hung in a test process";
		else if (code != 0) why = failure("Windows' H.264 encoder crashed in a test process", static_cast<HRESULT>(code));
		else safe = true;
	});

	reason = why;
	return safe;
}

int run_h264_probe()
{
	// A crash is the failure being looked for, and must not stop on an error dialog.
	SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
	probing = true;

	const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	const bool started = SUCCEEDED(MFStartup(MF_VERSION));
	{
		// The size and settings the call itself uses, so the probe takes the same paths.
		h264_settings settings;
		settings.width = 320;
		settings.height = 240;
		settings.frames_per_second = 15;
		settings.kbit_per_second = 52;
		settings.face_offset = -6;

		image_u8 frame(settings.width, settings.height);
		for (int y = 0; y < frame.height(); ++y)
			for (int x = 0; x < frame.width(); ++x) frame(x, y) = static_cast<uint8_t>((x * 3 + y * 5) & 0xff);

		// Errors are ordinary unavailability, which the caller meets and reports itself; only
		// a crash or a hang changes the outcome.
		h264_wire wire;
		std::string error;
		std::vector<uint8_t> bytes;
		image_u8 picture;
		bool working = wire.open(settings, error);
		for (int i = 0; working && i < 3; ++i)
			working = wire.encode(frame, {rect_i::from_size(100, 60, 120, 120)}, bytes, error)
				&& wire.decode(bytes, picture, error);
	}
	if (started) MFShutdown();
	if (SUCCEEDED(com)) CoUninitialize();
	return 0;
}