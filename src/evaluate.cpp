// Offline rate-distortion evaluation over the PNG files in ../samples.
#include <SDKDDKVer.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "codec.h"
#include "engine.h"
#include "face.h"
#include "landmarks.h"

namespace
{
	using Microsoft::WRL::ComPtr;

	struct evaluation_sample
	{
		std::wstring name;
		video_frame frame;
		image_u8 importance;
		std::vector<std::vector<point_i>> parts;
		int faces = 0;
	};

	std::wstring sample_pattern()
	{
		wchar_t module[MAX_PATH] = {};
		const DWORD length = GetModuleFileNameW(nullptr, module, MAX_PATH);
		if (length == 0 || length >= MAX_PATH) return L"samples\\*.png";
		std::wstring path(module, length);
		const size_t slash = path.find_last_of(L'\\');
		path.resize(slash == std::wstring::npos ? 0 : slash + 1);
		return path + L"..\\samples\\*.png";
	}

	uint8_t clamp_byte(const int value)
	{
		return static_cast<uint8_t>(std::clamp(value, 0, 255));
	}

	bool load_png(IWICImagingFactory* factory, const std::wstring& path, video_frame& result)
	{
		ComPtr<IWICBitmapDecoder> decoder;
		ComPtr<IWICBitmapFrameDecode> source;
		ComPtr<IWICBitmapScaler> scaler;
		ComPtr<IWICFormatConverter> converter;
		if (FAILED(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
			WICDecodeMetadataCacheOnDemand, &decoder))) return false;
		if (FAILED(decoder->GetFrame(0, &source))) return false;
		if (FAILED(factory->CreateBitmapScaler(&scaler))) return false;
		if (FAILED(scaler->Initialize(source.Get(), 640, 480, WICBitmapInterpolationModeFant))) return false;
		if (FAILED(factory->CreateFormatConverter(&converter))) return false;
		if (FAILED(converter->Initialize(scaler.Get(), GUID_WICPixelFormat32bppBGRA,
			WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) return false;

		std::vector<uint8_t> pixels(640u * 480u * 4u);
		if (FAILED(converter->CopyPixels(nullptr, 640 * 4, static_cast<UINT>(pixels.size()), pixels.data())))
			return false;

		result.luma.resize(640, 480);
		result.cb.resize(320, 480);
		result.cr.resize(320, 480);
		for (int y = 0; y < 480; ++y)
		{
			for (int x = 0; x < 640; x += 2)
			{
				int cb_sum = 0;
				int cr_sum = 0;
				for (int offset = 0; offset < 2; ++offset)
				{
					const uint8_t* pixel = pixels.data() + (static_cast<size_t>(y) * 640 + x + offset) * 4;
					const int blue = pixel[0];
					const int green = pixel[1];
					const int red = pixel[2];
					result.luma(x + offset, y) = clamp_byte(16 + ((66 * red + 129 * green + 25 * blue + 128) >> 8));
					cb_sum += 128 + ((-38 * red - 74 * green + 112 * blue + 128) >> 8);
					cr_sum += 128 + ((112 * red - 94 * green - 18 * blue + 128) >> 8);
				}
				result.cb(x / 2, y) = clamp_byte(cb_sum / 2);
				result.cr(x / 2, y) = clamp_byte(cr_sum / 2);
			}
		}
		return true;
	}
}

int run_sample_evaluation()
{
	const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	const bool uninitialise = SUCCEEDED(com);
	if (FAILED(com) && com != RPC_E_CHANGED_MODE)
	{
		std::printf("could not initialise COM (0x%08lx)\n", static_cast<unsigned long>(com));
		return 1;
	}

	ComPtr<IWICImagingFactory> factory;
	if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
		IID_PPV_ARGS(&factory))))
	{
		std::printf("could not create Windows image decoder\n");
		if (uninitialise) CoUninitialize();
		return 1;
	}

	shape_predictor predictor;
	std::string model_error;
	const bool have_landmarks = predictor.load(asset_path(L"shape_predictor_68_face_landmarks.dat"), model_error);
	if (!have_landmarks) std::printf("landmark model unavailable: %s\n", model_error.c_str());
	const face_detector detector;

	const std::wstring pattern = sample_pattern();
	WIN32_FIND_DATAW found = {};
	const HANDLE search = FindFirstFileW(pattern.c_str(), &found);
	if (search == INVALID_HANDLE_VALUE)
	{
		std::printf("no PNG samples found\n");
		if (uninitialise) CoUninitialize();
		return 1;
	}

	const size_t slash = pattern.find_last_of(L'\\');
	const std::wstring directory = pattern.substr(0, slash + 1);
	std::vector<evaluation_sample> samples;
	do
	{
		evaluation_sample sample;
		sample.name = found.cFileName;
		if (!load_png(factory.Get(), directory + found.cFileName, sample.frame))
		{
			std::printf("could not decode %ls\n", found.cFileName);
			continue;
		}

		sample.importance.resize(sample.frame.width(), sample.frame.height());
		sample.importance.fill(32);
		const face_detection detection = detector.detect(sample.frame);
		sample.faces = static_cast<int>(detection.faces.size());
		for (const rect_i& face : detection.faces)
		{
			const std::vector<point_i> parts = have_landmarks ? predictor.predict(sample.frame.luma, face)
				: std::vector<point_i>{};
			add_face_importance(sample.importance, face, parts);
			if (!parts.empty())
			{
				sample.parts.push_back(parts);
			}
		}
		samples.push_back(std::move(sample));
	}
	while (FindNextFileW(search, &found));
	FindClose(search);

	std::sort(samples.begin(), samples.end(), [](const evaluation_sample& a, const evaluation_sample& b)
	{
		return a.name < b.name;
	});

	patch_autoencoder neural;
	for (int epoch = 0; epoch < 50; ++epoch)
		for (const auto& sample : samples) neural.train(sample.frame.luma, 256, &sample.importance);

	transform_codec traditional;
	double traditional_bits = 0.0;
	double neural_bits = 0.0;
	double traditional_psnr = 0.0;
	double traditional_weighted_psnr = 0.0;
	double neural_psnr = 0.0;
	double neural_weighted_psnr = 0.0;

	std::printf("sample,faces,traditional_kbit,traditional_psnr,traditional_face_psnr,neural_kbit,neural_psnr,neural_face_psnr\n");
	for (const auto& sample : samples)
	{
		image_u8 traditional_image;
		image_u8 neural_image;
		temporal_transform_stream traditional_wire;
		temporal_landmark_stream traditional_point_wire;
		const transform_packet traditional_packet = traditional_wire.encode(
			traditional, sample.frame.luma, &sample.importance, true, 0);
		const landmark_packet traditional_points = traditional_point_wire.encode(sample.parts, true);
		std::vector<std::vector<point_i>> traditional_decoded_parts;
		if (!traditional_wire.decode(traditional, traditional_packet, sample.frame.width(), sample.frame.height(),
			traditional_image) || !traditional_point_wire.decode(traditional_points, traditional_decoded_parts))
		{
			std::printf("could not decode traditional evaluation packet for %ls\n", sample.name.c_str());
			continue;
		}
		const int64_t sample_traditional_bits = static_cast<int64_t>(traditional_packet.bytes.size()
			+ traditional_points.bytes.size()) * 8;
		temporal_patch_stream image_wire;
		temporal_landmark_stream landmark_wire;
		const latent_packet image_packet = image_wire.encode(neural, sample.frame.luma, &sample.importance, true, 1500);
		const landmark_packet point_packet = landmark_wire.encode(sample.parts, true);
		std::vector<std::vector<point_i>> decoded_parts;
		if (!image_wire.decode(neural, image_packet, sample.frame.width(), sample.frame.height(), neural_image)
			|| !landmark_wire.decode(point_packet, decoded_parts))
		{
			std::printf("could not decode evaluation packet for %ls\n", sample.name.c_str());
			continue;
		}
		const int64_t sample_neural_bits = static_cast<int64_t>(image_packet.bytes.size()
			+ point_packet.bytes.size()) * 8;
		const reconstruction_quality tq = measure_quality(sample.frame.luma, traditional_image, &sample.importance);
		const reconstruction_quality nq = measure_quality(sample.frame.luma, neural_image, &sample.importance);

		std::printf("%ls,%d,%.1f,%.2f,%.2f,%.1f,%.2f,%.2f\n", sample.name.c_str(), sample.faces,
			sample_traditional_bits / 1000.0, tq.psnr, tq.weighted_psnr,
			sample_neural_bits / 1000.0, nq.psnr, nq.weighted_psnr);
		traditional_bits += sample_traditional_bits;
		neural_bits += sample_neural_bits;
		traditional_psnr += tq.psnr;
		traditional_weighted_psnr += tq.weighted_psnr;
		neural_psnr += nq.psnr;
		neural_weighted_psnr += nq.weighted_psnr;
	}

	if (!samples.empty())
	{
		const double count = static_cast<double>(samples.size());
		std::printf("mean,%zu,%.1f,%.2f,%.2f,%.1f,%.2f,%.2f\n", samples.size(),
			traditional_bits / count / 1000.0, traditional_psnr / count, traditional_weighted_psnr / count,
			neural_bits / count / 1000.0, neural_psnr / count, neural_weighted_psnr / count);
		int max_faces = 0;
		for (const auto& sample : samples) max_faces = std::max(max_faces, static_cast<int>(sample.parts.size()));
		const double mean_keyframe_bytes = neural_bits / count / 8.0;
		const double neural_reference_kbit_per_second = (mean_keyframe_bytes + 14.0 * (1500.0 + max_faces * 80.0))
			* 8.0 / 3.0 / 1000.0;
		const double traditional_reference_kbit_per_second = (1200.0 + max_faces * 80.0) * 5.0 * 8.0 / 1000.0;
		std::printf("configured_traditional_5hz_ceiling_kbit_per_second,%.1f\n",
			traditional_reference_kbit_per_second);
		std::printf("configured_neural_5hz_reference_kbit_per_second,%.1f\n",
			neural_reference_kbit_per_second);
	}

	factory.Reset();
	if (uninitialise) CoUninitialize();
	return samples.empty() ? 1 : 0;
}
