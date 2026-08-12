// The two low bandwidth codecs, their wire streams and the quality metrics that compare them.
// One is a hand-designed transform codec, the other a learned patch autoencoder; both are fed
// the same face importance map and both schedule their packets by the same rule.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "image.h"

struct reconstruction_quality
{
	double rmse = 0.0;
	double psnr = 0.0;
	double weighted_rmse = 0.0;
	double weighted_psnr = 0.0;
};

reconstruction_quality measure_quality(const image_u8& source, const image_u8& reconstructed,
	                                    const image_u8* importance = nullptr);
void add_face_importance(image_u8& importance, const rect_i& face, const std::vector<point_i>& landmarks);

// An intra-frame transform codec in the same family as traditional MPEG codecs. Luma is
// transformed in 8x8 DCT blocks, quantised, run-length accounted, and reconstructed. The
// importance map lowers quantisation around facial features without changing the algorithm.
class transform_codec
{
public:
	static constexpr int block_size = 8;
	static constexpr int block_coefficients = block_size * block_size;

	// Straight intra coding of a whole frame, with no wire format involved. The streams below
	// are what actually measure rate; this is here for comparisons that only need the picture.
	void reconstruct(const image_u8& source, const image_u8* importance, image_u8& reconstructed) const;
	void encode_coefficients(const image_u8& source, const image_u8* importance,
		std::vector<int16_t>& coefficients, std::vector<uint8_t>& quantizers) const;
	bool reconstruct_coefficients(const std::vector<int16_t>& coefficients,
		const std::vector<uint8_t>& quantizers, int width, int height, image_u8& reconstructed) const;

	// Quantised transform of an 8x8 window of an already decoded frame, displaced by a motion
	// vector. Both endpoints run this over the same reference, so a block that merely moved
	// costs its vector and a residual instead of a fresh copy of its texture.
	void predict_block(const image_u8& reference, int block_x, int block_y, int dx, int dy,
		int quantizer, int16_t* coefficients) const;
};

struct transform_packet
{
	std::vector<uint8_t> bytes;
	bool keyframe = false;
	int updated_blocks = 0;
};

// Serializes 8x8 transform blocks over time. A block is coded either against what the decoder
// already holds for it or against a motion-compensated window of the previous decoded frame,
// whichever is cheaper. Blocks then compete for a fixed packet budget on distortion removed
// per byte, so a region the face has moved away from can outbid the face itself once its
// error is large enough. Without that, stale blocks are never repaired and a moving head
// leaves copies of itself behind.
//
// Every packet from encode must be handed to decode. The encoder advances its reference frame
// assuming the packet lands, so a dropped or rejected packet leaves the two sides predicting
// from different pictures.
class temporal_transform_stream
{
public:
	void reset();
	transform_packet encode(const transform_codec& codec, const image_u8& source,
		const image_u8* importance, bool force_keyframe, size_t max_packet_bytes);
	bool decode(const transform_codec& codec, const transform_packet& packet,
		int width, int height, image_u8& reconstructed);

private:
	std::vector<int16_t> encoder_coefficients;
	std::vector<int16_t> decoder_coefficients;
	std::vector<uint8_t> encoder_quantizers;
	std::vector<uint8_t> decoder_quantizers;
	std::vector<uint16_t> encoder_age; // packets a block has waited, to stop starvation
	image_u8 encoder_reference; // the encoder's copy of what the decoder is showing
	image_u8 decoder_reference; // what motion vectors in the next packet may point into
	int frame_width = 0;
	int frame_height = 0;
};

// A minimal autoencoder used as a stand-in for a learned video codec.
//
// The frame is cut into fixed 20x20 luma patches. Each patch's mean is sent exactly and the
// remaining texture is squeezed through a narrow bottleneck. Training is online; transmission
// uses the temporal packet stream below so unchanged and low-priority patches do not consume
// a full-frame payload.
class patch_autoencoder
{
public:
	static constexpr int patch_size = 20;
	static constexpr int patch_pixels = patch_size * patch_size;
	static constexpr int code_size = 3;
	static constexpr int hidden_size = 32;

	// One exact mean byte plus the latent code. Brightness is by far the highest variance
	// direction in a patch, and eight bits buy it outright for less than a bottleneck unit
	// spent trying to represent it.
	static constexpr int bytes_per_patch = code_size + 1;

	patch_autoencoder();

	void reset();

	// Runs a batch of SGD steps on random patches of the frame and returns the mean squared
	// error over that batch.
	float train(const image_u8& luma, int batch = 256, const image_u8* importance = nullptr);

	void reconstruct(const image_u8& src, image_u8& dst) const;

	// bytes_per_patch signed values per patch: the mean biased by -128, then the code.
	void encode_codes(const image_u8& src, std::vector<int8_t>& codes) const;
	bool reconstruct_codes(const std::vector<int8_t>& codes, int width, int height, image_u8& dst) const;

	int steps() const { return training_steps; }
	float loss() const { return recent_loss; }

	static int64_t bits_per_frame(int width, int height)
	{
		return static_cast<int64_t>(width / patch_size) * (height / patch_size) * bytes_per_patch * 8;
	}

private:
	// The analysis side stays deliberately cheap and the synthesis side carries the capacity,
	// which is the shape every practical learned codec settles on: one encode per patch, but
	// the decoder is what has to invent detail from three numbers.
	struct dense
	{
		int inputs = 0;
		int outputs = 0;
		std::vector<float> weights; // outputs x inputs
		std::vector<float> bias;
		std::vector<float> weight_rate; // running mean square gradient, per weight
		std::vector<float> bias_rate;

		void reset(int inputs, int outputs, std::mt19937& rng);
		void forward(const float* input, float* output, bool activate) const;
		void backward(const float* input, const float* gradient, float* input_gradient);
	};

	dense analysis; // patch_pixels -> code_size, tanh
	dense synthesis_hidden; // code_size -> hidden_size, tanh
	dense synthesis_pixels; // hidden_size -> patch_pixels, linear

	std::mt19937 rng;
	int training_steps = 0;
	float recent_loss = 0.0f;

	// Zero-mean, unit-ish scaled patch. Returns the quantised mean that was removed.
	int normalise(const image_u8& src, int px, int py, float* input) const;
	void encode(const float* input, float* code) const;
	void decode(const float* code, float* hidden, float* output) const;
};

struct latent_packet
{
	std::vector<uint8_t> bytes;
	bool keyframe = false;
	int updated_patches = 0;
};

// Stateful encoder and decoder for the learned patch representation. A keyframe contains all
// latent bytes in raster order. Delta packets carry patch indices and signed latent deltas,
// both variable-length encoded, ranked by latent error per byte with face importance as a
// bounded multiplier, and never exceed max_packet_bytes.
class temporal_patch_stream
{
public:
	void reset();
	latent_packet encode(const patch_autoencoder& model, const image_u8& source,
		const image_u8* importance, bool force_keyframe, size_t max_packet_bytes);
	bool decode(const patch_autoencoder& model, const latent_packet& packet,
		int width, int height, image_u8& reconstructed);

private:
	std::vector<int8_t> encoder_codes;
	std::vector<int8_t> decoder_codes;
	std::vector<uint16_t> encoder_age; // packets a patch has waited, to stop starvation
	int frame_width = 0;
	int frame_height = 0;
};
