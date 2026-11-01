// The low-ban picture codec, its wire stream and the quality metrics that compare it with the
// H.264 baseline. The codec is a hand-designed block transform: the face importance map steers
// its quantiser and its packet scheduling, and the transmitted landmarks warp the face it
// already holds before a single block is coded.
#pragma once

#include <cstdint>
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

	// Straight intra coding of a whole frame, with no wire format involved. The stream below
	// is what actually measures rate; this is here for comparisons that only need the picture.
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
	bool warped = false; // the packet commits a landmark warp of the face the decoder holds
	int updated_blocks = 0;
};

// Serializes 8x8 transform blocks over time. Before any block is coded, both ends warp the
// face they hold onto the newest transmitted geometry, so a face that only moved or changed
// expression costs nothing beyond the landmarks. A block is then coded against what the
// decoder holds for it or against a motion-compensated window of the previous decoded frame,
// whichever is cheaper, and blocks compete for a fixed packet budget on distortion removed
// per byte, so a region the face has moved away from can outbid the face itself once its
// error is large enough. Without that, stale blocks are never repaired and a moving head
// leaves copies of itself behind.
//
// `geometry` is the transmitted landmark geometry in the coded picture's coordinates, and the
// encoder must be given exactly what the decoder will hold when the packet arrives. Every
// packet from encode must be handed to decode: the encoder advances its reference frame and
// geometry assuming the packet lands, so a dropped or rejected packet leaves the two sides
// predicting from different pictures.
class temporal_transform_stream
{
public:
	void reset();
	transform_packet encode(const transform_codec& codec, const image_u8& source,
		const image_u8* importance, bool force_keyframe, size_t max_packet_bytes,
		const face_geometry& geometry = {});
	bool decode(const transform_codec& codec, const transform_packet& packet,
		int width, int height, image_u8& reconstructed, const face_geometry& geometry = {});

	// What the receiver shows before the next packet arrives: its decoded picture with the
	// face warped onto newer geometry, unless the encoder's last word on the warp was that it
	// is wrong for this picture. Uses decoder state only.
	bool animate(const face_geometry& geometry, image_u8& shown) const;

private:
	std::vector<int16_t> encoder_coefficients;
	std::vector<int16_t> decoder_coefficients;
	std::vector<uint8_t> encoder_quantizers;
	std::vector<uint8_t> decoder_quantizers;
	std::vector<uint16_t> encoder_age; // packets a block has waited, to stop starvation
	image_u8 encoder_reference; // the encoder's copy of what the decoder is showing
	image_u8 decoder_reference; // what motion vectors in the next packet may point into
	face_geometry encoder_geometry; // the geometry each reference was decoded at
	face_geometry decoder_geometry;
	bool encoder_animating = false; // the encoder's copy of whether the receiver animates
	bool decoder_animating = false;
	int frame_width = 0;
	int frame_height = 0;
};
