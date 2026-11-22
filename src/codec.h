// The low-ban wire and the quality metrics that compare it with the H.264 baseline.
//
// One packet a tick carries everything the receiver needs. When the faces have moved it
// carries their new landmarks, and with them, when it helps, the warp of the face the receiver
// holds onto them. Then come the 16x16 macroblocks that most need repair: each skipped along
// the vector its neighbours predict, moved by one quarter-pixel vector or by one for each of
// its 8x8 blocks, or predicted from the picture around it along one of nine directions, and
// corrected by quantised DCT residuals. A loop filter then smooths the block edges the packet
// touched, at both ends. The face importance map steers each macroblock's quantiser and the
// order macroblocks are served in. Every syntax element is range coded (entropy.h) against
// models both ends adapt identically.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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

constexpr int transform_block = 8;

// Quantiser steps: the quantiser doubles every six, from 4 at step 0. A wire is coded at a
// quality, the step the background gets; face macroblocks are finer by up to nine steps, in
// proportion to their importance.
constexpr int quantizer_steps = 37;

// A whole picture intra coded exactly as a keyframe at `quality` codes it, loop filter and
// all, with no wire involved.
void intra_picture(const image_u8& source, const image_u8* importance, int quality, image_u8& reconstructed);

// The loop filter over every block edge of a picture coded wholly intra, as a keyframe's
// decoder runs it; `steps` holds each 8x8 block's quantiser step in raster order. Where the
// step across an edge is larger than the quantiser explains it is the scene's, and stays.
void deblock_intra(image_u8& picture, const std::vector<uint8_t>& steps);

struct lowban_packet
{
	std::vector<uint8_t> bytes;
	bool keyframe = false;
	bool geometry = false;     // carries new landmarks
	bool warped = false;       // and moves the face the receiver holds onto them
	bool limited = false;      // macroblocks worth sending were left for want of bytes
	int updated_blocks = 0;    // 8x8 blocks, in the macroblocks it updated
	size_t geometry_bytes = 0; // of `bytes`, to the nearest byte
};

struct wire_models;

// Both ends of the low-ban wire, as an in-process encoder and decoder whose state never mixes.
//
// Landmarks are in coded-picture pixels. Before a block is coded, both ends can warp the face
// they hold from the landmarks it was drawn at onto the new ones, so a face that only moved
// costs its landmarks; the encoder commits that warp only when it brings the picture closer to
// the source. Macroblocks then compete for the packet's bytes on distortion removed per bit,
// weighted by a bounded face importance and promoted once they have waited too long, so a
// region the face moved off is always repaired eventually.
//
// Every packet from encode must be handed to decode: the encoder advances its picture, its
// landmarks and its models assuming the packet lands. A packet decode rejects changes nothing.
class lowban_wire
{
public:
	void reset();

	// The first packet, and any whose picture size differs, is a keyframe that codes every
	// macroblock. After that a packet carries what is worth its bits within `max_packet_bytes`, 0
	// meaning no cap; new landmarks are always sent. Empty bytes mean there is nothing worth
	// sending, and then nothing is committed. `quality` is the background's quantiser step.
	lowban_packet encode(const image_u8& source, const image_u8* importance, const face_geometry& faces,
	                     bool force_keyframe, size_t max_packet_bytes, int quality);
	bool decode(const lowban_packet& packet, image_u8& picture);

	// What the receiver shows: its decoded picture, which the loop filter has already smoothed.
	// Decoder state only.
	bool show(image_u8& shown) const;

	// The landmarks each end's picture is drawn at.
	const face_geometry& transmitted() const { return encoder.geometry; }
	const face_geometry& received() const { return decoder.geometry; }

	// Whether the encoder holds exactly what the decoder holds: the picture, every block's
	// quantiser and the landmarks. True after every packet that reached decode.
	bool in_sync() const;

private:
	struct side
	{
		image_u8 picture;
		std::vector<uint8_t> steps;  // each 8x8 block's quantiser step, raster order
		std::vector<uint8_t> recent; // the macroblocks the last packet updated
		face_geometry geometry;
		std::shared_ptr<const wire_models> models; // replaced, never changed, so a copy is cheap
	};

	side encoder;
	side decoder;
	std::vector<uint16_t> encoder_age; // packets a macroblock with work to do has been passed over
};
