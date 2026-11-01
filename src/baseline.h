// The H.264 baseline: Windows' own encoder and decoder at the same bitrate as the low-ban wire,
// fed the same face boxes as a region of interest quantiser offset. It is a real serialized
// wire - the picture shown is what the decoder made of the encoder's bytes - so the comparison
// with low-ban is between what each receiver actually gets.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "image.h"

struct h264_settings
{
	int width = 0;
	int height = 0;
	int frames_per_second = 15;

	// Constant bit rate target. Zero codes every frame at `quantizer` instead, which is how
	// the still-image evaluation searches for the finest picture that fits a byte budget.
	int kbit_per_second = 0;
	int quantizer = 28;

	// Quantiser offset inside the union of the face boxes; zero turns region of interest
	// coding off. Negative is finer, and six steps halve the quantiser.
	int face_offset = 0;
};

class h264_wire
{
public:
	h264_wire() = default;
	~h264_wire();

	h264_wire(const h264_wire&) = delete;
	h264_wire& operator=(const h264_wire&) = delete;

	// Media Foundation must be started and COM initialised on the calling thread, which is
	// the only thread that may use the wire afterwards. Fails, rather than opening, where
	// h264_safe_to_use says the encoder would crash the process.
	bool open(const h264_settings& settings, std::string& error);
	void close();
	bool is_open() const;

	// Whether the encoder accepted region of interest offsets. Only meaningful once open.
	bool face_offset_active() const;

	// Encodes one luma frame at the next frame time; chroma is sent flat grey, which costs
	// H.264 next to nothing. `faces` are in frame coordinates. `bytes` is what crosses the
	// wire for this frame: every unit the decoder needs, including parameter sets the encoder
	// kept out of band, and no access unit delimiters or SEI messages, which it does not.
	bool encode(const image_u8& luma, const std::vector<rect_i>& faces, std::vector<uint8_t>& bytes,
		std::string& error);

	// Decodes the bytes `encode` produced. `picture` changes only when the decoder emits one.
	bool decode(const std::vector<uint8_t>& bytes, image_u8& picture, std::string& error);

private:
	struct impl;
	impl* state = nullptr;
};

// Whether Windows' H.264 encoder can be driven in this process without crashing it. GitHub's
// Windows Server 2025 runners, for one, register an encoder that is an older build than the rest
// of their Media Foundation, and it crashes on its own worker thread during the first frame,
// which no error code reports and no handler can recover from. The first call therefore runs a
// short round trip in a child process - this executable with /probe-h264 - and later calls
// reuse the answer. Ordinary failures, such as a missing transform, pass here and are reported
// by the call that meets them; only a probe that crashes or hangs makes this false.
bool h264_safe_to_use(std::string& reason);

// The child's side of that probe: one short round trip through both transforms. Returns the
// process exit code, zero unless the process is about to die anyway.
int run_h264_probe();
