// The low-ban entropy coder: an adaptive binary range coder in the style of LZMA's, with
// Exp-Golomb binarisation for integers. Every syntax element of a packet is coded as binary
// decisions, each against a model of how often that decision has gone which way so far, so a
// decision that almost always goes one way costs a small fraction of a bit.
//
// Models live with the stream that owns them and adapt across packets; both ends make every
// decision against the same model in the same order, so they stay identical. The coder
// finishes with four bytes and the decoder has to consume exactly what was written, so a
// truncated or padded packet is detected.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// The probability that the next decision is 0. Two estimates adapt at different speeds and
// the coder uses their mean: the fast one follows a change within a few decisions, the slow
// one settles on a precise figure for a decision whose statistics hold still.
struct bit_model
{
	uint16_t fast = 32768;
	uint16_t slow = 32768;

	// In 1/4096ths, never 0 or 4096.
	uint32_t zero() const
	{
		const uint32_t p = (static_cast<uint32_t>(fast) + slow) >> 5;
		return p < 1 ? 1 : p > 4095 ? 4095 : p;
	}

	void update(const int bit)
	{
		if (bit == 0)
		{
			fast = static_cast<uint16_t>(fast + ((65536u - fast) >> 4));
			slow = static_cast<uint16_t>(slow + ((65536u - slow) >> 7));
		}
		else
		{
			fast = static_cast<uint16_t>(fast - (fast >> 4));
			slow = static_cast<uint16_t>(slow - (slow >> 7));
		}
	}
};

// The cost in bits of a decision whose probability is p/4096, for every p a model produces.
// Built before main, so the encoder's rate estimates read it without a guard.
struct decision_cost_table
{
	float bits[4097] = {};
	decision_cost_table();
};

extern const decision_cost_table decision_costs;

// What coding `bit` against `model` would cost, in bits, without coding it.
inline double bit_cost(const bit_model& model, const int bit)
{
	const uint32_t zero = model.zero();
	return decision_costs.bits[bit == 0 ? zero : 4096 - zero];
}

class range_encoder
{
public:
	void encode(bit_model& model, int bit);
	// The low `bits` bits of `value`, most significant first, each at even odds.
	void encode_bypass(uint32_t value, int bits);

	// The packet's length if it were finished now; exact, not an estimate.
	size_t size_if_finished() const { return shifts + 4; }

	// Flushes and returns everything written. The encoder is spent afterwards.
	std::vector<uint8_t> finish();

private:
	void shift_low();

	uint64_t low = 0;
	uint32_t range = 0xffffffffu;
	uint8_t cache = 0;
	size_t pending = 1;
	size_t shifts = 0;
	bool leading = true; // LZMA's first byte is always zero, so it is never written
	std::vector<uint8_t> out;
};

class range_decoder
{
public:
	range_decoder(const uint8_t* data, size_t size);

	int decode(bit_model& model);
	uint32_t decode_bypass(int bits);

	// Marks the input as malformed: a value out of range, a count that cannot be right.
	void fail() { failed = true; }
	// Nothing read past the end and nothing marked malformed so far.
	bool ok() const { return !failed && !overrun; }
	// ok, and every byte consumed: what a whole, well formed packet leaves behind.
	bool finished() const { return ok() && offset == size; }

private:
	uint8_t next_byte();

	const uint8_t* data = nullptr;
	size_t size = 0;
	size_t offset = 0;
	uint32_t range = 0xffffffffu;
	uint32_t code = 0;
	bool overrun = false;
	bool failed = false;
};

// Exp-Golomb: the bit length of value + 1 in unary, each unary decision against its own model
// (the last one shared by every longer length), then the bits below the leading one at even
// odds. Small values cost a decision or two; any 24-bit value still fits.
constexpr int golomb_max_bits = 24;

void encode_golomb(range_encoder& out, bit_model* prefix, int prefix_models, uint32_t value);
uint32_t decode_golomb(range_decoder& in, bit_model* prefix, int prefix_models);
double golomb_cost(const bit_model* prefix, int prefix_models, uint32_t value);

// A signed value: whether it is zero, then its sign at even odds and its magnitude less one.
struct signed_models
{
	bit_model zero;
	bit_model magnitude[6];
};

void encode_signed(range_encoder& out, signed_models& models, int value);
int decode_signed(range_decoder& in, signed_models& models);
double signed_cost(const signed_models& models, int value);
