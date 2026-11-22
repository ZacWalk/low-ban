#include "entropy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace
{
	constexpr uint32_t top = 1u << 24;

	int floor_log2(uint32_t value)
	{
		int bits = 0;
		while (value > 1)
		{
			value >>= 1;
			++bits;
		}
		return bits;
	}
}

decision_cost_table::decision_cost_table()
{
	bits[0] = 24.0f;
	for (int p = 1; p <= 4096; ++p) bits[p] = static_cast<float>(-std::log2(p / 4096.0));
}

const decision_cost_table decision_costs;

void range_encoder::shift_low()
{
	++shifts;
	if (static_cast<uint32_t>(low) < 0xff000000u || (low >> 32) != 0)
	{
		const auto carry = static_cast<uint8_t>(low >> 32);
		uint8_t held = cache;
		do
		{
			if (leading) leading = false;
			else out.push_back(static_cast<uint8_t>(held + carry));
			held = 0xff;
		} while (--pending != 0);
		cache = static_cast<uint8_t>(low >> 24);
	}
	++pending;
	low = (low & 0x00ffffffu) << 8;
}

void range_encoder::encode(bit_model& model, const int bit)
{
	const uint32_t bound = (range >> 12) * model.zero();
	if (bit == 0) range = bound;
	else
	{
		low += bound;
		range -= bound;
	}
	model.update(bit);
	while (range < top)
	{
		range <<= 8;
		shift_low();
	}
}

void range_encoder::encode_bypass(const uint32_t value, const int bits)
{
	for (int i = bits - 1; i >= 0; --i)
	{
		range >>= 1;
		if (((value >> i) & 1) != 0) low += range;
		while (range < top)
		{
			range <<= 8;
			shift_low();
		}
	}
}

std::vector<uint8_t> range_encoder::finish()
{
	for (int i = 0; i < 5; ++i) shift_low();
	return std::move(out);
}

range_decoder::range_decoder(const uint8_t* const data_in, const size_t size_in) : data(data_in), size(size_in)
{
	for (int i = 0; i < 4; ++i) code = code << 8 | next_byte();
}

uint8_t range_decoder::next_byte()
{
	if (offset >= size)
	{
		overrun = true;
		return 0;
	}
	return data[offset++];
}

int range_decoder::decode(bit_model& model)
{
	const uint32_t bound = (range >> 12) * model.zero();
	int bit = 0;
	if (code < bound) range = bound;
	else
	{
		code -= bound;
		range -= bound;
		bit = 1;
	}
	model.update(bit);
	while (range < top)
	{
		range <<= 8;
		code = code << 8 | next_byte();
	}
	return bit;
}

uint32_t range_decoder::decode_bypass(const int bits)
{
	uint32_t value = 0;
	for (int i = 0; i < bits; ++i)
	{
		range >>= 1;
		uint32_t bit = 0;
		if (code >= range)
		{
			code -= range;
			bit = 1;
		}
		value = value << 1 | bit;
		while (range < top)
		{
			range <<= 8;
			code = code << 8 | next_byte();
		}
	}
	return value;
}

void encode_golomb(range_encoder& out, bit_model* const prefix, const int prefix_models, const uint32_t value)
{
	const uint32_t shifted = value + 1;
	const int length = floor_log2(shifted);
	for (int i = 0; i < length; ++i) out.encode(prefix[std::min(i, prefix_models - 1)], 1);
	out.encode(prefix[std::min(length, prefix_models - 1)], 0);
	if (length > 0) out.encode_bypass(shifted, length);
}

uint32_t decode_golomb(range_decoder& in, bit_model* const prefix, const int prefix_models)
{
	int length = 0;
	while (in.decode(prefix[std::min(length, prefix_models - 1)]) == 1)
	{
		if (++length >= golomb_max_bits || !in.ok())
		{
			in.fail();
			return 0;
		}
	}
	const uint32_t below = length > 0 ? in.decode_bypass(length) : 0;
	return ((1u << length) | below) - 1;
}

double golomb_cost(const bit_model* const prefix, const int prefix_models, const uint32_t value)
{
	const int length = floor_log2(value + 1);
	double bits = length;
	for (int i = 0; i < length; ++i) bits += bit_cost(prefix[std::min(i, prefix_models - 1)], 1);
	return bits + bit_cost(prefix[std::min(length, prefix_models - 1)], 0);
}

void encode_signed(range_encoder& out, signed_models& models, const int value)
{
	out.encode(models.zero, value != 0 ? 1 : 0);
	if (value == 0) return;
	out.encode_bypass(value < 0 ? 1 : 0, 1);
	encode_golomb(out, models.magnitude, 6, static_cast<uint32_t>(std::abs(value) - 1));
}

int decode_signed(range_decoder& in, signed_models& models)
{
	if (in.decode(models.zero) == 0) return 0;
	const bool negative = in.decode_bypass(1) != 0;
	const int magnitude = static_cast<int>(decode_golomb(in, models.magnitude, 6)) + 1;
	return negative ? -magnitude : magnitude;
}

double signed_cost(const signed_models& models, const int value)
{
	if (value == 0) return bit_cost(models.zero, 0);
	return bit_cost(models.zero, 1) + 1.0 +
	       golomb_cost(models.magnitude, 6, static_cast<uint32_t>(std::abs(value) - 1));
}
