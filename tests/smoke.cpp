#include <string_view>
#include <cstdio>

int main()
{
	constexpr std::string_view name = "low-ban";
	std::puts("low-ban: build-only smoke; behavioural counts come from low_ban_selftest");
	return name.empty() ? 1 : 0;
}
