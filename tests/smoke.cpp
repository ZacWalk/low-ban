#include <string_view>

int main()
{
    constexpr std::string_view name = "low-ban";
    return name.empty() ? 1 : 0;
}
