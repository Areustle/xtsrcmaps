#include "xtsrcmaps/bilerp.hxx"

#include <algorithm>
#include <cassert>
#include <fmt/format.h>

auto
Fermi::lerp(std::span<double const, std::dynamic_extent> const sp, double const v)
    -> std::pair<double, size_t>
{
    auto upper = std::upper_bound(std::cbegin(sp), std::cend(sp), v);
    auto idx   = std::distance(std::cbegin(sp), upper);

    // if (upper == std::end(sp)) { fmt::print("({}) : {}\n", fmt::join(sp, " "), v); }
    assert(upper != std::begin(sp));
    assert(upper != std::end(sp));

    auto tt = (v - *(upper - 1)) / (*upper - *(upper - 1));

    return { tt, idx };
}

auto
Fermi::greatest_lower(std::span<double const, std::dynamic_extent> const sp,
                      double const                                       v) -> size_t
{
    auto upper = std::upper_bound(std::cbegin(sp), std::cend(sp), v);
    auto idx   = std::distance(std::cbegin(sp), upper);
    return idx;
}
