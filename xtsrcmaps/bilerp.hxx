#pragma once

#include <span>
#include <utility>

namespace Fermi
{

auto
lerp(std::span<double const, std::dynamic_extent> const sp, double const v)
    -> std::pair<double, size_t>;

auto
bilerp(
    double const tt, double const uu, size_t const cx, size_t const ex, auto const IP)
    -> double
{
    return (1. - tt) * (1. - uu) * IP(cx - 1, ex - 1) //
           + (tt) * (1. - uu) * IP(cx, ex - 1)        //
           + (1. - tt) * (uu)*IP(cx - 1, ex)          //
           + (tt) * (uu)*IP(cx, ex);                  //
}

auto
greatest_lower(std::span<double const, std::dynamic_extent> const sp, double const v)
    -> size_t;

} // namespace Fermi
