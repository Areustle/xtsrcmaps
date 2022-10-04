#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace Fermi
{

auto
lerp_pars(std::span<double const, std::dynamic_extent> const,
          double const,
          std::optional<double> const = std::nullopt)
    -> std::tuple<double, double, size_t>;

auto
lerp_pars(std::vector<double> const&, double const)
    -> std::tuple<double, double, size_t>;

auto
lerp_pars(std::vector<double> const&,
          std::vector<double> const&,
          std::optional<double> const = std::nullopt)
    -> std::vector<std::tuple<double, double, size_t>>;


inline auto
bilerp(std::tuple<double, double, size_t> const& ct,
       std::tuple<double, double, size_t> const& et,
       auto const&                               IP) -> double
{
    auto const& [c_weight, c_complement, c_index] = ct;
    auto const& [e_weight, e_complement, e_index] = et;

    return c_complement * e_complement * IP(c_index - 1, e_index - 1) //
           + c_weight * e_complement * IP(c_index, e_index - 1)       //
           + c_complement * e_weight * IP(c_index - 1, e_index)       //
           + c_weight * e_weight * IP(c_index, e_index);              //
}

auto
greatest_lower(std::span<double const, std::dynamic_extent> const sp, double const v)
    -> size_t;

} // namespace Fermi
