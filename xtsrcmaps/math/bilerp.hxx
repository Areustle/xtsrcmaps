#pragma once

#include "xtsrcmaps/math/tensor_types.hxx"

#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace Fermi {

auto lerp_pars(std::span<double const, std::dynamic_extent> const,
               double const,
               std::optional<double> const = std::nullopt)
    -> std::tuple<double, double, size_t>;

auto lerp_pars(std::vector<double> const&, double const)
    -> std::tuple<double, double, size_t>;

auto lerp_pars(std::vector<double> const&,
               std::vector<double> const&,
               std::optional<double> const = std::nullopt)
    -> std::vector<std::tuple<double, double, size_t>>;

auto lerp_pars(Tensor1d const&,
               std::vector<double> const&,
               std::optional<double> const = std::nullopt)
    -> std::vector<std::tuple<double, double, size_t>>;

inline auto
bilerp(std::tuple<double, double, size_t> const& et,
       std::tuple<double, double, size_t> const& ct,
       Tensor2d const&                           IP) -> double {
    auto const& [c_weight, c_complement, c_index] = ct;
    auto const& [e_weight, e_complement, e_index] = et;

    return c_complement * e_complement * IP(e_index - 1, c_index - 1) //
           + c_complement * e_weight * IP(e_index, c_index - 1)       //
           + c_weight * e_complement * IP(e_index - 1, c_index)       //
           + c_weight * e_weight * IP(e_index, c_index);              //
}

} // namespace Fermi
