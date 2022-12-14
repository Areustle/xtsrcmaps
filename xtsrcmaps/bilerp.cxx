#include "xtsrcmaps/bilerp.hxx"

// #include "xtsrcmaps/psf.hxx"

#include <algorithm>
#include <cassert>
#include <fmt/format.h>

auto
Fermi::lerp_pars(std::span<double const, std::dynamic_extent> const sp,
                 double const                                       v,
                 std::optional<double> const zero_lower_bound_value)
    -> std::tuple<double, double, size_t>
{

    if (zero_lower_bound_value && v < zero_lower_bound_value.value())
    {
        return { 0.0, 0.0, 1 };
    }

    auto upper = std::upper_bound(std::cbegin(sp), std::cend(sp), v);
    auto idx   = std::distance(std::cbegin(sp), upper);

    assert(upper != std::begin(sp));
    assert(upper != std::end(sp));

    // Weight
    auto tt = (v - *(upper - 1)) / (*upper - *(upper - 1));

    // Weight, Weight_Complement, Upper Bound Index
    return { tt, 1 - tt, idx };
}

auto
Fermi::lerp_pars(std::vector<double> const& rng, double const v)
    -> std::tuple<double, double, size_t>
{
    return lerp_pars(std::span(rng), v, rng[1]);
}

auto
Fermi::lerp_pars(std::vector<double> const&  rng,
                 std::vector<double> const&  vals,
                 std::optional<double> const zero_lower_bound)
    -> std::vector<std::tuple<double, double, size_t>>
{
    auto sp    = std::span(rng);
    auto lerps = std::vector<std::tuple<double, double, size_t>>(vals.size());
    std::transform(vals.cbegin(),
                   vals.cend(),
                   lerps.begin(), //
                   [&](auto const& v) -> std::tuple<double, double, size_t> {
                       return lerp_pars(sp, v, zero_lower_bound);
                   });
    return lerps;
}

auto
Fermi::lerp_pars(Tensor1d const&             rng,
                 std::vector<double> const&  vals,
                 std::optional<double> const zero_lower_bound)
    -> std::vector<std::tuple<double, double, size_t>>
{
    auto sp    = std::span(rng.data(), rng.size());
    auto lerps = std::vector<std::tuple<double, double, size_t>>(vals.size());
    std::transform(vals.cbegin(),
                   vals.cend(),
                   lerps.begin(), //
                   [&](auto const& v) -> std::tuple<double, double, size_t> {
                       return lerp_pars(sp, v, zero_lower_bound);
                   });
    return lerps;
}

// auto
// Fermi::separations_lerp_pars(double const ds, std::vector<double> const& sep)
//     -> std::tuple<double, double, size_t>
// {
//     double      vd     = PSF::inverse_separation(ds);
//     size_t      i      = size_t(vd);
//     auto const& sep_lo = sep[i];
//     auto const& sep_hi = sep[i + 1];
//     auto const  tt     = (vd - sep_lo) / (sep_hi - sep_lo);
//     return { tt, 1 - tt, i };
// }


auto
Fermi::greatest_lower(std::span<double const, std::dynamic_extent> const sp,
                      double const                                       v) -> size_t
{
    auto upper = std::upper_bound(std::cbegin(sp), std::cend(sp), v);
    auto idx   = std::distance(std::cbegin(sp), upper);
    return idx;
}
