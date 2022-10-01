#include "xtsrcmaps/bilerp.hxx"

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
Fermi::greatest_lower(std::span<double const, std::dynamic_extent> const sp,
                      double const                                       v) -> size_t
{
    auto upper = std::upper_bound(std::cbegin(sp), std::cend(sp), v);
    auto idx   = std::distance(std::cbegin(sp), upper);
    return idx;
}

// (2      ,     0.5      ,       1.96875,2.03125,
// (2.09997,     0.0995546,       2.09375,2.15625,
// (2.19994,     0.699109 ,       2.15625,2.21875,
// (2.29992,     0.298663 ,       2.28125,2.34375,
// (2.39989,     0.898219 ,       2.34375,2.40625,
// (2.49986,     0.497773 ,       2.46875,2.53125,
// (2.59983,     0.0973276,       2.59375,2.65625,
// (2.69981,     0.696882 ,       2.65625,2.71875,
// (2.79978,     0.296437 ,       2.78125,2.84375,
// (2.89975,     0.895991 ,       2.84375,2.90625,
// (2.99972,     0.495546 ,       2.96875,3.03125,
// (3.09969,     0.0951007,       3.09375,3.15625,
// (3.19967,     0.694655 ,       3.15625,3.21875,
// (3.29964,     0.294209 ,       3.28125,3.34375,
// (3.39961,     0.893764 ,       3.34375,3.40625,
// (3.49958,     0.493319 ,       3.46875,3.53125,
// (3.59955,     0.0928739,       3.59375,3.65625,
// (3.69953,     0.692428 ,       3.65625,3.71875,
// (3.7995,,     0.291983 ,       3.78125,3.84375,
// (3.89947,     0.891537 ,       3.84375,3.90625,
// (3.99944,     0.491091 ,       3.96875,4.03125,
// (4.09942,     0.0906463,       4.09375,4.15625,
// (4.19939,     0.690201 ,       4.15625,4.21875,
// (4.29936,     0.859837 ,       4.21875,4.3125 ,
// (4.39933,     0.694655 ,        4.3125,4.4375 ,
// (4.4993 ,     0.494432 ,        4.4375,4.5625 ,
// (4.59928,     0.29421  ,        4.5625,4.6875 ,
// (4.69925,     0.0939871,        4.6875,4.8125 ,
// (4.79922,     0.893764 ,        4.6875,4.8125 ,
// (4.89919,     0.693542 ,        4.8125,4.9375 ,
// (4.99916,     0.493319 ,        4.9375,5.0625 ,
// (5.09914,     0.293096 ,        5.0625,5.1875 ,
// (5.19911,     0.0928735,        5.1875,5.3125 ,
// (5.29908,     0.892651 ,        5.1875,5.3125 ,
// (5.39905,     0.692428 ,        5.3125,5.4375 ,
// (5.49903,     0.492205 ,        5.4375,5.5625 ,
// (5.599  ,     0.291983 ,        5.5625,5.6875 ,
// (5.69897,     0.0917601,        5.6875,5.8125 ,
