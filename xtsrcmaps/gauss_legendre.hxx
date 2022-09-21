#pragma once

#include <cmath>
#include <tuple>
#include <vector>


namespace
{
// Estimate the polynomial's root and derivative
template <size_t N>
auto
poly_root_deriv(double const x) -> std::pair<double, double>
{
    double       val      = x;
    double       deriv    = 0.0;
    double       prev_val = 1;
    const double delta    = 1 / (x * x - 1);
    for (size_t j = 2; j <= N; j++)
    {
        const double next_val = ((2 * j - 1) * x * val - (j - 1) * prev_val) / j;
        deriv                 = j * delta * (x * next_val - val);
        prev_val              = val;
        val                   = next_val;
    }

    return { val, deriv };
}


} // namespace

namespace Fermi
{

template <size_t N = 8>
auto
legendre_poly_rw(double const eps = 1e-15) -> std::vector<std::pair<double, double>>
{
    auto result = std::vector<std::pair<double, double>>();
    result.reserve(N);
    for (size_t i = 1; i <= N; ++i)
    {
        double abcissa      = std::cos(M_PI * (i - 0.25) / (N + 0.5));
        auto [value, deriv] = ::poly_root_deriv<N>(abcissa);

        // Use Newtons method to improve polynomial root estimate.
        double ratio        = value / deriv;

        while (fabs(ratio) > eps)
        {
            abcissa -= ratio;
            std::tie(value, deriv) = ::poly_root_deriv<N>(abcissa);
            ratio                  = value / deriv;
        }
        abcissa -= ratio;

        result.emplace_back(abcissa,
                            2.0 / ((1 - (abcissa * abcissa)) * (deriv * deriv)));
    }
    return result;
}

template <typename F>
auto
gauss_legendre_integral(
    double const                                  a,
    double const                                  b,
    std::vector<std::pair<double, double>> const& legendre_poly_rts_wgts,
    F&&                                           f //
    ) -> double
{
    const double width         = 0.5 * (b - a);
    const double mean          = 0.5 * (a + b);
    double       gaussLegendre = 0.0;

    for (auto const& [root, weight] : legendre_poly_rts_wgts)
    {
        gaussLegendre += weight * f(width * root + mean);
    }

    return gaussLegendre * width;
}

} // namespace Fermi
