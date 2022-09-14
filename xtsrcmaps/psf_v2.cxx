#include "xtsrcmaps/psf.hxx"

#include "experimental/mdspan"
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <vector>

using std::pair;
using std::vector;
using std::experimental::full_extent;
using std::experimental::mdspan;
using std::experimental::submdspan;

//////
auto
Fermi::separations(double const xmin, double const xmax, size_t const N)
    -> std::vector<double>
{
    auto   sep   = std::vector<double>(N, 0.0);
    double xstep = std::log(xmax / xmin) / double(N - 1);
    for (size_t i = 0; i < N - 1; ++i) sep[i + 1] = xmin * std::exp(i * xstep);
    // fmt::print("sep: ({} ... {})\t size = {}\n",
    //            fmt::join(sep.begin(), sep.begin() + 4, " "),
    //            fmt::join(sep.end() - 5, sep.end() - 1, " "),
    //            sep.size());
    return sep;
}

inline auto
king_single(double const& sep,
            // mdspan<double, extents<size_t, 6>> const& pars
            auto const& pars) noexcept -> double
{
    assert(pars.extent(0) >= 6);
    double const& ncore = pars[0];
    double const& ntail = pars[1];
    double const& score = pars[2]; //* scaleFactor(energy) }; // Prescale the PSF "s"
    double const& stail = pars[3]; //* scaleFactor(energy) }; // pars by scaleFactor(IE)
    double const& gcore = pars[4]; // assured not to be 1.0
    double const& gtail = pars[5]; // assured not to be 1.0

    double rc           = sep / score;
    double uc           = rc * rc / 2.;

    double rt           = sep / stail;
    double ut           = rt * rt / 2.;

    // scaled king function
    return (ncore * (1. - 1. / gcore) * std::pow(1. + uc / gcore, -gcore)
            + ntail * ncore * (1. - 1. / gcore) * std::pow(1. + ut / gtail, -gtail));
    // should be able to compute x^-g as exp(-g*ln(x)) with SIMD log and exp.
}

// A               [Nd, Mc, Me]
// D (Separations) [Nd]
// P (IRF Params)  [Mc, Me, 6]
inline void
co_king_base(auto A, auto D, auto P) noexcept
{
    assert(P.extent(2) >= 6);

    for (size_t d = 0; d < A.extent(0); ++d)
    {
        for (size_t c = 0; c < A.extent(1); ++c)
        {
            for (size_t e = 0; e < A.extent(2); ++e)
            {
                A(d, c, e) = king_single(D[d], submdspan(P, c, e, full_extent));
            }
        }
    }
}

inline bool
is_co_psf_base(size_t const& Nd, size_t const& Mc, size_t const& Me) noexcept
{
    return (Nd <= 4 && Mc <= 4 && Me <= 4);
}

// A               [Nd, Mc, Me]
// D (Separations) [Nd]
// P (IRF Params)  [Mc, Me, 6]
void
co_king(auto A, auto D, auto P) noexcept
{
    auto largest_dim
        = [](size_t const& Nd, size_t const& Mc, size_t const& Me) noexcept -> char {
        char          midx   = 0;
        size_t const* valptr = &Nd;
        /* clang-format off */
        if (*valptr < Mc) { midx = 1; valptr = &Mc; }
        if (*valptr < Me) { midx = 2; valptr = &Me; }
        /* clang-format on */
        return midx;
    };

    // check for base case
    if (is_co_psf_base(A.extent(0), A.extent(1), A.extent(2)))
    {
        // Do base case computation
        return co_king_base(A, D, P);
    }

    char const ld = largest_dim(A.extent(0), A.extent(1), A.extent(2));

    if (ld == 0) // Nd is the largest. Cut A[0] and D[0]
    {
        auto const& z  = A.extent(0);
        auto        A1 = submdspan(A, pair(0, z / 2), full_extent, full_extent);
        auto        A2 = submdspan(A, pair(z / 2, z), full_extent, full_extent);
        auto        D1 = submdspan(D, pair(0, z / 2));
        auto        D2 = submdspan(D, pair(z / 2, z));
        co_king(A1, D1, P);
        co_king(A2, D2, P);
    }
    else if (ld == 1) // Mc is largest. Split A[1], P[0]
    {
        auto const& z  = A.extent(1);
        auto        A1 = submdspan(A, full_extent, pair(0, z / 2), full_extent);
        auto        A2 = submdspan(A, full_extent, pair(z / 2, z), full_extent);
        auto        P1 = submdspan(P, pair(0, z / 2), full_extent, full_extent);
        auto        P2 = submdspan(P, pair(z / 2, z), full_extent, full_extent);
        co_king(A1, D, P1);
        co_king(A2, D, P2);
    }
    else // (ld == 2) -- Me is largest. Split A[2] and P[1]
    {
        auto const& z  = A.extent(2);
        auto        A1 = submdspan(A, full_extent, full_extent, pair(0, z / 2));
        auto        A2 = submdspan(A, full_extent, full_extent, pair(z / 2, z));
        auto        P1 = submdspan(P, full_extent, pair(0, z / 2), full_extent);
        auto        P2 = submdspan(P, full_extent, pair(z / 2, z), full_extent);
        co_king(A1, D, P1);
        co_king(A2, D, P2);
    }
}

auto
Fermi::psf_fixed_grid(vector<double> const& deltas, IrfData3 const& pars)
    -> vector<double>
{

    auto kings = vector<double>(
        deltas.size() * pars.params.extent(0) * pars.params.extent(1), 0.0);
    assert(pars.params.extent(0) == pars.cosths.size());
    assert(pars.params.extent(1) == pars.logEs.size());
    auto A = mdspan(
        kings.data(), deltas.size(), pars.params.extent(0), pars.params.extent(1));
    auto D = mdspan(deltas.data(), deltas.size());
    auto P = mdspan(pars.params.data(),
                    pars.params.extent(0),
                    pars.params.extent(1),
                    pars.params.extent(2));

    co_king(A, D, P);

    return kings;
}
