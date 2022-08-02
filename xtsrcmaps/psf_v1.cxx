#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/psf3_base.hxx"

#include "experimental/mdspan"
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <vector>

using std::tuple;
using std::vector;
using std::experimental::extents;
using std::experimental::full_extent;
using std::experimental::mdspan;
using std::experimental::submdspan;


// inline bool
// is_co_psf_base(size_t const dz, size_t const ez, size_t const sz) noexcept
// {
//     // All 4
//     if (dz <= 4 && ez <= 4 && sz <= 4) { return true; }
//     return false;
// }

// A                   [Ns, Nd, Ne]
// C  (Src X costheta) [Ns, Nc]
// D  (Separations)    [Nd]
// E  (Energies)       [Ne]
// IP (IRF Params)     [Me, Mc, 6]
// IE (IRF energies)   [Me]
// IC (IRF costheta)   [Mc]
// inline void
// co_psf_base_loop(auto A, auto C, auto D, auto E, auto IP, auto IE, auto IC) noexcept
// {
//     for (size_t s = 0; s < A.extent(0); ++s)
//     {
//         for (size_t d = 0; d < A.extent(1); ++d)
//         {
//             for (size_t e = 0; e < A.extent(2); ++e)
//             {
//                 for (size_t c = 0; c < C.extent(1); ++c) { A(s, d, e) += }
//             }
//         }
//     }
// }

inline char
largest_dim(size_t const& Ns,
            size_t const& Nd,
            size_t const& Ne,
            size_t const& Nc) noexcept
{
    char          midx   = 0;
    size_t const* valptr = &Ns;
    /* clang-format off */
    if (*valptr < Nd) { midx = 1; valptr = &Nd; }
    if (*valptr < Ne) { midx = 2; valptr = &Ne; }
    if (*valptr < Nc) { midx = 3; valptr = &Nc; }
    /* clang-format on */
    return midx;
}

inline auto
split_IRF_Energies(auto const&  IP,
                   auto const&  IE,
                   float const& e1,
                   float const& e2) noexcept -> auto
{
    // Find lower bound in IE which contains e1.
    auto const it1 = std::lower_bound(std::begin(IE), std::end(IE), e1);
    // Find lower bound in IE which contains e2. Must be at least it1.
    auto const it2 = --std::upper_bound(it1, std::end(IE), e2);

    auto const z1  = std::distance(std::begin(IE), it1);
    auto const z2  = std::distance(std::begin(IE), it2);
    return tuple {
        submdspan(IP, (0, z1), full_extent, full_extent),
        submdspan(IP, (z2, IP.extent(0)), full_extent, full_extent),
        submdspan(IE, (0, z1)),
        submdspan(IE, (z2, IE.extent(0))),
    };
}

inline auto
split_IRF_Angles(auto const& IP,
                 auto const& IC,
                 auto const& C1,
                 auto const& C2) noexcept -> auto
{
    // Find largest element in C1. C1 is not expected to be sorted.
    auto const max1 = std::max_element(std::begin(C1), std::end(C1));
    // Find least upper bound in IC which contains all of C1.
    auto const it1  = std::lower_bound(std::begin(IC), std::end(IC), max1);

    // Find smallest element in C2. C2 is not expected to be sorted.
    auto const min2 = std::min_element(std::begin(C2), std::end(C2));
    // Find greatest lower bound in IC which contains all of C2.
    auto const it2  = --std::upper_bound(std::begin(IC), std::end(IC), min2);

    auto const z1   = std::distance(std::begin(IC), it1);
    auto const z2   = std::distance(std::begin(IC), it2);
    return tuple {
        submdspan(IP, full_extent, (0, z1), full_extent),
        submdspan(IP, full_extent, (z2, IP.extent(0)), full_extent),
        submdspan(IC, (0, z1)),
        submdspan(IC, (z2, IC.extent(0))),
    };
}


// A                 [Ns, Nd, Ne]
// C  (Src X costheta) [Ns, Nc]
// D  (Separations)    [Nd]
// E  (Energies)       [Ne]
// IP (IRF Params)     [Me, Mc, 6]
// IE (IRF energies)   [Me]
// IC (IRF costheta)   [Mc]
void
co_moffat(auto A, auto C, auto D, auto E, auto IP, auto IE, auto IC) noexcept
{
    // Invariant Assumptions
    // These extent pairs must be equal for the algoritm to be valid:
    //    A[0] == C[0], A[1] == D[0], A[2] == E[0]
    //    IP[0] == IE[0],  IP[1] == IC[0]
    // The elements of the following dimensions are sorted along the rows and are
    // therefore monotonically increasing.
    //    C[1] (costheta),  D[0], E[0], IE[0], IC[0]

    // check for base case
    if (is_co_psf_base(A.extent(0), A.extent(1), A.extent(2), C.extent(1)))
    {
        // Do base case computation
        return co_psf_base_loop(A, C, D, E, IP, IE, IC);
    }

    char const ld = largest_dim(A.extent(0), A.extent(1), A.extent(2), C.extent(1));

    if (ld == 0) // Ns is the largest. Cut A[0] and C[0]
    {
        auto const& z    = A.extent(0);
        auto        Out1 = submdspan(A, (0, z / 2), full_extent, full_extent);
        auto        Out2 = submdspan(A, (z / 2, z), full_extent, full_extent);
        auto        B1   = submdspan(C, (0, z / 2), full_extent);
        auto        B2   = submdspan(C, (z / 2, z), full_extent);
        co_moffat(Out1, B1, D, E, IP, IE, IC);
        co_moffat(Out2, B2, D, E, IP, IE, IC);
    }
    else if (ld == 1) // Nd is the largest. Split A[1] and D[0]
    {
        auto const& z    = A.extent(1);
        auto        Out1 = submdspan(A, full_extent, (0, z / 2), full_extent);
        auto        Out2 = submdspan(A, full_extent, (z / 2, z), full_extent);
        auto        D1   = submdspan(D, (0, z / 2));
        auto        D2   = submdspan(D, (z / 2, z));
        co_moffat(Out1, C, D1, E, IP, IE, IC);
        co_moffat(Out2, C, D2, E, IP, IE, IC);
    }
    else if (ld == 2) // Ne is largest. Split A[2], E[0], and IP[0], IE[0]
    {
        auto const& z             = A.extent(2);
        auto        Out1          = submdspan(A, full_extent, full_extent, (0, z / 2));
        auto        Out2          = submdspan(A, full_extent, full_extent, (z / 2, z));
        auto        E1            = submdspan(E, (0, z / 2));
        auto        E2            = submdspan(E, (z / 2, z));
        // Search + split IP[0], IE
        auto [IP1, IP2, IE1, IE2] = split_IRF_Energies(IP, IE, E1[-1], E2[0]);
        co_moffat(Out1, C, D, E1, IP1, IE1, IC);
        co_moffat(Out2, C, D, E2, IP2, IE2, IC);
    }
    else // (ld == 3) -- costheta is largest. Split C[1] and IP[1], IC[0]
    {
        auto const& z             = C.extent(0);
        auto        B1            = submdspan(C, full_extent, (0, z / 2));
        auto        B2            = submdspan(C, full_extent, (z / 2, z));
        // Search + split IP[1], IC
        auto [IP1, IP2, IC1, IC2] = split_IRF_Angles(
            IP, IE, submdspan(B1, full_extent, -1), submdspan(B2, full_extent, 0));

        // Note, Must be sequential! Otherwise A will see a data race.
        co_moffat(A, B1, D, E, IP1, IE, IC1);
        co_moffat(A, B2, D, E, IP2, IE, IC2);
    }
}

// auto
// Fermi::psf_fixed_grid(vector<pair<double, double>> const& dirs,
//                       vector<double> const&               energies) ->
//                       vector<double>
// {
//
//     auto sep = separations(1e-4, 70.0, 400);
//
//     auto out = vector<double>(dirs.size() * energies.size() * sep.size(), 0.0);
//
//     auto A   = mdspan(out.data(), dirs.size(), energies.size(), sep.size());
//     auto D   = std::span(dirs.begin(), dirs.end());
//     auto E   = std::span(energies.begin(), energies.end());
//     auto S   = std::span(sep.begin(), sep.end());
//
//     fmt::print("D: {}\t", D.size());
//     fmt::print("E: {}\t", E.size());
//     fmt::print("S: {}\n\n", S.size());
//
//     co_psf(A, D, E, S);
//
//     // fmt::print("{}\n", fmt::join(out, ", "));
//     fmt::print("{}\n", out.back());
//
//     return out;
// }
