#include "xtsrcmaps/psf.hxx"
// #include "xtsrcmaps/psf3_base.hxx"

#include "experimental/mdspan"
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

using std::pair;
using std::span;
using std::tuple;
using std::vector;
// using std::experimental::extents;
using std::experimental::full_extent;
using std::experimental::mdspan;
using std::experimental::submdspan;


inline bool
is_co_psf_base(size_t const& Ns, size_t const& Nd, size_t const& Ne) noexcept
{
    return (Ns <= 4 && Nd <= 4 && Ne <= 4);
}

inline auto
lerp(auto sp, double const v) -> pair<double, size_t>
{
    auto upper = std::upper_bound(std::begin(sp), std::end(sp), v);
    auto idx   = std::distance(std::begin(sp), upper);
    auto tt    = (v - *(upper - 1)) / (*upper - *(upper - 1));

    return { tt, idx };
}

inline auto
bilerp(double const tt,
       double const uu,
       size_t const d,
       size_t const ex,
       size_t const cx,
       auto         IP) -> double
{
    return (1. - tt) * (1. - uu) * IP(d, ex - 1, cx - 1)
           + (tt) * (1. - uu) * IP(d, ex, cx - 1) //
           + (1. - tt) * (uu)*IP(d, ex - 1, cx)   //
           + (tt) * (uu)*IP(d, ex, cx);
}


// B                   [Ns, Nd, Ne]
// C  (Src X costheta) [Ns, Nc]
// E  (Energies)       [Ne]
// IP (IRF Params)     [Nd, Me, Mc]
// IE (IRF energies)   [Me]
// IC (IRF costheta)   [Mc]
inline void
co_aeff_value_base(auto        R,
               auto const& C,
               auto const& E,
               auto const& IP,
               auto const& IE,
               auto const& IC) noexcept
{

    for (size_t s = 0; s < R.extent(0); ++s)
    {
        auto elerps = vector<pair<double, size_t>>(R.extent(2));
        for (size_t i = 0; i < R.extent(2); ++i) { elerps[i] = lerp(IE, E(i)); }
        auto clerps = vector<pair<double, size_t>>(C.extent(1));
        for (size_t i = 0; i < C.extent(1); ++i) { clerps[i] = lerp(IC, C(s, i)); }

        for (size_t d = 0; d < R.extent(1); ++d)
        {
            for (size_t e = 0; e < R.extent(2); ++e)
            {
                // auto [tt, eidx] = lerp(std::cbegin(IE), std::cend(IE), E(e));
                for (size_t c = 0; c < C.extent(1); ++c)
                {
                    // auto [uu, cidx] = lerp(std::cbegin(IC), std::cend(IC), C(s, c));
                    auto& tt   = std::get<0>(elerps[e]);
                    auto& eidx = std::get<1>(elerps[e]);
                    auto& uu   = std::get<0>(clerps[c]);
                    auto& cidx = std::get<1>(clerps[c]);

                    R(s, d, e) += C(s, c) * bilerp(tt, uu, d, eidx, cidx, IP);
                }
            }
        }
    }
}

inline char
largest_dim(size_t const& Ns,
            size_t const& Nd,
            size_t const& Ne,
            size_t const& Nc) noexcept
{
    // Would that std::max_element worked on variadic parameter packs.
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
    // Find lower bound in IE which contains clamped e1.
    auto it1      = std::upper_bound(std::cbegin(IE), std::cend(IE), e1);
    // Find lower bound in IE which contains clamped e2. Must be at least it1.
    auto it2      = --std::upper_bound(it1, std::cend(IE), e2);

    auto const z1 = std::distance(std::cbegin(IE), it1);
    auto const z2 = std::distance(std::cbegin(IE), it2);
    return tuple {
        submdspan(IP, pair(0, z1), full_extent, full_extent),
        submdspan(IP, pair(z2, IP.extent(0)), full_extent, full_extent),
        IE.subspan(0, z1),             // submdspan(IE, pair(0, z1)),
        IE.subspan(z2, IE.size() - z2) // submdspan(IE, pair(z2, IE.extent(0))),
    };
}

inline auto
split_IRF_Angles(auto const& IP,
                 auto const& IC,
                 auto const& C1,
                 auto const& C2) noexcept -> auto
{
    // Find largest element in C1. C1 is not expected to be sorted.
    // auto max1     = std::max_element(std::cbegin(C1), std::cend(C1));
    auto max1 = C1[0];
    for (size_t i = 1; i < C1.extent(0); ++i) { max1 = max1 > C1[i] ? max1 : C1[i]; }
    // Find upper bound in IC which contains all of C1.
    auto it1  = std::upper_bound(std::cbegin(IC), std::cend(IC), max1);

    // Find smallest element in C2. C2 is not expected to be sorted.
    // auto min2     = std::min_element(std::cbegin(C2), std::cend(C2));
    auto min2 = C2[0];
    for (size_t i = 1; i < C2.extent(0); ++i) { max1 = max1 > C2[i] ? max1 : C2[i]; }
    // Find greatest lower bound in IC which contains all of C2.
    auto it2      = --std::upper_bound(std::cbegin(IC), std::cend(IC), min2);

    auto const z1 = std::distance(std::cbegin(IC), it1);
    auto const z2 = std::distance(std::cbegin(IC), it2);
    return tuple {
        submdspan(IP, full_extent, pair(0, z1), full_extent),
        submdspan(IP, full_extent, pair(z2, IP.extent(0)), full_extent),
        IC.subspan(0, z1),             // submdspan(IC, pair(0, z1)),
        IC.subspan(z2, IC.size() - z2) // submdspan(IC, pair(z2, IC.extent(0))),
    };
}

// B                   [Ns, Nd, Ne]
// C  (Src X costheta) [Ns, Nc]
// E  (Energies)       [Ne]
// IP (IRF Params)     [Nd, Me, Mc]
// IE (IRF energies)   [Me]
// IC (IRF costheta)   [Mc]
void
co_bilerp(auto R, auto C, auto E, auto IP, auto IE, auto IC) noexcept
{
    /*
     * Invariant Assumptions
     * These extent pairs must be equal for the algoritm to be valid:
     *    B[0] == C[0], B[1] == IP[0], B[2] == E[0]
     *    IP[1] == IE[0],  IP[2] == IC[0]
     * The elements of the following dimensions are sorted along the rows and are
     * therefore monotonically increasing.
     *    C[1] (costheta),  E[0], IE[0], IC[0]
     */

    // check for base case
    if (is_co_psf_base(R.extent(0), R.extent(1), R.extent(2)))
    {
        // Do base case computation
        return co_aeff_value_base(R, C, E, IP, IE, IC);
    }

    char const ld = largest_dim(R.extent(0), R.extent(1), R.extent(2), C.extent(1));

    if (ld == 0) // Ns is the largest. Cut B[0] and C[0]
    {
        auto const& z    = R.extent(0);
        auto        Out1 = submdspan(R, pair(0, z / 2), full_extent, full_extent);
        auto        Out2 = submdspan(R, pair(z / 2, z), full_extent, full_extent);
        auto        C1   = submdspan(C, pair(0, z / 2), full_extent);
        auto        C2   = submdspan(C, pair(z / 2, z), full_extent);
        co_bilerp(Out1, C1, E, IP, IE, IC);
        co_bilerp(Out2, C2, E, IP, IE, IC);
        return;
    }
    else if (ld == 1) // Nd is the largest. Split B[1] and IP[0]
    {
        auto const& z    = R.extent(1);
        auto        Out1 = submdspan(R, full_extent, pair(0, z / 2), full_extent);
        auto        Out2 = submdspan(R, full_extent, pair(z / 2, z), full_extent);
        auto        IP1  = submdspan(IP, pair(0, z / 2), full_extent, full_extent);
        auto        IP2  = submdspan(IP, pair(z / 2, z), full_extent, full_extent);
        co_bilerp(Out1, C, E, IP1, IE, IC);
        co_bilerp(Out2, C, E, IP2, IE, IC);
        return;
    }
    else if (ld == 2) // Ne is largest. Split B[2], E[0], and IP[1], IE[0]
    {
        auto const& z    = R.extent(2);
        auto        Out1 = submdspan(R, full_extent, full_extent, pair(0, z / 2));
        auto        Out2 = submdspan(R, full_extent, full_extent, pair(z / 2, z));
        auto        E1   = submdspan(E, pair(0, z / 2));
        auto        E2   = submdspan(E, pair(z / 2, z));
        // Search + split IP[0], IE
        auto [IP1, IP2, IE1, IE2] = split_IRF_Energies(IP, IE, E1[-1], E2[0]);
        co_bilerp(Out1, C, E1, IP1, IE1, IC);
        co_bilerp(Out2, C, E2, IP2, IE2, IC);
        return;
    }
    else // (ld == 3) -- costheta is largest. Split C[1] and IP[2], IC[0]
    {
        auto const& z  = C.extent(0);
        auto        C1 = submdspan(C, full_extent, pair(0, z / 2));
        auto        C2 = submdspan(C, full_extent, pair(z / 2, z));
        // Search + split IP[1], IC
        auto [IP1, IP2, IC1, IC2]
            = split_IRF_Angles(IP,
                               IC,
                               submdspan(C1, full_extent, C1.extent(1) - 1),
                               submdspan(C2, full_extent, 0));
        // Note, Must be sequential! Otherwise B will see a data race.
        co_bilerp(R, C1, E, IP1, IE, IC1);
        co_bilerp(R, C2, E, IP2, IE, IC2);
        return;
    }
}

auto
Fermi::bilerp(vector<double> const& kings,
              vector<double> const& logEs,
              vector<double> const& cosBins,
              IrfData3 const&        pars) -> vector<double>
{

    size_t N_src   = cosBins.size() / 40;
    auto   bilerps = vector<double>(N_src * 400 * logEs.size(), 0.0);
    auto   R       = mdspan(bilerps.data(), N_src, 400, logEs.size());
    auto   C       = mdspan(cosBins.data(), N_src, 40);
    auto   E       = mdspan(logEs.data(), logEs.size());
    auto   IP      = mdspan(kings.data(), 400, 10, 25);
    auto   IE      = span(pars.logEs.cbegin(), pars.logEs.cend());
    auto   IC      = span(pars.cosths.cbegin(), pars.cosths.cend());

    co_aeff_value_base(R, C, E, IP, IE, IC);

    return bilerps;
}

// auto
// Fermi::bilerp(vector<double> const& kings,
//               vector<double> const& logEs,
//               vector<double> const& cosBins,
//               PsfData const&        pars) -> vector<double>
// {
//
//     size_t N_src   = cosBins.size() / 40;
//     auto   bilerps = vector<double>(N_src * 400 * logEs.size(), 0.0);
//     auto   B       = mdspan(bilerps.data(), N_src, 400, logEs.size());
//     auto   C       = mdspan(cosBins.data(), N_src, 40);
//     auto   E       = mdspan(logEs.data(), logEs.size());
//     auto   IP      = mdspan(kings.data(), 400, 10, 25);
//     auto   IE      = span(pars.logEs.cbegin(), pars.logEs.cend());
//     auto   IC      = span(pars.cosths.cbegin(), pars.cosths.cend());
//
//     co_bilerp_base(B, C, E, IP, IE, IC);
//
//     return bilerps;
// }
