#include "xtsrcmaps/psf.hxx"

#include "experimental/mdspan"
#include <fmt/format.h>

#include <cmath>
#include <span>
#include <tuple>
#include <vector>

using std::pair;
// using std::tuple;
using std::vector;
using std::experimental::full_extent;
using std::experimental::mdspan;
using std::experimental::submdspan;

using dir_t = pair<double, double>;

auto
separations(double xmin, double xmax, size_t N) -> std::vector<double>
{
    auto   sep   = std::vector<double>(N + 1, 0.0);
    double xstep = std::log(xmax / xmin) / double(N - 1);
    for (size_t i = 0; i < N; ++i) sep[i + 1] = xmin * std::exp(i * xstep);
    return sep;
}

inline bool
is_co_psf_base(size_t const dz, size_t const ez, size_t const sz) noexcept
{
    // SIMD Base case
    if (dz <= 4 && ez <= 4 && sz <= 4) { return true; }
    return false;
}

inline void
co_psf_base_loop(auto A, auto D, auto E, auto S) noexcept
{
    for (size_t di = 0; di < A.extent(0); ++di)
    {
        for (size_t ei = 0; ei < A.extent(1); ++ei)
        {
            for (size_t si = 0; si < A.extent(2); ++si)
            {
                A(di, ei, si)
                    += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[si];
            }
        }
    }
}

void
co_psf_base_i1(auto A, auto D, auto E, auto S) noexcept
{
    for (size_t di = 0; di < A.extent(0); ++di)
    {
        for (size_t ei = 0; ei < A.extent(1); ++ei)
        {
            A(di, ei, 0) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[0];
        }
    }
}

void
co_psf_base_i2(auto A, auto D, auto E, auto S) noexcept
{
    for (size_t di = 0; di < A.extent(0); ++di)
    {
        for (size_t ei = 0; ei < A.extent(1); ++ei)
        {
            A(di, ei, 0) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[0];
            A(di, ei, 1) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[1];
        }
    }
}

void
co_psf_base_i3(auto A, auto D, auto E, auto S) noexcept
{
    for (size_t di = 0; di < A.extent(0); ++di)
    {
        for (size_t ei = 0; ei < A.extent(1); ++ei)
        {
            A(di, ei, 0) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[0];
            A(di, ei, 1) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[1];
            A(di, ei, 2) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[2];
        }
    }
}

void
co_psf_base_i4(auto A, auto D, auto E, auto S) noexcept
{
    for (size_t di = 0; di < A.extent(0); ++di)
    {
        for (size_t ei = 0; ei < A.extent(1); ++ei)
        {
            A(di, ei, 0) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[0];
            A(di, ei, 1) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[1];
            A(di, ei, 2) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[2];
            A(di, ei, 3) += std::get<0>(D[di]) + std::get<1>(D[di]) + E[ei] + S[3];
        }
    }
}

inline char
largest_dim(size_t const dz, size_t const ez, size_t const sz) noexcept
{
    if (dz >= ez)
    {
        if (dz >= sz) { return 0; } // dz > {ez, sz}
        return 2;                   // sz > dz > ez
    }
    // ez > dz
    if (ez >= sz) { return 1; } // ez > {dz, sz}
    return 2;                   // sz > ez > dz
}

template <typename T>
inline auto
split_in(std::span<T> const& in) -> std::pair<std::span<T> const, std::span<T> const>
{
    size_t const z1 = in.size() / 2;
    size_t const z2 = in.size() - z1;
    return { in.subspan(0, z1), in.subspan(z1, z2) };
}

inline auto
split_out_0(auto const& A) -> auto
{
    auto const z = (A.extent(0) / 2);
    return pair { submdspan(A, pair(0, z), full_extent, full_extent),
                  submdspan(A, pair(z, A.extent(0)), full_extent, full_extent) };
}

inline auto
split_out_1(auto const& A) -> auto
{
    auto const z = (A.extent(1) / 2);
    return pair { submdspan(A, full_extent, pair(0, z), full_extent),
                  submdspan(A, full_extent, pair(z, A.extent(1)), full_extent) };
}

inline auto
split_out_2(auto const& A) -> auto
{
    auto const z = (A.extent(2) / 2);
    return pair { submdspan(A, full_extent, full_extent, pair(0, z)),
                  submdspan(A, full_extent, full_extent, pair(z, A.extent(2))) };
}

void
co_psf(auto A, auto D, auto E, auto S) noexcept
{
    // check for base case
    if (is_co_psf_base(D.size(), E.size(), S.size()))
    {
        // Do base case computation
        // co_psf_base_loop(A, D, E, S);
        if (S.size() == 4) { return co_psf_base_i4(A, D, E, S); }
        if (S.size() == 3) { return co_psf_base_i3(A, D, E, S); }
        if (S.size() == 2) { return co_psf_base_i2(A, D, E, S); }
        if (S.size() == 1) { return co_psf_base_i1(A, D, E, S); }
        return;
    }

    char const ld = largest_dim(D.size(), E.size(), S.size());

    if (ld == 0)
    {
        auto const [D1, D2] = split_in(D);
        auto const [A1, A2] = split_out_0(A);
        co_psf(A1, D1, E, S);
        co_psf(A2, D2, E, S);
    }
    else if (ld == 1)
    {
        auto const [E1, E2] = split_in(E);
        auto const [A1, A2] = split_out_1(A);
        co_psf(A1, D, E1, S);
        co_psf(A2, D, E2, S);
    }
    else /* (ld == 2) */
    {
        auto const [S1, S2] = split_in(S);
        auto const [A1, A2] = split_out_2(A);
        co_psf(A1, D, E, S1);
        co_psf(A2, D, E, S2);
    }
}

auto
Fermi::psf_fixed_grid(vector<pair<double, double>> const& dirs,
                      vector<double> const&               energies) -> vector<double>
{

    auto sep   = separations(1e-4, 70.0, 400);
    auto logEs = vector<double>(energies.size());
    for (size_t i = 0; i < logEs.size(); ++i) { logEs[i] = std::log(energies[i]); }

    auto out = vector<double>(dirs.size() * logEs.size() * sep.size(), 0.0);

    auto A   = mdspan(out.data(), dirs.size(), logEs.size(), sep.size());
    auto D   = std::span(dirs.cbegin(), dirs.cend());
    auto E   = std::span(logEs.cbegin(), logEs.cend());
    auto S   = std::span(sep.cbegin(), sep.cend());

    fmt::print("D: {}\t", D.size());
    fmt::print("E: {}\t", E.size());
    fmt::print("S: {}\n\n", S.size());

    co_psf(A, D, E, S);

    // fmt::print("{}\n", fmt::join(out, ", "));
    fmt::print("{}\n", out.back());

    return out;
}
