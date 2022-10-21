#pragma once

#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include "unsupported/Eigen/CXX11/Tensor"

namespace Fermi::ModelMap
{


template <size_t N>
auto constexpr integ_delta_steps() -> std::array<double, N> // Assumes Linear Binning
{
    auto delta = std::array<double, N>();
    for (size_t i = 0; i < N; ++i)
    {
        delta[i] = double(i) / double(N) - 0.5 + 1. / (2. * double(N));
    }
    return delta;
}

template <size_t N>
auto constexpr integ_delta_lo() -> std::array<double, N>
{

    auto constexpr delta = integ_delta_steps<N>();
    auto delta_lo        = std::array<double, N>();
    for (size_t i = 0; i < N / 2; ++i)
    {
        delta_lo[i * 2u + 0u] = 1.0 - delta[i];
        delta_lo[i * 2u + 1u] = delta[i];
    }

    return delta_lo;
}

template <size_t N>
auto constexpr integ_delta_hi() -> std::array<double, N>
{

    auto constexpr delta = integ_delta_steps<N>();
    auto delta_hi        = std::array<double, N>();
    for (size_t i = 0; i < N / 2; ++i)
    {
        delta_hi[i * 2u + 0u] = 1.0 - delta[i + (N / 2)];
        delta_hi[i * 2u + 1u] = delta[i + (N / 2)];
    }

    return delta_hi;
}

template <size_t N>
auto constexpr integ_delta() -> std::array<double, 2 * N>
{

    auto constexpr steps = integ_delta_steps<N>();
    auto delta           = std::array<double, 2 * N>();
    for (size_t i = 0; i < N; ++i)
    {
        delta[i * 2u + 0u] = 1.0 - steps[i];
        delta[i * 2u + 1u] = steps[i];
    }

    return delta;
}

template <size_t N>
auto constexpr integ_delta_0() -> std::array<double, 3 * N>
{

    auto constexpr steps = integ_delta_steps<N>();
    auto delta           = std::array<double, 3 * N>();
    for (size_t i = 0; i < N / 2; ++i)
    {
        delta[i * 3 + 0]             = 1.0 - steps[i];
        delta[i * 3 + 1]             = steps[i];
        delta[i * 3 + 2]             = 0.0;
        delta[(i + (N / 2)) * 3 + 0] = 0.0;
        delta[(i + (N / 2)) * 3 + 1] = 1.0 - steps[i + (N / 2)];
        delta[(i + (N / 2)) * 3 + 2] = steps[i + (N / 2)];
    }

    return delta;
}

template <size_t N>
auto constexpr integ_delta_lo1() -> std::array<double, N>
{

    auto constexpr steps = integ_delta_steps<N>();
    auto delta           = std::array<double, N>();
    for (size_t i = 0; i < N / 2; ++i)
    {
        delta[i * 2 + 0] = -steps[i];
        delta[i * 2 + 1] = 1.0 + steps[i];
    }
    return delta;
}

template <size_t N>
auto constexpr integ_delta_hi1() -> std::array<double, N>
{

    auto constexpr steps = integ_delta_steps<N>();
    auto delta           = std::array<double, N>();
    for (size_t i = 0; i < N / 2; ++i)
    {
        delta[i * 2 + 0] = 1.0 - steps[i + N / 2];
        delta[i * 2 + 1] = steps[i + N / 2];
    }
    return delta;
}

template <size_t N>
auto constexpr integ_delta_1() -> std::array<double, 3 * N>
{

    auto constexpr steps = integ_delta_steps<N>();
    auto delta           = std::array<double, 3 * N>();
    for (size_t i = 0; i < N / 2; ++i)
    {
        delta[0 + i * 3]             = -steps[i];
        delta[1 + i * 3]             = 1.0 + steps[i];
        delta[2 + i * 3]             = 0.0;
        delta[0 + (i + (N / 2)) * 3] = 0.0;
        delta[1 + (i + (N / 2)) * 3] = 1.0 - steps[i + (N / 2)];
        delta[2 + (i + (N / 2)) * 3] = steps[i + (N / 2)];
    }

    return delta;
}

// 00 01 02 03  |
// 10 11 12 13  |
// 20 21 22 23  |
// 30 31 32 33  V
//
// 00x 10x 20x 30x 01x 11x 21x 31x 02x 12x 22x 32x 03x 13x 23x 33x  |
// 00y 10y 20y 30y 01y 11y 21y 31y 02y 12y 22y 32y 03y 13y 23y 33y  V
//
// 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3  |
// 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3  V

template <size_t N>
auto constexpr integ_delta_lin() -> std::array<double, 2 * N * N>
{

    auto constexpr steps = integ_delta_steps<N>();
    auto delta           = std::array<double, 2 * N * N>();
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            delta[0 + i * 2 + j * 2 * N] = steps[j];
            delta[1 + i * 2 + j * 2 * N] = steps[i];
        }
    }

    return delta;
}


auto
mean_psf(double const d, Tensor2d const& uPsf) -> Tensor1d;

template <size_t Rank>
auto
mean_psf(Tensor<double, Rank> const& Sep, Tensor2d const& uPsf) -> Tensor1d
{
    long const& Ne = uPsf.dimension(0);
    long const& ND = Sep.size();
    Tensor1d    result(Ne);
    for (long d = 0; d < ND; ++d) { result += mean_psf(Sep(d), uPsf); }
    return result / (4.0 * ND);
}

template <short N>
auto
rectangular_comb_separations(long const                      pw,
                             long const                      ph,
                             double const                    ref_size,
                             std::pair<double, double> const src_pix,
                             Eigen::MatrixXd const&          Offsets)
    -> Eigen::Matrix<double, N, N>
{
    auto constexpr _dvec = Fermi::ModelMap::integ_delta_1<N>();
    Eigen::Map<Eigen::Matrix<double, 3, N> const> const D(_dvec.data());

    Eigen::Matrix<double, N, N> SD
        = D.transpose() * Offsets.block<3, 3>(pw - 1, ph - 1) * D;

    auto constexpr dsmatv = Fermi::ModelMap::integ_delta_lin<N>();
    Eigen::Map<Eigen::Matrix<double, 2, N * N> const> const dsm(
        dsmatv.data(), 2, N * N);

    Eigen::Vector2d spv(src_pix.first - ph, src_pix.second - pw);

    SD = ref_size * (dsm.colwise() - spv).colwise().norm().reshaped(N, N).array()
         * (1. + SD.array());
    return SD;
}

template <short Ndelta, short NMAX = 64>
auto
integrate_psf_adapt_recurse(long const      px,
                            long const      py,
                            Tensor2d const& Offsets,
                            Tensor2d const& uPsf, // Ne, Nd
                            Tensor1d const& v0) -> Tensor1d
{
    double constexpr ftol_threshold = 0.001; // config.psfEstimatorFtol();
    size_t constexpr Nhalf          = Ndelta / 2;
    size_t const& Ne                = uPsf.dimension(0);
    Idx2 constexpr e32              = { 3, 2 };
    Idx2 constexpr e22              = { 2, Nhalf };
    Idx2 constexpr o2l              = { 0, 0 };
    Idx2 constexpr o2h              = { 1, 0 };
    Idx2 off                        = { px, py };
    IdxPair1 constexpr cdimA        = { Eigen::IndexPair<long>(1, 0) };
    IdxPair1 constexpr cdimB        = { Eigen::IndexPair<long>(0, 0) };
    auto constexpr delta_lo_arr     = integ_delta_lo<Ndelta>();
    auto constexpr delta_hi_arr     = integ_delta_hi<Ndelta>();
    Eigen::TensorMap<Tensor2d const> const Dlo(delta_lo_arr.data(), 2, Nhalf);
    Eigen::TensorMap<Tensor2d const> const Dhi(delta_hi_arr.data(), 2, Nhalf);

    Tensor1d                 v1(Ne);
    Tensor2d                 SD(Nhalf, Nhalf);
    Tensor2d                 ID(3, Nhalf);
    Eigen::Tensor<double, 2> P = Offsets.slice(off, e32);

    // [3,2][2,Nhalf] = [3,Nhalf]
    P.contract(Dlo, cdimA, ID);

    // [2,Nhalf][2,Nhalf] = [Nhalf,Nhalf]
    Dlo.contract(ID.slice(o2l, e22), cdimB, SD.setZero());
    v1 += mean_psf<2>(SD, uPsf);
    Dhi.contract(ID.slice(o2h, e22), cdimB, SD.setZero());
    v1 += mean_psf<2>(SD, uPsf);
    off = { px, py + 1 };
    P   = Offsets.slice(off, e32);
    P.contract(Dhi, cdimA, ID);
    Dlo.contract(ID.slice(o2l, e22), cdimB, SD.setZero());
    v1 += mean_psf<2>(SD, uPsf);
    Dhi.contract(ID.slice(o2h, e22), cdimB, SD.setZero());
    v1 += mean_psf<2>(SD, uPsf);

    // if constexpr (Ndelta >= 64) return v1;

    Tensor0b all_zero_or_true = v1.all();
    if (all_zero_or_true(0)) { return v1; }

    Tensor1d vdiff   = v1 - v0;

    all_zero_or_true = vdiff.all();
    if (all_zero_or_true(0)) { return v1; }

    all_zero_or_true = ((vdiff / v1).abs() < ftol_threshold).all();
    if (all_zero_or_true(0)) { return v1; }

    return integrate_psf_adapt_recurse<2 * Ndelta, NMAX>(px, py, Offsets, uPsf, v1);
}


auto
integrate_psf_adaptive(long const      px,
                       long const      py,
                       Tensor2d const& Offsets,
                       Tensor2d const& uPsf,    // Ne, Nd
                       Tensor1d const& uPsfPeak // Ne
                       ) -> Tensor1d;

auto
point_src_model_map_wcs(long const      Nx,
                        long const      Ny,
                        vpd const&      src_dirs,
                        Tensor3d const& uPsf,
                        Tensor2d const& uPsfPeak,
                        SkyGeom const&  skygeom) -> Tensor4d;

} // namespace Fermi::ModelMap
