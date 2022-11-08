#pragma once

#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include "unsupported/Eigen/CXX11/Tensor"

namespace Fermi::ModelMap
{

using MatCoord3 = Eigen::Matrix<SkyGeom::coord3, Eigen::Dynamic, Eigen::Dynamic>;

auto
pix_dirs_with_padding(SkyGeom const& skygeom, long const Nw, long const Nh)
    -> MatCoord3;

auto
pixel_angular_offset_from_source_with_padding(SkyGeom::coord3 const& src_dir,
                                              SkyGeom::coord2 const& src_pix,
                                              double const           ref_size,
                                              SkyGeom const&         skygeom,
                                              long const             NpW,
                                              long const NpH) -> Eigen::MatrixXd;

auto
pixel_angular_offset_from_source_with_padding(MatCoord3 const&       pdirs,
                                              SkyGeom::coord3 const& src_dir,
                                              SkyGeom::coord2 const& src_pix,
                                              double const           ref_size,
                                              SkyGeom const&         skygeom)
    -> Eigen::MatrixXd;

// Index in PSF Lookup Table of Pixel given by dispacement from source
auto
psf_idx_sep(SkyGeom::coord3 const& src_dir,
            MatCoord3 const&       pdirs,
            SkyGeom const&         skygeom) -> Eigen::MatrixXd;

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
pixels_to_integrate(Eigen::Ref<Eigen::MatrixXd const> const& mean_psf_v0,
                    Eigen::Ref<Eigen::VectorXd const> const& suPeak,
                    double const                             peak_threshold,
                    long const                               Nw,
                    long const Nh) -> std::vector<std::pair<long, long>>;

template <typename D0, typename D1>
void
psf_lut(Eigen::MatrixBase<D0>&      mean_psf_v0,
        Eigen::ArrayBase<D1> const& sep,
        Eigen::MatrixXd const&      suPsf)
{
    Eigen::ArrayXd r = sep - sep.floor();

    mean_psf_v0      = suPsf(sep.floor(), Eigen::all).array().colwise() * (1. - r)
                  + suPsf(sep.floor() + 1, Eigen::all).array().colwise() * r;
}

template <typename D1>
auto
psf_lut(Eigen::ArrayBase<D1> const& isep, Eigen::MatrixXd const& suPsf)
    -> Eigen::MatrixXd
{
    Eigen::ArrayXd r = isep - isep.floor();

    return suPsf(isep.floor(), Eigen::all).array().colwise() * (1. - r)
           + suPsf(isep.floor() + 1, Eigen::all).array().colwise() * r;
}

template <typename D0, typename D1>
void
mean_psf_lut(Eigen::MatrixBase<D0>&      mean_psf,
             Eigen::ArrayBase<D1> const& sep,
             Eigen::MatrixXd const&      suPsf)
{
    Eigen::ArrayXd r = sep - sep.floor();

    mean_psf         = (suPsf(sep.floor(), Eigen::all).array().colwise() * (1. - r)
                + suPsf(sep.floor() + 1, Eigen::all).array().colwise() * r)
                   .colwise()
                   .mean();
    // std::cout << mean_psf << std::endl << std::endl;
}

namespace detail
{
template <typename F>
struct visitor_wrapper : F
{
    template <typename S, typename I>
    auto
    init(const S& v, I i, I j) -> void
    {
        return F::operator()(v, i, j);
    }
};
} // namespace detail

template <typename M, typename F>
void
visit_lambda(const M& m, const F& f)
{
    auto v = detail::visitor_wrapper<F>(f);
    m.visit(v);
}

auto
is_integ_psf_converged(Eigen::Ref<Eigen::MatrixXd const> const& v0,
                       Eigen::Ref<Eigen::MatrixXd const> const& v1,
                       double const                             ftol_threshold) -> bool;


// template <short N>
// auto
// rectangular_comb_separations(long const                      pw,
//                              long const                      ph,
//                              double const                    ref_size,
//                              std::pair<double, double> const src_pix,
//                              Eigen::MatrixXd const&          Offsets)
//     -> Eigen::Matrix<double, N, N>
// {
//     // Linear Interpolation delta steps as a function of Integral Depth N.
//     auto constexpr _dvec = Fermi::ModelMap::integ_delta_1<N>();
//     Eigen::Map<Eigen::Matrix<double, 3, N> const> const D(_dvec.data());
//
//     // Perform Lerp of 3x3 offxet region around target pixel offset.
//     Eigen::Matrix<double, N, N> SD
//         = D.transpose() * Offsets.block<3, 3>(pw - 1, ph - 1) * D;
//
//     // Delta steps for displacement-from-source computation.
//     auto constexpr dsmatv = Fermi::ModelMap::integ_delta_lin<N>();
//     Eigen::Map<Eigen::Matrix<double, 2, N * N> const> const dsm(
//         dsmatv.data(), 2, N * N);
//
//     // Target pixel displacement from source
//     Eigen::Vector2d spv(src_pix.first - ph, src_pix.second - pw);
//
//     // Separation of offset from
//     SD = ref_size * (dsm.colwise() - spv).colwise().norm().reshaped(N, N).array()
//          * (1. + SD.array());
//     return SD;
// }


template <short N>
auto
rectangular_true_separations(long const            pw,
                             long const            ph,
                             SkyGeom::coord3 const src_dir,
                             SkyGeom const& skygeom) -> Eigen::Matrix<double, N, N>
{
    auto constexpr ds = Fermi::ModelMap::integ_delta_steps<N>();
    Eigen::Matrix<double, N, N> SD;
    for (long h = 0; h < N; ++h)
    {
        for (long w = 0; w < N; ++w)
        {
            SD(w, h) = skygeom.srcpixoff(src_dir, { ph + ds[h], pw + ds[w] });
        }
    }
    return SD;
}


template <unsigned short Ndelta>
auto
integrate_psf_(long const             w,
               long const             h,
               SkyGeom::coord3 const  src_dir,
               SkyGeom const&         skygeom,
               Eigen::MatrixXd const& suPsf // Nd, Ne
               ) -> Eigen::MatrixXd
{
    long const&                           Ne = suPsf.cols();
    Eigen::Matrix<double, Ndelta, Ndelta> sep
        = rectangular_true_separations<Ndelta>(w, h, src_dir, skygeom);

    sep = (sep.array() < 1e-4)
              .select(1e4 * sep, 1. + ((sep.array() * 1e4).log() / sep_step));
    // sep.unaryExpr(&PSF::linear_inverse_separation);

    Eigen::MatrixXd v1(1, Ne);
    // std::cout << sep << std::endl << std::endl;
    mean_psf_lut(v1, sep.reshaped().array(), suPsf);
    // std::cout << v1 << std::endl << std::endl;
    return v1;
}

template <unsigned short Ndelta = 2u>
auto
integrate_psf_recursive(long const                               pw,
                        long const                               ph,
                        SkyGeom::coord3 const                    src_dir,
                        SkyGeom const&                           skygeom,
                        double const                             ftol_threshold,
                        Eigen::MatrixXd const&                   suPsf, // Nd, Ne
                        Eigen::Ref<Eigen::MatrixXd const> const& v0) -> Eigen::MatrixXd
{
    Eigen::MatrixXd v1 = v0;
    // auto const v1 = integrate_psf_<Ndelta>(pw, ph, src_dir, skygeom, suPsf);
    // std::cout << v1 << std::endl << std::endl;
    // std::cout << v1 - v0 << std::endl << std::endl;


    return is_integ_psf_converged(v0, v1, ftol_threshold)
               ? v1
               : integrate_psf_recursive<2u * Ndelta>(
                   pw, ph, src_dir, skygeom, ftol_threshold, suPsf, v1);
}

template <>
auto
integrate_psf_recursive<64u>(long const                               pw,
                             long const                               ph,
                             SkyGeom::coord3 const                    src_dir,
                             SkyGeom const&                           skygeom,
                             double const                             ftol_threshold,
                             Eigen::MatrixXd const&                   suPsf, // Nd, Ne
                             Eigen::Ref<Eigen::MatrixXd const> const& v0)
    -> Eigen::MatrixXd;


// // Approximate separation computation
// template <unsigned short Ndelta>
// auto
// integrate_psf_(long const                      pw,
//                long const                      ph,
//                double const                    ref_size,
//                std::pair<double, double> const src_pix,
//                Eigen::MatrixXd const&          Offsets,
//                Eigen::MatrixXd const&          suPsf // Nd, Ne
//                ) -> Eigen::VectorXd
// {
//     long const&                           Ne = suPsf.cols();
//     Eigen::Matrix<double, Ndelta, Ndelta> sep
//         = rectangular_comb_separations<Ndelta>(pw, ph, ref_size, src_pix, Offsets);
//     sep = (sep.array() < 1e-4)
//               .select(1e4 * sep, ((sep.array() * 1e4).log() / sep_step) + 1.);
//
//     Eigen::MatrixXd v1(1, Ne);
//     mean_psf_lut(v1, sep.reshaped().array(), suPsf);
//     return v1;
// }
//
//
// template <unsigned short Ndelta>
// auto
// integrate_psf_recursive(long const                               pw,
//                         long const                               ph,
//                         double const                             ref_size,
//                         std::pair<double, double> const          src_pix,
//                         double const                             ftol_threshold,
//                         Eigen::MatrixXd const&                   Offsets,
//                         Eigen::MatrixXd const&                   suPsf, // Nd, Ne
//                         Eigen::Ref<Eigen::MatrixXd const> const& v0) ->
//                         Eigen::VectorXd
// {
//     Eigen::Matrix v1
//         = integrate_psf_<Ndelta>(pw, ph, ref_size, src_pix, Offsets, suPsf);
//
//     return is_integ_psf_converged(v0, v1, ftol_threshold)
//                ? v1
//                : integrate_psf_recursive<2u * Ndelta>(
//                    pw, ph, ref_size, src_pix, ftol_threshold, Offsets, suPsf, v1);
// }



auto
point_src_model_map_wcs(long const      Nx,
                        long const      Ny,
                        vpd const&      src_dirs,
                        Tensor3d const& uPsf,
                        Tensor2d const& uPsfPeak,
                        SkyGeom const&  skygeom) -> Tensor4d;

// template <short Ndelta, short NMAX = 64>
// auto
// integrate_psf_adapt_recurse(long const      px,
//                             long const      py,
//                             Tensor2d const& Offsets,
//                             Tensor2d const& uPsf, // Ne, Nd
//                             Tensor1d const& v0) -> Tensor1d
// {
//     double constexpr ftol_threshold = 0.001; // config.psfEstimatorFtol();
//     size_t constexpr Nhalf          = Ndelta / 2;
//     size_t const& Ne                = uPsf.dimension(0);
//     Idx2 constexpr e32              = { 3, 2 };
//     Idx2 constexpr e22              = { 2, Nhalf };
//     Idx2 constexpr o2l              = { 0, 0 };
//     Idx2 constexpr o2h              = { 1, 0 };
//     Idx2 off                        = { px, py };
//     IdxPair1 constexpr cdimA        = { Eigen::IndexPair<long>(1, 0) };
//     IdxPair1 constexpr cdimB        = { Eigen::IndexPair<long>(0, 0) };
//     auto constexpr delta_lo_arr     = integ_delta_lo<Ndelta>();
//     auto constexpr delta_hi_arr     = integ_delta_hi<Ndelta>();
//     Eigen::TensorMap<Tensor2d const> const Dlo(delta_lo_arr.data(), 2, Nhalf);
//     Eigen::TensorMap<Tensor2d const> const Dhi(delta_hi_arr.data(), 2, Nhalf);
//
//     Tensor1d                 v1(Ne);
//     Tensor2d                 SD(Nhalf, Nhalf);
//     Tensor2d                 ID(3, Nhalf);
//     Eigen::Tensor<double, 2> P = Offsets.slice(off, e32);
//
//     // [3,2][2,Nhalf] = [3,Nhalf]
//     P.contract(Dlo, cdimA, ID);
//
//     // [2,Nhalf][2,Nhalf] = [Nhalf,Nhalf]
//     Dlo.contract(ID.slice(o2l, e22), cdimB, SD.setZero());
//     v1 += mean_psf<2>(SD, uPsf);
//     Dhi.contract(ID.slice(o2h, e22), cdimB, SD.setZero());
//     v1 += mean_psf<2>(SD, uPsf);
//     off = { px, py + 1 };
//     P   = Offsets.slice(off, e32);
//     P.contract(Dhi, cdimA, ID);
//     Dlo.contract(ID.slice(o2l, e22), cdimB, SD.setZero());
//     v1 += mean_psf<2>(SD, uPsf);
//     Dhi.contract(ID.slice(o2h, e22), cdimB, SD.setZero());
//     v1 += mean_psf<2>(SD, uPsf);
//
//     // if constexpr (Ndelta >= 64) return v1;
//
//     Tensor0b all_zero_or_true = v1.all();
//     if (all_zero_or_true(0)) { return v1; }
//
//     Tensor1d vdiff   = v1 - v0;
//
//     all_zero_or_true = vdiff.all();
//     if (all_zero_or_true(0)) { return v1; }
//
//     all_zero_or_true = ((vdiff / v1).abs() < ftol_threshold).all();
//     if (all_zero_or_true(0)) { return v1; }
//
//     return integrate_psf_adapt_recurse<2 * Ndelta, NMAX>(px, py, Offsets, uPsf, v1);
// }

} // namespace Fermi::ModelMap
