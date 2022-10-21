#include "xtsrcmaps/model_map.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/psf.hxx"

#include "unsupported/Eigen/CXX11/Tensor"

auto
Fermi::ModelMap::mean_psf(double const d, Tensor2d const& uPsf) -> Tensor1d
{
    long const& Ne = uPsf.dimension(0);
    if (d >= 70.0 || d < 0.0) return Tensor1d(Ne);

    long const i = long(d);
    FixTen1d_2 pt;
    pt[0]          = 1.0 - (double(i) - d);
    pt[1]          = 1.0 + pt[0];
    Tensor2d psf_s = uPsf.slice(Idx2 { 0, i }, Idx2 { Ne, 2 });
    return psf_s.contract(pt, IdxPair1 { { { 1, 0 } } });
}


auto
Fermi::ModelMap::integrate_psf_adaptive(long const      px,
                                        long const      py,
                                        Tensor2d const& Offsets,
                                        Tensor2d const& uPsf,    // Ne, Nd
                                        Tensor1d const& uPsfPeak // Ne
                                        ) -> Tensor1d
{
    double constexpr peak_threshold = 1e-6; // config.psfEstimatorPeakTh();
    double const offset             = Offsets(px, py);

    Tensor1d v0                     = mean_psf(offset, uPsf);
    Tensor1d ones                   = v0.constant(1.0);
    Tensor1d zeros                  = v0.constant(0.0);
    Tensor1b zero_uPsfPeak          = uPsfPeak.cast<bool>();
    Tensor1d safeUPsfPeak           = v0 / zero_uPsfPeak.select(ones, uPsfPeak);
    Tensor1d peak_ratio             = zero_uPsfPeak.select(zeros, safeUPsfPeak);
    Tensor0b all_below_peak         = (peak_ratio < peak_threshold).all();
    if (all_below_peak(0)) return v0;

    return integrate_psf_adapt_recurse<2, 64>(px, py, Offsets, uPsf, v0);
}

template <>
auto
Fermi::ModelMap::integrate_psf_adapt_recurse<64, 64>(long const      px,
                                                     long const      py,
                                                     Tensor2d const& Offsets,
                                                     Tensor2d const& uPsf, // Ne, Nd
                                                     Tensor1d const& v0) -> Tensor1d
{
    (void)(v0);
    size_t constexpr Nhalf      = 32;
    size_t const& Ne            = uPsf.dimension(0);
    Idx2 constexpr e32          = { 3, 2 };
    Idx2 constexpr e22          = { 2, Nhalf };
    Idx2 constexpr o2l          = { 0, 0 };
    Idx2 constexpr o2h          = { 1, 0 };
    Idx2 off                    = { px, py };
    IdxPair1 constexpr cdimA    = { { { 1, 0 } } };
    IdxPair1 constexpr cdimB    = { { { 0, 0 } } };
    auto constexpr delta_lo_arr = integ_delta_lo<64>();
    auto constexpr delta_hi_arr = integ_delta_hi<64>();
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

    return v1;
}

auto
Fermi::ModelMap::point_src_model_map_wcs(long const      Npx,
                                         long const      Npy,
                                         vpd const&      dirs,
                                         Tensor3d const& uPsf,
                                         Tensor2d const& uPsfPeak,
                                         SkyGeom const&  skygeom) -> Tensor4d
{
    long const& Ns = uPsf.dimension(0);
    long const& Ne = uPsf.dimension(1);
    long const& Nd = uPsf.dimension(2);
    assert(dirs.size() == 263);
    assert(Ns == 263);
    assert(Ne == 38);
    assert(Nd == 401);

    Tensor4d mm(Ns, Npx, Npy, Ne);

    Eigen::Tensor<double, 3> PixDirs(3, Npx, Npy);
    for (long px = 0; px < Npx; ++px)
    {
        for (long py = 0; py < Npy; ++py)
        {
            auto pdir           = skygeom.pix2dir({ px + 1.0, py + 1.0 });
            PixDirs(0l, px, py) = std::get<0>(pdir);
            PixDirs(1l, px, py) = std::get<1>(pdir);
            PixDirs(2l, px, py) = std::get<2>(pdir);
        }
    }

    for (int s = 0; s < Ns; ++s)
    {
        auto     src_dir = skygeom.sph2dir(dirs[s]); // CLHEP Style 3
        Tensor2d Offsets(Npx + 2, Npy + 2);
        Offsets.setZero();
        for (long px = 0; px < Npx; ++px)
        {
            for (long py = 0; py < Npy; ++py)
            {
                Offsets(px, py) = PSF::inverse_separations(skygeom.srcpixoff(
                    src_dir,
                    { PixDirs(0, px, py), PixDirs(1, px, py), PixDirs(2, px, py) }));
            }
        }

        Tensor2d const suPsf
            = uPsf.slice(Idx3 { s, 0, 0 }, Idx3 { 1, Ne, Nd }).reshape(Idx2 { Ne, Nd });
        Tensor1d const suPsfPeak
            = uPsfPeak.slice(Idx2 { s, 0 }, Idx2 { 1, Ne }).reshape(Idx1 { Ne });

        for (long px = 0; px < Npx; ++px)
        {
            for (long py = 0; py < Npy; ++py)
            {
                mm.slice(Idx4 { s, px, py, 0 }, Idx4 { 1, 1, 1, Ne })
                    .reshape(Idx1 { Ne })
                    = integrate_psf_adaptive(px, py, Offsets, suPsf, suPsfPeak);
            }
        }
    }

    return mm;
}
