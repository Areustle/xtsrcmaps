#pragma once


#include "xtsrcmaps/math/tensor_types.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"

#ifdef __APPLE__
#include "wcs.h"
#else
#include "wcslib/wcs.h"
#endif

namespace Fermi {

class SkyGeom {

    std::array<char, 3000> m_wcs_struct {};
    wcsprm*                m_wcs;
    // = std::unique_ptr<wcsprm>(new (m_wcs_struct.data()) wcsprm);
    std::string m_proj_name;
    bool        m_is_galactic;
    bool        m_wcspih_used = false;

  public:
    SkyGeom(Obs::CCubePixels const&);
    ~SkyGeom();

    // // Delete all the other constructor types.
    // SkyGeom()          = delete;
    // SkyGeom(SkyGeom&&) = delete;
    // SkyGeom
    // operator=(SkyGeom&&)
    //     = delete;
    // SkyGeom
    // operator=(SkyGeom const&)
    //     = delete;

    auto sph2pix(Vector2d const&) const -> Vector2d;

    auto sph2pix(std::pair<double, double> const&) const
        -> std::pair<double, double>;

    auto pix2sph(Vector2d const&) const -> Vector2d;

    auto pix2sph(double const, double const) const -> Vector2d;

    auto sph2pix(Obs::sphcrd_v_t const&) const -> Obs::sphcrd_v_t;

    auto pix2sph(Eigen::Matrix2Xd const&) const -> Eigen::Matrix2Xd;

    // auto
    // pix2sph(Eigen::Matrix2Xd const&) const -> Eigen::Matrix2Xd;

    auto dir2sph(Vector3d const&) const -> Vector2d;

    auto pix2dir(Vector2d const&) const -> Vector3d;

    auto sph2dir(Vector2d const&) const -> Vector3d;

    auto sph2dir(std::pair<double, double> const&) const -> Vector3d;

    auto srcpixoff(Vector3d const& src, Vector2d const& pix) const -> double;

    auto srcpixoff(Vector3d const& src, Vector3d const& pix) const -> double;
};


inline auto
dir_diff(Vector3d const& L, Vector3d const& R) -> double {
    return 2. * asin(0.5 * (L - R).norm());
};

auto
pix_diff(Vector2d const& L, Vector2d const& R, Fermi::SkyGeom const& skygeom)
    -> double;

auto sph_pix_diff(std::pair<double, double> const& L,
                  Vector2d const&                  R,
                  SkyGeom const&                   skygeom) -> double;

} // namespace Fermi
