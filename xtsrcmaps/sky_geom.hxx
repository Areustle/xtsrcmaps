#pragma once

#include <memory>

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include "wcslib/wcs.h"

namespace Fermi
{

class SkyGeom
{

    std::array<char, 3000> m_wcs_struct {};
    wcsprm*                m_wcs;
    // = std::unique_ptr<wcsprm>(new (m_wcs_struct.data()) wcsprm);
    std::string m_proj_name;
    bool        m_is_galactic;
    bool        m_wcspih_used = false;

  public:
    using coord2     = std::pair<double, double>;
    using coord3     = std::tuple<double, double, double>;
    using vec_coord2 = std::vector<std::pair<double, double>>;
    using vec_coord3 = std::vector<std::tuple<double, double, double>>;


    SkyGeom(fits::CCubePixels const&);
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

    auto
    sph2pix(coord2 const&) const -> coord2;

    auto
    pix2sph(coord2 const&) const -> coord2;

    auto
    sph2pix(vec_coord2 const&) const -> vec_coord2;

    auto
    pix2sph(vec_coord2 const&) const -> vec_coord2;

    auto
    dir2sph(coord3 const&) const -> coord2;

    auto
    pix2dir(coord2 const&) const -> coord3;

    auto
    sph2dir(coord2 const&) const -> coord3;

    auto
    srcpixoff(coord3 const& src, coord2 const& pix) const -> double;

    auto
    srcpixoff(coord3 const& src, coord3 const& pix) const -> double;
};

} // namespace Fermi
