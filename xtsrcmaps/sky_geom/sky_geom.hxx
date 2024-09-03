#pragma once


#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"

#include <array>
#include <string>

#ifdef __APPLE__
#include "wcs.h"
#else
#include "wcslib/wcs.h"
#endif

namespace Fermi {

template <typename T>
class SkyGeom {

    std::array<char, 3000> m_wcs_struct {};
    wcsprm*                m_wcs { nullptr }; // Initialize to nullptr
    std::string            m_proj_name;
    bool                   m_is_galactic { false };
    bool                   m_wcspih_used { false };

  public:
    SkyGeom() {
        // Initialize wcsprm with a safe state
        m_wcs = new (m_wcs_struct.data()) wcsprm;
        std::memset(m_wcs, 0, sizeof(wcsprm)); // Ensure it's zeroed out
        m_wcs->flag = -1;
    }

    SkyGeom(Obs::CCubePixels const& pars) {
        // Mostly copied from SkyProj.cxx in Sciencetools Likelihood.

        m_wcs         = new (m_wcs_struct.data()) wcsprm;
        m_wcs->flag   = -1;
        m_proj_name   = pars.proj_name;
        m_is_galactic = pars.is_galactic;
        m_wcspih_used = false;

        wcsini(1, 2, m_wcs);

        std::string lon_type = (pars.is_galactic ? "GLON" : "RA"),
                    lat_type = (pars.is_galactic ? "GLAT" : "DEC");

        if (pars.proj_name.compare("") != 0) {
            lon_type += (pars.is_galactic ? "-" : "---") + pars.proj_name;
            lat_type += (pars.is_galactic ? "-" : "--") + pars.proj_name;
        }

        strcpy(m_wcs->ctype[0], lon_type.c_str());
        strcpy(m_wcs->ctype[1], lat_type.c_str());

        m_wcs->crval[0] = pars.crval[0]; // reference value
        m_wcs->crval[1] = pars.crval[1]; // reference value
        //
        m_wcs->crpix[0] = pars.crpix[0]; // pixel coordinate
        m_wcs->crpix[1] = pars.crpix[1]; // pixel coordinate
        //
        m_wcs->cdelt[0] = pars.cdelt[0]; // scale factor
        m_wcs->cdelt[1] = pars.cdelt[1]; // scale factor

        if (m_wcs->crval[0] > 180) { m_wcs->crval[0] -= 360; }

        // Set wcs to use CROTA rotations instead of PC or CD  transformations
        m_wcs->altlin |= 4;
        m_wcs->crota[1] = pars.axis_rot;

#ifdef WIN32
        int status = wcsset2(p.wcs.get());
#else
        int status = wcsset(m_wcs);
#endif
        (void)(status);
    }
    // Destructor to clean up wcsprm
    ~SkyGeom() {
        if (m_wcs) { wcsfree(m_wcs); }
    }

    // Add copy constructor and assignment operators if necessary
    SkyGeom(SkyGeom const& other) {
        m_wcs = new (m_wcs_struct.data()) wcsprm;
        std::memcpy(m_wcs, other.m_wcs, sizeof(wcsprm));
        m_proj_name   = other.m_proj_name;
        m_is_galactic = other.m_is_galactic;
        m_wcspih_used = other.m_wcspih_used;
    }

    SkyGeom& operator=(SkyGeom const& other) {
        if (this != &other) {
            std::memcpy(m_wcs, other.m_wcs, sizeof(wcsprm));
            m_proj_name   = other.m_proj_name;
            m_is_galactic = other.m_is_galactic;
            m_wcspih_used = other.m_wcspih_used;
        }
        return *this;
    }

    // Add move constructor and move assignment operators if necessary
    SkyGeom(SkyGeom&& other) noexcept {
        m_wcs = new (m_wcs_struct.data()) wcsprm;
        std::memcpy(m_wcs, other.m_wcs, sizeof(wcsprm));
        m_proj_name   = std::move(other.m_proj_name);
        m_is_galactic = other.m_is_galactic;
        m_wcspih_used = other.m_wcspih_used;
        other.m_wcs   = nullptr; // Leave other in a valid state
    }

    SkyGeom& operator=(SkyGeom&& other) noexcept {
        if (this != &other) {
            std::memcpy(m_wcs, other.m_wcs, sizeof(wcsprm));
            m_proj_name   = std::move(other.m_proj_name);
            m_is_galactic = other.m_is_galactic;
            m_wcspih_used = other.m_wcspih_used;
            other.m_wcs   = nullptr; // Leave other in a valid state
        }
        return *this;
    }

    auto sph2pix(std::array<T, 2> const& ss) const -> std::array<T, 2> {
        double    s1 = ss[0], s2 = ss[1];
        int const ncoords = 1;
        int const nelem   = 2;
        double    imgcrd[2], pixcrd[2];
        double    phi[1], theta[1];
        int       stat[1];

        // WCS projection routines require the input coordinates are in degrees
        // and in the range of [-90,90] for the lat and [-180,180] for the lon.
        // So correct for this effect.
        bool wrap_pos = false;
        if (s1 > 180) {
            s1 -= 360.;
            wrap_pos = false;
        }
        bool wrap_neg = false;
        if (s1 < -180) {
            s1 += 360.;
            wrap_neg = false;
        }
        double worldcrd[] = { s1, s2 };
        wcss2p(
            m_wcs, ncoords, nelem, worldcrd, phi, theta, imgcrd, pixcrd, stat);
        if (wrap_pos) { pixcrd[0] += 360. / m_wcs->cdelt[0]; }
        if (wrap_neg) { pixcrd[0] -= 360. / m_wcs->cdelt[0]; }
        return { static_cast<T>(pixcrd[0]), static_cast<T>(pixcrd[1]) };
    }

    auto sph2pix(std::pair<T, T> const& ss) const -> std::pair<T, T> {
        auto [s1, s2]     = ss;
        int const ncoords = 1;
        int const nelem   = 2;
        double    imgcrd[2], pixcrd[2];
        double    phi[1], theta[1];
        int       stat[1];

        // WCS projection routines require the input coordinates are in degrees
        // and in the range of [-90,90] for the lat and [-180,180] for the lon.
        // So correct for this effect.
        bool wrap_pos = false;
        if (s1 > 180) {
            s1 -= 360.;
            wrap_pos = false;
        }
        bool wrap_neg = false;
        if (s1 < -180) {
            s1 += 360.;
            wrap_neg = false;
        }
        double worldcrd[] = { s1, s2 };
        wcss2p(
            m_wcs, ncoords, nelem, worldcrd, phi, theta, imgcrd, pixcrd, stat);
        if (wrap_pos) { pixcrd[0] += 360. / m_wcs->cdelt[0]; }
        if (wrap_neg) { pixcrd[0] -= 360. / m_wcs->cdelt[0]; }
        return { static_cast<T>(pixcrd[0]), static_cast<T>(pixcrd[1]) };
    }

    auto pix2sph(std::array<T, 2> const& px) const -> std::array<T, 2> {
        int    ncoords = 1;
        int    nelem   = 2;
        double worldcrd[2], imgcrd[2];
        double phi[1], theta[1];
        int    stat[1];
        double pixcrd[] = { px[0], px[1] };
        wcsp2s(
            m_wcs, ncoords, nelem, pixcrd, imgcrd, phi, theta, worldcrd, stat);
        double s1 = worldcrd[0];
        while (s1 < 0) s1 += 360.;
        while (s1 >= 360) s1 -= 360.;
        return { static_cast<T>(s1), static_cast<T>(worldcrd[1]) };
    }

    auto pix2sph(T const first, T const second) const -> std::array<T, 2> {
        int    ncoords = 1;
        int    nelem   = 2;
        double worldcrd[2], imgcrd[2];
        double phi[1], theta[1];
        int    stat[1];
        double pixcrd[] = { first, second };
        wcsp2s(
            m_wcs, ncoords, nelem, pixcrd, imgcrd, phi, theta, worldcrd, stat);
        double s1 = worldcrd[0];
        while (s1 < 0) s1 += 360.;
        while (s1 >= 360) s1 -= 360.;
        return { static_cast<T>(s1), static_cast<T>(worldcrd[1]) };
    }

    auto dir2sph(std::array<T, 3> const& dir) const -> std::array<T, 2> {
        T ra = atan2(dir[1], dir[0]) * rad2deg;
        // fold RA into the range (0,360)
        while (ra < 0) ra += 360.;
        while (ra > 360) ra -= 360.;
        T dec = asin(dir[2]) * rad2deg;
        return { ra, dec };
    }


    auto pix2dir(std::array<T, 2> const& px) const -> std::array<T, 3> {
        return sph2dir(pix2sph(px));
    }

    auto sph2dir(std::pair<T, T> const& s) const -> std::array<T, 3> {
        T cos_ra  = cos(s.first * deg2rad);
        T cos_dec = cos(s.second * deg2rad);
        T sin_ra  = sin(s.first * deg2rad);
        T sin_dec = sin(s.second * deg2rad);
        return { cos_ra * cos_dec, sin_ra * cos_dec, sin_dec };
    }

    auto srcpixoff(std::array<T, 3> const& src_dir_coord,
                   std::array<T, 2> const& delta_pix) const -> T {
        auto const dpx = pix2dir(delta_pix);
        return srcpixoff(src_dir_coord, dpx);
    }

    static inline auto
    dir_diff(std::array<T, 3> const& L, std::array<T, 3> const& R) -> T {
        std::array<T, 3> tmp = { L[0] - R[0], L[1] - R[1], L[2] - R[2] };
        T norm = std::sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2]);
        return 2. * asin(0.5 * norm);
    };

    static auto pix_diff(std::array<T, 2> const& L,
                         std::array<T, 2> const& R,
                         SkyGeom const&          skygeom) -> T {
        return dir_diff(skygeom.pix2dir(L), skygeom.pix2dir(R));
    };

    static auto sph_pix_diff(std::pair<T, T> const&  L,
                             std::array<T, 2> const& R,
                             SkyGeom const&          skygeom) -> T;
    static auto sph_pix_diff(std::pair<double, double> const& L,
                             std::array<float, 2> const&      R,
                             SkyGeom const& skygeom) -> double {
        return dir_diff(skygeom.sph2dir(L), skygeom.pix2dir(R));
    };

    auto srcpixoff(std::array<float, 3> const& src_dir_coord,
                   std::array<float, 3> const& pix) const -> double {
        return dir_diff(src_dir_coord, pix) * rad2deg;
    }
};

} // namespace Fermi
