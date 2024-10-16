#pragma once


#include "xtsrcmaps/misc/misc.hxx"

#include <array>
#include <string>

#ifdef __APPLE__
#include "wcs.h"
#else
#include "wcslib/wcs.h"
#endif

namespace Fermi {

struct WcsConfig {
    std::array<long, 3>   naxes; // NAXIS extent of image
    std::array<double, 3> crpix; // Reference Pixel Coordinate
    std::array<double, 3> crval; // World Coord Value (RA, Dec) at Ref Pixel.
    std::array<double, 3> cdelt;
    double                axis_rot;
    std::string           proj_name;
    bool                  is_galactic;
};

template <typename T>
class SkyGeom {

    std::array<char, 3000> m_wcs_struct {};
    wcsprm*                m_wcs { nullptr }; // Initialize to nullptr
    std::string            m_proj_name;
    bool                   m_is_galactic { false };
    bool                   m_wcspih_used { false };
    std::array<long, 2>    m_naxes;

  public:
    SkyGeom() {
        // Initialize wcsprm with a safe state
        m_wcs = new (m_wcs_struct.data()) wcsprm { -1 };
        std::memset(m_wcs, 0, sizeof(wcsprm)); // Ensure it's zeroed out
    }

    SkyGeom(long const         naxis1,
            long const         naxis2,
            std::string const& proj_name,
            bool const         is_galactic,
            double const       crpix1,
            double const       crpix2,
            double const       crval1,
            double const       crval2,
            double const       cdelt1,
            double const       cdelt2,
            double const       axis_rot)
        : m_naxes({ naxis1, naxis2 })
        , m_wcs(new(m_wcs_struct.data()) wcsprm { -1 })
        , m_proj_name(proj_name)
        , m_is_galactic(is_galactic)
        , m_wcspih_used(false) {
        initialize_wcs(
            crpix1, crpix2, crval1, crval2, cdelt1, cdelt2, axis_rot);
    }

    SkyGeom(WcsConfig const& cfg)
        : SkyGeom(cfg.naxes[0],
                  cfg.naxes[1],
                  cfg.proj_name,
                  cfg.is_galactic,
                  cfg.crpix[0],
                  cfg.crpix[1],
                  cfg.crval[0],
                  cfg.crval[1],
                  cfg.cdelt[0],
                  cfg.cdelt[1],
                  cfg.axis_rot) {};


  private:
    void initialize_wcs(double crpix1,
                        double crpix2,
                        double crval1,
                        double crval2,
                        double cdelt1,
                        double cdelt2,
                        double axis_rot) {

        // Initialize WCS structure
        wcsini(1, 2, m_wcs);

        // Construct coordinate types
        std::string lon_type = m_is_galactic ? "GLON" : "RA";
        std::string lat_type = m_is_galactic ? "GLAT" : "DEC";

        if (!m_proj_name.empty()) {
            std::string separator = m_is_galactic ? "-" : "---";
            lon_type += separator + m_proj_name;
            lat_type += separator.substr(0, separator.size() - 1) + m_proj_name;
        }

        std::strncpy(
            m_wcs->ctype[0], lon_type.c_str(), sizeof(m_wcs->ctype[0]) - 1);
        std::strncpy(
            m_wcs->ctype[1], lat_type.c_str(), sizeof(m_wcs->ctype[1]) - 1);

        // Set reference pixels and values element by element
        m_wcs->crpix[0] = crpix1;
        m_wcs->crpix[1] = crpix2;

        // Set reference values
        m_wcs->crval[0] = (crval1 > 180) ? crval1 - 360 : crval1;
        m_wcs->crval[1] = crval2;

        // Set scale factors element by element
        m_wcs->cdelt[0] = cdelt1;
        m_wcs->cdelt[1] = cdelt2;

        // Set rotation using CROTA (alternating linear) instead of PC or CD
        // matrices
        m_wcs->altlin |= 4;
        m_wcs->crota[1] = axis_rot;

        // Set WCS parameters
        int status      = wcsset(m_wcs);
        (void)status; // Avoid unused variable warning
    }

  public:
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

    long const*   naxes() const { return &m_naxes[0]; }
    double const* crval() const { return m_wcs->crval; }
    double const* crpix() const { return m_wcs->crpix; }
    double const* cdelt() const { return m_wcs->cdelt; }
    double const  axis_rot() const { return m_wcs->crota[1]; }

    auto sph2pix(std::array<T, 2> const& ss) const -> std::array<T, 2> {
        double    s1 = ss[0], s2 = ss[1];
        int const ncoords = 1;
        int const nelem   = 2;
        double    imgcrd[2], pixcrd[2];
        double    phi[1], theta[1];
        int       stat[1];

        // WCS projection routines require the input coordinates are in
        // degrees and in the range of [-90,90] for the lat and [-180,180]
        // for the lon. So correct for this effect.
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

        // WCS projection routines require the input coordinates are in
        // degrees and in the range of [-90,90] for the lat and [-180,180]
        // for the lon. So correct for this effect.
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

    auto refpix() const -> std::array<T, 2> {
        return { m_wcs->crpix[0], m_wcs->crpix[1] };
    }

    auto refsph() const -> std::array<T, 2> {
        return { m_wcs->crval[0], m_wcs->crval[1] };
    }

    auto refdir() const -> std::array<T, 3> { return sph2dir(refsph()); }

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

    auto srcpixoff(std::array<T, 3> const& src_dir_coord,
                   std::array<T, 3> const& pix) const -> T {
        return dir_diff(src_dir_coord, pix) * rad2deg;
    }

    // auto refpixoff(std::array<T, 2> const& delta_pix) const -> T {
    //     auto const dir = pix2dir(delta_pix);
    //     auto const ref = pix2dir({ m_wcs->crpix[0], m_wcs->crpix[1] });
    //     return dir_diff(src_dir_coord, pix) * rad2deg;
    // }

    auto wcs_config() -> WcsConfig {
        return {
            .naxes       = {      m_naxes[0],      m_naxes[1], 1ul },
            .crpix       = { m_wcs->crpix[0], m_wcs->crpix[1], 1.0 },
            .crval       = { m_wcs->crval[0], m_wcs->crval[1], 1.0 },
            .cdelt       = { m_wcs->cdelt[0], m_wcs->cdelt[1], 1.0 },
            .axis_rot    = m_wcs->crota[1],
            .proj_name   = m_proj_name,
            .is_galactic = m_is_galactic
        };
    }
};

template <typename T = double>
auto
rebin_wcs_config(WcsConfig orig, double target_pix_size) -> WcsConfig {
    double cdelt1 = orig.cdelt[0] < 0. ? -target_pix_size : target_pix_size;
    double cdelt2 = orig.cdelt[1] < 0. ? -target_pix_size : target_pix_size;
    double scale1 = std::abs(orig.cdelt[0] / cdelt1);
    double scale2 = std::abs(orig.cdelt[1] / cdelt2);
    long   naxis1 = static_cast<long>(std::round(orig.naxes[0] * scale1));
    long   naxis2 = static_cast<long>(std::round(orig.naxes[1] * scale2));
    double crpix1 = orig.crpix[0] * scale1;
    double crpix2 = orig.crpix[1] * scale2;
    return {
        .naxes       = { naxis1, naxis2, orig.naxes[2] },
        .crpix       = { crpix1, crpix2, orig.crpix[2] },
        .crval       = orig.crval,
        .cdelt       = { cdelt1, crpix2, orig.cdelt[2] },
        .axis_rot    = orig.axis_rot,
        .proj_name   = orig.proj_name,
        .is_galactic = orig.is_galactic
    };
};

template <typename T = double>
auto
rebin_skygeom(SkyGeom<T> skygeom, double target_pix_size) -> SkyGeom<T> {
    WcsConfig par = rebin_wcs_config(skygeom.wcs_config(), target_pix_size);
    return { par };
};


} // namespace Fermi
