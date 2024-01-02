#include "xtsrcmaps/sky_geom/sky_geom.hxx"

#include "xtsrcmaps/misc/misc.hxx"

#include <cmath>
#include <string.h>


Fermi::SkyGeom::SkyGeom(Obs::CCubePixels const& pars) {
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

Fermi::SkyGeom::~SkyGeom() { wcsfree(m_wcs); }


auto
Fermi::SkyGeom::sph2pix(Vector2d const& ss) const -> Vector2d {
    double    s1 = ss(0), s2 = ss(1);
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
    wcss2p(m_wcs, ncoords, nelem, worldcrd, phi, theta, imgcrd, pixcrd, stat);
    if (wrap_pos) { pixcrd[0] += 360. / m_wcs->cdelt[0]; }
    if (wrap_neg) { pixcrd[0] -= 360. / m_wcs->cdelt[0]; }
    return { pixcrd[0], pixcrd[1] };
}

auto
Fermi::SkyGeom::sph2pix(std::pair<double, double> const& ss) const
    -> std::pair<double, double> {
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
    wcss2p(m_wcs, ncoords, nelem, worldcrd, phi, theta, imgcrd, pixcrd, stat);
    if (wrap_pos) { pixcrd[0] += 360. / m_wcs->cdelt[0]; }
    if (wrap_neg) { pixcrd[0] -= 360. / m_wcs->cdelt[0]; }
    return { pixcrd[0], pixcrd[1] };
}

auto
Fermi::SkyGeom::pix2sph(Vector2d const& px) const -> Vector2d {
    int    ncoords = 1;
    int    nelem   = 2;
    double worldcrd[2], imgcrd[2];
    double phi[1], theta[1];
    int    stat[1];
    double pixcrd[] = { px(0), px(1) };
    wcsp2s(m_wcs, ncoords, nelem, pixcrd, imgcrd, phi, theta, worldcrd, stat);
    double s1 = worldcrd[0];
    while (s1 < 0) s1 += 360.;
    while (s1 >= 360) s1 -= 360.;
    return { s1, worldcrd[1] };
}

auto
Fermi::SkyGeom::pix2sph(double const first, double const second) const
    -> Vector2d {
    int    ncoords = 1;
    int    nelem   = 2;
    double worldcrd[2], imgcrd[2];
    double phi[1], theta[1];
    int    stat[1];
    double pixcrd[] = { first, second };
    wcsp2s(m_wcs, ncoords, nelem, pixcrd, imgcrd, phi, theta, worldcrd, stat);
    double s1 = worldcrd[0];
    while (s1 < 0) s1 += 360.;
    while (s1 >= 360) s1 -= 360.;
    return { s1, worldcrd[1] };
}

auto
Fermi::SkyGeom::sph2pix(Obs::sphcrd_v_t const& ss) const -> Obs::sphcrd_v_t {
    auto v = std::vector<std::pair<double, double>>(ss.size());
    std::transform(ss.begin(), ss.end(), v.begin(), [&](auto const& x) {
        return sph2pix(x);
    });
    return v;
}

auto
Fermi::SkyGeom::pix2sph(Eigen::Matrix2Xd const& px) const -> Eigen::Matrix2Xd {
    // auto v = std::vector<std::pair<double, double>>(px.size());
    // std::transform(px.begin(), px.end(), v.begin(), [&](auto const& x) {
    //     return pix2sph(x);
    // });
    // return v;
    Eigen::Matrix2Xd S(2, px.cols());
    for (long i = 0; i < px.cols(); ++i) {
        Vector2d v       = px(Eigen::all, i);
        S(Eigen::all, i) = pix2sph(v);
    }
    return S;
}


auto
Fermi::SkyGeom::dir2sph(Vector3d const& dir) const -> Vector2d {
    double ra = atan2(dir(1), dir(0)) * rad2deg;
    // fold RA into the range (0,360)
    while (ra < 0) ra += 360.;
    while (ra > 360) ra -= 360.;
    double dec = asin(dir(2)) * rad2deg;
    return Vector2d(ra, dec);
}

auto
Fermi::SkyGeom::pix2dir(Vector2d const& px) const -> Vector3d {
    auto   s       = pix2sph(px);
    double cos_ra  = cos(s(0) * deg2rad);
    double cos_dec = cos(s(1) * deg2rad);
    double sin_ra  = sin(s(0) * deg2rad);
    double sin_dec = sin(s(1) * deg2rad);
    return Vector3d(cos_ra * cos_dec, sin_ra * cos_dec, sin_dec);
    // auto ra_rad  = s(0) * deg2rad;
    // auto dec_rad = s(1) * deg2rad;
    // return { cos(ra_rad) * cos(dec_rad), sin(ra_rad) * cos(dec_rad),
    // sin(dec_rad) };
}

auto
Fermi::SkyGeom::sph2dir(Vector2d const& s) const -> Vector3d {
    // auto ra_rad  = s(0) * deg2rad;
    // auto dec_rad = s(1) * deg2rad;
    double cos_ra  = cos(s(0) * deg2rad);
    double cos_dec = cos(s(1) * deg2rad);
    double sin_ra  = sin(s(0) * deg2rad);
    double sin_dec = sin(s(1) * deg2rad);
    return Vector3d(cos_ra * cos_dec, sin_ra * cos_dec, sin_dec);
}

auto
Fermi::SkyGeom::sph2dir(std::pair<double, double> const& s) const -> Vector3d {
    double cos_ra  = cos(s.first * deg2rad);
    double cos_dec = cos(s.second * deg2rad);
    double sin_ra  = sin(s.first * deg2rad);
    double sin_dec = sin(s.second * deg2rad);
    return Vector3d(cos_ra * cos_dec, sin_ra * cos_dec, sin_dec);
    // return { cos_ra * cos_dec, sin_ra * cos_dec, sin_dec };
    // auto ra_rad  = s.first * deg2rad;
    // auto dec_rad = s.second * deg2rad;
    // return { cos(ra_rad) * cos(dec_rad), sin(ra_rad) * cos(dec_rad),
    // sin(dec_rad) };
}

auto
Fermi::SkyGeom::srcpixoff(Vector3d const& src_dir_coord,
                          Vector2d const& delta_pix) const -> double {
    auto const dpx = pix2dir(delta_pix);
    return srcpixoff(src_dir_coord, dpx);
}


auto
Fermi::pix_diff(Vector2d const& L, Vector2d const& R, SkyGeom const& skygeom)
    -> double {
    return dir_diff(skygeom.pix2dir(L), skygeom.pix2dir(R));
};

auto
Fermi::sph_pix_diff(std::pair<double, double> const& L,
                    Vector2d const&                  R,
                    SkyGeom const&                   skygeom) -> double {
    return dir_diff(skygeom.sph2dir(L), skygeom.pix2dir(R));
};

auto
Fermi::SkyGeom::srcpixoff(Vector3d const& src_dir_coord,
                          Vector3d const& pix) const -> double {
    // src = sph
    // dpix = pix
    // auto  src  = sph2dir(src_sph_coord);
    // auto const& s0   = std::get<0>(src_dir_coord);
    // auto const& s1   = std::get<1>(src_dir_coord);
    // auto const& s2   = std::get<2>(src_dir_coord);
    // auto const& p0   = std::get<0>(pix);
    // auto const& p1   = std::get<1>(pix);
    // auto const& p2   = std::get<2>(pix);
    // auto        diff = coord3 { s0 - p0, s1 - p1, s2 - p2 };
    // diff
    // auto&       d0   = std::get<0>(diff);
    // auto&       d1   = std::get<1>(diff);
    // auto&       d2   = std::get<2>(diff);
    //
    // double mag       = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    //
    // // double x  = 0.5 * (m_dir - other.dir()).mag();
    // // return 2. * asin(x);
    // return 2. * asin(0.5 * mag) * rad2deg;
    return dir_diff(src_dir_coord, pix) * rad2deg;
}
