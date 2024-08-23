#include "xtsrcmaps/healpix/healpix.hxx"

#include "xtsrcmaps/misc/misc.hxx"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <tuple>


constexpr uint16_t utab[] = {
#define Z(a) 0x##a##0, 0x##a##1, 0x##a##4, 0x##a##5
#define Y(a) Z(a##0), Z(a##1), Z(a##4), Z(a##5)
#define X(a) Y(a##0), Y(a##1), Y(a##4), Y(a##5)
    X(0), X(1), X(4), X(5)
#undef X
#undef Y
#undef Z
};

constexpr uint16_t ctab[] = {
#define Z(a) a, a + 1, a + 256, a + 257
#define Y(a) Z(a), Z(a + 2), Z(a + 512), Z(a + 514)
#define X(a) Y(a), Y(a + 4), Y(a + 1024), Y(a + 1028)
    X(0), X(8), X(2048), X(2056)
#undef X
#undef Y
#undef Z
};

const int64_t jrll[] = { 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 },
              jpll[] = { 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7 };

#ifndef __BMI2__

inline auto
spread_bits(int64_t const v) -> int64_t {
    return int64_t(utab[v & 0xff]) | (int64_t(utab[(v >> 8) & 0xff]) << 16)
           | (int64_t(utab[(v >> 16) & 0xff]) << 32)
           | (int64_t(utab[(v >> 24) & 0xff]) << 48);
}

inline int64_t
compress_bits(int64_t const v) {
    int64_t raw = v & 0x5555555555555555ull;
    raw |= raw >> 15;
    return ctab[raw & 0xff] | (ctab[(raw >> 8) & 0xff] << 4)
           | (ctab[(raw >> 32) & 0xff] << 16)
           | (ctab[(raw >> 40) & 0xff] << 20);
}

#else

#include <x86intrin.h>

inline int64_t
spread_bits(int64_t v) {
    return _pdep_u64(v, 0x5555555555555555ull);
}

inline int64_t
compress_bits(int64_t v) {
    return _pext_u64(v, 0x5555555555555555ull);
}

#endif

inline auto
nest2xyf(int64_t const pix,
         int64_t const order_,
         int64_t const npface_) -> std::tuple<int64_t, int64_t, int64_t> {
    int64_t face_num = pix >> (2 * order_);
    int64_t tmp      = pix & (npface_ - 1);
    int64_t ix       = compress_bits(tmp);
    int64_t iy       = compress_bits(tmp >> 1);
    return { ix, iy, face_num };
}


inline auto
xyf2nest(uint64_t const ix,
         uint64_t const iy,
         uint64_t const face_num,
         uint64_t const order_) -> uint64_t {
    return (uint64_t(face_num) << (2 * order_)) + spread_bits(ix)
           + (spread_bits(iy) << 1);
}


// Returns the remainder of the division v1/v2.
// The result is non-negative.
// v1 can be positive or negative; v2 must be positive.
inline auto
fmodulo(double const v1, double const v2) -> double {
    if (v1 >= 0) return (v1 < v2) ? v1 : std::fmod(v1, v2);
    double tmp = std::fmod(v1, v2) + v2;
    return (tmp == v2) ? 0. : tmp;
}


/* Returns the largest integer n that fulfills 2^n<=arg. */
inline auto
ilog2(int64_t arg) -> int64_t {
#ifdef __GNUC__
    if (arg == 0) return 0;
    return 8 * sizeof(uint64_t) - 1 - __builtin_clzll(arg);
#endif
    int64_t res = 0;
    while (arg > 0xFFFF) {
        res += 16;
        arg >>= 16;
    }
    if (arg > 0x00FF) {
        res |= 8;
        arg >>= 8;
    }
    if (arg > 0x000F) {
        res |= 4;
        arg >>= 4;
    }
    if (arg > 0x0003) {
        res |= 2;
        arg >>= 2;
    }
    if (arg > 0x0001) { res |= 1; }
    return res;
}

inline auto
nside2order(int64_t const nside) -> int64_t {
    assert(nside > int64_t(0));
    return ((nside) & (nside - 1)) ? -1 : ilog2(nside);
}

// Get the angle pair for a healpix nested ordering map from a pixel index
auto
Fermi::Healpix::pix2ang(uint64_t const pix,
                        int64_t const  nside_) -> std::pair<double, double> {
    int64_t const npface_   = nside_ * nside_;
    int64_t const npix_     = 12 * npface_;
    double const  fact2_    = 4. / npix_;
    double const  fact1_    = (nside_ << 1) * fact2_;
    int64_t const order_    = nside2order(nside_);
    auto [ix, iy, face_num] = nest2xyf(pix, order_, npface_);

    int64_t jjr             = (jrll[face_num] << order_) - ix - iy - 1;

    int64_t r               = 0;
    double  z = 0.0, sintheta = 0.0;
    bool    have_sth = false;
    if (jjr < nside_) {
        r          = jjr;
        double tmp = (r * r) * fact2_;
        z          = 1 - tmp;
        if (z > 0.99) {
            sintheta = sqrt(tmp * (2. - tmp));
            have_sth = true;
        }
    } else if (jjr > 3 * nside_) {
        r          = nside_ * 4 - jjr;
        double tmp = (r * r) * fact2_;
        z          = tmp - 1;
        if (z < -0.99) {
            sintheta = sqrt(tmp * (2. - tmp));
            have_sth = true;
        }
    } else {
        r = nside_;
        z = (2 * nside_ - jjr) * fact1_;
    }

    int64_t t = jpll[face_num] * r + ix - iy;
    assert(t < 8 * r);
    if (t < 0) t += 8 * r;
    double phi
        = (r == nside_) ? 0.75 * halfpi * t * fact1_ : (0.5 * halfpi * t) / r;

    return { have_sth ? std::atan2(sintheta, z) : std::acos(z), phi };
}

// Get the pixel index for a healpix nested ordering map from an angle
auto
Fermi::Healpix::ang2pix(double const  theta,
                        double const  phi,
                        int64_t const nside_) -> uint64_t {

    bool const     have_sth = ((theta < 0.01) || (theta > pi - 0.01));
    double const   sth      = have_sth ? std::sin(theta) : 0.;
    double const   z        = std::cos(theta);
    double const   za       = std::abs(z);
    double const   tt       = fmodulo(phi * inv_halfpi, 4.0); // in [0,4)
    uint64_t const order_   = nside2order(nside_);
    {
        if (za <= twothird) // Equatorial region
        {
            double   temp1 = nside_ * (0.5 + tt);
            double   temp2 = nside_ * (z * 0.75);
            uint64_t jp
                = uint64_t(temp1 - temp2); // index of  ascending edge line
            uint64_t jm
                = uint64_t(temp1 + temp2); // index of descending edge line
            uint64_t ifp = jp >> order_;   // in {0,4}
            uint64_t ifm = jm >> order_;
            uint64_t face_num
                = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));

            uint64_t ix = jm & (nside_ - 1);
            uint64_t iy = nside_ - (jp & (nside_ - 1)) - 1;
            return xyf2nest(ix, iy, face_num, order_);
        } else /* polar region, za > 2/3 */
        {
            uint64_t ntt = std::min(3_u64, uint64_t(tt));
            double   tp  = tt - ntt;
            double   tmp = ((za < 0.99) || (!have_sth))
                               ? nside_ * sqrt(3 * (1 - za))
                               : nside_ * sth / sqrt((1. + za) / 3.);

            int64_t jp   = int64_t(tp * tmp); // increasing edge line index
            int64_t jm
                = int64_t((1.0 - tp) * tmp); // decreasing edge line index
            jp = std::min(jp,
                          nside_ - 1); // for points too close to the boundary
            jm = std::min(jm, nside_ - 1);
            return (z >= 0)
                       ? xyf2nest(nside_ - jm - 1, nside_ - jp - 1, ntt, order_)
                       : xyf2nest(jp, jm, ntt + 8, order_);
        }
    }
}

auto
Fermi::Healpix::ang2pix(std::pair<double, double> const ang,
                        int64_t const                   nside_) -> uint64_t {
    return ang2pix(ang.first, ang.second, nside_);
}

/* auto */
/* Fermi::Healpix::ang2pix(  */
/*     std::vector<std::pair<double, double>> const& angs, */
/*     int64_t const nside_) -> std::vector<uint64_t> { */
/*     auto pixs = std::vector<uint64_t>(angs.size(), 0); */
/*     std::transform(angs.begin(), angs.end(), pixs.begin(), [&nside_](auto p)
 * { */
/*         return ang2pix(p, nside_); */
/*     }); */
/*     return pixs; */
/* } */
