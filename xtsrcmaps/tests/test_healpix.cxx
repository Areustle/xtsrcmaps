#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include <cmath>

#include "fmt/format.h"

#include "xtsrcmaps/healpix.hxx"


TEST_CASE("Fermi HEALPix pix2pix Test")
{

    for (int64_t i = 0; i < 21; ++i)
    {
        int64_t const nside = 1ull << i;

        auto f              = [nside](uint64_t const pix) {
            return Fermi::Healpix::ang2pix(Fermi::Healpix::pix2ang(pix, nside), nside);
        };

        for (int64_t pix = 1; pix <= nside + 1; ++pix)
        {
            REQUIRE_MESSAGE(pix == f(pix), fmt::format("i: {} nside: {}", i, nside));
        }
    }
}

// // constexpr double twopi      = 6.283185307179586476925286766559005768394;
//
// TEST_CASE("Fermi HEALPix ilog2 Test")
// {
//     for (uint64_t n = 0; n < 16; ++n)
//     {
//         for (uint64_t i = (1ull << n); i < (2ull << n); ++i) { REQUIRE(ilog2(i) ==
//         n); }
//     }
//     for (uint64_t n = 16; n < 63; ++n) { REQUIRE(ilog2(1ull << n) == n); }
// }
//
// TEST_CASE("Fermi HEALPix nside2order Test")
// {
//     for (uint64_t n = 1; n < 29; ++n) { REQUIRE(nside2order(1ull << n) == n); }
// }
//
// TEST_CASE("Fermi HEALPix nest2xyf Test")
// {
//     for (uint64_t n = 1; n < 8; ++n)
//     {
//         int64_t const nside_  = 1ull << n;
//         int64_t const npface_ = nside_ * nside_;
//         int64_t const npix_   = 12 * npface_;
//         int64_t const order_  = nside2order(nside_);
//
//         for (int64_t i = 0; i < npix_; ++i)
//         {
//             auto [x, y, f] = nest2xyf(i, order_, npface_);
//             REQUIRE(x >= 0);
//             REQUIRE(x < nside_);
//             REQUIRE(y >= 0);
//             REQUIRE(y < nside_);
//             REQUIRE(f >= i >> (2 * order_));
//         }
//     }
// }
