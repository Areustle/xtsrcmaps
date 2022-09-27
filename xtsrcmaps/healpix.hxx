#pragma once


#include <cstdint>
#include <utility>

namespace Fermi::Healpix
{

// Get the pixel index for a healpix nested ordering map from a spherical coordinate
auto
ang2pix(double const z, double const phi, int64_t const nside_) -> uint64_t;

auto
ang2pix(std::pair<double, double> const ang, int64_t const nside_) -> uint64_t;

// Get the pixel index for a healpix nested ordering map from a spherical coordinate
auto
pix2ang(uint64_t const pix, int64_t const nside_) -> std::pair<double, double>;

} // namespace Fermi::Healpix
