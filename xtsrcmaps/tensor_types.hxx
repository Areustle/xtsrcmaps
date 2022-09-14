#pragma once

#include "experimental/mdarray"
#include "experimental/mdspan"

using dyext1   = std::experimental::extents<uint64_t, std::dynamic_extent>;
using dyext2   = std::experimental::extents<uint64_t,
                                          std::dynamic_extent, //
                                          std::dynamic_extent>;
using dyext3   = std::experimental::extents<uint64_t,
                                          std::dynamic_extent, //
                                          std::dynamic_extent,
                                          std::dynamic_extent>;
using dyext4   = std::experimental::extents<uint64_t,
                                          std::dynamic_extent,
                                          std::dynamic_extent,
                                          std::dynamic_extent,
                                          std::dynamic_extent>;

// mdarray
using mdarray1 = std::experimental::mdarray<double, dyext1>;
using mdarray2 = std::experimental::mdarray<double, dyext2>;
using mdarray3 = std::experimental::mdarray<double, dyext3>;
using mdarray4 = std::experimental::mdarray<double, dyext4>;
// mdspan
using mdspan1  = std::experimental::mdspan<double, dyext1>;
using mdspan2  = std::experimental::mdspan<double, dyext2>;
using mdspan3  = std::experimental::mdspan<double, dyext3>;
using mdspan4  = std::experimental::mdspan<double, dyext4>;
