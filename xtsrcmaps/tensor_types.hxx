#pragma once

#include "experimental/mdarray"
#include "experimental/mdspan"

#include "unsupported/Eigen/CXX11/Tensor"

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




using vpd = std::vector<std::pair<double, double>>;

using Eigen::Index;
using Eigen::Sizes;
using Eigen::Tensor;
using Eigen::TensorFixedSize;
using FixTen1d_2 = TensorFixedSize<double, Sizes<2>>;

using Tensor0b   = Tensor<bool, 0>;
using Tensor1b   = Tensor<bool, 1>;
using Tensor2b   = Tensor<bool, 2>;
using Tensor3b   = Tensor<bool, 3>;

using Tensor0d   = Tensor<double, 0>;
using Tensor1d   = Tensor<double, 1>;
using Tensor2d   = Tensor<double, 2>;
using Tensor3d   = Tensor<double, 3>;
using Tensor4d   = Tensor<double, 4>;

using IdxPair    = Eigen::IndexPair<Eigen::Index>;
using Idx1       = Eigen::array<Eigen::Index, 1>;
using Idx2       = Eigen::array<Eigen::Index, 2>;
using Idx3       = Eigen::array<Eigen::Index, 3>;
using Idx4       = Eigen::array<Eigen::Index, 4>;
using IdxPair1   = Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>;
