#pragma once

#include "unsupported/Eigen/CXX11/Tensor"

using vpd    = std::vector<std::pair<double, double>>;

using Eigen::Index;
using Eigen::Sizes;
using Eigen::Tensor;
using Eigen::TensorFixedSize;
using FixTen1d_2 = TensorFixedSize<double, Sizes<2>>;

using Eigen::Array2Xd;
using Eigen::Array3Xd;
using Eigen::ArrayX2d;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

using Matrix23d = Eigen::Matrix<double, 2, 3>;

template <typename T>
using Map = Eigen::Map<T>;
template <typename T>
using TensorMap   = Eigen::TensorMap<T>;

using Tensor0b    = Tensor<bool, 0>;
using Tensor1b    = Tensor<bool, 1>;
using Tensor2b    = Tensor<bool, 2>;
using Tensor3b    = Tensor<bool, 3>;
using Tensor0d    = Tensor<double, 0>;
using Tensor1d    = Tensor<double, 1>;
using Tensor2d    = Tensor<double, 2>;
using Tensor3d    = Tensor<double, 3>;
using Tensor4d    = Tensor<double, 4>;
using Tensor0f    = Tensor<float, 0>;
using Tensor1f    = Tensor<float, 1>;
using Tensor2f    = Tensor<float, 2>;
using Tensor3f    = Tensor<float, 3>;
using Tensor4f    = Tensor<float, 4>;

using Tensor1idx  = Tensor<Eigen::Index, 1>;
using Tensor1byt  = Tensor<unsigned char, 1>;

using Tensor0i    = Tensor<Eigen::DenseIndex, 0>;
using Tensor1i    = Tensor<Eigen::DenseIndex, 1>;
using Tensor2i    = Tensor<Eigen::DenseIndex, 2>;
using Tensor3i    = Tensor<Eigen::DenseIndex, 3>;
using Tensor4i    = Tensor<Eigen::DenseIndex, 4>;

using IdxPair     = Eigen::IndexPair<Eigen::Index>;
using Idx1        = Eigen::array<Eigen::Index, 1>;
using Idx2        = Eigen::array<Eigen::Index, 2>;
using Idx3        = Eigen::array<Eigen::Index, 3>;
using Idx4        = Eigen::array<Eigen::Index, 4>;
using IdxPair1    = Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>;

using IdxVec      = Eigen::VectorX<Eigen::DenseIndex>;

using RowTensor1d = Eigen::Tensor<double, 1, Eigen::RowMajor>;
using RowTensor2d = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using RowTensor3d = Eigen::Tensor<double, 3, Eigen::RowMajor>;
using RowTensor1f = Eigen::Tensor<float, 1, Eigen::RowMajor>;
using RowTensor2f = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using RowTensor3f = Eigen::Tensor<float, 3, Eigen::RowMajor>;
