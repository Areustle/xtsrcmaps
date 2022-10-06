#pragma once

#include "xtsrcmaps/tensor_types.hxx"

#include <vector>

namespace Fermi
{

auto
contract210(mdarray2 const& A, mdarray2 const& B) -> mdarray2;

auto
contract3210(mdarray3 const& A, mdarray2 const& B) -> mdarray3;

auto
mul210(mdarray2 const& A, std::vector<double> const& v) -> mdarray2;

auto
mul310(mdarray3 const& A, std::vector<double> const& v) -> mdarray3;

auto
mul32_1(mdarray3 const& A, mdarray2 const& B) -> mdarray3;

auto
mul322(mdarray3 const& A, mdarray2 const& B) -> mdarray3;

auto
sum2_2(mdarray2 const& A, mdarray2 const& B) -> mdarray2;

auto
sum3_3(mdarray3 const& A, mdarray3 const& B) -> mdarray3;

auto
safe_reciprocal(mdarray2 const& A) -> mdarray2;

} // namespace Fermi
