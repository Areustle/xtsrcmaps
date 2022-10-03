#pragma once

#include "xtsrcmaps/tensor_types.hxx"

#include <vector>

namespace Fermi
{

auto
contract210(mdarray2 const& A, mdarray2 const& B) -> mdarray2;

auto
mul210(mdarray2 const& A, std::vector<double> const& v) -> mdarray2;

auto
sum2_2(mdarray2 const& A, mdarray2 const& B) -> mdarray2;

auto
safe_reciprocal(mdarray2 const& A) -> mdarray2;

} // namespace Fermi
