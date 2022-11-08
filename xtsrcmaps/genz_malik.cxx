#include "xtsrcmaps/genz_malik.hxx"
#include "xtsrcmaps/tensor_types.hxx"

namespace Fermi::Genz
{

constexpr double alpha2
    = 0.35856858280031809199064515390793749545406372969943071; // √(9 / 70)
constexpr double alpha4
    = 0.94868329805051379959966806332981556011586654179756505; // √(9/10)
constexpr double alpha5
    = 0.68824720161168529772162873429362352512689535661564885; // √(9/19)
// constexpr double ratio
//     = 0.14285714285714285714285714285714285714285714285714281; // ⍺₂² / ⍺₄²

auto
fullsym(Tensor1d c, Tensor1d l2, Tensor1d l4, Tensor1d l5) -> Tensor2d
{
    // {c, l2, l4, l5} shape = [2];
    // points shape = [2, 17]
    // k0
    Tensor2d points = c.broadcast(Idx2 { 1, 17 }); //[2, 17]
    // k1
    points(0, 1) -= l2(0);
    points(0, 2) += l2(0);
    points(0, 3) -= l4(0);
    points(0, 4) += l4(0);
    points(1, 5) -= l2(1);
    points(1, 6) += l2(1);
    points(1, 7) -= l4(1);
    points(1, 8) += l4(1);
    // k2
    points(0, 9) -= l4(0);
    points(1, 9) -= l4(1);
    points(0, 10) += l4(0);
    points(1, 10) -= l4(1);
    points(0, 11) -= l4(0);
    points(1, 11) += l4(1);
    points(0, 12) += l4(0);
    points(1, 12) += l4(1);
    // k3
    points(0, 13) -= l5(0);
    points(1, 13) -= l5(1);
    points(0, 14) -= l5(0);
    points(1, 14) += l5(1);
    points(0, 15) += l5(0);
    points(1, 15) -= l5(1);
    points(0, 16) += l5(0);
    points(1, 16) += l5(1);

    return points;
}
// def fullsym(c: NPF, l2: NPF, l4: NPF, l5: NPF) -> NPF:
//
//     p: NPF = full_kn(c, num_points)
//     _, d1, d2 = num_points_full(p.shape[0])
//
//     pts_k0k1(c, l2, l4, p=p[:, 0:d1, ...])
//     pts_k2(c, l4, p=p[:, d1:d2, ...])
//     pts_k6(c, l5, p=p[:, d2:, ...])
//
//     return p

// ### [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
// ### integration Over an N-dimensional rectangular region", updated by Bernstein,
// ### Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
// ### Multiple Integrals"
auto
rule(Tensor1d center, Tensor1d halfwidth, Tensor1d volume) -> auto
{ // std::tuple<double, double, short>{
    Tensor2d points
        = fullsym(center, halfwidth * alpha2, halfwidth * alpha4, halfwidth * alpha5);
}

} // namespace Fermi::Genz
