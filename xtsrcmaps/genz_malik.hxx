#pragma once

#include <iostream>

#include "xtsrcmaps/tensor_types.hxx"

#include "unsupported/Eigen/CXX11/Tensor"

namespace Fermi::Genz
{

// The number of sample points within a 2D region in a Genz_Malik rule.
constexpr long Ncnt = 17;

constexpr double alpha2
    = 0.35856858280031809199064515390793749545406372969943071; // √(9 / 70)
constexpr double alpha4
    = 0.94868329805051379959966806332981556011586654179756505; // √(9/10)
constexpr double alpha5
    = 0.68824720161168529772162873429362352512689535661564885; // √(9/19)
constexpr double ratio
    = 0.14285714285714285714285714285714285714285714285714281; // ⍺₂² / ⍺₄²

// [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
// integration Over an N-dimensional rectangular region", updated by Bernstein,
// Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
// Multiple Integrals"

constexpr std::array<double, 5> genz_malik_weights
    = { (12824. - 9120. * 2. + 400. * 2. * 2.) / 19683.,
        980. / 6561.,
        (1820. - 400. * 2.) / 19683.,
        200. / 19683.,
        (6859. / 19683.) / 4. };

constexpr std::array<double, 17> genz_malik_weights_17 = {
    (12824. - 9120. * 2. + 400. * 4.) / 19683., // 0
    980. / 6561.,                               // 1
    980. / 6561.,                               // 5
    980. / 6561.,                               // 2
    980. / 6561.,                               // 6
    (1820. - 400. * 2.) / 19683.,               // 3
    (1820. - 400. * 2.) / 19683.,               // 4
    (1820. - 400. * 2.) / 19683.,               // 7
    (1820. - 400. * 2.) / 19683.,               // 8
    200. / 19683.,                              // 9
    200. / 19683.,                              // a
    200. / 19683.,                              // b
    200. / 19683.,                              // c
    (6859. / 19683.) / 4.,                      // d
    (6859. / 19683.) / 4.,                      // e
    (6859. / 19683.) / 4.,                      // f
    (6859. / 19683.) / 4.,                      // 10
};

constexpr std::array<double, 4> genz_malik_err_weights
    = { (729.0 - 950.0 * (2.) + 50.0 * (2.) * (2.)) / 729.0,
        245. / 486.,
        (265.0 - 100.0 * (2.)) / 1458.0,
        25. / 729. };

constexpr std::array<double, 17> genz_malik_err_weights_17 = {
    (729.0 - (950.0 * 2.) + 50.0 * 4.) / 729., // 0
    245. / 486.,                               // 1
    245. / 486.,                               // 2
    245. / 486.,                               // 3
    245. / 486.,                               // 4
    (265.0 - 100.0 * 2.) / 1458.0,             // 5
    (265.0 - 100.0 * 2.) / 1458.0,             // 6
    (265.0 - 100.0 * 2.) / 1458.0,             // 7
    (265.0 - 100.0 * 2.) / 1458.0,             // 8
    25. / 729.,                                // 9
    25. / 729.,                                // a
    25. / 729.,                                // b
    25. / 729.,                                // c
    0.,                                        // d
    0.,                                        // e
    0.,                                        // f
    0.,                                        // 10
};


auto
fullsym(Tensor2d const& c, Tensor2d const& l2, Tensor2d const& l4, Tensor2d const& l5)
    -> Tensor3d;

auto
region(Tensor3d const& low, Tensor3d const& high, long const Nevts)
    -> std::tuple<Tensor2d, Tensor2d, Tensor2d>;


auto
result_err(Tensor3d const& vals) -> std::tuple<Tensor2d, Tensor2d>;

// [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
// integration Over an N-dimensional rectangular region", updated by Bernstein,
// Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
// Multiple Integrals"
auto
result_err(Tensor3d const& vals, Tensor2d const& volume)
    -> std::tuple<Tensor2d, Tensor2d>;


// [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
// integration Over an N-dimensional rectangular region", updated by Bernstein,
// Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
// Multiple Integrals"
template <typename F>
auto
rule(F&& f, Array3Xd const& points, Tensor2d const& halfwidth, Tensor2d const& volume)
    // -> Tensor2d
    -> std::tuple<Tensor2d, Tensor2d>
{
    // // dim = p.shape[0] //2
    // // d1 = points.num_k0k1(dim) // 9
    // // d2 = points.num_k2(dim) // 4
    // // d3 = d1 + d2 // 13

    // // # vals shape [ range_dim, points, Nevts ]
    Tensor3d const vals = f(points); // [Ne, 17, Nevts]
    // return vals;
    long const Ne       = vals.dimension(0);
    // long const Nevts = vals.dimension(2);

    // // vc = vals[:, 0:1, Nevts]  # center integrand value. shape = [ rdim, 1, Nevts ]
    // // // Tensor2d [Ne, 1, Nevts]
    // Tensor3d vc      = vals.slice(Idx3 { 0, 0, 0 }, Idx3 { Ne, 1, Nevts });
    // // # [ range_dim, domain_dim, Nevts ]
    // // v01 = vals[:, 1:d1:4, Nevts] + vals[:, 2:d1:4, Nevts]
    // // v23 = vals[:, 3:d1:4, Nevts] + vals[:, 4:d1:4, Nevts]
    // Tensor3d v01
    //     = vals.slice(Idx3 { 0, 1, 0 }, Idx3 { Ne, 8, Nevts }).stride(Idx3 { 1, 4, 1
    //     })
    //       + vals.slice(Idx3 { 0, 2, 0 }, Idx3 { Ne, 8, Nevts })
    //             .stride(Idx3 { 1, 4, 1 });
    // Tensor3d v23
    //     = vals.slice(Idx3 { 0, 3, 0 }, Idx3 { Ne, 8, Nevts }).stride(Idx3 { 1, 4, 1
    //     })
    //       + vals.slice(Idx3 { 0, 4, 0 }, Idx3 { Ne, 8, Nevts })
    //             .stride(Idx3 { 1, 4, 1 });
    //
    // // // # Compute the 4th divided difference to determine dimension on which to
    // split.
    // // Tensor1d diff
    // //     = ((v01 - 2. * vc.reshape(Idx2 { Ne, 1 }).broadcast(Idx2 { 1, 2 }))
    // //        - ratio * (v23 - 2. * vc.reshape(Idx2 { Ne, 1 }).broadcast(Idx2 { 1, 2
    // //        })))
    // //           .abs()
    // //           .sum(Idx1 { 0 });
    // //
    // // // s2 = np.sum(v01, axis=1)  # [ range_dim, ... ]
    // // // s3 = np.sum(v23, axis=1)  # [ range_dim, ... ]
    // // // s4 = np.sum(vals[:, d1:d3, ...], axis=1)  # [ range_dim, ... ]
    // // // s5 = np.sum(vals[:, d3:, ...], axis=1)  # [ range_dim, ... ]
    // auto s2 = v01.sum(Idx1 { 1 });
    // auto s3 = v23.sum(Idx1 { 1 });
    // auto s4 = vals.slice(Idx3 { 0, 9, 0 }, Idx3 { Ne, 4, Nevts }).sum(Idx1 { 1 });
    // auto s5 = vals.slice(Idx3 { 0, 13, 0 }, Idx3 { Ne, 4, Nevts }).sum(Idx1 { 1 });

    // w = genz_malik_weights(dim)  # [5]
    // wE = genz_malik_err_weights(dim)  # [4]
    TensorMap<Tensor1d const> const w(genz_malik_weights_17.data(), 17);
    TensorMap<Tensor1d const> const wE(genz_malik_err_weights_17.data(), 17);


    // # [5] . [5,range_dim, ... ] = [range_dim, ... ]
    // result = volumes * np.tensordot(w, (vc, s2, s3, s4, s5), (0, 0))
    // auto result
    //     = volume(0) * (w(0) * vc + w(1) * s2 + w(2) * s3 + w(3) * s4 + w(4) * s5);
    Tensor2d result = volume.broadcast(Idx2 { Ne, 1 })
                      * w.contract(vals, IdxPair1 { { { 0, 1 } } });
    // // # [4] . [4,range_dim, ... ] = [range_dim, ... ]
    // // res5th = volumes * np.tensordot(wE, (vc, s2, s3, s4), (0, 0))
    // Tensor3d const& val5th = vals.slice(Idx3 { 0, 0, 0 }, Idx3 { Ne, 13, Nevts });
    Tensor2d res5th = wE.contract(vals, IdxPair1 { { { 0, 1 } } });

    // // err = np.abs(res5th - result)  # [range_dim, ... ]
    Tensor2d err    = (res5th - result).abs(); //  # [range_dim, ... ]
    //
    // // # determine split dimension
    // Tensor<long, 1> split_dim  = diff.argmax().reshape(Idx1 { 1 });
    // Tensor<long, 1> widest_dim = halfwidth.argmax().reshape(Idx1 { 1 });
    //
    // // df = np.sum(err, axis=0) / (volumes * 10 ** dim)  # [ ... ]
    // Tensor1d df
    //     = err.sum(Idx1 { 0 }).reshape(Idx1 { 1 }) / (100 * volume); //  # [ ... ]
    // // delta = np.reshape(diff[split_i] - diff[widest_i], diff.shape[1:])  # [ ... ]
    // auto delta = diff(split_dim(0)) - diff(widest_dim(0));
    // // too_close = delta <= df
    // split_dim  = (df > delta).select(split_dim, widest_dim);
    // return { result, err, split_dim };
    return { result, err };
}

template <typename F>
auto
rule(F&& f, Tensor2d const& center, Tensor2d const& halfwidth, Tensor2d const& volume)
    -> Tensor3d
{
    Tensor3d const points
        = fullsym(center, halfwidth * alpha2, halfwidth * alpha4, halfwidth * alpha5);

    return rule(f, points, halfwidth, volume);
}

auto
converged_indices(Tensor2d const& value,
                  Tensor2d const& error,
                  double const    ftol_threshold)
    -> std::tuple<std::vector<long>, std::vector<long>>;

auto
split_dims(Tensor3d const&          evals,
           Tensor2d const&          err,
           Tensor2d const&          hwUcnv,
           Tensor2d const&          volUcnv,
           std::vector<long> const& not_converged) -> Tensor1byt;

auto
region_split(Tensor2d&         center,
             Tensor2d&         halfwidth,
             Tensor2d&         volume,
             Tensor1byt const& split_dim,
             Tensor2d const&   cenUcv,
             Tensor2d const&   hwUcv,
             Tensor2d const&   volUcv) -> void;

// def split(centers: NPF, halfwidth: NPF, volumes: NPF, split_dim: NPI):
//
//     # centers.shape   [ domain_dim, regions_events ]
//     # split_dim.shape [ 1, regions_events ]
//
//     if np.amin(split_dim) < 0 or np.amax(split_dim) >= (centers.shape[0]):
//         raise IndexError("split dimension invalid")
//
//     if split_dim.ndim < centers.ndim:
//         split_dim = np.expand_dims(split_dim, 0)
//
//     ## {center, hwidth}  [ domain_dim, (regions, events) ]
//
//     mask = np.zeros_like(centers, dtype=np.bool_)
//     np.put_along_axis(mask, split_dim, True, 0)
//
//     h = np.copy(halfwidth)
//     h[mask] *= 0.5
//
//     v = np.copy(volumes)
//     v *= 0.5
//
//     c1 = np.copy(centers)
//     c2 = np.copy(centers)
//     c1[mask] -= h[mask]
//     c2[mask] += h[mask]
//
//     c = np.concatenate((c1, c2), axis=1)
//     h = np.concatenate((h, h), axis=1)
//     v = np.concatenate((v, v), axis=0)
//
//     return c, h, v
//

} // namespace Fermi::Genz
