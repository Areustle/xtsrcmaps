#include "xtsrcmaps/genz_malik.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include <iostream>

namespace Fermi::Genz
{

auto
fullsym(Tensor2d const& c, Tensor2d const& l2, Tensor2d const& l4, Tensor2d const& l5)
    -> Tensor3d
{
    // {c, l2, l4, l5} shape = [2];
    // points shape = [2, 17]
    // k0
    long     Nevts = c.dimension(1);
    Tensor3d points(2, 17, Nevts);
    points = c.reshape(Idx3 { 2, 1, Nevts }).broadcast(Idx3 { 1, 17, 1 });
    // k1
    points.slice(Idx3 { 0, 1, 0 }, Idx3 { 1, 1, Nevts })
        -= l2.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 2, 0 }, Idx3 { 1, 1, Nevts })
        -= l2.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 0, 3, 0 }, Idx3 { 1, 1, Nevts })
        += l2.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 4, 0 }, Idx3 { 1, 1, Nevts })
        += l2.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });

    points.slice(Idx3 { 0, 5, 0 }, Idx3 { 1, 1, Nevts })
        -= l4.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 6, 0 }, Idx3 { 1, 1, Nevts })
        -= l4.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 0, 7, 0 }, Idx3 { 1, 1, Nevts })
        += l4.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 8, 0 }, Idx3 { 1, 1, Nevts })
        += l4.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    // k2
    points.slice(Idx3 { 0, 9, 0 }, Idx3 { 1, 1, Nevts })
        -= l4.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 9, 0 }, Idx3 { 1, 1, Nevts })
        -= l4.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 0, 10, 0 }, Idx3 { 1, 1, Nevts })
        += l4.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 10, 0 }, Idx3 { 1, 1, Nevts })
        -= l4.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 0, 11, 0 }, Idx3 { 1, 1, Nevts })
        -= l4.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 11, 0 }, Idx3 { 1, 1, Nevts })
        += l4.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 0, 12, 0 }, Idx3 { 1, 1, Nevts })
        += l4.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 12, 0 }, Idx3 { 1, 1, Nevts })
        += l4.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    // k3
    points.slice(Idx3 { 0, 13, 0 }, Idx3 { 1, 1, Nevts })
        -= l5.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 13, 0 }, Idx3 { 1, 1, Nevts })
        -= l5.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 0, 14, 0 }, Idx3 { 1, 1, Nevts })
        -= l5.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 14, 0 }, Idx3 { 1, 1, Nevts })
        += l5.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 0, 15, 0 }, Idx3 { 1, 1, Nevts })
        += l5.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 15, 0 }, Idx3 { 1, 1, Nevts })
        -= l5.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 0, 16, 0 }, Idx3 { 1, 1, Nevts })
        += l5.slice(Idx2 { 0, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    points.slice(Idx3 { 1, 16, 0 }, Idx3 { 1, 1, Nevts })
        += l5.slice(Idx2 { 1, 0 }, Idx2 { 1, Nevts }).reshape(Idx3 { 1, 1, Nevts });
    //
    return points;
}

auto
region(Tensor3d const& low, Tensor3d const& high, long const Nevts)
    -> std::tuple<Tensor2d, Tensor2d, Tensor2d>
{
    Tensor2d center    = ((high + low) * 0.5).reshape(Idx2 { 2, Nevts });
    Tensor2d halfwidth = ((high - low) * 0.5).reshape(Idx2 { 2, Nevts });
    Tensor2d volume    = (2. * halfwidth).prod(Idx1 { 0 }).reshape(Idx2 { 1, Nevts });
    return { center, halfwidth, volume };
}

// Result and Error computation in the special case where the full set of volumes is 1.
// This is the case in the initial run for every pixel and every souce.
auto
result_err(Tensor3d const& vals) -> std::tuple<Tensor2d, Tensor2d>
{
    TensorMap<Tensor1d const> const w(genz_malik_weights_17.data(), 17);
    TensorMap<Tensor1d const> const wE(genz_malik_err_weights_17.data(), 17);

    // # [5] . [5,range_dim, ... ] = [range_dim, ... ]
    Tensor2d result = w.contract(vals, IdxPair1 { { { 0, 1 } } });
    // # [4] . [4,range_dim, ... ] = [range_dim, ... ]
    Tensor2d res5th = wE.contract(vals, IdxPair1 { { { 0, 1 } } });

    // err = np.abs(res5th - result)  # [range_dim, ... ]
    Tensor2d err    = (res5th - result).abs(); //  # [range_dim, ... ]
    return { result, err };
}

auto
result_err(Tensor3d const& vals, Tensor2d const& volume)
    -> std::tuple<Tensor2d, Tensor2d>
{
    TensorMap<Tensor1d const> const w(genz_malik_weights_17.data(), 17);
    TensorMap<Tensor1d const> const wE(genz_malik_err_weights_17.data(), 17);

    // # [5] . [5,range_dim, ... ] = [range_dim, ... ]
    Tensor2d result = volume.broadcast(Idx2 { vals.dimension(0), 1 })
                      * w.contract(vals, IdxPair1 { { { 0, 1 } } });
    // # [4] . [4,range_dim, ... ] = [range_dim, ... ]
    Tensor2d res5th = volume.broadcast(Idx2 { vals.dimension(0), 1 })
                      * wE.contract(vals, IdxPair1 { { { 0, 1 } } });

    // err = np.abs(res5th - result)  # [range_dim, ... ]
    Tensor2d err = (res5th - result).abs(); //  # [range_dim, ... ]
    return { result, err };
}

auto
converged_indices(Tensor2d const& value,
                  Tensor2d const& error,
                  double const    ftol_threshold)
    -> std::tuple<std::vector<long>, std::vector<long>>
{
    Tensor1d const abserr = error.abs().maximum(Idx1 { 0 });
    Tensor1d const relerr = (error / value.abs()).maximum(Idx1 { 0 });
    // std::cout << err.reshape(Idx2 { 100, 100 }).slice(Idx2 { 0, 0 }, Idx2 { 45, 30 })
    //           << std::endl;

    auto converged        = std::vector<long> {};
    auto not_converged    = std::vector<long> {};

    for (long i = 0; i < relerr.size(); ++i)
    {
        if (relerr(i) > ftol_threshold && abserr(i) > ftol_threshold)
        {
            not_converged.push_back(i);
        }
        else { converged.push_back(i); }
    }

    return { converged, not_converged };
}


auto
split_dims(Tensor3d const&          evals,
           Tensor2d const&          error,
           Tensor2d const&          hwUcnv,
           Tensor2d const&          volUcnv,
           std::vector<long> const& not_converged) -> Tensor1byt
{
    long const Ne    = evals.dimension(0);
    long const Nevts = evals.dimension(2);
    long const Nucnv = not_converged.size();

    // SPLIT DIM
    Tensor2d diff(2, Nucnv);

    /* BLOCK */
    {
        Tensor3d evalUcnv(Ne, Ncnt, Nucnv);
        Map<MatrixXd>(evalUcnv.data(), Ne * Ncnt, Nucnv) = Map<MatrixXd const>(
            evals.data(), Ne * Ncnt, Nevts)(Eigen::all, not_converged);
        Tensor3d vc  = 2. * evalUcnv.slice(Idx3 { 0, 0, 0 }, Idx3 { Ne, 1, Nucnv });

        // # [ range_dim, domain_dim, ... ]
        Tensor3d v01 = evalUcnv.slice(Idx3 { 0, 1, 0 }, Idx3 { Ne, 2, Nucnv })
                       + evalUcnv.slice(Idx3 { 0, 3, 0 }, Idx3 { Ne, 2, Nucnv });
        Tensor3d v23 = evalUcnv.slice(Idx3 { 0, 5, 0 }, Idx3 { Ne, 2, Nucnv })
                       + evalUcnv.slice(Idx3 { 0, 7, 0 }, Idx3 { Ne, 2, Nucnv });

        // # Compute the 4th divided difference to determine dimension on which to
        // split.

        diff = ((v01 - vc.broadcast(Idx3 { 1, 2, 1 }))
                - ratio * (v23 - vc.broadcast(Idx3 { 1, 2, 1 })))
                   .abs()
                   .sum(Idx1 { 0 });
    }

    Tensor2d errUcnv(Ne, Nucnv);
    Map<MatrixXd>(errUcnv.data(), Ne, Nucnv)
        = Map<MatrixXd const>(error.data(), Ne, Nevts)(Eigen::all, not_converged);

    Tensor1d df = (errUcnv.sum(Idx1 { 0 }) / (100. * volUcnv).reshape(Idx1 { Nucnv }));

    Tensor1byt zeros(Nucnv);
    Tensor1byt ones(Nucnv);
    zeros.setZero();
    ones.setConstant(1);

    Tensor1byt split_dim
        = (diff.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
           >= diff.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }))
              .select(zeros, ones);

    Tensor1byt widest_dim
        = (hwUcnv.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
           >= hwUcnv.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }))
              .select(zeros, ones);

    // std::cout << "split  " << split_dim.reshape(Idx2 { 1, Nucnv }) << std::endl
    //           << std::endl;
    // std::cout << "widest " << widest_dim.reshape(Idx2 { 1, Nucnv }) << std::endl
    //           << std::endl;
    //
    // delta = np.reshape(diff[split_i] - diff[widest_i], diff.shape[1:])  # [ ... ]

    Tensor1d delta
        = split_dim.select(
              diff.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }),
              diff.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }))
          - widest_dim.select(
              diff.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }),
              diff.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }));

    // too_close = delta <= df
    // split_dim[too_close] = widest_dim[too_close]
    split_dim = (delta <= df).select(split_dim, widest_dim);

    return split_dim;
}

auto
region_split(Tensor2d&         center,
             Tensor2d&         halfwidth,
             Tensor2d&         volume,
             Tensor1byt const& split_dim,
             Tensor2d const&   cenUcv,
             Tensor2d const&   hwUcv,
             Tensor2d const&   volUcv) -> void
{
    long const Nucnv     = split_dim.size();

    //     h = np.copy(halfwidth)
    //     h[mask] *= 0.5
    //     h = np.concatenate((h, h), axis=1)
    Tensor2d const qwUcv = hwUcv * 0.5;
    halfwidth.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
        = split_dim.select(
            hwUcv.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }),
            qwUcv.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }));

    halfwidth.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
        = split_dim.select(
            qwUcv.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }),
            hwUcv.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }) // 0
        );
    halfwidth.slice(Idx2 { 0, Nucnv }, Idx2 { 2, Nucnv })
        = halfwidth.slice(Idx2 { 0, 0 }, Idx2 { 2, Nucnv });


    //     v = np.copy(volumes)
    //     v *= 0.5
    //     v = np.concatenate((v, v), axis=0)
    volume.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }) = volUcv * 0.5;
    volume.slice(Idx2 { 0, Nucnv }, Idx2 { 1, Nucnv })
        = volume.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv });

    //     c1 = np.copy(centers)
    //     c1[mask] -= h[mask]
    //     c2 = np.copy(centers)
    //     c2[mask] += h[mask]
    //     c = np.concatenate((c1, c2), axis=1)

    //     c1
    Tensor2d const left = cenUcv - qwUcv;
    center.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
        = split_dim.select(
            cenUcv.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }), // 1
            left.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })    // 0
        );
    center.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
        = split_dim.select(
            left.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }),  // 1
            cenUcv.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }) // 0
        );

    // c2
    Tensor2d const right = cenUcv + qwUcv;
    center.slice(Idx2 { 0, Nucnv }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
        = split_dim.select(
            cenUcv.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }), // 1
            right.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })   // 0
        );
    center.slice(Idx2 { 1, Nucnv }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
        = split_dim.select(
            right.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }), // 1
            cenUcv.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }) // 0
        );
}

} // namespace Fermi::Genz
