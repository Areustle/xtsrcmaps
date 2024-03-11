#include "xtsrcmaps/math/genz_malik.hxx"
#include "xtsrcmaps/math/tensor_types.hxx"

namespace Fermi::Genz {

auto
fullsym(Tensor2d const& c, double const l2, double const l4, double const l5)
    -> Tensor3d {
    // {c, l2, l4, l5} shape = [2];
    // points shape = [2, 17]
    // k0
    long const Nevts = c.dimension(1);
    Idx3 const Ext   = { 1, 1, Nevts }; // Slice Extent
    Tensor3d   pts(2, 17, Nevts);
    // k0
    pts = c.reshape(Idx3 { 2, 1, Nevts }).broadcast(Idx3 { 1, 17, 1 });
    // k1
    pts.slice(Idx3 { 0, 1, 0 }, Ext) -= pts.slice(Idx3 { 0, 1, 0 }, Ext).constant(l2);
    pts.slice(Idx3 { 1, 2, 0 }, Ext) -= pts.slice(Idx3 { 1, 2, 0 }, Ext).constant(l2);
    pts.slice(Idx3 { 0, 3, 0 }, Ext) += pts.slice(Idx3 { 0, 3, 0 }, Ext).constant(l2);
    pts.slice(Idx3 { 1, 4, 0 }, Ext) += pts.slice(Idx3 { 1, 4, 0 }, Ext).constant(l2);
    pts.slice(Idx3 { 0, 5, 0 }, Ext) -= pts.slice(Idx3 { 0, 5, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 1, 6, 0 }, Ext) -= pts.slice(Idx3 { 1, 6, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 0, 7, 0 }, Ext) += pts.slice(Idx3 { 0, 7, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 1, 8, 0 }, Ext) += pts.slice(Idx3 { 1, 8, 0 }, Ext).constant(l4);
    // k2
    pts.slice(Idx3 { 0, 9, 0 }, Ext) -= pts.slice(Idx3 { 0, 9, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 1, 9, 0 }, Ext) -= pts.slice(Idx3 { 1, 9, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 0, 10, 0 }, Ext) += pts.slice(Idx3 { 0, 10, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 1, 10, 0 }, Ext) -= pts.slice(Idx3 { 1, 10, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 0, 11, 0 }, Ext) -= pts.slice(Idx3 { 0, 11, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 1, 11, 0 }, Ext) += pts.slice(Idx3 { 1, 11, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 0, 12, 0 }, Ext) += pts.slice(Idx3 { 0, 12, 0 }, Ext).constant(l4);
    pts.slice(Idx3 { 1, 12, 0 }, Ext) += pts.slice(Idx3 { 1, 12, 0 }, Ext).constant(l4);
    // k3
    pts.slice(Idx3 { 0, 13, 0 }, Ext) -= pts.slice(Idx3 { 0, 13, 0 }, Ext).constant(l5);
    pts.slice(Idx3 { 1, 13, 0 }, Ext) -= pts.slice(Idx3 { 1, 13, 0 }, Ext).constant(l5);
    pts.slice(Idx3 { 0, 14, 0 }, Ext) -= pts.slice(Idx3 { 0, 14, 0 }, Ext).constant(l5);
    pts.slice(Idx3 { 1, 14, 0 }, Ext) += pts.slice(Idx3 { 1, 14, 0 }, Ext).constant(l5);
    pts.slice(Idx3 { 0, 15, 0 }, Ext) += pts.slice(Idx3 { 0, 15, 0 }, Ext).constant(l5);
    pts.slice(Idx3 { 1, 15, 0 }, Ext) -= pts.slice(Idx3 { 1, 15, 0 }, Ext).constant(l5);
    pts.slice(Idx3 { 0, 16, 0 }, Ext) += pts.slice(Idx3 { 0, 16, 0 }, Ext).constant(l5);
    pts.slice(Idx3 { 1, 16, 0 }, Ext) += pts.slice(Idx3 { 1, 16, 0 }, Ext).constant(l5);
    //
    return pts;
}


auto
region(Tensor3d const& low, Tensor3d const& high, long const Nevts)
    -> std::tuple<Tensor2d, Tensor2d, Tensor2d> {
    Tensor2d center    = ((high + low) * 0.5).reshape(Idx2 { 2, Nevts });
    Tensor2d halfwidth = ((high - low) * 0.5).reshape(Idx2 { 2, Nevts });
    Tensor2d volume    = (2. * halfwidth).prod(Idx1 { 0 }).reshape(Idx2 { 1, Nevts });
    return { center, halfwidth, volume };
}

auto
pixel_region(Tensor3d const& pixels) -> std::tuple<Tensor2d, double, double> {
    long const Nevts  = pixels.dimension(1) * pixels.dimension(2);
    Tensor2d   center = pixels.reshape(Idx2 { 2, Nevts });

    return { center, 0.5, 1. };
}

auto
rule(Tensor3d const& vals, double const volume) -> std::tuple<Tensor2d, Tensor2d> {
    TensorMap<Tensor1d const> const w(genz_malik_weights_17.data(), 17);
    TensorMap<Tensor1d const> const wE(genz_malik_err_weights_17.data(), 17);

    // # [17] . [17,range_dim, ... ] = [range_dim, ... ]
    Tensor2d result = volume * w.contract(vals, IdxPair1 { { { 0, 0 } } });

    // # [17] . [17,range_dim, ... ] = [range_dim, ... ]
    Tensor2d res5th = volume * wE.contract(vals, IdxPair1 { { { 0, 0 } } });

    // err = np.abs(res5th - result)  # [range_dim, ... ]
    Tensor2d err    = (res5th - result).abs(); //  # [range_dim, ... ]
    return { result, err };
}

auto
converged_indices(Tensor2d const& value,
                  Tensor2d const& error,
                  double const    ftol_threshold)
    -> std::tuple<std::vector<long>, std::vector<long>> {
    // Tensor1d const abserr = error.abs().maximum(Idx1 { 0 });
    Tensor1d const relerr = (error / value.abs()).maximum(Idx1 { 0 });

    auto converged        = std::vector<long> {};
    auto not_converged    = std::vector<long> {};

    for (long i = 0; i < relerr.size(); ++i) {
        if (relerr(i) > ftol_threshold) // && abserr(i) > ftol_threshold)
        {
            not_converged.push_back(i);
        } else {
            converged.push_back(i);
        }
    }

    return { converged, not_converged };
}


auto
dims_to_split(Tensor3d const&          evals,
              Tensor2d const&          error,
              double const             halfwidth,
              double const             volume,
              std::vector<long> const& not_converged) -> Tensor1byt {
    long const Ne    = evals.dimension(1);
    long const Nevts = evals.dimension(2);
    long const Nucnv = not_converged.size();

    // SPLIT DIM
    Tensor2d diff(2, Nucnv);

    Tensor3d evalUcnv(Ncnt, Ne, Nucnv);
    Map<MatrixXd>(evalUcnv.data(), Ncnt * Ne, Nucnv) = Map<MatrixXd const>(
        evals.data(), Ncnt * Ne, Nevts)(Eigen::all, not_converged);
    Tensor3d vc  = 2. * evalUcnv.slice(Idx3 { 0, 0, 0 }, Idx3 { 1, Ne, Nucnv });

    // # [ range_dim, domain_dim, ... ]
    Tensor3d v01 = evalUcnv.slice(Idx3 { 1, 0, 0 }, Idx3 { 2, Ne, Nucnv })
                   + evalUcnv.slice(Idx3 { 3, 0, 0 }, Idx3 { 2, Ne, Nucnv });
    Tensor3d v23 = evalUcnv.slice(Idx3 { 5, 0, 0 }, Idx3 { 2, Ne, Nucnv })
                   + evalUcnv.slice(Idx3 { 7, 0, 0 }, Idx3 { 2, Ne, Nucnv });

    // Compute the 4th divided difference to determine dimension on which to split.
    diff = ((v01 - vc.broadcast(Idx3 { 2, 1, 1 }))
            - ratio * (v23 - vc.broadcast(Idx3 { 2, 1, 1 })))
               .abs()
               .sum(Idx1 { 1 });

    Tensor2d errUcnv(Ne, Nucnv);
    Map<MatrixXd>(errUcnv.data(), Ne, Nucnv)
        = Map<MatrixXd const>(error.data(), Ne, Nevts)(Eigen::all, not_converged);

    Tensor1d df = (errUcnv.sum(Idx1 { 0 }) / (100. * volume));

    Tensor1byt zeros(Nucnv);
    Tensor1byt ones(Nucnv);
    zeros.setZero();
    ones.setConstant(1);

    Tensor1byt split_dim
        = (diff.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
           >= diff.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }))
              .select(zeros, ones);

    // Tensor1byt widest_dim
    //     = (halfwidth.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv })
    //        >= halfwidth.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv
    //        }))
    //           .select(zeros, ones);
    Tensor1byt widest_dim = zeros;

    Tensor1d delta
        = split_dim.select(
              diff.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }),
              diff.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }))
          // - widest_dim.select(
          //     diff.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }),
          //     diff.slice(Idx2 { 0, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv }));
          - diff.slice(Idx2 { 1, 0 }, Idx2 { 1, Nucnv }).reshape(Idx1 { Nucnv });

    // too_close = delta <= df
    // split_dim[too_close] = widest_dim[too_close]
    split_dim = (delta <= df).select(split_dim, widest_dim);

    return split_dim;
}

auto
region_split(Tensor2d&         center,
             double&           halfwidth,
             double&           volume,
             Tensor1byt const& split_dim,
             Tensor2d const&   cenUcv) -> void {
    long const Nucnv = split_dim.size();

    halfwidth *= 0.5;
    volume *= 0.5;

    //     c1 = np.copy(centers)
    //     c1[mask] -= h[mask]
    //     c2 = np.copy(centers)
    //     c2[mask] += h[mask]
    //     c = np.concatenate((c1, c2), axis=1)

    //     c1
    Tensor2d const left = cenUcv - halfwidth;
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
    Tensor2d const right = cenUcv + halfwidth;
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
