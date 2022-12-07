// Multi-dimensional Integration over an N-dimensional rectangular region. Algorithm
// from A.C. Genz, A.A. Malik, An adaptive algorithm for numerical integration over an
// N-dimensional rectangular region, J. Comput. Appl. Math. 6 (1980) 295-302.
//
// Updated by Bernstein, Espelid, Genz in "An Adaptive Algorithm for the Approximate
// Calculation of Multiple Integrals"

#pragma once

#include <iostream>

#include "xtsrcmaps/tensor_types.hxx"

#include "unsupported/Eigen/CXX11/Tensor"

namespace Fermi::Genz
{

// The number of sample points within a 2D region in a Genz_Malik rule.
constexpr long Ncnt = 17;

constexpr double alpha2
    = 0.35856858280031809199064515390793749545406372969943071; // √(9/70)
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

alignas(64) constexpr std::array<double, 17> genz_malik_weights_17 = {
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

alignas(64) constexpr std::array<double, 17> genz_malik_err_weights_17 = {
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
fullsym(Tensor2d const& c, double const l2, double const l4, double const l5)
    -> Tensor3d;

auto
region(Tensor3d const& low, Tensor3d const& high, long const Nevts)
    -> std::tuple<Tensor2d, Tensor2d, Tensor2d>;

auto
pixel_region(Tensor3d const& points) -> std::tuple<Tensor2d, double, double>;


auto
converged_indices(Tensor2d const& value,
                  Tensor2d const& error,
                  double const    ftol_threshold)
    -> std::tuple<std::vector<long>, std::vector<long>>;

auto
dims_to_split(Tensor3d const&          evals,
              Tensor2d const&          err,
              double const             halfwidth,
              double const             volume,
              std::vector<long> const& not_converged) -> Tensor1byt;

auto
region_split(Tensor2d&         center,
             double&           halfwidth,
             double&           volume,
             Tensor1byt const& split_dim,
             Tensor2d const&   cenUcv) -> void;

// [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
// integration Over an N-dimensional rectangular region", updated by Bernstein,
// Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
// Multiple Integrals"
auto
rule(Tensor3d const& vals, double const volume) -> std::tuple<Tensor2d, Tensor2d>;

template <typename F, typename G>
auto
integrate_region(F&&                  integrand,
                 G&&                  dir_pix_f,
                 Eigen::Ref<MatrixXd> result_value,
                 Tensor2d             centers,
                 double               halfwidth,
                 double               volume,
                 Array3Xd             dir_points,
                 double const         ftol_threshold,
                 size_t const         max_iteration_depth = 6) -> void
{
    // auto t0                = std::chrono::high_resolution_clock::now();

    long const Nevts       = dir_points.cols() / Genz::Ncnt;
    // Index of events.
    IdxVec evtidx          = IdxVec::LinSpaced(Nevts, 0, Nevts - 1);

    // The initial integrand evaluation.
    // dir_points             = dir_pix_f(genz_points); // Pre-Computed
    Tensor3d   evals       = integrand(dir_points);
    long const Nrange      = evals.dimension(1); // [Ne]

    // auto t1                = std::chrono::high_resolution_clock::now();
    // [range_dim, Nevts]
    auto [value, error]    = rule(evals, volume);
    // auto t2                = std::chrono::high_resolution_clock::now();

    size_t iteration_depth = 1;
    while (iteration_depth <= max_iteration_depth)
    {
        // Determine which regions are converged
        auto [converged, not_converged]
            = converged_indices(value, error, ftol_threshold);

        // Accumulate converged region results into correct event
        Map<MatrixXd> valM(value.data(), Nrange, value.dimension(1));
        result_value(Eigen::all, evtidx(converged)) += valM(Eigen::all, converged);

        long const Nucnv = not_converged.size();
        if (Nucnv == 0) { break; }
        // break;

        // evtidx = np.tile(evtidx[nmask], 2)
        IdxVec uevx = evtidx(not_converged);
        evtidx.resize(Nucnv * 2);
        evtidx << uevx, uevx;

        Tensor2d centerUcnv(2, Nucnv);

        Map<MatrixXd>(centerUcnv.data(), 2, Nucnv)
            = Map<MatrixXd>(centers.data(), 2, Nevts)(Eigen::all, not_converged);

        centers.resize(2, Nucnv * 2);

        Tensor1byt split_dim
            = dims_to_split(evals, error, halfwidth, volume, not_converged);
        region_split(centers, halfwidth, volume, split_dim, centerUcnv);

        // Pixel points in observation relative Spherical shell direction space.
        Tensor3d genz_points = fullsym(
            centers, halfwidth * alpha2, halfwidth * alpha4, halfwidth * alpha5);

        // The next integrand evaluation.
        dir_points             = dir_pix_f(genz_points);
        evals                  = integrand(dir_points);
        std::tie(value, error) = rule(evals, volume);

        ++iteration_depth;
    }

    // auto t3  = std::chrono::high_resolution_clock::now();
    // auto d10 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    // auto d21 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    // auto d32 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    // std::cout << " gm: [" << d10 << " " << d21 << " " << d32 << "] " << std::flush;
}

} // namespace Fermi::Genz
