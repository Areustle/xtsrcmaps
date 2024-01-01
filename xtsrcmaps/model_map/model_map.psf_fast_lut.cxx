#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/math/genz_malik.hxx"
#include "xtsrcmaps/misc/misc.hxx"

auto
Fermi::ModelMap::psf_fast_lut(Array3Xd const& points3,
                              ArrayXd const&  src_d,
                              Tensor2d const& table) -> Tensor3d {
    // Dimensions
    long const Npts           = points3.cols();
    long const Ne             = table.dimension(0);
    long const Nevts          = Npts / Genz::Ncnt;

    // Given sample points on the sphere in 3-direction-space, compute the
    // separation.
    auto diff                 = points3.colwise() - src_d;
    auto mag                  = diff.colwise().norm();
    auto off                  = 2. * rad2deg * Eigen::asin(0.5 * mag);

    // From the spherical offset, use logarithmic interpolation to get the index
    // val. Similar to implementation of Fermi::PSF::fast_separation_lower_index
    auto           scaled_off = 1e4 * off;
    ArrayXXd const separation_index
        = (scaled_off < 1.)
              .select(scaled_off, 1. + (scaled_off.log() / sep_step));
    TensorMap<Tensor1d const> const idxs(separation_index.data(), Npts);

    // Sample the PSF lookup table by finding sequential elements which share a
    // table column (by virtue of having the same separation index). Then use
    // tensor contraction to contract all of these together with the correct
    // alpha multiplier of the psf values.

    // Allocate a result buffer [Ne, 17, Nevts]
    // Tensor3d vals(Ne, Genz::Ncnt, Npts / Genz::Ncnt);
    Tensor3d vals(Genz::Ncnt, Ne, Nevts);

    // iterate over every point
    long i = 0;
    while (i < Npts) {

        // Lookup table's separation index.
        double const index = std::floor(idxs(i));
        // run length of points which share a separation index.
        long Nlen          = 1;
        // Iterate sequential points until a new index value is seen
        while ((i + Nlen < Npts) && index == std::floor(idxs(i + Nlen))) {
            ++Nlen;
        }
        // Get a view Linear of the same-separation points.
        TensorMap<Tensor1d const> const ss(idxs.data() + i, Nlen);
        // Compute the interpolation weights for every ss point.
        Tensor2d weights(Nlen, 2);
        TensorMap<Tensor1d>(weights.data() + Nlen, Nlen) = ss - index;
        TensorMap<Tensor1d>(weights.data(), Nlen)
            = 1. - TensorMap<Tensor1d>(weights.data() + Nlen, Nlen);

        // Get a view of the psf lookup table.
        TensorMap<Tensor2d const> const lut(
            table.data() + long(index) * Ne, Ne, 2);
        // Contract the weights with the lookup table entries, thereby computing
        // the PSF values for every energy in the table and every ss point. [Ne,
        // Nlen]
        Tensor2d vv = lut.contract(weights, IdxPair1 { { { 1, 1 } } });

        // // Write the Energies into the result buffer via a veiw.
        // TensorMap<Tensor2d>(vals.data() + i * Ne, Ne, Nlen) = vv;
        for (long j = 0; j < Nlen; ++j) {
            long evoff = (i + j) / Genz::Ncnt;
            long gzoff = (i + j) % Genz::Ncnt;

            // (vals.data() + (evoff * Ne * Genz::Ncnt) + gzoff) = 0.;
            for (long k = 0; k < Ne; ++k) { vals(gzoff, k, evoff) = vv(k, j); }
        }

        // Shift the target point by the length of ss points to ensure we start
        // at an unseen point
        i += Nlen;
    }
    return vals;
}
