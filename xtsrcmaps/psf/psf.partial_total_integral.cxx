#include "xtsrcmaps/psf/psf.hxx"

#include "xtsrcmaps/misc.hxx"


auto
Fermi::PSF::partial_total_integral(Tensor3d const& mean_psf /* [Nd, Ne, Ns] */
                                   ) -> std::pair<Tensor3d, Tensor2d> {
    long   Nd   = mean_psf.dimension(0);
    long   Ne   = mean_psf.dimension(1);
    long   Ns   = mean_psf.dimension(2);
    SepArr seps = separations();
    /* assert(seps.size() == Nd); */
    TensorMap<Tensor1d> delta(seps.data(), Nd);
    /* assert(Nd == delta.size()); */
    delta = deg2rad * delta;

    Tensor3d parint(Nd, Ne, Ns);
    parint.setZero();
    Tensor3d totint(1, Ne, Ns);
    totint.setConstant(1.);

    Idx3 const i0 = { 0, 0, 0 };
    Idx3 const i1 = { 1, 0, 0 };
    Idx3 const i2 = { Nd - 1, 0, 0 };
    Idx3 const i3 = { Nd - 1, 1, 1 };
    Idx3 const i4 = { Nd, 1, 1 };
    Idx3 const i5 = { 1, Ne, Ns };
    Idx3 const i6 = { Nd - 1, Ne, Ns };
    // Idx3 const i7 = { Nd, Ne, Ns };

    // Use Midpoint Rule to compute approximate sum of psf from each separation entry
    // over the lookup table.

    // [Nd, 1, 1]
    Tensor3d X    = delta.slice(Idx1 { 0 }, Idx1 { Nd }).reshape(i4);
    // [Nd-1, 1, 1]
    Tensor3d DX   = X.slice(i1, i3) - X.slice(i0, i3);
    Tensor3d SX   = X.slice(i1, i3) + X.slice(i0, i3);
    // [Nd, Ne, Ns]
    Tensor3d Y = X.unaryExpr([](double t) { return twopi * std::sin(t); }).broadcast(i5)
                 * mean_psf;
    // [Nd-1, Ne, Ns]
    // Tensor3d DY          = Y.slice(i1, i6) - Y.slice(i0, i6);
    Tensor3d M           = (Y.slice(i1, i6) - Y.slice(i0, i6)) / DX.broadcast(i5);
    Tensor3d B           = Y.slice(i0, i6) - (M * X.slice(i0, i3).broadcast(i5));
    Tensor3d V           = (0.5 * M * SX.broadcast(i5) + B) * DX.broadcast(i5);
    // [Nd, Ne, Ns]
    parint.slice(i1, i6) = V.cumsum(0);
    //
    // Normalize the partial along the separation dimension.
    totint               = parint.slice(i2, i5);
    Tensor3d invtotint   = totint.inverse();
    Tensor3d zeros       = totint.constant(0.0);
    parint *= (totint == 0.0).select(zeros, invtotint).broadcast(i4);

    return { parint, totint.reshape(Idx2 { Ne, Ns }) };
}
