#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <cassert>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif


auto
Fermi::exp_contract(Fermi::Tensor<double, 2> const& A,
                    Fermi::Tensor<double, 2> const& B)
    -> Fermi::Tensor<double, 2> {
    /* ===========================================================
     * Tensor Contractions as DGEMM Matrix Multiplies
     */
    size_t const N0 = A.extent(0);
    size_t const N1 = A.extent(1);
    size_t const N2 = B.extent(1);

    // C[s, e] = (A[s, c] * B[c, e])
    Fermi::Tensor<double, 2> C(N0, N2);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                N0,
                N2,
                N1,
                1.0,
                A.data(),
                N1,
                B.data(),
                N2,
                0.0,
                C.data(),
                N2);
    return C;
}
