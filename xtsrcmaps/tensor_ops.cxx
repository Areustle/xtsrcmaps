#include "xtsrcmaps/tensor_ops.hxx"

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <utility>

using std::vector;

auto
Fermi::contract210(mdarray2 const& A, mdarray2 const& B) -> mdarray2
{
    assert(A.extent(1) == B.extent(0));

    size_t const& Ni = A.extent(0);
    size_t const& Nj = B.extent(1);
    size_t const& Nk = B.extent(0);

    auto rv          = vector<double>(Ni * Nj, 0.0);
    auto R           = mdarray2(rv, Ni, Nj);

    // Hillariously naiive. Chunk this!
    for (size_t i = 0; i < Ni; ++i)
    {
        for (size_t j = 0; j < Nj; ++j)
        {
            for (size_t k = 0; k < Nk; ++k) { R(i, j) += A(i, k) * B(k, j); }
        }
    }

    return R;
}


// TODO Use Eigen or BLAS GEMM
auto
Fermi::contract3210(mdarray3 const& A, mdarray2 const& B) -> mdarray3
{
    using MatrixXdR
        = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Eigen::Map;
    // A[c,e,d]
    // B[s,c]
    assert(A.extent(0) == B.extent(1)); // C

    size_t const& Ns = B.extent(0); // S
    size_t const& Nc = B.extent(1); // C
    size_t const& Ne = A.extent(1); // E
    size_t const& Nd = A.extent(2); // D

    // R[s,e,d]
    auto rv          = vector<double>(Ns * Ne * Nd);

    Map<MatrixXdR const> mA(A.data(), Nc, Ne * Nd);
    Map<MatrixXdR const> mB(B.data(), Ns, Nc);
    Map<MatrixXdR>       mC(rv.data(), Ns, Ne * Nd);
    mC.noalias() = mB * mA;

    assert(mC.rows() == long(Ns));
    assert(mC.cols() == long(Ne * Nd));

    return mdarray3(rv, Ns, Ne, Nd);
}

auto
Fermi::mul210(mdarray2 const& A, vector<double> const& v) -> mdarray2
{
    // R[s,e] = A[s,e] * v[e]
    assert(A.extent(1) == v.size());

    size_t const& Ni = A.extent(0);
    size_t const& Nj = A.extent(1);

    auto rv          = vector<double>(A.container().size());
    auto R           = mdarray2(rv, A.extent(0), A.extent(1));

    // Hillariously naiive. Chunk this!
    for (size_t i = 0; i < Ni; ++i)
    {
        for (size_t j = 0; j < Nj; ++j) { R(i, j) = A(i, j) * v[j]; }
    }

    return R;
};

auto
Fermi::mul310(mdarray3 const& A, vector<double> const& v) -> mdarray3
{
    // R[s,e,d] = A[s,e,d] * v[e]
    assert(A.extent(1) == v.size());

    size_t const& Ns = A.extent(0); // S
    size_t const& Ne = A.extent(1); // E
    size_t const& Nd = A.extent(2); // D

    auto rv          = vector<double>(A.size());
    auto R           = mdarray3(rv, A.extent(0), A.extent(1), A.extent(2));

    // Hillariously naiive. Chunk this!
    for (size_t s = 0; s < Ns; ++s)
    {
        for (size_t e = 0; e < Ne; ++e)
        {
            for (size_t d = 0; d < Nd; ++d) { R(s, e, d) = A(s, e, d) * v[e]; }
        }
    }

    return R;
};

auto
Fermi::mul32_1(mdarray3 const& A, mdarray2 const& B) -> mdarray3
{
    // R[s,e,d] = A[s,e,d] * B[s,e]
    assert(A.extent(0) == B.extent(0));
    assert(A.extent(1) == B.extent(1));

    size_t const& Ns = A.extent(0); // S
    size_t const& Ne = A.extent(1); // E
    size_t const& Nd = A.extent(2); // D

    auto rv          = vector<double>(A.container().size());
    auto R           = mdarray3(rv, A.extent(0), A.extent(1), A.extent(2));

    // Hillariously naiive. Chunk this!
    for (size_t s = 0; s < Ns; ++s)
    {
        for (size_t e = 0; e < Ne; ++e)
        {
            for (size_t d = 0; d < Nd; ++d) { R(s, e, d) = A(s, e, d) * B(s, e); }
        }
    }

    return R;
};


auto
Fermi::mul322(mdarray3 const& A, mdarray2 const& B) -> mdarray3
{
    // R[c,e,d] = A[c,e,d] * B[c,e]
    assert(A.extent(0) == B.extent(0));
    assert(A.extent(1) == B.extent(1));

    size_t const& Nc = A.extent(0); // C
    size_t const& Ne = A.extent(1); // E
    size_t const& Nd = A.extent(2); // D

    auto rv          = vector<double>(A.container().size());
    auto R           = mdarray3(rv, A.extent(0), A.extent(1), A.extent(2));

    // Hillariously naiive. Chunk this!
    for (size_t c = 0; c < Nc; ++c)
    {
        for (size_t e = 0; e < Ne; ++e)
        {
            for (size_t d = 0; d < Nd; ++d) { R(c, e, d) = A(c, e, d) * B(c, e); }
        }
    }

    return R;
};

auto
Fermi::sum2_2(mdarray2 const& A, mdarray2 const& B) -> mdarray2
{
    // R[i,j] = A[i,j] * B[i,j]
    assert(A.size() == B.size());
    assert(A.extent(0) == B.extent(0));
    assert(A.extent(1) == B.extent(1));

    auto rv = vector<double>(A.size(), 0.0);

    std::transform(A.container().cbegin(),
                   A.container().cend(),
                   B.container().cbegin(),
                   rv.begin(),
                   std::plus<> {});

    return mdarray2(rv, A.extent(0), A.extent(1));
}

auto
Fermi::sum3_3(mdarray3 const& A, mdarray3 const& B) -> mdarray3
{
    // R[i,j,k] = A[i,j,k] + B[i,j,k]
    assert(A.size() == B.size());
    assert(A.extent(0) == B.extent(0));
    assert(A.extent(1) == B.extent(1));
    assert(A.extent(2) == B.extent(2));

    auto rv = vector<double>(A.size(), 0.0);

    std::transform(A.container().cbegin(),
                   A.container().cend(),
                   B.container().cbegin(),
                   rv.begin(),
                   std::plus<> {});

    return mdarray3(rv, A.extent(0), A.extent(1), A.extent(2));
}

auto
Fermi::safe_reciprocal(mdarray2 const& A) -> mdarray2
{
    auto rv = vector<double>(A.size());
    std::transform(A.container().cbegin(),
                   A.container().cend(),
                   rv.begin(),
                   [](auto const x) { return x <= 0. ? 0. : 1. / x; });
    return mdarray2(rv, A.extent(0), A.extent(1));
};


// inline void
// co_contract_base(auto R, auto const A, auto const B)
// {
//
//     size_t const& Ns = B.extent(0); // N
//     size_t const& Nd = A.extent(0); // D
//     size_t const& Nc = A.extent(1); // C
//     size_t const& Ne = A.extent(2); // E
//
//     for (size_t n = 0; n < Ns; ++n)
//     {
//         for (size_t d = 0; d < Nd; ++d)
//         {
//             for (size_t c = 0; c < Nc; ++c)
//             {
//                 for (size_t e = 0; e < Ne; ++e)
//                 {
//                     //
//                     R(n, d, e) += A(d, c, e) * B(n, c);
//                 }
//             }
//         }
//     }
// }
//
// inline bool
// is_co_contract_base(size_t const Ns,
//                     size_t const Nd,
//                     size_t const Nc,
//                     size_t const Ne) noexcept
// {
//     return (Ns <= 4 && Nd <= 4 && Nc <= 4 && Ne <= 4);
// }

// // R  [Ns, Nd, Ne]
// // A  [Nd, Nc, Ne]
// // B  [Ns, Nc]
// void
// co_contract3210(auto R, auto const A, auto const B)
// {
// using std::pair;
// using std::experimental::full_extent;
// using std::experimental::submdspan;
//     auto largest_dim = [](size_t const Ns,
//                           size_t const Nd,
//                           size_t const Nc,
//                           size_t const Ne) noexcept -> char {
//         char          midx   = 0;
//         size_t const* valptr = &Ns;
//         /* clang-format off */
//         if (*valptr < Nd) { midx = 1; valptr = &Nd; }
//         if (*valptr < Nc) { midx = 2; valptr = &Nc; }
//         if (*valptr < Ne) { midx = 3; valptr = &Ne; }
//         /* clang-format on */
//         return midx;
//     };
//
//     size_t const& Ns = B.extent(0); // S
//     size_t const& Nd = A.extent(0); // D
//     size_t const& Nc = A.extent(1); // C
//     size_t const& Ne = A.extent(2); // E
//
//     // check for base case
//     if (is_co_contract_base(Ns, Nd, Nc, Ne))
//     {
//         // Do base case computation
//         return co_contract_base(R, A, B);
//     }
//
//     char const ld = largest_dim(Ns, Nd, Nc, Ne);
//
//     if (ld == 0) // Ns is the largest. Cut R[0] and B[0]
//     {
//         auto p1 = pair(0, Ns / 2);
//         auto p2 = pair(Ns / 2, Ns);
//         auto R1 = submdspan(R, p1, full_extent, full_extent);
//         auto R2 = submdspan(R, p2, full_extent, full_extent);
//         auto B1 = submdspan(B, p1, full_extent);
//         auto B2 = submdspan(B, p2, full_extent);
//         co_contract3210(R1, A, B1);
//         co_contract3210(R2, A, B2);
//     }
//     else if (ld == 1) // Nd is largest. Split R[1], A[0]
//     {
//         auto p1 = pair(0, Nd / 2);
//         auto p2 = pair(Nd / 2, Nd);
//         auto R1 = submdspan(R, full_extent, p1, full_extent);
//         auto R2 = submdspan(R, full_extent, p2, full_extent);
//         auto A1 = submdspan(A, p1, full_extent, full_extent);
//         auto A2 = submdspan(A, p2, full_extent, full_extent);
//         co_contract3210(R1, A1, B);
//         co_contract3210(R2, A2, B);
//     }
//     else if (ld == 2) // -- Nc is largest. Split A[1] and B[1]
//     {
//         auto p1 = pair(0, Nc / 2);
//         auto p2 = pair(Nc / 2, Nc);
//         auto A1 = submdspan(A, full_extent, p1, full_extent);
//         auto A2 = submdspan(A, full_extent, p2, full_extent);
//         auto B1 = submdspan(B, full_extent, p1);
//         auto B2 = submdspan(B, full_extent, p2);
//         co_contract3210(R, A1, B1);
//         co_contract3210(R, A2, B2);
//     }
//     else // if (ld == 3) // -- Ne is largest. Split R[2] and A[2]
//     {
//         auto p1 = pair(0, Ne / 2);
//         auto p2 = pair(Ne / 2, Ne);
//         auto R1 = submdspan(R, full_extent, full_extent, p1);
//         auto R2 = submdspan(R, full_extent, full_extent, p2);
//         auto A1 = submdspan(A, full_extent, full_extent, p1);
//         auto A2 = submdspan(A, full_extent, full_extent, p2);
//         co_contract3210(R1, A1, B);
//         co_contract3210(R2, A2, B);
//     }
// }

// auto
// eig_contract3210(mdarray3 const& A, mdarray2 const& B) -> mdarray3
// {
// }
