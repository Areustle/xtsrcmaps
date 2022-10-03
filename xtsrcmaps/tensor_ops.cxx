#include "xtsrcmaps/tensor_ops.hxx"

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
Fermi::sum2_2(mdarray2 const& A, mdarray2 const& B) -> mdarray2
{
    // R[i,j] = A[i,j] * B[i,j]
    assert(A.extent(0) == B.extent(0));
    assert(A.extent(1) == B.extent(1));
    assert(A.size() == B.size());

    auto rv = vector<double>(A.size(), 0.0);

    std::transform(A.container().cbegin(),
                   A.container().cend(),
                   B.container().cbegin(),
                   rv.begin(),
                   std::plus<> {});

    return mdarray2(rv, A.extent(0), A.extent(1));
}

auto
Fermi::safe_reciprocal(mdarray2 const& A) -> mdarray2
{
    auto rv = vector<double>(A.size());
    std::transform(A.container().cbegin(),
                   A.container().cend(),
                   rv.begin(),
                   [](auto const x) { return x == 0.0 ? 1. / x : 0.0; });
    return mdarray2(rv, A.extent(0), A.extent(1));
};
