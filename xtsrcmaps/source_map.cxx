#include "xtsrcmaps/source_map.hxx"

template <size_t N = 64>
auto
integ_delta_steps() -> std::array<double, N>
{
    auto delta = std::array<double, N>();
    for (size_t i = 0; i < N; ++i)
    {
        delta[i] = double(i) / double(N) - 0.5 + 1. / (2. * double(N));
    }
    return delta;
}

auto
Fermi::point_src_model_map_wcs(std::vector<double> const&                    Px,
                               std::vector<double> const&                    Py,
                               std::vector<std::pair<double, double>> const& dirs)
    -> mdarray3
{

    size_t constexpr Ndelta = 64;
    size_t const Nsrc       = dirs.size();
    size_t const Npx        = Px.size();
    size_t const Npy        = Py.size();
    size_t const Nx         = Ndelta * Npx;
    size_t const Ny         = Ndelta * Npx;


    auto delta              = integ_delta_steps<Ndelta>();
    auto xs_arr             = std::vector<double>(Nx, 0.0);
    auto xs                 = mdspan2(xs_arr.data(), Npx, Ndelta);
    for (size_t i = 0; i < Npx; ++i)
        for (size_t l = 0; l < Ndelta; ++l) xs(i, l) = Px[i] + delta[l];


    auto ys_arr = std::vector<double>(Ny, 0.0);
    auto ys     = mdspan2(ys_arr.data(), Npy, Ndelta);
    for (size_t i = 0; i < Npy; ++i)
        for (size_t l = 0; l < Ndelta; ++l) xs(i, l) = Py[i] + delta[l];

    auto model_map = std::vector<double>(Nsrc * Npx * Npy);
    auto mm        = mdspan3(model_map.data(), Nsrc, Npx, Npy);

    ////
    for (size_t s = 0; s < dirs.size(); ++s)
    {
        double const sx = dirs[s].first;  // RA
        double const sy = dirs[s].second; // dec

        auto dsx_arr    = std::vector<double>(Nx, 0.0);
        auto dsy_arr    = std::vector<double>(Ny, 0.0);

        auto dsx        = mdspan2(dsx_arr.data(), Npx, Ndelta);
        auto dsy        = mdspan2(dsx_arr.data(), Npy, Ndelta);

        for (size_t i = 0; i < Npx; ++i)
            for (size_t d = 0; d < Ndelta; ++d)
                dsx(i, d) = (sx - xs(i, d)) * (sx - xs(i, d));

        for (size_t i = 0; i < Npy; ++i)
            for (size_t d = 0; d < Ndelta; ++d)
                dsy(i, d) = (sy - ys(i, d)) * (sy - ys(i, d));

        for (size_t px = 0; px < Npx; ++px)
        {
            for (size_t py = 0; py < Npy; ++py)
            {
                for (size_t dx = 0; dx < Ndelta; ++dx)
                {
                    for (size_t dy = 0; dy < Ndelta; ++dy) {

                                    }
                }
            }
        }
        //
    }

    return {};
}
