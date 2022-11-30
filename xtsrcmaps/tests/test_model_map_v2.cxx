#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/model_map.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/source_utils.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

TEST_CASE("Pixel Source Separation and Offset")
{
    auto       cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::directions_from_point_sources(srcs);

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(opt_ccube);
    auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
    Fermi::SkyGeom skygeom(ccube);
    long const     Nw  = 100;
    long const     Nh  = 100;
    long const     Ns  = dirs.size();
    long const     Nd  = 401;
    long const     Ne  = 38;

    auto const st_off0 = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_initial_offset.bin", Ns, Nw, Nh);
    assert(st_off0.dimension(0) == Nh);
    assert(st_off0.dimension(1) == Nw);
    assert(st_off0.dimension(2) == Ns);

    auto const st_offi0 = Fermi::row_major_file_to_col_major_tensor<unsigned short>(
        "./xtsrcmaps/tests/expected/src_initial_offset_idx.bin", Ns, Nw, Nh);
    assert(st_offi0.dimension(0) == Nh);
    assert(st_offi0.dimension(1) == Nw);
    assert(st_offi0.dimension(2) == Ns);

    auto const st_offs0 = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_initial_offset_scalar.bin", Ns, Nw, Nh);
    assert(st_offs0.dimension(0) == Nh);
    assert(st_offs0.dimension(1) == Nw);
    assert(st_offs0.dimension(2) == Ns);

    Tensor4d const st_v0 = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_mean_psf_v0.bin", Ns, Nw, Nh, Ne);
    assert(st_v0.dimension(0) == Ne);
    assert(st_v0.dimension(1) == Nh);
    assert(st_v0.dimension(2) == Nw);
    assert(st_v0.dimension(3) == Ns);

    Tensor3d const uPsf = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/uPsf_normalized_SED.bin", Ns, Ne, Nd);
    assert(uPsf.dimension(0) == Nd);
    assert(uPsf.dimension(1) == Ne);
    assert(uPsf.dimension(2) == Ns);

    MatrixXd const init_points = Fermi::ModelMap::get_init_points(Nh, Nw);
    Array3Xd const init_dirs   = Fermi::ModelMap::get_dir_points(
        Fermi::ModelMap::sub_pixel_points(init_points, 0), skygeom);

    for (long s = 0; s < Ns; ++s)
    {
        auto           src_dir = skygeom.sph2dir(dirs[s]); // CLHEP Style 3
        Eigen::ArrayXd src_d(3, 1);
        src_d << std::get<0>(src_dir), std::get<1>(src_dir), std::get<2>(src_dir);

        ArrayXd const init_sep = Fermi::ModelMap::separation(init_dirs, src_d);
        for (long w = 0; w < Nw; ++w)
        {
            for (long h = 0; h < Nh; ++h)
            {
                REQUIRE_MESSAGE(doctest::Approx(init_sep(h + w * Nh)).epsilon(1e-5)
                                    == st_off0(h, w, s),
                                h << " " << w << " " << s << ": ");
            }
        }

        Tensor1d isep = Fermi::ModelMap::index_from_sep(init_sep);
        // for (long w = 0; w < Nw; ++w)
        // {
        //     for (long h = 0; h < Nh; ++h)
        //     {
        //
        //         long logidx = long(isep(h + w * Nh));
        //         CHECK_MESSAGE(logidx == st_offi0(h, w, s),
        //                       h << " " << w << " " << s << ": " << isep(h + w * Nh));
        //     }
        // }
        Tensor2d const tuPsf_DE
            = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 }).reshape(Idx2 { Nd, Ne });

        for (long e = 0; e < Ne; ++e)
        {
            Tensor2d y0 = Fermi::ModelMap::psf_single_energy(isep, tuPsf_DE, e, 1);
            for (long w = 0; w < Nw; ++w)
            {
                for (long h = 0; h < Nh; ++h)
                {
                    REQUIRE_MESSAGE(doctest::Approx(y0(0, h + w * Nh)).epsilon(1e-3)
                                        == st_v0(e, h, w, s),
                                    e << " " << h << " " << w << " " << s << "\n");
                }
            }
        }
    }
}
