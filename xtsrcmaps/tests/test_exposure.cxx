#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/tests/fits/exposure_cosbins_expected.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/exposure.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/source_utils.hxx"


TEST_CASE("Test Source Exposure Cosine Bins.")
{
    auto cfg          = Fermi::XtCfg();
    auto srcs         = Fermi::parse_src_xml(cfg.srcmdl);
    auto dirs         = Fermi::directions_from_point_sources(srcs);
    auto opt_exp_map  = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto opt_wexp_map = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    REQUIRE(opt_exp_map);
    REQUIRE(opt_wexp_map);
    auto exp_map      = Fermi::exp_map(opt_exp_map.value());
    auto wexp_map     = Fermi::exp_map(opt_wexp_map.value());
    auto exp_cosbins  = Fermi::src_exp_cosbins(dirs, exp_map);
    auto wexp_cosbins = Fermi::src_exp_cosbins(dirs, wexp_map);

    REQUIRE(exp_cosbins.size() == FermiTest::exposure_source_cosine_bins.size());
    for (size_t i = 0; i < FermiTest::exposure_source_cosine_bins.size(); ++i)
    {
        REQUIRE(exp_cosbins.container()[i]
                == doctest::Approx(FermiTest::exposure_source_cosine_bins[i]));
    }
    REQUIRE(wexp_cosbins.size()
            == FermiTest::weighted_exposure_source_cosine_bins.size());
    for (size_t i = 0; i < FermiTest::weighted_exposure_source_cosine_bins.size(); ++i)
    {
        REQUIRE(wexp_cosbins.container()[i]
                == doctest::Approx(FermiTest::weighted_exposure_source_cosine_bins[i]));
    }
}


TEST_CASE("Test Source Exposure CosineThetas.")
{
    auto cfg          = Fermi::XtCfg();
    auto opt_exp_map  = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto opt_wexp_map = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    REQUIRE(opt_exp_map);
    REQUIRE(opt_wexp_map);
    auto exp_costhetas  = Fermi::exp_costhetas(opt_exp_map.value());
    auto wexp_costhetas = Fermi::exp_costhetas(opt_wexp_map.value());

    REQUIRE(exp_costhetas.size() == FermiTest::exp_costhetas.size());
    REQUIRE(wexp_costhetas.size() == FermiTest::exp_costhetas.size());
    for (size_t i = 0; i < FermiTest::exp_costhetas.size(); ++i)
    {
        REQUIRE(exp_costhetas[i] == doctest::Approx(FermiTest::exp_costhetas[i]));
    }
    for (size_t i = 0; i < FermiTest::exp_costhetas.size(); ++i)
    {
        REQUIRE(wexp_costhetas[i] == doctest::Approx(FermiTest::exp_costhetas[i]));
    }
}

TEST_CASE("Test Aeff Value Front")
{
    auto const cfg      = Fermi::XtCfg();
    auto const opt_aeff = Fermi::load_aeff(cfg.aeff_name);
    auto const oexpmap  = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto const owexpmap = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    auto const oen      = Fermi::fits::ccube_energies(cfg.cmap);
    REQUIRE(opt_aeff);
    REQUIRE(oexpmap);
    REQUIRE(owexpmap);
    REQUIRE(oen);
    auto const aeff           = opt_aeff.value();
    auto const logEs          = Fermi::log10_v(oen.value());
    auto const exp_costhetas  = Fermi::exp_costhetas(oexpmap.value());
    auto const wexp_costhetas = Fermi::exp_costhetas(owexpmap.value());

    auto aeff_f = Fermi::aeff_value(exp_costhetas, logEs, aeff.front.effective_area);
    auto aeff_b = Fermi::aeff_value(exp_costhetas, logEs, aeff.back.effective_area);

    REQUIRE(aeff_f.container().size() == FermiTest::aeff_front_c_e.size());
    REQUIRE(aeff_b.container().size() == FermiTest::aeff_back_c_e.size());

    for (size_t i = 0; i < FermiTest::aeff_front_c_e.size(); ++i)
    {
        REQUIRE(aeff_f.container()[i] == doctest::Approx(FermiTest::aeff_front_c_e[i]));
    }
    for (size_t i = 0; i < FermiTest::aeff_back_c_e.size(); ++i)
    {
        REQUIRE(aeff_b.container()[i] == doctest::Approx(FermiTest::aeff_back_c_e[i]));
    }
}

TEST_CASE("test bilerps")
{
    auto cfg         = Fermi::XtCfg();

    auto oaeff       = Fermi::load_aeff(cfg.aeff_name);
    auto opt_exp_map = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto oen         = Fermi::fits::ccube_energies(cfg.cmap);
    REQUIRE(oaeff);
    REQUIRE(opt_exp_map);
    REQUIRE(oen);
    auto aeff          = oaeff.value();
    auto exp_costhetas = Fermi::exp_costhetas(opt_exp_map.value());
    auto logE          = Fermi::log10_v(oen.value());

    // clang-format off
    auto cweight       = std::vector<float> {
        0.9875,    0.8875,  0.6875,  0.387499, 0.99375,  0.74375,  0.44375,  0.0937497,
        0.69375,   0.24375, 0.74375, 0.19375,  0.59375,  0.94375,  0.24375,  0.49375,
        0.69375,   0.84375, 0.94375, 0.99375,  0.99375,  0.94375,  0.84375,  0.69375,
        0.49375,   0.24375, 0.94375, 0.59375,  0.19375,  0.74375,  0.24375,  0.69375,
        0.0937503, 0.44375, 0.74375, 0.999871, 0.962758, 0.924613, 0.885438, 0.845232
    };
    // clang-format on

    auto const clbound = std::vector<float> {
        0.9875, 0.9875, 0.9875, 0.9875, 0.9625, 0.9625, 0.9625, 0.9625, 0.9375, 0.9375,
        0.9125, 0.9125, 0.8875, 0.8625, 0.8625, 0.8375, 0.8125, 0.7875, 0.7625, 0.7375,
        0.7125, 0.6875, 0.6625, 0.6375, 0.6125, 0.5875, 0.5375, 0.5125, 0.4875, 0.4375,
        0.4125, 0.3625, 0.3375, 0.2875, 0.2375, -1,     -1,     -1,     -1,     -1
    };

    auto const cubound = std::vector<float> {
        1,      1,      1,      1,      0.9875, 0.9875, 0.9875, 0.9875, 0.9625, 0.9625,
        0.9375, 0.9375, 0.9125, 0.8875, 0.8875, 0.8625, 0.8375, 0.8125, 0.7875, 0.7625,
        0.7375, 0.7125, 0.6875, 0.6625, 0.6375, 0.6125, 0.5625, 0.5375, 0.5125, 0.4625,
        0.4375, 0.3875, 0.3625, 0.3125, 0.2625, 0.2125, 0.2125, 0.2125, 0.2125, 0.2125
    };


    auto Cs = aeff.front.effective_area.cosths;
    auto IC = std::span(Cs);

    for (size_t i = 0; i < exp_costhetas.size(); ++i)
    {
        auto p = Fermi::lerp_pars(IC, exp_costhetas[i]);
        REQUIRE(std::get<0>(p) == doctest::Approx(cweight[i]));
        REQUIRE(Cs[std::get<2>(p)] == doctest::Approx(cubound[i]));
    }

    auto eweight = std::vector<float> {
        0.5,       0.0995546, 0.699109,  0.298663, 0.898219,  0.497773, 0.0973276,
        0.696882,  0.296437,  0.895991,  0.495546, 0.0951007, 0.694655, 0.294209,
        0.893764,  0.493319,  0.0928739, 0.692428, 0.291983,  0.891537, 0.491091,
        0.0906463, 0.690201,  0.859837,  0.694655, 0.494432,  0.29421,  0.0939871,
        0.893764,  0.693542,  0.493319,  0.293096, 0.0928735, 0.892651, 0.692428,
        0.492205,  0.291983,  0.0917601
    };

    auto elbound = std::vector<float> {
        1.96875, 2.09375, 2.15625, 2.28125, 2.34375, 2.46875, 2.59375, 2.65625,
        2.78125, 2.84375, 2.96875, 3.09375, 3.15625, 3.28125, 3.34375, 3.46875,
        3.59375, 3.65625, 3.78125, 3.84375, 3.96875, 4.09375, 4.15625, 4.21875,
        4.3125,  4.4375,  4.5625,  4.6875,  4.6875,  4.8125,  4.9375,  5.0625,
        5.1875,  5.1875,  5.3125,  5.4375,  5.5625,  5.6875
    };

    auto eubound = std::vector<float> {
        2.03125, 2.15625, 2.21875, 2.34375, 2.40625, 2.53125, 2.65625, 2.71875,
        2.84375, 2.90625, 3.03125, 3.15625, 3.21875, 3.34375, 3.40625, 3.53125,
        3.65625, 3.71875, 3.84375, 3.90625, 4.03125, 4.15625, 4.21875, 4.3125,
        4.4375,  4.5625,  4.6875,  4.8125,  4.8125,  4.9375,  5.0625,  5.1875,
        5.3125,  5.3125,  5.4375,  5.5625,  5.6875,  5.8125
    };

    auto Es = aeff.front.effective_area.logEs;
    auto IE = std::span(Es);

    for (size_t i = 0; i < logE.size(); ++i)
    {
        auto p = Fermi::lerp_pars(IE, logE[i]);
        REQUIRE(std::get<0>(p) == doctest::Approx(eweight[i]));
        REQUIRE(Es[std::get<2>(p)] == doctest::Approx(eubound[i]));
    }
}
