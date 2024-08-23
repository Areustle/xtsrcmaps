#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"


#include "tests/fermi_tests.hxx"
#include "tests/fits/aeff_expected.hxx"
#include "tests/fits/exposure_cosbins_expected.hxx"
#include "tests/fits/exposure_expected.hxx"

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/math/bilerp.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/source/source.hxx"
#include "xtsrcmaps/tensor/read_file_tensor.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"


TEST_CASE("test lerp_pars") {
    auto const cfg         = Fermi::XtCfg();
    auto const oaeff       = Fermi::load_aeff(cfg.aeff_file);
    auto const opt_exp_map = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto const oen         = Fermi::fits::ccube_energies(cfg.cmap);
    REQUIRE(oaeff);
    REQUIRE(opt_exp_map);
    REQUIRE(oen);
    auto const aeff          = oaeff.value();
    auto const exp_costhetas = Fermi::exp_costhetas(opt_exp_map.value());
    auto const logE          = Fermi::log10_v(oen.value());

    // clang-format off
    auto cweight       = std::vector<float> {
        0.9875,    0.8875,  0.6875,  0.387499, 0.99375,  0.74375,  0.44375,  0.0937497,
        0.69375,   0.24375, 0.74375, 0.19375,  0.59375,  0.94375,  0.24375,  0.49375,
        0.69375,   0.84375, 0.94375, 0.99375,  0.99375,  0.94375,  0.84375,  0.69375,
        0.49375,   0.24375, 0.94375, 0.59375,  0.19375,  0.74375,  0.24375,  0.69375,
        0.0937503, 0.44375, 0.74375, 0.999871, 0.962758, 0.924613, 0.885438, 0.845232
    };
    // clang-format on

    auto const clbound       = std::vector<float> {
        0.9875, 0.9875, 0.9875, 0.9875, 0.9625, 0.9625, 0.9625, 0.9625,
        0.9375, 0.9375, 0.9125, 0.9125, 0.8875, 0.8625, 0.8625, 0.8375,
        0.8125, 0.7875, 0.7625, 0.7375, 0.7125, 0.6875, 0.6625, 0.6375,
        0.6125, 0.5875, 0.5375, 0.5125, 0.4875, 0.4375, 0.4125, 0.3625,
        0.3375, 0.2875, 0.2375, -1,     -1,     -1,     -1,     -1
    };

    auto const cubound = std::vector<float> {
        1,      1,      1,      1,      0.9875, 0.9875, 0.9875, 0.9875,
        0.9625, 0.9625, 0.9375, 0.9375, 0.9125, 0.8875, 0.8875, 0.8625,
        0.8375, 0.8125, 0.7875, 0.7625, 0.7375, 0.7125, 0.6875, 0.6625,
        0.6375, 0.6125, 0.5625, 0.5375, 0.5125, 0.4625, 0.4375, 0.3875,
        0.3625, 0.3125, 0.2625, 0.2125, 0.2125, 0.2125, 0.2125, 0.2125
    };


    auto Cs = aeff.front.effective_area.cosths;
    auto IC = std::span(Cs.data(), Cs.extent(0));

    for (long i = 0; i < exp_costhetas.size(); ++i) {
        auto p = Fermi::lerp_pars(IC, exp_costhetas[i]);
        CHECK(std::get<0>(p) == doctest::Approx(cweight[i]));
        CHECK(Cs[std::get<2>(p)] == doctest::Approx(cubound[i]));
    }

    auto eweight = std::vector<float> {
        0.5,       0.0995546, 0.699109,  0.298663,  0.898219,  0.497773,
        0.0973276, 0.696882,  0.296437,  0.895991,  0.495546,  0.0951007,
        0.694655,  0.294209,  0.893764,  0.493319,  0.0928739, 0.692428,
        0.291983,  0.891537,  0.491091,  0.0906463, 0.690201,  0.859837,
        0.694655,  0.494432,  0.29421,   0.0939871, 0.893764,  0.693542,
        0.493319,  0.293096,  0.0928735, 0.892651,  0.692428,  0.492205,
        0.291983,  0.0917601
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
    auto IE = std::span(Es.data(), Es.extent(0));

    for (long i = 0; i < logE.size(); ++i) {
        auto p = Fermi::lerp_pars(IE, logE[i]);
        REQUIRE(std::get<0>(p) == doctest::Approx(eweight[i]));
        REQUIRE(Es[std::get<2>(p)] == doctest::Approx(eubound[i]));
    }
}

TEST_CASE("Test Source Exposure Cosine Bins.") {
    auto cfg         = Fermi::XtCfg();
    auto srcs        = Fermi::parse_src_xml(cfg.srcmdl);
    auto dirs        = Fermi::spherical_coords_from_point_sources(srcs);
    auto opt_exp_map = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto opt_wexp_map
        = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    REQUIRE(opt_exp_map);
    REQUIRE(opt_wexp_map);
    auto exp_map                       = Fermi::exp_map(opt_exp_map.value());
    auto wexp_map                      = Fermi::exp_map(opt_wexp_map.value());
    auto src_exposure_cosbins          = Fermi::src_exp_cosbins(dirs, exp_map);
    auto src_weighted_exposure_cosbins = Fermi::src_exp_cosbins(dirs, wexp_map);

    et2comp(src_exposure_cosbins, FermiTest::exposure_source_cosine_bins);
    et2comp(src_weighted_exposure_cosbins,
            FermiTest::weighted_exposure_source_cosine_bins);
}


TEST_CASE("Test Source Exposure CosineThetas.") {
    auto cfg         = Fermi::XtCfg();
    auto opt_exp_map = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto opt_wexp_map
        = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    REQUIRE(opt_exp_map);
    REQUIRE(opt_wexp_map);
    auto exp_costhetas  = Fermi::exp_costhetas(opt_exp_map.value());
    auto wexp_costhetas = Fermi::exp_costhetas(opt_wexp_map.value());

    REQUIRE(exp_costhetas.size() == FermiTest::exp_costhetas.size());
    REQUIRE(wexp_costhetas.size() == FermiTest::exp_costhetas.size());
    for (long i = 0; i < FermiTest::exp_costhetas.size(); ++i) {
        REQUIRE(exp_costhetas[i]
                == doctest::Approx(FermiTest::exp_costhetas[i]));
    }
    for (long i = 0; i < FermiTest::exp_costhetas.size(); ++i) {
        REQUIRE(wexp_costhetas[i]
                == doctest::Approx(FermiTest::exp_costhetas[i]));
    }
}

TEST_CASE("Test Aeff Value Front") {
    auto const cfg      = Fermi::XtCfg();
    auto const opt_aeff = Fermi::load_aeff(cfg.aeff_file);
    auto const oexpmap  = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto const owexpmap
        = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    auto const oen = Fermi::fits::ccube_energies(cfg.cmap);
    REQUIRE(opt_aeff);
    REQUIRE(oexpmap);
    REQUIRE(owexpmap);
    REQUIRE(oen);
    auto const aeff          = opt_aeff.value();
    auto const logEs         = Fermi::log10_v(oen.value());
    auto const exp_costhetas = Fermi::exp_costhetas(oexpmap.value());

    auto front_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff.front.effective_area);
    auto back_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff.back.effective_area);

    et2comp(front_aeff, FermiTest::aeff_front_c_e);
    et2comp(back_aeff, FermiTest::aeff_back_c_e);
}

TEST_CASE("Test Exposure") {
    auto const cfg      = Fermi::XtCfg();
    auto const opt_aeff = Fermi::load_aeff(cfg.aeff_file);
    auto const oexpmap  = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto const owexpmap
        = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    auto const oen  = Fermi::fits::ccube_energies(cfg.cmap);
    auto const srcs = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs = Fermi::spherical_coords_from_point_sources(srcs);
    REQUIRE(opt_aeff);
    REQUIRE(oexpmap);
    REQUIRE(owexpmap);
    REQUIRE(oen);
    auto const aeff      = opt_aeff.value();
    auto const logEs     = Fermi::log10_v(oen.value());
    auto const front_LTF = Fermi::livetime_efficiency_factors(
        logEs, aeff.front.efficiency_params);
    // auto back_LTF  = Fermi::lt_effic_factors(logEs,
    // aeff.back.efficiency_params);

    for (long i = 0; i < front_LTF.extent(1); ++i) {
        CHECK(front_LTF[0, i]
              == doctest::Approx(FermiTest::Aeff::livetime_factor1[i]));
        CHECK(front_LTF[1, i]
              == doctest::Approx(FermiTest::Aeff::livetime_factor2[i]));
    }

    auto const exp_costhetas        = Fermi::exp_costhetas(oexpmap.value());
    auto const wexp_costhetas       = Fermi::exp_costhetas(owexpmap.value());
    auto const exp_map              = Fermi::exp_map(oexpmap.value());
    auto const wexp_map             = Fermi::exp_map(owexpmap.value());
    auto const src_exposure_cosbins = Fermi::src_exp_cosbins(dirs, exp_map);
    auto const src_weighted_exposure_cosbins
        = Fermi::src_exp_cosbins(dirs, wexp_map);
    size_t const Nsrc = src_exposure_cosbins.extent(0);
    /* size_t const Nc   = src_exposure_cosbins.extent(1); */
    size_t const Ne   = front_LTF.extent(1);

    auto const front_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff.front.effective_area);
    auto const back_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff.back.effective_area);
    SUBCASE("EXPECTED AEFF") {
        et2comp(front_aeff, FermiTest::aeff_front_c_e);
        et2comp(back_aeff, FermiTest::aeff_back_c_e);
    }

    // auto const exp_aeff_f = Fermi::contract210(src_exposure_cosbins,
    // front_aeff);
    /* Tensor2d const exp_aeff_f */
    /*     = front_aeff.contract(src_exposure_cosbins, IdxPair1 { { { 1, 0 } }
     * }); */
    // ExpC[s, e] = Sum_c (ECB[s, c] * Aeff[c, e])
    Tensor2d const exp_aeff_f /*(Nsrc, Ne)*/
        = Fermi::exp_contract(src_exposure_cosbins, front_aeff);
    SUBCASE("EXPECTED FRONT EXPOSURE VALUE AFTER CONTRACTION WITH COSBINS") {
        REQUIRE(exp_aeff_f.extent(0) == Nsrc);
        REQUIRE(exp_aeff_f.extent(1) == Ne);
        et2comp(exp_aeff_f, FermiTest::Aeff::Front::exp_aeff);
    }

    // auto const exp_aeff_b = Fermi::contract210(src_exposure_cosbins,
    // back_aeff);
    Tensor2d const exp_aeff_b
        /* = back_aeff.contract(src_exposure_cosbins, IdxPair1 { { { 1, 0 } }
           }); */
        = Fermi::exp_contract(src_exposure_cosbins, back_aeff);
    SUBCASE("EXPECTED BACK EXPOSURE VALUE AFTER CONTRACTION WITH COSBINS") {
        REQUIRE(exp_aeff_b.extent(0) == Nsrc);
        REQUIRE(exp_aeff_b.extent(1) == Ne);
        et2comp(exp_aeff_b, FermiTest::Aeff::Back::exp_aeff);
    }

    // auto const wexp_aeff_f
    //     = Fermi::contract210(src_weighted_exposure_cosbins, front_aeff);
    Tensor2d const wexp_aeff_f
        = Fermi::exp_contract(src_weighted_exposure_cosbins, front_aeff);
    SUBCASE("EXPECTED FRONT WEIGHTED_EXPOSURE VALUE AFTER CONTRACTION WITH "
            "COSBINS") {
        REQUIRE(wexp_aeff_f.extent(0) == Nsrc);
        REQUIRE(wexp_aeff_f.extent(1) == Ne);
        et2comp(wexp_aeff_f, FermiTest::Aeff::Front::wexp_aeff);
    }

    // auto const wexp_aeff_b
    //     = Fermi::contract210(src_weighted_exposure_cosbins, back_aeff);
    Tensor2d const wexp_aeff_b
        = Fermi::exp_contract(src_weighted_exposure_cosbins, back_aeff);
    SUBCASE("EXPECTED BACK WEIGHTED_EXPOSURE VALUE AFTER CONTRACTION WITH "
            "COSBINS") {
        REQUIRE(wexp_aeff_b.extent(0) == Nsrc);
        REQUIRE(wexp_aeff_b.extent(1) == Ne);
        et2comp(wexp_aeff_b, FermiTest::Aeff::Back::wexp_aeff);
    }

    SUBCASE("EXPECTED EXPOSURE BIN MATCHES EXPOSURE TEXTFILE") {
        const size_t sz_exp = 9994;
        REQUIRE(FermiTest::expected_exposure.size() == sz_exp);
        Tensor2d const stexp = Fermi::read_file_tensor(
            "./tests/expected/exposure.bin", 263uz, 38uz);

        et2comp(stexp, FermiTest::expected_exposure);
    }

    /* TensorMap<Tensor2d const> LTFe(front_LTF.first.data(), Ne, 1); */
    /* TensorMap<Tensor2d const> LTFw(front_LTF.second.data(), Ne, 1); */
    auto LTFe = front_LTF.slice({ 0, 0 }, { 1, Ne }).broadcast({ Nsrc, Ne });
    auto LTFw = front_LTF.slice({ 1, 0 }, { 1, Ne }).broadcast({ Nsrc, Ne });

    // auto const lef = Fermi::mul210(exp_aeff_f, front_LTF.first);
    Tensor2d const lef = exp_aeff_f * LTFe;
    SUBCASE("EXPECTED FRONT EXPOSURE VALUE AFTER LTF SCALING") {
        filecomp(lef, "exposure_lef");
    }

    // auto const lwf        = Fermi::mul210(wexp_aeff_f, LTFw);
    Tensor2d const lwf = wexp_aeff_f * LTFw;
    SUBCASE("EXPECTED FRONT WEIGHTED_EXPOSURE VALUE AFTER LTF SCALING") {
        filecomp(lwf, "exposure_lwf");
    }

    // auto const leb        = Fermi::mul210(exp_aeff_b, LTFe);
    Tensor2d const leb = exp_aeff_b * LTFe;
    SUBCASE("EXPECTED BACK EXPOSURE VALUE AFTER LTF SCALING (Front LTF "
            "apparently).") {
        filecomp(leb, "exposure_leb");
    }

    // auto const lwb        = Fermi::mul210(wexp_aeff_b, LTFw);
    Tensor2d const lwb = wexp_aeff_b * LTFw;
    SUBCASE(
        "EXPECTED BACK WEIGHTED_EXPOSURE VALUE AFTER LTF SCALING (Front LTF "
        "apparently).") {
        filecomp(lwb, "exposure_lwb");
    }

    Tensor2d const exposure = Fermi::exposure(src_exposure_cosbins,
                                              src_weighted_exposure_cosbins,
                                              front_aeff,
                                              back_aeff,
                                              front_LTF);
    SUBCASE("EXPECTED EXPOSURE!") { filecomp(exposure, "exposure"); }
}
