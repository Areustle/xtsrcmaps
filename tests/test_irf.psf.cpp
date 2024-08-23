#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/fits/fits.hxx"
/* #include "xtsrcmaps/irf/irf.hxx" */
#include "xtsrcmaps/irf/_irf_private.hpp"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include "tests/fermi_tests.hxx"

#include <algorithm> // For std::transform
#include <cctype>    // For std::toupper

void
test_psf_irf(std::string direction) {
    // Convert the direction to uppercase for IRF naming consistency
    std::transform(
        direction.begin(), direction.end(), direction.begin(), ::toupper);

    // Convert direction to lowercase for filename consistency
    std::string direction_lower = direction;
    std::transform(direction_lower.begin(),
                   direction_lower.end(),
                   direction_lower.begin(),
                   ::tolower);

    auto        cfg       = Fermi::XtCfg();
    std::string rpsf_name = "RPSF_" + direction;
    auto o_rpsf_raw       = Fermi::fits::read_irf_pars(cfg.psf_file, rpsf_name);
    REQUIRE(o_rpsf_raw);
    auto rpsf_pars = o_rpsf_raw.value();

    REQUIRE(rpsf_pars.extents.size() == 10);

    Fermi::IrfData3 const rpsf = irf_private::prepare_grid(rpsf_pars);

    size_t const Mt            = 10;
    size_t const Me            = 25;

    SUBCASE("PSF IRF Derived cosths") {
        CHECK(rpsf.cosths.extent(0) == Mt);
        auto thetas
            = Fermi::Tensor<float, 1> { 180.0,   75.5225, 69.5127, 63.2563,
                                        56.633,  49.4584, 41.4096, 31.7883,
                                        18.1949, 0.0 };

        for (size_t i = 0; i < Mt; ++i)
            CHECK_MESSAGE(rpsf.cosths[i]
                              == doctest::Approx(std::cos(deg2rad * thetas[i])),
                          i);
    }

    SUBCASE("PSF IRF Derived logEs") {
        CHECK(rpsf.logEs.extent(0) == Me);
        auto energies = Fermi::Tensor<float, 1> {
            1,       7.49894, 13.3352,     23.7137,     42.1696,
            74.9894, 133.352, 237.137,     421.696,     749.894,
            1333.52, 2371.37, 4216.96,     7498.94,     13335.2,
            23713.7, 42169.7, 74989.4,     133352,      237137,
            421697,  749894,  1.33352e+06, 2.37137e+06, 1e+10
        };

        for (size_t i = 0; i < Me; ++i)
            CHECK_MESSAGE(
                rpsf.logEs[i] == doctest::Approx(std::log10(energies[i])), i);
    }

    SUBCASE("PSF IRF Derived Params") {
        CHECK(rpsf.params.extent(0) == Mt);
        CHECK(rpsf.params.extent(1) == Me);
        REQUIRE(rpsf.params.extent(2) == 6);

        filecomp<double>(rpsf.params, "psf3_rpsf." + direction_lower);
    }

    std::string scalepar_name = "PSF_SCALING_PARAMS_" + direction;
    auto        o_scalepar_raw
        = Fermi::fits::read_irf_pars(cfg.psf_file, scalepar_name);
    REQUIRE(o_scalepar_raw);
    auto scalepar = o_scalepar_raw.value();

    auto psfdata  = Fermi::irf::psf::Data {
        rpsf,
        irf_private::prepare_scale(scalepar),
    };

    irf_private::normalize_rpsf(psfdata);

    SUBCASE("PSF IRF Normalized Params") {
        CHECK(psfdata.rpsf.params.extent(0) == Mt);
        CHECK(psfdata.rpsf.params.extent(1) == Me);
        REQUIRE(psfdata.rpsf.params.extent(2) == 6);
        auto const sp_b = Fermi::read_file_tensor(
            "./tests/expected/psf3_normalized_rpsf." + direction_lower + ".bin",
            psfdata.rpsf.params.extents());

        Fermi::IrfData3&       data  = psfdata.rpsf;
        Fermi::IrfScale const& scale = psfdata.psf_scaling_params;

        auto scaleFactor             = [sp0 = (scale.scale0 * scale.scale0),
                            sp1 = (scale.scale1 * scale.scale1),
                            si  = scale.scale_index](double const energy) {
            double const tt = std::pow(energy * 1.e-2, si);
            return std::sqrt(sp0 * tt * tt + sp1);
        };

        for (long t = 0l; t < Mt; ++t) {
            for (long e = 0l; e < Me; ++e) {
                //
                double const energy = std::pow(10.0, data.logEs[e]);
                double const sf     = scaleFactor(energy);
                REQUIRE(psfdata.rpsf.params[t, e, 0]
                        == doctest::Approx(sp_b[t, e, 0]));
                REQUIRE_MESSAGE((psfdata.rpsf.params[t, e, 0]
                                 == doctest::Approx(sp_b[t, e, 0])),
                                t << " " << e);
                REQUIRE(psfdata.rpsf.params[t, e, 1]
                        == doctest::Approx(sp_b[t, e, 1]));
                REQUIRE_MESSAGE((psfdata.rpsf.params[t, e, 1]
                                 == doctest::Approx(sp_b[t, e, 1])),
                                t << " " << e);
                //
                REQUIRE(psfdata.rpsf.params[t, e, 2]
                        == doctest::Approx(sf * sp_b[t, e, 2]));
                REQUIRE_MESSAGE((psfdata.rpsf.params[t, e, 2]
                                 == doctest::Approx(sf * sp_b[t, e, 2])),
                                t << " " << e);
                REQUIRE(psfdata.rpsf.params[t, e, 3]
                        == doctest::Approx(sf * sp_b[t, e, 3]));
                REQUIRE_MESSAGE((psfdata.rpsf.params[t, e, 3]
                                 == doctest::Approx(sf * sp_b[t, e, 3])),
                                t << " " << e);
                //
                REQUIRE(psfdata.rpsf.params[t, e, 4]
                        == doctest::Approx(sp_b[t, e, 4]));
                REQUIRE_MESSAGE((psfdata.rpsf.params[t, e, 4]
                                 == doctest::Approx(sp_b[t, e, 4])),
                                t << " " << e);
                REQUIRE(psfdata.rpsf.params[t, e, 5]
                        == doctest::Approx(sp_b[t, e, 5]));
                REQUIRE_MESSAGE((psfdata.rpsf.params[t, e, 5]
                                 == doctest::Approx(sp_b[t, e, 5])),
                                t << " " << e);
            }
        }
    }
}

TEST_CASE("Load PSF IRF FRONT") { test_psf_irf("FRONT"); }

TEST_CASE("Load PSF IRF BACK") { test_psf_irf("BACK"); }
