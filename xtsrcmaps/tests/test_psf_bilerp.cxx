#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include <fstream>

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/exposure.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/king.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/source_utils.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

auto
veq(auto v1, auto v2) -> void
{
    REQUIRE(v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); ++i) REQUIRE(v1[i] == doctest::Approx(v2[i]));
}

auto
filecomp3(mdarray3 const& computed, std::string const& filebase) -> void
{
    const size_t sz_exp = computed.extent(0) * computed.extent(1) * computed.extent(2);
    REQUIRE(computed.size() == sz_exp);
    auto expected = std::vector<double>(sz_exp);

    std::ifstream ifs("./xtsrcmaps/tests/expected/" + filebase + ".bin",
                      std::ios::in | std::ios::binary);
    ifs.read((char*)(&expected[0]), sizeof(double) * sz_exp);
    ifs.close();

    // md2comp(computed, expected);
    REQUIRE(computed.size() == expected.size());
    auto sp_b = std::experimental::mdspan(
        expected.data(), computed.extent(0), computed.extent(1), computed.extent(2));
    for (size_t i = 0; i < computed.extent(0); ++i)
        for (size_t j = 0; j < computed.extent(1); ++j)
            for (size_t k = 0; k < computed.extent(2); ++k)
                REQUIRE_MESSAGE(computed(i, j, k) == doctest::Approx(sp_b(i, j, k)),
                                i << " " << j << " "
                                  << " " << k << " " << filebase);
}

TEST_CASE("Test Psf_bilerp")
{
    auto const cfg     = Fermi::XtCfg();
    auto const opt_psf = Fermi::load_psf(cfg.psf_name);
    auto const oexpmap = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto const oen     = Fermi::fits::ccube_energies(cfg.cmap);
    auto const srcs    = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs    = Fermi::directions_from_point_sources(srcs);
    REQUIRE(opt_psf);
    REQUIRE(oexpmap);
    REQUIRE(oen);
    auto const psf           = opt_psf.value();
    auto const logEs         = Fermi::log10_v(oen.value());
    auto const exp_costhetas = Fermi::exp_costhetas(oexpmap.value());
    // auto const exp_map              = Fermi::exp_costhetas(oexpmap.value());
    // auto const src_exposure_cosbins = Fermi::src_exp_cosbins(dirs, exp_map);

    auto const separations   = Fermi::PSF::separations(1e-4, 70., 400);
    auto const kings         = Fermi::king(separations, psf.front);
    auto const psf_vals      = Fermi::PSF::bilerp(
        exp_costhetas, logEs, psf.front.rpsf.cosths, psf.front.rpsf.logEs, kings);

    filecomp3(psf_vals, "psf_val_front");
}
