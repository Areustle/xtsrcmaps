#include "gtest/gtest.h"

#include <sstream>

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/irf/irf.hxx"
/* #include "xtsrcmaps/model_map/mm_cubature/convolve_pixel_psf_searchsep.hpp"
 */
#include "xtsrcmaps/model_map/mm_cubature/convolve_pixel_psf_logsep.hpp"
#include "xtsrcmaps/model_map/mm_cubature/cubature.hpp"
#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/observation/observation.hxx"
#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"
#include "xtsrcmaps/tensor/write_file_tensor.hpp"

#include "tests/fermi_tests.hxx"

class TestModelMap : public ::testing::Test {
  protected:
    Fermi::XtCfg           cfg = Fermi::XtCfg();
    Fermi::XtObs           obs;
    Fermi::XtIrf           irf;
    Fermi::XtExp           exp;
    Fermi::XtPsf           psf;
    Fermi::SkyGeom<double> skygeom;

    size_t                  Ns;
    size_t                  Nh;
    size_t                  Nw;
    size_t                  Ne;
    static constexpr size_t NPIX  = 64;
    static constexpr double dstep = 1.0 / static_cast<double>(NPIX);

    void SetUp() override {
        cfg.srcmdl = "./analysis/single_point.xml";
        obs        = Fermi::collect_observation_data(cfg);
        irf        = Fermi::collect_irf_data(cfg, obs);
        exp        = Fermi::compute_exposure_data(cfg, obs, irf);
        psf        = Fermi::PSF::compute_psf_data(obs, irf, exp);
        skygeom    = { obs.ccube };
        Ns         = obs.src_sph.extent(0);
        Nh         = obs.Nh;
        Nw         = obs.Nw;
        Ne         = psf.uPsf.extent(2);
    }
};

/* TEST_F(TestModelMap, NaiveWeights4096) { */
/**/
/*     Fermi::Tensor<double, 4> model_map({ Ns, Nh, Nw, Ne }); */
/*     model_map.clear(); */
/**/
/*     std::string const filename = "./tests/expected/mm_pixel_epixxy_4096.bin";
 */
/*     auto              expected = Fermi::read_file_tensor( */
/*         filename, std::array<size_t, 5> { Nh, Nw, NPIX, NPIX, 2uz }); */
/**/
/*     for (size_t h = 0; h < Nw; ++h) { */
/*         for (size_t w = 0; w < Nw; ++w) { */
/**/
/*             double vh = 1. + h; */
/*             double vw = 1. + w; */
/**/
/*             for (size_t p = 0; p < NPIX * NPIX; ++p) { */
/*                 size_t i = p / NPIX; */
/*                 size_t j = p % NPIX; */
/*                 double u = vh + (-0.5 + (i + 0.5) * dstep); */
/*                 double v = vw + (-0.5 + (j + 0.5) * dstep); */
/**/
/*                 ASSERT_TRUE(NearRelative(u, expected[w, h, i, j, 0], 1e-12))
 */
/*                     << h << " " << w << " | " << i << " " << j << std::endl;
 */
/*                 ASSERT_TRUE(NearRelative(v, expected[w, h, i, j, 1], 1e-12))
 */
/*                     << h << " " << w << " | " << i << " " << j << std::endl;
 */
/*             } */
/*         } */
/*     } */
/* } */


/* TEST_F(TestModelMap, NaiveDisplacement4096) { */
/**/
/*     Fermi::Tensor<double, 4> model_map({ Ns, Nh, Nw, Ne }); */
/*     model_map.clear(); */
/**/
/*     std::string const filename =
 * "./tests/expected/mm_pixel_offsets_4096.bin"; */
/*     auto              expected = Fermi::read_file_tensor( */
/*         filename, std::array<size_t, 4> { Nw, Nh, 64, 64 }); */
/*     Fermi::Tensor<double, 4> computed({ Nh, Nw, 64, 64 }); */
/**/
/*     auto points_weights = Fermi::naive_4096_ptswts(Nh, Nw, skygeom); */
/*     auto source     = skygeom.sph2dir({ obs.src_sph[0, 0], obs.src_sph[0, 1]
 * }); */
/*     auto const seps = Fermi::PSF::separations(1e-4, 70.); */
/*     for (size_t h = 0; h < Nh; ++h) { */
/*         for (size_t w = 0; w < Nw; ++w) { */
/*             for (size_t i = 0; i < 64; ++i) { */
/*                 for (size_t j = 0; j < 64; ++j) { */
/*                     size_t p = (i * 64 + j) * 4; */
/**/
/*                     // Spherical Distance */
/*                     double x = points_weights[h, w, p] - source[0]; */
/*                     double y = points_weights[h, w, p + 1] - source[1]; */
/*                     double z = points_weights[h, w, p + 2] - source[2]; */
/*                     double d = 0.5f * std::sqrt(x * x + y * y + z * z); */
/*                     double s = 2.0f * rad2deg * std::asin(d); */
/*                     computed[h, w, i, j] = s; */
/*                 } */
/*             } */
/*         } */
/*     } */
/**/
/*     for (size_t h = 0; h < Nh; ++h) { */
/*         for (size_t w = 0; w < Nw; ++w) { */
/*             for (size_t i = 0; i < 64; ++i) { */
/*                 for (size_t j = 0; j < 64; ++j) { */
/*                     ASSERT_TRUE(NearRelative( */
/*                         computed[h, w, i, j], expected[w, h, i, j], 1e-6));
 */
/*                 } */
/*             } */
/*         } */
/*     } */
/* } */


/* TEST_F(TestModelMap, ModelMap4096psf_value) { */
/**/
/*     Fermi::Tensor<double, 4> model_map({ 1uz, Nw, Nh, Ne }); */
/*     model_map.clear(); */
/**/
/*     auto points_weights = Fermi::naive_4096_ptswts(Nh, Nw, skygeom); */
/*     auto source = skygeom.sph2dir({ obs.src_sph[0, 0], obs.src_sph[0, 1] });
 */
/**/
/* #pragma omp parallel for schedule(static, 16) */
/*     for (size_t w = 0; w < Nw; ++w) { */
/*         for (size_t h = 0; h < Nh; ++h) { */
/*             Fermi::convolve_pixel_psf_searchsep<double, 4096>( */
/*                 &(model_map[0, w, h, 0]), */
/*                 Ne, */
/*                 &(points_weights[h, w, 0]), */
/*                 &(psf.uPsf[0, 0, 0]), */
/*                 source); */
/*         } */
/*     } */
/**/
/*     filecomp<double, double, 4>(model_map, "mm_psf_value_4096", 1e-5, 1e-5);
 */
/* } */


TEST_F(TestModelMap, ModelMapSquareDeg7psf_value) {

    Fermi::Tensor<double, 4> model_map({ 1uz, Nw, Nh, Ne });
    model_map.clear();

    /* auto points_weights = Fermi::deg12_asymsquare_ptswts(Nh, Nw, skygeom); */
    auto points_weights = Fermi::square_ptswts(
        Nh, Nw, skygeom, Fermi::CubatureSets::square_deg7);
    auto source = skygeom.sph2dir({ obs.src_sph[0, 0], obs.src_sph[0, 1] });

    /* #pragma omp parallel for schedule(static, 16) */
    for (size_t w = 0; w < Nw; ++w) {
        for (size_t h = 0; h < Nh; ++h) {
            Fermi::convolve_pixel_psf_logsep<double,
                                             Fermi::CubatureSets::N_SQUARE_D7>(
                &(model_map[0, w, h, 0]),
                Ne,
                &(points_weights[h, w, 0]),
                &(psf.uPsf[0, 0, 0]),
                source);
        }
    }

    /* Fermi::write_file_tensor<double, 4>("tests/expected/mm_sqd3.double.bin",
     */
    /*                                     model_map); */

    filecomp<double, double, 4>(model_map, "mm_psf_value_4096", 1e-5, 1e-5);
}


/* TEST_F(TestModelMap, ModelMapArpistDeg4psf_value) { */
/**/
/*     Fermi::Tensor<double, 4> model_map({ 1uz, Nw, Nh, Ne }); */
/*     model_map.clear(); */
/**/
/*     auto points_weights = Fermi::deg4x2_arpist_ptswts(Nh, Nw, skygeom); */
/*     auto source = skygeom.sph2dir({ obs.src_sph[0, 0], obs.src_sph[0, 1] });
 */
/**/
/*     for (size_t w = 0; w < Nw; ++w) { */
/*         for (size_t h = 0; h < Nh; ++h) { */
/*             Fermi::convolve_pixel_psf_searchsep<double, 12>( */
/*                 &(model_map[0, w, h, 0]), */
/*                 Ne, */
/*                 &(points_weights[h, w, 0]), */
/*                 &(psf.uPsf[0, 0, 0]), */
/*                 source); */
/*         } */
/*     } */
/**/
/*     Fermi::write_file_tensor<double, 4>( */
/*         "tests/expected/mm_arpist4_single_src.double.bin", model_map); */
/**/
/*     filecomp<double, double, 4>(model_map, "mm_psf_value_4096", 1e-3, 1e-2);
 */
/* } */
