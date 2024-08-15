#include "xtsrcmaps/irf/irf.hxx"

#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/math/gauss_legendre.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <cassert>
#include <cmath>
#include <mdspan>


#include "fmt/color.h"

using std::mdspan;
using std::sin;
using std::string;
using std::vector;


auto
Fermi::collect_irf_data(XtCfg const& cfg, XtObs const& obs) -> XtIrf {

    fmt::print(fg(fmt::color::light_pink),
               "Collecting Instrument Response Functions.\n");

    //**************************************************************************
    // Read IRF Fits Files.
    //**************************************************************************
    auto opt_aeff  = Fermi::load_aeff(cfg.aeff_file);
    auto opt_psf   = Fermi::load_psf(cfg.psf_file);
    auto aeff_irf  = good(opt_aeff, "Cannot read AEFF Irf FITS file!");
    auto psf_irf   = good(opt_psf, "Cannot read PSF Irf FITS file!");

    auto front_LTF = Fermi::livetime_efficiency_factors(
        obs.logEs, aeff_irf.front.efficiency_params);

    return { .aeff_irf = aeff_irf, .psf_irf = psf_irf, .front_LTF = front_LTF };
}

//************************************************************************************
// Convert the raw arrays read in from an IRF FITS file into a useable set
// of gridded data tensors for future sampling.
//
// The format of an IRF grid is a single FITS row with 5 or more columns.
// The first 4 columns are the low and high columns of the energy and
// costheta range vectors. Every subsequent column is a data entry for 1
// paramater entry in a linearized grid.
//
// column 0: energy low
// column 1: energy high
// column 2: costheta low
// column 3: costheta high
// column 4: Data parameter in linearized (costheta x energy) grid.
// column n: Data parameter in linearized (costheta x energy) grid.
//************************************************************************************
auto
prepare_grid(Fermi::fits::TablePars const& pars) -> Fermi::IrfData3 {

    assert(pars.extents.size() >= 5);
    assert(pars.extents[0] > 1);
    assert(pars.extents[0] == pars.extents[1]);
    assert(pars.extents[2] > 1);
    assert(pars.extents[2] == pars.extents[3]);
    assert(pars.extents[4] == pars.extents[0] * pars.extents[2]);
    assert(pars.rowdata.extent(0) == 1uz);

    auto const& extents               = pars.extents;
    auto const& offsets               = pars.offsets;
    /* Fermi::Tensor<float, 1> const row */
    /*     = pars.rowdata.reshape(Idx1 { pars.rowdata.dimension(0) }); */

    size_t const             M_e_base = extents[0];
    size_t const             M_e      = extents[0] + 2;
    size_t const             M_t_base = extents[2];
    size_t const             M_t      = extents[2] + 2;
    size_t const             off_cos0 = offsets[2];
    size_t const             off_cos1 = offsets[3];
    Fermi::Tensor<double, 1> cosths(M_t);
    /* cosths.clear(); */
    // Arithmetic mean of cosine bins
    auto const cos0view = std::span { &pars.rowdata[0, off_cos0], M_t_base };
    auto const cos1view = std::span { &pars.rowdata[0, off_cos1], M_t_base };
    std::transform(cos0view.begin(),
                   cos0view.end(),
                   cos1view.begin(),
                   cosths.begin(),
                   [](auto const& c0, auto const& c1) {
                       return 0.5 * (c0 + c1);
                   });
    /* cosths.slice(Idx1 { 1 }, Idx1 { M_t_base }) */
    /*     = 0.5 */
    /*       * (row.slice(Idx1 { off_cos0 }, Idx1 { M_t_base }) */
    /*          + row.slice(Idx1 { off_cos1 }, Idx1 { M_t_base })) */
    /*             .cast<double>(); */
    cosths[0]                        = -1.;
    cosths[M_t - 1]                  = 1.;

    // scale and pad the energy data
    size_t const             off_Es0 = offsets[0];
    size_t const             off_Es1 = offsets[1];
    Fermi::Tensor<double, 1> logEs(M_e);
    /* logEs.(); */
    auto const Es0view = std::span { &pars.rowdata[0, off_Es0], M_e_base };
    auto const Es1view = std::span { &pars.rowdata[0, off_Es1], M_e_base };
    std::transform(Es0view.begin(),
                   Es0view.end(),
                   Es1view.begin(),
                   logEs.begin(),
                   [](auto const& c0, auto const& c1) {
                       return 0.5 * std::log10((c0 * c1));
                   });
    /* logEs.slice(Idx1 { 1 }, Idx1 { M_e_base }) */
    /*     = (row.slice(Idx1 { off_Es0 }, Idx1 { M_e_base }) */
    /*        * row.slice(Idx1 { off_Es1 }, Idx1 { M_e_base })) */
    /*           .sqrt() */
    /*           .unaryExpr([](float x) -> float { return std::log10(x); }) */
    /*           .cast<double>(); */
    logEs[0]                        = 0.;
    logEs[M_e - 1]                  = 10.;

    size_t const             Ngrids = extents.size() - 4;
    Fermi::Tensor<double, 3> params(M_t, M_e, Ngrids);
    std::mdspan              pv {
        pars.rowdata.data() + offsets[4], Ngrids, M_t_base, M_e_base
    };
    /* TensorMap<Tensor3f const> pv( */
    /*     row.data() + offsets[4], M_e_base, M_t_base, Ngrids); */

    // Let's assign the data values into the params block structure. Pad by
    // value duplication.
    for (size_t p = 0; p < Ngrids; ++p) {  // params
        for (size_t t = 0; t < M_t; ++t) { // costheta
            size_t const t_ = t == 0 ? 0 : t >= M_t_base ? M_t_base - 1 : t - 1;
            for (size_t e = 0; e < M_e; ++e) { // energy
                size_t const e_ = e == 0          ? 0
                                  : e >= M_e_base ? M_e_base - 1
                                                  : e - 1;
                params[t, e, p] = pv[p, t_, e_];
            }
        }
    }

    return { cosths, logEs, params, pars.rowdata[0, off_cos0] };
}

auto
prepare_scale(Fermi::fits::TablePars const& pars) -> Fermi::IrfScale {
    assert(pars.rowdata.extent(0) == 1);
    assert(pars.rowdata.extent(1) == 3);

    return { pars.rowdata[0, 0], pars.rowdata[0, 1], pars.rowdata[0, 2] };
}

auto
prepare_effic(Fermi::fits::TablePars const& pars) -> Fermi::IrfEffic {

    assert(pars.extents.size() == 1);
    assert(pars.extents[0] == 6);
    assert(pars.rowdata.extent(0) == 2);
    assert(pars.rowdata.extent(1) == 6);

    auto p0 = std::array<float, 6> { 0.0 };
    auto p1 = std::array<float, 6> { 0.0 };
    std::copy(&pars.rowdata[0, 0], &pars.rowdata[0, 6], p0.begin());
    std::copy(&pars.rowdata[1, 0], &pars.rowdata[1, 6], p1.begin());

    return { p0, p1 };
}

auto
normalize_rpsf(Fermi::irf::psf::Data& psfdata) -> void {

    Fermi::IrfData3&       data  = psfdata.rpsf;
    Fermi::IrfScale const& scale = psfdata.psf_scaling_params;
    // Next normalize and scale.

    auto scaleFactor             = [sp0 = (scale.scale0 * scale.scale0),
                        sp1 = (scale.scale1 * scale.scale1),
                        si  = scale.scale_index](double const energy) {
        double const tt = std::pow(energy / 100., si);
        return std::sqrt(sp0 * tt * tt + sp1);
    };

    // An integration is required below, so let's precompute the orthogonal
    // legendre polynomials here for future use.
    auto const polypars = Fermi::legendre_poly_rw<64>(1e-15);

    for (size_t c = 0; c < data.params.extent(0); ++c) // costheta
    {
        for (size_t e = 0; e < data.params.extent(1); ++e) // energy
        {
            double const energy = std::pow(10.0, data.logEs[e]);
            double const sf     = scaleFactor(energy);
            double const norm
                = energy < 120. //
                      ? Fermi::gauss_legendre_integral(
                            0.0,
                            90.,
                            polypars,
                            [&](auto const& v) -> double {
                                double x = Fermi::evaluate_king(
                                    v * deg2rad,
                                    sf,
                                    std::span { &data.params[c, e, 0], 6 });
                                double y = sin(v * deg2rad) * twopi * deg2rad;
                                return x * y;
                            })
                      : Fermi::psf3_psf_base_integral(
                            90.0, sf, std::span { &data.params[c, e, 0], 6 });

            data.params[c, e, 0uz] /= norm;
            data.params[c, e, 2uz] *= sf;
            data.params[c, e, 3uz] *= sf;
            data.params[c, e, 4uz]
                = data.params[c, e, 4uz] == 1. ? 1.001 : data.params[c, e, 4uz];
            data.params[c, e, 5uz]
                = data.params[c, e, 5uz] == 1. ? 1.001 : data.params[c, e, 5uz];
        }
    }
};

//************************************************************************************
//************************************************************************************
auto
prepare_psf_data(Fermi::fits::TablePars const& front_rpsf,
                 Fermi::fits::TablePars const& front_scaling,
                 Fermi::fits::TablePars const& front_fisheye,
                 Fermi::fits::TablePars const& back_rpsf,
                 Fermi::fits::TablePars const& back_scaling,
                 Fermi::fits::TablePars const& back_fisheye)
    -> Fermi::irf::psf::Pass8FB {

    auto front = Fermi::irf::psf::Data { prepare_grid(front_rpsf),
                                         prepare_scale(front_scaling),
                                         prepare_grid(front_fisheye) };
    auto back  = Fermi::irf::psf::Data { prepare_grid(back_rpsf),
                                        prepare_scale(back_scaling),
                                        prepare_grid(back_fisheye) };

    normalize_rpsf(front);
    normalize_rpsf(back);

    return { front, back };
}

auto
prepare_aeff_data(Fermi::fits::TablePars const& front_eff_area,
                  Fermi::fits::TablePars const& front_phi_dep,
                  Fermi::fits::TablePars const& front_effici,
                  Fermi::fits::TablePars const& back_eff_area,
                  Fermi::fits::TablePars const& back_phi_dep,
                  Fermi::fits::TablePars const& back_effici)
    -> Fermi::irf::aeff::Pass8FB {
    auto front = Fermi::irf::aeff::Data { prepare_grid(front_eff_area),
                                          prepare_grid(front_phi_dep),
                                          prepare_effic(front_effici) };
    auto back  = Fermi::irf::aeff::Data { prepare_grid(back_eff_area),
                                         prepare_grid(back_phi_dep),
                                         prepare_effic(back_effici) };


    return { front, back };
}

auto
read_opt(auto&&             F,
         std::string const& filename,
         std::string const& tablename) -> auto {
    auto opt_irf = F(filename, tablename);
    if (!opt_irf) fmt::print("Cannot read {} table {}!\n", filename, tablename);
    return opt_irf;
}

//************************************************************************************
//************************************************************************************
auto
Fermi::load_aeff(std::string const& filename)
    -> std::optional<irf::aeff::Pass8FB> {
    auto f   = fits::read_irf_pars;
    auto o0f = read_opt(f, filename, "EFFECTIVE AREA_FRONT");
    auto o1f = read_opt(f, filename, "PHI_DEPENDENCE_FRONT");
    auto o2f = read_opt(f, filename, "EFFICIENCY_PARAMS_FRONT");
    auto o0b = read_opt(f, filename, "EFFECTIVE AREA_BACK");
    auto o1b = read_opt(f, filename, "PHI_DEPENDENCE_BACK");
    auto o2b = read_opt(f, filename, "EFFICIENCY_PARAMS_BACK");

    if (o0f && o1f && o2f && o0b && o1b && o2b) {
        return { prepare_aeff_data(o0f.value(),
                                   o1f.value(),
                                   o2f.value(),
                                   o0b.value(),
                                   o1b.value(),
                                   o2b.value()) };
    } else {
        return std::nullopt;
    }
}

auto
Fermi::load_psf(std::string const& filename)
    -> std::optional<irf::psf::Pass8FB> {
    auto f   = fits::read_irf_pars;
    auto o0f = read_opt(f, filename, "RPSF_FRONT");
    auto o1f = read_opt(f, filename, "PSF_SCALING_PARAMS_FRONT");
    auto o2f = read_opt(f, filename, "FISHEYE_CORRECTION_FRONT");
    auto o0b = read_opt(f, filename, "RPSF_BACK");
    auto o1b = read_opt(f, filename, "PSF_SCALING_PARAMS_BACK");
    auto o2b = read_opt(f, filename, "FISHEYE_CORRECTION_BACK");

    if (o0f && o1f && o2f && o0b && o1b && o2b) {
        return { prepare_psf_data(o0f.value(),
                                  o1f.value(),
                                  o2f.value(),
                                  o0b.value(),
                                  o1b.value(),
                                  o2b.value()) };
    } else {
        return std::nullopt;
    }
}

auto
Fermi::livetime_efficiency_factors(vector<double> const& logEs,
                                   IrfEffic const&       effic)
    -> Tensor<double, 2> /* [2, Ne] */ {
    // pair<vector<double>, vector<double>> {

    auto single_factor = [](auto const& p) -> auto {
        auto const&  a0     = p.at(0);
        auto const&  b0     = p.at(1);
        auto const&  a1     = p.at(2);
        auto const&  logEb1 = p.at(3);
        auto const&  a2     = p.at(4);
        auto const&  logEb2 = p.at(5);
        double const b1     = (a0 - a1) * logEb1 + b0;
        double const b2     = (a1 - a2) * logEb2 + b1;

        return [=](double const logE) -> double {
            return logE < logEb1   ? a0 * logE + b0
                   : logE < logEb2 ? a1 * logE + b1
                                   : a2 * logE + b2;
        };
    };

    /* auto lf1 = vector<double>(logEs.size(), 0.); */
    /* auto lf0 = vector<double>(logEs.size(), 0.); */
    Tensor<double, 2> ltf(2UZ, logEs.size());
    // Intentionally ordered that way. ¯\_(ツ)_/¯
    std::transform(logEs.cbegin(),
                   logEs.cend(),
                   ltf.begin_at(1, 0),
                   single_factor(effic.p0));
    std::transform(logEs.cbegin(),
                   logEs.cend(),
                   ltf.begin_at(0, 0),
                   single_factor(effic.p1));

    return ltf; // [2, Ne]
    // Intentionally ordered that way. ¯\_(ツ)_/¯
    /* return { lf1, lf0 }; */
}
