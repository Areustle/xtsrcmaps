#include "xtsrcmaps/irf.hxx"

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/gauss_legendre.hxx"
#include "xtsrcmaps/misc.hxx"

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <utility>

#include <fmt/format.h>
// #include <xtsrcmaps/fmt_source.hxx>

using std::pair;
using std::sin;
using std::string;
using std::vector;

//************************************************************************************
// Convert the raw arrays read in from an IRF FITS file into a useable set of gridded
// data tensors for future sampling.
//
// The format of an IRF grid is a single FITS row with 5 or more columns. The first 4
// columns are the low and high columns of the energy and costheta range vectors. Every
// subsequent column is a data entry for 1 paramater entry in a linearized grid.
//
// column 0: energy low
// column 1: energy high
// column 2: costheta low
// column 3: costheta high
// column 4: Data parameter in linearized (costheta x energy) grid.
// column n: Data parameter in linearized (costheta x energy) grid.
//************************************************************************************
auto
prepare_grid(Fermi::fits::TablePars const& pars) -> Fermi::IrfData3
{

    assert(pars.extents.size() >= 5);
    assert(pars.extents[0] > 1);
    assert(pars.extents[0] == pars.extents[1]);
    assert(pars.extents[2] > 1);
    assert(pars.extents[2] == pars.extents[3]);
    assert(pars.extents[4] == pars.extents[0] * pars.extents[2]);
    assert(pars.rowdata.dimension(1) == 1);

    auto const&    extents = pars.extents;
    auto const&    offsets = pars.offsets;
    Tensor1f const row     = pars.rowdata.reshape(Idx1 { pars.rowdata.dimension(0) });

    long const M_e_base    = extents[0];
    long const M_e         = extents[0] + 2;
    long const M_t_base    = extents[2];
    long const M_t         = extents[2] + 2;



    // vector<double> cosths(M_t, 0.0);
    // for (long k(0); k < M_t_base; k++)
    // {
    //     // Arithmetic mean of cosine bins
    //     cosths[1 + k] = 0.5 * (row[offsets[2] + k] + row[offsets[3] + k]);
    // }
    // // padded cosine bin values.
    // cosths.front() = -1.0;
    // cosths.back()  = 1.0;
    long const off_cos0    = offsets[2];
    long const off_cos1    = offsets[3];
    Tensor1d   cosths(M_t);
    cosths.setZero();
    // Arithmetic mean of cosine bins
    cosths.slice(Idx1 { 1 }, Idx1 { M_t_base })
        = 0.5
          * (row.slice(Idx1 { off_cos0 }, Idx1 { M_t_base })
             + row.slice(Idx1 { off_cos1 }, Idx1 { M_t_base }))
                .cast<double>();
    cosths(0)          = -1.;
    cosths(M_t - 1)    = 1.;

    // // scale and pad the energy data
    // vector<double> logEs(M_e, 0.0);
    // for (long k(0); k < M_e_base; k++)
    // {
    //     // Geometric mean of energy bins
    //     logEs[1 + k] = std::log10(std::sqrt(row[offsets[0] + k] * row[offsets[1] +
    //     k]));
    // }
    // // padded energy bin values.
    // logEs.front()   = 0.0;
    // logEs.back()    = 10.0;
    long const off_Es0 = offsets[0];
    long const off_Es1 = offsets[1];
    Tensor1d   logEs(M_e);
    logEs.setZero();
    logEs.slice(Idx1 { 1 }, Idx1 { M_e_base })
        = (row.slice(Idx1 { off_Es0 }, Idx1 { M_e_base })
           * row.slice(Idx1 { off_Es1 }, Idx1 { M_e_base }))
              .sqrt()
              .unaryExpr([](float x) -> float { return std::log10(x); })
              .cast<double>();
    logEs(0)        = 0.;
    logEs(M_e - 1)  = 10.;

    long     Ngrids = extents.size() - 4;
    Tensor3d params(Ngrids, M_e, M_t); // Note column major Tensor.

    TensorMap<Tensor3f const> pv(row.data() + offsets[4], M_e_base, M_t_base, Ngrids);

    // Let's assign the data values into the params block structure. Pad by value
    // duplication.
    for (long p = 0; p < Ngrids; ++p) // params
    {
        for (long t = 0; t < M_t; ++t) // costheta
        {
            long const t_ = t == 0 ? 0 : t >= M_t_base ? M_t_base - 1 : t - 1;
            for (long e = 0; e < M_e; ++e) // energy
            {
                long const e_   = e == 0 ? 0 : e >= M_e_base ? M_e_base - 1 : e - 1;
                params(p, e, t) = pv(e_, t_, p);
            }
        }
    }

    return { cosths, logEs, params, row(off_cos0) };
}

auto
prepare_scale(Fermi::fits::TablePars const& pars) -> Fermi::IrfScale
{
    assert(pars.rowdata.dimension(0) == 3);
    assert(pars.rowdata.dimension(1) == 1);

    return { pars.rowdata(0, 0), pars.rowdata(1, 0), pars.rowdata(2, 0) };
}

auto
prepare_effic(Fermi::fits::TablePars const& pars) -> Fermi::IrfEffic
{

    assert(pars.extents.size() == 1);
    assert(pars.extents[0] == 6);
    // assert(pars.rowdata.size() == 2);
    // assert(pars.rowdata[0].size() == 6);
    // assert(pars.rowdata[1].size() == 6);
    assert(pars.rowdata.dimension(0) == 6);
    assert(pars.rowdata.dimension(1) == 2);

    auto p0 = std::array<float, 6> { 0.0 };
    auto p1 = std::array<float, 6> { 0.0 };
    std::copy(&pars.rowdata(0, 0), &pars.rowdata(6, 0), p0.begin());
    std::copy(&pars.rowdata(0, 1), &pars.rowdata(6, 1), p1.begin());

    return { p0, p1 };
}

//
inline auto
evaluate_king(double const sep, double const scale_factor, Tensor1d const& pars)
    -> double
{
    auto f = [](double const u, double gamma) -> double {
        // ugly kluge because of sloppy programming in handoff_response
        // when setting boundaries of fit parameters for the PSF.
        if (gamma == 1) { gamma = 1.001; }
        return (1. - 1. / gamma) * std::pow(1. + u / gamma, -gamma);
    };

    double const ncore = pars(0);
    double const ntail = pars(1);
    double const score = pars(2) * scale_factor;
    double const stail = pars(3) * scale_factor;
    double const gcore = pars(4);
    double const gtail = pars(5);

    double rc          = sep / score;
    double uc          = rc * rc / 2.;

    double rt          = sep / stail;
    double ut          = rt * rt / 2.;
    return (ncore * f(uc, gcore) + ntail * ncore * f(ut, gtail));
}

inline auto
psf3_psf_base_integral(double const    radius,
                       double const    scale_factor,
                       Tensor1d const& pars) -> double
{
    auto f = [](double u, double gamma) -> double {
        double arg(1. + u / gamma);
        if (arg < 0) { throw std::runtime_error("neg. arg to pow"); }
        return 1. - std::pow(arg, 1. - gamma);
    };

    double const ncore = pars(0);
    double const ntail = pars(1);
    double const score = pars(2) * scale_factor;
    double const stail = pars(3) * scale_factor;
    double const gcore = pars(4);
    double const gtail = pars(5);

    double sep         = radius * deg2rad;
    double rc          = sep / score;
    double uc          = rc * rc / 2.;

    double rt          = sep / stail;
    double ut          = rt * rt / 2.;

    if (gcore < 0. || gtail < 0.) { throw std::runtime_error("gamma < 0"); }

    return (ncore * f(uc, gcore) * twopi * score * score
            + ntail * ncore * f(ut, gtail) * twopi * stail * stail);
}

auto
normalize_rpsf(Fermi::irf::psf::Data& psfdata) -> void
{

    Fermi::IrfData3&       data  = psfdata.rpsf;
    Fermi::IrfScale const& scale = psfdata.psf_scaling_params;
    // Next normalize and scale.

    auto scaleFactor             = [sp0 = (scale.scale0 * scale.scale0),
                        sp1 = (scale.scale1 * scale.scale1),
                        si  = scale.scale_index](double const energy) {
        double const tt = std::pow(energy / 100., si);
        return std::sqrt(sp0 * tt * tt + sp1);
    };

    // An integration is required below, so let's precompute the orthogonal legendre
    // polynomials here for future use.
    auto const polypars = Fermi::legendre_poly_rw<64>(1e-15);

    for (long c = 0; c < data.params.dimension(2); ++c) // costheta
    {
        for (long e = 0; e < data.params.dimension(1); ++e) // energy
        {
            double const energy = std::pow(10, data.logEs[e]);
            double const sf     = scaleFactor(energy);
            double const norm
                = energy < 120. //
                      ? Fermi::gauss_legendre_integral(
                          0.0,
                          90.,
                          polypars,
                          [&](auto const& v) -> double {
                              double x = evaluate_king(
                                  v * deg2rad,
                                  sf,
                                  data.params.slice(Idx3 { 0, e, c }, Idx3 { 6, 1, 1 })
                                      .reshape(Idx1 { 6 }));
                              double y = sin(v * deg2rad) * twopi * deg2rad;
                              return x * y;
                          })
                      : psf3_psf_base_integral(
                          90.0,
                          sf,
                          data.params.slice(Idx3 { 0, e, c }, Idx3 { 6, 1, 1 })
                              .reshape(Idx1 { 6 }));

            data.params(0, e, c) /= norm;
            data.params(2, e, c) *= sf;
            data.params(3, e, c) *= sf;
            data.params(4, e, c)
                = data.params(4, e, c) == 1. ? 1.001 : data.params(4, e, c);
            data.params(5, e, c)
                = data.params(5, e, c) == 1. ? 1.001 : data.params(5, e, c);
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
                 Fermi::fits::TablePars const& back_fisheye) -> Fermi::irf::psf::Pass8FB
{

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
    -> Fermi::irf::aeff::Pass8FB
{
    auto front = Fermi::irf::aeff::Data { prepare_grid(front_eff_area),
                                          prepare_grid(front_phi_dep),
                                          prepare_effic(front_effici) };
    auto back  = Fermi::irf::aeff::Data { prepare_grid(back_eff_area),
                                         prepare_grid(back_phi_dep),
                                         prepare_effic(back_effici) };


    return { front, back };
}

auto
read_opt(auto&& F, std::string const& filename, std::string const& tablename) -> auto
{
    auto opt_irf = F(filename, tablename);
    if (!opt_irf) fmt::print("Cannot read {} table {}!\n", filename, tablename);
    return opt_irf;
}

//************************************************************************************
//************************************************************************************
auto
Fermi::load_aeff(std::string const& filename) -> std::optional<irf::aeff::Pass8FB>
{
    auto f   = fits::read_irf_pars;
    auto o0f = read_opt(f, filename, "EFFECTIVE AREA_FRONT");
    auto o1f = read_opt(f, filename, "PHI_DEPENDENCE_FRONT");
    auto o2f = read_opt(f, filename, "EFFICIENCY_PARAMS_FRONT");
    auto o0b = read_opt(f, filename, "EFFECTIVE AREA_BACK");
    auto o1b = read_opt(f, filename, "PHI_DEPENDENCE_BACK");
    auto o2b = read_opt(f, filename, "EFFICIENCY_PARAMS_BACK");

    if (o0f && o1f && o2f && o0b && o1b && o2b)
    {
        return { prepare_aeff_data(o0f.value(),
                                   o1f.value(),
                                   o2f.value(),
                                   o0b.value(),
                                   o1b.value(),
                                   o2b.value()) };
    }
    else { return std::nullopt; }
}

auto
Fermi::load_psf(std::string const& filename) -> std::optional<irf::psf::Pass8FB>
{
    auto f   = fits::read_irf_pars;
    auto o0f = read_opt(f, filename, "RPSF_FRONT");
    auto o1f = read_opt(f, filename, "PSF_SCALING_PARAMS_FRONT");
    auto o2f = read_opt(f, filename, "FISHEYE_CORRECTION_FRONT");
    auto o0b = read_opt(f, filename, "RPSF_BACK");
    auto o1b = read_opt(f, filename, "PSF_SCALING_PARAMS_BACK");
    auto o2b = read_opt(f, filename, "FISHEYE_CORRECTION_BACK");

    if (o0f && o1f && o2f && o0b && o1b && o2b)
    {
        return { prepare_psf_data(o0f.value(),
                                  o1f.value(),
                                  o2f.value(),
                                  o0b.value(),
                                  o1b.value(),
                                  o2b.value()) };
    }
    else { return std::nullopt; }
}

auto
Fermi::lt_effic_factors(vector<double> const& logEs, IrfEffic const& effic)
    -> pair<vector<double>, vector<double>>
{
    auto single_factor = [](auto const& p) -> auto
    {
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

    auto lf1 = vector<double>(logEs.size(), 0.);
    auto lf0 = vector<double>(logEs.size(), 0.);
    std::transform(logEs.cbegin(), logEs.cend(), lf0.begin(), single_factor(effic.p0));
    std::transform(logEs.cbegin(), logEs.cend(), lf1.begin(), single_factor(effic.p1));

    // Intentionally ordered that way. ¯\_(ツ)_/¯
    return { lf1, lf0 };
}
