#include <xtsrcmaps/config.hxx>
#include <xtsrcmaps/exposure.hxx>
#include <xtsrcmaps/fitsfuncs.hxx>
#include <xtsrcmaps/irf.hxx>
#include <xtsrcmaps/misc.hxx>
#include <xtsrcmaps/parse_src_mdl.hxx>
#include <xtsrcmaps/psf.hxx>
#include <xtsrcmaps/source.hxx>
#include <xtsrcmaps/source_utils.hxx>

#include <fmt/format.h>
#include <xtsrcmaps/fmt_source.hxx>

#include <algorithm>
#include <numeric>
#include <vector>

using std::vector;

int
main()
{
    // : parse-cli : Parse the command line arguments, parfiles, etc.
    // Just use st_app? hoops? ape?
    // rewrite hoops ape locally?
    // No. Goal is to prototype a faster srcmaps, not rewrite stapp+hoops+ape.
    // Just use command line parameters and a pre-defined struct.
    auto cfg          = Fermi::XtCfg();

    // : parse-xml : Parse the XML source model file.
    auto srcs         = Fermi::parse_src_xml(cfg.srcmdl);
    // fmt::print("{}\n", fmt::join(srcs, "\n"));
    auto dirs         = Fermi::directions_from_point_sources(srcs);
    // fmt::print("{}\n", fmt::join(dirs, "\n"));

    // : load-counts : Load the fits file count maps
    auto opt_energies = Fermi::fits::ccube_energies(cfg.cmap);
    if (!opt_energies)
    {
        fmt::print("Cannot read ccube_energies file!\n");
        return 1;
    }
    auto energies = opt_energies.value();
    auto logEs    = vector<double>(energies.size(), 0.0);
    std::transform(energies.cbegin(),
                   energies.cend(),
                   logEs.begin(),
                   [](auto const& v) { return std::log10(v); });

    // fmt::print("CMap Energies: {}\n", fmt::join(energies, ", "));
    // fmt::print("CMap Log Enrg: {}\n", fmt::join(logEs, ", "));

    // skipping ROI cuts.
    // skipping edisp_bin expansion.

    // : load-exposure : Load the fits file exposure maps
    auto opt_exp_map  = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto opt_wexp_map = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    if (!opt_exp_map || !opt_wexp_map)
    {
        fmt::print("Cannot read exposure cube map file table!\n");
        return 1;
    }
    auto exp_map      = Fermi::exp_map(opt_exp_map.value());
    auto wexp_map     = Fermi::exp_map(opt_wexp_map.value());
    auto exp_cosbins  = Fermi::src_exp_cosbins(dirs, exp_map);
    auto wexp_cosbins = Fermi::src_exp_cosbins(dirs, wexp_map);

    //********************************************************************************
    // Read IRF Fits Files.
    //********************************************************************************
    auto opt_aeff     = Fermi::load_aeff(cfg.aeff_name);
    auto opt_psf      = Fermi::load_psf(cfg.psf_name);
    if (!opt_aeff || !opt_psf) { return 1; }

    auto aeff = opt_aeff.value();
    auto psf  = opt_psf.value();
    // fmt::print("aeff.front.effective_area.cosths \n{:.25}\n",
    //            fmt::join(aeff.front.effective_area.cosths, ", "));
    // fmt::print("aeff.front.effective_area.logEs \n{:.25}\n",
    //            fmt::join(aeff.front.effective_area.logEs, ", "));
    // fmt::print("aeff.front.effective_area.params \n{}\n",
    //            aeff.front.effective_area.params);

    // auto exp_a_f = Fermi::aeff_value(exp_costheta, logEs, aeff.front.effective_area);
    // auto exp_a_b = Fermi::aeff_value(exp_costheta, logEs, aeff.back.effective_area);

    // // // : compute-psf : Compute the actual PSF
    // // Need to figure out how to determine if the phiDepPars or m_usePhiDependence
    // // parameters are set. If so this calculation can be skipped entirely and just
    // // the unmodulated Aeff value used.
    // auto exp_phid = Fermi::phi_mod(exp_costheta, logEs, aeff_phidep, false);
    // auto expo     = Fermi::exposure(exp_area, exp_phid, exp_costheta);
    // fmt::print("expo: {:+4.2g}\n",
    //            fmt::join(expo.container().begin(), expo.container().end(), " "));
    //
    // auto seps  = Fermi::separations(1e-4, 70.0, 400);
    // auto kings = Fermi::psf_fixed_grid(seps, psf_rpsf);
    // fmt::print("kings: {}\n", std::reduce(kings.begin(), kings.end(), 0.0));
    //
    // // auto bilerps = Fermi::bilerp(kings, logEs, exp_costheta, psf_rpsf);
    // // fmt::print("bilerps: {}\n", std::reduce(bilerps.begin(), bilerps.end(), 0.0));
    //
    // // : convolve-psf :
    //
    // // : write-results : write the results back out to file
    // return 0;
}
