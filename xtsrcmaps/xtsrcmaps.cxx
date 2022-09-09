#include <xtsrcmaps/config.hxx>
#include <xtsrcmaps/fitsfuncs.hxx>
#include <xtsrcmaps/irf.hxx>
#include <xtsrcmaps/misc.hxx>
#include <xtsrcmaps/parse_src_mdl.hxx>
#include <xtsrcmaps/psf.hxx>
#include <xtsrcmaps/source.hxx>
#include <xtsrcmaps/source_utils.hxx>

#include <fmt/format.h>
#include <xtsrcmaps/fmt_source.hxx>

#include <numeric>
#include <ranges>
#include <vector>

using std::vector;
using std::ranges::views::transform;

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
    auto energies   = opt_energies.value();
    auto logEs      = to<vector<double>>(energies | transform(::log10));

    // skipping ROI cuts.
    // skipping edisp_bin expansion.

    // : load-exposure : Load the fits file exposure maps
    auto opt_ltcube = Fermi::fits::read_ltcube(cfg.expcube);
    if (!opt_ltcube)
    {
        fmt::print("Cannot read ltcube file!\n");
        return 1;
    }
    auto ltcube = opt_ltcube.value();
    // auto exp_costheta = vector<double>(dirs.size() * 40);
    // for (size_t i = 0; i < exp_costheta.size(); ++i)
    // {
    //     exp_costheta[i] = double(i) / double(exp_costheta.size());
    // }
    // fmt::print("cosths: {}\n",
    //            std::reduce(exp_costheta.cbegin(), exp_costheta.cend(), 0.0));

    auto opt_aeff_area
        = Fermi::fits::read_irf_grid(cfg.aeff_name, "EFFECTIVE AREA_FRONT");
    if (!opt_aeff_area)
    {
        fmt::print("Cannot read Aeff table EFFECTIVE AREA_FRONT!\n");
        return 1;
    }
    auto aeff_area = opt_aeff_area.value();
    //
    auto opt_aeff_phidep
        = Fermi::fits::read_irf_grid(cfg.aeff_name, "PHI_DEPENDENCE_FRONT");
    if (!opt_aeff_phidep)
    {
        fmt::print("Cannot read Aeff table PHI_DEPENDENCE_FRONT!\n");
        return 1;
    }
    auto aeff_phidep  = opt_aeff_phidep.value();
    //
    auto opt_psf_rpsf = Fermi::fits::read_irf_grid(cfg.psf_name, "RPSF_FRONT");
    if (!opt_psf_rpsf)
    {
        fmt::print("Cannot read PSF table RPSF_FRONT!\n");
        return 1;
    }
    auto psf_rpsf = opt_psf_rpsf.value();
    //
    auto opt_psf_scale
        = Fermi::fits::read_irf_scale(cfg.psf_name, "PSF_SCALING_PARAMS_FRONT");
    if (!opt_psf_scale)
    {
        fmt::print("Cannot read PSF table PSF_SCALING_PARAMS_FRONT!\n");
        return 1;
    }
    auto psf_scale = opt_psf_scale.value();

    // : load-psf parameters
    // auto opt_psfpars = Fermi::fits::read_psf(cfg.psf_name);
    // auto raw_psfpars = opt_psfpars.value();
    // : compute-psf : Compute the actual PSF
    // auto psfdata     = Fermi::prepare_psf_data(raw_psfpars);
    auto psfdata   = Fermi::prepare_irf_data(psf_rpsf);
    Fermi::normalize_irf_data(psfdata, psf_scale);

    fmt::print("psfdata: {}\t",
               std::reduce(psfdata.logEs.begin(), psfdata.logEs.end(), 0.0));
    fmt::print("{}\t", std::reduce(psfdata.cosths.begin(), psfdata.cosths.end(), 0.0));
    fmt::print("{}\n", std::reduce(psfdata.params.begin(), psfdata.params.end(), 0.0));

    // auto kings   = Fermi::psf_fixed_grid(psfdata);
    // fmt::print("kings: {}\n", std::reduce(kings.begin(), kings.end(), 0.0));

    // auto bilerps = Fermi::bilerp(kings, logEs, exp_costheta, psfdata);
    // fmt::print("bilerps: {}\n", std::reduce(bilerps.begin(), bilerps.end(), 0.0));

    // : convolve-psf :

    // : write-results : write the results back out to file
    return 0;
}
