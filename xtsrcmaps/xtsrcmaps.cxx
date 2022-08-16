#include <xtsrcmaps/config.hxx>
#include <xtsrcmaps/fitsfuncs.hxx>
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
    auto energies     = opt_energies.value();
    auto logEs        = to<vector<double>>(energies | transform(::log10));

    // skipping ROI cuts.
    // skipping edisp_bin expansion.

    // : load-exposure : Load the fits file exposure maps
    auto opt_ltcube   = Fermi::fits::read_ltcube(cfg.expcube).value();
    auto exp_costheta = vector<double>(dirs.size() * 40);
    for (size_t i = 0; i < exp_costheta.size(); ++i)
    {
        exp_costheta[i] = double(i) / double(exp_costheta.size());
    }
    fmt::print("cosths: {}\n",
               std::reduce(exp_costheta.cbegin(), exp_costheta.cend(), 0.0));

    // : load-psf parameters
    auto opt_psfpars = Fermi::fits::read_psf(cfg._psf_name);
    auto raw_psfpars = opt_psfpars.value();
    // : compute-psf : Compute the actual PSF
    // vector<std::pair<double, double>> dirs(8, { 1.0, 1.0 });
    auto psfdata     = Fermi::prepare_psf_data(raw_psfpars);
    fmt::print("psfdata: {}\t",
               std::reduce(psfdata.logEs.begin(), psfdata.logEs.end(), 0.0));
    fmt::print("{}\t", std::reduce(psfdata.cosths.begin(), psfdata.cosths.end(), 0.0));
    fmt::print("{}\n", std::reduce(psfdata.params.begin(), psfdata.params.end(), 0.0));
    auto kings = Fermi::psf_fixed_grid(psfdata);
    fmt::print("kings: {}\n", std::reduce(kings.begin(), kings.end(), 0.0));
    auto bilerps = Fermi::bilerp(kings, logEs, exp_costheta, psfdata);
    fmt::print("bilerps: {}\n", std::reduce(bilerps.begin(), bilerps.end(), 0.0));

    // : convolve-psf :

    // : write-results : write the results back out to file
    return 0;
}
