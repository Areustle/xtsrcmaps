#include <xtsrcmaps/config.hxx>
#include <xtsrcmaps/fitsfuncs.hxx>
#include <xtsrcmaps/parse_src_mdl.hxx>
#include <xtsrcmaps/psf.hxx>
#include <xtsrcmaps/source.hxx>
#include <xtsrcmaps/source_utils.hxx>

#include <fmt/format.h>
#include <xtsrcmaps/fmt_source.hxx>

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
    // : load-psf parameters
    auto opt_psfpars  = Fermi::fits::read_psf(cfg._psf_name);
    auto raw_psfpars  = opt_psfpars.value();

    // skipping ROI cuts.
    // skipping edisp_bin expansion.

    // : load-exposure : Load the fits file exposure maps
    // auto exp_costheta = std::vector<double>(dirs.size() * 40, 1.0);

    // : compute-psf : Compute the actual PSF
    // std::vector<std::pair<double, double>> dirs(8, { 1.0, 1.0 });
    // std::vector<double>                    energies(8, 10.0);
    auto psfdata      = Fermi::prepare_psf_data(raw_psfpars);
    auto moffats      = Fermi::psf_fixed_grid(psfdata);
    fmt::print("{} {}\n", moffats.front(), moffats.back());

    // : convolve-psf :

    // : write-results : write the results back out to file
    return 0;
}
