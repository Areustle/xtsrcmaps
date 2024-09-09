#pragma once

#include <optional>
#include <string>

namespace Fermi {
namespace Config {

// Configuration struct for xtsrcmaps
//
struct XtCfg {
    //"Spacecraft data extension"
    std::string sctable   = "SC_DATA";
    //"Exposure hypercube file"
    std::string expcube   = "analysis/3C279_binned_ltcube.fits";
    //"Counts map file"
    std::string cmap      = "analysis/3C279_binned_ccube.fits";
    //"Source model file"
    std::string srcmdl    = "analysis/3C279_input_model_point.xml";
    //"Binned exposure map"
    std::string bexpmap   = "analysis/3C279_binned_allsky_expcube.fits";
    //"Likelihood weights map"
    std::string wmap      = "none";
    //"Source maps output file"
    std::string outfile   = "analysis/3C279_binned_srcmaps_local.fits";
    //"Response functions"
    /* std::string irfs          = "CALDB"; */
    std::string psf_file  = "analysis/psf_P8R3_SOURCE_V2_FB.fits";
    std::string aeff_file = "analysis/aeff_P8R3_SOURCE_V2_FB.fits";
    /* double      minbinsz      = 0.1; // "Minimum pixel size for rebinning
     * fine maps" */
    /* int         evtype        = -1;  //"Event type selections" */
    /* int         rfactor       = 2;   //"Resampling factor" */
    /* // "Number of extra bins to compute for energy dispersion purposes" */
    /* int  edisp_bins           = 0; */
    /* bool convol               = true; //"Perform convolution with psf" */
    /* bool resample             = true; //,,,"Resample input counts map for
     * convolution" */
    /* bool ptsrc                = true; //"Compute point source maps" */
    /* bool psfcorr              = true; //"Apply psf integral corrections" */
    /* bool emapbnds             = true; //"Enforce boundaries of exposure map"
     */
    /* // Copy all source maps from input counts map file to output" */
    /* bool copyall              = false; */
};

auto parse_parfile(const std::string& filename) -> std::optional<XtCfg>;

auto validate_cfg(XtCfg const& ocfg) -> XtCfg;

} // namespace Config
} // namespace Fermi
