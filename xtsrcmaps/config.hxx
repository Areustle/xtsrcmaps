#pragma once

#include <string>

namespace Fermi
{

// Configuration struct for xtsrcmaps
//
struct XtCfg
{
    std::string analysis_base = "/home/areustle/nasa/fermi/xtsrcmaps/analysis/";
    std::string scfile        = "";        //,,,"Spacecraft data file"
    std::string sctable       = "SC_DATA"; //,,,Spacecraft data extension
    //"Exposure hypercube file"
    std::string expcube       = analysis_base + "3C279_binned_ltcube.fits";
    //"Counts map file"
    std::string cmap          = analysis_base + "3C279_binned_ccube.fits";
    //"Source model file"
    std::string srcmdl        = analysis_base + "3C279_input_model_point.xml";
    //"Binned exposure map"
    std::string bexpmap       = analysis_base + "3C279_binned_allsky_expcube.fits";
    //"Likelihood weights map"
    std::string wmap          = "none";
    //"Source maps output file"
    std::string outfile       = analysis_base + "3C279_binned_srcmaps_local.fits";
    std::string irfs          = "CALDB"; //"Response functions"
    std::string psf_name      = analysis_base + "psf_P8R3_SOURCE_V2_FB.fits";
    std::string aeff_name     = analysis_base + "aeff_P8R3_SOURCE_V2_FB.fits";
    double      minbinsz      = 0.1; // "Minimum pixel size for rebinning fine maps"
    int         evtype        = -1;  //"Event type selections"
    int         rfactor       = 2;   //"Resampling factor"
    // "Number of extra bins to compute for energy dispersion purposes"
    int  edisp_bins           = 0;
    bool convol               = true; //"Perform convolution with psf"
    bool resample             = true; //,,,"Resample input counts map for convolution"
    bool ptsrc                = true; //"Compute point source maps"
    bool psfcorr              = true; //"Apply psf integral corrections"
    bool emapbnds             = true; //"Enforce boundaries of exposure map"
    // Copy all source maps from input counts map file to output"
    bool copyall              = false;
};

} // namespace Fermi
