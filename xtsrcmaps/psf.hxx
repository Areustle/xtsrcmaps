#pragma once

#include <vector>

#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/tensor_types.hxx"

namespace Fermi::PSF
{

///////////////////////////////////////////////////////////////////////////////////////
/// Given a PSF IRF grid and a set of separations, compute the King/Moffat results for
/// every entry in the table and every separation.
///////////////////////////////////////////////////////////////////////////////////////
auto
king(std::vector<double> const& deltas, irf::psf::Data const& data)
    -> mdarray3; //[Nd, Me, Mc]


auto
separations(double const xmin = 1e-4, double const xmax = 70., size_t const N = 400)
    -> std::vector<double>;

auto
inverse_separations(double const s) -> double;

// auto
// psf_fixed_grid(std::vector<double> const& deltas, IrfData3 const& pars)
//     -> std::vector<double>;
//
auto
bilerp(std::vector<double> const& costhetas,
       std::vector<double> const& logEs,
       std::vector<double> const& par_cosths,
       std::vector<double> const& par_logEs,
       mdarray3 const&            kings) -> mdarray3;

auto
corrected_exposure_psf(
    mdarray3 const& obs_psf,
    mdarray2 const& obs_aeff,
    mdarray2 const& src_exposure_cosbins,
    mdarray2 const& src_weighted_exposure_cosbins,
    std::pair<std::vector<double>, std::vector<double>> const& front_LTF /*[Ne]*/
    ) -> mdarray3;

auto
mean_psf(                                //
    mdarray3 const& front_corrected_psf, /*[Nd, Nc, Ne]*/
    mdarray3 const& back_corrected_psf,  /*[Nd, Nc, Ne]*/
    mdarray2 const& exposure /*[Ns, Ne]*/) -> mdarray3;

auto
partial_total_integral(std::vector<double> const& deltas, mdarray3 const& mean_psf)
    -> std::pair<mdarray3, mdarray2>;

auto
integral(std::vector<double> deltas,
         mdarray3 const&     partial_integrals,
         mdarray3 const&     mean_psf) -> mdarray3;

auto
normalize(mdarray3& mean_psf, mdarray2 const& total_integrals) -> void;

auto
peak_psf(mdarray3 const& mean_psf) -> mdarray2;


/* Compute the SourceMap model values for a Point source.
   This version is called for WCS-based Counts Maps

   pointSrc    : The source in question
   dataMap     : The counts map in question
   energies    : The vector of energies to consider
   config      : Parameters for PSF integration
   meanpsf     : The average PSF across the ROI
   formatter   : Stream for writting progress messages
   modelmap    : Filled with the model values
   mapType     : Enum that specifies how to store the map
   kmin        : Minimum energy layer
   kmax        : Maximum energy layer

   return 0 for success, error code otherwise
*/

auto
makePointSourceMap_wcs( // const PointSource&          pointSrc,
                        //  const CountsMap&            dataMap,
    const std::vector<double>& energies,
    // const PsfIntegConfig&       config,
    const mdarray3& meanpsf,
    // const BinnedExposureBase*   bexpmap,
    // st_stream::StreamFormatter& formatter,
    std::vector<float>& modelmap,
    // FileUtils::SrcMapType&      mapType,
    int kmin = 0,
    int kmax = -1) -> std::vector<double>;
//
//
//
} // namespace Fermi::PSF
