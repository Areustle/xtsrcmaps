#pragma once

#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/skyimage/skyimage.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

#include "fitsio.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Fermi {
namespace fits {

auto safe_open(std::string const& filename)
    -> std::unique_ptr<fitsfile, std::function<void(fitsfile*)>>;

// Error handling function
template <typename T = int>
auto
handleError(T status) {
    if (status) {
        fits_report_error(stderr, status);
        throw std::runtime_error("FITSIO error occurred");
    }
};

// template <typename T = double>
// auto
// makeSkyGeom(WcsConfig const& m) -> SkyGeom<T> {
//     return { m.proj_name, m.is_galactic, m.crpix[0], m.crpix[1], m.crval[0],
//              m.crval[1],  m.cdelt[0],    m.cdelt[1], m.axis_rot };
// }

auto read_energies(std::string const&) noexcept
    -> std::optional<std::vector<double>>;

auto read_image_meta(std::string const&) noexcept -> std::optional<WcsConfig>;


auto read_expcube(std::string const&,
                  std::string const&) -> std::optional<Obs::ExposureCubeData>;

struct TablePars {
    std::vector<size_t> extents;
    std::vector<size_t> offsets;
    Tensor<float, 2>    rowdata;
};

auto read_irf_pars(std::string const&, std::string) -> std::optional<TablePars>;

auto write_src_model(std::string const&              filename,
                     Tensor<double, 4>&              model_map,
                     std::vector<std::string> const& srcs) -> void;

auto
read_allsky_cropped(SkyGeom<double> const& roiGeom,
                    std::string const& allskyfilename) -> SkyImage<float, 3>;

} // namespace fits
} // namespace Fermi
