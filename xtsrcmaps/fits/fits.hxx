#pragma once

#include "xtsrcmaps/math/tensor_types.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/source/source.hxx"

#include <optional>
#include <string>
#include <vector>

namespace Fermi {
namespace fits {


auto ccube_energies(std::string const&) noexcept
    -> std::optional<std::vector<double>>;

auto
ccube_pixels(std::string const&) noexcept -> std::optional<Obs::CCubePixels>;


auto read_expcube(std::string const&, std::string const&)
    -> std::optional<Obs::ExposureCubeData>;

struct TablePars {
    std::vector<size_t> extents;
    std::vector<size_t> offsets;
    Tensor2f            rowdata;
};

auto read_irf_pars(std::string const&, std::string const&)
    -> std::optional<TablePars>;

auto write_src_model(std::string const&                filename,
                     Tensor4f const&                   model_map,
                     std::vector<Fermi::Source> const& srcs) -> void;


} // namespace fits
} // namespace Fermi
