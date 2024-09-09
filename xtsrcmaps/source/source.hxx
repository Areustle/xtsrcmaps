#pragma once

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <concepts>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace Fermi {
namespace Source {

struct SourceParameter {
    std::string   name  = "";
    unsigned long free  = 0ul;
    float         value = 0.0f;
    float         scale = 1.0f;
    float         min   = 0.0f;
    float         max   = 0.0f;
};

/*******************************************************************************
 * The various Spectrum types, here defined as sepearate structs and unified
 *into the arithmetic sum type "Fermi::Spectrum" with a std::variant
 *(effectively a memory-safe) union.
 ******************************************************************************/

enum class SpectrumType {
    BPLExpCutoff,
    BrokenPowerLaw,
    BrokenPowerLaw2,
    ConstantValue,
    ExpCutoff,
    FileFunction,
    Gaussian,
    LogParabola,
    PowerLaw,
    PowerLaw2,
    PLSuperExpCutoff2,
    PLSuperExpCutoff3,
    PLSuperExpCutoff4,
    Unknown,
};

struct NormalSpectrum {
    std::vector<SourceParameter> params = {};
    SpectrumType                 type   = SpectrumType::Unknown;
};

struct FileSpectrum {
    std::string                  file   = "";
    std::vector<SourceParameter> params = {};
    bool                         edisp  = false;
};

using Spectrum = std::variant<NormalSpectrum, FileSpectrum>;

/*******************************************************************************
 * The various SpatialModel types, here defined as sepearate structs and unified
 *into the arithmetic sum type "Fermi::SpatialModel" with a std::variant
 *(effectively a memory-safe) union.
 ******************************************************************************/

struct ConstantValueSpatialModel {
    std::vector<SourceParameter> params = {};
};

struct MapCubeFunctionSpatialModel {
    std::string                  file   = "";
    std::vector<SourceParameter> params = {};
};

struct SkyDirFunctionSpatialModel {
    std::vector<SourceParameter> params = {};
};

struct UnknownSpatialModel {};

using SpatialModel = std::variant<ConstantValueSpatialModel,
                                  MapCubeFunctionSpatialModel,
                                  SkyDirFunctionSpatialModel,
                                  UnknownSpatialModel>;

/*******************************************************************************
 * The various Source types, here defined as sepearate structs and unified into
 *the arithmetic sum type "Fermi::Source" with a std::variant (effectively a
 *memory-safe) union.
 ******************************************************************************/

struct PointSource {
    std::string  name            = "";
    Spectrum     spectrum        = {};
    SpatialModel spatial_model   = {};
    float        roi_center_dist = 0.0;
};

struct DiffuseSource {
    std::string  name          = "";
    Spectrum     spectrum      = {};
    SpatialModel spatial_model = {};
};

struct CompositeSource {
    std::string  name          = "";
    Spectrum     spectrum      = {};
    SpatialModel spatial_model = {};
};

struct UnknownSource {
    std::string name = "";
};

using Source
    = std::variant<PointSource, DiffuseSource, CompositeSource, UnknownSource>;

// Concept to constrain templates to be one of the allowed Fermi::Source types
template <typename T>
concept SourceConcept
    = std::same_as<T, PointSource> || std::same_as<T, DiffuseSource>
      || std::same_as<T, CompositeSource> || std::same_as<T, UnknownSource>;


template <SourceConcept T>
struct SourceData {
    std::vector<T>           srcs;
    Fermi::Tensor<double, 2> sph_locs;
    std::vector<std::string> names;
};


struct XtSrc {
    SourceData<PointSource>   point;
    SourceData<DiffuseSource> diffuse;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Utility Functions ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////


auto
spherical_coords(std::vector<PointSource> const&) -> Fermi::Tensor<double, 2>;
auto
spherical_coords(std::vector<DiffuseSource> const&,
                 std::pair<double, double> const) -> Fermi::Tensor<double, 2>;


/* auto names_from_point_sources(std::vector<Source> const&) */
/*     -> std::vector<std::string>; */
template <SourceConcept T>
auto
source_names(std::vector<T> const& srcs) -> std::vector<std::string> {

    auto names = std::vector<std::string>();
    std::transform(srcs.cbegin(),
                   srcs.cend(),
                   std::back_inserter(names),
                   [](auto const& s) -> std::string { return s.name; });
    return names;
}


struct SourceGroup {
    std::vector<PointSource>   point;
    std::vector<DiffuseSource> diffuse;
};

auto parse_src_xml(std::string const& src_file_name) -> SourceGroup;

auto collect_source_model(Config::XtCfg const&, Obs::XtObs const&) -> XtSrc;

} // namespace Source
} // namespace Fermi
