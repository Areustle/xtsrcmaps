#pragma once

#include <string>
#include <variant>
#include <vector>

namespace Fermi
{

struct SourceParameter
{
    std::string   name  = "";
    unsigned long free  = 0ul;
    float         value = 0.0f;
    float         scale = 1.0f;
    float         min   = 0.0f;
    float         max   = 0.0f;
};

/*************************************************************************************
 * The various Spectrum types, here defined as sepearate structs and unified into the
 * arithmetic sum type "Fermi::Spectrum" with a std::variant (effectively a memory-safe)
 * union.
 ************************************************************************************/

enum class SpectrumType
{
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

struct NormalSpectrum
{
    std::vector<SourceParameter> params = {};
    SpectrumType                 type   = SpectrumType::Unknown;
};

struct FileSpectrum
{
    std::string                  file   = "";
    std::vector<SourceParameter> params = {};
    bool                         edisp  = false;
};

using Spectrum = std::variant<NormalSpectrum, FileSpectrum>;

/*************************************************************************************
 * The various SpatialModel types, here defined as sepearate structs and unified into
 * the arithmetic sum type "Fermi::SpatialModel" with a std::variant (effectively a
 * memory-safe) union.
 ************************************************************************************/

struct ConstantValueSpatialModel
{
    std::vector<SourceParameter> params = {};
};

struct MapCubeFunctionSpatialModel
{
    std::string                  file   = "";
    std::vector<SourceParameter> params = {};
};

struct SkyDirFunctionSpatialModel
{
    std::vector<SourceParameter> params = {};
};

struct UnknownSpatialModel
{
};

using SpatialModel = std::variant<ConstantValueSpatialModel,
                                  MapCubeFunctionSpatialModel,
                                  SkyDirFunctionSpatialModel,
                                  UnknownSpatialModel>;

/*************************************************************************************
 * The various Source types, here defined as sepearate structs and unified into the
 * arithmetic sum type "Fermi::Source" with a std::variant (effectively a memory-safe)
 * union.
 ************************************************************************************/

struct PointSource
{
    std::string  name            = "";
    Spectrum     spectrum        = {};
    SpatialModel spatial_model   = {};
    float        roi_center_dist = 0.0;
};

struct DiffuseSource
{
    std::string  name          = "";
    Spectrum     spectrum      = {};
    SpatialModel spatial_model = {};
};

struct CompositeSource
{
    std::string  name          = "";
    Spectrum     spectrum      = {};
    SpatialModel spatial_model = {};
};

struct UnknownSource
{
    std::string name = "";
};

using Source = std::variant<PointSource, DiffuseSource, CompositeSource, UnknownSource>;

} // namespace Fermi
