#pragma once

#include <xtsrcmaps/source.hxx>
#include <xtsrcmaps/tensor_types.hxx>

#include <fmt/format.h>

#include <string>

////////////////////////////////////////////////////////////////////////////////////////
/// Fermi::SourceParameter section.
////////////////////////////////////////////////////////////////////////////////////////

template <>
struct fmt::formatter<Fermi::SourceParameter>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::SourceParameter const& par, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "({}, {}, {}, {}, {}, {})",
                              par.name,
                              par.free,
                              par.value,
                              par.scale,
                              par.min,
                              par.max);
    }
};

////////////////////////////////////////////////////////////////////////////////////////
/// Fermi::Spectrum section.
////////////////////////////////////////////////////////////////////////////////////////

template <>
struct fmt::formatter<Fermi::SpectrumType>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::SpectrumType const& st, FormatContext& ctx)
    {

        return fmt::format_to(
            ctx.out(), "{}", [](Fermi::SpectrumType t) noexcept -> std::string {
                if (t == Fermi::SpectrumType::BPLExpCutoff) return "BPLExpCutoff";
                if (t == Fermi::SpectrumType::BrokenPowerLaw) return "BrokenPowerLaw";
                if (t == Fermi::SpectrumType::BrokenPowerLaw2) return "BrokenPowerLaw2";
                if (t == Fermi::SpectrumType::ConstantValue) return "ConstantValue";
                if (t == Fermi::SpectrumType::ExpCutoff) return "ExpCutoff";
                if (t == Fermi::SpectrumType::FileFunction) return "FileFunction";
                if (t == Fermi::SpectrumType::Gaussian) return "Gaussian";
                if (t == Fermi::SpectrumType::LogParabola) return "LogParabola";
                if (t == Fermi::SpectrumType::PowerLaw) return "PowerLaw";
                if (t == Fermi::SpectrumType::PowerLaw2) return "PowerLaw2";
                if (t == Fermi::SpectrumType::PLSuperExpCutoff2)
                    return "PLSuperExpCutoff2";
                if (t == Fermi::SpectrumType::PLSuperExpCutoff3)
                    return "PLSuperExpCutoff3";
                if (t == Fermi::SpectrumType::PLSuperExpCutoff4)
                    return "PLSuperExpCutoff4";
                if (t == Fermi::SpectrumType::Unknown) return "Unknown";
                return "";
            }(st));
    }
};

template <>
struct fmt::formatter<Fermi::NormalSpectrum>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::NormalSpectrum const& spec, FormatContext& ctx)
    {

        return fmt::format_to(ctx.out(),
                              "<NormalSpectrum> type: {} params[{}]",
                              spec.type,
                              fmt::join(spec.params, ", "));
    }
};

template <>
struct fmt::formatter<Fermi::FileSpectrum>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::FileSpectrum const& spec, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "<FileFunction> file: {}  edisp: {}  params[{}]",
                              spec.file,
                              spec.edisp,
                              fmt::join(spec.params, ", "));
    }
};


template <>
struct fmt::formatter<Fermi::Spectrum>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::Spectrum const& spec, FormatContext& ctx)
    {
        if (auto const* ns = std::get_if<Fermi::NormalSpectrum>(&spec))
        {
            return fmt::format_to(ctx.out(), "{}", *ns);
        }
        if (auto const* fs = std::get_if<Fermi::FileSpectrum>(&spec))
        {
            return fmt::format_to(ctx.out(), "{}", *fs);
        }
        return fmt::format_to(ctx.out(), "[]", "");
    }
};

////////////////////////////////////////////////////////////////////////////////////////
/// Fermi::SpatialModel section.
////////////////////////////////////////////////////////////////////////////////////////

template <>
struct fmt::formatter<Fermi::ConstantValueSpatialModel>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::ConstantValueSpatialModel const& v, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "<ConstantValueSpatialModel> params[{}]",
                              fmt::join(v.params, ", "));
    }
};

template <>
struct fmt::formatter<Fermi::MapCubeFunctionSpatialModel>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::MapCubeFunctionSpatialModel const& v, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "<MapCubeFunctionSpatialModel> file: {}  params[{}]",
                              v.file,
                              fmt::join(v.params, ", "));
    }
};

template <>
struct fmt::formatter<Fermi::SkyDirFunctionSpatialModel>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::SkyDirFunctionSpatialModel const& v, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "<SkyDirFunctionSpatialModel> params[{}]",
                              fmt::join(v.params, ", "));
    }
};

template <>
struct fmt::formatter<Fermi::UnknownSpatialModel>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::UnknownSpatialModel const& v, FormatContext& ctx)
    {
        (void)(v);
        return fmt::format_to(ctx.out(), "<UnknownSpatialModel>", "");
    }
};


template <>
struct fmt::formatter<Fermi::SpatialModel>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::SpatialModel const& v, FormatContext& ctx)
    {
        if (auto const* cv = std::get_if<Fermi::ConstantValueSpatialModel>(&v))
        {
            return fmt::format_to(ctx.out(), "{}", *cv);
        }
        if (auto const* mv = std::get_if<Fermi::MapCubeFunctionSpatialModel>(&v))
        {
            return fmt::format_to(ctx.out(), "{}", *mv);
        }
        if (auto const* sv = std::get_if<Fermi::SkyDirFunctionSpatialModel>(&v))
        {
            return fmt::format_to(ctx.out(), "{}", *sv);
        }
        if (auto const* uv = std::get_if<Fermi::UnknownSpatialModel>(&v))
        {
            return fmt::format_to(ctx.out(), "{}", *uv);
        }
        return fmt::format_to(ctx.out(), "[]", "");
    }
};

////////////////////////////////////////////////////////////////////////////////////////
/// Fermi::Source section.
////////////////////////////////////////////////////////////////////////////////////////

template <>
struct fmt::formatter<Fermi::PointSource>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::PointSource const& v, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "<PointSource> name: {}  roi_center_distance: {}\n"
                              "\t spectrum: {}\n"
                              "\t spatial_model: {}",
                              v.name,
                              v.roi_center_dist,
                              v.spectrum,
                              v.spatial_model);
    }
};

template <>
struct fmt::formatter<Fermi::DiffuseSource>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::DiffuseSource const& v, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "<DiffuseSource> name: {}\n"
                              "\t spectrum: {}\n"
                              "\t spatial_model: {}",
                              v.name,
                              v.spectrum,
                              v.spatial_model);
    }
};

template <>
struct fmt::formatter<Fermi::CompositeSource>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::CompositeSource const& v, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "<DiffuseSource> name: {}\n"
                              "\t spectrum: {}\n"
                              "\t spatial_model: {}",
                              v.name,
                              v.spectrum,
                              v.spatial_model);
    }
};


template <>
struct fmt::formatter<Fermi::Source>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(Fermi::Source const& v, FormatContext& ctx)
    {
        if (auto const* ps = std::get_if<Fermi::PointSource>(&v))
        {
            return fmt::format_to(ctx.out(), "{}", *ps);
        }
        if (auto const* ds = std::get_if<Fermi::DiffuseSource>(&v))
        {
            return fmt::format_to(ctx.out(), "{}", *ds);
        }
        if (auto const* cs = std::get_if<Fermi::CompositeSource>(&v))
        {
            return fmt::format_to(ctx.out(), "{}", *cs);
        }
        return fmt::format_to(ctx.out(), "[]", "");
    }
};

////////////////////////////////////////////////////////////////////////////////
/// Direction Pair
////////////////////////////////////////////////////////////////////////////////
///
template <>
struct fmt::formatter<std::pair<double, double>>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(std::pair<double, double> const& par, FormatContext& ctx)
    {
        return fmt::format_to(
            ctx.out(), "({}, {})", std::get<0>(par), std::get<1>(par));
    }
};



////////////////////////////////////////////////////////////////////////////////
/// Index Pair
////////////////////////////////////////////////////////////////////////////////
///
template <>
struct fmt::formatter<std::pair<long, long>>
{
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(std::pair<long, long> const& par, FormatContext& ctx)
    {
        return fmt::format_to(
            ctx.out(), "({}, {})", std::get<0>(par), std::get<1>(par));
    }
};
