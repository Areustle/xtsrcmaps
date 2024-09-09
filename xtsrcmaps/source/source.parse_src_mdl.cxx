// An implementation converting an XML file into a vector of Fermi::Source
// objects.
#include "xtsrcmaps/source/source.hxx"

#include "xtsrcmaps/rapidxml/rapidxml.hxx"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>


using RXNPtr = rapidxml::xml_node<>*;

// Forward Declarations
auto parse_source_node(RXNPtr src) -> Fermi::Source::Source;
auto parse_spatial_model_node(RXNPtr spamo) -> Fermi::Source::SpatialModel;
auto parse_spectrum_node(RXNPtr spec) -> Fermi::Source::Spectrum;
auto src_par_vec(RXNPtr parent) -> std::vector<Fermi::Source::SourceParameter>;


template <Fermi::Source::SourceConcept T>
void
sort_source_vector(std::vector<T>& srcs) {
    std::sort(
        std::begin(srcs), std::end(srcs), [](auto const& lhs, auto const& rhs) {
            return lhs.name < rhs.name;
        });
}

// Return a Fermi::Source given a RapidXML node .
auto
Fermi::Source::parse_src_xml(std::string const& src_file_name) -> SourceGroup {
    auto result          = Fermi::Source::SourceGroup {};
    auto sorted_srcs     = std::vector<Fermi::Source::Source>();
    auto doc             = rapidxml::xml_document<>();
    auto src_file_stream = std::ifstream(src_file_name);
    if (!src_file_stream.is_open()) {
        std::cerr << "Cannot open " << src_file_name << std::endl;
        exit(1);
    }
    auto src_file_buf
        = std::vector(std::istreambuf_iterator { src_file_stream },
                      std::istreambuf_iterator<char>());
    src_file_buf.push_back('\0');
    doc.parse<0>(&src_file_buf[0]);

    // The source_library root node must be a sequence of sources, which we
    // shall parse
    auto root     = RXNPtr(doc.first_node("source_library"));
    auto src_node = RXNPtr(root->first_node("source"));
    for (; src_node; src_node = src_node->next_sibling()) {
        Fermi::Source::Source src = parse_source_node(src_node);
        if (std::holds_alternative<PointSource>(src)) {
            result.point.push_back(std::get<PointSource>(src));
        } else if (std::holds_alternative<DiffuseSource>(src)) {
            result.diffuse.push_back(std::get<DiffuseSource>(src));
        }
        /*  else if (std::holds_alternative<CompositeSource>(src)) { */
        /*     result.composite.push_back(std::get<CompositeSource>(src)); */
        /* } else if (std::holds_alternative<UnknownSource>(src)) { */
        /*     result.unknown.push_back(std::get<UnknownSource>(src)); */
        /* } */
    }

    // Sort sources by name. Default sorting occurs on variant index first, then
    // name.
    sort_source_vector(result.point);
    sort_source_vector(result.diffuse);

    return result;
}

// Return a Fermi::Source given a RapidXML node .
auto
parse_source_node(RXNPtr src) -> Fermi::Source::Source {

    auto parse_string_attribute
        = [](RXNPtr src, const char* attr_name) -> std::string {
        if (auto attr = src->first_attribute(attr_name)) {
            return attr->value();
        }
        throw std::runtime_error("Missing required string attribute: "
                                 + std::string(attr_name));
    };

    auto parse_float_attribute
        = [](RXNPtr src, const char* attr_name) -> float {
        if (auto attr = src->first_attribute(attr_name)) {
            try {
                return std::stof(attr->value());
            } catch (const std::invalid_argument&) {
                throw std::runtime_error("Invalid float value for attribute: "
                                         + std::string(attr_name));
            }
        }
        throw std::runtime_error("Missing required float attribute: "
                                 + std::string(attr_name));
    };

    auto parse_pt_src = [&](RXNPtr src) -> Fermi::Source::Source {
        return Fermi::Source::PointSource {
            parse_string_attribute(src, "name"),
            parse_spectrum_node(src->first_node("spectrum")),
            parse_spatial_model_node(src->first_node("spatialModel")),
            parse_float_attribute(src, "ROI_Center_Distance")
        };
    };

    auto parse_diff_src = [&](RXNPtr src) -> Fermi::Source::Source {
        return Fermi::Source::DiffuseSource {
            parse_string_attribute(src, "name"),
            parse_spectrum_node(src->first_node("spectrum")),
            parse_spatial_model_node(src->first_node("spatialModel"))
        };
    };

    auto parse_comp_src = [&](RXNPtr src) -> Fermi::Source::Source {
        return Fermi::Source::CompositeSource { parse_string_attribute(
            src, "name") };
    };

    auto parse_unknown_src = [&](RXNPtr src) -> Fermi::Source::Source {
        return Fermi::Source::UnknownSource { parse_string_attribute(src,
                                                                     "name") };
    };

    auto src_parser = [=](std::string const& s)
        -> std::function<Fermi::Source::Source(RXNPtr)> {
        if (s == "PointSource") { return parse_pt_src; }
        if (s == "DiffuseSource") { return parse_diff_src; }
        if (s == "CompositeSource") { return parse_comp_src; }
        return parse_unknown_src;
    }(parse_string_attribute(src, "type"));

    return src_parser(src);
}

// Return a vector of Fermi::SourceParameters by iterating through the
// 'parameter' child nodes of the given RapidXML node .
auto
src_par_vec(RXNPtr parent) -> std::vector<Fermi::Source::SourceParameter> {
    auto result = std::vector<Fermi::Source::SourceParameter>();
    auto par    = RXNPtr(parent->first_node("parameter"));

    for (; par; par = par->next_sibling()) {
        // why does emplace_back fail?
        result.push_back({
            std::string(par->first_attribute("name")->value()),
            std::stoul(par->first_attribute("free")->value()),
            std::stof(par->first_attribute("value")->value()),
            std::stof(par->first_attribute("scale")->value()),
            std::stof(par->first_attribute("min")->value()),
            std::stof(par->first_attribute("max")->value()),
        });
    }
    return result;
}

// Return a Fermi::SpatialModel given a RapidXML node .
auto
parse_spatial_model_node(RXNPtr spamo) -> Fermi::Source::SpatialModel {
    auto parse_const_spamo = [](RXNPtr spamo) -> Fermi::Source::SpatialModel {
        return Fermi::Source::ConstantValueSpatialModel { src_par_vec(spamo) };
    };
    auto parse_map_cube_spamo
        = [](RXNPtr spamo) -> Fermi::Source::SpatialModel {
        return Fermi::Source::MapCubeFunctionSpatialModel {
            spamo->first_attribute("file")->value(),
            src_par_vec(spamo),
        };
    };
    auto parse_sky_dir_spamo = [](RXNPtr spamo) -> Fermi::Source::SpatialModel {
        return Fermi::Source::SkyDirFunctionSpatialModel { src_par_vec(spamo) };
    };
    auto parse_unkknown_spamo
        = [](RXNPtr spamo) -> Fermi::Source::SpatialModel {
        (void)spamo;
        return Fermi::Source::UnknownSpatialModel {};
    };

    auto spatial_model_parser = [=](std::string const& s)
        -> std::function<Fermi::Source::SpatialModel(RXNPtr)> {
        if (s == "ConstantValue") { return parse_const_spamo; }
        if (s == "MapCubeFunction") { return parse_map_cube_spamo; }
        if (s == "SkyDirFunction") { return parse_sky_dir_spamo; }
        return parse_unkknown_spamo;
    }(spamo->first_attribute("type")->value());

    return spatial_model_parser(spamo);
}

// Return a Fermi::Spectrum given a RapidXML node .
auto
parse_spectrum_node(RXNPtr spec) -> Fermi::Source::Spectrum {

    if (spec == nullptr) return {};

    auto type = std::string(spec->first_attribute("type")->value());
    if (type == "FileFunction") {
        return Fermi::Source::FileSpectrum {
            spec->first_attribute("file")->value(),
            src_par_vec(spec),
            std::string(spec->first_attribute("apply_edisp")->value())
                == "true",
        };
    }
    return Fermi::Source::NormalSpectrum {
        src_par_vec(spec),
        [](std::string const& s) -> Fermi::Source::SpectrumType {
            if (s == "BPLExpCutoff")
                return Fermi::Source::SpectrumType::BPLExpCutoff;
            if (s == "BrokenPowerLaw")
                return Fermi::Source::SpectrumType::BrokenPowerLaw;
            if (s == "BrokenPowerLaw2")
                return Fermi::Source::SpectrumType::BrokenPowerLaw2;
            if (s == "ConstantValue")
                return Fermi::Source::SpectrumType::ConstantValue;
            if (s == "ExpCutoff") return Fermi::Source::SpectrumType::ExpCutoff;
            if (s == "FileFunction")
                return Fermi::Source::SpectrumType::FileFunction;
            if (s == "Gaussian") return Fermi::Source::SpectrumType::Gaussian;
            if (s == "LogParabola")
                return Fermi::Source::SpectrumType::LogParabola;
            if (s == "PowerLaw") return Fermi::Source::SpectrumType::PowerLaw;
            if (s == "PowerLaw2") return Fermi::Source::SpectrumType::PowerLaw2;
            if (s == "PLSuperExpCutoff2")
                return Fermi::Source::SpectrumType::PLSuperExpCutoff2;
            if (s == "PLSuperExpCutoff3")
                return Fermi::Source::SpectrumType::PLSuperExpCutoff3;
            if (s == "PLSuperExpCutoff4")
                return Fermi::Source::SpectrumType::PLSuperExpCutoff4;
            return Fermi::Source::SpectrumType::Unknown;
        }(type),
    };
}
