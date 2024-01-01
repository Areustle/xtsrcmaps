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
auto parse_source_node(RXNPtr src) -> Fermi::Source;
auto parse_spatial_model_node(RXNPtr spamo) -> Fermi::SpatialModel;
auto parse_spectrum_node(RXNPtr spec) -> Fermi::Spectrum;
auto src_par_vec(RXNPtr parent) -> std::vector<Fermi::SourceParameter>;

// Return a Fermi::Source given a RapidXML node .
auto
Fermi::parse_src_xml(std::string const& src_file_name)
    -> std::vector<Fermi::Source> {

    auto result          = std::vector<Fermi::Source>();
    auto doc             = rapidxml::xml_document<>();
    auto src_file_stream = std::ifstream(src_file_name);
    if (!src_file_stream.is_open()){
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
        result.push_back(parse_source_node(src_node));
    }

    // Sort sources by name. Default sorting occurs on variant index first, then
    // name.
    std::sort(std::begin(result),
              std::end(result),
              [](auto const& lhs, auto const& rhs) {
                  auto name = [](auto const& x) {
                      return std::visit([](auto const& e) { return e.name; },
                                        x);
                  };
                  return name(lhs) < name(rhs);
              });

    return result;
}

// Return a Fermi::Source given a RapidXML node .
auto
parse_source_node(RXNPtr src) -> Fermi::Source {
    auto parse_pt_src = [](RXNPtr src) -> Fermi::Source {
        return Fermi::PointSource {
            src->first_attribute("name")->value(),
            parse_spectrum_node(src->first_node("spectrum")),
            parse_spatial_model_node(RXNPtr(src->first_node("spatialModel"))),
            std::stof(src->first_attribute("ROI_Center_Distance")->value())
        };
    };

    auto parse_diff_src = [](RXNPtr src) -> Fermi::Source {
        return Fermi::DiffuseSource {
            src->first_attribute("name")->value(),
            parse_spectrum_node(src->first_node("spectrum")),
            parse_spatial_model_node(src->first_node("spatialModel")),
        };
    };

    auto parse_comp_src = [](RXNPtr src) -> Fermi::Source {
        return Fermi::CompositeSource { src->first_attribute("name")->value() };
    };

    auto parse_unknown_src = [](RXNPtr src) -> Fermi::Source {
        return Fermi::UnknownSource { src->first_attribute("name")->value() };
    };

    auto src_parser
        = [=](std::string const& s) -> std::function<Fermi::Source(RXNPtr)> {
        if (s == "PointSource") { return parse_pt_src; }
        if (s == "DiffuseSource") { return parse_diff_src; }
        if (s == "CompositeSource") { return parse_comp_src; }
        return parse_unknown_src;
    }(src->first_attribute("type")->value());

    return src_parser(src);
}

// Return a vector of Fermi::SourceParameters by iterating through the
// 'parameter' child nodes of the given RapidXML node .
auto
src_par_vec(RXNPtr parent) -> std::vector<Fermi::SourceParameter> {
    auto result = std::vector<Fermi::SourceParameter>();
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
parse_spatial_model_node(RXNPtr spamo) -> Fermi::SpatialModel {
    auto parse_const_spamo = [](RXNPtr spamo) -> Fermi::SpatialModel {
        return Fermi::ConstantValueSpatialModel { src_par_vec(spamo) };
    };
    auto parse_map_cube_spamo = [](RXNPtr spamo) -> Fermi::SpatialModel {
        return Fermi::MapCubeFunctionSpatialModel {
            spamo->first_attribute("file")->value(),
            src_par_vec(spamo),
        };
    };
    auto parse_sky_dir_spamo = [](RXNPtr spamo) -> Fermi::SpatialModel {
        return Fermi::SkyDirFunctionSpatialModel { src_par_vec(spamo) };
    };
    auto parse_unkknown_spamo = [](RXNPtr spamo) -> Fermi::SpatialModel {
        (void)spamo;
        return Fermi::UnknownSpatialModel {};
    };

    auto spatial_model_parser = [=](std::string const& s)
        -> std::function<Fermi::SpatialModel(RXNPtr)> {
        if (s == "ConstantValue") { return parse_const_spamo; }
        if (s == "MapCubeFunction") { return parse_map_cube_spamo; }
        if (s == "SkyDirFunction") { return parse_sky_dir_spamo; }
        return parse_unkknown_spamo;
    }(spamo->first_attribute("type")->value());

    return spatial_model_parser(spamo);
}

// Return a Fermi::Spectrum given a RapidXML node .
auto
parse_spectrum_node(RXNPtr spec) -> Fermi::Spectrum {

    if (spec == nullptr) return {};

    auto type = std::string(spec->first_attribute("type")->value());
    if (type == "FileFunction") {
        return Fermi::FileSpectrum {
            spec->first_attribute("file")->value(),
            src_par_vec(spec),
            std::string(spec->first_attribute("apply_edisp")->value())
                == "true",
        };
    }
    return Fermi::NormalSpectrum {
        src_par_vec(spec),
        [](std::string const& s) -> Fermi::SpectrumType {
            if (s == "BPLExpCutoff") return Fermi::SpectrumType::BPLExpCutoff;
            if (s == "BrokenPowerLaw")
                return Fermi::SpectrumType::BrokenPowerLaw;
            if (s == "BrokenPowerLaw2")
                return Fermi::SpectrumType::BrokenPowerLaw2;
            if (s == "ConstantValue") return Fermi::SpectrumType::ConstantValue;
            if (s == "ExpCutoff") return Fermi::SpectrumType::ExpCutoff;
            if (s == "FileFunction") return Fermi::SpectrumType::FileFunction;
            if (s == "Gaussian") return Fermi::SpectrumType::Gaussian;
            if (s == "LogParabola") return Fermi::SpectrumType::LogParabola;
            if (s == "PowerLaw") return Fermi::SpectrumType::PowerLaw;
            if (s == "PowerLaw2") return Fermi::SpectrumType::PowerLaw2;
            if (s == "PLSuperExpCutoff2")
                return Fermi::SpectrumType::PLSuperExpCutoff2;
            if (s == "PLSuperExpCutoff3")
                return Fermi::SpectrumType::PLSuperExpCutoff3;
            if (s == "PLSuperExpCutoff4")
                return Fermi::SpectrumType::PLSuperExpCutoff4;
            return Fermi::SpectrumType::Unknown;
        }(type),
    };
}
