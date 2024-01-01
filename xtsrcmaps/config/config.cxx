#include "xtsrcmaps/config/config.hxx"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace fs = std::filesystem;

// Function to parse the CSV parfile and return XtCfg struct
std::optional<Fermi::XtCfg>
Fermi::parse_parfile(const std::string& filename) {

    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening parfile: " << filename << std::endl;
        return {};
    }

    std::unordered_map<std::string, std::string> tempMap;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') { continue; }

        std::istringstream iss(line);
        std::string        key, value;
        std::getline(iss, key, ',');
        std::getline(iss, value, ',');
        std::getline(iss, value, ',');
        std::getline(iss, value, ',');

        // Trim leading and trailing whitespaces from the value
        value.erase(0, value.find_first_not_of(" \t\n\r\f\v\"\'"));
        value.erase(value.find_last_not_of(" \t\n\r\f\v\"\'") + 1);

        tempMap[key] = value;
    }

    return {
        {
         .sctable   = tempMap["sctable"],
         .expcube   = tempMap["expcube"],
         .cmap      = tempMap["cmap"],
         .srcmdl    = tempMap["srcmdl"],
         .bexpmap   = tempMap["bexpmap"],
         .wmap      = tempMap["wmap"],
         .outfile   = tempMap["outfile"],
         .psf_file  = tempMap["psf_file"],
         .aeff_file = tempMap["aeff_file"],
         }
    };
}


auto
absolute_path(std::string const& path) -> std::string {
    fs::path inputPath(path);

    if (!inputPath.is_absolute()) {
        fs::path currentDir = fs::current_path();
        inputPath           = currentDir / inputPath;
    }

    return inputPath.lexically_normal().string();
}

auto
valid_file(std::string const& path) -> std::string {
    auto abspath = absolute_path(path);
    if (fs::exists(abspath)) {
        return abspath;
    } else {
        if (abspath == path) {
            std::cerr << "File not found! " << abspath << std::endl;
            exit(1);
        } else {
            std::cerr << "File not found locally: " << path
                      << "\nNor globally: " << abspath << std::endl;
            exit(1);
        }
    }
}

auto
Fermi::validate_cfg(Fermi::XtCfg const& cfg) -> Fermi::XtCfg {
    return {
        .sctable   = cfg.sctable,
        .expcube   = valid_file(cfg.expcube),
        .cmap      = valid_file(cfg.cmap),
        .srcmdl    = valid_file(cfg.srcmdl),
        .bexpmap   = valid_file(cfg.bexpmap),
        .wmap      = cfg.wmap,
        .outfile   = cfg.outfile,
        .psf_file  = valid_file(cfg.psf_file),
        .aeff_file = valid_file(cfg.aeff_file),
    };

    return {};
}
