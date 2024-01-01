#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/misc/misc.hxx"

#include "cxxopts.hpp"

#include <iostream>
#include <string>

using std::string;

namespace Fermi {

auto
parse_cli_to_cfg(int const argc, char** argv) -> Fermi::XtCfg {

    cxxopts::Options options("YourProgram", "Description");
    options.add_options()("help", "Print help")(
        "parfile,p", "Input parfile", cxxopts::value<string>())(
        "sctable", "", cxxopts::value<string>())(
        "expcube", "", cxxopts::value<string>())(
        "cmap", "", cxxopts::value<string>())(
        "srcmdl", "", cxxopts::value<string>())(
        "bexpmap", "", cxxopts::value<string>())(
        "wmap", "", cxxopts::value<string>())(
        "outfile", "", cxxopts::value<string>())(
        "psf_file", "", cxxopts::value<string>())(
        "aeff_file", "", cxxopts::value<string>());

    auto vm = options.parse(argc, argv);

    if (vm.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    auto cfg = good(vm.count("parfile")
                        ? Fermi::parse_parfile(vm["parfile"].as<string>())
                        : Fermi::parse_parfile("gtsrcmaps.par"),
                    "Cannot Read Parfile!");

    // Override values with user-provided arguments
    if (vm.count("sctable")) cfg.sctable = vm["sctable"].as<string>();
    if (vm.count("expcube")) cfg.expcube = vm["expcube"].as<string>();
    if (vm.count("cmap")) cfg.cmap = vm["cmap"].as<string>();
    if (vm.count("srcmdl")) cfg.srcmdl = vm["srcmdl"].as<string>();
    if (vm.count("bexpmap")) cfg.bexpmap = vm["bexpmap"].as<string>();
    if (vm.count("wmap")) cfg.wmap = vm["wmap"].as<string>();
    if (vm.count("outfile")) cfg.outfile = vm["outfile"].as<string>();
    if (vm.count("psf_file")) cfg.psf_file = vm["psf_file"].as<string>();
    if (vm.count("aeff_file")) cfg.aeff_file = vm["aeff_file"].as<string>();

    cfg = Fermi::validate_cfg(cfg);

    return cfg;
}
} // namespace Fermi
