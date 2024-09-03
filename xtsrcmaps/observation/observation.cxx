#include "xtsrcmaps/observation/observation.hxx"
#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/source/source.hxx"

#include "fmt/color.h"

auto
Fermi::collect_observation_data(Fermi::XtCfg const& cfg) -> XtObs {

    /* fmt::print(fg(fmt::color::light_pink), "Collecting Observation Data.\n"); */

    auto const energies = good(Fermi::fits::ccube_energies(cfg.cmap),
                               "Cannot read ccube_energies file!");

    /* fmt::print(fg(fmt::color::light_green), */
    /*            "\t{} Energy Bins\n", */
    /*            energies.size()); */

    auto const ccube = good(Fermi::fits::ccube_pixels(cfg.cmap),
                            "Cannot read counts cube map file!");

    /* fmt::print(fg(fmt::color::light_green), */
    /*            "\t{} x {} pixel grid\n", */
    /*            ccube.naxes[0], */
    /*            ccube.naxes[1]); */

    auto const srcs = Fermi::parse_src_xml(cfg.srcmdl);

    /* fmt::print(fg(fmt::color::light_green), "\t{} Sources\n", srcs.size()); */

    //**************************************************************************
    // Read Exposure Cube Fits File.
    //**************************************************************************
    auto const exp_cube
        = good(Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE"),
               "Cannot read exposure cube map file!");
    auto const wexp_cube
        = good(Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE"),
               "Cannot read exposure cube map file!");

    return {
        .energies          = energies,
        .logEs             = Fermi::log10_v(energies),
        .ccube             = ccube,
        .srcs              = srcs,
        .src_sph           = Fermi::spherical_coords_from_point_sources(srcs),
        .src_names         = Fermi::names_from_point_sources(srcs),
        .exp_cube          = exp_cube,
        .weighted_exp_cube = wexp_cube,
        .Nh                = ccube.naxes[0],
        .Nw                = ccube.naxes[1],
    };
};
