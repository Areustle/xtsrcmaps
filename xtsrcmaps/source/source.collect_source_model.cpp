#include "xtsrcmaps/source/source.hxx"

auto
Fermi::Source::collect_source_model(Config::XtCfg const& cfg,
                                    Obs::XtObs const&    obs) -> XtSrc {
    auto const srcs = Fermi::Source::parse_src_xml(cfg.srcmdl);
    return {
        .point = {
            .srcs          = srcs.point,
            .sph_locs      = spherical_coords(srcs.point),
            .names         = source_names(srcs.point),
        },
        .diffuse = {
            .srcs          = srcs.diffuse,
            .sph_locs      = spherical_coords(srcs.diffuse, {obs.skygeom.crval()[0], obs.skygeom.crval()[1]}), // maybe need to adjust for galactic projections. not sure.
            .names         = source_names(srcs.diffuse),
        },
    };
}
