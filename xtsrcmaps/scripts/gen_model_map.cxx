////
#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/model_map.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/source_utils.hxx"
#include "xtsrcmaps/tensor_ops.hxx"



int
main()
{

    auto       cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const src_sph   = Fermi::spherical_coords_from_point_sources(srcs);

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    auto const ccube     = good(opt_ccube, "Cannot read counts cube map file!");
    auto       skygeom   = Fermi::SkyGeom(ccube);
    long const Ne        = 38;
    long const Nh        = 100;
    long const Nw        = 100;
    long const Ns        = src_sph.size();
    long const Nd        = 401;

    Tensor3d norm_uPsf   = Fermi::row_major_file_to_col_major_tensor(
        "/home/areustle/nasa/fermi/xtsrcmaps/xtsrcmaps/tests/expected/"
          "uPsf_normalized_SED.bin",
        Ns,
        Ne,
        Nd);

    Tensor2d peak_uPsf = Fermi::row_major_file_to_col_major_tensor(
        "/home/areustle/nasa/fermi/xtsrcmaps/xtsrcmaps/tests/expected/"
        "uPsf_peak_SE.bin",
        Ns,
        Ne);

    Tensor4d psfEst = Fermi::ModelMap::pixel_mean_psf_riemann(
        Nh, Nw, src_sph, norm_uPsf, peak_uPsf, { ccube }, 1e-3);

    std::ofstream ofs_pr("xt_psfEstimate.bin",
                         std::ios::out | std::ios::binary | std::ios::ate);
    ofs_pr.write(reinterpret_cast<char*>(psfEst.data()),
                 sizeof(double) * Ne * 10000 * Ns);
    ofs_pr.close();

    // Tensor4d st_psfEst = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/src_psfEstimate.bin", Ns, Nw, Nh, Ne);
}
