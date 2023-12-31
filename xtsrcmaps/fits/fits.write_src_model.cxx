#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/source/source.hxx"

#include "fitsio.h"

#include <functional>

auto
Fermi::fits::write_src_model(std::string const&                filename,
                             Tensor4f const&                   model_map,
                             std::vector<Fermi::Source> const& srcs) -> void {

    // create the file
    int       status = 0;
    fitsfile* fp;
    fits_create_file(&fp, filename.c_str(), &status);

    // input a dummy primary image
    fits_create_img(fp, FLOAT_IMG, 0, nullptr, &status);
    char key_name[16] = { "HDUNAME" };
    char primary[16]  = { "PRIMARY" };
    fits_update_key(fp, TSTRING, key_name, primary, 0, &status);

    // Prepare to write the sources
    auto const names = Fermi::names_from_point_sources(srcs);

    long const Ne    = model_map.dimension(0);
    long const Nh    = model_map.dimension(1);
    long const Nw    = model_map.dimension(2);
    long const Ns    = model_map.dimension(3);

    // Loop over sources
    for (long s = 0; s < Ns; ++s) {

        // Create the image space in the fits output file.
        std::vector<long> naxes      = { Nh, Nw, Ne };
        auto              image_size = std::accumulate(
            naxes.begin(), naxes.end(), 1, std::multiplies {});
        fits_create_img(fp, FLOAT_IMG, 3, naxes.data(), &status);

        char key_name[16] = { "EXTNAME" };
        fits_update_key(fp,
                        TSTRING,
                        key_name,
                        const_cast<char*>(names[s].c_str()),
                        0,
                        &status);

        // Transpose the image to H,W,E
        Tensor3f img
            = model_map.slice(Idx4 { 0, 0, 0, s }, Idx4 { Ne, Nh, Nw, 1 })
                  .reshape(Idx3 { Ne, Nh, Nw })
                  .shuffle(Idx3 { 1, 2, 0 });

        std::vector<signed long> coord = { 1, 1, 1 };

        // Write the image
        fits_write_pix(
            fp, TFLOAT, &*coord.begin(), image_size, img.data(), &status);

        fits_write_chksum(fp, &status);
    }
    fits_close_file(fp, &status);
}
