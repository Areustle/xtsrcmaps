#include "xtsrcmaps/fits/fits.hxx"

#include "xtsrcmaps/tensor/reorder_tensor.hpp"

#include "fitsio.h"
#include "fmt/color.h"

#include <functional>

auto
Fermi::fits::write_src_model(std::string const&              filename,
                             Tensor<float, 4> &         model_map,
                             std::vector<std::string> const& src_names)
    -> void {

    fmt::print(fg(fmt::color::light_pink),
               "Writing Source Maps to file: " + filename + "\n");

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

    long const Ns                  = model_map.extent(0);
    long const Nh                  = model_map.extent(1);
    long const Nw                  = model_map.extent(2);
    long const Ne                  = model_map.extent(3);

    std::vector<signed long> coord = { 1l, 1l, 1l };
    std::vector<long>        naxes = { Nw, Nh, Ne };
    auto                     image_size
        = std::accumulate(naxes.begin(), naxes.end(), 1l, std::multiplies {});

    //           SHWE -> SEHW;
    auto img = reorder_tensor(model_map, { 0, 3, 2, 1 });

    // Loop over sources
    for (long s = 0; s < Ns; ++s) {

        // Create the image space in the fits output file.
        fits_create_img(fp, FLOAT_IMG, 3, naxes.data(), &status);

        char key_name[16] = { "EXTNAME" };
        fits_update_key(fp,
                        TSTRING,
                        key_name,
                        const_cast<char*>(src_names[s].c_str()),
                        0,
                        &status);

        // // Transpose the image to H,W,E
        // Tensor<float, 3> img
        //     = model_map.slice(Idx4 { 0, 0, 0, s }, Idx4 { Ne, Nh, Nw, 1 })
        //           .reshape(Idx3 { Ne, Nh, Nw })
        //           .shuffle(Idx3 { 1, 2, 0 });


        // Write the image
        fits_write_pix(
            fp, TFLOAT, &*coord.begin(), image_size, &img[s, 0, 0, 0], &status);

        fits_write_chksum(fp, &status);
    }
    fits_close_file(fp, &status);
}
