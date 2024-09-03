#include "xtsrcmaps/tensor/read_file_tensor.hpp"

#include <iostream>

int
main(int const argc, char** argv) {

    /* filecomp(front_obs_psf, "obs_psf_front_CDE.bin");  */
    auto const sp_b = Fermi::read_file_tensor<double, 3>(
        "./tests/expected/obs_psf_front_CDE.bin", { 40uz, 401uz, 38uz });
    std::cout << sp_b[0, 0, 0] << std::endl;
}
