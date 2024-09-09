#include "xtsrcmaps/fits/fits.hxx"

auto
Fermi::fits::safe_open(std::string const& filename)
    -> std::unique_ptr<fitsfile, std::function<void(fitsfile*)>> {
    fitsfile* fptr   = nullptr;
    int       status = 0;

    fits_open_file(&fptr, filename.c_str(), READONLY, &status);
    if (status) {
        fits_report_error(stderr, status);
        throw std::runtime_error("FITSIO error occurred");
    }

    // Custom deleter lambda that closes the FITS file
    auto deleter = [](fitsfile* fptr) {
        int status = 0;
        if (fptr) {
            fits_close_file(fptr, &status);
            if (status) { fits_report_error(stderr, status); }
        }
    };

    return std::unique_ptr<fitsfile, decltype(deleter)>(fptr, deleter);
}
