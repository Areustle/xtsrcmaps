#include "xtsrcmaps/fits/fits.hxx"

#include "fitsio.h"
#include "fmt/core.h"

using std::string;

/// Read the header keyword values from the counts cube.
auto
Fermi::fits::ccube_pixels(std::string const& filename) noexcept
    -> std::optional<Obs::CCubePixels> {
    // Use CFITSIO to open the ccube and read the energies in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    if (status != 0) {
        fmt::print(
            "Failed to open {}. Status {}. Returning an empty optional.\n",
            filename,
            status);
        return std::nullopt;
    }

    fits_movabs_hdu(ifile, 1, nullptr, &status);
    if (status) {
        fmt::print("Failed to access header 1 in FITS file {}. Status {}.\n",
                   filename,
                   status);
        return std::nullopt;
    }
    //
    unsigned long naxis = 0;
    fits_read_key(ifile, TULONG, "NAXIS", &naxis, nullptr, &status);
    if (naxis != 3) {
        fmt::print("File Format Error in file {}. Expected 3 NAXES, got {}.\n",
                   filename,
                   naxis);
        return std::nullopt;
    }

    long cols = 0;
    fits_read_key(ifile, TLONG, "NAXIS1", &cols, nullptr, &status);
    if (status) fmt::print("Failed to read NAXIS1. Status {}.\n", status);
    if (status) return std::nullopt;
    long rows = 0;
    fits_read_key(ifile, TLONG, "NAXIS2", &rows, nullptr, &status);
    if (status) fmt::print("Failed to read NAXIS2. Status {}.\n", status);
    if (status) return std::nullopt;
    long channels = 0;
    fits_read_key(ifile, TLONG, "NAXIS3", &channels, nullptr, &status);
    if (status) fmt::print("Failed to read NAXIS3. Status {}.\n", status);
    if (status) return std::nullopt;

    std::array<double, 3> crpix;
    fits_read_key(ifile, TDOUBLE, "CRPIX1", &crpix[0], nullptr, &status);
    if (status) fmt::print("Failed to read CRPIX1. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TDOUBLE, "CRPIX2", &crpix[1], nullptr, &status);
    if (status) fmt::print("Failed to read CRPIX2. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TDOUBLE, "CRPIX3", &crpix[2], nullptr, &status);
    if (status) fmt::print("Failed to read CRPIX3. Status {}.\n", status);
    if (status) return std::nullopt;

    std::array<double, 3> crval;
    fits_read_key(ifile, TDOUBLE, "CRVAL1", &crval[0], nullptr, &status);
    if (status) fmt::print("Failed to read CRVAL1. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TDOUBLE, "CRVAL2", &crval[1], nullptr, &status);
    if (status) fmt::print("Failed to read CRVAL2. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TDOUBLE, "CRVAL3", &crval[2], nullptr, &status);
    if (status) fmt::print("Failed to read CRVAL3. Status {}.\n", status);
    if (status) return std::nullopt;

    std::array<double, 3> cdelt;
    fits_read_key(ifile, TDOUBLE, "CDELT1", &cdelt[0], nullptr, &status);
    if (status) fmt::print("Failed to read CDELT1. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TDOUBLE, "CDELT2", &cdelt[1], nullptr, &status);
    if (status) fmt::print("Failed to read CDELT2. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TDOUBLE, "CDELT3", &cdelt[2], nullptr, &status);
    if (status) fmt::print("Failed to read CDELT3. Status {}.\n", status);
    if (status) return std::nullopt;

    double axis_rot;
    fits_read_key(ifile, TDOUBLE, "CROTA2", &axis_rot, nullptr, &status);
    if (status) fmt::print("Failed to read CROTA2. Status {}.\n", status);
    if (status) return std::nullopt;

    char loncoord[128];
    fits_read_key(ifile, TSTRING, "CTYPE1", &loncoord, nullptr, &status);
    if (status) fmt::print("Failed to read CTYPE1. Status {}.\n", status);
    if (status) return std::nullopt;

    fits_close_file(ifile, &status);


    string                   input(loncoord);
    std::vector<std::string> tokens;
    std::string::size_type   j;
    while ((j = input.find_first_of("-")) != std::string::npos) {
        if (j != 0) tokens.push_back(input.substr(0, j));
        input = input.substr(j + 1);
    }
    tokens.push_back(input);
    if (tokens.back() == "") tokens.pop_back();
    bool   is_galactic = tokens.front() == "GLON";
    string proj_name   = tokens.back();

    return {
        {{ cols, rows, channels },
         crpix, crval,
         cdelt, axis_rot,
         proj_name, is_galactic}
    };
}
