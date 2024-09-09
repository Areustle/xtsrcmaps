#include "xtsrcmaps/fits/fits.hxx"

#include "fitsio.h"
#include "fmt/core.h"

auto
Fermi::fits::read_energies(std::string const& filename) noexcept
    -> std::optional<std::vector<double>> {
    // Use CFITSIO to open the ccube and read the energies in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    if (status) {
        fmt::print(
            "Failed to open {}. Status {}. Returning an empty optional.\n",
            filename,
            status);
        return std::nullopt;
    }

    fits_movabs_hdu(ifile, 2, nullptr, &status);
    unsigned long rows = 0;
    fits_read_key(ifile, TULONG, "NAXIS2", &rows, nullptr, &status);

    auto energies = std::vector<double>(rows + 1, 0.0);

    int anynul    = 0;
    fits_read_col(
        ifile, TDOUBLE, 2, 1, 1, rows, nullptr, &energies[0], &anynul, &status);
    fits_read_col(ifile,
                  TDOUBLE,
                  3,
                  rows,
                  1,
                  1,
                  nullptr,
                  &energies[rows],
                  &anynul,
                  &status);

    fits_close_file(ifile, &status);

    for (auto&& e : energies) { e /= 1000.0; }

    return energies;
}
