#include "xtsrcmaps/fitsfuncs.hxx"

#include "fmt/format.h"

#include "fitsio.h"

using std::optional;
using std::string;
using std::vector;
// using std::experimental::extents;
// using std::experimental::full_extent;
// using std::experimental::mdspan;
// using std::experimental::submdspan;

auto
Fermi::fits::ccube_energies(std::string const& filename) noexcept
    -> std::optional<std::vector<double>>
{
    // Use CFITSIO to open the ccube and read the energies in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    if (status != 0)
    {
        fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
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
    fits_read_col(
        ifile, TDOUBLE, 2, rows, 1, 1, nullptr, &energies[rows], &anynul, &status);

    fits_close_file(ifile, &status);

    for (auto&& e : energies) { e /= 1000.0; }

    return energies;
}

auto
Fermi::fits::read_psf(std::string const& filename) -> std::optional<PsfParamData>
{

    // Use CFITSIO to open the ccube and read the energies in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    // open the file, or return a null optional if it cannot open.
    if (status)
    {
        fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
                   filename,
                   status);
        return std::nullopt;
    }

    // Read the RPSF parameters
    char rpsf[] { "RPSF_FRONT" };
    fits_movnam_hdu(ifile, BINARY_TBL, rpsf, 0, &status);
    if (status) fmt::print("Failed to access RPSF header. Status {}\n", status);
    if (status) return std::nullopt;

    // Populate the local RPSF Param vectors.
    auto energ_lo = vector<float>(23, 0.0f);
    auto energ_hi = vector<float>(23, 0.0f);
    auto cthet_lo = vector<float>(8, 0.0f);
    auto cthet_hi = vector<float>(8, 0.0f);
    auto ncore    = vector<float>(184, 0.0f);
    auto ntail    = vector<float>(184, 0.0f);
    auto score    = vector<float>(184, 0.0f);
    auto stail    = vector<float>(184, 0.0f);
    auto gcore    = vector<float>(184, 0.0f);
    auto gtail    = vector<float>(184, 0.0f);

    fits_read_col(ifile, TFLOAT, 1, 1, 1, 23, nullptr, &energ_lo[0], nullptr, &status);
    if (status) fmt::print("Failed to access T1. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 2, 1, 1, 23, nullptr, &energ_hi[0], nullptr, &status);
    if (status) fmt::print("Failed to access T2. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 3, 1, 1, 8, nullptr, &cthet_lo[0], nullptr, &status);
    if (status) fmt::print("Failed to access T3. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 4, 1, 1, 8, nullptr, &cthet_hi[0], nullptr, &status);
    if (status) fmt::print("Failed to access T4. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 5, 1, 1, 184, nullptr, &ncore[0], nullptr, &status);
    if (status) fmt::print("Failed to access T5. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 6, 1, 1, 184, nullptr, &ntail[0], nullptr, &status);
    if (status) fmt::print("Failed to access T6. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 7, 1, 1, 184, nullptr, &score[0], nullptr, &status);
    if (status) fmt::print("Failed to access T7. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 8, 1, 1, 184, nullptr, &stail[0], nullptr, &status);
    if (status) fmt::print("Failed to access T8. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 9, 1, 1, 184, nullptr, &gcore[0], nullptr, &status);
    if (status) fmt::print("Failed to access T9. Status {}\n", status);
    fits_read_col(ifile, TFLOAT, 10, 1, 1, 184, nullptr, &gtail[0], nullptr, &status);
    if (status) fmt::print("Failed to access T10. Status {}\n", status);
    if (status) return std::nullopt;

    // Read the PSF scaling factors.
    char psf_scale[] { "PSF_SCALING_PARAMS_FRONT" };
    fits_movnam_hdu(ifile, BINARY_TBL, psf_scale, 0, &status);
    if (status) fmt::print("Failed to access PSF_SCALING header. Status {}\n", status);
    if (status) return std::nullopt;

    // Populate the psf scaling factors.
    auto scale = vector<float>(3, 0.0f);
    fits_read_col(ifile, TFLOAT, 1, 1, 1, 3, nullptr, &scale[0], nullptr, &status);
    if (status) fmt::print("Failed to access P1. Status {}\n", status);
    if (status) return std::nullopt;

    fits_close_file(ifile, &status);
    if (status) fmt::print("Failed to Close. Status {}\n", status);
    if (status) return std::nullopt;

    return {
        {energ_lo,
         energ_hi, cthet_lo,
         cthet_hi, ncore,
         ntail, score,
         stail, gcore,
         gtail, scale[0],
         scale[1],
         scale[2]}
    };
}
