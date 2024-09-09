#include "xtsrcmaps/fits/fits.hxx"

#include "fitsio.h"
#include "fmt/core.h"

using std::vector;

auto
Fermi::fits::read_expcube(std::string const& filename,
                          std::string const& tblname)
    -> std::optional<Obs::ExposureCubeData> {
    // Use CFITSIO to open the ltcube and read the values in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    // open the file, or return a null optional if it cannot open.
    if (status) {
        fmt::print(
            "Failed to open {}. Status {}. Returning an empty optional.\n",
            filename,
            status);
        return std::nullopt;
    }

    // Read the Exposure parameters
    char   c_tbl[128] {};
    size_t tbsz = tblname.size() > 127 ? 127 : tblname.size();
    tblname.copy(c_tbl, tbsz);
    c_tbl[tbsz] = '\0';
    fits_movnam_hdu(ifile, BINARY_TBL, c_tbl, 0, &status);
    if (status)
        fmt::print("Failed to access {} header. Status {}\n", c_tbl, status);
    if (status) return std::nullopt;

    // HDU keys
    // fits_read_key (fitsfile *fptr, int datatype, char *keyname, > DTYPE
    // *value,
    //    char *comment, int *status);
    unsigned int nside, nbrbins;
    double       cosmin;
    char         ordering[128], coordsys[128], thetabin[128];

    fits_read_key(ifile, TUINT, "NSIDE", &nside, nullptr, &status);
    if (status) fmt::print("Failed to read NSIDE. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TUINT, "NBRBINS", &nbrbins, nullptr, &status);
    if (status) fmt::print("Failed to read NBRBINS. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TDOUBLE, "COSMIN", &cosmin, nullptr, &status);
    if (status) fmt::print("Failed to read COSMIN. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TSTRING, "ORDERING", &ordering, nullptr, &status);
    if (status) fmt::print("Failed to read ORDERING. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TSTRING, "COORDSYS", &coordsys, nullptr, &status);
    if (status) fmt::print("Failed to read COORDSYS. Status {}.\n", status);
    if (status) return std::nullopt;
    fits_read_key(ifile, TSTRING, "THETABIN", &thetabin, nullptr, &status);
    if (status) fmt::print("Failed to read THETABIN. Status {}.\n", status);
    if (status) return std::nullopt;

    if (std::string { ordering } != "NESTED") {
        fmt::print("LiveTimeCube Healpix Ordering {} is Unsupported. Please "
                   "use NESTED "
                   "order.\n",
                   ordering);
        return std::nullopt;
    }
    if (std::string { coordsys } != "EQU") {
        fmt::print("LiveTimeCube COORDSYS {} is Unsupported. Please use EQU "
                   "coordinates.\n",
                   ordering);
        return std::nullopt;
    }

    // Populate the local Exposure Param vectors.
    size_t const npix    = 12 * nside * nside;
    auto         cosbins = vector<float>(nbrbins * npix, 0.0f);
    auto         ra      = vector<float>(npix, 0.0f);
    auto         dec     = vector<float>(npix, 0.0f);

    fits_read_col(ifile,
                  TFLOAT,
                  1,
                  1,
                  1,
                  nbrbins * npix,
                  nullptr,
                  &cosbins[0],
                  nullptr,
                  &status);
    if (status) fmt::print("Failed to access T1. Status {}\n", status);
    if (status) return std::nullopt;
    fits_read_col(
        ifile, TFLOAT, 2, 1, 1, npix, nullptr, &ra[0], nullptr, &status);
    if (status) fmt::print("Failed to access T2. Status {}\n", status);
    if (status) return std::nullopt;
    fits_read_col(
        ifile, TFLOAT, 3, 1, 1, npix, nullptr, &dec[0], nullptr, &status);
    if (status) fmt::print("Failed to access T3. Status {}\n", status);
    if (status) return std::nullopt;

    fits_close_file(ifile, &status);
    if (status) fmt::print("Failed to Close. Status {}\n", status);
    if (status) return std::nullopt;

    return {
        { cosbins,
         ra, dec,
         nside, nbrbins,
         cosmin, { ordering },
         { coordsys },
         (std::string(thetabin) != "COSTHETA") }
    };
}
