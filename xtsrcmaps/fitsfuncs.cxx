#include "xtsrcmaps/fitsfuncs.hxx"

#include "fitsio.h"
#include "fmt/format.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <numeric>
#include <optional>

using std::optional;
using std::string;
using std::vector;

auto
Fermi::fits::ccube_energies(std::string const& filename) noexcept
    -> std::optional<std::vector<double>> {
    // Use CFITSIO to open the ccube and read the energies in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    if (status != 0) {
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
        ifile, TDOUBLE, 3, rows, 1, 1, nullptr, &energies[rows], &anynul, &status);

    fits_close_file(ifile, &status);

    for (auto&& e : energies) { e /= 1000.0; }

    return energies;
}

/// Read the header keyword values from the counts cube.
auto
Fermi::fits::ccube_pixels(std::string const& filename) noexcept
    -> std::optional<CCubePixels> {
    // Use CFITSIO to open the ccube and read the energies in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    if (status != 0) {
        fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
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

    unsigned long cols = 0;
    fits_read_key(ifile, TULONG, "NAXIS1", &cols, nullptr, &status);
    if (status) fmt::print("Failed to read NAXIS1. Status {}.\n", status);
    if (status) return std::nullopt;
    unsigned long rows = 0;
    fits_read_key(ifile, TULONG, "NAXIS2", &rows, nullptr, &status);
    if (status) fmt::print("Failed to read NAXIS2. Status {}.\n", status);
    if (status) return std::nullopt;
    unsigned long channels = 0;
    fits_read_key(ifile, TULONG, "NAXIS3", &channels, nullptr, &status);
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

auto
Fermi::fits::read_expcube(std::string const& filename, std::string const& tblname)
    -> std::optional<ExposureCubeData> {
    // Use CFITSIO to open the ltcube and read the values in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    // open the file, or return a null optional if it cannot open.
    if (status) {
        fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
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
    if (status) fmt::print("Failed to access {} header. Status {}\n", c_tbl, status);
    if (status) return std::nullopt;

    // HDU keys
    // fits_read_key (fitsfile *fptr, int datatype, char *keyname, > DTYPE *value,
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
        fmt::print("LiveTimeCube Healpix Ordering {} is Unsupported. Please use NESTED "
                   "order.\n",
                   ordering);
        return std::nullopt;
    }
    if (std::string { coordsys } != "EQU") {
        fmt::print(
            "LiveTimeCube COORDSYS {} is Unsupported. Please use EQU coordinates.\n",
            ordering);
        return std::nullopt;
    }

    // Populate the local Exposure Param vectors.
    size_t const npix    = 12 * nside * nside;
    auto         cosbins = vector<float>(nbrbins * npix, 0.0f);
    auto         ra      = vector<float>(npix, 0.0f);
    auto         dec     = vector<float>(npix, 0.0f);

    fits_read_col(
        ifile, TFLOAT, 1, 1, 1, nbrbins * npix, nullptr, &cosbins[0], nullptr, &status);
    if (status) fmt::print("Failed to access T1. Status {}\n", status);
    if (status) return std::nullopt;
    fits_read_col(ifile, TFLOAT, 2, 1, 1, npix, nullptr, &ra[0], nullptr, &status);
    if (status) fmt::print("Failed to access T2. Status {}\n", status);
    if (status) return std::nullopt;
    fits_read_col(ifile, TFLOAT, 3, 1, 1, npix, nullptr, &dec[0], nullptr, &status);
    if (status) fmt::print("Failed to access T3. Status {}\n", status);
    if (status) return std::nullopt;

    fits_close_file(ifile, &status);
    if (status) fmt::print("Failed to Close. Status {}\n", status);
    if (status) return std::nullopt;

    return {
        {cosbins,
         ra, dec,
         nside, nbrbins,
         cosmin, { ordering },
         { coordsys },
         (std::string(thetabin) != "COSTHETA")}
    };
}

auto
Fermi::fits::read_irf_pars(std::string const& filename, std::string const& tblname)
    -> std::optional<TablePars> {

    // Use CFITSIO to open the ccube and read the energies in the header.
    int       status = 0;
    fitsfile* ifile;
    fits_open_file(&ifile, filename.c_str(), READONLY, &status);

    // open the file, or return a null optional if it cannot open.
    if (status) {
        fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
                   filename,
                   status);
        return std::nullopt;
    }

    char   c_tbl[128] {};
    size_t tbsz = tblname.size() > 127 ? 127 : tblname.size();
    tblname.copy(c_tbl, tbsz);
    c_tbl[tbsz] = '\0';
    fits_movnam_hdu(ifile, BINARY_TBL, c_tbl, 0, &status);
    if (status) fmt::print("Failed to access {} header. Status {}\n", c_tbl, status);
    if (status) return std::nullopt;

    // read param count (number of param grids)
    // read num cols.
    int ncols = 0;
    fits_get_num_cols(ifile, &ncols, &status);
    if (status) fmt::print("Failed to get numcols in {}. Status {}\n", c_tbl, status);
    // int fits_read_key(fitsfile *fptr, int datatype, char *keyname, void *value, char
    // *comment, int *status)

    // read dimensions
    auto col_repeat = vector<long>(ncols, 0);
    for (int t = 0; t < ncols; ++t) {
        auto tf       = t + 1;
        int  typecode = 0;
        long width    = 0;
        fits_get_coltype(ifile, tf, &typecode, &col_repeat[t], &width, &status);
        if (status) {
            fmt::print(
                "Failed reading FITS column size in {} at column {}. Status {}\n",
                c_tbl,
                tf,
                status);
            return std::nullopt;
        }
        if (typecode != TFLOAT) {
            fmt::print(
                "In FITS file {}, Expected TFLOAT column but got {} at column {}\n",
                c_tbl,
                typecode,
                tf);
            return std::nullopt;
        }
        assert(4 == width);
    }

    // Get full row width.
    size_t const row_width = std::reduce(col_repeat.cbegin(), col_repeat.cend(), 0);
    auto         extents   = vector<size_t>(ncols, 0);
    std::copy(col_repeat.cbegin(), col_repeat.cend(), extents.begin());

    // Get num rows.
    long nrows = 0;
    fits_get_num_rows(ifile, &nrows, &status);
    if (status) fmt::print("Failed to get numrows in {}. Status {}", c_tbl, status);

    // Populate the rows vectors.
    // auto rowdata = vector<vector<float>>(nrows, vector<float>(row_width, 0.0f));
    RowTensor2f rowdata(nrows, row_width);

    for (unsigned int n = 0; n < nrows; ++n) {
        auto const nf    = n + 1;
        // Read the fits row data into a correctly sized, pre-allocated array.
        auto const sz    = sizeof(float) * row_width;
        auto       bytes = vector<uint8_t>(sz, 0);
        fits_read_tblbytes(ifile, nf, 1, sz, &bytes[0], &status);
        if (status) {
            fmt::print("Failed reading FITS tblbytes in {} at row {}. Status {}\n",
                       c_tbl,
                       nf,
                       status);
            return std::nullopt;
        }
        // Optionally swap the endianness of the bytes
        if (std::endian::native == std::endian::little) {
            for (size_t i = 0; i < row_width; ++i) {
                std::reverse(&bytes[i * sizeof(float)],
                             &bytes[(i + 1) * sizeof(float)]);
            }
        }
        // Copy the raw bytes into the float array.
        std::memcpy(&rowdata(n, 0), bytes.data(), sz);
    }

    fits_close_file(ifile, &status);
    if (status) fmt::print("Failed to Close. Status {}\n", status);
    if (status) return std::nullopt;

    // compute the offsets for each fits vector in the row data.
    auto offsets = std::vector<size_t>(extents.size(), 0);
    std::exclusive_scan(extents.cbegin(), extents.cend(), offsets.begin(), 0.0);


    return {
        {extents, offsets, rowdata.swap_layout()}
    };
}
