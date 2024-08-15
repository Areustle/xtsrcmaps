#include "xtsrcmaps/fits/fits.hxx"

#include "fitsio.h"
#include "fmt/core.h"

#include <bit>
#include <cassert>
#include <vector>

using std::vector;

auto
Fermi::fits::read_irf_pars(std::string const& filename,
                           std::string tblname) -> std::optional<TablePars> {

    // Use CFITSIO to open the ccube and read the energies in the header.
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

    tblname += '\0';
    fits_movnam_hdu(ifile, BINARY_TBL, tblname.data(), 0, &status);
    if (status)
        fmt::print(
            "Failed to access {} header. Status {}\n", tblname.data(), status);
    if (status) return std::nullopt;

    // read param count (number of param grids)
    // read num cols.
    int ncols = 0;
    fits_get_num_cols(ifile, &ncols, &status);
    if (status)
        fmt::print(
            "Failed to get numcols in {}. Status {}\n", tblname.data(), status);
    // int fits_read_key(fitsfile *fptr, int datatype, char *keyname, void
    // *value, char *comment, int *status)

    // read dimensions
    auto col_repeat = vector<long>(ncols, 0);
    for (int t = 0; t < ncols; ++t) {
        auto tf       = t + 1;
        int  typecode = 0;
        long width    = 0;
        fits_get_coltype(ifile, tf, &typecode, &col_repeat[t], &width, &status);
        if (status) {
            fmt::print("Failed reading FITS column size in {} at column {}. "
                       "Status {}\n",
                       tblname.data(),
                       tf,
                       status);
            return std::nullopt;
        }
        if (typecode != TFLOAT) {
            fmt::print("In FITS file {}, Expected TFLOAT column but got {} at "
                       "column {}\n",
                       tblname.data(),
                       typecode,
                       tf);
            return std::nullopt;
        }
        assert(4 == width);
    }

    // Get full row width.
    size_t const row_width
        = std::reduce(col_repeat.cbegin(), col_repeat.cend(), 0);
    auto extents = vector<size_t>(ncols, 0);
    std::copy(col_repeat.cbegin(), col_repeat.cend(), extents.begin());

    // Get num rows.
    long nrows = 0;
    fits_get_num_rows(ifile, &nrows, &status);
    if (status)
        fmt::print(
            "Failed to get numrows in {}. Status {}", tblname.data(), status);

    // Populate the rows vectors.
    Fermi::Tensor<float, 2> rowdata(nrows, row_width);

    for (unsigned int n = 0; n < nrows; ++n) {
        auto const nf    = n + 1;
        // Read the fits row data into a correctly sized, pre-allocated array.
        auto const sz    = sizeof(float) * row_width;
        auto       bytes = vector<uint8_t>(sz, 0);
        fits_read_tblbytes(ifile, nf, 1, sz, &bytes[0], &status);
        if (status) {
            fmt::print(
                "Failed reading FITS tblbytes in {} at row {}. Status {}\n",
                tblname.data(),
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
        std::memcpy(&rowdata[n, 0], bytes.data(), sz);
    }

    fits_close_file(ifile, &status);
    if (status) fmt::print("Failed to Close. Status {}\n", status);
    if (status) return std::nullopt;

    // compute the offsets for each fits vector in the row data.
    auto offsets = std::vector<size_t>(extents.size(), 0);
    std::exclusive_scan(extents.cbegin(), extents.cend(), offsets.begin(), 0.0);


    return {
        { extents, offsets, rowdata }
    };
}
