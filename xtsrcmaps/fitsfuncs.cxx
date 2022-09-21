#include "xtsrcmaps/fitsfuncs.hxx"

#include "fitsio.h"
#include "fmt/format.h"

#include <cassert>
#include <numeric>

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


/*
 *
 */
auto
Fermi::fits::read_ltcube(std::string const& filename) -> std::optional<LiveTimeCubeData>
{
    // Use CFITSIO to open the ltcube and read the values in the header.
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

    // Read the Exposure parameters
    char expsr[] { "EXPOSURE" };
    fits_movnam_hdu(ifile, BINARY_TBL, expsr, 0, &status);
    if (status) fmt::print("Failed to access Exposure header. Status {}\n", status);
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

    // Populate the local Exposure Param vectors.
    auto cosbins = vector<float>(nbrbins * 12 * nside * nside, 0.0f);
    auto ra      = vector<float>(12 * nside * nside, 0.0f);
    auto dec     = vector<float>(12 * nside * nside, 0.0f);

    fits_read_col(ifile, TFLOAT, 1, 1, 1, 40, nullptr, &cosbins[0], nullptr, &status);
    if (status) fmt::print("Failed to access T1. Status {}\n", status);
    if (status) return std::nullopt;
    fits_read_col(ifile, TFLOAT, 2, 1, 1, 1, nullptr, &ra[0], nullptr, &status);
    if (status) fmt::print("Failed to access T2. Status {}\n", status);
    if (status) return std::nullopt;
    fits_read_col(ifile, TFLOAT, 3, 1, 1, 1, nullptr, &dec[0], nullptr, &status);
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
         !strcmp(thetabin, "COSTHETA")}
    };
}

auto
Fermi::fits::read_irf_pars(std::string const& filename, std::string const& tblname)
    -> std::optional<TablePars>
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
    for (int t = 0; t < ncols; ++t)
    {
        auto tf       = t + 1;
        int  typecode = 0;
        long width    = 0;
        fits_get_coltype(ifile, tf, &typecode, &col_repeat[t], &width, &status);
        if (status)
        {
            fmt::print(
                "Failed reading FITS column size in {} at column {}. Status {}\n",
                c_tbl,
                tf,
                status);
            return std::nullopt;
        }
        if (typecode != TFLOAT)
        {
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
    auto rowdata = vector<vector<float>>(nrows, vector<float>(row_width, 0.0f));

    for (unsigned int n = 0; n < nrows; ++n)
    {
        auto const nf    = n + 1;
        // Read the fits row data into a correctly sized, pre-allocated array.
        auto const sz    = sizeof(float) * row_width;
        auto       bytes = vector<unsigned char>(sz, 0);
        fits_read_tblbytes(ifile, nf, 1, sz, &bytes[0], &status);
        if (status)
        {
            fmt::print("Failed reading FITS tblbytes in {} at row {}. Status {}\n",
                       c_tbl,
                       nf,
                       status);
            return std::nullopt;
        }
        // Copy the raw bytes into the float array.
        std::memcpy(rowdata[n].data(), bytes.data(), sz);
    }

    fits_close_file(ifile, &status);
    if (status) fmt::print("Failed to Close. Status {}\n", status);
    if (status) return std::nullopt;

    return {
        {extents, rowdata}
    };
}

//
// /*
//  * Given an IRF Point Spread Function fits file, read it and return an optional
//  wrapped
//  * PsfParamData object if successful, or a nullopt if not.
//  */
// auto
// Fermi::fits::read_psf(std::string const& filename) -> std::optional<PsfParamData>
// {
//
//     // Use CFITSIO to open the ccube and read the energies in the header.
//     int       status = 0;
//     fitsfile* ifile;
//     fits_open_file(&ifile, filename.c_str(), READONLY, &status);
//
//     // open the file, or return a null optional if it cannot open.
//     if (status)
//     {
//         fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
//                    filename,
//                    status);
//         return std::nullopt;
//     }
//
//     // Read the RPSF parameters
//     char rpsf[] { "RPSF_FRONT" };
//     fits_movnam_hdu(ifile, BINARY_TBL, rpsf, 0, &status);
//     if (status) fmt::print("Failed to access RPSF header. Status {}\n", status);
//     if (status) return std::nullopt;
//
//     // Populate the local RPSF Param vectors.
//     auto energ_lo = vector<float>(23, 0.0f);
//     auto energ_hi = vector<float>(23, 0.0f);
//     auto cthet_lo = vector<float>(8, 0.0f);
//     auto cthet_hi = vector<float>(8, 0.0f);
//     auto ncore    = vector<float>(184, 0.0f);
//     auto ntail    = vector<float>(184, 0.0f);
//     auto score    = vector<float>(184, 0.0f);
//     auto stail    = vector<float>(184, 0.0f);
//     auto gcore    = vector<float>(184, 0.0f);
//     auto gtail    = vector<float>(184, 0.0f);
//
//     fits_read_col(ifile, TFLOAT, 1, 1, 1, 23, nullptr, &energ_lo[0], nullptr,
//     &status); if (status) fmt::print("Failed to access T1. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 2, 1, 1, 23, nullptr, &energ_hi[0], nullptr,
//     &status); if (status) fmt::print("Failed to access T2. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 3, 1, 1, 8, nullptr, &cthet_lo[0], nullptr,
//     &status); if (status) fmt::print("Failed to access T3. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 4, 1, 1, 8, nullptr, &cthet_hi[0], nullptr,
//     &status); if (status) fmt::print("Failed to access T4. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 5, 1, 1, 184, nullptr, &ncore[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T5. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 6, 1, 1, 184, nullptr, &ntail[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T6. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 7, 1, 1, 184, nullptr, &score[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T7. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 8, 1, 1, 184, nullptr, &stail[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T8. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 9, 1, 1, 184, nullptr, &gcore[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T9. Status {}\n", status);
//     fits_read_col(ifile, TFLOAT, 10, 1, 1, 184, nullptr, &gtail[0], nullptr,
//     &status); if (status) fmt::print("Failed to access T10. Status {}\n", status); if
//     (status) return std::nullopt;
//
//     // Read the PSF scaling factors.
//     char psf_scale[] { "PSF_SCALING_PARAMS_FRONT" };
//     fits_movnam_hdu(ifile, BINARY_TBL, psf_scale, 0, &status);
//     if (status) fmt::print("Failed to access PSF_SCALING header. Status {}\n",
//     status); if (status) return std::nullopt;
//
//     // Populate the psf scaling factors.
//     auto scale = vector<float>(3, 0.0f);
//     fits_read_col(ifile, TFLOAT, 1, 1, 1, 3, nullptr, &scale[0], nullptr, &status);
//     if (status) fmt::print("Failed to access P1. Status {}\n", status);
//     if (status) return std::nullopt;
//
//     fits_close_file(ifile, &status);
//     if (status) fmt::print("Failed to Close. Status {}\n", status);
//     if (status) return std::nullopt;
//
//     return {
//         {energ_lo,
//          energ_hi, cthet_lo,
//          cthet_hi, ncore,
//          ntail, score,
//          stail, gcore,
//          gtail, scale[0],
//          scale[1],
//          scale[2]}
//     };
// }
//
//
// auto
// Fermi::fits::read_irf_grid(std::string const& filename, std::string const& tblname)
//     -> std::optional<IrfGrid>
// {
//
//     // Use CFITSIO to open the ccube and read the energies in the header.
//     int       status = 0;
//     fitsfile* ifile;
//     fits_open_file(&ifile, filename.c_str(), READONLY, &status);
//
//     // open the file, or return a null optional if it cannot open.
//     if (status)
//     {
//         fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
//                    filename,
//                    status);
//         return std::nullopt;
//     }
//
//     char c_tbl[128] {};
//     tblname.copy(c_tbl, tblname.size() > 127 ? 127 : tblname.size());
//     c_tbl[127] = '\0';
//     fits_movnam_hdu(ifile, BINARY_TBL, c_tbl, 0, &status);
//     if (status) fmt::print("Failed to access {} header. Status {}\n", c_tbl, status);
//     if (status) return std::nullopt;
//
//     // read param count (number of param grids)
//     unsigned int tfields = 0;
//     fits_read_key(ifile, TUINT, "TFIELDS", &tfields, NULL, &status);
//     if (status) fmt::print("Failed TFIELDS. Status {}\n", c_tbl, status);
//     // read dimensions
//     char cen[32] {};
//     fits_read_key(ifile, TSTRING, "TFORM1", &cen, NULL, &status);
//     if (status) fmt::print("Failed TFORM1. Status {}\n", c_tbl, status);
//     size_t Nenergy = std::stoi(string(cen).substr(0, string(cen).find_first_of('E')));
//
//     char ccn[32] {};
//     fits_read_key(ifile, TSTRING, "TFORM3", &ccn, NULL, &status);
//     if (status) fmt::print("Failed TFORM3. Status {}\n", c_tbl, status);
//     size_t Ncosthe = std::stoi(string(ccn).substr(0, string(ccn).find_first_of('E')));
//
//     size_t pn      = Nenergy * Ncosthe;
//     // fmt::print("AEFF:\t{}\t{}\t{}\n", tfields, Nenergy, Ncosthe);
//     // read each param grid
//
//     // // Populate the local RPSF Param vectors.
//     auto energ_lo  = vector<float>(Nenergy, 0.0f);
//     auto energ_hi  = vector<float>(Nenergy, 0.0f);
//     auto cthet_lo  = vector<float>(Ncosthe, 0.0f);
//     auto cthet_hi  = vector<float>(Ncosthe, 0.0f);
//
//     fits_read_col(
//         ifile, TFLOAT, 1, 1, 1, Nenergy, nullptr, &energ_lo[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T1. Status {}\n", status);
//     fits_read_col(
//         ifile, TFLOAT, 2, 1, 1, Nenergy, nullptr, &energ_hi[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T2. Status {}\n", status);
//     fits_read_col(
//         ifile, TFLOAT, 3, 1, 1, Ncosthe, nullptr, &cthet_lo[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T3. Status {}\n", status);
//     fits_read_col(
//         ifile, TFLOAT, 4, 1, 1, Ncosthe, nullptr, &cthet_hi[0], nullptr, &status);
//     if (status) fmt::print("Failed to access T4. Status {}\n", status);
//
//     size_t Ngrids = tfields - 4;
//
//     auto params   = vector<vector<float>>(Ngrids);
//
//     for (size_t i = 0; i < Ngrids; ++i)
//     {
//         params.emplace(params.begin() + i, pn, 0.0f);
//         fits_read_col(
//             ifile, TFLOAT, i + 5, 1, 1, pn, nullptr, &params[i][0], nullptr, &status);
//         if (status) fmt::print("Failed to access T5. Status {}\n", status);
//     }
//
//     fits_close_file(ifile, &status);
//     if (status) fmt::print("Failed to Close. Status {}\n", status);
//     if (status) return std::nullopt;
//
//     return {
//         {energ_lo, energ_hi, cthet_lo, cthet_hi, params, Nenergy, Ncosthe, Ngrids}
//     };
// }
// auto
// Fermi::fits::read_irf_scale(std::string const& filename, std::string const& tblname)
//     -> std::optional<IrfScale>
// {
//
//     // Use CFITSIO to open the ccube and read the energies in the header.
//     int       status = 0;
//     fitsfile* ifile;
//     fits_open_file(&ifile, filename.c_str(), READONLY, &status);
//
//     // open the file, or return a null optional if it cannot open.
//     if (status)
//     {
//         fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
//                    filename,
//                    status);
//         return std::nullopt;
//     }
//
//     char c_tbl[128] {};
//     tblname.copy(c_tbl, tblname.size() > 127 ? 127 : tblname.size());
//     c_tbl[127] = '\0';
//     // Read the PSF scaling factors.
//     fits_movnam_hdu(ifile, BINARY_TBL, c_tbl, 0, &status);
//     if (status) fmt::print("Failed to access {} header. Status {}\n", c_tbl, status);
//     if (status) return std::nullopt;
//
//     // Populate the psf scaling factors.
//     auto scale = vector<float>(3, 0.0f);
//     fits_read_col(ifile, TFLOAT, 1, 1, 1, 3, nullptr, &scale[0], nullptr, &status);
//     if (status)
//         fmt::print("Failed to access Scaling Params in table {}. Status {}\n",
//                    tblname,
//                    status);
//     if (status) return std::nullopt;
//
//     fits_close_file(ifile, &status);
//     if (status) fmt::print("Failed to Close. Status {}\n", status);
//     if (status) return std::nullopt;
//
//     return {
//         {scale[0], scale[1], scale[2]}
//     };
// }
//
// auto
// Fermi::fits::read_irf_efficiency(std::string const& filename,
//                                  std::string const& tblname) -> std::optional<IrfEffic>
// {
//
//     // Use CFITSIO to open the ccube and read the energies in the header.
//     int       status = 0;
//     fitsfile* ifile;
//     fits_open_file(&ifile, filename.c_str(), READONLY, &status);
//
//     // open the file, or return a null optional if it cannot open.
//     if (status)
//     {
//         fmt::print("Failed to open {}. Status {}. Returning an empty optional.\n",
//                    filename,
//                    status);
//         return std::nullopt;
//     }
//
//     char c_tbl[128] {};
//     tblname.copy(c_tbl, tblname.size() > 127 ? 127 : tblname.size());
//     c_tbl[127] = '\0';
//     // Move to the efficiency params table
//     fits_movnam_hdu(ifile, BINARY_TBL, c_tbl, 0, &status);
//     if (status) fmt::print("Failed to access {} header. Status {}\n", c_tbl, status);
//     if (status) return std::nullopt;
//
//     // Populate the Efficiency parameters.
//     auto p0 = std::array<float, 6> { 0.0 };
//     auto p1 = std::array<float, 6> { 0.0 };
//     fits_read_col(ifile, TFLOAT, 1, 1, 1, 6, nullptr, &p0[0], nullptr, &status);
//     fits_read_col(ifile, TFLOAT, 1, 2, 1, 6, nullptr, &p1[0], nullptr, &status);
//     if (status)
//         fmt::print("Failed to access Efficiency Params in table {}. Status {}\n",
//                    tblname,
//                    status);
//     if (status) return std::nullopt;
//
//     fits_close_file(ifile, &status);
//     if (status) fmt::print("Failed to Close. Status {}\n", status);
//     if (status) return std::nullopt;
//
//     return {
//         {p0, p1}
//     };
// }
