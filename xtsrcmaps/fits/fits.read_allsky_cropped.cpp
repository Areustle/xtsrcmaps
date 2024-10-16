#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/skyimage/skyimage.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

#include "fitsio.h"

#include <memory>

using namespace Fermi;

// Read WCS metadata from a FITS file
WcsConfig
readWcsParams(fitsfile* fptr) {
    WcsConfig meta;
    int       status = 0;
    fits_get_img_dim(fptr, nullptr, &status);
    fits_get_img_size(fptr, 3, meta.naxes.data(), &status);
    fits::handleError(status);

    auto readKey = [fptr, &status](const char* key, double& value) {
        fits_read_key(fptr, TDOUBLE, key, &value, nullptr, &status);
        if (status == KEY_NO_EXIST) {
            status = 0; // If key doesn't exist, default to 0
            value  = 0.0;
        }
        fits::handleError(status);
    };

    readKey("CRPIX1", meta.crpix[0]);
    readKey("CRPIX2", meta.crpix[1]);
    meta.crpix[2] = 1.0; // Default for 2D images
    readKey("CRVAL1", meta.crval[0]);
    readKey("CRVAL2", meta.crval[1]);
    readKey("CDELT1", meta.cdelt[0]);
    readKey("CDELT2", meta.cdelt[1]);
    readKey("CROTA2", meta.axis_rot);

    char ctype1[80];
    fits_read_key(fptr, TSTRING, "CTYPE1", ctype1, nullptr, &status);
    fits::handleError(status);
    meta.is_galactic = (strstr(ctype1, "GLON") != nullptr);
    meta.proj_name   = std::string(ctype1).substr(5);
    return meta;
}

// Read a subset of the image data
void
readImageSubset(fitsfile*            fptr,
                std::array<long, 3>& fpixel,
                std::array<long, 3>& lpixel,
                float*               buffer) {
    std::array<long, 3> inc     = { 1, 1, 1 };
    float               nullval = 0.0;
    int                 anynul  = 0;
    int                 status  = 0;

    // Use fits_read_subset to read the subset of the image
    if (fits_read_subset(fptr,
                         TFLOAT,
                         fpixel.data(),
                         lpixel.data(),
                         inc.data(),
                         &nullval,
                         buffer,
                         &anynul,
                         &status)) {
        fits::handleError(status);
    }
}

// Adjust sky coordinates and handle wrap-around at 0/360 degrees and poles
auto
adjustSkyCoords(std::array<double, 2>& coord) {
    coord[0] = fmod(coord[0] + 360.0, 360.0); // Wrap longitude into [0, 360)
    coord[1] = std::clamp(coord[1], -90.0, 90.0); // Clamp latitude to [-90, 90]
};

// Merges two image segments into a single full ROI array
// Takes two segments (seg1 and seg2) and combines them into the full ROI array
// (fullROI). It handles multi-channel fits files by iterating over channels
// (NAXIS3) and rows (NAXIS2), copying data from the first segment, then the
// second segment, into the correct positions in the FullROI output array.
void
mergeImageSegments(const tensor_details::AlignedVector<float>& seg1,
                   const tensor_details::AlignedVector<float>& seg2,
                   float*                                      output,
                   long fullw, // FullWidth
                   long seg1w  // Seg1 Width
) {
    long seg2w     = fullw - seg1w; // Width of the second segment

    // Determine the total number of elements in one channel
    long nchannels = seg1.size() / (seg1w * seg2w);
    long nrows     = seg1.size() / seg1w / nchannels;

    // Iterate over each channel and row to merge segments into the full ROI
    for (long z = 0; z < nchannels; ++z) { // For each channel
        for (long y = 0; y < nrows; ++y) { // For each row
            // Copy data from Segment 1
            std::copy(seg1.begin() + (z * nrows * seg1w) + (y * seg1w),
                      seg1.begin() + (z * nrows * seg1w) + (y * seg1w) + seg1w,
                      output + (z * nrows * fullw) + (y * fullw));

            // Copy data from Segment 2
            std::copy(seg2.begin() + (z * nrows * seg2w) + (y * seg2w),
                      seg2.begin() + (z * nrows * seg2w) + (y * seg2w) + seg2w,
                      output + (z * nrows * fullw) + (y * fullw) + seg1w);
        }
    }
}

// Read segments across boundaries if needed
void
readSegmentsAcrossBoundary(fitsfile*                  fptr,
                           std::array<long, 3>&       fpixel,
                           std::array<long, 3>&       lpixel,
                           float*                     output,
                           const std::array<long, 3>& maxBounds) {
    if (fpixel[0] <= lpixel[0]) { // Normal read without boundary crossing
        readImageSubset(fptr, fpixel, lpixel, output);
    } else { // Split read across boundary
        std::array<long, 3> seg1_fp = { fpixel[0], fpixel[1], fpixel[2] };
        std::array<long, 3> seg1_lp = { maxBounds[0], lpixel[1], lpixel[2] };
        tensor_details::AlignedVector<float> buffer1(
            (seg1_lp[0] - seg1_fp[0] + 1) * (seg1_lp[1] - seg1_fp[1] + 1)
            * (seg1_lp[2] - seg1_fp[2] + 1));
        readImageSubset(fptr, seg1_fp, seg1_lp, buffer1.data());

        std::array<long, 3> seg2_fp = { 1, fpixel[1], fpixel[2] };
        std::array<long, 3> seg2_lp = { lpixel[0], lpixel[1], lpixel[2] };
        tensor_details::AlignedVector<float> buffer2(
            (seg2_lp[0] - seg2_fp[0] + 1) * (seg2_lp[1] - seg2_fp[1] + 1)
            * (seg2_lp[2] - seg2_fp[2] + 1));
        readImageSubset(fptr, seg2_fp, seg2_lp, buffer2.data());

        // Combine the buffers
        mergeImageSegments(buffer1,
                           buffer2,
                           output,
                           maxBounds[0],
                           seg1_lp[0] - seg1_fp[0] + 1);
    }
}



auto
Fermi::fits::read_allsky_cropped(SkyGeom<double> const& roiGeom,
                                 std::string const&     allskyfilename)
    -> Fermi::SkyImage<float, 3> {
    try {

        // Open all-sky and ROI FITS files
        auto skyFptr    = fits::safe_open(allskyfilename.c_str());

        // Initialize SkyGeom objects
        auto allSkyMeta = readWcsParams(skyFptr.get());
        auto allSkyGeom = SkyGeom<double>(allSkyMeta);

        // Define ROI in pixel coordinates and convert to sky coordinates
        std::array<double, 2> roiPixMin = { 1.0, 1.0 };
        std::array<double, 2> roiPixMax
            = { static_cast<double>(roiGeom.naxes()[0]),
                static_cast<double>(roiGeom.naxes()[1]) };
        auto allSkySphMin = roiGeom.pix2sph(roiPixMin);
        auto allSkySphMax = roiGeom.pix2sph(roiPixMax);
        adjustSkyCoords(allSkySphMin);
        adjustSkyCoords(allSkySphMax);

        // Convert sky coordinates to pixel coordinates in all-sky image
        auto allSkyPixMin = allSkyGeom.sph2pix(allSkySphMin);
        auto allSkyPixMax = allSkyGeom.sph2pix(allSkySphMax);

        // Determine pixel range to read
        auto bounpix      = [](double v, long max) {
            return std::max(1L,
                            std::min(static_cast<long>(std::round(v)), max));
        };
        std::array<long, 3> fpixel
            = { bounpix(std::min(allSkyPixMin[0], allSkyPixMax[0]),
                        allSkyMeta.naxes[0]),
                bounpix(std::min(allSkyPixMin[1], allSkyPixMax[1]),
                        allSkyMeta.naxes[1]),
                1 };

        std::array<long, 3> lpixel
            = { bounpix(std::max(allSkyPixMin[0], allSkyPixMax[0]),
                        allSkyMeta.naxes[0]),
                bounpix(std::max(allSkyPixMin[1], allSkyPixMax[1]),
                        allSkyMeta.naxes[1]),
                (allSkyMeta.naxes[2] > 1) ? allSkyMeta.naxes[2] : 1 };

        // Read image segments across boundary if necessary
        Fermi::Tensor<float, 3> allskyBuffer(
            roiGeom.naxes()[0], roiGeom.naxes()[1], allSkyGeom.naxes()[2]);
        readSegmentsAcrossBoundary(skyFptr.get(),
                                   fpixel,
                                   lpixel,
                                   allskyBuffer.data(),
                                   allSkyMeta.naxes);

        auto energies = good(fits::read_energies(allskyfilename),
                             "Cannot read energies from diffuse file");

        return { .data     = allskyBuffer,
                 .skygeom  = allSkyGeom,
                 .energies = energies };


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return {};
    }
}
