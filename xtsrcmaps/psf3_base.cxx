/**************************************************************************************
 *
 *************************************************************************************/
#include "xtsrcmaps/psf.hxx"

#include "experimental/mdspan"

#include <cmath>

using std::experimental::extents;
using std::experimental::full_extent;
using std::experimental::mdspan;
using std::experimental::submdspan;

template <typename Lp>
inline auto
moffat_single(double const& sep,
              // float const&                                energy,
              mdspan<float, extents<float, 6>, Lp> const& pars) noexcept -> double
{
    float const& ncore = pars[0];
    float const& ntail = pars[1];
    float const& score = pars[2]; //* scaleFactor(energy) }; // Prescale the PSF "s"
    float const& stail = pars[3]; //* scaleFactor(energy) }; // pars by scaleFactor(IE)
    float const& gcore = pars[4]; // assured not to be 1.0
    float const& gtail = pars[5]; // assured not to be 1.0

    double rc          = sep / score;
    double uc          = rc * rc / 2.;

    double rt          = sep / stail;
    double ut          = rt * rt / 2.;

    // scaled king function
    return (ncore * (1. - 1. / gcore) * std::pow(1. + uc / gcore, -gcore)
            + ntail * ncore * (1. - 1. / gcore) * std::pow(1. + ut / gtail, -gtail));
    // should be able to compute x^-g as exp(-g*ln(x)) with SIMD log and exp.
}

template <typename L0, typename L1, typename L2>
inline auto
psf3_single(double const&                                     costh,
            double const&                                     sep,
            double const&                                     energy,
            mdspan<float, extents<float, 2, 2, 6>, L0> const& IP,
            mdspan<float, extents<float, 2>, L1> const&       IE,
            mdspan<float, extents<float, 2>, L2> const&       IC) noexcept -> double
{
    auto ee = (energy - IE[0]) / (IE[1] - IE[0]);
    auto cc = (costh - IC[0]) / (IC[1] - IC[0]);

    // compute the king function on each set of parameters to evaluate the PSF.
    // scaled, bilinear interpolation of evaluated psf.
    // clang-format off
    return costh * (
        (1.-ee)*(1.-cc) * moffat_single(sep, submdspan(IP, 0, 0, full_extent))
        +    ee*(1.-cc) * moffat_single(sep, submdspan(IP, 0, 1, full_extent))
        +        ee*cc  * moffat_single(sep, submdspan(IP, 1, 1, full_extent))
        +   (1.-ee)*cc  * moffat_single(sep, submdspan(IP, 1, 0, full_extent))
    );
    // return costh * (
    //     (1.-tt)*(1.-uu) * moffat_single(sep, IE[0], submdspan(IP, 0, 0, full_extent))
    //     +    tt*(1.-uu) * moffat_single(sep, IE[1], submdspan(IP, 0, 1, full_extent))
    //     +        tt*uu  * moffat_single(sep, IE[1], submdspan(IP, 1, 1, full_extent))
    //     +   (1.-tt)*uu  * moffat_single(sep, IE[0], submdspan(IP, 1, 0, full_extent))
    // );
    // clang-format on
    // should be able to compute x^-g as exp(-g*ln(x)) with SIMD log and exp.
}

// double Bilinear::evaluate(double tt, double uu,
//                           const double * zvals) {
//    double value = ( (1. - tt)*(1. - uu)*zvals[0]
//                     + tt*(1. - uu)*zvals[1]
//                     + tt*uu*zvals[2]
//                     + (1. - tt)*uu*zvals[3] );
//    return value;
// }


// template <typename Lx, typename Le, typename Lp>
// auto
// psf3_avx2(double const&                                  sep,
//           mdspan<float, extents<float, 4>, Le> const&    x,
//           mdspan<float, extents<float, 2>, Le> const&    scaleFactors,
//           mdspan<float, extents<float, 4, 6>, Lp> const& pars) noexcept -> double
// {
// }
//
// void
// Psf3::getCornerPars(double               energy,
//                     double               theta,
//                     double&              tt,
//                     double&              uu,
//                     std::vector<double>& cornerEnergies,
//                     std::vector<size_t>& indx) const
// {
//     double logE(std::log10(energy));
//     double costh(std::cos(theta * M_PI / 180.));
//     int    i(findIndex(m_logEs, logE));
//     int    j(findIndex(m_cosths, costh));
//
//     tt                = (logE - m_logEs[i - 1]) / (m_logEs[i] - m_logEs[i - 1]);
//     uu                = (costh - m_cosths[j - 1]) / (m_cosths[j] - m_cosths[j - 1]);
//     cornerEnergies[0] = m_energies[i - 1];
//     cornerEnergies[1] = m_energies[i];
//     cornerEnergies[2] = m_energies[i];
//     cornerEnergies[3] = m_energies[i - 1];
//
//     size_t xsize(m_energies.size());
//     indx[0] = xsize * (j - 1) + (i - 1);
//     indx[1] = xsize * (j - 1) + (i);
//     indx[2] = xsize * (j) + (i);
//     indx[3] = xsize * (j) + (i - 1);
// }
//
// int
// Psf3::findIndex(const std::vector<double>& xx, double x)
// {
//     typedef std::vector<double>::const_iterator const_iterator_t;
//
//     const_iterator_t ix(std::upper_bound(xx.begin(), xx.end(), x));
//     if (ix == xx.end() && x != xx.back())
//     {
//         std::cout << xx.front() << "  " << x << "  " << xx.back() << std::endl;
//         throw std::invalid_argument("Psf3::findIndex: x out of range");
//     }
//     if (x == xx.back()) { ix = xx.end() - 1; }
//     else if (x <= xx.front()) { ix = xx.begin() + 1; }
//     int i(ix - xx.begin());
//     return i;
// }
