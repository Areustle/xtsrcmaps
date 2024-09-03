#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

namespace Fermi {
template <size_t N3>
auto
square_ptswts(size_t const                  Nh,
              size_t const                  Nw,
              Fermi::SkyGeom<double> const& skygeom,
              std::array<double, N3> cubature_pw) -> Fermi::Tensor<double, 3> {

    constexpr size_t         CARD = N3 / 3;
    Fermi::Tensor<double, 3> points_weights({ Nh, Nw, 4 * CARD });

    // UL d ------- c UR
    //    |         |
    //    |    .    |
    //    |         |
    // LL a ------- b LR

    for (size_t h = 0; h < Nw; ++h) {
        for (size_t w = 0; w < Nw; ++w) {

            double ph = 1. + h;
            double pw = 1. + w;

            std::array<double, 2> const LL { ph - 0.5, pw - 0.5 };
            std::array<double, 2> const LR { ph - 0.5, pw + 0.5 };
            std::array<double, 2> const UL { ph + 0.5, pw - 0.5 };
            std::array<double, 2> const UR { ph + 0.5, pw + 0.5 };

            auto const a = skygeom.pix2dir(LL);
            auto const b = skygeom.pix2dir(LR);
            auto const c = skygeom.pix2dir(UR);
            auto const d = skygeom.pix2dir(UL);

            for (size_t p = 0; p < CARD; ++p) {
                double const& u  = cubature_pw[3 * p];
                double const& v  = cubature_pw[3 * p + 1];
                double        nn = (1.0 - u) * (1.0 - v);
                double        pn = (1.0 + u) * (1.0 - v);
                double        pp = (1.0 + u) * (1.0 + v);
                double        np = (1.0 - u) * (1.0 + v);
                points_weights[h, w, 4 * p]
                    = 0.25 * (nn * a[0] + pn * b[0] + pp * c[0] + np * d[0]);
                points_weights[h, w, 4 * p + 1]
                    = 0.25 * (nn * a[1] + pn * b[1] + pp * c[1] + np * d[1]);
                points_weights[h, w, 4 * p + 2]
                    = 0.25 * (nn * a[2] + pn * b[2] + pp * c[2] + np * d[2]);

                points_weights[h, w, 4 * p + 3] = 0.25 * cubature_pw[3 * p + 2];
            }
        }
    }
    return points_weights;
}
} // namespace Fermi
