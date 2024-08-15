#pragma once

#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"


namespace Fermi::ModelMap {

auto compute_srcmaps(XtObs const& obs,
                     XtExp const& exp,
                     XtPsf const& psf) -> Tensor<float, 4>;

auto
point_src_model_map_wcs(size_t const                    Nh,
                        size_t const                    Nw,
                        Tensor<double, 2> const&        src_sph,
                        std::vector<std::string> const& src_names,
                        Tensor<float, 3> const&         uPsf,
                        SkyGeom<float> const&           skygeom,
                        Tensor<double, 2> const&        exposures,
                        Tensor<float, 3> const& partial_integrals /* [SDE] */
                        ) -> Tensor<float, 4>;


void scale_map_by_solid_angle(Tensor<float, 4>&     model_map,
                              SkyGeom<float> const& skygeom);

void scale_map_by_exposure(Tensor<float, 4>&        model_map,
                           Tensor<double, 2> const& exposures);

auto map_correction_factor(Tensor<float, 2> const& MapInteg,

                           Tensor<double, 1> const& psf_radius,
                           std::vector<bool> const& is_in_fov,
                           Tensor<float, 3> const&  mean_psf,         // [SED]
                           Tensor<float, 3> const&  partial_integrals // [SED]
                           ) -> Tensor<float, 2>;

void scale_map_by_correction_factors(Tensor<float, 4>&       model_map,
                                     Tensor<float, 2> const& factor //[SE]
);

auto psf_boundary_radius(size_t const             Nh,
                         size_t const             Nw,
                         Tensor<double, 2> const& src_sph,
                         SkyGeom<float> const&    skygeom)
    -> std::pair<Tensor<double, 1>, std::vector<bool>>;

auto map_integral(Tensor<float, 4> const&  model_map,
                  Tensor<double, 2> const& src_sph,
                  SkyGeom<float> const&    skygeom,
                  Tensor<double, 1> const& psf_radius,
                  std::vector<bool> const& is_in_fov) -> Tensor<float, 2>;

auto integral(Tensor<double, 1> const& angles,
              Tensor<float, 3> const&  mean_psf,         // [D,E,S]
              Tensor<float, 3> const&  partial_integrals // [D,E,S]
              ) -> Tensor<float, 2>;

template <typename T>
Fermi::Tensor<T, 1>
filter_in(std::vector<T> const& A, std::vector<bool> const& M) {
    std::size_t Nt = std::count(M.begin(), M.end(), true);

    assert(A.size() == M.size()); // Ensure the Tensors have the same size
    assert(A.size() >= Nt);

    Fermi::Tensor<T, 1> B(Nt);

    size_t t = 0;
    for (size_t i = 0; i < A.size(); ++i) {
        if (M[i]) { B[t++] = A[i]; }
    }

    return B;
}

} // namespace Fermi::ModelMap
