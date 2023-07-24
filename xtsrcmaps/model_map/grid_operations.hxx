#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include "unsupported/Eigen/CXX11/Tensor"

/**
 * Center of pixel grid elements in 2d (pix) points.
 **/
auto
pixel_grid_centers(long const Nh, long const Nw) -> Tensor3d;

/**
 * Return the points which make up the 4 point corners of all pixels in 3-vector
 * cartesian coordinates (dir / xyz).
 **/
auto
pixel_grid_vertices(long const Nh, long const Nw, Fermi::SkyGeom const& skygeom)
    -> Tensor3d;

/**
 * Compute the cartesian vector normals to the planes formed by the great circles of all
 * grid lines from the vertices of the pixel grid.
 **/
auto
pixel_grid_normals(Tensor3d const& vertices) -> std::pair<Tensor2d, Tensor2d>;
