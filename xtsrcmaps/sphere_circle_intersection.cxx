
#include "xtsrcmaps/sky_geom.hxx"
#include "unsupported/Eigen/CXX11/Tensor"

auto 
Fermi::naive_csg_intersection_trig(){
    // P0 is the center point of the source-circle
    // r is the spherical length of the source circle. a.k.a. the spherical distance
    // from the source-circle to P0.
    // P1, P2 are non-antipodal points along geodesic
    //
    // N = P1 x P2 / || P1 x P2 ||
    // theta = r/(R=1)
    //
    // if |asin(P0.N)| < theta: 2-point intersection
    // else if |asin(P0.N)| == theta: 1-point intersection
    // else |asin(P0.N)| > theta: no intersection
    //
    // c0 = cos(theta)P0
    //
    // P3 = P2 - (P1.P2)P1 / ||P3||
    //
    // A = P1.P0
    // B = P3.P0
    // t0 = atan2(B, A)
    // u = acos((cos(theta)) / sqrt(A*A + B*B))
    // t = t0 +/- u
    // Q1 = P1 cos(t0+u) + P3 sin(t0+u)
    // Q2 = P1 cos(t0-u) + P3 sin(t0-u)
}


auto
Fermi::naive_csg_intersection_vec(){
}
