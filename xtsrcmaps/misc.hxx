#pragma once

#include <cmath>

constexpr double pi         = 3.141592653589793238462643383279502884197;
constexpr double twopi      = 6.283185307179586476925286766559005768394;
constexpr double halfpi     = 1.570796326794896619231321691639751442099;
constexpr double inv_halfpi = 0.6366197723675813430755350534900574;
constexpr double pi_180    = 0.017453292519943295769236907684886127134;
constexpr double deg2rad    = 0.017453292519943295769236907684886127134;
constexpr double rad2deg    = 57.295779513082320876798154814105170332405;
constexpr double twothird   = 2.0 / 3.0;

/*
 * Provide missing stl conversion function from a range to a container.
 * Example usage:
 *
 * class Node {};
 *
 * int main()
 * {
 *     std::vector<std::shared_ptr<Node>> nodes;
 *
 *     auto raw_nodes = to<std::vector<Node*>>(std::views::transform(nodes, [] (auto&
 * node) { return node.get();
 *     }));
 * }
 */
template <typename ContainerT, typename RangeT>
ContainerT
to(RangeT&& range)
{
    return ContainerT(begin(range), end(range));
}


/*
 * User defined literal to make converting degrees to radians simpler.
 */
// constexpr double operator"" _deg(long double deg) { return deg * deg2rad; }

constexpr double
radians(double const x)
{
    // return x * M_PI / 180.;
    return x * deg2rad;
}
constexpr double
degrees(double const x)
{
    // return x * 180. / M_PI;
    return x * rad2deg;
}
