#pragma once

#include <cmath>

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
constexpr double operator"" _deg(long double deg) { return deg * M_PI / 180; }

constexpr double
radians(double const x)
{
    return x * M_PI / 180.;
}
constexpr double
degrees(double const x)
{
    return x * 180. / M_PI;
}
