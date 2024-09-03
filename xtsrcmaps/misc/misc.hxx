#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>


constexpr double pi         = 3.141592653589793238462643383279502884197;
constexpr double twopi      = 6.283185307179586476925286766559005768394;
constexpr double halfpi     = 1.570796326794896619231321691639751442099;
constexpr double inv_halfpi = 0.6366197723675813430755350534900574;
constexpr double pi_180     = 0.017453292519943295769236907684886127134;
constexpr double deg2rad    = pi_180;
constexpr double rad2deg    = 57.295779513082320876798154814105170332405;
constexpr double D2R        = deg2rad;
constexpr double R2D        = rad2deg;
constexpr double T2D        = 2.0 * rad2deg;
constexpr double twothird   = 2.0 / 3.0;
constexpr double sep_step
    = 0.033731417579011382769913057686378448039204491; // ln(7e5)/399
constexpr float sep_step_f
    = 0.033731417579011382769913057686378448039204491; // ln(7e5)/399
constexpr double _ss = -0.033731417579011382769913057686378448039204491;
constexpr float _ss_f = -0.033731417579011382769913057686378448039204491;
constexpr double recipstep // 399 / ln(7e5)
    = 29.645952402019046699874076387601854602589360539942632073727038565;
constexpr float recipstep_f // 399 / ln(7e5)
    = 29.645952402019046699874076387601854602589360539942632073727038565;
// exp(ln(7e5)/399)
constexpr double sep_delta
    = 1.0343067728020497121475244691736145227943806311378006312643513;
// sep_delta - 1.
constexpr double edm1
    = 0.0343067728020497121475244691736145227943806311378006312643513;
constexpr double xmedm1 = 1e4 / edm1; // 1 / (1e-4 * edm1)
constexpr double _redm1 = 29.148763300179999028597222671574207003288142769360422095500496366;

auto
good(auto opt, std::string const& msg, int const return_code = 1) -> auto {
    if (!opt) {
        std::cerr << msg << std::endl;
        exit(return_code);
    }
    return opt.value();
}

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
 *     auto raw_nodes = to<std::vector<Node*>>(std::views::transform(nodes, []
 * (auto& node) { return node.get();
 *     }));
 * }
 */
template <typename ContainerT, typename RangeT>
ContainerT
to(RangeT&& range) {
    return ContainerT(begin(range), end(range));
}

/*
 * User defined literal for uint64_t
 */
constexpr uint64_t
operator"" _u64(unsigned long long int const x) {
    return uint64_t(x);
}

inline constexpr double
to_radians(double const x) {
    return x * deg2rad;
}
inline constexpr double
to_degrees(double const x) {
    return x * rad2deg;
}

/*
 * User defined literal to make converting degrees to radians simpler.
 */
constexpr double
operator"" _deg(long double deg) {
    return to_radians(deg);
}

namespace Fermi {
template <typename T>
auto
log10_v(std::vector<T> const& v) -> std::vector<T> {
    auto logEs = std::vector<double>(v.size(), 0.0);
    std::transform(v.cbegin(), v.cend(), logEs.begin(), [](auto const& x) {
        return std::log10(x);
    });
    return logEs;
};
} // namespace Fermi
