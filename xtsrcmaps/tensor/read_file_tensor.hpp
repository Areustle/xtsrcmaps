#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"

#include <fstream>
#include <string>

namespace Fermi {

template <typename T = double, size_t R>
    requires(R > 0uz)
auto
read_file_tensor(std::string const&           filename,
                 std::array<size_t, R> const& extents) {

    using VT       = std::conditional_t<std::is_same_v<T, bool>,
                                        std::vector<char>,
                                        std::vector<T>>;

    size_t bufsize = std::accumulate(
        extents.begin(), extents.end(), 1uz, std::multiplies {});
    VT            v = VT(bufsize);
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (!ifs) { throw std::runtime_error("Failed to open file: " + filename); }
    ifs.read(reinterpret_cast<char*>(v.data()), sizeof(T) * (bufsize));
    if (!ifs) {
        throw std::runtime_error(
            "Failed to read the expected amount of data from file: "
            + filename);
    }
    ifs.close();

    Tensor<T, R> new_tensor(v, extents);

    return new_tensor;
}

template <typename T = double, typename... ExtentsType>
    requires(sizeof...(ExtentsType) > 0uz
             && std::conjunction_v<std::is_integral<ExtentsType>...>)
auto
read_file_tensor(std::string const& filename, ExtentsType... extents) {
    return read_file_tensor(
        filename, std::array<size_t, sizeof...(ExtentsType)> { extents... });
}
} // namespace Fermi
