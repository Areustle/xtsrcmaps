#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"

#include <fstream>
#include <string>

namespace Fermi {

template <typename T = double, typename... ExtentsType>
    requires(sizeof...(ExtentsType) > 0uz
             && std::conjunction_v<std::is_integral<ExtentsType>...>)
auto
read_file_tensor(std::string const& filename, ExtentsType... extents) {

    using VT        = std::conditional_t<std::is_same_v<T, bool>,
                                         std::vector<char>,
                                         std::vector<T>>;

    VT            v = VT((... * extents));
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (!ifs) { throw std::runtime_error("Failed to open file: " + filename); }
    ifs.read(reinterpret_cast<char*>(v.data()), sizeof(T) * (... * extents));
    if (!ifs) {
        throw std::runtime_error(
            "Failed to read the expected amount of data from file: "
            + filename);
    }
    ifs.close();

    Tensor<T, sizeof...(ExtentsType)> new_tensor(v, extents...);

    return new_tensor;
}
} // namespace Fermi
