#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"

#include <fstream>
#include <string>

namespace Fermi {

template <typename T = double, size_t R>
    requires(R > 0uz)
auto
write_file_tensor(std::string const&         filename,
                  Fermi::Tensor<T, R> const& tensor) {

    using VT = std::conditional_t<std::is_same_v<T, bool>,
                                  std::vector<char>,
                                  std::vector<T>>;

    // Flatten the tensor data into a vector for writing
    VT v(tensor.begin(), tensor.end());

    std::ofstream ofs(filename,
                      std::ios::out | std::ios::binary | std::ios::trunc);

    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing: "
                                 + filename);
    }

    ofs.write(reinterpret_cast<const char*>(v.data()), sizeof(T) * v.size());

    if (!ofs) {
        std::string error_message
            = "Error: Failed to write the expected amount of data to file: "
              + filename + ". Expected to write " + std::to_string(v.size())
              + " elements of size " + std::to_string(sizeof(T))
              + " bytes each, but encountered an error.";
        ofs.close();
        throw std::runtime_error(error_message);
    }

    ofs.close();
}

template <typename T = double, typename... ExtentsType>
    requires(sizeof...(ExtentsType) > 0uz
             && std::conjunction_v<std::is_integral<ExtentsType>...>)
auto
write_file_tensor(std::string const&                              filename,
                  Fermi::Tensor<T, sizeof...(ExtentsType)> const& tensor) {
    write_file_tensor<T, sizeof...(ExtentsType)>(filename, tensor);
}

} // namespace Fermi
