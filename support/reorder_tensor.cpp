#include "xtsrcmaps/tensor/reorder_tensor.hpp" // Include your reorder_tensor function here
#include "xtsrcmaps/tensor/read_file_tensor.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

// Function to split a string by a delimiter and convert to size_t
std::array<std::size_t, 3>
split_string_to_size_t(const std::string& str, char delimiter) {
    std::array<std::size_t, 3> result;
    std::stringstream          ss(str);
    std::string                token;
    auto                       it = result.begin();
    while (std::getline(ss, token, delimiter)) {
        *it = (static_cast<std::size_t>(std::stoul(token)));
        ++it;
    }
    return result;
}

int
main(int argc, char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0]
                  << " -i input_file -o output_file -e extents -r reorder\n"
                  << "  -i input_file   : Path to the input binary file with "
                     "C++ doubles data.\n"
                  << "  -o output_file  : Path to the output binary file.\n"
                  << "  -e extents      : Comma-separated list of extents, "
                     "e.g., '4,3,2' for a 3D tensor.\n"
                  << "  -r reorder      : Comma-separated list of reordered "
                     "extents, e.g., '2,0,1' to swap dimensions.\n";
        return 1;
    }

    std::string                input_file, output_file;
    std::array<std::size_t, 3> extents, reorder;

    // Parse command line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string option = argv[i];
        std::string value  = argv[i + 1];
        if (option == "-i") {
            input_file = value;
        } else if (option == "-o") {
            output_file = value;
        } else if (option == "-e") {
            extents = split_string_to_size_t(value, ',');
        } else if (option == "-r") {
            reorder = split_string_to_size_t(value, ',');
        } else {
            std::cerr << "Unknown option: " << option << "\n";
            return 1;
        }
    }

    auto const input          = Fermi::read_file_tensor(input_file, extents);
    auto       reorder_tensor = Fermi::reorder_tensor(input, reorder);

    try {
        // Open output file and write reordered data
        std::ofstream output(output_file, std::ios::binary);
        if (!output) {
            std::cerr << "Error: Unable to open output file: " << output_file
                      << "\n";
            return 1;
        }
        output.write(reinterpret_cast<char*>(reorder_tensor.data()),
                     reorder_tensor.size() * sizeof(double));
        output.close();

        std::cout << "Reordered tensor successfully written to: " << output_file
                  << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
