#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/reorder_tensor.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

TEST_CASE("Tensor writable operator[]") {
    using Tensor3D = Fermi::Tensor<float, 3>;

    // Create a 3x3x3 tensor
    Tensor3D tensor({ 3, 3, 3 });

    // Write to the tensor using operator[]
    tensor[0, 0, 0] = 1.0f;
    tensor[1, 1, 1] = 2.0f;
    tensor[2, 2, 2] = 3.0f;

    // Read from the tensor and check values
    CHECK(tensor[0, 0, 0] == 1.0f);
    CHECK(tensor[1, 1, 1] == 2.0f);
    CHECK(tensor[2, 2, 2] == 3.0f);

    // Modify existing values
    tensor[0, 0, 0] += 1.0f;
    tensor[1, 1, 1] *= 2.0f;
    tensor[2, 2, 2] -= 1.0f;

    // Read the modified values and check
    CHECK(tensor[0, 0, 0] == 2.0f);
    CHECK(tensor[1, 1, 1] == 4.0f);
    CHECK(tensor[2, 2, 2] == 2.0f);

    // Test boundary values
    tensor[2, 2, 1] = 5.0f;
    CHECK(tensor[2, 2, 1] == 5.0f);
}

TEST_CASE("Tensor address of elements using operator&") {
    using Tensor3D = Fermi::Tensor<float, 3>;

    // Create a 3x3x3 tensor
    Tensor3D tensor({ 3, 3, 3 });

    // Write to the tensor using operator[]
    tensor[1, 1, 1] = 42.0f;

    // Get the address of an element using operator&
    float* p        = &tensor[1, 1, 1];

    // Check if the address points to the correct value
    CHECK(*p == 42.0f);

    // Modify the value through the pointer
    *p = 100.0f;

    // Check if the modification is reflected in the tensor
    CHECK(tensor[1, 1, 1] == 100.0f);
}
