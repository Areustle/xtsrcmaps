#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include "xtsrcmaps/tensor/reorder_tensor.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

/* // 2D Non-Square Tensor Test */
/* TEST_CASE("Reorder Tensor: 2D Non-Square Tensor") { */
/**/
/*     using Tensor2D = Fermi::Tensor<int, 2>; */
/**/
/*     Tensor2D tensor({ 2, 3 }); */
/*     // 1 2 3 */
/*     // 4 5 6 */
/*     tensor[0, 0]                              = 1; */
/*     tensor[0, 1]                              = 2; */
/*     tensor[0, 2]                              = 3; */
/*     tensor[1, 0]                              = 4; */
/*     tensor[1, 1]                              = 5; */
/*     tensor[1, 2]                              = 6; */
/**/
/*     // Test 1: No reordering (identity) */
/*     std::array<std::size_t, 2> identity_order = { 0, 1 }; */
/*     auto tensor_identity = Fermi::reorder_tensor(tensor, identity_order); */
/*     CHECK(tensor_identity.extent(0) == 2); */
/*     CHECK(tensor_identity.extent(1) == 3); */
/*     CHECK(tensor_identity[0, 0] == 1); */
/*     CHECK(tensor_identity[0, 1] == 2); */
/*     CHECK(tensor_identity[0, 2] == 3); */
/*     CHECK(tensor_identity[1, 0] == 4); */
/*     CHECK(tensor_identity[1, 1] == 5); */
/*     CHECK(tensor_identity[1, 2] == 6); */
/**/
/*     // Test 2: Reversing the dimensions */
/*     std::array<std::size_t, 2> reverse_order = { 1, 0 }; */
/*     auto tensor_reversed = Fermi::reorder_tensor(tensor, reverse_order); */
/*     CHECK(tensor_reversed.extent(0) == 3); */
/*     CHECK(tensor_reversed.extent(1) == 2); */
/*     CHECK(tensor_reversed[0, 0] == 1); */
/*     CHECK(tensor_reversed[0, 1] == 4); */
/*     CHECK(tensor_reversed[1, 0] == 2); */
/*     CHECK(tensor_reversed[1, 1] == 5); */
/*     CHECK(tensor_reversed[2, 0] == 3); */
/*     CHECK(tensor_reversed[2, 1] == 6); */
/* } */
/**/
/* // Rank 3 Tensor Test */
/* TEST_CASE("Reorder Tensor: 3D Tensor [3,4,5]") { */
/*     using Tensor3D = Fermi::Tensor<int, 3>; */
/**/
/*     // Create a 3x4x5 tensor and fill with sequential values */
/*     Tensor3D tensor({ 3, 4, 5 }); */
/*     int      value = 1; */
/*     for (std::size_t i = 0; i < 3; ++i) { */
/*         for (std::size_t j = 0; j < 4; ++j) { */
/*             for (std::size_t k = 0; k < 5; ++k) { tensor[i, j, k] = value++; } */
/*         } */
/*     } */
/**/
/*     // Test 1: No reordering (identity) */
/*     std::array<std::size_t, 3> identity_order = { 0, 1, 2 }; */
/*     auto tensor_identity = Fermi::reorder_tensor(tensor, identity_order); */
/*     CHECK(tensor_identity.extent(0) == 3); */
/*     CHECK(tensor_identity.extent(1) == 4); */
/*     CHECK(tensor_identity.extent(2) == 5); */
/*     value = 1; */
/*     for (std::size_t i = 0; i < 3; ++i) { */
/*         for (std::size_t j = 0; j < 4; ++j) { */
/*             for (std::size_t k = 0; k < 5; ++k) { */
/*                 CHECK(tensor_identity[i, j, k] == value++); */
/*             } */
/*         } */
/*     } */
/**/
/*     // Test 2: Reorder dimensions */
/*     std::array<std::size_t, 3> reorder = { 1, 2, 0 }; */
/*     auto tensor_reordered              = Fermi::reorder_tensor(tensor, reorder); */
/*     CHECK(tensor_reordered.extent(0) == 4); */
/*     CHECK(tensor_reordered.extent(1) == 5); */
/*     CHECK(tensor_reordered.extent(2) == 3); */
/**/
/*     // Check some values to ensure reordering is correct */
/*     CHECK(tensor_reordered[0, 0, 0] == 1); */
/*     CHECK(tensor_reordered[0, 1, 0] == 2); */
/*     CHECK(tensor_reordered[0, 2, 0] == 3); */
/*     CHECK(tensor_reordered[0, 0, 1] == 21); */
/*     CHECK(tensor_reordered[0, 0, 2] == 41); */
/* } */
/**/
/* // Rank 4 Tensor Test */
/* TEST_CASE("Reorder Tensor: 4D Tensor [3,5,7,9]") { */
/*     using Tensor4D = Fermi::Tensor<int, 4>; */
/**/
/*     // Create a 3x5x7x9 tensor and fill with sequential values */
/*     Tensor4D tensor({ 3, 5, 7, 9 }); */
/*     int      value = 1; */
/*     for (std::size_t i = 0; i < 3; ++i) { */
/*         for (std::size_t j = 0; j < 5; ++j) { */
/*             for (std::size_t k = 0; k < 7; ++k) { */
/*                 for (std::size_t l = 0; l < 9; ++l) { */
/*                     tensor[i, j, k, l] = value++; */
/*                 } */
/*             } */
/*         } */
/*     } */
/**/
/*     // Test 1: No reordering (identity) */
/*     std::array<std::size_t, 4> identity_order = { 0, 1, 2, 3 }; */
/*     auto tensor_identity = Fermi::reorder_tensor(tensor, identity_order); */
/*     CHECK(tensor_identity.extent(0) == 3); */
/*     CHECK(tensor_identity.extent(1) == 5); */
/*     CHECK(tensor_identity.extent(2) == 7); */
/*     CHECK(tensor_identity.extent(3) == 9); */
/*     value = 1; */
/*     for (std::size_t i = 0; i < 3; ++i) { */
/*         for (std::size_t j = 0; j < 5; ++j) { */
/*             for (std::size_t k = 0; k < 7; ++k) { */
/*                 for (std::size_t l = 0; l < 9; ++l) { */
/*                     CHECK(tensor_identity[i, j, k, l] == value++); */
/*                 } */
/*             } */
/*         } */
/*     } */
/**/
/*     // Test 2: Reorder dimensions */
/*     std::array<std::size_t, 4> reorder = { 3, 2, 1, 0 }; */
/*     auto tensor_reordered              = Fermi::reorder_tensor(tensor, reorder); */
/*     CHECK(tensor_reordered.extent(0) == 9); */
/*     CHECK(tensor_reordered.extent(1) == 7); */
/*     CHECK(tensor_reordered.extent(2) == 5); */
/*     CHECK(tensor_reordered.extent(3) == 3); */
/**/
/*     // Check some values to ensure reordering is correct */
/*     CHECK(tensor_reordered[0, 0, 0, 0] == 1); */
/*     CHECK(tensor_reordered[0, 0, 0, 1] == 316); */
/*     CHECK(tensor_reordered[0, 0, 1, 0] == 64); */
/*     CHECK(tensor_reordered[0, 1, 0, 0] == 10); */
/*     CHECK(tensor_reordered[8, 6, 4, 2] == 945); */
/* } */
