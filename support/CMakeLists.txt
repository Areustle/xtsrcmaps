add_executable(reorder_tensor #
               reorder_tensor.cpp)
target_link_libraries(reorder_tensor PRIVATE xtsrcmaps)
target_include_directories(reorder_tensor PRIVATE ${CMAKE_SOURCE_DIR})
