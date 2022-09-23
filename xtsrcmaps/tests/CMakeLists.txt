# Make test executable
#
add_executable(tests
    test_fitsfuncs.cxx
    test_main.cpp)
target_link_libraries(
  tests PRIVATE fmt::fmt std::mdspan PkgConfig::cfitsio doctest::doctest
                fitsfuncs
)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME tests COMMAND tests)
