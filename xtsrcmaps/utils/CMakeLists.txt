include_directories(${CMAKE_SOURCE_DIR})

add_library(utils STATIC bilerp.cxx)

add_library(Fermi::utils ALIAS utils)
target_link_libraries(
    utils
    PUBLIC Eigen3::Eigen
    PRIVATE fmt::fmt
)
