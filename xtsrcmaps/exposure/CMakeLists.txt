target_sources(xtsrcmaps PRIVATE exposure.cxx)
if(APPLE)
    target_compile_definitions(xtsrcmaps PRIVATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(xtsrcmaps PRIVATE "-framework Accelerate")
else()
    # do something for openblas or local link
    # target_compile_definitions(xtsrcmaps PRIVATE ACCELERATE_NEW_LAPACK)
    # target_link_options(xtsrcmaps PRIVATE "-framework Accelerate")
endif()
