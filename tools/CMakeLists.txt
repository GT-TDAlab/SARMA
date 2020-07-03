add_executable(bintospart bintospart.cpp)
add_executable(validate_csr validate_csr.cpp)
add_executable(sps_timing sps_timing.cpp)

if (NOT TARGET tools)
    add_custom_target(tools)
    add_dependencies(tools bintospart validate_csr sps_timing)
endif ()
