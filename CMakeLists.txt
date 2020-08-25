# Crated by mbenlioglu on May 29, 2020.
# -------------------------------------------------------------
cmake_minimum_required(VERSION 3.13)
project(sarma
        LANGUAGES C CXX
        VERSION 1.1.0
        DESCRIPTION "Symmetric rectilinear partitioning algorithms")
include(GNUInstallDirs)

# =============================================================================
# CXX Standard

# Require GCC>=7.0.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        message(FATAL_ERROR "GCC version >= 7 is required")
    elseif (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        message(STATUS "C++ standard 14 is set. Execution will be sequential.")
        set(CMAKE_CXX_STANDARD 14)
    else ()
        message(STATUS "C++ standard 17 is set. Execution can be parallel.")
        set(CMAKE_CXX_STANDARD 17)
    endif ()
else ()
    message(FATAL_ERROR "GCC version >= 7 is required")
endif ()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# CXX flags -pedantic maybe?
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES x86_64)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif ()

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES ppc64le)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=power9")
endif ()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif ()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Other Flags
set(THREADS_PREFER_PTHREADS_FLAG ON)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/.local" ${CMAKE_PREFIX_PATH})
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# =============================================================================
# Link Libraries
find_package(Threads REQUIRED)
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    find_package(TBB REQUIRED)
endif ()

# header files for the sarma library
file(GLOB_RECURSE header_files RELATIVE_PATH include "*.h" "*.hpp")

add_subdirectory(external)

# Define sarma interface (header-only) library
add_library(libsarma INTERFACE)
target_include_directories(libsarma INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/mmio/include>
        $<INSTALL_INTERFACE:include/sarma>
        )

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    target_link_libraries(libsarma INTERFACE
            Threads::Threads
            TBB::tbb
            stdc++fs
            mmio
            )
else ()
    target_link_libraries(libsarma INTERFACE
            Threads::Threads
            stdc++fs
            mmio
            )
endif ()

link_libraries(
        libsarma
)

add_executable(sarma src/sarma.cpp)
add_executable(experiment src/experiment.cpp)
add_executable(sarma-bench src/sarma-bench.cpp)

find_package(GUROBI QUIET)
if (GUROBI_FOUND)
    include_directories(${GUROBI_INCLUDE_DIRS})

    target_link_libraries(sarma ${GUROBI_CXX_LIBRARY})
    target_link_libraries(sarma ${GUROBI_LIBRARY})
    target_compile_definitions(sarma PUBLIC GUROBI_FOUND)

    target_link_libraries(experiment ${GUROBI_CXX_LIBRARY})
    target_link_libraries(experiment ${GUROBI_LIBRARY})
    target_compile_definitions(experiment PUBLIC GUROBI_FOUND)
else ()
    message(WARNING "Gurobi is needed for the mip algorithm, excluding mip!")
endif ()

add_custom_target(compare
        COMMAND git stash push -m compare-w-\${HASH}
        COMMAND git checkout \${HASH}
        COMMAND ${CMAKE_COMMAND} --build . --target sarma-bench ${CMAKE_CURRENT_SOURCE_DIR} -- -j 4
        COMMAND mv sarma-bench sarma-bench-\${HASH}
        COMMAND git checkout -
        COMMAND ${CMAKE_COMMAND} --build . --target sarma-bench ${CMAKE_CURRENT_SOURCE_DIR} -- -j 4
        COMMAND git stash list | grep "compare-w-\${HASH}" &&
        git stash pop stash@{\$$\(git stash list | grep "compare-w-\${HASH}" | awk -F'[{}]' '{print \$$2}'\)}
        || exit 0
        COMMAND mkdir -p compare
        COMMAND cat \${CONFIG} | ./sarma-bench-\${HASH} | ./sarma-bench -o compare
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# =============================================================================
# Python bindings
add_subdirectory(python)

# =============================================================================
# Test support
enable_testing()
add_subdirectory(tests)

# =============================================================================
# Tools
# add_subdirectory(tools)

# =============================================================================
# Docs
add_subdirectory(docs)


# =============================================================================
# Install
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/SARMAConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

install(TARGETS libsarma sarma sarma-bench
        EXPORT SARMATargets
        LIBRARY DESTINATION lib COMPONENT Runtime
        ARCHIVE DESTINATION lib COMPONENT Development
        RUNTIME DESTINATION bin COMPONENT Runtime
        PUBLIC_HEADER DESTINATION include COMPONENT Development
        BUNDLE DESTINATION bin COMPONENT Runtime
        )
install(TARGETS mmio
        EXPORT SARMATargets
        LIBRARY DESTINATION lib COMPONENT Runtime
        ARCHIVE DESTINATION lib COMPONENT Development
        RUNTIME DESTINATION bin COMPONENT Runtime
        PUBLIC_HEADER DESTINATION include/sarma COMPONENT Development
        BUNDLE DESTINATION bin COMPONENT Runtime
        )

configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/SARMAConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/SARMAConfig.cmake"
        INSTALL_DESTINATION lib/cmake/SARMA
)

# Define namespace for link_libraries(SARMA::libsarma) to work
install(EXPORT SARMATargets NAMESPACE SARMA:: DESTINATION lib/cmake/SARMA)
install(FILES "${PROJECT_BINARY_DIR}/SARMAConfigVersion.cmake"
        "${PROJECT_BINARY_DIR}/SARMAConfig.cmake"
        DESTINATION lib/cmake/SARMA)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include/sarma)

# Uninstall
if (NOT TARGET uninstall)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif ()