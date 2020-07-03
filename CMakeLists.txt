# Crated by mbenlioglu on May 29, 2020.
# -------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)

project(sarma
        LANGUAGES C CXX
        VERSION 1.0.0
        DESCRIPTION "Symmetric rectilinear partitioning algorithms")

# =============================================================================
# CXX Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# CXX flags -pedantic maybe?
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

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
find_package(TBB REQUIRED)

# header files for the sarma library
file(GLOB_RECURSE header_files RELATIVE_PATH include "*.h" "*.hpp")

add_subdirectory(external)

include_directories(
        PUBLIC
        include
        external/mmio/include
)


link_libraries(
        Threads::Threads
        TBB::tbb
        stdc++fs
        mmio
)

add_executable(sarma src/sarma.cpp)
add_executable(experiment src/experiment.cpp)
add_executable(sarma-bench src/sarma-bench.cpp)

find_package(GUROBI QUIET)
if (GUROBI_FOUND)
    include_directories(${GUROBI_INCLUDE_DIRS})

    target_link_libraries(sarma optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
    target_link_libraries(sarma ${GUROBI_LIBRARY})
    target_compile_definitions(sarma PUBLIC GUROBI_FOUND)

    target_link_libraries(experiment optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
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
add_subdirectory(tools)

# =============================================================================
# Docs
add_subdirectory(docs)
