set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests-bin)
set(CTEST_BINARY_DIRECTORY ${PROJECT_BINARY_DIR}/tests-bin)

add_custom_target(build-tests)

# Find all files in the unit tests directory
file(GLOB unit_test_files "*.cpp")

foreach (test_src ${unit_test_files})
    get_filename_component(exe ${test_src} NAME_WE)
    add_executable(test_${exe} ${test_src})
    add_dependencies(build-tests test_${exe})

    # Note: library linking, C++ flags, etc. occur globally
    # They are not per-target in this setup
    add_test(NAME unit_${exe} COMMAND
            ${CTEST_BINARY_DIRECTORY}/test_${exe}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(unit_${exe}
            PROPERTIES
            TIMEOUT 100
            SKIP_RETURN_CODE 2
            ENVIRONMENT TEST_MTX_DIR=${CMAKE_SOURCE_DIR}/tests/system/matrices/)
endforeach ()
