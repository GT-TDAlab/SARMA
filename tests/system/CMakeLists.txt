# Test basic help and syntax
add_test (NAME sys_helpstring COMMAND
    bash -x -c "
        # If the help string is changed, update this
        \"${PROJECT_BINARY_DIR}/sarma\" -h 2>&1 | shasum | grep 955db9cb4f6f1e6d13c5efe69b748a88598e1188
    " --
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Gather mtx matrix files
file (GLOB matrices "matrices/*.mtx")

# Test all parameters options for each algorithm, on each matrix, combinatorially
foreach (P 32 8)
    foreach (Q 32 8)
        foreach (sparse 1.0 0.1)
            foreach (order nat)
                foreach (alg nic sgo pal opal rac) # in decreasing order of speed
                    foreach (seed 2711481)
                        # Symmetric partition algs skip when P != Q
                        if (NOT (P EQUAL Q))
                            if ((alg EQUAL pal) OR (alg EQUAL opal) OR (alg EQUAL rac) OR (alg EQUAL uni))
                                continue()
                            endif()
                        endif()

                        # Run the given parameters against each matrix
                        foreach (matrix ${matrices})
                            string (REGEX REPLACE "(^.*/|\\.mtx$)" "" test ${matrix})
                            set(test_name "sys_${test}-${alg}-${seed}-${sparse}-${order}-${P}-${Q}")
                            add_test (NAME ${test_name} COMMAND
                                bash -x -c "
                                    timeout -k 85s 85s \"${PROJECT_BINARY_DIR}/sarma\" --graph \"${matrix}\" --alg ${alg} --order ${order} --p ${P} --q ${Q} 1>outputs/${test_name}.out 2>outputs/${test_name}.err
                                    ret=$?
                                    pushd outputs
                                    python3 ../parse_res.py ${test_name}.out ${test_name}
                                    grep ${test_name} valid >${test_name}.sha
                                    #depending on whether ctest is run with the environment variable
                                    #SARMA_BUILD_VALID defined, the pre/post below will be either:
                                    if [[ $SARMA_BUILD_VALID ]]; then
                                        if [ -s ${test_name} ]; then
                                            # which will always return 0 and save the output to the file valid
                                            until mkdir valid.lock; do sleep 1; done;
                                            sed '/${test_name}/d' valid > valid_rem;
                                            mv valid_rem valid;
                                            shasum ${test_name} >> valid;
                                            rmdir valid.lock
                                            sharet=0
                                        else
                                            sharet=1
                                        fi
                                    else
                                        # which will return 0 for an error code on the file ${test_name} matching the sum
                                        shasum -c ${test_name}.sha
                                        sharet=$?
                                    fi
                                    ((ret|=sharet))
                                    SARMA_SYS_TEST_CLEANUP=\${SARMA_SYS_TEST_CLEANUP:-1}
                                    if [ $SARMA_SYS_TEST_CLEANUP -ne 0 ]; then
                                        rm -v ${test_name}*
                                    fi
                                    popd
                                    exit $ret
                                " --
                                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
                            set_tests_properties(${test_name} PROPERTIES TIMEOUT 90)
                        endforeach()
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endforeach()
