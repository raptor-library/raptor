add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg raptor ${MPI_LIBRARIES} googletest pthread)
add_test(TestCG ./test_cg)

add_executable(test_bicgstab test_bicgstab.cpp)
target_link_libraries(test_bicgstab raptor ${MPI_LIBRARIES} googletest pthread)
add_test(TestBiCGStab ./test_bicgstab)

if (WITH_MPI)
    add_executable(test_par_cg test_par_cg.cpp)
    target_link_libraries(test_par_cg raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParCG ${MPIRUN} -n 1 ${HOST} ./test_par_cg)

    if (WITH_HIP AND WITH_MFEM)
        add_executable(test_par_hip_cg test_par_hip_cg.cpp)
        target_link_libraries(test_par_hip_cg raptor ${MPI_LIBRARIES} googletest pthread)
        add_test(TestParHipCG ${MPIRUN} -n 1 ${HOST} ./test_par_hip_cg)
        add_test(TestParHipCGMPI ${MPIRUN} -n ${TEST_COMM_SIZE} ${HOST} ./test_par_hip_cg)
        set_tests_properties(TestParHipCG TestParHipCGMPI PROPERTIES
            SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
    endif()
        
    add_executable(test_par_bicgstab test_par_bicgstab.cpp)
    target_link_libraries(test_par_bicgstab raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParBiCGStab ${MPIRUN} -n 1 ${HOST} ./test_par_bicgstab)

endif()


