add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg raptor)
add_test(TestCG ./test_cg)

add_executable(test_bicgstab test_bicgstab.cpp)
target_link_libraries(test_bicgstab raptor)
add_test(TestBiCGStab ./test_bicgstab)

if (WITH_MPI)
    add_executable(test_par_cg test_par_cg.cpp)
    target_link_libraries(test_par_cg raptor ${MPI_LIBRARIES})
    add_test(TestParCG ${MPIRUN} -n 1 ./test_par_cg)
    
    add_executable(test_par_bicgstab test_par_bicgstab.cpp)
    target_link_libraries(test_par_bicgstab raptor ${MPI_LIBRARIES})
    add_test(TestParBiCGStab ${MPIRUN} -n 1 ./test_par_bicgstab)
    
    add_executable(test_par_prebicgstab test_par_prebicgstab.cpp)
    target_link_libraries(test_par_prebicgstab raptor ${MPI_LIBRARIES})
    add_test(TestParPreBiCGStab ./test_par_prebicgstab)

    add_executable(test_bicgstab_reproducibility test_bicgstab_reproducibility.cpp)
    target_link_libraries(test_bicgstab_reproducibility raptor ${MPI_LIBRARIES})
    add_test(TestBiCGStabReprod ${MPIRUN} -n 1 ./test_bicgstab_reproducibility)

    add_executable(test_partial_inner test_partial_inner.cpp)
    target_link_libraries(test_partial_inner raptor ${MPI_LIBRARIES})
    add_test(TestPartialInner ${MPIRUN} -n 1 ./test_partial_inner)
    
endif()


