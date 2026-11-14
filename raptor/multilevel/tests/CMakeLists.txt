add_executable(test_amg test_amg.cpp)
target_link_libraries(test_amg raptor ${MPI_LIBRARIES} googletest pthread )
add_test(AMGTest ./test_amg)

if (WITH_MPI)
    add_executable(test_par_amg test_par_amg.cpp)
    target_link_libraries(test_par_amg raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParAMGTest_n1 ${MPIRUN} -n 1 ${HOST} ./test_par_amg)
    add_test(ParAMGTest_n2 ${MPIRUN} -n 2 ${HOST} ./test_par_amg)

endif()
