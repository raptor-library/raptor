add_executable(test_amg test_amg.cpp)
target_link_libraries(test_amg raptor googletest pthread )
add_test(AMGTest ./test_amg)

if (WITH_MPI)
    add_executable(test_par_amg test_par_amg.cpp)
    target_link_libraries(test_par_amg raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParAMGTest_1 mpirun -n 1 ./test_par_amg)
    add_test(ParAMGTest_2 mpirun -n 2 ./test_par_amg)

    add_executable(test_par_sparsify test_par_sparsify.cpp)
    target_link_libraries(test_par_sparsify raptor ${MPI_LIBRARIES} googletest
        pthread )
    add_test(ParSparsifyTest_1 mpirun -n 1 ./test_par_sparsify)
    add_test(ParSparsifyTest_16 mpirun -n 16 ./test_par_sparsify)
endif()
