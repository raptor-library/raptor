add_executable(test_amg test_amg.cpp)
target_link_libraries(test_amg raptor)
add_test(AMGTest ./test_amg)

if (WITH_MPI)
    add_executable(test_par_amg test_par_amg.cpp)
    target_link_libraries(test_par_amg raptor ${MPI_LIBRARIES})
    add_test(ParAMGTest mpirun -n 2 ./test_par_amg)
endif()
