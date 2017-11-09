if (WITH_MPI)
    add_executable(test_par_comm test_par_comm.cpp)
    target_link_libraries(test_par_comm raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParCommTest mpirun -n 2 ./test_par_comm)

    add_executable(test_tap_comm test_tap_comm.cpp)
    target_link_libraries(test_tap_comm raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TAPCommTest  mpirun -n 2 ./test_tap_comm)

    add_executable(test_par_matrix test_par_matrix.cpp)
    target_link_libraries(test_par_matrix raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParMatrixTest mpirun -n 2 ./test_par_matrix)

    add_executable(test_par_vector test_par_vector.cpp)
    target_link_libraries(test_par_vector raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParVectorTest mpirun -n 2 ./test_par_vector)
endif ()

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix raptor googletest pthread )
add_test(MatrixTest ./test_matrix)


