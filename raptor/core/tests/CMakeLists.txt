if (WITH_MPI)
    add_executable(test_par_comm test_par_comm.cpp)
    target_link_libraries(test_par_comm raptor)
    add_test(ParCommTest ./test_par_comm)

    add_executable(test_tap_comm test_tap_comm.cpp)
    target_link_libraries(test_tap_comm raptor)
    add_test(TAPCommTest ./test_tap_comm)

    add_executable(test_par_matrix test_par_matrix.cpp)
    target_link_libraries(test_par_matrix raptor)
    add_test(ParMatrixTest ./test_par_matrix)

    add_executable(test_par_vector test_par_vector.cpp)
    target_link_libraries(test_par_vector raptor)
    add_test(ParVectorTest ./test_par_vector)
endif ()

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix raptor)
add_test(MatrixTest ./test_matrix)


