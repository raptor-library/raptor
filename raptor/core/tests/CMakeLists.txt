add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix raptor ${MPI_LIBRARIES})
add_test(MatrixTest ./test_matrix)

