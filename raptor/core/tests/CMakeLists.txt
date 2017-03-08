add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix raptor)
add_test(MatrixTest ./test_matrix)

