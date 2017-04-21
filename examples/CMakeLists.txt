include_directories(${raptor_INCDIR})

add_executable(benchmark_matrix benchmark_matrix.cpp)
target_link_libraries(benchmark_matrix raptor ${MPI_LIBRARIES})

add_executable(benchmark_tap_spmv benchmark_tap_spmv.cpp)
target_link_libraries(benchmark_tap_spmv raptor ${MPI_LIBRARIES})

add_executable(benchmark_matmult benchmark_matmult.cpp)
target_link_libraries(benchmark_matmult raptor ${MPI_LIBRARIES})

add_executable(benchmark_tap_matmult benchmark_tap_matmult.cpp)
target_link_libraries(benchmark_tap_matmult raptor ${MPI_LIBRARIES})
