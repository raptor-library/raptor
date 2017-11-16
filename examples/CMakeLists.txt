include_directories(${raptor_INCDIR})

if (WITH_HYPRE)
    add_executable(benchmark_rss benchmark_rss.cpp)
    target_link_libraries(benchmark_rss raptor ${MPI_LIBRARIES})
endif()
if (WITH_MPI)
    add_executable(benchmark_spmv benchmark_spmv.cpp)
    target_link_libraries(benchmark_spmv raptor ${MPI_LIBRARIES})

    add_executable(benchmark_matmult benchmark_matmult.cpp)
    target_link_libraries(benchmark_matmult raptor ${MPI_LIBRARIES})
endif()


