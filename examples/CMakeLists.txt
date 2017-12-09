include_directories(${raptor_INCDIR})

add_executable(example example.cpp)
target_link_libraries(example raptor ${MPI_LIBRARIES})

if (WITH_HYPRE)
    add_executable(benchmark_rss benchmark_rss.cpp)
    target_link_libraries(benchmark_rss raptor ${MPI_LIBRARIES})

    add_executable(benchmark_nek5000 benchmark_nek5000.cpp)
    target_link_libraries(benchmark_nek5000 raptor ${MPI_LIBRARIES})
endif()
if (WITH_MPI)
    add_executable(benchmark_spmv benchmark_spmv.cpp)
    target_link_libraries(benchmark_spmv raptor ${MPI_LIBRARIES})

    add_executable(benchmark_matmult benchmark_matmult.cpp)
    target_link_libraries(benchmark_matmult raptor ${MPI_LIBRARIES})
endif()


