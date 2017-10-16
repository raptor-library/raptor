include_directories(${raptor_INCDIR})

if (WITH_HYPRE)
    add_executable(benchmark_rss benchmark_rss.cpp)
    target_link_libraries(benchmark_rss raptor ${MPI_LIBRARIES})
endif()

#add_executable(benchmark_matmult benchmark_matmult.cpp)
#target_link_libraries(benchmark_matmult raptor ${MPI_LIBRARIES})

#add_executable(benchmark_tap_matmult benchmark_tap_matmult.cpp)
#target_link_libraries(benchmark_tap_matmult raptor ${MPI_LIBRARIES})

#add_executable(benchmark_exxon_comm benchmark_exxon_comm.cpp)
#target_link_libraries(benchmark_exxon_comm raptor ${MPI_LIBRARIES})

