include_directories(${raptor_INCDIR})

if (WITH_HYPRE)
    add_executable(benchmark_rss benchmark_rss.cpp)
    target_link_libraries(benchmark_rss raptor ${MPI_LIBRARIES})
endif()


