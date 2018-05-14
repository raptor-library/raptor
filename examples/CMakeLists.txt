include_directories(${raptor_INCDIR})

add_executable(example example.cpp)
target_link_libraries(example raptor ${MPI_LIBRARIES})

if (WITH_HYPRE)
    add_executable(benchmark_rss benchmark_rss.cpp)
    target_link_libraries(benchmark_rss raptor ${MPI_LIBRARIES})
endif()
if (WITH_MPI)
    add_executable(benchmark_setup benchmark_setup.cpp)
    target_link_libraries(benchmark_setup raptor ${MPI_LIBRARIES})

    add_executable(benchmark_solve benchmark_solve.cpp)
    target_link_libraries(benchmark_solve raptor ${MPI_LIBRARIES})

endif()


