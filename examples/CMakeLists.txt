include_directories(${raptor_INCDIR})
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark raptor ${MPI_LIRARIES})

if(WITH_HYPRE)
    add_executable(benchmark_hypre benchmark_hypre.cpp)
    target_link_libraries(benchmark_hypre raptor ${MPI_LIRARIES})
endif(WITH_HYPRE)

if(WITH_MFEM)
    add_executable(benchmark_mfem benchmark_mfem.cpp)
    target_link_libraries(benchmark_mfem raptor ${MPI_LIRARIES})
endif(WITH_MFEM)

    add_executable(benchmark_charm benchmark_charm.cpp)
    target_link_libraries(benchmark_charm raptor ${MPI_LIRARIES})



