include_directories(${raptor_INCDIR})

#    add_executable(hypre_matmult hypre_matmult.cpp)
#    target_link_libraries(hypre_matmult raptor ${MPI_LIRARIES})

#    add_executable(benchmark_matmult benchmark_matmult.cpp)
#    target_link_libraries(benchmark_matmult raptor ${MPI_LIRARIES})

#    add_executable(benchmark_probe benchmark_probe.cpp)
#    target_link_libraries(benchmark_probe ${MPI_LIRARIES})
   
#    add_executable(benchmark_topo_spmv benchmark_topo_spmv.cpp)
#    target_link_libraries(benchmark_topo_spmv raptor ${MPI_LIRARIES})

#    add_executable(benchmark_mfem benchmark_mfem.cpp)
#    target_link_libraries(benchmark_mfem raptor ${MPI_LIBRARIES})

    add_executable(benchmark_matrix benchmark_matrix.cpp)
    target_link_libraries(benchmark_matrix raptor ${MPI_LIBRARIES})

