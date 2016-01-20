include_directories(${raptor_INCDIR})
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark raptor ${MPI_LIBRARIES} ${MFEM_LIBRARIES} 
    ${METIS_LIBRARIES} ${HYPRE_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})


if(WITH_HYPRE)
    add_executable(benchmark_hypre benchmark_hypre.cpp)
    target_link_libraries(benchmark_hypre raptor ${MPI_LIBRARIES} ${MFEM_LIBRARIES} 
    ${METIS_LIBRARIES} ${HYPRE_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif(WITH_HYPRE)

if(WITH_MFEM)
	add_executable(benchmark_mfem benchmark_mfem.cpp)
    target_link_libraries(benchmark_mfem raptor ${MPI_LIBRARIES} ${MFEM_LIBRARIES} 
    ${METIS_LIBRARIES} ${HYPRE_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif(WITH_MFEM)

