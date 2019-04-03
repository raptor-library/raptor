include_directories(${raptor_INCDIR})

add_executable(example example.cpp)
target_link_libraries(example raptor ${MPI_LIBRARIES})

add_executable(benchmark_spmv benchmark_spmv.cpp)
target_link_libraries(benchmark_spmv raptor)

if (WITH_HYPRE)
    add_executable(benchmark_tap_amg benchmark_tap_amg.cpp)
    target_link_libraries(benchmark_tap_amg raptor ${MPI_LIBRARIES})
endif()
if (WITH_MPI)
    add_executable(calc_model_params calc_model_params.cpp)
    target_link_libraries(calc_model_params raptor ${MPI_LIBRARIES})
endif()


