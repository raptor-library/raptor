include_directories(${raptor_INCDIR})

add_executable(example example.cpp)
target_link_libraries(example raptor ${MPI_LIBRARIES})

add_executable(benchmark_spmv benchmark_spmv.cpp)
target_link_libraries(benchmark_spmv raptor)

if (WITH_HYPRE)
    add_executable(benchmark_rss benchmark_rss.cpp)
    target_link_libraries(benchmark_rss raptor ${MPI_LIBRARIES})

    add_executable(profile_rss profile_rss.cpp)
    target_link_libraries(profile_rss raptor ${MPI_LIBRARIES})
     
    add_executable(benchmark_tap_amg benchmark_tap_amg.cpp)
    target_link_libraries(benchmark_tap_amg raptor ${MPI_LIBRARIES})

    add_executable(benchmark_matops benchmark_matops.cpp)
    target_link_libraries(benchmark_matops raptor ${MPI_LIBRARIES})
endif()
if (WITH_MPI)
    add_executable(benchmark_setup benchmark_setup.cpp)
    target_link_libraries(benchmark_setup raptor ${MPI_LIBRARIES})

    add_executable(benchmark_solve benchmark_solve.cpp)
    target_link_libraries(benchmark_solve raptor ${MPI_LIBRARIES})

    add_executable(benchmark_tap_spmv benchmark_tap_spmv.cpp)
    target_link_libraries(benchmark_tap_spmv raptor ${MPI_LIBRARIES})

    add_executable(benchmark_tap_spgemm benchmark_tap_spgemm.cpp)
    target_link_libraries(benchmark_tap_spgemm raptor ${MPI_LIBRARIES})

    add_executable(profile_tap_spgemm profile_tap_spgemm.cpp)
    target_link_libraries(profile_tap_spgemm raptor ${MPI_LIBRARIES})

    add_executable(benchmark_par_spmv benchmark_par_spmv.cpp)
    target_link_libraries(benchmark_par_spmv raptor ${MPI_LIBRARIES})

    add_executable(benchmark_commpkg benchmark_commpkg.cpp)
    target_link_libraries(benchmark_commpkg raptor ${MPI_LIBRARIES})

    add_executable(benchmark_amg benchmark_amg.cpp)
    target_link_libraries(benchmark_amg raptor ${MPI_LIBRARIES})

    add_executable(benchmark_setup_sweeps benchmark_setup_sweeps.cpp)
    target_link_libraries(benchmark_setup_sweeps raptor ${MPI_LIBRARIES})
endif()


