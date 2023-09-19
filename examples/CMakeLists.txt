include_directories(${raptor_INCDIR})

add_executable(benchmark_spmv benchmark_spmv.cpp)
target_link_libraries(benchmark_spmv raptor)

if (WITH_HYPRE)
    add_executable(benchmark_rss benchmark_rss.cpp)
    target_link_libraries(benchmark_rss raptor ${MPI_LIBRARIES})

    add_executable(benchmark_tap_amg benchmark_tap_amg.cpp)
    target_link_libraries(benchmark_tap_amg raptor ${MPI_LIBRARIES})

    add_executable(benchmark_matops benchmark_matops.cpp)
    target_link_libraries(benchmark_matops raptor ${MPI_LIBRARIES})

    add_executable(benchmark_par_spmv benchmark_par_spmv.cpp)
    target_link_libraries(benchmark_par_spmv raptor ${MPI_LIBRARIES})

endif()
if (WITH_MPI)
    add_executable(example example.cpp)
    target_link_libraries(example raptor ${MPI_LIBRARIES})

    add_executable(matop_example matop_example.cpp)
    target_link_libraries(matop_example raptor ${MPI_LIBRARIES})

    add_executable(benchmark_setup benchmark_setup.cpp)
    target_link_libraries(benchmark_setup raptor ${MPI_LIBRARIES})

    add_executable(coo_example coo_example.cpp)
    target_link_libraries(coo_example raptor ${MPI_LIBRARIES})

    add_executable(csr_example csr_example.cpp)
    target_link_libraries(csr_example raptor ${MPI_LIBRARIES})

    add_executable(benchmark_solve benchmark_solve.cpp)
    target_link_libraries(benchmark_solve raptor ${MPI_LIBRARIES})

    add_executable(benchmark_reader benchmark_reader.cpp)
    target_link_libraries(benchmark_reader raptor ${MPI_LIBRARIES})

    add_executable(benchmark_tap_spmv benchmark_tap_spmv.cpp)
    target_link_libraries(benchmark_tap_spmv raptor ${MPI_LIBRARIES})

    add_executable(benchmark_tap_spgemm benchmark_tap_spgemm.cpp)
    target_link_libraries(benchmark_tap_spgemm raptor ${MPI_LIBRARIES})

    add_executable(profile_tap_spgemm profile_tap_spgemm.cpp)
    target_link_libraries(profile_tap_spgemm raptor ${MPI_LIBRARIES})

    add_executable(benchmark_amg benchmark_amg.cpp)
    target_link_libraries(benchmark_amg raptor ${MPI_LIBRARIES})

    add_executable(benchmark_setup_sweeps benchmark_setup_sweeps.cpp)
    target_link_libraries(benchmark_setup_sweeps raptor ${MPI_LIBRARIES})

    add_executable(profile_tap_rss profile_tap_rss.cpp)
    target_link_libraries(profile_tap_rss raptor ${MPI_LIBRARIES})

    add_executable(profile_tap_sas profile_tap_sas.cpp)
    target_link_libraries(profile_tap_sas raptor ${MPI_LIBRARIES})

    add_executable(profile_tap_amg profile_tap_amg.cpp)
    target_link_libraries(profile_tap_amg raptor ${MPI_LIBRARIES})

    add_executable(profile_tap_steps profile_tap_steps.cpp)
    target_link_libraries(profile_tap_steps raptor ${MPI_LIBRARIES})

    add_executable(model_tap_steps model_tap_steps.cpp)
    target_link_libraries(model_tap_steps raptor ${MPI_LIBRARIES})

    add_executable(benchmark_amg_spgemm benchmark_amg_spgemm.cpp)
    target_link_libraries(benchmark_amg_spgemm raptor ${MPI_LIBRARIES})
endif()


