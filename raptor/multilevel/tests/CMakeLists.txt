add_executable(test_amg test_amg.cpp)
target_link_libraries(test_amg raptor ${MPI_LIBRARIES} googletest pthread )
add_test(AMGTest ./test_amg)

if (WITH_MPI)
    add_executable(test_par_amg test_par_amg.cpp)
    target_link_libraries(test_par_amg raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParAMGTest ${MPIRUN} -n 1 ${HOST} ./test_par_amg)
    add_test(ParAMGTest ${MPIRUN} -n 2 ${HOST} ./test_par_amg)
    
    add_executable(test_par_block_amg test_par_block_amg.cpp)
    target_link_libraries(test_par_block_amg raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParBlockAMGTest ${MPIRUN} -n 1 ${HOST} ./test_par_block_amg)
    add_test(ParBlockAMGTest ${MPIRUN} -n 2 ${HOST} ./test_par_block_amg)
    add_test(ParBlockAMGTest ${MPIRUN} -n 4 ${HOST} ./test_par_block_amg)
    add_test(ParBlockAMGTest ${MPIRUN} -n 8 ${HOST} ./test_par_block_amg)
    
    add_executable(par_block_amg_profiling par_block_amg_profiling.cpp)
    target_link_libraries(par_block_amg_profiling raptor ${MPI_LIBRARIES} )

endif()
