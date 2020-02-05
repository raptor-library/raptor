add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg raptor ${MPI_LIBRARIES} googletest pthread)
add_test(TestCG ./test_cg)

add_executable(test_bicgstab test_bicgstab.cpp)
target_link_libraries(test_bicgstab raptor ${MPI_LIBRARIES} googletest pthread)
add_test(TestBiCGStab ./test_bicgstab)

if (WITH_MPI)
    add_executable(test_par_cg test_par_cg.cpp)
    target_link_libraries(test_par_cg raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParCG ${MPIRUN} -n 1 ${HOST} ./test_par_cg)
    add_test(TestParCG ${MPIRUN} -n 2 ${HOST} ./test_par_cg)
        
    add_executable(test_par_bicgstab test_par_bicgstab.cpp)
    target_link_libraries(test_par_bicgstab raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParBiCGStab ${MPIRUN} -n 1 ${HOST} ./test_par_bicgstab)
    add_test(TestParBiCGStab ${MPIRUN} -n 2 ${HOST} ./test_par_bicgstab)
    
    add_executable(test_par_srecg test_par_srecg.cpp)
    target_link_libraries(test_par_srecg raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParSRECG ${MPIRUN} -n 1 ${HOST} ./test_par_srecg)
    add_test(TestParSRECG ${MPIRUN} -n 2 ${HOST} ./test_par_srecg)
    
    add_executable(test_par_psrecg test_par_psrecg.cpp)
    target_link_libraries(test_par_psrecg raptor ${MPI_LIBRARIES} googletest pthread)
    
    add_executable(test_par_ekcg test_par_ekcg.cpp)
    target_link_libraries(test_par_ekcg raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParSRECG ${MPIRUN} -n 1 ${HOST} ./test_par_ekcg)
    #add_test(TestParSRECG ${MPIRUN} -n 2 ${HOST} ./test_par_ekcg)
    
    add_executable(par_srecg_profiling par_srecg_profiling.cpp)
    target_link_libraries(par_srecg_profiling raptor ${MPI_LIBRARIES} googletest pthread)
    
    add_executable(par_ekcg_profiling par_ekcg_profiling.cpp)
    target_link_libraries(par_ekcg_profiling raptor ${MPI_LIBRARIES} googletest pthread)
    
    add_executable(par_cg_profiling par_cg_profiling.cpp)
    target_link_libraries(par_cg_profiling raptor ${MPI_LIBRARIES} googletest pthread)

endif()


