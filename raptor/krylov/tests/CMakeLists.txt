add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg raptor googletest pthread)
add_test(TestCG ./test_cg)

add_executable(test_bicgstab test_bicgstab.cpp)
target_link_libraries(test_bicgstab raptor googletest pthread)
add_test(TestBiCGStab ./test_bicgstab)

if (WITH_MPI)
    add_executable(test_par_cg test_par_cg.cpp)
    target_link_libraries(test_par_cg raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParCG ${MPIRUN} -n 1 ./test_par_cg)
    add_test(TestParCG ${MPIRUN} -n 2 ./test_par_cg)
    add_test(TestParCG ${MPIRUN} -n 4 ./test_par_cg)
    add_test(TestParCG ${MPIRUN} -n 8 ./test_par_cg)
    add_test(TestParCG ${MPIRUN} -n 16 ./test_par_cg)
        
    add_executable(test_par_bicgstab test_par_bicgstab.cpp)
    target_link_libraries(test_par_bicgstab raptor ${MPI_LIBRARIES} googletest pthread)
    #add_test(TestParBiCGStab ${MPIRUN} -n 1 ./test_par_bicgstab)
    #add_test(TestParBiCGStab ${MPIRUN} -n 2 ./test_par_bicgstab)
    #add_test(TestParBiCGStab ${MPIRUN} -n 4 ./test_par_bicgstab)
    #add_test(TestParBiCGStab ${MPIRUN} -n 8 ./test_par_bicgstab)
    #add_test(TestParBiCGStab ${MPIRUN} -n 16 ./test_par_bicgstab)
    
    add_executable(test_par_prebicgstab test_par_prebicgstab.cpp)
    target_link_libraries(test_par_prebicgstab raptor ${MPI_LIBRARIES} googletest pthread)
    #add_test(TestParBiCGStab ${MPIRUN} -n 1 ./test_par_bicgstab)
    
    add_executable(test_par_srecg test_par_srecg.cpp)
    target_link_libraries(test_par_srecg raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParSRECG ${MPIRUN} -n 1 ./test_par_srecg)
    add_test(TestParSRECG ${MPIRUN} -n 2 ./test_par_srecg)
    add_test(TestParSRECG ${MPIRUN} -n 4 ./test_par_srecg)
    add_test(TestParSRECG ${MPIRUN} -n 8 ./test_par_srecg)
    add_test(TestParSRECG ${MPIRUN} -n 16 ./test_par_srecg)

endif()


