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
        
    add_executable(test_par_bicgstab test_par_bicgstab.cpp)
    target_link_libraries(test_par_bicgstab raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestParBiCGStab ${MPIRUN} -n 1 ./test_par_bicgstab)

endif()


