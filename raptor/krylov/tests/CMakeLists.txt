add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg raptor)
add_test(TestCG ./test_cg)

add_executable(test_bicgstab test_bicgstab.cpp)
target_link_libraries(test_bicgstab raptor)
add_test(TestBiCGStab ./test_bicgstab)

if (WITH_MPI)
    add_executable(test_par_cg test_par_cg.cpp)
    target_link_libraries(test_par_cg raptor ${MPI_LIBRARIES})
    add_test(TestParCG ./test_par_cg)
    
    add_executable(test_par_bicgstab test_par_bicgstab.cpp)
    target_link_libraries(test_par_bicgstab raptor ${MPI_LIBRARIES})
    add_test(TestParBiCGStab ./test_par_bicgstab)

    add_executable(test_partial_inner test_partial_inner.cpp)
    target_link_libraries(test_partial_inner raptor ${MPI_LIBRARIES})
    add_test(TestPartialInner ./test_partial_inner)
endif()

