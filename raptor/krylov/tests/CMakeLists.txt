add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg raptor)
add_test(TestCG ./test_cg)

if (WITH_MPI)
    add_executable(test_par_cg test_par_cg.cpp)
    target_link_libraries(test_par_cg raptor ${MPI_LIBRARIES})
    add_test(TestParCG ./test_par_cg)
endif()

