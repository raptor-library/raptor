add_executable(test_mgs test_mgs.cpp)
target_link_libraries(test_mgs raptor)
add_test(TestMGS ./test_mgs)

if (WITH_MPI)
    #add_executable(test_par_mgs test_par_mgs.cpp)
    #target_link_libraries(test_par_mgs raptor ${MPI_LIBRARIES})
    #add_test(TestParMGS ./test_par_mgs)
endif()


