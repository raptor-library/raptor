add_executable(test_cf_splitting test_cf_splitting.cpp)
target_link_libraries(test_cf_splitting raptor ${MPI_LIBRARIES})
add_test(TestCFSplitting ./test_cf_splitting)
