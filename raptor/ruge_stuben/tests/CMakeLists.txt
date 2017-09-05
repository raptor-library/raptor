add_executable(test_cf_splitting test_cf_splitting.cpp)
target_link_libraries(test_cf_splitting raptor ${MPI_LIBRARIES})
add_test(TestCFSplitting ./test_cf_splitting)

add_executable(test_ruge_stuben test_ruge_stuben.cpp)
target_link_libraries(test_ruge_stuben raptor ${MPI_LIBRARIES})
add_test(TestRugeStuben ./test_ruge_stuben)
