add_executable(test_relaxation test_relaxation.cpp)
target_link_libraries(test_relaxation raptor ${MPI_LIBRARIES})
add_test(RelaxTest ./test_relaxation)

