add_executable(test_relaxation test_relaxation.cpp)
target_link_libraries(test_relaxation raptor ${MPI_LIBRARIES})
add_test(RelaxTest ./test_relaxation)

add_executable(test_RAP test_RAP.cpp)
target_link_libraries(test_RAP raptor ${MPI_LIBRARIES})
add_test(RAPTest ./test_RAP)
