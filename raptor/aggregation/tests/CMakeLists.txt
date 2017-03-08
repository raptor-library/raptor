add_executable(test_strength test_strength.cpp)
target_link_libraries(test_strength raptor ${MPI_LIBRARIES})
add_test(StrengthTest ./test_strength)

add_executable(test_aggregation test_aggregation.cpp)
target_link_libraries(test_aggregation raptor ${MPI_LIBRARIES})
add_test(AggregationTest ./test_aggregation)
