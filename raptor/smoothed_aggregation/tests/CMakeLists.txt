add_executable(test_seq_strength seq/test_strength.cpp)
target_link_libraries(test_seq_strength raptor ${MPI_LIBRARIES})
add_test(SeqStrengthTest ./test_seq_strength)

add_executable(test_seq_aggregation seq/test_aggregation.cpp)
target_link_libraries(test_seq_aggregation raptor ${MPI_LIBRARIES})
add_test(SeqAggregationTest ./test_seq_aggregation)

add_executable(test_par_strength test_par_strength.cpp)
target_link_libraries(test_par_strength raptor ${MPI_LIBRARIES})
add_test(ParStrengthTest ./test_par_strength)

add_executable(test_par_mis test_par_mis.cpp)
target_link_libraries(test_par_mis raptor ${MPI_LIBRARIES})
add_test(ParMISTest ./test_par_mis)

add_executable(test_par_aggregation test_par_aggregation.cpp)
target_link_libraries(test_par_aggregation raptor ${MPI_LIBRARIES})
add_test(ParAggregationTest ./test_par_aggregation)


