add_executable(test_mis test_mis.cpp)
target_link_libraries(test_mis raptor googletest pthread )
add_test(TestMIS ./test_mis)

add_executable(test_aggregate test_aggregate.cpp)
target_link_libraries(test_aggregate raptor googletest pthread )
add_test(TestAggregate ./test_aggregate)

add_executable(test_candidates test_candidates.cpp)
target_link_libraries(test_candidates raptor googletest pthread )
add_test(TestCandidates ./test_candidates)

add_executable(test_prolongation test_prolongation.cpp)
target_link_libraries(test_prolongation raptor googletest pthread )
add_test(TestProlongation ./test_prolongation)

add_executable(test_smoothed_aggregation test_smoothed_aggregation.cpp)
target_link_libraries(test_smoothed_aggregation raptor googletest pthread )
add_test(TestSmoothedAggregation ./test_smoothed_aggregation)

if (WITH_MPI)
    add_executable(test_par_mis test_par_mis.cpp)
    target_link_libraries(test_par_mis raptor googletest pthread )
    add_test(TestParMIS mpirun -n 1 ./test_par_mis)
    add_test(TestParMIS mpirun -n 4 ./test_par_mis)
    add_test(TestParMIS mpirun -n 16 ./test_par_mis)

    add_executable(test_par_aggregate test_par_aggregate.cpp)
    target_link_libraries(test_par_aggregate raptor googletest pthread )
    add_test(TestParAggregate mpirun -n 1 ./test_par_aggregate)
    add_test(TestParAggregate mpirun -n 4 ./test_par_aggregate)
    add_test(TestParAggregate mpirun -n 16 ./test_par_aggregate)

    add_executable(test_par_candidates test_par_candidates.cpp)
    target_link_libraries(test_par_candidates raptor googletest pthread )
    add_test(TestParCandidates mpirun -n 1 ./test_par_candidates)
    add_test(TestParCandidates mpirun -n 4 ./test_par_candidates)
    add_test(TestParCandidates mpirun -n 16 ./test_par_candidates)

    add_executable(test_par_prolongation test_par_prolongation.cpp)
    target_link_libraries(test_par_prolongation raptor googletest pthread )
    add_test(TestParProlongation mpirun -n 1 ./test_par_prolongation)
    add_test(TestParProlongation mpirun -n 4 ./test_par_prolongation)
    add_test(TestParProlongation mpirun -n 16 ./test_par_prolongation)

    add_executable(test_par_smoothed_aggregation test_par_smoothed_aggregation.cpp)
    target_link_libraries(test_par_smoothed_aggregation raptor googletest pthread )
    add_test(TestParSmoothedAggregation mpirun -n 1 ./test_par_smoothed_aggregation)
    add_test(TestParSmoothedAggregation mpirun -n 4 ./test_par_smoothed_aggregation)
    add_test(TestParSmoothedAggregation mpirun -n 16 ./test_par_smoothed_aggregation)
endif()




