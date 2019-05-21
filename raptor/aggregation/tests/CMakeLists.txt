add_executable(test_mis test_mis.cpp)
target_link_libraries(test_mis raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestMIS ./test_mis)

add_executable(test_aggregate test_aggregate.cpp)
target_link_libraries(test_aggregate raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestAggregate ./test_aggregate)

add_executable(test_candidates test_candidates.cpp)
target_link_libraries(test_candidates raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestCandidates ./test_candidates)

add_executable(test_prolongation test_prolongation.cpp)
target_link_libraries(test_prolongation raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestProlongation ./test_prolongation)

add_executable(test_smoothed_aggregation test_smoothed_aggregation.cpp)
target_link_libraries(test_smoothed_aggregation raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestSmoothedAggregation ./test_smoothed_aggregation)

if (WITH_MPI)
    add_executable(test_par_mis test_par_mis.cpp)
    target_link_libraries(test_par_mis raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParMIS ${MPIRUN} -n 1 ${HOST} ./test_par_mis)
    add_test(TestParMIS ${MPIRUN} -n 4 ${HOST} ./test_par_mis)
    add_test(TestParMIS ${MPIRUN} -n 16 ${HOST} ./test_par_mis)

    add_executable(test_tap_mis test_tap_mis.cpp)
    target_link_libraries(test_tap_mis raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestTAPMIS ${MPIRUN} -n 16 ${HOST} ./test_tap_mis)

    add_executable(test_par_aggregate test_par_aggregate.cpp)
    target_link_libraries(test_par_aggregate raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParAggregate ${MPIRUN} -n 1 ${HOST} ./test_par_aggregate)
    add_test(TestParAggregate ${MPIRUN} -n 4 ${HOST} ./test_par_aggregate)
    add_test(TestParAggregate ${MPIRUN} -n 16 ${HOST} ./test_par_aggregate)

    add_executable(test_tap_aggregate test_tap_aggregate.cpp)
    target_link_libraries(test_tap_aggregate raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestTAPAggregate ${MPIRUN} -n 16 ${HOST} ./test_tap_aggregate)

    add_executable(test_par_candidates test_par_candidates.cpp)
    target_link_libraries(test_par_candidates raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParCandidates ${MPIRUN} -n 1 ${HOST} ./test_par_candidates)
    add_test(TestParCandidates ${MPIRUN} -n 4 ${HOST} ./test_par_candidates)
    add_test(TestParCandidates ${MPIRUN} -n 16 ${HOST} ./test_par_candidates)

    add_executable(test_tap_candidates test_tap_candidates.cpp)
    target_link_libraries(test_tap_candidates raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestTAPCandidates ${MPIRUN} -n 16 ${HOST} ./test_tap_candidates)

    add_executable(test_par_prolongation test_par_prolongation.cpp)
    target_link_libraries(test_par_prolongation raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParProlongation ${MPIRUN} -n 1 ${HOST} ./test_par_prolongation)
    add_test(TestParProlongation ${MPIRUN} -n 4 ${HOST} ./test_par_prolongation)
    add_test(TestParProlongation ${MPIRUN} -n 16 ${HOST} ./test_par_prolongation)

    add_executable(test_tap_prolongation test_tap_prolongation.cpp)
    target_link_libraries(test_tap_prolongation raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestTAPProlongation ${MPIRUN} -n 16 ${HOST} ./test_tap_prolongation)

    add_executable(test_par_smoothed_aggregation test_par_smoothed_aggregation.cpp)
    target_link_libraries(test_par_smoothed_aggregation raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParSmoothedAggregation ${MPIRUN} -n 1 ${HOST} ./test_par_smoothed_aggregation)
    add_test(TestParSmoothedAggregation ${MPIRUN} -n 4 ${HOST} ./test_par_smoothed_aggregation)
    add_test(TestParSmoothedAggregation ${MPIRUN} -n 16 ${HOST} ./test_par_smoothed_aggregation)
endif()




