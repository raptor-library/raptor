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

if (WITH_MPI)

endif()




