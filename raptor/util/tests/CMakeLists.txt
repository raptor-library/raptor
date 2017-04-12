add_executable(test_relaxation test_relaxation.cpp)
target_link_libraries(test_relaxation raptor ${MPI_LIBRARIES})
add_test(RelaxTest ./test_relaxation)

add_executable(test_RAP test_RAP.cpp)
target_link_libraries(test_RAP raptor ${MPI_LIBRARIES})
add_test(RAPTest ./test_RAP)

if (WITH_MPI)
    add_executable(test_par_spmv test_par_spmv.cpp)
    target_link_libraries(test_par_spmv raptor ${MPI_LIBRARIES})
    add_test(ParSpMVTest ./test_par_spmv)

    add_executable(test_tap_spmv test_tap_spmv.cpp)
    target_link_libraries(test_tap_spmv raptor ${MPI_LIBRARIES})
    add_test(TAPSpMVTest ./test_tap_spmv)

    add_executable(test_par_matmult test_par_matmult.cpp)
    target_link_libraries(test_par_matmult raptor ${MPI_LIBRARIES})
    add_test(ParMatmultTest ./test_par_matmult)
endif()
