if (WITH_MPI)
    add_executable(test_par_comm test_par_comm.cpp)
    target_link_libraries(test_par_comm raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParCommTest_1  mpirun -n 1 ./test_par_comm)
    add_test(ParCommTest_16 mpirun -n 16 ./test_par_comm)

    add_executable(test_tap_comm test_tap_comm.cpp)
    target_link_libraries(test_tap_comm raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TAPCommTest_1   mpirun -n 1  ./test_tap_comm)
    add_test(TAPCommTest_16  mpirun -n 16 ./test_tap_comm)

    add_executable(test_par_matrix test_par_matrix.cpp)
    target_link_libraries(test_par_matrix raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParMatrixTest_1  mpirun -n 1  ./test_par_matrix)
    add_test(ParMatrixTest_16 mpirun -n 16 ./test_par_matrix)

    add_executable(test_par_vector test_par_vector.cpp)
    target_link_libraries(test_par_vector raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParVectorTest_1  mpirun -n 1  ./test_par_vector)
    add_test(ParVectorTest_16 mpirun -n 16 ./test_par_vector)

    add_executable(test_par_transpose test_par_transpose.cpp)
    target_link_libraries(test_par_transpose raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParTransposeTest_1 mpirun -n 1 ./test_par_transpose)
    add_test(ParTransposeTest_16 mpirun -n 16 ./test_par_transpose)    
    
    add_executable(test_par_bsr test_par_bsr.cpp)
    target_link_libraries(test_par_bsr raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParBSRTest_1 mpirun -n 1 ./test_par_bsr)
    add_test(ParBSRTest_3 mpirun -n 3 ./test_par_bsr)
    add_test(ParBSRTest_4 mpirun -n 6 ./test_par_bsr)
endif ()

add_executable(test_bsr_matrix test_bsr_matrix.cpp)
target_link_libraries(test_bsr_matrix raptor googletest pthread )
add_test(BSRMatrixTest ./test_bsr_matrix)

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix raptor googletest pthread )
add_test(MatrixTest ./test_matrix)

add_executable(test_transpose test_transpose.cpp)
target_link_libraries(test_transpose raptor googletest pthread )
add_test(TransposeTest ./test_transpose)


