if (WITH_MPI)
    add_executable(test_par_comm test_par_comm.cpp)
    target_link_libraries(test_par_comm raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParCommTest ${MPIRUN} -n 1 ${HOST} ./test_par_comm)
    add_test(ParCommTest ${MPIRUN} -n 4 ${HOST} ./test_par_comm)
    add_test(ParCommTest ${MPIRUN} -n 16 ${HOST} ./test_par_comm)

    add_executable(test_tap_comm test_tap_comm.cpp)
    target_link_libraries(test_tap_comm raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TAPCommTest ${MPIRUN} -n 1 ${HOST} ./test_tap_comm)
    add_test(TAPCommTest ${MPIRUN} -n 4 ${HOST} ./test_tap_comm)
    add_test(TAPCommTest ${MPIRUN} -n 16 ${HOST} ./test_tap_comm)

    add_executable(test_par_matrix test_par_matrix.cpp)
    target_link_libraries(test_par_matrix raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParMatrixTest ${MPIRUN} -n 1 ${HOST} ./test_par_matrix)
    add_test(ParMatrixTest ${MPIRUN} -n 4 ${HOST} ./test_par_matrix)
    add_test(ParMatrixTest ${MPIRUN} -n 16 ${HOST} ./test_par_matrix)

    add_executable(test_par_vector test_par_vector.cpp)
    target_link_libraries(test_par_vector raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParVectorTest ${MPIRUN} -n 1 ${HOST} ./test_par_vector)
    add_test(ParVectorTest ${MPIRUN} -n 4 ${HOST} ./test_par_vector)
    add_test(ParVectorTest ${MPIRUN} -n 16 ${HOST} ./test_par_vector)

    add_executable(test_par_transpose test_par_transpose.cpp)
    target_link_libraries(test_par_transpose raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParTransposeTest ${MPIRUN} -n 1 ${HOST} ./test_par_transpose)
    add_test(ParTransposeTest ${MPIRUN} -n 4 ${HOST} ./test_par_transpose)
    add_test(ParTransposeTest ${MPIRUN} -n 16 ${HOST} ./test_par_transpose)
        
    add_executable(test_par_block_matrix test_par_block_matrix.cpp)
    target_link_libraries(test_par_block_matrix raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParBlockMatrixTest ${MPIRUN} -n 1 ${HOST} ./test_par_block_matrix)
    add_test(ParBlockMatrixTest ${MPIRUN} -n 4 ${HOST} ./test_par_block_matrix)
    add_test(ParBlockMatrixTest ${MPIRUN} -n 16 ${HOST} ./test_par_block_matrix)
    
    add_executable(test_par_block_conversion test_par_block_conversion.cpp)
    target_link_libraries(test_par_block_conversion raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParBlockConversionTest ${MPIRUN} -n 1 ${HOST} ./test_par_block_conversion)
    add_test(ParBlockConversionTest ${MPIRUN} -n 4 ${HOST} ./test_par_block_conversion)
    add_test(ParBlockConversionTest ${MPIRUN} -n 16 ${HOST} ./test_par_block_conversion)
    
    add_executable(test_par_block_vector test_par_block_vector.cpp)
    target_link_libraries(test_par_block_vector raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParBlockVectorTest ${MPIRUN} -n 1 ${HOST} ./test_par_block_vector)
    add_test(ParBlockVectorTest ${MPIRUN} -n 4 ${HOST} ./test_par_block_vector)
    add_test(ParBlockVectorTest ${MPIRUN} -n 16 ${HOST} ./test_par_block_vector)

endif ()

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix raptor ${MPI_LIBRARIES} googletest pthread )
add_test(MatrixTest ./test_matrix)

add_executable(test_block_matrix test_block_matrix.cpp)
target_link_libraries(test_block_matrix raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BlockMatrixTest ./test_block_matrix)

add_executable(test_transpose test_transpose.cpp)
target_link_libraries(test_transpose raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TransposeTest ./test_transpose)


