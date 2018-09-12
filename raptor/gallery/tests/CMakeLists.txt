add_executable(test_stencil test_stencil.cpp)
target_link_libraries(test_stencil raptor googletest pthread )
add_test(StencilTest ./test_stencil)

add_executable(test_laplacian test_laplacian.cpp)
target_link_libraries(test_laplacian raptor googletest pthread )
add_test(LaplacianTest ./test_laplacian)

add_executable(test_aniso test_aniso.cpp)
target_link_libraries(test_aniso raptor googletest pthread )
add_test(AnisoTest ./test_aniso)

add_executable(test_matrix_market test_matrix_market.cpp)
target_link_libraries(test_matrix_market raptor googletest pthread )
add_test(MatrixMarketTest ./test_matrix_market)


if (WITH_MPI)
    add_executable(test_par_laplacian test_par_laplacian.cpp)
    target_link_libraries(test_par_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParLaplacianTest ${MPIRUN} -n 1 ./test_par_laplacian)
    add_test(ParLaplacianTest ${MPIRUN} -n 2 ./test_par_laplacian)

    add_executable(test_par_aniso test_par_aniso.cpp)
    target_link_libraries(test_par_aniso raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParAnisoTest ${MPIRUN} -n 1 ./test_par_aniso)
    add_test(ParAnisoTest ${MPIRUN} -n 2 ./test_par_aniso)

    add_executable(test_par_matrix_market test_par_matrix_market.cpp)
    target_link_libraries(test_par_matrix_market raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParMatrixMarketTest mpirun -n 1 ./test_par_matrix_market)
    add_test(ParMatrixMarketTest mpirun -n 2 ./test_par_matrix_market)
endif()

