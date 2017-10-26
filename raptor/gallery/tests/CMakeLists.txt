add_executable(test_stencil test_stencil.cpp)
target_link_libraries(test_stencil raptor)
add_test(StencilTest ./test_stencil)

add_executable(test_laplacian test_laplacian.cpp)
target_link_libraries(test_laplacian raptor)
add_test(LaplacianTest ./test_laplacian)

add_executable(test_aniso test_aniso.cpp)
target_link_libraries(test_aniso raptor)
add_test(AnisoTest ./test_aniso)

if (WITH_MPI)
    add_executable(test_par_laplacian test_par_laplacian.cpp)
    target_link_libraries(test_par_laplacian raptor ${MPI_LIBRARIES})
    add_test(ParLaplacianTest ./test_par_laplacian)

    add_executable(test_par_aniso test_par_aniso.cpp)
    target_link_libraries(test_par_aniso raptor ${MPI_LIBRARIES})
    add_test(ParAnisoTest ./test_par_aniso)

    add_executable(test_exxon_reader test_exxon_reader.cpp)
    target_link_libraries(test_exxon_reader raptor ${MPI_LIBRARIES})
endif()


