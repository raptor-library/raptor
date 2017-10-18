add_executable(test_stencil test_stencil.cpp)
target_link_libraries(test_stencil raptor googletest pthread )
add_test(StencilTest ./test_stencil)

add_executable(test_laplacian test_laplacian.cpp)
target_link_libraries(test_laplacian raptor googletest pthread )
add_test(LaplacianTest ./test_laplacian)

add_executable(test_aniso test_aniso.cpp)
target_link_libraries(test_aniso raptor googletest pthread )
add_test(AnisoTest ./test_aniso)

if (WITH_MPI)
    add_executable(test_par_laplacian test_par_laplacian.cpp)
    target_link_libraries(test_par_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParLaplacianTest ./test_par_laplacian)

    add_executable(test_par_aniso test_par_aniso.cpp)
    target_link_libraries(test_par_aniso raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParAnisoTest ./test_par_aniso)
endif()

