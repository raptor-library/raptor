add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg raptor)
add_test(TestCG ./test_cg)

add_executable(test_bicgstab test_bicgstab.cpp)
target_link_libraries(test_bicgstab raptor)
add_test(TestBiCGStab ./test_bicgstab)

if (WITH_MPI)
    add_executable(test_par_cg test_par_cg.cpp)
    target_link_libraries(test_par_cg raptor ${MPI_LIBRARIES})
    add_test(TestParCG ./test_par_cg)
    
    add_executable(test_par_bicgstab test_par_bicgstab.cpp)
    target_link_libraries(test_par_bicgstab raptor ${MPI_LIBRARIES})
    add_test(TestParBiCGStab ./test_par_bicgstab)
    
    add_executable(test_par_prebicgstab test_par_prebicgstab.cpp)
    target_link_libraries(test_par_prebicgstab raptor ${MPI_LIBRARIES})
    add_test(TestParPreBiCGStab ./test_par_prebicgstab)

    add_executable(test_bicgstab_reproducibility test_bicgstab_reproducibility.cpp)
    target_link_libraries(test_bicgstab_reproducibility raptor ${MPI_LIBRARIES})
    add_test(TestBiCGStabReprod ./test_bicgstab_reproducibility)

    add_executable(test_partial_inner test_partial_inner.cpp)
    target_link_libraries(test_partial_inner raptor ${MPI_LIBRARIES})
    add_test(TestPartialInner ./test_partial_inner)
    
    add_executable(test_par_bicgstab_partial_inner_aniso test_par_bicgstab_partial_inner_aniso.cpp)
    target_link_libraries(test_par_bicgstab_partial_inner_aniso raptor ${MPI_LIBRARIES})
    add_test(TestParBiCGStabPartialInnerAniso ./test_par_bicgstab_partial_inner_aniso)
    
    add_executable(test_par_bicgstab_partial_inner_laplacian test_par_bicgstab_partial_inner_laplacian.cpp)
    target_link_libraries(test_par_bicgstab_partial_inner_laplacian raptor ${MPI_LIBRARIES})
    add_test(TestParBiCGStabPartialInnerLaplacian ./test_par_bicgstab_partial_inner_laplacian)
    
    add_executable(test_par_bicgstab_partial_inner_infile test_par_bicgstab_partial_inner_infile.cpp)
    target_link_libraries(test_par_bicgstab_partial_inner_infile raptor ${MPI_LIBRARIES})
    add_test(TestParBiCGStabPartialInnerInputFile ./test_par_bicgstab_partial_inner_infile)
endif()

