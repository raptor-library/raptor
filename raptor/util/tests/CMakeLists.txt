add_executable(test_spmv_laplacian test_spmv_laplacian.cpp)
target_link_libraries(test_spmv_laplacian raptor)
add_test(LaplacianSpMVTest ./test_spmv_laplacian)

add_executable(test_spmv_aniso test_spmv_aniso.cpp)
target_link_libraries(test_spmv_aniso raptor)
add_test(AnisoSpMVTest ./test_spmv_aniso)

add_executable(test_spmv_random test_spmv_random.cpp)
target_link_libraries(test_spmv_random raptor)
add_test(RandomSpMVTest ./test_spmv_random)

if (WITH_MPI)
    add_executable(test_par_spmv_laplacian test_par_spmv_laplacian.cpp)
    target_link_libraries(test_par_spmv_laplacian raptor)
    add_test(ParLaplacianSpMVTest_2 mpirun -n 2 ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest_4 mpirun -n 4 ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest_8 mpirun -n 8 ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest_16 mpirun -n 16 ./test_par_spmv_laplacian)

    add_executable(test_par_spmv_aniso test_par_spmv_aniso.cpp)
    target_link_libraries(test_par_spmv_aniso raptor)
    add_test(ParAnisoSpMVTest_2 mpirun -n 2 ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest_4 mpirun -n 4 ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest_8 mpirun -n 8 ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest_16 mpirun -n 16 ./test_par_spmv_aniso)

    add_executable(test_par_spmv_random test_par_spmv_random.cpp)
    target_link_libraries(test_par_spmv_random raptor)
    add_test(ParRandomSpMVTest_2 mpirun -n 2 ./test_par_spmv_random)
    add_test(ParRandomSpMVTest_4 mpirun -n 4 ./test_par_spmv_random)
    add_test(ParRandomSpMVTest_8 mpirun -n 8 ./test_par_spmv_random)
    add_test(ParRandomSpMVTest_16 mpirun -n 16 ./test_par_spmv_random)

<<<<<<< HEAD
    add_executable(test_repartition test_repartition.cpp)
    target_link_libraries(test_repartition raptor)
=======
    if (WITH_PTSCOTCH)
        add_executable(test_repartition test_repartition.cpp)
        target_link_libraries(test_repartition raptor)
    endif()
>>>>>>> b75c9892c1abfbf8a21b29863ec471395fb32a3f
endif()
