add_executable(test_spmv_laplacian test_spmv_laplacian.cpp)
target_link_libraries(test_spmv_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(LaplacianSpMVTest ./test_spmv_laplacian)

add_executable(test_spmv_aniso test_spmv_aniso.cpp)
target_link_libraries(test_spmv_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(AnisoSpMVTest ./test_spmv_aniso)

add_executable(test_spmv_random test_spmv_random.cpp)
target_link_libraries(test_spmv_random raptor ${MPI_LIBRARIES} googletest pthread )
add_test(RandomSpMVTest ./test_spmv_random)

add_executable(test_bsr_spmv_laplacian test_bsr_spmv_laplacian.cpp)
target_link_libraries(test_bsr_spmv_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRLaplacianSpMVTest ./test_bsr_spmv_laplacian)

add_executable(test_bsr_spmv_aniso test_bsr_spmv_aniso.cpp)
target_link_libraries(test_bsr_spmv_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRAnisoSpMVTest ./test_bsr_spmv_aniso)

add_executable(test_bsr_spmv_random test_bsr_spmv_random.cpp)
target_link_libraries(test_bsr_spmv_random raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRRandomSpMVTest ./test_bsr_spmv_random)


add_executable(test_jacobi_aniso test_jacobi_aniso.cpp)
target_link_libraries(test_jacobi_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(AnisoJacobiTest ./test_jacobi_aniso)

add_executable(test_jacobi_laplacian test_jacobi_laplacian.cpp)
target_link_libraries(test_jacobi_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(LaplaceJacobiTest ./test_jacobi_laplacian)

add_executable(test_gs_aniso test_gs_aniso.cpp)
target_link_libraries(test_gs_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(AnisoGSTest ./test_gs_aniso)

add_executable(test_gs_laplacian test_gs_laplacian.cpp)
target_link_libraries(test_gs_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(LaplaceGSTest ./test_gs_laplacian)

# CANNOT CURRENTLY RUN THESE TESTS, BUT RAPTOR SEEMS CORRECT
# TODO : UNCOMMENT WHEN PYAMG BUG IS FIXED
#
#add_executable(test_sor_aniso test_sor_aniso.cpp)
#target_link_libraries(test_sor_aniso raptor ${MPI_LIBRARIES} googletest pthread )
#add_test(AnisoSORTest ./test_sor_aniso)
#
#add_executable(test_sor_laplacian test_sor_laplacian.cpp)
#target_link_libraries(test_sor_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
#add_test(LaplaceSORTest ./test_sor_laplacian)

#add_executable(test_bsr_jacobi_aniso test_bsr_jacobi_aniso.cpp)
#target_link_libraries(test_bsr_jacobi_aniso raptor ${MPI_LIBRARIES} googletest pthread )
#add_test(BSRAnisoJacobiTest ./test_bsr_jacobi_aniso)

add_executable(test_bsr_jacobi_aniso test_bsr_jacobi_aniso.cpp)
target_link_libraries(test_bsr_jacobi_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRAnisoJacobiTest ./test_bsr_jacobi_aniso)

add_executable(test_bsr_jacobi_laplacian test_bsr_jacobi_laplacian.cpp)
target_link_libraries(test_bsr_jacobi_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRLaplaceJacobiTest ./test_bsr_jacobi_laplacian)

add_executable(test_bsr_gs_aniso test_bsr_gs_aniso.cpp)
target_link_libraries(test_bsr_gs_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRAnisoGSTest ./test_bsr_gs_aniso)

add_executable(test_bsr_gs_laplacian test_bsr_gs_laplacian.cpp)
target_link_libraries(test_bsr_gs_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRLaplaceGSTest ./test_bsr_gs_laplacian)

if (WITH_MPI)
    add_executable(test_par_add test_par_add.cpp)
    target_link_libraries(test_par_add raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParAddTest ${MPIRUN} -n 1 ${HOST} ./test_par_add)
    add_test(ParAddTest ${MPIRUN} -n 16 ${HOST} ./test_par_add)

    add_executable(test_par_spmv_laplacian test_par_spmv_laplacian.cpp)
    target_link_libraries(test_par_spmv_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParLaplacianSpMVTest ${MPIRUN} -n 1 ${HOST} ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest ${MPIRUN} -n 2 ${HOST} ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest ${MPIRUN} -n 4 ${HOST} ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest ${MPIRUN} -n 8 ${HOST} ./test_par_spmv_laplacian)
    add_test(ParLaplacianSpMVTest ${MPIRUN} -n 16 ${HOST} ./test_par_spmv_laplacian)

    add_executable(test_par_spmv_aniso test_par_spmv_aniso.cpp)
    target_link_libraries(test_par_spmv_aniso raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParAnisoSpMVTest ${MPIRUN} -n 1 ${HOST} ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest ${MPIRUN} -n 2 ${HOST} ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest ${MPIRUN} -n 4 ${HOST} ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest ${MPIRUN} -n 8 ${HOST} ./test_par_spmv_aniso)
    add_test(ParAnisoSpMVTest ${MPIRUN} -n 16 ${HOST} ./test_par_spmv_aniso)

    add_executable(test_par_spmv_random test_par_spmv_random.cpp)
    target_link_libraries(test_par_spmv_random ${MPI_LIBRARIES} raptor googletest pthread )
    add_test(ParRandomSpMVTest ${MPIRUN} -n 1 ${HOST} ./test_par_spmv_random)
    add_test(ParRandomSpMVTest ${MPIRUN} -n 2 ${HOST} ./test_par_spmv_random)
    add_test(ParRandomSpMVTest ${MPIRUN} -n 4 ${HOST} ./test_par_spmv_random)
    add_test(ParRandomSpMVTest ${MPIRUN} -n 8 ${HOST} ./test_par_spmv_random)
    add_test(ParRandomSpMVTest ${MPIRUN} -n 16 ${HOST} ./test_par_spmv_random)
    
    add_executable(test_tap_spmv_laplacian test_tap_spmv_laplacian.cpp)
    target_link_libraries(test_tap_spmv_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TAPLaplacianSpMVTest ${MPIRUN} -n 2 ${HOST} ./test_tap_spmv_laplacian)
    add_test(TAPLaplacianSpMVTest ${MPIRUN} -n 4 ${HOST} ./test_tap_spmv_laplacian)
    add_test(TAPLaplacianSpMVTest ${MPIRUN} -n 8 ${HOST} ./test_tap_spmv_laplacian)
    add_test(TAPLaplacianSpMVTest ${MPIRUN} -n 16 ${HOST} ./test_tap_spmv_laplacian)

    add_executable(test_tap_spmv_aniso test_tap_spmv_aniso.cpp)
    target_link_libraries(test_tap_spmv_aniso raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TAPAnisoSpMVTest ${MPIRUN} -n 2 ${HOST} ./test_tap_spmv_aniso)
    add_test(TAPAnisoSpMVTest ${MPIRUN} -n 4 ${HOST} ./test_tap_spmv_aniso)
    add_test(TAPAnisoSpMVTest ${MPIRUN} -n 8 ${HOST} ./test_tap_spmv_aniso)
    add_test(TAPAnisoSpMVTest ${MPIRUN} -n 16 ${HOST} ./test_tap_spmv_aniso)

    add_executable(test_tap_spmv_random test_tap_spmv_random.cpp)
    target_link_libraries(test_tap_spmv_random ${MPI_LIBRARIES} raptor googletest pthread )
    add_test(TAPRandomSpMVTest ${MPIRUN} -n 2 ${HOST} ./test_tap_spmv_random)
    add_test(TAPRandomSpMVTest ${MPIRUN} -n 4 ${HOST} ./test_tap_spmv_random)
    add_test(TAPRandomSpMVTest ${MPIRUN} -n 8 ${HOST} ./test_tap_spmv_random)

    add_executable(test_par_scale_aniso test_par_scale_aniso.cpp)
    target_link_libraries(test_par_scale_aniso raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(ParScaleAnisoTest ${MPIRUN} -n 1 ${HOST} ./test_par_scale_aniso)
    add_test(ParScaleAnisoTest ${MPIRUN} -n 2 ${HOST} ./test_par_scale_aniso)
    add_test(ParScaleAnisoTest ${MPIRUN} -n 3 ${HOST} ./test_par_scale_aniso)
    add_test(ParScaleAnisoTest ${MPIRUN} -n 6 ${HOST} ./test_par_scale_aniso)

    add_executable(test_repartition test_repartition.cpp)
    target_link_libraries(test_repartition raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(RepartitionTest ${MPIRUN} -n 1 ${HOST} ./test_repartition)
    add_test(RepartitionTest ${MPIRUN} -n 2 ${HOST} ./test_repartition)
    add_test(RepartitionTest ${MPIRUN} -n 3 ${HOST} ./test_repartition)
    add_test(RepartitionTest ${MPIRUN} -n 6 ${HOST} ./test_repartition)
    add_test(RepartitionTest ${MPIRUN} -n 16 ${HOST} ./test_repartition)

    if (WITH_PTSCOTCH)
        add_executable(test_ptscotch test_ptscotch.cpp)
        target_link_libraries(test_ptscotch raptor ${MPI_LIBRARIES} googletest pthread )
        add_test(PTScotchTest ${MPIRUN} -n 1 ${HOST} ./test_ptscotch)
        add_test(PTScotchTest ${MPIRUN} -n 2 ${HOST} ./test_ptscotch)
        add_test(PTScotchTest ${MPIRUN} -n 3 ${HOST} ./test_ptscotch)
        add_test(PTScotchTest ${MPIRUN} -n 6 ${HOST} ./test_ptscotch)
    endif()

    if (WITH_PARMETIS)
        add_executable(test_parmetis test_parmetis.cpp)
        target_link_libraries(test_parmetis raptor ${MPI_LIBRARIES} googletest pthread )
        add_test(ParMetisTest ${MPIRUN} -n 1 ${HOST} ./test_parmetis)
        add_test(ParMetisTest ${MPIRUN} -n 2 ${HOST} ./test_parmetis)
        add_test(ParMetisTest ${MPIRUN} -n 3 ${HOST} ./test_parmetis)
        add_test(ParMetisTest ${MPIRUN} -n 6 ${HOST} ./test_parmetis)
    endif()
endif()
