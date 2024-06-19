add_executable(test_rap test_rap.cpp)
target_link_libraries(test_rap raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestRAP ./test_rap)

add_executable(test_splitting test_splitting.cpp)
target_link_libraries(test_splitting raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestSplitting ./test_splitting)

add_executable(test_interpolation test_interpolation.cpp)
target_link_libraries(test_interpolation raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestInterpolation ./test_interpolation)

add_executable(test_ruge_stuben test_ruge_stuben.cpp)
target_link_libraries(test_ruge_stuben raptor ${MPI_LIBRARIES} googletest pthread )
add_test(TestRugeStuben ./test_ruge_stuben)

if (WITH_MPI)
    add_executable(test_par_rap test_par_rap.cpp)
    target_link_libraries(test_par_rap raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParRAP ${MPIRUN} -n 16 ${HOST} ./test_par_rap)

    add_executable(test_tap_rap test_tap_rap.cpp)
    target_link_libraries(test_tap_rap raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestTAPRAP ${MPIRUN} -n 16 ${HOST} ./test_tap_rap)

    add_executable(test_par_splitting test_par_splitting.cpp)
    target_link_libraries(test_par_splitting raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParSplitting ${MPIRUN} -n 16 ${HOST} ./test_par_splitting)

    add_executable(test_tap_splitting test_tap_splitting.cpp)
    target_link_libraries(test_tap_splitting raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestTAPSplitting ${MPIRUN} -n 16 ${HOST} ./test_tap_splitting)

    add_executable(test_par_interpolation test_par_interpolation.cpp)
    target_link_libraries(test_par_interpolation raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParInterpolation ${MPIRUN} -n 16 ${HOST} ./test_par_interpolation)

    add_executable(test_air test_air.cpp)
    target_link_libraries(test_air raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestAIR ${MPIRUN} -n 4 ${HOST} ./test_air)

    add_executable(test_tap_interpolation test_tap_interpolation.cpp)
    target_link_libraries(test_tap_interpolation raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestTAPInterpolation ${MPIRUN} -n 16 ${HOST} ./test_tap_interpolation)

    add_executable(test_par_ruge_stuben test_par_ruge_stuben.cpp)
    target_link_libraries(test_par_ruge_stuben raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParRugeStuben ${MPIRUN} -n 16 ${HOST} ./test_par_ruge_stuben)

    add_executable(test_tap_ruge_stuben test_tap_ruge_stuben.cpp)
    target_link_libraries(test_tap_ruge_stuben raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestTAPRugeStuben ${MPIRUN} -n 16 ${HOST} ./test_tap_ruge_stuben)
endif()

if (WITH_HYPRE)
    add_executable(test_hypre test_hypre.cpp)
    target_link_libraries(test_hypre raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestHypre ${MPIRUN} -n 1 ./test_hypre)
    add_test(TestHypre ${MPIRUN} -n 2 ./test_hypre)

    add_executable(test_hypre_agg test_hypre_agg.cpp)
    target_link_libraries(test_hypre_agg raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestHypreAgg ${MPIRUN} -n 1 ./test_hypre_agg)
    add_test(TestHypreAgg ${MPIRUN} -n 2 ./test_hypre_agg)

    add_executable(test_hypre_strength test_hypre_strength.cpp)
    target_link_libraries(test_hypre_strength raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestHypreStr ${MPIRUN} -n 1 ./test_hypre_strength)
    add_test(TestHypreStr ${MPIRUN} -n 2 ./test_hypre_strength)

    add_executable(test_hypre_splitting test_hypre_splitting.cpp)
    target_link_libraries(test_hypre_splitting raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestHypreSplit ${MPIRUN} -n 1 ./test_hypre_splitting)
    add_test(TestHypreSplit ${MPIRUN} -n 2 ./test_hypre_splitting)

    add_executable(test_hypre_interpolation test_hypre_interpolation.cpp)
    target_link_libraries(test_hypre_interpolation raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestHypreInterp ${MPIRUN} -n 1 ./test_hypre_interpolation)
    add_test(TestHypreInterp ${MPIRUN} -n 2 ./test_hypre_interpolation)

    add_executable(test_hypre_unknown test_hypre_unknown.cpp)
    target_link_libraries(test_hypre_unknown raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestHypreUnknown ${MPIRUN} -n 1 ./test_hypre_unknown)
    add_test(TestHypreUnknown ${MPIRUN} -n 2 ./test_hypre_unknown)
endif()

if (WITH_MFEM)
    add_executable(test_mfem_amg test_mfem_amg.cpp)
    target_link_libraries(test_mfem_amg raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestMFEM ${MPIRUN} -n 16 ./test_mfem_amg)
endif()


