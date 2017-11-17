add_executable(test_rap test_rap.cpp)
target_link_libraries(test_rap raptor googletest pthread )
add_test(TestRAP ./test_rap)

add_executable(test_splitting test_splitting.cpp)
target_link_libraries(test_splitting raptor googletest pthread )
add_test(TestSplitting ./test_splitting)

add_executable(test_interpolation test_interpolation.cpp)
target_link_libraries(test_interpolation raptor googletest pthread )
add_test(TestInterpolation ./test_interpolation)

add_executable(test_ruge_stuben test_ruge_stuben.cpp)
target_link_libraries(test_ruge_stuben raptor googletest pthread )
add_test(TestRugeStuben ./test_ruge_stuben)

if (WITH_MPI)
    add_executable(test_par_rap test_par_rap.cpp)
    target_link_libraries(test_par_rap raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParRAP_1 mpirun -n 1 ./test_par_rap)
    add_test(TestParRAP_2 mpirun -n 2 ./test_par_rap)

    add_executable(test_par_splitting test_par_splitting.cpp)
    target_link_libraries(test_par_splitting raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParSplitting_1 mpirun -n 1 ./test_par_splitting)
    add_test(TestParSplitting_2 mpirun -n 2 ./test_par_splitting)

    add_executable(test_par_interpolation test_par_interpolation.cpp)
    target_link_libraries(test_par_interpolation raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParInterpolation_1 mpirun -n 1 ./test_par_interpolation)
    add_test(TestParInterpolation_2 mpirun -n 2 ./test_par_interpolation)

    add_executable(test_par_ruge_stuben test_par_ruge_stuben.cpp)
    target_link_libraries(test_par_ruge_stuben raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(TestParRugeStuben_1 mpirun -n 1 ./test_par_ruge_stuben)
    add_test(TestParRugeStuben_2 mpirun -n 2 ./test_par_ruge_stuben)
endif()

if (WITH_HYPRE)
    add_executable(test_hypre test_hypre.cpp)
    target_link_libraries(test_hypre raptor ${MPI_LIBRARIES} googletest pthread)
    add_test(TestHypre_1 mpirun -n 1 ./test_hypre)
    add_test(TestHypre_2 mpirun -n 2 ./test_hypre)
endif()



