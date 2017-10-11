add_executable(test_rap test_rap.cpp)
target_link_libraries(test_rap raptor)
add_test(TestRAP ./test_rap)

add_executable(test_splitting test_splitting.cpp)
target_link_libraries(test_splitting raptor)
add_test(TestSplitting ./test_splitting)

add_executable(test_interpolation test_interpolation.cpp)
target_link_libraries(test_interpolation raptor)
add_test(TestInterpolation ./test_interpolation)

add_executable(test_ruge_stuben test_ruge_stuben.cpp)
target_link_libraries(test_ruge_stuben raptor)
add_test(TestRugeStuben ./test_ruge_stuben)

if (WITH_MPI)
    add_executable(test_par_rap test_par_rap.cpp)
    target_link_libraries(test_par_rap raptor ${MPI_LIBRARIES})
    add_test(TestParRAP ./test_par_rap)

    add_executable(test_par_splitting test_par_splitting.cpp)
    target_link_libraries(test_par_splitting raptor ${MPI_LIBRARIES})
    add_test(TestParSplitting ./test_par_splitting)

    add_executable(test_par_interpolation test_par_interpolation.cpp)
    target_link_libraries(test_par_interpolation raptor ${MPI_LIBRARIES})
    add_test(TestParInterpolation ./test_par_interpolation)

    add_executable(test_par_ruge_stuben test_par_ruge_stuben.cpp)
    target_link_libraries(test_par_ruge_stuben raptor ${MPI_LIBRARIES})
    add_test(TestParRugeStuben ./test_par_ruge_stuben)
endif()


