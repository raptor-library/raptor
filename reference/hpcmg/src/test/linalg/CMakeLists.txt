add_executable(linalg-test seqvector_test.cpp 
                           csrmatrix_test.cpp
                           main.cpp
                           )
target_link_libraries(linalg-test linalg gtest gtest_main m)
add_test(linalg-test linalg-test)

add_executable(par-linalg-test parvector_test.cpp
    main.cpp
    )
target_link_libraries(par-linalg-test linalg gtest gtest_main m
    ${MPI_LIBRARIES})
add_mpi_test(par-linalg-test 30)
