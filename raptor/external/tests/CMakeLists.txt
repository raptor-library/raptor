if (WITH_PETSC)
    add_executable(test_petsc_wrapper test_petsc_wrapper.cpp)
    target_link_libraries(test_petsc_wrapper raptor::petsc_wrapper ${MPI_LIBRARIES} googletest )
    add_test(PetscWrapperTest_n1 ${MPIRUN} -n 1 ${HOST} ./test_petsc_wrapper)
    add_test(PetscWrapperTest_n4 ${MPIRUN} -n 4 ${HOST} ./test_petsc_wrapper)
    add_test(PetscWrapperTest_n16 ${MPIRUN} -n 16 ${HOST} ./test_petsc_wrapper)
endif ()

if (WITH_MUELU)
    add_executable(test_trilinos_wrapper test_trilinos_wrapper.cpp)
    target_link_libraries(test_trilinos_wrapper raptor ${MPI_LIBRARIES} googletest )
    add_test(TrilinosWrapperTest_n1 ${MPIRUN} -n 1 ${HOST} ./test_trilinos_wrapper)
    add_test(TrilinosWrapperTest_n4 ${MPIRUN} -n 4 ${HOST} ./test_trilinos_wrapper)
    add_test(TrilinosWrapperTest_n16 ${MPIRUN} -n 16 ${HOST} ./test_trilinos_wrapper)
endif ()





