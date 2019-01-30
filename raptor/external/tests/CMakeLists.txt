if (WITH_PETSC)
    add_executable(test_petsc_wrapper test_petsc_wrapper.cpp)
    target_link_libraries(test_petsc_wrapper raptor ${MPI_LIBRARIES} googletest pthread )
    add_test(PetscWrapperTest ${MPIRUN} -n 1 ./test_petsc_wrapper)
    add_test(PetscWrapperTest ${MPIRUN} -n 4 ./test_petsc_wrapper)
    add_test(PetscWrapperTest ${MPIRUN} -n 16 ./test_petsc_wrapper)
endif ()




