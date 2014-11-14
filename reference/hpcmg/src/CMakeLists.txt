include_directories("." ${MPI_INCLUDE_PATH} "./includes")
add_subdirectory(drivers)
add_subdirectory(linalg)
add_subdirectory(test/linalg)
