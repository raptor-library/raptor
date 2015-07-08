include_directories(${raptor_INCDIR})
add_executable(vecnorm vecnorm.cpp)
target_link_libraries(vecnorm raptor ${MPI_LIBRARIES})
