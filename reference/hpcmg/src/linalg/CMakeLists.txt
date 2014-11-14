add_library(linalg SHARED 
    seq/vector.cpp 
    seq/csrmatrix.cpp
    seq/csrmatvec.cpp
    par/vector.cpp
    par/partition.cpp
    par/csrmatrix.cpp
    par/comm_pkg.cpp
    par/csrmatvec.cpp
    )
target_link_libraries(linalg ${MPI_LIBRARIES})
