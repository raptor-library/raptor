# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called linalg_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_linalg_SOURCES
        util/linalg/spmv.cpp
        util/linalg/matmult.cpp
        )
else ()
    set (par_linalg_SOURCES 
        ""
        )
endif()

set(linalg_SOURCES 
    ${par_linalg_SOURCES}
    util/linalg/seq/spmv.cpp
    util/linalg/seq/subtract.cpp
    util/linalg/seq/matmult.cpp
    util/linalg/seq/relax.cpp
    PARENT_SCOPE
    )




