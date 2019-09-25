# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called linalg_SOURCES containing all .cpp files:

if (WITH_MPI)
    set(par_linalg_HEADERS
        linalg/par_diag_scale.hpp
        )
    set(par_linalg_SOURCES
        linalg/par_spmv.cpp
        linalg/par_vecops.cpp
        linalg/par_matmult.cpp
        linalg/par_add.cpp
        )
else ()
    set(par_linalg_HEADERS
        ""
        )
    set (par_linalg_SOURCES 
        ""
        )
endif()

set(linalg_HEADERS
    ${par_linalg_HEADERS}
    PARENT_SCOPE
    )
set(linalg_SOURCES 
    linalg/matmult.cpp
    linalg/add.cpp
    linalg/spmv.cpp
    linalg/vecops.cpp
    ${par_linalg_SOURCES}
    PARENT_SCOPE
    )




