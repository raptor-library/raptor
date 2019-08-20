# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called linalg_SOURCES containing all .cpp files:

if (WITH_PTSCOTCH)
    set(external_linalg_HEADERS
        util/linalg/external/ptscotch_wrapper.hpp
        )
else()
    set(external_linalg_HEADERS
        ""
        )
endif()

if (WITH_MPI)
    set(par_linalg_HEADERS
        util/linalg/repartition.hpp
        util/linalg/par_relax.hpp
        util/linalg/par_diag_scale.hpp
        )
    set(par_linalg_SOURCES
        util/linalg/par_spmv.cpp
        util/linalg/par_vecops.cpp
        util/linalg/par_matmult.cpp
        util/linalg/par_add.cpp
        util/linalg/par_relax.cpp
        util/linalg/repartition.cpp
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
    util/linalg/relax.hpp
    ${par_linalg_HEADERS}
    ${external_linalg_HEADERS}
    PARENT_SCOPE
    )
set(linalg_SOURCES 
    util/linalg/matmult.cpp
    util/linalg/relax.cpp
    util/linalg/add.cpp
    util/linalg/spmv.cpp
    util/linalg/vecops.cpp
    ${par_linalg_SOURCES}
    PARENT_SCOPE
    )




