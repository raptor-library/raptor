# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called linalg_SOURCES containing all .cpp files:

if (WITH_MPI)
    set(par_linalg_HEADERS
        util/linalg/repartition.hpp
        util/linalg/par_relax.hpp
        util/linalg/par_diag_scale.hpp
        util/linalg/par_spectral_radius.hpp
        )
    set(par_linalg_SOURCES
        util/linalg/par_spmv.cpp
        util/linalg/par_matmult.cpp
        util/linalg/par_add.cpp
        util/linalg/par_relax.cpp
        util/linalg/repartition.cpp
        util/linalg/par_diag_scale.cpp
        util/linalg/par_spectral_radius.cpp
        )
else ()
    set(par_linalg_HEADERS
        ""
        )
    set (par_linalg_SOURCES 
        ""
        )
endif()

if (WITH_HIP)
    set(hip_linalg_HEADERS util/linalg/hip_spmv.hpp util/linalg/hip_vector.hpp)
    set(hip_linalg_SOURCES util/linalg/hip_spmv.cpp util/linalg/hip_vector.cpp)
    if (WITH_MPI)
        list(APPEND hip_linalg_HEADERS util/linalg/par_hip_spmv.hpp)
        list(APPEND hip_linalg_SOURCES util/linalg/par_hip_spmv.cpp)
        list(APPEND hip_linalg_HEADERS util/linalg/par_hip_vector.hpp)
        list(APPEND hip_linalg_SOURCES util/linalg/par_hip_vector.cpp)
    endif()
endif()

set(linalg_HEADERS
    util/linalg/lapack_wrapper.hpp
    util/linalg/relax.hpp
    ${hip_linalg_HEADERS}
    ${par_linalg_HEADERS}
    ${external_linalg_HEADERS}
    PARENT_SCOPE
    )
set(linalg_SOURCES 
    util/linalg/lapack_wrapper.cpp
    util/linalg/matmult.cpp
    util/linalg/relax.cpp
    util/linalg/add.cpp
    util/linalg/spmv.cpp
    ${hip_linalg_SOURCES}
    ${par_linalg_SOURCES}
    PARENT_SCOPE
    )



