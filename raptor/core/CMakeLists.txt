# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_core_HEADERS
        core/mpi_types.hpp
        core/topology.hpp
        core/partition.hpp
        core/comm_data.hpp
        core/comm_pkg.hpp
        core/par_vector.hpp
        core/par_matrix.hpp
		core/matrix_traits.hpp
        )
    set(par_core_SOURCES
        core/mpi_types.cpp
        core/comm_data.cpp
        core/tap_comm.cpp
        core/comm_pkg.cpp
        core/comm_mat.cpp
        core/par_vector.cpp
        core/par_matrix.cpp
        )
else ()
    set(par_core_HEADERS
        ""
        )

    set (par_core_SOURCES
        ""
        )
endif()

if (WITH_HIP)
    set(hip_core_SOURCES core/hip_matrix.cpp)
    set(hip_core_HEADERS core/hip_matrix.hpp core/hip_error.hpp)
    if (WITH_MPI)
        list(APPEND hip_core_SOURCES core/par_hip_matrix.cpp)
        list(APPEND hip_core_HEADERS core/par_hip_matrix.hpp)
    endif()
endif()

set(core_SOURCES 
    core/vector.cpp
    core/matrix.cpp
    ${hip_core_SOURCES}
    ${par_core_SOURCES}
    PARENT_SCOPE
    )
set(core_HEADERS
    core/types.hpp
    core/vector.hpp
    core/matrix.hpp
    core/utilities.hpp
    ${hip_core_HEADERS}
    ${par_core_HEADERS}
    PARENT_SCOPE
    )
