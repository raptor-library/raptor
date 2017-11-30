# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_core_HEADERS
        core/topology.hpp
        core/partition.hpp
        core/comm_data.hpp
        core/comm_pkg.hpp
        core/par_vector.hpp
        core/par_matrix.hpp
        )
    set(par_core_SOURCES
        core/comm_data.cpp
        core/tap_comm.cpp
        core/comm_pkg.cpp
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

set(core_SOURCES 
    core/vector.cpp
    core/matrix.cpp
    ${par_core_SOURCES}
    PARENT_SCOPE
    )
set(core_HEADERS
    core/types.hpp
    core/vector.hpp
    core/matrix.hpp
    ${par_core_HEADERS}
    PARENT_SCOPE
    )
