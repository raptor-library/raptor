# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_core_comm_SOURCES
        core/comm_data.hpp
        core/comm_pkg.hpp
        core/tap_comm.cpp
        core/comm_pkg.cpp
        )
    set(par_core_SOURCES
        core/par_vector.hpp
        core/par_vector.cpp
        core/par_matrix.hpp
        core/par_matrix.cpp
        )
else ()
    set(par_core_comm_SOURCES
        ""
        )

    set (par_core_SOURCES
        ""
        )
endif()

set(core_SOURCES 
    core/types.hpp
    core/vector.hpp
    core/vector.cpp
    core/matrix.hpp
    core/matrix.cpp
    ${par_core_comm_SOURCES}
    ${par_core_SOURCES}
    PARENT_SCOPE
    )

