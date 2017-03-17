# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_core_SOURCES
        core/par_matrix.cpp
        core/par_vector.cpp
        core/par_comm.cpp
        )
else ()
    set (par_core_SOURCES 
        ""
        )
endif()

set(core_SOURCES 
    core/matrix.cpp
    core/vector.cpp
    ${par_core_SOURCES}
    PARENT_SCOPE
    )

