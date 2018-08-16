# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called linalg_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_linalg_HEADERS
        )
    set(par_linalg_SOURCES
        profiling/profile_comm.cpp
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
    ${par_linalg_SOURCES}
    PARENT_SCOPE
    )




