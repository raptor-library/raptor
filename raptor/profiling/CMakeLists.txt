# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called linalg_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_profile_HEADERS
        )
    set(par_profile_SOURCES
        profiling/profile_comm.cpp
        )
else ()
    set(par_profile_HEADERS
        ""
        )
    set (par_profile_SOURCES 
        ""
        )
endif()

set(profile_HEADERS
    ${par_profile_HEADERS}
    PARENT_SCOPE
    )
set(profile_SOURCES 
    ${par_profile_SOURCES}
    PARENT_SCOPE
    )




