# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called linalg_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(profile_HEADERS
        )
    set(profile_SOURCES
        profiling/profile_comm.cpp
        profiling/profile_block_comm.cpp
        )
else ()
    set(profile_HEADERS
        ""
        )
    set (profile_SOURCES 
        ""
        )
endif()

set(linalg_HEADERS
    ${profile_HEADERS}
    PARENT_SCOPE
    )
set(linalg_SOURCES 
    ${profile_SOURCES}
    PARENT_SCOPE
    )




