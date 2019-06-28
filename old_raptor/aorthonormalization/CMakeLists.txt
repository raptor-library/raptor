# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (WITH_MPI)
    set(par_aorthonormalization_SOURCES
        aorthonormalization/par_cgs.cpp
        aorthonormalization/par_cgs_timed.cpp
        )
    set(par_aorthonormalization_HEADERS
        aorthonormalization/par_cgs.hpp
        aorthonormalization/par_cgs_timed.hpp
        )
else()
    set(par_aorthonormalization_SOURCES
        )
    set(par_aorthonormalization_HEADERS
        )
endif()
 
set(aorthonormalization_HEADERS
    aorthonormalization/cgs.cpp
    ${par_aorthonormalization_HEADERS}
    PARENT_SCOPE)
# Create a variable called linalg_SOURCES containing all .cpp files:
set(aorthonormalization_SOURCES 
    aorthonormalization/cgs.hpp
    ${par_aorthonormalization_SOURCES}
PARENT_SCOPE)


