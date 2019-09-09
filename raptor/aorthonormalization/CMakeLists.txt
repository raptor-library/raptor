# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a variable called par_aorthonormolization_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_aorthonormalization_HEADERS
        aorthonormalization/par_cgs.hpp)
    set(par_aorthonormalization_SOURCES
        aorthonormalization/par_cgs.cpp)
else()
    set(par_aorthonormalization_HEADERS
    ""
    )
    set(par_aorthonormalization_SOURCES
    ""
    )
endif()

set(aorthonormalization_HEADERS
    ${par_aorthonormalization_HEADERS}
    PARENT_SCOPE)
set(aorthonormalization_SOURCES
    ${par_aorthonormalization_SOURCES}
    PARENT_SCOPE)
