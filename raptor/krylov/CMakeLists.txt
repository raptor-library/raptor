# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (WITH_MPI)
    set(par_krylov_SOURCES
        krylov/par_cg.cpp
        )
    set(par_krylov_HEADERS
        krylov/par_cg.hpp
        )
else()
    set(par_krylov_SOURCES
        )
    set(par_krylov_HEADERS
        )
endif()
 
set(krylov_HEADERS
    krylov/cg.hpp
    ${par_krylov_HEADERS}
    PARENT_SCOPE)
# Create a variable called linalg_SOURCES containing all .cpp files:
set(krylov_SOURCES 
    krylov/cg.cpp
    ${par_krylov_SOURCES}
PARENT_SCOPE)
