# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called relaxation_SOURCES containing all .cpp files:

if (WITH_MPI)
    set(par_relaxation_HEADERS
        relaxation/par_relax.hpp
        )
    set(par_relaxation_SOURCES
        relaxation/par_relax.cpp
        )
else ()
    set(par_relaxation_HEADERS
        ""
        )
    set (par_relaxation_SOURCES 
        ""
        )
endif()

set(relaxation_HEADERS
    relaxation/relax.hpp
    ${par_relaxation_HEADERS}
    PARENT_SCOPE
    )
set(relaxation_SOURCES 
    relaxation/relax.cpp
    ${par_relaxation_SOURCES}
    PARENT_SCOPE
    )




