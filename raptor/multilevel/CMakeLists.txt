# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_multilevel_HEADERS
        multilevel/par_level.hpp
        multilevel/par_multilevel.hpp
        )
    set(par_multilevel_SOURCES
        )
else ()
    set (par_multilevel_HEADERS
        ""
        )
    set (par_multilevel_SOURCES
        ""
        )
endif()

set(multilevel_HEADERS 
    multilevel/level.hpp
    multilevel/multilevel.hpp
    ${par_multilevel_HEADERS}
    PARENT_SCOPE
    )

set(multilevel_SOURCES
    ${par_multilevel_SOURCES}
    PARENT_SCOPE
    )


