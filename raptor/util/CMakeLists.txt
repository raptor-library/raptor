# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called util_SOURCES containing all .cpp files:

if (WITH_PTSCOTCH)
    set(external_util_HEADERS
        util/external/ptscotch_wrapper.hpp
        )
else()
    set(external_util_HEADERS
        ""
        )
endif()

if (WITH_MPI)
    set(par_util_HEADERS
        util/repartition.hpp
        )
    set(par_util_SOURCES
        util/repartition.cpp
        )
else ()
    set(par_util_HEADERS
        ""
        )
    set (par_util_SOURCES 
        ""
        )
endif()

set(util_HEADERS
    ${par_util_HEADERS}
    ${external_util_HEADERS}
    PARENT_SCOPE
    )
set(util_SOURCES 
    ${par_util_SOURCES}
    PARENT_SCOPE
    )




