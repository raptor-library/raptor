# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#Create a variable called linalg_SOURCES containing all .cpp files:

if (WITH_PTSCOTCH)
    set(ptscotch_linalg_HEADERS
        util/linalg/external/ptscotch_wrapper.hpp
        )
else()
    set(ptscotch_linalg_HEADERS
        ""
        )
endif()

if (WITH_PARMETIS)
    set(parmetis_linalg_HEADERS
        util/linalg/external/parmetis_wrapper.hpp
        )
else()
    set(parmetis_linalg_HEADERS
        ""
        )
endif()

set(ext_linalg_HEADERS
    ${ptscotch_linalg_HEADERS}
    ${parmetis_linalg_HEADERS}
    PARENT_SCOPE
    )




