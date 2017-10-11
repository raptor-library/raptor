# from https://cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)
pkg_check_modules(PTSCOTCH QUIET ptscotch)

find_path(PTSCOTCH_INCLUDE_DIR ptscotch.h
            HINTS ${PC_PTSCOTCH_INCLUDEDIR} ${PC_PTSCOTCH_INCLUDE_DIRS} 
            ${PTSCOTCH_DIR}/include)

find_library(PTSCOTCH_LIBRARY NAMES ptscotch
            HINTS ${PC_PTSCOTCH_LIBDIR} ${PC_PTSCOTCH_LIBRARY_DIRS} 
            ${PTSCOTCH_DIR}/lib)

find_library(PTSCOTCH_ERR_LIBRARY NAMES ptscotcherr
            HINTS ${PC_PTSCOTCH_LIBDIR} ${PC_PTSCOTCH_LIBRARY_DIRS} 
            ${PTSCOTCH_DIR}/lib)

find_library(PTSCOTCH_ERR_EXIT_LIBRARY NAMES ptscotcherrexit
            HINTS ${PC_PTSCOTCH_LIBDIR} ${PC_PTSCOTCH_LIBRARY_DIRS}
            ${PTSCOTCH_DIR}/lib)

find_library(PTSCOTCH_PARMETIS_LIBRARY NAMES ptscotchparmetis
            HINTS ${PC_PTSCOTCH_LIBDIR} ${PC_PTSCOTCH_LIBRARY_DIRS}
            ${PTSCOTCH_DIR}/lib)

find_library(SCOTCH_LIBRARY NAMES scotch
            HINTS ${PC_PTSCOTCH_LIBDIR} ${PC_PTSCOTCH_LIBRARY_DIRS} 
            ${PTSCOTCH_DIR}/lib)

find_library(SCOTCH_ERR_LIBRARY NAMES scotcherr
            HINTS ${PC_PTSCOTCH_LIBDIR} ${PC_PTSCOTCH_LIBRARY_DIRS} 
            ${PTSCOTCH_DIR}/lib)

find_library(SCOTCH_ERR_EXIT_LIBRARY NAMES scotcherrexit
            HINTS ${PC_PTSCOTCH_LIBDIR} ${PC_PTSCOTCH_LIBRARY_DIRS}
            ${PTSCOTCH_DIR}/lib)

find_library(SCOTCH_METIS_LIBRARY NAMES scotchmetis
            HINTS ${PC_PTSCOTCH_LIBDIR} ${PC_PTSCOTCH_LIBRARY_DIRS}
            ${PTSCOTCH_DIR}/lib)

set(PTSCOTCH_LIBRARIES ${PTSCOTCH_LIBRARY} ${PTSCOTCH_ERR_LIBRARY}
    ${PTSCOTCH_ERR_EXIT_LIBRARY} ${PTSCOTCH_PARMETIS_LIBRARY}
    ${SCOTCH_LIBRARY} ${SCOTCH_ERR_LIBRARY}
    ${SCOTCH_ERR_EXIT_LIBRARY} ${SCOTCH_METIS_LIBRARY})
set(PTSCOTCH_INCLUDE_DIRS ${PTSCOTCH_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PTScotch  DEFAULT_MSG
                                  PTSCOTCH_LIBRARY PTSCOTCH_INCLUDE_DIR)

mark_as_advanced(PTSCOTCH_INCLUDE_DIR PTSCOTCH_LIBRARY)

