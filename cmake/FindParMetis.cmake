# from https://cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)
pkg_check_modules(PARMETIS QUIET parmetis)

find_path(PARMETIS_INCLUDE_DIR parmetis.h
            HINTS ${PC_PARMETIS_INCLUDEDIR} ${PC_PARMETIS_INCLUDE_DIRS} 
            ${PARMETIS_DIR}/include $ENV{PARMETIS_DIR}/include)

find_library(PARMETIS_LIBRARY NAMES libparmetis.a 
            HINTS ${PC_PARMETIS_LIBDIR} ${PC_PARMETIS_LIBRARY_DIRS} 
            ${PARMETIS_DIR}/lib $ENV{PARMETIS_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ParMetis  DEFAULT_MSG
                                  PARMETIS_LIBRARY PARMETIS_INCLUDE_DIR)

set(PARMETIS_LIBRARIES ${PARMETIS_LIBRARY} )
set(PARMETIS_INCLUDE_DIRS ${PARMETIS_INCLUDE_DIR} )

mark_as_advanced(PARMETIS_INCLUDE_DIR PARMETIS_LIBRARY)


