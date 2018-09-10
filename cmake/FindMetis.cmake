# from https://cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)
pkg_check_modules(METIS QUIET metis)

find_path(METIS_INCLUDE_DIR metis.h
          HINTS ${PC_METIS_INCLUDEDIR} ${PC_METIS_INCLUDE_DIRS}
          ${METIS_DIR}/Lib $ENV{METIS_DIR}/Lib)

find_library(METIS_LIBRARY NAMES libmetis.a
             HINTS ${PC_METIS_LIBDIR} ${PC_METIS_LIBRARY_DIRS}
             ${METIS_DIR} $ENV{METIS_DIR})

set(METIS_LIBRARIES ${METIS_LIBRARY} )
set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metis  DEFAULT_MSG
                                  METIS_LIBRARY METIS_INCLUDE_DIR)

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY )
