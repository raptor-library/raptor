# from https://cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)
pkg_check_modules(hypre QUIET hypre)

find_path(HYPRE_INCLUDE_DIR hypre.h
          HINTS ${PC_HYPRE_INCLUDEDIR} ${PC_HYPRE_INCLUDE_DIRS}
          PATH_SUFFIXES hypre)

find_library(HYPRE_LIBRARY NAMES hypre
             HINTS ${PC_HYPRE_LIBDIR} ${PC_HYPRE_LIBRARY_DIRS})

set(HYPRE_LIBRARIES ${HYPRE_LIBRARY} )
set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hypre  DEFAULT_MSG
                                  HYPRE_LIBRARY HYPRE_INCLUDE_DIR)

mark_as_advanced(HYPRE_INCLUDE_DIR HYPRE_LIBRARY )
