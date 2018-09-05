# from https://cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)
pkg_check_modules(HYPRE QUIET hypre)

find_path(HYPRE_INCLUDE_DIR HYPRE.h
          HINTS ${PC_HYPRE_INCLUDEDIR} ${PC_HYPRE_INCLUDE_DIRS} 
          $ENV{HYPRE_DIR}/include $ENV{HYPRE_DIR}/src/hypre/include)

find_library(HYPRE_LIBRARY NAMES HYPRE
             HINTS ${PC_HYPRE_LIBDIR} ${PC_HYPRE_LIBRARY_DIRS} 
             $ENV{HYPRE_DIR}/lib $ENV{HYPRE_DIR}/src/hypre/lib)

set(HYPRE_LIBRARIES ${HYPRE_LIBRARY} )
set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hypre  DEFAULT_MSG
                                  HYPRE_LIBRARY HYPRE_INCLUDE_DIR)

mark_as_advanced(HYPRE_INCLUDE_DIR HYPRE_LIBRARY )
