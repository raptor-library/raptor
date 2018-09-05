# from https://cmake.org/Wiki/CMake:How_To_Find_Libraries

find_package(PkgConfig)
pkg_check_modules(MFEM QUIET mfem)

find_path(MFEM_INCLUDE_DIR mfem.hpp
          HINTS ${PC_MFEM_INCLUDEDIR} ${PC_MFEM_INCLUDE_DIRS}
			$ENV{MFEM_DIR} $ENV{MFEM_DIR}/include)

find_library(MFEM_LIBRARY NAMES mfem
             HINTS ${PC_MFEM_LIBDIR} ${PC_MFEM_LIBRARY_DIRS}
				$ENV{MFEM_DIR} $ENV{MFEM_DIR}/lib)

set(MFEM_LIBRARIES ${MFEM_LIBRARY} )
set(MFEM_INCLUDE_DIRS ${MFEM_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Mfem  DEFAULT_MSG
                                  MFEM_LIBRARY MFEM_INCLUDE_DIR)

mark_as_advanced(MFEM_INCLUDE_DIR MFEM_LIBRARY )
