#TODO -- Currently using enviroment variables for HYPRE_INC, HYPRE_LIB, METIS_INC, METIS_LIB, MFEM_INC, MFEM_LIB (or HYPRE_DIR, METIS_DIR, MFEM_DIR).  Should we be using something else?

cmake_minimum_required(VERSION 2.8)
project(raptor)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(options)
include(testing)
include(cxx11)
include(source)
include(mpi_config)
include(cxx_config)

include_directories("external")
set(raptor_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/raptor)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

if (DEFINED ENV{HYPRE_DIR})
    set(ENV{HYPRE_INC} "$ENV{HYPRE_DIR}/hypre/src/hypre/include")
    set(ENV{HYPRE_LIB} "$ENV{HYPRE_DIR}/hypre/src/hypre/lib")
endif()

if (DEFINED ENV{MFEM_DIR})
    set(ENV{MFEM_INC} "$ENV{MFEM_DIR}/mfem")
    set(ENV{MFEM_LIB} "$ENV{MFEM_DIR}/mfem")
    message( STATUS "MFEM_INC= $ENV{MFEM_INC} and MFEM_LIB = $ENV{MFEM_LIB}")
endif()

if (DEFINED ENV{METIS_DIR})
    set(ENV{METIS_INC} "$ENV{METIS_DIR}/build/Linux-x86_64/include")
    set(ENV{METIS_LIB} "$ENV{METIS_DIR}/build/Linux-x86_64/libmetis")
    message( STATUS "METIS_INC= $ENV{METIS_INC} and METIS_LIB = $ENV{METIS_LIB}")
endif()

set(WITH_HYPRE false)
set(WITH_MFEM false)
if (DEFINED ENV{HYPRE_INC} AND DEFINED ENV{HYPRE_LIB})
    set(WITH_HYPRE true)
    if (DEFINED ENV{MFEM_INC} AND DEFINED ENV{MFEM_LIB} AND DEFINED ENV{METIS_INC} AND DEFINED ENV{METIS_LIB})
        set(WITH_MFEM true)
    else()
        message( "To include mfem_wrapper, set MFEM_INC, MFEM_LIB, METIS_INC, and METIS_DIR environment variables")
    endif()
else()
    message( "To include hypre_wrapper, set HYPRE_INC and HYPRE_LIB environment variables.  To include mfem_wrapper, also set MFEM_INC, MFEM_LIB, METIS_INC, and METIS_LIB environment variables")
endif()

add_subdirectory(raptor)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
