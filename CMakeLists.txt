cmake_minimum_required(VERSION 3.18...3.20)
project(raptor VERSION 1.0)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Default to an optimized build when no build type is given
# (single-configuration generators only, e.g. Makefiles and Ninja).
get_property(RAPTOR_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (NOT RAPTOR_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected; defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wredundant-decls -Wcast-align -Wshadow")

include(FeatureSummary)

option(WITH_PTSCOTCH "Enable PTScotch Partitioning" OFF)
option(WITH_PARMETIS "Enable ParMetis Partitioning" OFF)
option(WITH_HYPRE "Add Hypre" OFF)
option(WITH_MUELU "Add Trilinos MueLu" OFF)
option(WITH_MFEM "Add MFEM" OFF)
option(WITH_PETSC "Add Petsc" OFF)
option(WITH_AMPI "Using AMPI" OFF)
option(WITH_MPI "Using MPI" ON)
option(WITH_HOSTFILE "Use a Hostfile with MPI" OFF)

add_feature_info(hypre WITH_HYPRE "Hypre preconditioner")
add_feature_info(ml WITH_MUELU "Trilinos MueLu preconditioner")
add_feature_info(mfem WITH_MFEM "MFEM matrix gallery")
add_feature_info(petsc WITH_PETSC "Petsc Interface")
add_feature_info(ampi WITH_AMPI "Compile with AMPI")
add_feature_info(ptscotch WITH_PTSCOTCH "Enable PTScotch Partitioning")
add_feature_info(parmetis WITH_PARMETIS "Enable ParMetis Partitioning")
add_feature_info(hostfile WITH_HOSTFILE "Enable Hostfile for MPIRUN")

include(options)
include(testing)
include(source)
SET(MPIRUN "")

if (WITH_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    # use the same MPI for both build and launch
    # note: this is only an issue with multiple mpi installs
    if (MPIEXEC_EXECUTABLE)
        SET(MPIRUN ${MPIEXEC_EXECUTABLE})
    else()
        SET(MPIRUN mpirun)
    endif()
endif (WITH_MPI)

#include_directories("external")
set(raptor_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/raptor)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

SET(HYPRE_DIR "" CACHE STRING "Directory of HYPRE")
SET(METIS_DIR "" CACHE STRING "Directory of metis")
SET(MFEM_DIR "" CACHE STRING "Directory of mfem")
SET(MFEM_MESH_DIR "" CACHE STRING "Directory containing MFEM mesh files")
SET(PETSC_DIR "" CACHE STRING "Directory of petsc")
SET(PTSCOTCH_DIR "" CACHE STRING "Directory of Ptscotch")
SET(PARMETIS_DIR "" CACHE STRING "Directory of ParMetis")
SET(HOSTFILE "" CACHE STRING "Set Hostfile")

set(RAPTOR_BLA_VENDOR "${BLA_VENDOR}")
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(EXTERNAL_LIBS LAPACK::LAPACK BLAS::BLAS)

if (WITH_HOSTFILE)
    if (NOT EXISTS "${HOSTFILE}")
        message(FATAL_ERROR "WITH_HOSTFILE is set but HOSTFILE (\"${HOSTFILE}\") does not exist.")
    endif()
    set(HOST "--hostfile" "${HOSTFILE}" "--bind-to" "none")
endif()

if (WITH_PTSCOTCH)
    find_package(Ptscotch)
    if (NOT PTSCOTCH_FOUND)
        message(FATAL_ERROR "Cannot find PTSCOTCH.  Set PTSCOTCH_DIR" )
    endif()
endif(WITH_PTSCOTCH)

if (WITH_PARMETIS)
    find_package(ParMetis)
    if (NOT PARMETIS_FOUND)
        message(FATAL_ERROR "Cannot find PARMETIS.  Set PARMETIS_DIR" )
    endif()
endif(WITH_PARMETIS)

if (WITH_HYPRE)
    find_package(Hypre)
    if (NOT HYPRE_FOUND)
        message(FATAL_ERROR "Cannot find HYPRE. Set HYPRE_DIR" )
    endif()
endif(WITH_HYPRE)

if (WITH_MUELU)
    find_package(MueLu CONFIG REQUIRED)
endif(WITH_MUELU)

if (WITH_MFEM)
    set(WITH_HYPRE ON)
    if (NOT HYPRE_FOUND)
        find_package(Hypre)
        if (NOT HYPRE_FOUND)
            message(FATAL_ERROR "Cannot find HYPRE (required for MFEM). Set HYPRE_DIR.")
        endif()
    endif()

    find_package(Metis)
    if (NOT METIS_FOUND)
        message(FATAL_ERROR "Cannot find metis. Set METIS_DIR.")
    endif()

    find_package(Mfem)
    if (NOT MFEM_FOUND)
        message(FATAL_ERROR "Cannot find mfem. Set MFEM_DIR.")
    endif()
    message(STATUS "MFEM dir: " ${MFEM_MESH_PATH})

    # Mesh directory for the MFEM tests and examples
    file(GENERATE
        OUTPUT "${PROJECT_BINARY_DIR}/raptor_mfem_config.hpp"
        CONTENT "#pragma once\n\n#define MFEM_MESH_DIR \"${MFEM_MESH_PATH}\"\n")
    include_directories(${PROJECT_BINARY_DIR})
endif(WITH_MFEM)

if (WITH_PETSC)
    find_package(PETSc)
    if (NOT PETSC_FOUND)
        message(FATAL_ERROR "Cannot find petsc.  Try setting PETSC_DIR.")
    endif()
endif(WITH_PETSC)

add_subdirectory(raptor)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

find_package(Doxygen)

if (DOXYGEN_FOUND)
    # make the output directory
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)

    # copy the doxygen file with @whatever@ substitutions
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    message("build Doxygen files")

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "generating Doxygen output"
        VERBATIM
        )
else (DOXYGEN_FOUND)
    message("Doxygen required")
endif (DOXYGEN_FOUND)
