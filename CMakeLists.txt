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
    add_definitions ( -DUSING_MPI )
    # skip deprecated MPI C++ bindings
    add_definitions ( -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX )
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
    add_definitions ( -DUSING_PTSCOTCH )
    find_package(Ptscotch)
    if (PTSCOTCH_FOUND)
        include_directories(${PTSCOTCH_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Cannot find PTSCOTCH.  Try setting PTSCOTCH_DIR" )
        set(WITH_PTSCOTCH ON)
    endif(PTSCOTCH_FOUND)
endif(WITH_PTSCOTCH)

if (WITH_PARMETIS)
    add_definitions ( -DUSING_PARMETIS )
    find_package(ParMetis)
    if (PARMETIS_FOUND)
        include_directories(${PARMETIS_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Cannot find PARMETIS.  Try setting PARMETIS_DIR" )
        set(WITH_PARMETIS ON)
    endif(PARMETIS_FOUND)
endif(WITH_PARMETIS)

if(WITH_HYPRE)
    add_definitions ( -DUSING_HYPRE )
    find_package(Hypre)
    if(HYPRE_FOUND)
        include_directories(${HYPRE_INCLUDE_DIRS})
	else()
	    message(FATAL_ERROR "Cannot find HYPRE. Try setting HYPRE_DIR" )
		set(WITH_HYPRE OFF)
    endif(HYPRE_FOUND)
endif(WITH_HYPRE)

if (WITH_MUELU)
    add_definitions ( -DUSING_MUELU )
    find_package(MueLu CONFIG REQUIRED)
endif(WITH_MUELU)

if(WITH_MFEM)
    add_definitions ( -DUSING_MFEM )
    add_definitions ( -DUSING_HYPRE )
	set(WITH_HYPRE ON)
	if(NOT HYPRE_FOUND)
		find_package(Hypre)
		if(HYPRE_FOUND)
			include_directories(${HYPRE_INCLUDE_DIRS})
		else()
			message(FATAL_ERROR "Cannot find HYPRE (required for MFEM). Try setting HYPRE_DIR.")
		endif(HYPRE_FOUND)
	endif(NOT HYPRE_FOUND)

    add_definitions ( -DUSING_METIS )
	find_package(Metis)
	if(METIS_FOUND)
		include_directories(${METIS_INCLUDE_DIRS})
	else()
		message(FATAL_ERROR "Cannot find metis (required for MFEM). Try setting METIS_DIR.")
	endif(METIS_FOUND)

	find_package(Mfem)
	if (MFEM_FOUND)
		include_directories(${MFEM_INCLUDE_DIRS})
		message(STATUS "MFEM Mesh Directory: " ${MFEM_MESH_PATH})
	else()
		message(FATAL_ERROR "Cannot find mfem. Try setting MFEM_DIR.")
	endif(MFEM_FOUND)
    file(GENERATE
        OUTPUT "${PROJECT_BINARY_DIR}/raptor_mfem_config.hpp"
        CONTENT "#pragma once\n\n#define MFEM_MESH_DIR \"${MFEM_MESH_PATH}\"\n")
    include_directories(${PROJECT_BINARY_DIR})
endif(WITH_MFEM)

if(WITH_PETSC)
    add_definitions( -DUSING_PETSC )
    find_package(PETSc)
    if (PETSC_FOUND)
        include_directories(${PETSC_INCLUDE_DIR} ${PETSC_INCLUDE_CONF})
    else()
        message(FATAL_ERROR "Cannot find petsc.  Try setting PETSC_DIR.")
    endif(PETSC_FOUND)
endif(WITH_PETSC)

if (WITH_AMPI)
	add_definitions(-DUSE_AMPI)
endif(WITH_AMPI)

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
