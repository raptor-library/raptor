cmake_minimum_required(VERSION 3.12...3.20)
project(raptor)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wredundant-decls -Wcast-align -Wshadow")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -funroll-loops")

include(FeatureSummary)

option(CRAYXE "Running on Cray XE" OFF)
option(BGQ "Running on BGQ" OFF)
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
add_feature_info(crayxe CRAYXE "Compile on CrayXE")
add_feature_info(bgq BGQ "Compile on BGQ")
add_feature_info(ptscotch WITH_PTSCOTCH "Enable PTScotch Partitioning")
add_feature_info(parmetis WITH_PARMETIS "Enable ParMetis Partitioning")
add_feature_info(hostfile WITH_HOSTFILE "Enable Hostfile for MPIRUN")

include(options)
include(testing)
include(source)
SET(MPIRUN "")

if (WITH_MPI)
    add_definitions ( -DUSING_MPI )
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    SET(MPIRUN mpirun)
endif (WITH_MPI)

#include_directories("external")
set(raptor_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/raptor)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

SET(HYPRE_DIR "" CACHE STRING "Directory of HYPRE")
SET(MUELU_DIR "" CACHE_STRING "Directory of MueLu")
SET(METIS_DIR "" CACHE STRING "Directory of metis")
SET(MFEM_DIR "" CACHE STRING "Directory of mfem")
SET(MFEM_MESH_DIR "" CACHE STRING "Directory containing MFEM mesh files")
SET(PETSC_DIR "" CACHE STRING "Directory of petsc")
SET(PTSCOTCH_DIR "" CACHE STRING "Directory of Ptscotch")
SET(PARMETIS_DIR "" CACHE STRING "Directory of ParMetis")
SET(HOSTFILE "" CACHE STRING "Set Hostfile")

if (CRAYXE)
    set(EXTERNAL_LIBS "")
    set(MPIRUN aprun)
    set(CMAKE_AR  "gcc-ar-4.7")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_FINISH   true)
elseif (BGQ)
    find_library(LAPACK_LIB NAMES liblapack lapack HINTS
        "/soft/libraries/alcf/current/xl/LAPACK/lib")
    find_library(BLAS_LIB NAMES libblas blas HINTS
        "/soft/libraries/alcf/current/xl/BLAS/lib")
    find_library(ESSL_LIB NAMES libesslbg esslbg HINTS
        "/soft/libraries/essl/current/essl/5.1/lib64")
    find_library(XLF_LIB NAMES libxlf90_r.a xlf90_r HINTS
        "/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/bglib64")
    find_library(XLOPT_LIB NAMES libxlopt.a xlopt HINTS
        "/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/bglib64")
    find_library(XLFMATH_LIB NAMES libxlfmath.a xlfmath HINTS
        "/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/bglib64")
    find_library(XL_LIB NAMES libxl.a xl HINTS
        "/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/bglib64")
    find_library(XLSMP_LIB NAMES libxlomp_ser.a xlomp_ser HINTS
        "/soft/compilers/ibmcmp-may2016/xlsmp/bg/3.1/bglib64")
    set(EXTERNAL_LIBS ${LAPACK_LIB} ${BLAS_LIB} ${ESSL_LIB} ${XLF_LIB}
        ${XLOPT_LIB} ${XLFMATH_LIB} ${XL_LIB} ${XLSMP_LIB})
else()
	find_library(LAPACK_LIB NAMES liblapack.so.3 lapack HINTS "/usr/lib/x86_64-linux-gnu/")
	find_library(BLAS_LIB NAMES libblas.so.3 blas HINTS "/usr/lib/x86_64-linux-gnu/")
	set(EXTERNAL_LIBS ${LAPACK_LIB} ${BLAS_LIB})
endif()

if (WITH_HOSTFILE)
    find_file (FILE_OF_HOST, ${HOSTFILE})
    set(HOST "--hostfile" "${HOSTFILE}")
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
    find_package(MUELU)
    if (MUELU_FOUND)
        include_directories(${MUELU_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Cannot find MueLu. Try setting MUELU_DIR" )
        set(WITH_MUELU OFF)
    endif(MUELU_FOUND)
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
    add_definitions( -DMFEM_MESH_DIR=\"${MFEM_MESH_PATH}\" )
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
