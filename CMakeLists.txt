cmake_minimum_required(VERSION 2.8)
project(raptor)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(FeatureSummary)

option(CRAYXE "Running on Cray XE" OFF)
option(BGQ "Running on BGQ" OFF)
option(WITH_HYPRE "Add Hypre" OFF)
option(WITH_MFEM "Add MFEM" OFF)
option(WITH_AMPI "Using AMPI" OFF)
option(WITH_MPI "Using MPI" ON)
add_feature_info(hypre WITH_HYPRE "Hypre preconditioner")
add_feature_info(mfem WITH_MFEM "MFEM matrix gallery")
add_feature_info(ampi WITH_AMPI "Compile with AMPI")
add_feature_info(crayxe CRAYXE "Compile on CrayXE")
add_feature_info(bgq BGQ "Compile on BGQ")
add_feature_info(mpi WITH_MPI "Compile with MPI")

include(options)
include(testing)
include(cxx11)
include(source)
include(cxx_config)

if (WITH_MPI)
    include(mpi_config)
endif (WITH_MPI)

include_directories("external")
set(raptor_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/raptor)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

SET(HYPRE_DIR "" CACHE STRING "Directory of HYPRE")
SET(METIS_DIR "" CACHE STRING "Directory of metis")
SET(MFEM_DIR "" CACHE STRING "Directory of mfem")

if (CRAYXE)
    set(EXTERNAL_LIBS "")
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
    find_library(LAPACK_LIB NAMES liblapack lapack)
    find_library(BLAS_LIB NAMES libblas blas)
    set(EXTERNAL_LIBS ${LAPACK_LIB} ${BLAS_LIB})
endif()


if(WITH_HYPRE)
    find_package(Hypre)
    if(HYPRE_FOUND)
        include_directories(${HYPRE_INCLUDE_DIRS})
	else()
	    message(FATAL_ERROR "Cannot find HYPRE. Try setting HYPRE_DIR" )
		set(WITH_HYPRE OFF)
    endif(HYPRE_FOUND)
endif(WITH_HYPRE)

if(WITH_MFEM)
	set(WITH_HYPRE ON)
	if(NOT HYPRE_FOUND)
		find_package(Hypre)
		if(HYPRE_FOUND)
			include_directories(${HYPRE_INCLUDE_DIRS})
		else()
			message(FATAL_ERROR "Cannot find HYPRE (required for MFEM). Try setting HYPRE_DIR.")
		endif(HYPRE_FOUND)
	endif(NOT HYPRE_FOUND)

	find_package(Metis)
	if(METIS_FOUND)
		include_directories(${METIS_INCLUDE_DIRS})
	else()
		message(FATAL_ERROR "Cannot find metis (required for MFEM). Try setting METIS_DIR.")
	endif(METIS_FOUND)

	find_package(Mfem)
	if (MFEM_FOUND)
		include_directories(${MFEM_INCLUDE_DIRS})
	else()
		message(FATAL_ERROR "Cannot find mfem. Try setting MFEM_DIR.")	
	endif(MFEM_FOUND)
endif(WITH_MFEM)

if (WITH_AMPI)
	add_definitions(-DUSE_AMPI)
endif(WITH_AMPI)


add_subdirectory(raptor)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
