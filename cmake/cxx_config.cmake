check_for_cxx11_compiler(cxx11)

if (cxx11)
  enable_cxx11()
else()
  message(FAIL_ERROR "A C++11 compiler is required.")
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -flto -fivopts -flax-vector-conversions -funsafe-math-optimizations")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
#SET(CMAKE_AR  "gcc-ar")
#SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
#SET(CMAKE_CXX_ARCHIVE_FINISH   true)

#if (DEFINED ENV{HYPRE_LIB})
#    if (DEFINED ENV{MFEM_LIB} AND DEFINED ENV{METIS_LIB})
#        set(CMAKE_EXE_LINKER_FLAGS "-L$ENV{MFEM_LIB} -lmfem -L$ENV{METIS_LIB} -lmetis")
#    endif()
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L$ENV{HYPRE_LIB} -lHYPRE")
#endif()

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wall -llapack -lblas")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g -std=c++11")

if (NOT CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_definitions(-DDEBUG)
endif()
