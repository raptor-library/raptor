cmake_minimum_required(VERSION 2.6)
project(HPCMG)

add_subdirectory(src/test/gtest-1.7.0)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
function(add_mpi_test name no_mpi_proc)
    #target_link_libraries(${name} )
  set(test_parameters ${MPIEXEC_NUMPROC_FLAG} ${no_mpi_proc} "./${name}")
      add_test(NAME ${name} COMMAND ${MPIEXEC} ${test_parameters})
endfunction(add_mpi_test)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "-O2 -g -Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g -Wall -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -std=c++11")
add_subdirectory(src)
