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

add_subdirectory(raptor)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
