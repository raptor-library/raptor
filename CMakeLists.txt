cmake_minimum_required(VERSION 2.8)
project(raptor)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(cxx11)
include(source)
include(cxx_config)

option(BUILD_SHARED_LIBS "Build the shared library" ON)

include_directories("external")
add_subdirectory(raptor)
