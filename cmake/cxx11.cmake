macro(check_for_cxx11_compiler _CXX11CHECK)

    message(STATUS "Checking for C++11 compiler")

    set(${_CXX11CHECK})

    if((MSVC AND (MSVC10 OR MSVC11 OR MSVC12)) OR
        (CMAKE_COMPILER_IS_GNUCXX AND
            NOT ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 4.6) OR
        (CMAKE_CXX_COMPILER_ID STREQUAL "Intel" AND
            NOT ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 12.10) OR
        (CMAKE_CXX_COMPILER_ID STREQUAL "PGI" AND
            NOT ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 14.3) OR
        (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
            NOT ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.1))

        set(${_CXX11CHECK} 1)
        message(STATUS "Checking for C++11 compiler - yes")

    else()

        message(STATUS "Checking for C++11 compiler - no")

    endif()

endmacro()

macro(enable_cxx11)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11")
    endif(NOT CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
endmacro()
