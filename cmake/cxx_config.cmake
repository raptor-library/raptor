check_for_cxx11_compiler(cxx11)

if (cxx11)
  enable_cxx11()
else()
  message(FAIL_ERROR "A C++11 compiler is required.")
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fivopts -flax-vector-conversions -funsafe-math-optimizations -Wall -dynamic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g")

if (NOT CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_definitions(-DDEBUG)
endif()
