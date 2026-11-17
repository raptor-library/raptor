if (ENABLE_UNIT_TESTS)
  enable_testing()

  include(FetchContent)
  if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  FetchContent_Declare(
      googletest
      URL      https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
      URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
  )
  # Build only GoogleTest
  set(BUILD_GMOCK OFF CACHE INTERNAL "")
  set(INSTALL_GTEST OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(googletest)
  find_package(Threads REQUIRED)
  add_library(googletest INTERFACE IMPORTED)
  target_link_libraries(googletest INTERFACE gtest_main Threads::Threads)

  if(ENABLE_COVERAGE)
    target_compile_options(googletest INTERFACE -O0 -g -fprofile-arcs -ftest-coverage)
    target_link_options(googletest INTERFACE -fprofile-arcs -ftest-coverage)
  endif()
endif()
