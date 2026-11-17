if (ENABLE_UNIT_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        release-1.12.1
  )
  FetchContent_MakeAvailable(googletest)
  add_library(googletest INTERFACE IMPORTED)
  target_link_libraries(googletest INTERFACE gtest_main)

  if(ENABLE_COVERAGE)
    target_compile_options(googletest INTERFACE -O0 -g -fprofile-arcs -ftest-coverage)
    target_link_options(googletest INTERFACE -fprofile-arcs -ftest-coverage)
  endif()
endif()
