add_src_promisc(core)
add_library(raptor ${core_SOURCES})
include_directories(".")
if(ENABLE_UNIT_TESTS)
  add_tests_promisc(core)
  add_tests_promisc(util)
endif()