add_src_promisc(core)
add_src_promisc(gallery)
add_src_promisc(util)
add_library(raptor ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})


include_directories(".")
if(ENABLE_UNIT_TESTS)
  add_tests_promisc(core)
  add_tests_promisc(util)
endif()
