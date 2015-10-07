add_src_promisc(core)
add_src_promisc(gallery)
add_src_promisc(util)

add_library(raptor SHARED ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})
#add_library(raptor-shared SHARED ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})
#add_library(raptor-static STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})
#set_target_properties(raptor-shared raptor-static PROPERTIES OUTPUT_NAME raptor)

include_directories(".")
if(ENABLE_UNIT_TESTS)
  add_tests_promisc(core)
  add_tests_promisc(util)
endif()
