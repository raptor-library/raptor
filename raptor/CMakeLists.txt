add_src_promisc(core)
add_src_promisc(gallery)
add_src_promisc(util)

add_library(raptor STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})

include_directories(".")
if(ENABLE_UNIT_TESTS)
  add_subdirectory(core/tests)
endif()
