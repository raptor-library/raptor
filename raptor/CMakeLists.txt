include_directories(/home/bienz2/Documents/Parallel/hypre/src/hypre/include)

add_src_promisc(core)
add_src_promisc(gallery)
add_src_promisc(util)
add_src_promisc(hypre)

add_library(raptor STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES} ${hypre_SOURCES})

#add_library(raptor-shared SHARED ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})
#add_library(raptor-static STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})
#set_target_properties(raptor-shared raptor-static PROPERTIES OUTPUT_NAME raptor)

include_directories(".")
if(ENABLE_UNIT_TESTS)
  add_tests_promisc(core)
  add_tests_promisc(util)
endif()
