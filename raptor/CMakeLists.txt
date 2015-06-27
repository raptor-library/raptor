add_src_promisc(core)

include_directories("include")
add_library(raptor ${core_sources} ${tests_sources})
