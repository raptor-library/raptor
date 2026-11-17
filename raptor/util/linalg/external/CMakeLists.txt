# PT-Scotch interface (header only): raptor::ptscotch_wrapper
if (WITH_PTSCOTCH)
    add_library(ptscotch_wrapper INTERFACE)
    add_library(raptor::ptscotch_wrapper ALIAS ptscotch_wrapper)
    target_compile_definitions(ptscotch_wrapper INTERFACE USING_PTSCOTCH)
    target_link_libraries(ptscotch_wrapper INTERFACE raptor raptor::ptscotch)
    install(TARGETS ptscotch_wrapper EXPORT raptorTargets)
    install(FILES ptscotch_wrapper.hpp DESTINATION "include/raptor/util/linalg/external")
endif()

# ParMETIS interface (header only): raptor::parmetis_wrapper
if (WITH_PARMETIS)
    add_library(parmetis_wrapper INTERFACE)
    add_library(raptor::parmetis_wrapper ALIAS parmetis_wrapper)
    target_compile_definitions(parmetis_wrapper INTERFACE USING_PARMETIS)
    target_link_libraries(parmetis_wrapper INTERFACE raptor raptor::parmetis)
    install(TARGETS parmetis_wrapper EXPORT raptorTargets)
    install(FILES parmetis_wrapper.hpp DESTINATION "include/raptor/util/linalg/external")
endif()
