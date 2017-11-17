include_directories(.)

add_subdirectory(core)
add_subdirectory(gallery)
add_subdirectory(util)
add_subdirectory(ruge_stuben)
add_subdirectory(multilevel)

if (WITH_MPI)
    set(par_SOURCES
        par_strength.cpp
        )
else()
    set(par_SOURCES
        )
endif()

add_library(raptor STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES}
    ${par_SOURCES} strength.cpp ${ruge_stuben_SOURCES} ${multilevel_SOURCES})

target_link_libraries(raptor ${MFEM_LIBRARIES} ${METIS_LIBRARIES} ${HYPRE_LIBRARIES}
    ${PTSCOTCH_LIBRARIES} ${EXTERNAL_LIBS} )

set_target_properties(raptor PROPERTIES PUBLIC_HEADER "raptor.hpp")
INSTALL(TARGETS raptor DESTINATION "lib")

include_directories(".")
if(ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
    add_subdirectory(core/tests)
    add_subdirectory(util/tests)
    add_subdirectory(ruge_stuben/tests)
    add_subdirectory(multilevel/tests)
    add_subdirectory(gallery/tests)
endif()
