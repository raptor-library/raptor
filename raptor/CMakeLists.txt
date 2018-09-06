include_directories(.)

add_subdirectory(core)
add_subdirectory(gallery)
add_subdirectory(util)
add_subdirectory(ruge_stuben)
add_subdirectory(aggregation)
add_subdirectory(multilevel)
add_subdirectory(krylov)

if (WITH_MPI)
    set(par_SOURCES
        par_strength.cpp
        )
else()
    set(par_SOURCES
        )
endif()

add_library(raptor STATIC ${core_SOURCES} ${core_HEADERS} ${gallery_SOURCES}
    ${gallery_HEADERS} ${ext_gallery_HEADERS} ${util_SOURCES} ${util_HEADERS}
    ${par_SOURCES} strength.cpp ${ruge_stuben_SOURCES} ${ruge_stuben_HEADERS}
    ${aggregation_SOURCES} ${aggregation_HEADERS}
    ${multilevel_SOURCES} ${multilevel_HEADERS} ${krylov_SOURCES} ${krylov_HEADERS} )

target_link_libraries(raptor ${MFEM_LIBRARIES} ${METIS_LIBRARIES} ${HYPRE_LIBRARIES}
    ${PTSCOTCH_LIBRARIES} ${EXTERNAL_LIBS} )

install(TARGETS raptor DESTINATION "lib")
install(FILES raptor.hpp DESTINATION "include/raptor")
install(FILES ${core_HEADERS} DESTINATION "include/raptor/core")
install(FILES ${gallery_HEADERS}  DESTINATION "include/raptor/gallery")
install(FILES ${ext_gallery_HEADERS} DESTINATION "include/raptor/gallery/external")
install(FILES ${util_HEADERS} DESTINATION "include/raptor/util/linalg")
install(FILES ${ruge_stuben_HEADERS} DESTINATION "include/raptor/ruge_stuben")
install(FILES ${aggregation_HEADERS} DESTINATION "include/raptor/aggregation")
install(FILES ${multilevel_HEADERS} DESTINATION "include/raptor/multilevel")
install(FILES ${krylov_HEADERS} DESTINATION "include/raptor/krylov")


include_directories(".")
if(ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
    add_subdirectory(core/tests)
    add_subdirectory(util/tests)
    add_subdirectory(ruge_stuben/tests)
    add_subdirectory(aggregation/tests)
    add_subdirectory(multilevel/tests)
    add_subdirectory(gallery/tests)
    add_subdirectory(krylov/tests)
endif()
