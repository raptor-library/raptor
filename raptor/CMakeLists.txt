include_directories(.)

add_subdirectory(core)
add_subdirectory(gallery)
add_subdirectory(util)
add_subdirectory(smoothed_aggregation)
add_subdirectory(ruge_stuben)

#add_library(raptor STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES}
#    strength.cpp ${smoothed_aggregation_SOURCES} ${ruge_stuben_SOURCES})

if (WITH_MPI)
    set(par_SOURCES
        par_strength.cpp
        )
else()
    set(par_SOURCES
        )
endif()

add_library(raptor STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES}
    ${par_SOURCES} strength.cpp ${ruge_stuben_SOURCES})

target_link_libraries(raptor ${MFEM_LIBRARIES} ${METIS_LIBRARIES} ${HYPRE_LIBRARIES}
	${EXTERNAL_LIBS} )

include_directories(".")
if(ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
    add_subdirectory(core/tests)
    add_subdirectory(util/tests)
    #add_subdirectory(smoothed_aggregation/tests)
    add_subdirectory(ruge_stuben/tests)
    #add_subdirectory(multilevel/tests)
    add_subdirectory(gallery/tests)
endif()
