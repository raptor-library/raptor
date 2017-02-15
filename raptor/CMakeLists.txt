#if (DEFINED ENV{HYPRE_INC})
#    include_directories($ENV{HYPRE_INC})
#    if (DEFINED ENV{MFEM_INC} AND DEFINED ENV{METIS_INC})
#        include_directories($ENV{MFEM_INC})
#        include_directories($ENV{METIS_INC})
#    endif()
#endif()

include_directories(.)

add_subdirectory(core)
add_subdirectory(gallery)
add_subdirectory(util)

add_library(raptor STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})

if (WITH_AMPI)
#target_link_libraries(raptor ${MFEM_LIBRARIES} ${METIS_LIBRARIES} ${HYPRE_LIBRARIES}
#	${LAPACK_LIBRARY} ${BLAS_LIBRARY} ${ESSL_LIBRARY} "-module CommonLBs -lm")
#target_link_libraries(raptor ${MFEM_LIBRARIES} ${METIS_LIBRARIES} ${HYPRE_LIBRARIES}
#	${LAPACK_LIBRARY} ${BLAS_LIBRARY} "-module CommonLBs -tracemode projections -tracemode summary")
else (WITH_AMPI)
target_link_libraries(raptor ${MFEM_LIBRARIES} ${METIS_LIBRARIES} ${HYPRE_LIBRARIES}
	${EXTERNAL_LIBS} )
endif() 

include_directories(".")
if(ENABLE_UNIT_TESTS)
  add_subdirectory(core/tests)
  add_subdirectory(util/tests)
endif()
