add_subdirectory(core)
add_subdirectory(gallery)
add_subdirectory(util)
add_subdirectory(ruge_stuben)
add_subdirectory(aggregation)
add_subdirectory(multilevel)
add_subdirectory(krylov)
add_subdirectory(profiling)
add_subdirectory(external)

if (WITH_MPI)
    set(par_SOURCES
        par_strength.cpp
        )
else()
    set(par_SOURCES
        )
endif()

add_library(raptor ${core_SOURCES} ${core_HEADERS} 
	${gallery_SOURCES} ${gallery_HEADERS} ${ext_gallery_HEADERS} 
	${util_SOURCES} ${util_HEADERS} ${root_util_HEADERS}
	${par_SOURCES} strength.cpp 
	${ruge_stuben_SOURCES} ${ruge_stuben_HEADERS}
        ${aggregation_SOURCES} ${aggregation_HEADERS}
        ${multilevel_SOURCES} ${multilevel_HEADERS} 
        ${krylov_SOURCES} ${krylov_HEADERS} 
        ${profile_SOURCES} ${profile_HEADERS}
        ${external_SOURCES} ${external_HEADERS})
add_library(raptor::raptor ALIAS raptor)

if (WITH_MFEM)
    target_compile_definitions(raptor PUBLIC USING_MFEM)
    target_link_libraries(raptor PUBLIC mfem)
endif()

if (WITH_HYPRE)
    target_compile_definitions(raptor PUBLIC USING_HYPRE)
    target_include_directories(raptor PUBLIC ${HYPRE_INCLUDE_DIRS})
    target_link_libraries(raptor PUBLIC ${HYPRE_LIBRARIES})
endif()

if (WITH_PETSC)
    target_compile_definitions(raptor PUBLIC USING_PETSC)
    target_include_directories(raptor PUBLIC ${PETSC_INCLUDE_DIR} ${PETSC_INCLUDE_CONF})
    target_link_libraries(raptor PUBLIC ${PETSC_LIBRARIES})
endif()

if (WITH_PTSCOTCH)
    target_compile_definitions(raptor PUBLIC USING_PTSCOTCH)
    target_include_directories(raptor PUBLIC ${PTSCOTCH_INCLUDE_DIRS})
    target_link_libraries(raptor PUBLIC ${PTSCOTCH_LIBRARIES})
endif()

if (WITH_PARMETIS)
    target_compile_definitions(raptor PUBLIC USING_PARMETIS)
    target_include_directories(raptor PUBLIC ${PARMETIS_INCLUDE_DIRS})
    target_link_libraries(raptor PUBLIC ${PARMETIS_LIBRARIES})
endif()

if (WITH_MUELU)
    target_compile_definitions(raptor PUBLIC USING_MUELU)
    target_link_libraries(raptor PUBLIC MueLu::all_libs)
endif()

if (WITH_AMPI)
    target_compile_definitions(raptor PUBLIC USE_AMPI)
endif()

# BLAS and LAPACK
target_link_libraries(raptor PUBLIC ${EXTERNAL_LIBS})

if (WITH_MPI)
    target_compile_definitions(raptor PUBLIC USING_MPI OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
    target_link_libraries(raptor PUBLIC MPI::MPI_CXX)
endif()

if (ENABLE_COVERAGE)
    target_compile_options(raptor PRIVATE -O0 -g --coverage)
    target_link_options(raptor PRIVATE --coverage)
endif()

target_include_directories(raptor
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

include(GNUInstallDirs)
install(TARGETS raptor
  EXPORT raptorTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES raptor.hpp DESTINATION "include/raptor")
install(FILES ${core_HEADERS} DESTINATION "include/raptor/core")
install(FILES ${gallery_HEADERS}  DESTINATION "include/raptor/gallery")
install(FILES ${ext_gallery_HEADERS} DESTINATION "include/raptor/gallery/external")
install(FILES ${util_HEADERS} DESTINATION "include/raptor/util/linalg")
install(FILES ${root_util_HEADERS} DESTINATION "include/raptor/util")
install(FILES ${ext_util_HEADERS} DESTINATION "include/raptor/util/linalg/external")
install(FILES ${ruge_stuben_HEADERS} DESTINATION "include/raptor/ruge_stuben")
install(FILES ${aggregation_HEADERS} DESTINATION "include/raptor/aggregation")
install(FILES ${multilevel_HEADERS} DESTINATION "include/raptor/multilevel")
install(FILES ${krylov_HEADERS} DESTINATION "include/raptor/krylov")
install(FILES ${profile_HEADERS} DESTINATION "include/raptor/profiling")
install(FILES ${external_HEADERS} DESTINATION "include/raptor/external")
install(EXPORT raptorTargets
  FILE raptorTargets.cmake
  NAMESPACE "raptor::"
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/raptor)
export(EXPORT raptorTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/raptorTargets.cmake
  NAMESPACE "raptor::")

include(CMakePackageConfigHelpers)
configure_package_config_file(${PROJECT_SOURCE_DIR}/config/raptorConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/raptorConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/raptor)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/raptorConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/raptorConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/raptorConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/raptor)

if(ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
    add_subdirectory(core/tests)
    add_subdirectory(util/tests)
    add_subdirectory(ruge_stuben/tests)
    add_subdirectory(aggregation/tests)
    add_subdirectory(multilevel/tests)
    add_subdirectory(gallery/tests)
    add_subdirectory(krylov/tests)
    add_subdirectory(external/tests)
endif()
