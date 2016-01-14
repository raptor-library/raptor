if (DEFINED ENV{HYPRE_INC})
    include_directories($ENV{HYPRE_INC})
    if (DEFINED ENV{MFEM_INC} AND DEFINED ENV{METIS_INC})
        include_directories($ENV{MFEM_INC})
        include_directories($ENV{METIS_INC})
    endif()
endif()

include_directories(.)

add_subdirectory(core)
add_subdirectory(gallery)
add_subdirectory(util)

add_library(raptor STATIC ${core_SOURCES} ${gallery_SOURCES} ${util_SOURCES})

include_directories(".")
if(ENABLE_UNIT_TESTS)
  add_tests_promisc(core)
  add_tests_promisc(util)
endif()
