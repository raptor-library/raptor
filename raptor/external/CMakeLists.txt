# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# hypre interface: raptor::hypre_wrapper
if (WITH_HYPRE)
    add_library(hypre_wrapper hypre_wrapper.cpp hypre_wrapper.hpp)
    add_library(raptor::hypre_wrapper ALIAS hypre_wrapper)
    set_target_properties(hypre_wrapper PROPERTIES OUTPUT_NAME raptor_hypre_wrapper)
    target_compile_definitions(hypre_wrapper PUBLIC USING_HYPRE)
    target_link_libraries(hypre_wrapper PUBLIC raptor raptor::hypre)
    install(TARGETS hypre_wrapper EXPORT raptorTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES hypre_wrapper.hpp DESTINATION "include/raptor/external")
endif()

# MFEM test problems for RAPtor's tests and examples (not installed)
if (WITH_MFEM)
    add_library(raptor_mfem
        mfem/mfem_linear_elasticity.cpp
        mfem/mfem_laplacian.cpp
        mfem/mfem_grad_div.cpp
        mfem/mfem_adaptive_laplacian.cpp
        mfem/mfem_dg_diffusion.cpp
        mfem/mfem_dg_elasticity.cpp
        mfem_wrapper.hpp)
    target_compile_definitions(raptor_mfem PUBLIC USING_MFEM)
    target_link_libraries(raptor_mfem PUBLIC raptor::hypre_wrapper mfem)
endif()

# PETSc interface: raptor::petsc_wrapper
if (WITH_PETSC)
    add_library(petsc_wrapper petsc_wrapper.cpp petsc_wrapper.hpp)
    add_library(raptor::petsc_wrapper ALIAS petsc_wrapper)
    set_target_properties(petsc_wrapper PROPERTIES OUTPUT_NAME raptor_petsc_wrapper)
    target_compile_definitions(petsc_wrapper PUBLIC USING_PETSC)
    target_link_libraries(petsc_wrapper PUBLIC raptor raptor::petsc)
    install(TARGETS petsc_wrapper EXPORT raptorTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES petsc_wrapper.hpp DESTINATION "include/raptor/external")
endif()

if (WITH_MUELU)
    set(trilinos_HEADERS
        external/trilinos_wrapper.hpp
        )
    set(trilinos_SOURCES
        external/trilinos_wrapper.cpp
        )
endif()

set(external_HEADERS
    ${trilinos_HEADERS}
    PARENT_SCOPE)
set(external_SOURCES
    ${trilinos_SOURCES}
    PARENT_SCOPE)
