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

# MFEM interface: raptor::mfem_wrapper
if (WITH_MFEM)
    add_library(mfem_wrapper
        mfem/mfem_linear_elasticity.cpp
        mfem/mfem_laplacian.cpp
        mfem/mfem_grad_div.cpp
        mfem/mfem_adaptive_laplacian.cpp
        mfem/mfem_dg_diffusion.cpp
        mfem/mfem_dg_elasticity.cpp
        mfem_wrapper.hpp)
    add_library(raptor::mfem_wrapper ALIAS mfem_wrapper)
    set_target_properties(mfem_wrapper PROPERTIES OUTPUT_NAME raptor_mfem_wrapper)
    target_compile_definitions(mfem_wrapper PUBLIC USING_MFEM)
    target_link_libraries(mfem_wrapper PUBLIC raptor::hypre_wrapper mfem)
    install(TARGETS mfem_wrapper EXPORT raptorTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES mfem_wrapper.hpp DESTINATION "include/raptor/external")
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

# Trilinos MueLu interface: raptor::trilinos_wrapper
if (WITH_MUELU)
    add_library(trilinos_wrapper trilinos_wrapper.cpp trilinos_wrapper.hpp)
    add_library(raptor::trilinos_wrapper ALIAS trilinos_wrapper)
    set_target_properties(trilinos_wrapper PROPERTIES OUTPUT_NAME raptor_trilinos_wrapper)
    target_compile_definitions(trilinos_wrapper PUBLIC USING_MUELU)
    target_link_libraries(trilinos_wrapper PUBLIC raptor MueLu::all_libs)
    install(TARGETS trilinos_wrapper EXPORT raptorTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES trilinos_wrapper.hpp DESTINATION "include/raptor/external")
endif()
