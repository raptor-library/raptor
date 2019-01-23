# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called gallery_SOURCES containing all .cpp files:
if (WITH_MFEM)
    set(mfem_HEADERS
        external/mfem_wrapper.hpp
        )
    set(mfem_SOURCES 
        external/mfem/mfem_linear_elasticity.cpp
        external/mfem/mfem_laplacian.cpp
        external/mfem/mfem_grad_div.cpp
        external/mfem/mfem_adaptive_laplacian.cpp
        external/mfem/mfem_dg_diffusion.cpp
        external/mfem/mfem_dg_elasticity.cpp
       )
endif()

if (WITH_HYPRE)
    set(hypre_HEADERS
        external/hypre_wrapper.hpp
        )
    set(hypre_SOURCES
        external/hypre_wrapper.cpp
        )
endif()

if (WITH_PETSC)
    set(petsc_HEADERS
        external/petsc_wrapper.hpp
        )
    set(petsc_SOURCES
        external/petsc_wrapper.cpp
        )
endif()

set(external_HEADERS
    ${mfem_HEADERS}
    ${hypre_HEADERS}
    ${petsc_HEADERS}
    PARENT_SCOPE)
set(external_SOURCES
    ${mfem_SOURCES}
    ${hypre_SOURCES}
    ${petsc_SOURCES}
    PARENT_SCOPE)
