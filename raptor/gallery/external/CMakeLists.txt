# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called gallery_SOURCES containing all .cpp files:
if (WITH_MFEM)
    set(external_mfem_HEADERS
        gallery/external/mfem_wrapper.hpp
        )
    set(external_mfem_SOURCES 
        gallery/external/mfem/mfem_linear_elasticity.cpp
        gallery/external/mfem/mfem_laplacian.cpp
        gallery/external/mfem/mfem_grad_div.cpp
        gallery/external/mfem/mfem_adaptive_laplacian.cpp
        gallery/external/mfem/mfem_dg_diffusion.cpp
        gallery/external/mfem/mfem_dg_elasticity.cpp
       )
endif()

set(external_gallery_HEADERS
    gallery/external/hypre_wrapper.hpp
    ${external_mfem_HEADERS}
    PARENT_SCOPE)
set(external_gallery_SOURCES
    gallery/external/hypre_wrapper.cpp 
    ${external_mfem_SOURCES}
    PARENT_SCOPE)
