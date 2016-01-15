# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called gallery_SOURCES containing all .cpp files:
if (WITH_MFEM)
    set(external_gallery_SOURCES 
        gallery/external/mfem/mfem_laplace.cpp
        gallery/external/mfem/mfem_linear_elasticity.cpp
        gallery/external/mfem/mfem_electromagnetic_diffusion.cpp
        gallery/external/mfem/mfem_hdiv_diffusion.cpp
        gallery/external/mfem/mfem_isoparametric_laplace.cpp)
endif()

set(external_gallery_SOURCES
    ${external_gallery_SOURCES}
    gallery/external/hypre_wrapper.cpp 
    PARENT_SCOPE)

