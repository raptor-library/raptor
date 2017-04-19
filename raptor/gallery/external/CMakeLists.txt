# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called gallery_SOURCES containing all .cpp files:
if (WITH_MFEM)
    #set(external_mfem_SOURCES 
    #            gallery/external/mfem/mfem_laplace.cpp
    #            gallery/external/mfem/mfem_linear_elasticity.cpp
    #            gallery/external/mfem/mfem_darcy.cpp
    #            )
endif()

set(external_gallery_SOURCES
    gallery/external/hypre_wrapper.hpp
    gallery/external/hypre_wrapper.cpp 
    #    ${external_mfem_SOURCES}
    PARENT_SCOPE)

