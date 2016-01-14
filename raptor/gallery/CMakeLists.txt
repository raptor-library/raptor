# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a variable called gallery_SOURCES containing all .cpp files:
if(WITH_HYPRE)
    add_subdirectory(external)
endif(WITH_HYPRE)

set(gallery_SOURCES
    gallery/diffusion.cpp
    gallery/matrix_IO.cpp
    gallery/stencil.cpp
    ${external_gallery_SOURCES}
    PARENT_SCOPE)


