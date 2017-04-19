# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a variable called gallery_SOURCES containing all .cpp files:
if(WITH_HYPRE)
    add_subdirectory(external)
endif(WITH_HYPRE)

set(gallery_SOURCES
    gallery/diffusion.hpp
    gallery/stencil.hpp
    gallery/random.hpp
    gallery/par_stencil.hpp
    gallery/par_random.hpp
    gallery/exxon_reader.hpp
    ${external_gallery_SOURCES}
    PARENT_SCOPE)


