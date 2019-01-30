# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a variable called gallery_SOURCES containing all .cpp files:
if (WITH_MPI)
    set(par_gallery_HEADERS
        gallery/par_stencil.hpp
        gallery/par_random.hpp
        gallery/par_matrix_IO.hpp
        gallery/par_matrix_market.hpp
        )
    set(par_gallery_SOURCES
        gallery/par_matrix_IO.cpp
        gallery/par_matrix_market.cpp
        )
else ()
    set(par_gallery_HEADERS
        ""
        )
    set(par_gallery_SOURCES
        ""
        )
endif()

set(gallery_HEADERS
    gallery/diffusion.hpp
    gallery/laplacian27pt.hpp
    gallery/stencil.hpp
    gallery/random.hpp
    gallery/matrix_IO.hpp
    gallery/matrix_market.hpp
    ${par_gallery_HEADERS}
    PARENT_SCOPE)

set(gallery_SOURCES
    gallery/matrix_IO.cpp
    gallery/matrix_market.cpp
    ${par_gallery_SOURCES}
    PARENT_SCOPE)


