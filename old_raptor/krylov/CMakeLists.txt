# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (WITH_MPI)
    set(par_krylov_SOURCES
        krylov/par_cg.cpp
        krylov/par_cg_timed.cpp
        krylov/par_cg_openmp.cpp
	krylov/par_bicgstab.cpp
	krylov/partial_inner.cpp
        )
    set(par_krylov_HEADERS
        krylov/par_cg.hpp
        krylov/par_cg_timed.hpp
        krylov/par_cg_openmp.hpp
	krylov/par_bicgstab.hpp
	krylov/partial_inner.hpp
        )
else()
    set(par_krylov_SOURCES
        )
    set(par_krylov_HEADERS
        )
endif()
 
set(krylov_HEADERS
    krylov/cg.hpp
    krylov/bicgstab.hpp
    ${par_krylov_HEADERS}
    PARENT_SCOPE)
# Create a variable called linalg_SOURCES containing all .cpp files:
set(krylov_SOURCES 
    krylov/cg.cpp
    krylov/bicgstab.cpp
    ${par_krylov_SOURCES}
PARENT_SCOPE)


