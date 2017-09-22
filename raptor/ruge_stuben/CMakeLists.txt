# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (WITH_MPI)
    set(par_ruge_stuben_SOURCES
        ruge_stuben/par_cf_splitting.hpp
        ruge_stuben/par_cf_splitting.cpp
        ruge_stuben/par_interpolation.hpp
        ruge_stuben/par_interpolation.cpp
        )
else()
    set(par_ruge_stuben_SOURCES
        )
endif()
 
# Create a variable called linalg_SOURCES containing all .cpp files:
set(ruge_stuben_SOURCES 
    ruge_stuben/cf_splitting.hpp
    ruge_stuben/cf_splitting.cpp
    ruge_stuben/interpolation.hpp
    ruge_stuben/interpolation.cpp
    ${par_ruge_stuben_SOURCES}
PARENT_SCOPE)




