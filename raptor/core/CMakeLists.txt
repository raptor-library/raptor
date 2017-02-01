# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called core_SOURCES containing all .cpp files:
set(core_SOURCES 
core/matrix.cpp
core/vector.cpp
core/par_matrix.cpp 
core/par_vector.cpp 
core/par_comm.cpp 
#core/hierarchy.cpp
PARENT_SCOPE)

