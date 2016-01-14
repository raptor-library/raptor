# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called linalg_SOURCES containing all .cpp files:
set(linalg_SOURCES 
util/linalg/spmv.cpp
util/linalg/relax.cpp
util/linalg/gauss_elimination.cpp 
PARENT_SCOPE)




