# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called linalg_SOURCES containing all .cpp files:
set(ruge_stuben_SOURCES 
    ruge_stuben/cf_splitting.hpp
    ruge_stuben/cf_splitting.cpp
PARENT_SCOPE)




