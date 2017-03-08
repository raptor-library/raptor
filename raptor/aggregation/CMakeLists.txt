# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called linalg_SOURCES containing all .cpp files:
set(aggregation_SOURCES 
    aggregation/strength.cpp
    aggregation/aggregate.cpp
    aggregation/prolongation.cpp
PARENT_SCOPE)




