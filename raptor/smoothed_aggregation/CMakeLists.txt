# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called linalg_SOURCES containing all .cpp files:
set(smoothed_aggregation_SOURCES 
    aggregation/seq/strength.cpp
    aggregation/seq/aggregate.cpp
    aggregation/seq/prolongation.cpp
    aggregation/seq/candidates.cpp
    aggregation/aggregate.cpp
    aggregation/prolongation.hpp
PARENT_SCOPE)




