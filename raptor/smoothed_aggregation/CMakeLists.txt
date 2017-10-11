# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called linalg_SOURCES containing all .cpp files:
set(smoothed_aggregation_SOURCES 
    smoothed_aggregation/seq/strength.cpp
    smoothed_aggregation/seq/aggregate.cpp
    smoothed_aggregation/seq/prolongation.cpp
    smoothed_aggregation/seq/candidates.cpp
    smoothed_aggregation/aggregate.cpp
    smoothed_aggregation/prolongation.hpp
PARENT_SCOPE)




