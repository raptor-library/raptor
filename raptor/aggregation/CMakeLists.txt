# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (WITH_MPI)
    set(par_aggregation_HEADERS
        aggregation/par_mis.hpp
        aggregation/par_aggregate.hpp
        aggregation/par_candidates.hpp
        aggregation/par_prolongation.hpp
        )
    set(par_aggregation_SOURCES
        aggregation/par_mis.cpp
        aggregation/par_aggregate.cpp
        aggregation/par_candidates.cpp
        aggregation/par_prolongation.cpp
        )
else()
    set(par_aggregation_SOURCES
        )
    set(par_aggregation_HEADERS
        )
endif()
 
# Create a variable called linalg_SOURCES containing all .cpp files:
set(aggregation_HEADERS
    aggregation/mis.hpp
    aggregation/aggregate.hpp
    aggregation/candidates.hpp
    aggregation/prolongation.hpp
    ${par_aggregation_HEADERS}
    PARENT_SCOPE)
set(aggregation_SOURCES 
    aggregation/mis.cpp
    aggregation/aggregate.cpp
    aggregation/candidates.cpp
    aggregation/prolongation.cpp
    ${par_aggregation_SOURCES}
    PARENT_SCOPE)




