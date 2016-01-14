# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
add_subdirectory(linalg)

set(util_SOURCES
${linalg_SOURCES}
PARENT_SCOPE)

