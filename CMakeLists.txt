cmake_minimum_required (VERSION 3.0)
project (Raytrace)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

set (CMAKE_CXX_FLAGS "-fopenmp")
add_compile_options (-Wall -Wextra -Wpedantic -Ofast)
add_executable (main raytrace.cpp framework.cpp)
set (CMAKE_CXX_STANDARD 11)

target_link_libraries (main GL glut GLEW pthread)
