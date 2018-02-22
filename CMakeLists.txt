cmake_minimum_required(VERSION 3.8)
project(liblj CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)
set(TEST_SOURCE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
set(INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/includes)

set(INCLUDES_INSTALL_DIRECTORY /usr/include/librl)

add_library(rl INTERFACE)

install(DIRECTORY includes/
        DESTINATION /usr/include/
        FILES_MATCHING PATTERN "*.hpp"
        PERMISSIONS WORLD_READ WORLD_WRITE)

add_custom_target(gridworld
        COMMAND g++ -std=c++17 examples/gridworld/*.cpp -o bin/gridworld -Wall -std=c++14 -L/lib
)
