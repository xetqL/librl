cmake_minimum_required(VERSION 3.9)
project(liblj)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_BUILD_TYPE Release)

set(INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/includes)
set(INCLUDES_INSTALL_DIRECTORY ${CMAKE_INSTALL_PREFIX}/include/)

add_library(rl INTERFACE)

target_link_libraries(rl INTERFACE)

install(DIRECTORY includes/
        DESTINATION  ${INCLUDES_INSTALL_DIRECTORY}
        FILES_MATCHING PATTERN "*.hpp"
        PERMISSIONS WORLD_READ)

file(GLOB LIBRL_INC
        "*.hpp"
)

add_executable(tests examples/basic/main.cpp)
target_link_libraries(tests armadillo mlpack)

