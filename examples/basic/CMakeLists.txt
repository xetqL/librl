cmake_minimum_required(VERSION 3.9)
project(liblj)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

include_directories(/home/xetql/boost_1_66_0)
link_directories(/home/xetql/boost_1_66_0/stage/lib)

add_executable(basic main.cpp)
target_link_libraries(basic armadillo mlpack)
