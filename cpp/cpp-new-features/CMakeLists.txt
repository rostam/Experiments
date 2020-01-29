cmake_minimum_required(VERSION 3.13)
project(CPP17)

set(CMAKE_CXX_STANDARD 17)

add_executable(CPP17 cpp20.cpp)
target_link_libraries(CPP17 blas)