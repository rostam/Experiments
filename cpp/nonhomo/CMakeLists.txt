cmake_minimum_required(VERSION 3.8)
project(nonhomo)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp homsearch_lib.cpp)
add_executable(nonhomo ${SOURCE_FILES})