cmake_minimum_required(VERSION 3.15)
project(cgal_test)

set(CMAKE_CXX_STANDARD 17)
find_package(CGAL)
add_executable(cgal_test main.cpp)
target_link_libraries(cgal_test CGAL::CGAL)