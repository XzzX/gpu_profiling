cmake_minimum_required(VERSION 3.16)

project(kokkos-example LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)
add_executable(MMM
        MMM.cpp
        )
    target_link_libraries(MMM
        PUBLIC Kokkos::kokkos
        )
