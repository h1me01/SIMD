cmake_minimum_required(VERSION 3.27)
project(SIMD)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_executable(SIMD main.cpp
        dotProduct.h
        adam.h
        gradientsUpdate.h
        neuron.h
        common.h)
