cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(tgcat VERSION 1.0.0)

set(CMAKE_CXX_FLAGS " -pthread -std=c++17 -funroll-loops -O3 -march=native")
add_subdirectory(text_lightgbm)
include_directories("text_lightgbm/src")

add_library(tgcat SHARED tgcat.cpp preprocessor.cpp predictor.cpp)
