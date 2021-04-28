cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
OPTION(USE_OPENMP "Enable OpenMP" ON)

project(tgcat VERSION 1.0.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

if(USE_DEBUG)
    ADD_DEFINITIONS(-DDEBUG)
endif(USE_DEBUG)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(USE_OPENMP)


add_library(_lightgbm SHARED IMPORTED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/resources/LightGBM/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/resources/Tokenizer/include)

add_executable(tgcat text_lightgbm/src/main.cpp text_lightgbm/src/application.cpp text_lightgbm/src/tfidf_vectorizer.cc)
target_link_libraries(tgcat "${CMAKE_SOURCE_DIR}/resources/LightGBM/lib_lightgbm.so")
target_link_libraries(tgcat "${CMAKE_SOURCE_DIR}/resources/Tokenizer/build/libOpenNMTTokenizer.so")

