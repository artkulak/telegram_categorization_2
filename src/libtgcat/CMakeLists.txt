cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(tgcat VERSION 1.0.0)


add_library(fasttext SHARED IMPORTED)
set_target_properties(fasttext PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../../resources/fastText/src/"
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../resources/fastText/build/libfasttext.so"
)

ADD_SUBDIRECTORY(text_lightgbm)

add_library(tgcat SHARED tgcat.cpp preprocessor.cpp predictor.cpp)
target_link_libraries(tgcat _text_lightgbm fasttext)

