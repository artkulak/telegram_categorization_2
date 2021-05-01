cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
OPTION(USE_OPENMP "Enable OpenMP" ON)

project(_text_lightgbm VERSION 1.0.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

if(USE_DEBUG)
    ADD_DEFINITIONS(-DDEBUG)
endif(USE_DEBUG)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(USE_OPENMP)

add_library(_lightgbm SHARED IMPORTED)

set_target_properties(_lightgbm PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../../resources/LightGBM/include/"
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../resources/LightGBM/lib_lightgbm.so"
)

add_library(_omt_tokenizer SHARED IMPORTED)
set_target_properties(_omt_tokenizer PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../../resources/Tokenizer/include"
    INTERFACE_INCLUDE_EXPORT "${CMAKE_SOURCE_DIR}/../../resources/Tokenizer/build/onmt"
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../resources/Tokenizer/build/libOpenNMTTokenizer.so"
)

add_library(_text_lightgbm SHARED application.cpp tfidf_vectorizer.cc)
add_executable(text_lightgbm main.cpp application.cpp tfidf_vectorizer.cc)

target_link_libraries(_text_lightgbm _lightgbm _omt_tokenizer)

target_link_libraries(text_lightgbm _lightgbm _omt_tokenizer)

