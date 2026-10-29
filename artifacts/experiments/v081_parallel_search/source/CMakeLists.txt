cmake_minimum_required(VERSION 3.20)

project(
    HybridRetrievalCore
    VERSION 0.2.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
    retrieval_core
    STATIC
    src/search_engine.cpp
    src/corpus_loader.cpp
)

target_include_directories(
    retrieval_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    retrieval_core
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(
    test_search_engine
    tests/test_search_engine.cpp
)

target_link_libraries(
    test_search_engine
    PRIVATE
    retrieval_core
)

add_executable(
    benchmark_index
    benchmarks/benchmark_index.cpp
)

target_link_libraries(
    benchmark_index
    PRIVATE
    retrieval_core
)

enable_testing()

add_test(
    NAME SearchEngineTests
    COMMAND test_search_engine
)

add_executable(
    test_bm25_reference
    tests/test_bm25_reference.cpp
)

target_link_libraries(
    test_bm25_reference
    PRIVATE
    retrieval_core
)

add_test(
    NAME BM25ReferenceTests
    COMMAND test_bm25_reference
)

add_executable(
    test_dense_accumulator
    tests/test_dense_accumulator.cpp
)

target_link_libraries(
    test_dense_accumulator
    PRIVATE
    retrieval_core
)

add_test(
    NAME DenseAccumulatorTests
    COMMAND test_dense_accumulator
)

add_executable(
    benchmark_dense
    benchmarks/benchmark_dense.cpp
)

target_link_libraries(
    benchmark_dense
    PRIVATE
    retrieval_core
)

# --------------------------------------------------
# Python bindings (pybind11)
# --------------------------------------------------

option(
    BUILD_PYTHON_BINDINGS
    "Build Python bindings"
    OFF
)

if(BUILD_PYTHON_BINDINGS)
    find_package(
        Python
        COMPONENTS Interpreter Development.Module
        REQUIRED
    )

    find_package(pybind11 CONFIG REQUIRED)

    set_target_properties(
        retrieval_core
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    pybind11_add_module(
        retrieval_cpp
        python/bindings.cpp
    )

    target_link_libraries(
        retrieval_cpp
        PRIVATE
        retrieval_core
    )

    set_target_properties(
        retrieval_cpp
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/python_extension"
    )
endif()
