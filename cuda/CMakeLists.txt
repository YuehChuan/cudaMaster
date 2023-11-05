cmake_minimum_required(VERSION 3.17)

project(cuda CUDA)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include/tensor)
message(STATUS " ${INCLUDE_DIRECTORIES}")

aux_source_directory(src SOURCES_FILES)

add_executable(test_cublasMatMul2D "test/test_cublasMatMul2D.cu" "include/cublas_utils.cuh" "src/cublas-MatMul2D.cu"  "src/types.cpp")
target_link_libraries(test_cublasMatMul2D ${CUDA_cublas_LIBRARY} )

set_target_properties(test_cublasMatMul2D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_cublasMatMul2D PROPERTIES CUDA_ARCHITECTURES "70")