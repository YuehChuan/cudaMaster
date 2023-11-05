cmake_minimum_required(VERSION 3.17)

project(cudaMaster CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include/)

find_package(CUDA)
include_directories(${PROJECT_SOURCE_DIR}/include/ ${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/cuda/include)


file(GLOB_RECURSE SOURCE_FILES ./src/*)

set(BUILD_TEST "OFF")
set(CUDA_TEST "ON")

if(BUILD_TEST)
message("BUILD_TEST")

else()
endif()

if(CUDA_TEST)
 message("BUILD_CUDA!!!")
 include_directories(${PROJECT_SOURCE_DIR}/cuda/include)
 add_subdirectory(cuda)
else()
endif()