cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

PROJECT ("DEPTH_EXAMPLE")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )

set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(CUDA)
find_package(OpenCV REQUIRED)

add_executable(monodepth main.cpp)

target_link_libraries(monodepth ${OpenCV_LIBS} ${TORCH_LIBRARIES})

