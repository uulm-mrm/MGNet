cmake_minimum_required(VERSION 3.16)

project(trt_inference)

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(OpenCV)
# For local installation, substitute PATH below with absolute path to deps/pytorch-install/share/cmake/Torch
find_package(Torch PATHS /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch REQUIRED)

cuda_add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES})