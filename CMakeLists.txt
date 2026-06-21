cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(quickImage LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(quickimage 
    main.cpp 
    kernel.cu
)

# Find OpenCV Installation
find_package(OpenCV REQUIRED)

# Include OpenCV's header directories
target_include_directories(quickimage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link OpenCV libraries to executable
target_link_libraries(quickimage ${OpenCV_LIBS})

# This line is updated to include the CUDA Toolkit headers so g++ can find them!
target_include_directories(quickimage PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
