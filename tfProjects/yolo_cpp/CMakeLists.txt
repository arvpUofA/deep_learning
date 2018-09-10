cmake_minimum_required(VERSION 3.5)
project(yolo_cpp)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp utils.cpp utils.h)
add_executable(yolo_cpp ${SOURCE_FILES})

# OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_cpp ${OpenCV_LIBS})


# ==================== PATHS TO SPECIFY! ==================== #

# Eigen lib headers
include_directories("/usr/local/include/eigen3")

# TensorFlow headers
include_directories("/usr/local/include/tf/")
include_directories("/usr/local/include/tf/bazel-genfiles/")
include_directories("/usr/local/include/tf/tensorflow/")
include_directories("/usr/local/include/tf/third-party/")

# Link TensorFlow libs
target_link_libraries(yolo_cpp "/usr/local/lib/libtensorflow_cc.so")
target_link_libraries(yolo_cpp "/usr/local/lib/libtensorflow_framework.so")
