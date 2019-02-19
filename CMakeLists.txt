cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet-libtorch)

find_package(Torch REQUIRED)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(resnet_libtorch use_resnet.cpp)
target_link_libraries(resnet_libtorch "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET resnet_libtorch PROPERTY CXX_STANDARD 11)

