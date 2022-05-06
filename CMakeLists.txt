cmake_minimum_required(VERSION 3.16.3)

project( MyTLD )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_compile_options(-std=c++17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB MY_SOURCES "src/*.cpp" "src/*.h")
add_executable(my_tld ${MY_SOURCES})
# add_executable( my_tld src/Main.cpp
#     src/TLD.cpp src/TLD.h
#     src/MedianFlowTracker.cpp src/MedianFlowTracker.h
#     src/CascadeClassifier.cpp src/CascadeClassifier.h
#     src/EnsembleClassifier.cpp src/EnsembleClassifier.h
#     src/Utils.cpp src/Utils.h
#     src/ObjectModel.cpp src/ObjectModel.h
#     src/Params.cpp src/Params.h
#   )
target_link_libraries( my_tld ${OpenCV_LIBS} )