cmake_minimum_required(VERSION 3.10)

project(MS_ISSM_Feature_Extractor)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ms_feature_cal main.cpp)

target_link_libraries(ms_feature_cal ${PCL_LIBRARIES})