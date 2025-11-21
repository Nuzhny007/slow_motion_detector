cmake_minimum_required(VERSION 3.14)
project(very_slow_track)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -MMD -Wall -Wextra -Winit-self")

find_package(OpenCV REQUIRED)

find_package(MTTracking REQUIRED)
message("MTTracking_INCLUDE_DIR: ${MTTracking_INCLUDE_DIR}")

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${MTTracking_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} MTTracking::mdetection MTTracking::mtracking)
