cmake_minimum_required(VERSION 3.16)
project(sonic_ffmaster_dummy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(aimdk_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(dummy_elbow
  src/main.cc
  src/dummy_node.cc
)
target_include_directories(dummy_elbow PRIVATE include)
ament_target_dependencies(dummy_elbow rclcpp aimdk_msgs)
target_link_libraries(dummy_elbow yaml-cpp)

install(TARGETS dummy_elbow DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
