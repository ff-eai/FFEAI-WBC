cmake_minimum_required(VERSION 3.16)
project(sonic_ffmaster_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(aimdk_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# unitree_sdk2 lives in the deploy repo's thirdparty (prebuilt x86_64 + aarch64).
# Same import pattern as gear_sonic_deploy/CMakeLists.txt:227-240.
if(NOT DEFINED UNITREE_SDK2_DIR)
  # REALPATH first: workspaces often symlink this package, and the relative
  # hop to gear_sonic_deploy/thirdparty must be taken from the real location.
  get_filename_component(_pkg_real "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
  get_filename_component(UNITREE_SDK2_DIR
    "${_pkg_real}/../../../thirdparty/unitree_sdk2" ABSOLUTE)
endif()
if(NOT EXISTS "${UNITREE_SDK2_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "unitree_sdk2 not found at ${UNITREE_SDK2_DIR} — pass -DUNITREE_SDK2_DIR=...")
endif()
set(BUILD_EXAMPLES OFF CACHE BOOL "Build unitree_sdk2 examples" FORCE)
add_subdirectory(${UNITREE_SDK2_DIR} unitree_sdk2_build)

add_executable(bridge
  src/main.cc
  src/bridge_node.cc
)
target_include_directories(bridge PRIVATE include ${UNITREE_SDK2_DIR}/include)
ament_target_dependencies(bridge rclcpp aimdk_msgs sensor_msgs)
target_link_libraries(bridge unitree_sdk2 yaml-cpp pthread)

# libddscxx/libddsc live in the sdk's per-arch thirdparty dir; bake an RPATH so
# the bridge runs without LD_LIBRARY_PATH gymnastics on either host.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(_sdk_arch aarch64)
else()
  set(_sdk_arch x86_64)
endif()
set_target_properties(bridge PROPERTIES
  BUILD_RPATH   "${UNITREE_SDK2_DIR}/thirdparty/lib/${_sdk_arch}"
  INSTALL_RPATH "${UNITREE_SDK2_DIR}/thirdparty/lib/${_sdk_arch}")
# CRITICAL: force DT_RPATH (not DT_RUNPATH). Sourcing ROS puts /opt/ros/humble/lib
# on LD_LIBRARY_PATH, whose libddsc.so.0 (rmw_cyclonedds's) shadows the SDK's and
# heap-corrupts the unitree side (observed: crash in dds_create_topic_impl from
# the ROS libddsc). DT_RPATH is searched BEFORE LD_LIBRARY_PATH; DT_RUNPATH after.
# Also: never run this bridge with RMW_IMPLEMENTATION=rmw_cyclonedds_cpp — the
# ROS side must stay on FastDDS (Humble default) so only unitree loads libddsc.
target_link_options(bridge PRIVATE -Wl,--disable-new-dtags)

install(TARGETS bridge DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
