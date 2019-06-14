cmake_minimum_required(VERSION 3.5)
project(ukf)

#=============
# Setup
#=============

# Default to C99
if (NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif ()

# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif ()

# Find packages
find_package(ament_cmake REQUIRED)
find_package(Eigen3)

# Local includes
include_directories(
  include
)

#=============
# Test
#=============

add_executable(
  ukf_test
  src/ukf_test.cpp
  src/ukf.cpp
)

ament_target_dependencies(
  ukf_test
)

#=============
# Install
#=============

# Install targets
install(
  TARGETS ukf_test
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()