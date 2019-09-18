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
# Library
#=============

add_library(
  ukf SHARED
  src/ukf.cpp
)
ament_export_libraries(ukf)

#=============
# Test
#=============

add_executable(
  ukf_test
  src/ukf_test.cpp
  src/ukf.cpp
)

#=============
# Install
#=============

# Export include directory
ament_export_include_directories(include)

# Install include directory
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

## Install library
install(
  TARGETS ukf
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install executable
install(
  TARGETS ukf_test
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()