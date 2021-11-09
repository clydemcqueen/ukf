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
find_package(Eigen3 REQUIRED)

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

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(
    ukf_test
    src/ukf_test.cpp
    src/ukf.cpp
  )
endif()

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

ament_package()