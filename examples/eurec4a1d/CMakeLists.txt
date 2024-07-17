# set cmake version
if(NOT DEFINED CMAKE_MINIMUM_REQUIRED_VERSION)
  cmake_minimum_required(VERSION 3.18.0)
endif()

# subdirectories for specific examples of CLEO
add_subdirectory(stationary_collision_condensation/src EXCLUDE_FROM_ALL)
add_subdirectory(stationary_condensation/src EXCLUDE_FROM_ALL)
add_subdirectory(stationary_null_microphysics/src EXCLUDE_FROM_ALL)
add_subdirectory(stationary_coalrebu_condensation_small/src EXCLUDE_FROM_ALL)
add_subdirectory(stationary_coalrebu_condensation_large/src EXCLUDE_FROM_ALL)
