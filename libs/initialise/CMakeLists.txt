# set cmake version
cmake_minimum_required(VERSION 3.18.0)
#cmake_minimum_required(VERSION 3.21.1) # if using Kokkos with NVC++ compiler 

# setlibrary name
set(LIBNAME "initialise")

# print where exactly project source dir is for this CMakeLists.txt
message("${LIBNAME} LIBRARY_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

# explicitly set library executables path to /lib in top level of build tree
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

# add executables to library
set(SOURCES 
  "config.cpp"
  )
add_library("${LIBNAME}" SHARED ${SOURCES})

# add directories to include for targets
target_include_directories(${LIBNAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# some compilation properties
set_target_properties(${LIBNAME} PROPERTIES
  CMAKE_CXX_STANDARD_REQUIRED ON
  CMAKE_CXX_EXTENSIONS ON
  CXX_STANDARD 20)

set(compileops -Werror -Wall -pedantic -g -gdwarf-4)
#set(compileops -Werror -Wall -pedantic -g) # cannot use -gdwarf-4 flag for nvc++ compiler 

target_compile_options(${LIBNAME} PRIVATE ${compileops})