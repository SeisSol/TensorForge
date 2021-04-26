cmake_minimum_required(VERSION 3.5)
project(simple-bandwidth-test)

set(DEVICE_SUB_ARCH "gfx906" CACHE STRING "size of the floating point data type")
set_property(CACHE DEVICE_SUB_ARCH PROPERTY STRINGS "sm_60" "sm_61" "sm_70" "sm_71")

set(DEVICE_BACKEND "HIP" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "CUDA" "HIP")

set(REAL_SIZE_IN_BYTES 4)

add_subdirectory(../../submodules/Device build)

if(${DEVICE_BACKEND} STREQUAL "CUDA")
    include(nvidia.cmake)
elseif(${DEVICE_BACKEND} STREQUAL "HIP")
    include(amd.cmake)
elseif((${DEVICE_BACKEND} STREQUAL "ONEAPI") OR (${DEVICE_BACKEND} STREQUAL "HIPSYCL"))
    include(sycl.cmake)
endif()

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include
                                                         common)

find_package(yaml-cpp REQUIRED)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${YAML_CPP_INCLUDE_DIR} ../../submodules/Device)
target_link_libraries(${CMAKE_PROJECT_NAME} ${YAML_CPP_LIBRARIES} device)
