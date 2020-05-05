cmake_minimum_required(VERSION 3.5)
project(simple-bandwidth-test)

set(SM_ARCH "sm_61" CACHE STRING "size of the floating point data type")
set_property(CACHE SM_ARCH PROPERTY STRINGS "sm_60" "sm_61" "sm_70" "sm_71")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -std=c++11;
        -arch=${SM_ARCH};
        -res-usage;
        -O3;
        -Xptxas -v;
        -maxrregcount 64;
        -DREAL_SIZE=${REAL_SIZE_IN_BYTES})

cuda_add_executable(${CMAKE_PROJECT_NAME} global.cu
                                          include/gemmgen_aux.cu)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include
                                                         common)

find_package(yaml-cpp REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} ${YAML_CPP_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${YAML_CPP_INCLUDE_DIR})
