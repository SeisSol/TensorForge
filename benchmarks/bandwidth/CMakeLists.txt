cmake_minimum_required(VERSION 3.5)
project(simple-bandwidth-test)

set(SM_ARCH "sm_61" CACHE STRING "size of the floating point data type")
set_property(CACHE SM_ARCH PROPERTY STRINGS "sm_60" "sm_61" "sm_70" "sm_71")

set(DEVICE_BACKEND "CUDA" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "CUDA" "HIP")

if(${DEVICE_BACKEND} STREQUAL "CUDA")
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

elseif(${DEVICE_BACKEND} STREQUAL "HIP")

  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()

  #set the CMAKE_MODULE_PATH for the helper cmake files from HIP
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

  find_package(HIP REQUIRED)

  #Do NOT set this if you make an executable
  #set(HIP_COMPILER hcc)

  set(SOURCE_FILES global.cpp
                   include/gemmgen_aux.cpp)

  set(HIPCC_FLAGS -std=c++11; 
                  -O3) 
                  #-res-usage)
  set(HCC_FLAGS)
  set(NVCC_FLAGS)

  set_source_files_properties(${SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  hip_add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_FLAGS} HCC_OPTIONS ${HCC_FLAGS} NVCC_OPTIONS ${NVCC_FLAGS})

endif()

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include
                                                         common)

find_package(yaml-cpp REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} ${YAML_CPP_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${YAML_CPP_INCLUDE_DIR})
