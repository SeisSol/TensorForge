cmake_minimum_required(VERSION 3.5)
project(simple-gemm)

set(REAL_SIZE "4" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE PROPERTY STRINGS "8" "4")

set(SM_ARCH "gfx906" CACHE STRING "size of the floating point data type")
set_property(CACHE SM_ARCH PROPERTY STRINGS "sm_60" "sm_61" "sm_70" "sm_71" "gfx906")

set(REFERENCE_IMPL "GEMMGEN" CACHE STRING "size of the floating point data type")
set_property(CACHE REFERENCE_IMPL PROPERTY STRINGS "GEMMGEN" "OPENBLAS")

set(DEVICE_BACKEND "HIP" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "CUDA" "HIP")

set(MANUFACTURER "")

if (${DEVICE_BACKEND} STREQUAL "CUDA")
  set(MANUFACTURER "nvidia")
elseif (${DEVICE_BACKEND} STREQUAL "HIP")
  set(MANUFACTURER "amd")
endif()



add_executable(${CMAKE_PROJECT_NAME}
               main.cpp
               common/gemm.cpp
               common/aux.cpp
               common/test_drivers/simple_comparator.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC gen_code
                                                 common
                                                 common/test_drivers
                                                 include)


if (${REFERENCE_IMPL} STREQUAL "OPENBLAS")
    find_package(OpenBLAS REQUIRED)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OpenBLAS_LIBRARIES})
endif()
target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC "CONCRETE_CPU_BACKEND=${REFERENCE_IMPL}")


target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC "-std=c++11")
target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC "REAL_SIZE=${REAL_SIZE}")

if(${DEVICE_BACKEND} STREQUAL "CUDA")

  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
          -std=c++11;
          -arch=${SM_ARCH};
  #        -restrict;
          -res-usage;
          -O3;
  #        -g;
  #        -lineinfo;
          -Xptxas -v;
  #        -maxrregcount 64;
          -DREAL_SIZE=${REAL_SIZE})

  cuda_add_library(gpu_part STATIC common/test_drivers/simple_driver.cu
                                 include/gemmgen_aux.cu
                                 gen_code/kernels.cu)

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

  set(HIP_COMPILER hcc)

  find_package(HIP QUIET)
  if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
  else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
  endif()

  set(HIP_COMPILER hcc)

  set(SOURCE_FILES common/test_drivers/simple_driver.cpp
                   include/gemmgen_aux.cpp
                   gen_code/kernels.cpp)

  set(HIPCC_FLAGS -DREAL_SIZE=${REAL_SIZE}; 
                  -std=c++11; 
                  -O3)
  set(HCC_FLAGS)
  set(NVCC_FLAGS)

  set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

  set_source_files_properties(${SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  hip_add_library(gpu_part STATIC ${SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_FLAGS} HCC_OPTIONS ${HCC_FLAGS} NVCC_OPTIONS ${NVCC_FLAGS})
  
  target_link_libraries(gpu_part PUBLIC ${HIP_PATH}/lib/libamdhip64.so)

endif()

target_include_directories(gpu_part PRIVATE include
                                            common
                                            common/test_drivers
                                            gen_code)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE gpu_part)

find_package(yaml-cpp REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${YAML_CPP_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${YAML_CPP_INCLUDE_DIR})

add_custom_target(copy_params ALL
                  cmake -E copy ${CMAKE_SOURCE_DIR}/params.yaml ${CMAKE_BINARY_DIR}
                  COMMAND
                    cmake -E copy ${CMAKE_SOURCE_DIR}/config.yaml ${CMAKE_BINARY_DIR}
                  COMMENT
                    "copying the parameter file"
                  WORKING_DIRECTORY
                    ${CMAKE_SOURCE_DIR})

if(${DEVICE_BACKEND} STREQUAL "CUDA")

  add_custom_target(gpu_generator ALL
                    python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
                              --realsize=${REAL_SIZE} --manufacturer=${MANUFACTURER} --sub_arch=${SM_ARCH}
                    BYPRODUCTS
                      ${CMAKE_SOURCE_DIR}/gen_code/kernels.cu
                      ${CMAKE_SOURCE_DIR}/gen_code/kernels.h
                    COMMENT
                      "generating compute kernels"
                    WORKING_DIRECTORY
                      ${CMAKE_SOURCE_DIR})

elseif(${DEVICE_BACKEND} STREQUAL "HIP")

  add_custom_target(gpu_generator ALL
                    python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
                              --realsize=${REAL_SIZE} --manufacturer=${MANUFACTURER} --sub_arch=${SM_ARCH}
                    BYPRODUCTS
                      ${CMAKE_SOURCE_DIR}/gen_code/kernels.cpp
                      ${CMAKE_SOURCE_DIR}/gen_code/kernels.h
                    COMMENT
                      "generating compute kernels"
                    WORKING_DIRECTORY
                      ${CMAKE_SOURCE_DIR})
endif()

add_dependencies(gpu_part gpu_generator)
add_dependencies(${CMAKE_PROJECT_NAME} copy_params)
