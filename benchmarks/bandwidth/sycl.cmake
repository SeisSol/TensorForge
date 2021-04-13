set(SOURCE_FILES global_sycl.cpp
                 include/gemmgen_aux_sycl.cpp)
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

if (${DEVICE_BACKEND} STREQUAL "HIPSYCL")
    find_package(${CMAKE_PROJECT_NAME} CONFIG REQUIRED)
    add_sycl_to_target(TARGET gpu_part SOURCES ${DEVICE_SOURCE_FILES})
else()
    set(CMAKE_CXX_COMPILER dpcpp)
endif()

target_compile_options(${CMAKE_PROJECT_NAME}  PRIVATE "-std=c++17" "-O3")
target_compile_definitions(${CMAKE_PROJECT_NAME}  PRIVATE DEVICE_${DEVICE_BACKEND}_LANG REAL_SIZE=${REAL_SIZE})