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
elseif ((${DEVICE_BACKEND} STREQUAL "ONEAPI") OR (${DEVICE_BACKEND} STREQUAL "HIPSYCL"))
    set(MANUFACTURER "sycl")
endif()

set(REAL_SIZE_IN_BYTES ${REAL_SIZE})
set(DEVICE_SUB_ARCH ${SM_ARCH})

if(NOT DEFINED PREFERRED_DEVICE_TYPE)
    set(ENV{PREFERRED_DEVICE_TYPE} GPU)
endif()

add_subdirectory(../../submodules/Device build)

add_executable(${TARGET} ${TARGET_SOURCE_FILES})
target_compile_options(${TARGET} PUBLIC "-std=c++11")
target_compile_definitions(${TARGET} PUBLIC "REAL_SIZE=${REAL_SIZE}")
target_include_directories(${TARGET} PUBLIC ${TARGET_INCLUDE_DIRS})

if (${REFERENCE_IMPL} STREQUAL "OPENBLAS")
    find_package(OpenBLAS REQUIRED)
    target_include_directories(${TARGET} PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${OpenBLAS_LIBRARIES})
endif()
target_compile_definitions(${TARGET} PUBLIC "CONCRETE_CPU_BACKEND=${REFERENCE_IMPL}")
