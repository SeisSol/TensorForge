add_library(${GPU_TARGET} SHARED ${GPU_TARGET_SOURCE_FILES})

if (${DEVICE_BACKEND} STREQUAL "HIPSYCL")
    find_package(${GPU_TARGET} CONFIG REQUIRED)
    add_sycl_to_target(TARGET ${GPU_TARGET} SOURCES ${GPU_TARGET_SOURCE_FILES})
else()
    set(CMAKE_CXX_COMPILER dpcpp)

    if("$ENV{PREFERRED_DEVICE_TYPE}" STREQUAL "CPU")
        target_compile_options(${GPU_TARGET} PRIVATE "-fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice")
        set_target_properties(${GPU_TARGET} PROPERTIES LINK_FLAGS "-fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs \"-march=${SM_ARCH}\"")
    else()
        target_compile_options(${GPU_TARGET} PRIVATE "-fsycl-targets=spir64_gen-unknown-unknown-sycldevice")
        set_target_properties(${GPU_TARGET} PROPERTIES LINK_FLAGS "-fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device ${SM_ARCH}\"")
    endif()
endif()

target_compile_options(${GPU_TARGET} PRIVATE "-std=c++17" "-O3")
target_compile_definitions(${GPU_TARGET} PRIVATE DEVICE_${DEVICE_BACKEND}_LANG REAL_SIZE=${REAL_SIZE})