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
  message(FATAL_ERROR "Could not find HIP. \
  Ensure that HIP is either installed in /opt/rocm/hip \
  or the variable HIP_PATH is set to point to the right location.")
endif()

# set up HIP specific flag in case offloading to AMD GPUs
set(HCC_FLAGS)

# set up CUDA specific flag in case offloading to Nvidia GPUs
if (DEFINED ENV{HIP_PLATFORM})
  if ($ENV{HIP_PLATFORM} STREQUAL "nvidia")
    set(NVCC_FLAGS -arch=${DEVICE_SUB_ARCH};
                   -dc;
                   --expt-relaxed-constexpr;
                   -DCUDA_UNDERHOOD)
  endif()
endif()

# set up common compiler flags
set(HIPCC_FLAGS -DREAL_SIZE=${REAL_SIZE};
                -std=c++11;
                -O3)

set(CMAKE_HIP_CREATE_SHARED_LIBRARY
"${HIP_HIPCC_CMAKE_LINKER_HELPER} \
${HCC_PATH} \
<CMAKE_SHARED_LIBRARY_CXX_FLAGS> \
<LANGUAGE_COMPILE_FLAGS> \
<LINK_FLAGS> \
<CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> \
<OBJECTS> \
<LINK_LIBRARIES>")

set_source_files_properties(${GPU_TARGET_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

hip_add_library(${GPU_TARGET} SHARED ${GPU_TARGET_SOURCE_FILES}
                              HIPCC_OPTIONS ${HIPCC_FLAGS}
                              HCC_OPTIONS ${HCC_FLAGS}
                              NVCC_OPTIONS ${NVCC_FLAGS})

set_property(TARGET ${GPU_TARGET} PROPERTY HIP_ARCHITECTURES OFF)

if (DEFINED ENV{HIP_PLATFORM})
    if ($ENV{HIP_PLATFORM} STREQUAL "nvidia")
        set_target_properties(${GPU_TARGET} PROPERTIES LINKER_LANGUAGE HIP)
        target_link_options(${GPU_TARGET} PRIVATE -arch=${DEVICE_SUB_ARCH})
    else()
        target_link_libraries(${GPU_TARGET} PUBLIC ${HIP_PATH}/lib/libamdhip64.so)
    endif()
else()
    target_link_libraries(${GPU_TARGET} PUBLIC ${HIP_PATH}/lib/libamdhip64.so)
endif()
