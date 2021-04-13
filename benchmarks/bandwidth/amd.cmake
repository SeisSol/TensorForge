
  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()

  #set the CMAKE_MODULE_PATH for the helper cmake files from HIP
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

  find_package(HIP QUIET)
  if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
  else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
  endif()
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