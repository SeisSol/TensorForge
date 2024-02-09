include(FindPackageHandleStandardArgs)

execute_process(COMMAND realpath ${CMAKE_CURRENT_LIST_DIR}/../..
                OUTPUT_VARIABLE KERNELFORGE_PATH)
string(REGEX REPLACE "\n$" "" KERNELFORGE_PATH "${KERNELFORGE_PATH}")

set(KernelForge_INCLUDE_DIRS "${KERNELFORGE_PATH}/include")

set(KernelForge_ALLOWED "cuda" "hip" "oneapi" "hipsycl" "omptarget" "targetdart")
if(NOT DEFINED DEVICE_BACKEND)
  set(DEVICE_BACKEND "cuda" CACHE STRING "type of an interface")
  set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS ${KernelForge_ALLOWED})
else()
  list(FIND KernelForge_ALLOWED ${DEVICE_BACKEND} INDEX)
  set(KernelForge_WRONG_INDEX -1)
  if (${INDEX} EQUAL ${KernelForge_WRONG_INDEX})
    message(FATAL_ERROR "DEVICE_BACKEND=${DEVICE_BACKEND} is wrong. Allowed: ${KernelForge_ALLOWED}")
  endif()
endif()


if (${DEVICE_BACKEND} STREQUAL "cuda")
    set(KernelForge_SOURCES "${KernelForge_INCLUDE_DIRS}/kernelforge_aux.cu")
elseif (${DEVICE_BACKEND} STREQUAL "hip")
    set(KernelForge_SOURCES "${KernelForge_INCLUDE_DIRS}/kernelforge_aux.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "hipsycl")
    set(KernelForge_SOURCES "${KernelForge_INCLUDE_DIRS}/kernelforge_aux_sycl.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "oneapi")
    set(KernelForge_SOURCES "${KernelForge_INCLUDE_DIRS}/kernelforge_aux_sycl.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "omptarget")
    set(KernelForge_SOURCES "${KernelForge_INCLUDE_DIRS}/kernelforge_aux_target.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "targetdart")
    set(KernelForge_SOURCES "${KernelForge_INCLUDE_DIRS}/kernelforge_aux_target.cpp")
endif()

find_package_handle_standard_args(KernelForge
                                  KernelForge_INCLUDE_DIRS
                                  KernelForge_SOURCES)
