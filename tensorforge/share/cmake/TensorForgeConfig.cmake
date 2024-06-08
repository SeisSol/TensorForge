include(FindPackageHandleStandardArgs)

execute_process(COMMAND realpath ${CMAKE_CURRENT_LIST_DIR}/../..
                OUTPUT_VARIABLE TENSORFORGE_PATH)
string(REGEX REPLACE "\n$" "" TENSORFORGE_PATH "${TENSORFORGE_PATH}")

set(TensorForge_INCLUDE_DIRS "${TENSORFORGE_PATH}/include")

set(TensorForge_ALLOWED "cuda" "hip" "oneapi" "hipsycl" "omptarget" "targetdart")
if(NOT DEFINED DEVICE_BACKEND)
  set(DEVICE_BACKEND "cuda" CACHE STRING "type of an interface")
  set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS ${TensorForge_ALLOWED})
else()
  list(FIND TensorForge_ALLOWED ${DEVICE_BACKEND} INDEX)
  set(TensorForge_WRONG_INDEX -1)
  if (${INDEX} EQUAL ${TensorForge_WRONG_INDEX})
    message(FATAL_ERROR "DEVICE_BACKEND=${DEVICE_BACKEND} is wrong. Allowed: ${TensorForge_ALLOWED}")
  endif()
endif()


if (${DEVICE_BACKEND} STREQUAL "cuda")
    set(TensorForge_SOURCES "${TensorForge_INCLUDE_DIRS}/tensorforge_aux.cu")
elseif (${DEVICE_BACKEND} STREQUAL "hip")
    set(TensorForge_SOURCES "${TensorForge_INCLUDE_DIRS}/tensorforge_aux.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "hipsycl")
    set(TensorForge_SOURCES "${TensorForge_INCLUDE_DIRS}/tensorforge_aux_sycl.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "acpp")
    set(TensorForge_SOURCES "${TensorForge_INCLUDE_DIRS}/tensorforge_aux_sycl.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "oneapi")
    set(TensorForge_SOURCES "${TensorForge_INCLUDE_DIRS}/tensorforge_aux_sycl.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "dpcpp")
    set(TensorForge_SOURCES "${TensorForge_INCLUDE_DIRS}/tensorforge_aux_sycl.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "omptarget")
    set(TensorForge_SOURCES "${TensorForge_INCLUDE_DIRS}/tensorforge_aux_target.cpp")
elseif (${DEVICE_BACKEND} STREQUAL "targetdart")
    set(TensorForge_SOURCES "${TensorForge_INCLUDE_DIRS}/tensorforge_aux_target.cpp")
endif()

find_package_handle_standard_args(TensorForge
                                  TensorForge_INCLUDE_DIRS
                                  TensorForge_SOURCES)
