include(FindPackageHandleStandardArgs)

execute_process(COMMAND realpath ${CMAKE_CURRENT_LIST_DIR}/../..
                OUTPUT_VARIABLE GEMMFORGE_PATH)
string(REGEX REPLACE "\n$" "" GEMMFORGE_PATH "${GEMMFORGE_PATH}")

set(GemmForge_INCLUDE_DIRS "${GEMMFORGE_PATH}/include")

set(GemmForge_ALLOWED "cuda" "hip" "oneapi" "hipsycl")
if(NOT DEFINED DEVICE_BACKEND)
  set(DEVICE_BACKEND "cuda" CACHE STRING "type of an interface")
  set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS ${GemmForge_ALLOWED})
else()
  list(FIND GemmForge_ALLOWED ${DEVICE_BACKEND} INDEX)
  set(GemmForge_WRONG_INDEX -1)
  if (${INDEX} EQUAL ${GemmForge_WRONG_INDEX})
    message(FATAL_ERROR "DEVICE_BACKEND=${DEVICE_BACKEND} is wrong. Allowed: ${GemmForge_ALLOWED}")
  endif()
endif()


if (${DEVICE_BACKEND} STREQUAL "cuda")
    set(GemmForge_SOURCES "${GemmForge_INCLUDE_DIRS}/gemmforge_aux.cu")
elseif (${DEVICE_BACKEND} STREQUAL "hip")
    set(GemmForge_SOURCES "${GemmForge_INCLUDE_DIRS}/gemmforge_aux.cpp")
else()
    set(GemmForge_SOURCES "${GemmForge_INCLUDE_DIRS}/gemmforge_aux_sycl.cpp")
endif()

find_package_handle_standard_args(GemmForge
                                  GemmForge_INCLUDE_DIRS
                                  GemmForge_SOURCES)
