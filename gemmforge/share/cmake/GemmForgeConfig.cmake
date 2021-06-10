include(FindPackageHandleStandardArgs)

execute_process(COMMAND realpath ${CMAKE_CURRENT_LIST_DIR}/../..
                OUTPUT_VARIABLE GEMMFORGE_PATH)
string(REGEX REPLACE "\n$" "" GEMMFORGE_PATH "${GEMMFORGE_PATH}")

set(GemmForge_INCLUDE_DIRS "${GEMMFORGE_PATH}/include")

set(DEVICE_BACKEND "CUDA" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "CUDA" "HIP" "ONEAPI" "HIPSYCL")


if (${DEVICE_BACKEND} STREQUAL "CUDA")
    set(GemmForge_SOURCES "${GemmForge_INCLUDE_DIRS}/gemmforge_aux.cu")
elseif (${DEVICE_BACKEND} STREQUAL "HIP")
    set(GemmForge_SOURCES "${GemmForge_INCLUDE_DIRS}/gemmforge_aux.cpp")
else()
    set(GemmForge_SOURCES "${GemmForge_INCLUDE_DIRS}/gemmforge_aux_sycl.cpp")
endif()

find_package_handle_standard_args(GemmForge
                                  GemmForge_INCLUDE_DIRS
                                  GemmForge_SOURCES)