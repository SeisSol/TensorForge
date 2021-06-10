set(REAL_SIZE "4" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE PROPERTY STRINGS "8" "4")

set(SM_ARCH "sm_60" CACHE STRING "size of the floating point data type")
set_property(CACHE SM_ARCH PROPERTY STRINGS "sm_60" "sm_61" "sm_70" "sm_71" "gfx906" "bdw"
                            "skl" "kbl" "cfl" "bxt" "glk" "icllp" "lkf" "ehl" "tgllp"
                            "rkl" "adls" "dg1" "Gen8" "Gen9" "Gen11" "Gen12LP")

set(REFERENCE_IMPL "GEMMFORGE" CACHE STRING "size of the floating point data type")
set_property(CACHE REFERENCE_IMPL PROPERTY STRINGS "GEMMFORGE" "OPENBLAS")

set(DEVICE_BACKEND "CUDA" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "CUDA" "HIP" "ONEAPI" "HIPSYCL")

set(MANUFACTURER "")

if (${DEVICE_BACKEND} STREQUAL "CUDA")
    set(MANUFACTURER "nvidia")
elseif (${DEVICE_BACKEND} STREQUAL "HIP")
    set(MANUFACTURER "amd")
elseif ((${DEVICE_BACKEND} STREQUAL "ONEAPI") OR (${DEVICE_BACKEND} STREQUAL "HIPSYCL"))
    string(TOLOWER ${DEVICE_BACKEND} MANUFACTURER)
endif()

set(REAL_SIZE_IN_BYTES ${REAL_SIZE})
set(DEVICE_SUB_ARCH ${SM_ARCH})

if(NOT DEFINED PREFERRED_DEVICE_TYPE)
    set(ENV{PREFERRED_DEVICE_TYPE} GPU)
endif()

