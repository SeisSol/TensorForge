set(REAL_SIZE "4" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE PROPERTY STRINGS "8" "4")

set(SM_ARCH "sm_60" CACHE STRING "size of the floating point data type")
set_property(CACHE SM_ARCH PROPERTY STRINGS sm_60 sm_61 sm_70 sm_71
    gfx906 gfx908
    dg1 bdw skl Gen8 Gen9 Gen11 Gen12LP)

set(REFERENCE_IMPL "GEMMFORGE" CACHE STRING "size of the floating point data type")
set_property(CACHE REFERENCE_IMPL PROPERTY STRINGS "GEMMFORGE" "OPENBLAS")

set(DEVICE_BACKEND "cuda" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "cuda" "hip" "oneapi" "hipsycl")

set(REAL_SIZE_IN_BYTES ${REAL_SIZE})
set(DEVICE_ARCH ${SM_ARCH})

if(NOT DEFINED PREFERRED_DEVICE_TYPE)
    set(ENV{PREFERRED_DEVICE_TYPE} GPU)
endif()

