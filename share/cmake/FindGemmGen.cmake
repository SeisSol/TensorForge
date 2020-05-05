include(FindPackageHandleStandardArgs)

execute_process(COMMAND realpath ${CMAKE_CURRENT_LIST_DIR}/../..
                OUTPUT_VARIABLE GEMMGEN_PATH)
string(REGEX REPLACE "\n$" "" GEMMGEN_PATH "${GEMMGEN_PATH}")

set(GemmGen_INCLUDE_DIRS "${GEMMGEN_PATH}/include")
set(GemmGen_SOURCES "${GemmGen_INCLUDE_DIRS}/gemmgen_aux.cu")

find_package_handle_standard_args(GemmGen
                                  GemmGen_INCLUDE_DIRS 
                                  GemmGen_SOURCES)