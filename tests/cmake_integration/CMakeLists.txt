cmake_minimum_required(VERSION 3.5)
project(cmake_integration)

execute_process(COMMAND python3 -c "import kernelforge; kernelforge.print_cmake_path()"
                OUTPUT_VARIABLE KERNELFORGE_PATH)

set(CMAKE_PREFIX_PATH "${KERNELFORGE_PATH}" ${CMAKE_MODULE_PATH})
find_package(Kernelforge REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} ./main.cpp)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${Kernelforge_INCLUDE_DIRS})