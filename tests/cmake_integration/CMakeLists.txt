cmake_minimum_required(VERSION 3.5)
project(cmake_integration)

execute_process(COMMAND python3 -c "import gemmforge; gemmforge.print_cmake_path()"
                OUTPUT_VARIABLE GEMMFORGE_PATH)

set(CMAKE_PREFIX_PATH "${GEMMFORGE_PATH}" ${CMAKE_MODULE_PATH})
find_package(GemmForge REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} ./main.cpp)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${GemmForge_INCLUDE_DIRS})