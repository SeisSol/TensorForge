cmake_minimum_required(VERSION 3.5)
project(cmake_integration)

execute_process(COMMAND python3 -c "import tensorforge; tensorforge.print_cmake_path()"
                OUTPUT_VARIABLE TENSORFORGE_PATH)

set(CMAKE_PREFIX_PATH "${TENSORFORGE_PATH}" ${CMAKE_MODULE_PATH})
find_package(Tensorforge REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} ./main.cpp)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${Tensorforge_INCLUDE_DIRS})