find_package(yaml-cpp REQUIRED)
target_include_directories(${TARGET} PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(${TARGET} PRIVATE ${YAML_CPP_LIBRARIES})