find_package(GTest REQUIRED)
target_link_libraries(${TARGET} PRIVATE ${GTEST_BOTH_LIBRARIES})
target_include_directories(${TARGET} PRIVATE ${GTEST_INCLUDE_DIR})