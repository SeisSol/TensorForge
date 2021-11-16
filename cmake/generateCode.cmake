if (NOT "${TEST_SUITE}" STREQUAL "")
    SET(TEST_SPEC "--specfile=${CMAKE_BINARY_DIR}/${TEST_SUITE}")
endif()

add_custom_target(gpu_generator ALL
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
        ${TEST_SPEC} --realsize=${REAL_SIZE} --backend=${DEVICE_BACKEND} --arch=${SM_ARCH}
        BYPRODUCTS
        ${GEN_COPY_PRODUCTS}
        COMMENT
        "generating compute kernels"
        WORKING_DIRECTORY
        ${CMAKE_SOURCE_DIR})


add_dependencies(${GPU_TARGET} gpu_generator)

SET(INT 0)
FOREACH(LETTER ${GEN_COPY_FILES})
add_custom_target(copy_params_${INT} ALL
        cmake -E copy ${LETTER} ${CMAKE_BINARY_DIR}
        COMMENT
        "copying the parameter file"
        WORKING_DIRECTORY
        ${CMAKE_SOURCE_DIR})

    add_dependencies(${TARGET} copy_params_${INT})
    MATH(EXPR INT "${INT}+1")
ENDFOREACH()

