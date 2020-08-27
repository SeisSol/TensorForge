execute_process(COMMAND python -c "import gemmforge; gemmforge.get_version()"
                OUTPUT_VARIABLE PACKAGE_VERSION)

# Check whether the requested PACKAGE_FIND_VERSION is compatible
if("${PACKAGE_VERSION}" VERSION_EQUAL "${PACKAGE_FIND_VERSION}")
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    set(PACKAGE_VERSION_EXACT TRUE)
else()
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
    set(PACKAGE_VERSION_EXACT FALSE)
endif()