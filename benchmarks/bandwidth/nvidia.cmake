    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                        -std=c++11;
                        -arch=${SM_ARCH};
                        -res-usage;
                        -O3;
                        -Xptxas -v;
                        -maxrregcount 64;
                        -DREAL_SIZE=${REAL_SIZE_IN_BYTES})

    cuda_add_executable(${CMAKE_PROJECT_NAME} global.cu
                                              include/gemmgen_aux.cu)