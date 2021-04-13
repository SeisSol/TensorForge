  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -std=c++11;
        -arch=${SUB_ARCH};
        -res-usage;
        -O3;
        -Xptxas -v;
        -DREAL_SIZE=${REAL_SIZE})

  cuda_add_library(gpu_part STATIC common/test_drivers/simple_driver.cu
                                 include/gemmgen_aux.cu
                                 gen_code/kernels.cu)