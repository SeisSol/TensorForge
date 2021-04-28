  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -std=c++11;
        -arch=${SUB_ARCH};
        #        -restrict;
        -res-usage;
        -O3;
        #        -g;
        #        -lineinfo;
        -Xptxas -v;
        #-maxrregcount 64;
        -DREAL_SIZE=${REAL_SIZE})

  cuda_add_library(gpu_part STATIC common/test_drivers/simple_driver.cpp
                                 include/gemmgen_aux.cu
                                 gen_code/kernels.cu)