  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
          -std=c++11;
          -arch=${SM_ARCH};
  #        -restrict;
          -res-usage;
          -O3;
  #        -g;
          -lineinfo;
          -Xptxas -v;
  #        -maxrregcount 64;
          -DREAL_SIZE=${REAL_SIZE})

  cuda_add_library(${GPU_TARGET} STATIC ${GPU_TARGET_SOURCE_FILES})
