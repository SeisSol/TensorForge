// === base name ===
kernel_227fc8550bc8cc23

// === header ===
void launcher_kernel_227fc8550bc8cc23(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_227fc8550bc8cc23(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_227fc8550bc8cc23, block.x * block.y * block.z, 128 * sizeof(float));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  dim3 grid (std::min(gridsize, numElements0), 1, 1);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_227fc8550bc8cc23, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_227fc8550bc8cc23<<<grid,block,128 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_227fc8550bc8cc23(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 20×9(20×9) {0..20}×{0..9} strided
    // m1 1×20(1×20) {0..1}×{0..20} strided
    // m2 1×9(1×9) {0..1}×{0..9} strided
    // m0 20×9(20×9) {0..20}×{0..9} strided({0..20}×{0..9})[0, 1] = m1 1×20(1×20) {0..1}×{0..20} strided({0..1}×{0..20})[-1, 0]×m2 1×9(1×9) {0..1}×{0..9} strided({0..1}×{0..9})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 180 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 20 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 9 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 32;
          bool v20_g = v19_lead < 20;
          #pragma unroll
          for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
            if (v20_g) {
              float v27_data = __ldcg(&glb_m1[(v16_i0 + v19_lead)]);
              r0[v16_i0] = v27_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 20), (0, 9)] [(0, 1)]
          float ir1[9]{};
          int32_t v34_lead = threadIdx.x % 32;
          if (v34_lead < 20) {
            float v36_data = r0[0];
            float v37_data = s0[0];
            float v39_data = ir1[0];
            ir1[0] = (v39_data + (v36_data * v37_data));
            float v42_data = s0[1];
            float v44_data = ir1[1];
            ir1[1] = (v44_data + (v36_data * v42_data));
            float v47_data = s0[2];
            float v49_data = ir1[2];
            ir1[2] = (v49_data + (v36_data * v47_data));
            float v52_data = s0[3];
            float v54_data = ir1[3];
            ir1[3] = (v54_data + (v36_data * v52_data));
            float v57_data = s0[4];
            float v59_data = ir1[4];
            ir1[4] = (v59_data + (v36_data * v57_data));
            float v62_data = s0[5];
            float v64_data = ir1[5];
            ir1[5] = (v64_data + (v36_data * v62_data));
            float v67_data = s0[6];
            float v69_data = ir1[6];
            ir1[6] = (v69_data + (v36_data * v67_data));
            float v72_data = s0[7];
            float v74_data = ir1[7];
            ir1[7] = (v74_data + (v36_data * v72_data));
            float v77_data = s0[8];
            float v79_data = ir1[8];
            ir1[8] = (v79_data + (v36_data * v77_data));
          }
          if (v34_lead < 20) {
            #pragma unroll
            for (int32_t v85_n1 = 0; v85_n1 < 9; ++v85_n1) {
              float v87_data = ir1[v85_n1];
              r1[v85_n1] = v87_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v34_lead < 20) {
            #pragma unroll
            for (int32_t v93_i1 = 0; v93_i1 < 9; ++v93_i1) {
              float v95_data = r1[v93_i1];
              glb_m0[(v34_lead + (v93_i1 * 20))] = v95_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

