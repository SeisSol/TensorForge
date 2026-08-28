// === base name ===
kernel_75d3097b00

// === header ===
void launcher_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_75d3097b00, block.x * block.y * block.z, 256 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_75d3097b00, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_75d3097b00<<<grid,block,256 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 20×9(20×9) {0..20}×{0..9} strided
    // m1 1×20(1×20) {0..1}×{0..20} strided
    // m2 1×9(1×9) {0..1}×{0..9} strided
    // m0 20×9(20×9) {0..20}×{0..9} strided({0..20}×{0..9})[0, 1] = m1 1×20(1×20) {0..1}×{0..20} strided({0..1}×{0..20})[-1, 0]×m2 1×9(1×9) {0..1}×{0..9} strided({0..1}×{0..9})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 180 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 20 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 9 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v11_lead = threadIdx.x % 32;
          bool v12_g = v11_lead < 20;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            if (v12_g) {
              float v19_data = __ldcg(&glb_m1[(v8_i0 + v11_lead)]);
              r0[v8_i0] = v19_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 20), (0, 9)] [(0, 1)]
          float ir1[9]{};
          int32_t v27_lead = threadIdx.x % 32;
          if (v27_lead < 20) {
            float v29_data = r0[0];
            float v30_data = s0[0];
            float v32_data = ir1[0];
            ir1[0] = (v32_data + (v29_data * v30_data));
            float v35_data = s0[1];
            float v37_data = ir1[1];
            ir1[1] = (v37_data + (v29_data * v35_data));
            float v40_data = s0[2];
            float v42_data = ir1[2];
            ir1[2] = (v42_data + (v29_data * v40_data));
            float v45_data = s0[3];
            float v47_data = ir1[3];
            ir1[3] = (v47_data + (v29_data * v45_data));
            float v50_data = s0[4];
            float v52_data = ir1[4];
            ir1[4] = (v52_data + (v29_data * v50_data));
            float v55_data = s0[5];
            float v57_data = ir1[5];
            ir1[5] = (v57_data + (v29_data * v55_data));
            float v60_data = s0[6];
            float v62_data = ir1[6];
            ir1[6] = (v62_data + (v29_data * v60_data));
            float v65_data = s0[7];
            float v67_data = ir1[7];
            ir1[7] = (v67_data + (v29_data * v65_data));
            float v70_data = s0[8];
            float v72_data = ir1[8];
            ir1[8] = (v72_data + (v29_data * v70_data));
          }
          if (v27_lead < 20) {
            #pragma unroll
            for (int32_t v78_n1 = 0; v78_n1 < 9; ++v78_n1) {
              float v80_data = ir1[v78_n1];
              r1[v78_n1] = v80_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v27_lead < 20) {
            #pragma unroll
            for (int32_t v86_i1 = 0; v86_i1 < 9; ++v86_i1) {
              float v88_data = r1[v86_i1];
              glb_m0[(v27_lead + (v86_i1 * 20))] = v88_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

