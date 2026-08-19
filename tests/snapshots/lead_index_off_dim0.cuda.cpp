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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 180 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 20 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 9 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 32;
          bool v4_g = v3_lead < 20;
          #pragma unroll
          for (int32_t v0_i0 = 0; v0_i0 < 1; ++v0_i0) {
            if (v4_g) {
              int32_t v10_a = v0_i0 + v3_lead;
              float v11_data;
              {
                v11_data = __ldcg(&glb_m1[v10_a]);
              }
              int32_t v12_a = v0_i0 + 0;
              r0[v12_a] = v11_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          pipeline.producer_acquire();
          if (threadIdx.x < 9) {
            cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<4>(4), pipeline);
          }
          __syncwarp();
          pipeline.producer_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[9]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 20), (0, 9)] [(0, 1)]
            float ir1[9]{};
            int32_t v15_lead = threadIdx.x % 32;
            if (v15_lead < 20) {
              float v17_data = r0[0];
              float v18_data = s0[0];
              float v20_data = ir1[0];
              ir1[0] = (v20_data + (v17_data * v18_data));
              float v23_data = s0[1];
              float v25_data = ir1[1];
              ir1[1] = (v25_data + (v17_data * v23_data));
              float v28_data = s0[2];
              float v30_data = ir1[2];
              ir1[2] = (v30_data + (v17_data * v28_data));
              float v33_data = s0[3];
              float v35_data = ir1[3];
              ir1[3] = (v35_data + (v17_data * v33_data));
              float v38_data = s0[4];
              float v40_data = ir1[4];
              ir1[4] = (v40_data + (v17_data * v38_data));
              float v43_data = s0[5];
              float v45_data = ir1[5];
              ir1[5] = (v45_data + (v17_data * v43_data));
              float v48_data = s0[6];
              float v50_data = ir1[6];
              ir1[6] = (v50_data + (v17_data * v48_data));
              float v53_data = s0[7];
              float v55_data = ir1[7];
              ir1[7] = (v55_data + (v17_data * v53_data));
              float v58_data = s0[8];
              float v60_data = ir1[8];
              ir1[8] = (v60_data + (v17_data * v58_data));
            }
            if (v15_lead < 20) {
              #pragma unroll
              for (int32_t v66_n1 = 0; v66_n1 < 9; ++v66_n1) {
                int32_t v67_a = 0 + v66_n1;
                float v69_data = ir1[v66_n1];
                int32_t v70_a = 0 + v66_n1;
                r1[v70_a] = v69_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v73_lead = threadIdx.x % 32;
          if (v73_lead < 20) {
            #pragma unroll
            for (int32_t v75_i1 = 0; v75_i1 < 9; ++v75_i1) {
              int32_t v76_a = 0 + v75_i1;
              float v78_data = r1[v75_i1];
              int32_t v85_a = v73_lead + (v75_i1 * 20);
              glb_m0[v85_a] = v78_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

