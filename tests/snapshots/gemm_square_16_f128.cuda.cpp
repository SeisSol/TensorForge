// === base name ===
kernel_0b2fc070b9

// === header ===
void launcher_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (2, 128, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0b2fc070b9, block.x * block.y * block.z, 1280 * sizeof(__float128));
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
        cudaFuncSetAttribute(kernel_kernel_0b2fc070b9, cudaFuncAttributeMaxDynamicSharedMemorySize, 1280 * sizeof(__float128));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_0b2fc070b9<<<grid,block,1280 * sizeof(__float128),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 2×2(2×2) {0..2}×{0..2} strided
    // m1 2×2(2×2) {0..2}×{0..2} strided
    // m2 2×2(2×2) {0..2}×{0..2} strided
    // m0 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, 1] = m1 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, -1]×m2 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<__float128*>(totalShrMemPtr);
      __float128* localShrMem0 = &totalShrMem[10 * threadIdx.y + 0];
      __float128* tempShrMem = &localShrMem0[8];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          __float128 *const __restrict__ glb_m0 = &m0[batchId0 * 4 + 0 + m0_extraOffset];
          const __float128 *const __restrict__ glb_m1 = &m1[batchId0 * 4 + 0 + m1_extraOffset];
          const __float128 *const __restrict__ glb_m2 = &m2[batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 2);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 2; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 2);
              __float128 v12_data;
              {
                v12_data = __ldcg(&glb_m1[v11_a]);
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          __float128* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          pipeline.producer_acquire();
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], cuda::aligned_size_t<16>(16), pipeline);
          cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 2], &glb_m2[0 + 0 + 1 * threadIdx.x + 2], cuda::aligned_size_t<16>(16), pipeline);
          __syncwarp();
          pipeline.producer_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          __float128 r1[2]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 2), (0, 2)] [(0, 2)]
            __float128 ir1[2]{};
            __float128 v17_data = r0[0];
            __float128 v18_data = s0[0];
            __float128 v20_data = ir1[0];
            ir1[0] = (v20_data + (v17_data * v18_data));
            __float128 v23_data = s0[2];
            __float128 v25_data = ir1[1];
            ir1[1] = (v25_data + (v17_data * v23_data));
            __float128 v30_data = r0[1];
            __float128 v31_data = s0[1];
            __float128 v33_data = ir1[0];
            ir1[0] = (v33_data + (v30_data * v31_data));
            __float128 v36_data = s0[3];
            __float128 v38_data = ir1[1];
            ir1[1] = (v38_data + (v30_data * v36_data));
            #pragma unroll
            for (int32_t v43_n0 = 0; v43_n0 < 1; ++v43_n0) {
              #pragma unroll
              for (int32_t v44_n1 = 0; v44_n1 < 2; ++v44_n1) {
                int32_t v45_a = v43_n0 + v44_n1;
                __float128 v46_data = ir1[v45_a];
                int32_t v47_a = v43_n0 + v44_n1;
                r1[v47_a] = v46_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v50_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v51_i0 = 0; v51_i0 < 1; ++v51_i0) {
            int32_t v59_lead = v50_lead + (v51_i0 * 2);
            #pragma unroll
            for (int32_t v52_i1 = 0; v52_i1 < 2; ++v52_i1) {
              int32_t v53_a = v51_i0 + v52_i1;
              __float128 v54_data = r1[v53_a];
              int32_t v61_a = v59_lead + (v52_i1 * 2);
              glb_m0[v61_a] = v54_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

