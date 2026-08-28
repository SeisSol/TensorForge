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
          int32_t v6_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 2;
            int32_t v13_lead = v6_lead + v12_lead;
            int32_t v20_lead = v6_lead + v12_lead;
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 2; ++v8_i1) {
              int32_t v14_a = v8_i1 * 2;
              int32_t v15_a = v13_lead + v14_a;
              __float128 v23_data = __ldcg(&glb_m1[(v20_lead + v14_a)]);
              int32_t v24_a = v7_i0 + v8_i1;
              r0[v24_a] = v23_data;
            }
          }
          __float128* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 16);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 2], &glb_m2[0 + 0 + 1 * threadIdx.x + 2], 16);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          __float128 r1[2]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 ir1[2]{};
          __float128 v33_data = r0[0];
          __float128 v34_data = s0[0];
          __float128 v36_data = ir1[0];
          ir1[0] = (v36_data + (v33_data * v34_data));
          __float128 v39_data = s0[2];
          __float128 v41_data = ir1[1];
          ir1[1] = (v41_data + (v33_data * v39_data));
          __float128 v46_data = r0[1];
          __float128 v47_data = s0[1];
          __float128 v49_data = ir1[0];
          ir1[0] = (v49_data + (v46_data * v47_data));
          __float128 v52_data = s0[3];
          __float128 v54_data = ir1[1];
          ir1[1] = (v54_data + (v46_data * v52_data));
          #pragma unroll
          for (int32_t v59_n0 = 0; v59_n0 < 1; ++v59_n0) {
            #pragma unroll
            for (int32_t v60_n1 = 0; v60_n1 < 2; ++v60_n1) {
              int32_t v61_a = v59_n0 + v60_n1;
              int32_t v62_a = v59_n0 + v60_n1;
              __float128 v63_data = ir1[v62_a];
              r1[v62_a] = v63_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
            int32_t v77_lead = v6_lead + (v68_i0 * 2);
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 2; ++v69_i1) {
              int32_t v70_a = v68_i0 + v69_i1;
              __float128 v72_data = r1[(v68_i0 + v69_i1)];
              glb_m0[(v77_lead + (v69_i1 * 2))] = v72_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

