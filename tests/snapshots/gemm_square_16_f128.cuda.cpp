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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          __float128 *const __restrict__ glb_m0 = &m0[batchId0 * 4 + 0 + m0_extraOffset];
          const __float128 *const __restrict__ glb_m1 = &m1[batchId0 * 4 + 0 + m1_extraOffset];
          const __float128 *const __restrict__ glb_m2 = &m2[batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 2);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 2; ++v12_i1) {
              __float128 v20_data = __ldcg(&glb_m1[(v17_lead + (v12_i1 * 2))]);
              r0[(v11_i0 + v12_i1)] = v20_data;
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
          __float128 v30_data = r0[0];
          __float128 v31_data = s0[0];
          __float128 v33_data = ir1[0];
          ir1[0] = (v33_data + (v30_data * v31_data));
          __float128 v36_data = s0[2];
          __float128 v38_data = ir1[1];
          ir1[1] = (v38_data + (v30_data * v36_data));
          __float128 v43_data = r0[1];
          __float128 v44_data = s0[1];
          __float128 v46_data = ir1[0];
          ir1[0] = (v46_data + (v43_data * v44_data));
          __float128 v49_data = s0[3];
          __float128 v51_data = ir1[1];
          ir1[1] = (v51_data + (v43_data * v49_data));
          #pragma unroll
          for (int32_t v56_n0 = 0; v56_n0 < 1; ++v56_n0) {
            #pragma unroll
            for (int32_t v57_n1 = 0; v57_n1 < 2; ++v57_n1) {
              int32_t v58_a = v56_n0 + v57_n1;
              __float128 v59_data = ir1[v58_a];
              r1[v58_a] = v59_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v64_i0 = 0; v64_i0 < 1; ++v64_i0) {
            int32_t v72_lead = v10_lead + (v64_i0 * 2);
            #pragma unroll
            for (int32_t v65_i1 = 0; v65_i1 < 2; ++v65_i1) {
              __float128 v67_data = r1[(v64_i0 + v65_i1)];
              glb_m0[(v72_lead + (v65_i1 * 2))] = v67_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

