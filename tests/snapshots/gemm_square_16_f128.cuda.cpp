// === base name ===
kernel_8d814dc4c4955c85

// === header ===
void launcher_kernel_8d814dc4c4955c85(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8d814dc4c4955c85(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (2, 64, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8d814dc4c4955c85, block.x * block.y * block.z, 640 * sizeof(__float128));
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
        cudaFuncSetAttribute(kernel_kernel_8d814dc4c4955c85, cudaFuncAttributeMaxDynamicSharedMemorySize, 640 * sizeof(__float128));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_8d814dc4c4955c85<<<grid,block,640 * sizeof(__float128),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_8d814dc4c4955c85(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 2×2(2×2) {0..2}×{0..2} strided
    // m1 2×2(2×2) {0..2}×{0..2} strided
    // m2 2×2(2×2) {0..2}×{0..2} strided
    // m0 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, 1] = m1 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, -1]×m2 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<__float128*>(totalShrMemPtr);
      __float128* localShrMem0 = &totalShrMem[10 * threadIdx.y + 0];
      __float128* tempShrMem = &localShrMem0[8];
      __float128 * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          __float128 *const __restrict__ glb_m0 = &m0[v4_batchId0 * 4 + 0 + m0_extraOffset];
          const __float128 *const __restrict__ glb_m1 = &m1[v4_batchId0 * 4 + 0 + m1_extraOffset];
          const __float128 *const __restrict__ glb_m2 = &m2[v4_batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v25_lead = v18_lead + (v19_i0 * 2);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 2; ++v20_i1) {
              __float128 v28_data = glb_m1[(v25_lead + (v20_i1 * 2))];
              r0[(v19_i0 + v20_i1)] = v28_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 2], &glb_m2[0 + 0 + 1 * threadIdx.x + 2], 16);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          __float128 r1[2]{};
          __syncwarp(0x00000003u << (threadIdx.y % 16 * 2));
          // r1 = +(r0 * s0) + None
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 ir1[2]{};
          __float128 v37_data = r0[0];
          __float128 v38_data = s0[0];
          __float128 v40_data = ir1[0];
          ir1[0] = (v40_data + (v37_data * v38_data));
          __float128 v43_data = s0[2];
          __float128 v45_data = ir1[1];
          ir1[1] = (v45_data + (v37_data * v43_data));
          __float128 v50_data = r0[1];
          __float128 v51_data = s0[1];
          __float128 v53_data = ir1[0];
          ir1[0] = (v53_data + (v50_data * v51_data));
          __float128 v56_data = s0[3];
          __float128 v58_data = ir1[1];
          ir1[1] = (v58_data + (v50_data * v56_data));
          #pragma unroll
          for (int32_t v63_n0 = 0; v63_n0 < 1; ++v63_n0) {
            #pragma unroll
            for (int32_t v64_n1 = 0; v64_n1 < 2; ++v64_n1) {
              int32_t v65_a = v63_n0 + v64_n1;
              __float128 v66_data = ir1[v65_a];
              r1[v65_a] = v66_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v71_i0 = 0; v71_i0 < 1; ++v71_i0) {
            int32_t v79_lead = v18_lead + (v71_i0 * 2);
            #pragma unroll
            for (int32_t v72_i1 = 0; v72_i1 < 2; ++v72_i1) {
              __float128 v74_data = r1[(v71_i0 + v72_i1)];
              glb_m0[(v79_lead + (v72_i1 * 2))] = v74_data;
            }
          }
          __syncwarp(0x00000003u << (threadIdx.y % 16 * 2));
        }
      }
    }
  }
}

