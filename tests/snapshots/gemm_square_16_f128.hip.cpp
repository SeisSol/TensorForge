// === base name ===
kernel_a7dbbcf5934af403

// === header ===
void launcher_kernel_a7dbbcf5934af403(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a7dbbcf5934af403(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (2, 128, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a7dbbcf5934af403, block.x * block.y * block.z, 256 * sizeof(__float128)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(__float128)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_a7dbbcf5934af403, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(__float128)));
          blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
        }
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a7dbbcf5934af403), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(__float128)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_a7dbbcf5934af403, grid, block, 256 * sizeof(__float128), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a7dbbcf5934af403(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      __float128* localShrMem0 = &totalShrMem[2 * threadIdx.y + 0];
      __float128* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t v3_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v3_batchId0 < numElements0; v3_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v4_ahead1 = v3_batchId0 + (gridDim.x * blockDim.y);
        size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<__float128, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<__float128, tensorforge::GlobalMemspace>)&m0[v3_batchId0 * 4 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace>)&m1[v3_batchId0 * 4 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const __float128, tensorforge::GlobalMemspace>)&m2[v3_batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
            int32_t v24_lead = v17_lead + (v18_i0 * 2);
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 2; ++v19_i1) {
              __float128 v27_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + (v19_i1 * 2))]);
              r0[(v18_i0 + v19_i1)] = v27_data;
            }
          }
          __float128 r1[2]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
            int32_t v39_lead = v17_lead + (v33_i0 * 2);
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 2; ++v34_i1) {
              __float128 v42_data = __builtin_nontemporal_load(&glb_m2[(v39_lead + (v34_i1 * 2))]);
              r1[(v33_i0 + v34_i1)] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          __float128 r2[2]{};
          // r2 = +(r0 * r1) + None
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 v45_data = r0[0];
          __float128 v46_data = r0[1];
          __float128 v47_acc{};
          __float128 v48_acc{};
          __float128 v49_data = r1[0];
          __float128 v50_data = r1[1];
          v47_acc += ((tensorforge::broadcast<2, 1, 0>(v49_data)) * v45_data);
          v47_acc += ((tensorforge::broadcast<2, 1, 1>(v49_data)) * v46_data);
          v48_acc += ((tensorforge::broadcast<2, 1, 0>(v50_data)) * v45_data);
          v48_acc += ((tensorforge::broadcast<2, 1, 1>(v50_data)) * v46_data);
          r2[0] = v47_acc;
          r2[1] = v48_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v62_i0 = 0; v62_i0 < 1; ++v62_i0) {
            int32_t v70_lead = v17_lead + (v62_i0 * 2);
            #pragma unroll
            for (int32_t v63_i1 = 0; v63_i1 < 2; ++v63_i1) {
              __float128 v65_data = r2[(v62_i0 + v63_i1)];
              glb_m0[(v70_lead + (v63_i1 * 2))] = v65_data;
            }
          }
        }
      }
    }
  }
}

