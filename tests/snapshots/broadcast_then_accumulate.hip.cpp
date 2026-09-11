// === base name ===
kernel_dd1c149d47084556

// === header ===
void launcher_kernel_dd1c149d47084556(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_dd1c149d47084556(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_dd1c149d47084556, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_dd1c149d47084556, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_dd1c149d47084556), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_dd1c149d47084556, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_dd1c149d47084556(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32(32) {0..32} pointer_based
    // m1 32×3(32×3) {0..32}×{0..3} pointer_based
    // m2 32×3(32×3) {0..32}×{0..3} pointer_based
    // t0 32(32) {0..32} strided({0..32})[0] = m0 32(32) {0..32} pointer_based({0..32})[0]
    // t1 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] = m1 32×3(32×3) {0..32}×{0..3} pointer_based({0..32}×{0..3})[0, 1]
    // t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] = t0 32(32) {0..32} strided({0..32})[0]
    // t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] += t1 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1]
    // m2 32×3(32×3) {0..32}×{0..3} pointer_based({0..32}×{0..3})[0, 1] = t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t v0_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v0_batchId0 < numElements0; v0_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1_ahead1 = v0_batchId0 + (gridDim.x * blockDim.y);
        size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v0_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0][0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v0_batchId0][0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v14_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
            float v21_data = __builtin_nontemporal_load(&glb_m0[(v14_lead + (v15_i0 * 32))]);
            r0[v15_i0] = v21_data;
          }
          float r2[3]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v32_lead = v14_lead + (v26_i0 * 32);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 3; ++v27_i1) {
              float v35_data = __builtin_nontemporal_load(&glb_m1[(v32_lead + (v27_i1 * 32))]);
              r2[(v26_i0 + v27_i1)] = v35_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 32)] []
          float v41_data = r0[0];
          float v42_data = r1[0];
          r1[0] = (v42_data + v41_data);
          // wait(r2 = load{g>r}(glb_m1););
          float r3[3]{};
          // r3 = +(r2) + None
          // [(0, 32), (0, 3)] []
          float v48_data = r2[0];
          float v49_data = r3[0];
          r3[0] = (v49_data + v48_data);
          float v51_data = r2[1];
          float v52_data = r3[1];
          r3[1] = (v52_data + v51_data);
          float v54_data = r2[2];
          float v55_data = r3[2];
          r3[2] = (v55_data + v54_data);
          float r4[3]{};
          // r4 = +(r1) + None
          // [(0, 32), (0, 3)] []
          float v61_data = r1[0];
          float v62_data = r4[0];
          r4[0] = (v62_data + v61_data);
          float v65_data = r4[1];
          r4[1] = (v65_data + v61_data);
          float v68_data = r4[2];
          r4[2] = (v68_data + v61_data);
          float r5[3]{};
          // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 3)] []
          float ir5[3]{};
          float v75_data = r3[0];
          float v76_data = ir5[0];
          ir5[0] = (v76_data + v75_data);
          float v78_data = r3[1];
          float v79_data = ir5[1];
          ir5[1] = (v79_data + v78_data);
          float v81_data = r3[2];
          float v82_data = ir5[2];
          ir5[2] = (v82_data + v81_data);
          #pragma unroll
          for (int32_t v87_n0 = 0; v87_n0 < 1; ++v87_n0) {
            #pragma unroll
            for (int32_t v88_n1 = 0; v88_n1 < 3; ++v88_n1) {
              int32_t v89_a = v87_n0 + v88_n1;
              float v90_data = ir5[v89_a];
              float v92_data = r4[v89_a];
              r5[v89_a] = (v92_data + v90_data);
            }
          }
          float r6[3]{};
          // r6 = +(r5) + None
          // [(0, 32), (0, 3)] []
          float v99_data = r5[0];
          float v100_data = r6[0];
          r6[0] = (v100_data + v99_data);
          float v102_data = r5[1];
          float v103_data = r6[1];
          r6[1] = (v103_data + v102_data);
          float v105_data = r5[2];
          float v106_data = r6[2];
          r6[2] = (v106_data + v105_data);
          // glb_m2 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v111_i0 = 0; v111_i0 < 1; ++v111_i0) {
            int32_t v119_lead = v14_lead + (v111_i0 * 32);
            #pragma unroll
            for (int32_t v112_i1 = 0; v112_i1 < 3; ++v112_i1) {
              float v114_data = r6[(v111_i0 + v112_i1)];
              glb_m2[(v119_lead + (v112_i1 * 32))] = v114_data;
            }
          }
        }
      }
    }
  }
}

