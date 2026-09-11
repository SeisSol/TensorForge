// === base name ===
kernel_d7bb9599e3c2e611

// === header ===
void launcher_kernel_d7bb9599e3c2e611(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_d7bb9599e3c2e611(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d7bb9599e3c2e611, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_d7bb9599e3c2e611, block.x * block.y * block.z, 0));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d7bb9599e3c2e611), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_d7bb9599e3c2e611, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d7bb9599e3c2e611(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 35×4(35×4) {0..35}×{0..4} strided
    // m1 35×8(35×8) {0..35}×{0..8} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
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
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v0_batchId0 * 140 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 280 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v0_batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v14_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
            int32_t v21_lead = v14_lead + (v15_i0 * 32);
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v16_i1 * 35))]);
              r0[(v15_i0 + (v16_i1 * 2))] = v24_data;
            }
          }
          if (v14_lead < 3) {
            int32_t v33_lead = v14_lead + 32_i32;
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
              float v36_data = __builtin_nontemporal_load(&glb_m1[(v33_lead + (v28_i1 * 35))]);
              r0[(1 + (v28_i1 * 2))] = v36_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m2);
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v44_i1 = 0; v44_i1 < 4; ++v44_i1) {
              float v52_data = __builtin_nontemporal_load(&glb_m2[(v14_lead + (v44_i1 * 8))]);
              r1[v44_i1] = v52_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 35), (0, 4)] [(0, 8)]
          float v55_data = r1[0];
          float v56_data = r1[1];
          float v57_data = r1[2];
          float v58_data = r1[3];
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          tensorforge::transpose4x4b32(v59_tp, v60_tp, v61_tp, v62_tp, v55_data, v56_data, v57_data, v58_data);
          tensorforge::VectorT<float, 4> v63_acc{};
          float v64_data = r0[0];
          float v65_data = r0[2];
          float v66_data = r0[4];
          float v67_data = r0[6];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v64_data, v63_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v65_data, v68_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v69_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v70_acc, 3, 0, 0);
          float v72_data = r0[8];
          float v73_data = r0[10];
          float v74_data = r0[12];
          float v75_data = r0[14];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v71_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v78_acc, 3, 1, 0);
          r2[0] = (v79_acc[0]);
          r2[2] = (v79_acc[1]);
          r2[4] = (v79_acc[2]);
          r2[6] = (v79_acc[3]);
          tensorforge::VectorT<float, 4> v84_acc{};
          float v85_data = r0[1];
          float v86_data = r0[3];
          float v87_data = r0[5];
          float v88_data = r0[7];
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v85_data, v84_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v86_data, v89_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v87_data, v90_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v88_data, v91_acc, 3, 0, 0);
          float v93_data = r0[9];
          float v94_data = r0[11];
          float v95_data = r0[13];
          float v96_data = r0[15];
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v93_data, v92_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v94_data, v97_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v95_data, v98_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v96_data, v99_acc, 3, 1, 0);
          r2[1] = (v100_acc[0]);
          r2[3] = (v100_acc[1]);
          r2[5] = (v100_acc[2]);
          r2[7] = (v100_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v108_i0 = 0; v108_i0 < 1; ++v108_i0) {
            int32_t v117_lead = v14_lead + (v108_i0 * 32);
            #pragma unroll
            for (int32_t v109_i1 = 0; v109_i1 < 4; ++v109_i1) {
              float v112_data = r2[(v108_i0 + (v109_i1 * 2))];
              glb_m0[(v117_lead + (v109_i1 * 35))] = v112_data;
            }
          }
          if (v14_lead < 3) {
            int32_t v129_lead = v14_lead + 32_i32;
            #pragma unroll
            for (int32_t v121_i1 = 0; v121_i1 < 4; ++v121_i1) {
              float v124_data = r2[(1 + (v121_i1 * 2))];
              glb_m0[(v129_lead + (v121_i1 * 35))] = v124_data;
            }
          }
        }
      }
    }
  }
}

