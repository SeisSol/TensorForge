// === base name ===
kernel_8e62c970e6f8c212

// === header ===
void launcher_kernel_8e62c970e6f8c212(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8e62c970e6f8c212(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8e62c970e6f8c212, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_8e62c970e6f8c212, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_8e62c970e6f8c212), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_8e62c970e6f8c212, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8e62c970e6f8c212(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×4(8×4) {0..8}×{0..4} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // C = abs(TMP)
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      __syncthreads();
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v4_batchId0 * 64 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v4_batchId0 * 32 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v4_batchId0 * 32 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v4_batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v19_lead = threadIdx.x % 64;
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
              float v29_data = __builtin_nontemporal_load(&glb_m0[(v19_lead + (v21_i1 * 8))]);
              r0[v21_i1] = v29_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m1);
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 4; ++v36_i1) {
              float v44_data = __builtin_nontemporal_load(&glb_m1[(v19_lead + (v36_i1 * 8))]);
              r1[v36_i1] = v44_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[4]{};
          // r3 = load{g>r}(glb_m2);
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v51_i1 = 0; v51_i1 < 4; ++v51_i1) {
              float v59_data = __builtin_nontemporal_load(&glb_m2[(v19_lead + (v51_i1 * 8))]);
              r3[v51_i1] = v59_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[4]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v62_data = r1[0];
          float v63_data = r1[1];
          float v64_data = r1[2];
          float v65_data = r1[3];
          float v66_tp{};
          float v67_tp{};
          float v68_tp{};
          float v69_tp{};
          tensorforge::transpose4x4b32(v66_tp, v67_tp, v68_tp, v69_tp, v62_data, v63_data, v64_data, v65_data);
          tensorforge::VectorT<float, 4> v70_acc{};
          float v71_data = r0[0];
          float v72_data = r0[1];
          float v73_data = r0[2];
          float v74_data = r0[3];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v70_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v72_data, v75_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v73_data, v76_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v74_data, v77_acc, 4, 0, 0);
          float v79_data = r0[4];
          float v80_data = r0[5];
          float v81_data = r0[6];
          float v82_data = r0[7];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v78_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v80_data, v83_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v81_data, v84_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v82_data, v85_acc, 4, 1, 0);
          r2[0] = (v86_acc[0]);
          r2[1] = (v86_acc[1]);
          r2[2] = (v86_acc[2]);
          r2[3] = (v86_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v95_i1 = 0; v95_i1 < 4; ++v95_i1) {
              float v97_data = r2[v95_i1];
              int32_t v104_a = v19_lead + (v95_i1 * 8);
              s0[(v104_a ^ ((v104_a >> 5) & 31))] = v97_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[4]{};
          // r4 = +(r0 * r3) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v109_data = r3[0];
          float v110_data = r3[1];
          float v111_data = r3[2];
          float v112_data = r3[3];
          float v113_tp{};
          float v114_tp{};
          float v115_tp{};
          float v116_tp{};
          tensorforge::transpose4x4b32(v113_tp, v114_tp, v115_tp, v116_tp, v109_data, v110_data, v111_data, v112_data);
          tensorforge::VectorT<float, 4> v117_acc{};
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v71_data, v117_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v72_data, v122_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v73_data, v123_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v74_data, v124_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v79_data, v125_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v80_data, v130_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v81_data, v131_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v82_data, v132_acc, 4, 1, 0);
          r4[0] = (v133_acc[0]);
          r4[1] = (v133_acc[1]);
          r4[2] = (v133_acc[2]);
          r4[3] = (v133_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v142_i1 = 0; v142_i1 < 4; ++v142_i1) {
              float v144_data = r4[v142_i1];
              int32_t v152_a = v19_lead + ((v142_i1 + 4) * 8);
              s0[(v152_a ^ ((v152_a >> 5) & 31))] = v144_data;
            }
          }
          // glb_m3 = abs(s0)
          if (v19_lead < 8) {
            #pragma unroll
            for (int32_t v160_k1 = 0; v160_k1 < 8; ++v160_k1) {
              int32_t v166_a = v160_k1 * 8;
              int32_t v167_a = v19_lead + v166_a;
              float v171_data = s0[(v167_a ^ ((v167_a >> 5) & 31))];
              glb_m3[(v19_lead + v166_a)] = (fabsf(v171_data));
            }
          }
        }
      }
    }
  }
}

