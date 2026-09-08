// === base name ===
kernel_924fd3d329

// === header ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_924fd3d329, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_924fd3d329), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_924fd3d329, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×4(8×4) {0..8}×{0..4} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // C = abs(TMP)
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      __syncthreads();
      float* __restrict__ s0 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 32 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 64;
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 8; ++v17_i1) {
              float v25_data = __builtin_nontemporal_load(&glb_m0[(v15_lead + (v17_i1 * 8))]);
              r0[v17_i1] = v25_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m1);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 4; ++v32_i1) {
              float v40_data = __builtin_nontemporal_load(&glb_m1[(v15_lead + (v32_i1 * 8))]);
              r1[v32_i1] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[4]{};
          // r3 = load{g>r}(glb_m2);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v47_i1 = 0; v47_i1 < 4; ++v47_i1) {
              float v55_data = __builtin_nontemporal_load(&glb_m2[(v15_lead + (v47_i1 * 8))]);
              r3[v47_i1] = v55_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[4]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v58_data = r1[0];
          float v59_data = r1[1];
          float v60_data = r1[2];
          float v61_data = r1[3];
          float v62_tp{};
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          tensorforge::transpose4x4b32(v62_tp, v63_tp, v64_tp, v65_tp, v58_data, v59_data, v60_data, v61_data);
          tensorforge::VectorT<float, 4> v66_acc{};
          float v67_data = r0[0];
          float v68_data = r0[1];
          float v69_data = r0[2];
          float v70_data = r0[3];
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v66_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v71_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v69_data, v72_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v70_data, v73_acc, 4, 0, 0);
          float v75_data = r0[4];
          float v76_data = r0[5];
          float v77_data = r0[6];
          float v78_data = r0[7];
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v74_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v79_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v77_data, v80_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v78_data, v81_acc, 4, 1, 0);
          r2[0] = (v82_acc[0]);
          r2[1] = (v82_acc[1]);
          r2[2] = (v82_acc[2]);
          r2[3] = (v82_acc[3]);
          // s0 = store{r>s}(localShrMem0, r2);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v91_i1 = 0; v91_i1 < 4; ++v91_i1) {
              float v93_data = r2[v91_i1];
              int32_t v100_a = v15_lead + (v91_i1 * 8);
              s0[(v100_a ^ ((v100_a >> 5) & 31))] = v93_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[4]{};
          // r4 = +(r0 * r3) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v105_data = r3[0];
          float v106_data = r3[1];
          float v107_data = r3[2];
          float v108_data = r3[3];
          float v109_tp{};
          float v110_tp{};
          float v111_tp{};
          float v112_tp{};
          tensorforge::transpose4x4b32(v109_tp, v110_tp, v111_tp, v112_tp, v105_data, v106_data, v107_data, v108_data);
          tensorforge::VectorT<float, 4> v113_acc{};
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v67_data, v113_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v68_data, v118_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v69_data, v119_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v70_data, v120_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v75_data, v121_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v76_data, v126_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v77_data, v127_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v78_data, v128_acc, 4, 1, 0);
          r4[0] = (v129_acc[0]);
          r4[1] = (v129_acc[1]);
          r4[2] = (v129_acc[2]);
          r4[3] = (v129_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v138_i1 = 0; v138_i1 < 4; ++v138_i1) {
              float v140_data = r4[v138_i1];
              int32_t v148_a = v15_lead + ((v138_i1 + 4) * 8);
              s0[(v148_a ^ ((v148_a >> 5) & 31))] = v140_data;
            }
          }
          // glb_m3 = abs(s0)
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v156_k1 = 0; v156_k1 < 8; ++v156_k1) {
              int32_t v162_a = v156_k1 * 8;
              int32_t v163_a = v15_lead + v162_a;
              float v167_data = s0[(v163_a ^ ((v163_a >> 5) & 31))];
              glb_m3[(v15_lead + v162_a)] = (fabsf(v167_data));
            }
          }
        }
      }
    }
  }
}

