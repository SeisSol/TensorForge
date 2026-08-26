// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ead773dd51, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_ead773dd51), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_ead773dd51, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 32×16(32×16) {0..32}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            int32_t v10_off = v2_lead + 4;
            int32_t v18_off = v2_lead + 4;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v11_a = v4_i1 * 32;
              int32_t v12_a = v10_off + v11_a;
              float v21_data = __builtin_nontemporal_load(&glb_m1[(v18_off + v11_a)]);
              int32_t v22_a = 0 + v4_i1;
              r0[v22_a] = v21_data;
            }
          }
          float r1[8]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v16 = glb_m2[16 + threadIdx.x * 1];
            r1[1] = v16;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[2] = v32;
            float v48 = glb_m2[48 + threadIdx.x * 1];
            r1[3] = v48;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r1[4] = v64;
            float v80 = glb_m2[80 + threadIdx.x * 1];
            r1[5] = v80;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r1[6] = v96;
            float v112 = glb_m2[112 + threadIdx.x * 1];
            r1[7] = v112;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 16)]
          auto& ir2 = r2;
          float v23_data = r1[0];
          float v24_data = r1[1];
          float v25_data = r1[2];
          float v26_data = r1[3];
          float v27_tp{};
          float v28_tp{};
          float v29_tp{};
          float v30_tp{};
          tensorforge::transpose4x4b32(v27_tp, v28_tp, v29_tp, v30_tp, v23_data, v24_data, v25_data, v26_data);
          tensorforge::VectorT<float, 4> v31_acc{};
          float v32_data = r0[0];
          float v33_data = r0[1];
          float v34_data = r0[2];
          float v35_data = r0[3];
          tensorforge::VectorT<float, 4> v36_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v32_data, v31_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v37_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v33_data, v36_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v38_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v34_data, v37_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v39_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v35_data, v38_acc, 2, 0, 0);
          float v40_data = r0[4];
          float v41_data = r0[5];
          float v42_data = r0[6];
          float v43_data = r0[7];
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v40_data, v39_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v41_data, v44_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v42_data, v45_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v43_data, v46_acc, 2, 1, 0);
          float v48_data = r0[8];
          float v49_data = r0[9];
          float v50_data = r0[10];
          float v51_data = r0[11];
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v48_data, v47_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v49_data, v52_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v50_data, v53_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v51_data, v54_acc, 2, 2, 0);
          float v56_data = r0[12];
          float v57_data = r0[13];
          float v58_data = r0[14];
          float v59_data = r0[15];
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v56_data, v55_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v57_data, v60_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v58_data, v61_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v59_data, v62_acc, 2, 3, 0);
          ir2[0] = (v63_acc[0]);
          ir2[1] = (v63_acc[1]);
          ir2[2] = (v63_acc[2]);
          ir2[3] = (v63_acc[3]);
          float v68_data = r1[4];
          float v69_data = r1[5];
          float v70_data = r1[6];
          float v71_data = r1[7];
          float v72_tp{};
          float v73_tp{};
          float v74_tp{};
          float v75_tp{};
          tensorforge::transpose4x4b32(v72_tp, v73_tp, v74_tp, v75_tp, v68_data, v69_data, v70_data, v71_data);
          tensorforge::VectorT<float, 4> v76_acc{};
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v32_data, v76_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v33_data, v81_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v34_data, v82_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v35_data, v83_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v40_data, v84_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v41_data, v89_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v42_data, v90_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v43_data, v91_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v48_data, v92_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v49_data, v97_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v50_data, v98_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v51_data, v99_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v56_data, v100_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v57_data, v105_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v58_data, v106_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v59_data, v107_acc, 2, 3, 0);
          ir2[4] = (v108_acc[0]);
          ir2[5] = (v108_acc[1]);
          ir2[6] = (v108_acc[2]);
          ir2[7] = (v108_acc[3]);
          // glb_m0 = store{r>g}(r2);
          int32_t v115_lead = threadIdx.x % 16;
          if (v115_lead < 12) {
            #pragma unroll
            for (int32_t v117_i1 = 0; v117_i1 < 8; ++v117_i1) {
              int32_t v118_a = 0 + v117_i1;
              float v120_data = r2[v117_i1];
              int32_t v127_a = v115_lead + (v117_i1 * 12);
              glb_m0[v127_a] = v120_data;
            }
          }
          ;
        }
      }
    }
  }
}

