// === base name ===
kernel_21138a3fa2

// === header ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_21138a3fa2, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_21138a3fa2), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_21138a3fa2, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 16;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
              int32_t v15_a = v9_i1 * 16;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + v9_i1)] = v24_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          tensorforge::VectorT<float, 4> v27_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[0 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[0] = v27_lin;
          tensorforge::VectorT<float, 4> v28_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[64 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[4] = v28_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float v30_data = r1[0];
          float v31_data = r1[1];
          float v32_data = r1[2];
          float v33_data = r1[3];
          float v34_tp{};
          float v35_tp{};
          float v36_tp{};
          float v37_tp{};
          tensorforge::transpose4x4b32(v34_tp, v35_tp, v36_tp, v37_tp, v30_data, v31_data, v32_data, v33_data);
          tensorforge::VectorT<float, 4> v38_acc{};
          float v39_data = r0[0];
          float v40_data = r0[1];
          float v41_data = r0[2];
          float v42_data = r0[3];
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v39_data, v38_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v40_data, v43_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v41_data, v44_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v42_data, v45_acc, 2, 0, 0);
          float v47_data = r0[4];
          float v48_data = r0[5];
          float v49_data = r0[6];
          float v50_data = r0[7];
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v47_data, v46_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v48_data, v51_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v49_data, v52_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v50_data, v53_acc, 2, 1, 0);
          float v55_data = r0[8];
          float v56_data = r0[9];
          float v57_data = r0[10];
          float v58_data = r0[11];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v55_data, v54_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v56_data, v59_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v57_data, v60_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v58_data, v61_acc, 2, 2, 0);
          float v63_data = r0[12];
          float v64_data = r0[13];
          float v65_data = r0[14];
          float v66_data = r0[15];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v63_data, v62_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v64_data, v67_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v65_data, v68_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v66_data, v69_acc, 2, 3, 0);
          r2[0] = (v70_acc[0]);
          r2[1] = (v70_acc[1]);
          r2[2] = (v70_acc[2]);
          r2[3] = (v70_acc[3]);
          float v75_data = r1[4];
          float v76_data = r1[5];
          float v77_data = r1[6];
          float v78_data = r1[7];
          float v79_tp{};
          float v80_tp{};
          float v81_tp{};
          float v82_tp{};
          tensorforge::transpose4x4b32(v79_tp, v80_tp, v81_tp, v82_tp, v75_data, v76_data, v77_data, v78_data);
          tensorforge::VectorT<float, 4> v83_acc{};
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v39_data, v83_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v40_data, v88_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v41_data, v89_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v42_data, v90_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v47_data, v91_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v48_data, v96_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v49_data, v97_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v50_data, v98_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v55_data, v99_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v56_data, v104_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v57_data, v105_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v58_data, v106_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v63_data, v107_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v64_data, v112_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v65_data, v113_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v66_data, v114_acc, 2, 3, 0);
          r2[4] = (v115_acc[0]);
          r2[5] = (v115_acc[1]);
          r2[6] = (v115_acc[2]);
          r2[7] = (v115_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v123_i0 = 0; v123_i0 < 1; ++v123_i0) {
            int32_t v132_lead = v7_lead + (v123_i0 * 16);
            #pragma unroll
            for (int32_t v124_i1 = 0; v124_i1 < 8; ++v124_i1) {
              int32_t v125_a = v123_i0 + v124_i1;
              float v127_data = r2[(v123_i0 + v124_i1)];
              glb_m0[(v132_lead + (v124_i1 * 16))] = v127_data;
            }
          }
        }
      }
    }
  }
}

