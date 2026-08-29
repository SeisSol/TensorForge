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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v14_i0 = 0; v14_i0 < 1; ++v14_i0) {
            int32_t v20_lead = v13_lead + (v14_i0 * 16);
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v20_lead + (v15_i1 * 16))]);
              r0[(v14_i0 + v15_i1)] = v23_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          tensorforge::VectorT<float, 4> v26_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[0 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[0] = v26_lin;
          tensorforge::VectorT<float, 4> v27_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[64 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[4] = v27_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float v29_data = r1[0];
          float v30_data = r1[1];
          float v31_data = r1[2];
          float v32_data = r1[3];
          float v33_tp{};
          float v34_tp{};
          float v35_tp{};
          float v36_tp{};
          tensorforge::transpose4x4b32(v33_tp, v34_tp, v35_tp, v36_tp, v29_data, v30_data, v31_data, v32_data);
          tensorforge::VectorT<float, 4> v37_acc{};
          float v38_data = r0[0];
          float v39_data = r0[1];
          float v40_data = r0[2];
          float v41_data = r0[3];
          tensorforge::VectorT<float, 4> v42_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v38_data, v37_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v39_data, v42_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v40_data, v43_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v41_data, v44_acc, 2, 0, 0);
          float v46_data = r0[4];
          float v47_data = r0[5];
          float v48_data = r0[6];
          float v49_data = r0[7];
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v46_data, v45_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v47_data, v50_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v48_data, v51_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v49_data, v52_acc, 2, 1, 0);
          float v54_data = r0[8];
          float v55_data = r0[9];
          float v56_data = r0[10];
          float v57_data = r0[11];
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v54_data, v53_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v55_data, v58_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v56_data, v59_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v57_data, v60_acc, 2, 2, 0);
          float v62_data = r0[12];
          float v63_data = r0[13];
          float v64_data = r0[14];
          float v65_data = r0[15];
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v62_data, v61_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v63_data, v66_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v64_data, v67_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v65_data, v68_acc, 2, 3, 0);
          r2[0] = (v69_acc[0]);
          r2[1] = (v69_acc[1]);
          r2[2] = (v69_acc[2]);
          r2[3] = (v69_acc[3]);
          float v74_data = r1[4];
          float v75_data = r1[5];
          float v76_data = r1[6];
          float v77_data = r1[7];
          float v78_tp{};
          float v79_tp{};
          float v80_tp{};
          float v81_tp{};
          tensorforge::transpose4x4b32(v78_tp, v79_tp, v80_tp, v81_tp, v74_data, v75_data, v76_data, v77_data);
          tensorforge::VectorT<float, 4> v82_acc{};
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v38_data, v82_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v39_data, v87_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v40_data, v88_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v41_data, v89_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v46_data, v90_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v47_data, v95_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v48_data, v96_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v49_data, v97_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v54_data, v98_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v55_data, v103_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v56_data, v104_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v57_data, v105_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v62_data, v106_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v63_data, v111_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v64_data, v112_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v65_data, v113_acc, 2, 3, 0);
          r2[4] = (v114_acc[0]);
          r2[5] = (v114_acc[1]);
          r2[6] = (v114_acc[2]);
          r2[7] = (v114_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v122_i0 = 0; v122_i0 < 1; ++v122_i0) {
            int32_t v130_lead = v13_lead + (v122_i0 * 16);
            #pragma unroll
            for (int32_t v123_i1 = 0; v123_i1 < 8; ++v123_i1) {
              float v125_data = r2[(v122_i0 + v123_i1)];
              glb_m0[(v130_lead + (v123_i1 * 16))] = v125_data;
            }
          }
        }
      }
    }
  }
}

