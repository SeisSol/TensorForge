// === base name ===
kernel_5c7e4b9ce8

// === header ===
void launcher_kernel_5c7e4b9ce8(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5c7e4b9ce8(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5c7e4b9ce8, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_5c7e4b9ce8), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_5c7e4b9ce8, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5c7e4b9ce8(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×11(16×11) {0..16}×{0..11} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×11(16×11) {0..16}×{0..11} strided
    // m0 16×11(16×11) {0..16}×{0..11} strided({0..16}×{0..11})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×11(16×11) {0..16}×{0..11} strided({0..16}×{0..11})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 176 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 176 + 0 + m2_extraOffset];
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
          float r1[11]{};
          // r1 = load{g>r}(glb_m2);
          tensorforge::VectorT<float, 4> v26_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[0 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[0] = v26_lin;
          tensorforge::VectorT<float, 4> v27_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[64 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[4] = v27_lin;
          tensorforge::VectorT<float, 2> v28_lin = *(tensorforge::VectorT<float, 2>*)&glb_m2[128 + threadIdx.x * 2];
          *(tensorforge::VectorRelaxedT<float, 2>*)&r1[8] = v28_lin;
          float v29_lin = glb_m2[160 + threadIdx.x * 1];
          r1[10] = v29_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[11]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 11)] [(0, 16)]
          float v31_data = r1[0];
          float v32_data = r1[1];
          float v33_data = r1[2];
          float v34_data = r1[3];
          float v35_tp{};
          float v36_tp{};
          float v37_tp{};
          float v38_tp{};
          tensorforge::transpose4x4b32(v35_tp, v36_tp, v37_tp, v38_tp, v31_data, v32_data, v33_data, v34_data);
          tensorforge::VectorT<float, 4> v39_acc{};
          float v40_data = r0[0];
          float v41_data = r0[1];
          float v42_data = r0[2];
          float v43_data = r0[3];
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v40_data, v39_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v41_data, v44_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v42_data, v45_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v43_data, v46_acc, 2, 0, 0);
          float v48_data = r0[4];
          float v49_data = r0[5];
          float v50_data = r0[6];
          float v51_data = r0[7];
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v48_data, v47_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v49_data, v52_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v50_data, v53_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v51_data, v54_acc, 2, 1, 0);
          float v56_data = r0[8];
          float v57_data = r0[9];
          float v58_data = r0[10];
          float v59_data = r0[11];
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v56_data, v55_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v57_data, v60_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v58_data, v61_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v59_data, v62_acc, 2, 2, 0);
          float v64_data = r0[12];
          float v65_data = r0[13];
          float v66_data = r0[14];
          float v67_data = r0[15];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v64_data, v63_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v65_data, v68_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v66_data, v69_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v67_data, v70_acc, 2, 3, 0);
          r2[0] = (v71_acc[0]);
          r2[1] = (v71_acc[1]);
          r2[2] = (v71_acc[2]);
          r2[3] = (v71_acc[3]);
          float v76_data = r1[4];
          float v77_data = r1[5];
          float v78_data = r1[6];
          float v79_data = r1[7];
          float v80_tp{};
          float v81_tp{};
          float v82_tp{};
          float v83_tp{};
          tensorforge::transpose4x4b32(v80_tp, v81_tp, v82_tp, v83_tp, v76_data, v77_data, v78_data, v79_data);
          tensorforge::VectorT<float, 4> v84_acc{};
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v40_data, v84_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v41_data, v89_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v42_data, v90_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v43_data, v91_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v48_data, v92_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v49_data, v97_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v50_data, v98_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v51_data, v99_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v56_data, v100_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v57_data, v105_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v58_data, v106_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v59_data, v107_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v64_data, v108_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v65_data, v113_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v66_data, v114_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v67_data, v115_acc, 2, 3, 0);
          r2[4] = (v116_acc[0]);
          r2[5] = (v116_acc[1]);
          r2[6] = (v116_acc[2]);
          r2[7] = (v116_acc[3]);
          float v121_data = r1[8];
          float v122_data = r1[9];
          float v123_data = r1[10];
          float v125_tp{};
          float v126_tp{};
          float v127_tp{};
          float v128_tp{};
          tensorforge::transpose4x4b32(v125_tp, v126_tp, v127_tp, v128_tp, v121_data, v122_data, v123_data, 0.0f);
          tensorforge::VectorT<float, 4> v129_acc{};
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v40_data, v129_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v41_data, v134_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v42_data, v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v43_data, v136_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v48_data, v137_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v49_data, v142_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v50_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v51_data, v144_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v56_data, v145_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v57_data, v150_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v58_data, v151_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v59_data, v152_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v64_data, v153_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v65_data, v158_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v66_data, v159_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v67_data, v160_acc, 2, 3, 0);
          r2[8] = (v161_acc[0]);
          r2[9] = (v161_acc[1]);
          r2[10] = (v161_acc[2]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v168_i0 = 0; v168_i0 < 1; ++v168_i0) {
            int32_t v176_lead = v13_lead + (v168_i0 * 16);
            #pragma unroll
            for (int32_t v169_i1 = 0; v169_i1 < 11; ++v169_i1) {
              float v171_data = r2[(v168_i0 + v169_i1)];
              glb_m0[(v176_lead + (v169_i1 * 16))] = v171_data;
            }
          }
        }
      }
    }
  }
}

