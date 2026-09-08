// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08703cce1d, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_08703cce1d), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_08703cce1d, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×32(12×6) {0..12}×{0..6} strided
    // m1 32×32(6×6) {0..6}×{0..6} strided
    // m2 32×32(12×6) {0..12}×{0..6} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
    // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
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
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v14_lead = threadIdx.x % 16;
          if (v14_lead < 12) {
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 6; ++v16_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m0[(v14_lead + (v16_i1 * 12))]);
              r0[v16_i1] = v24_data;
            }
          }
          float r1[6]{};
          // r1 = load{g>r}(glb_m1);
          if (v14_lead < 6) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 6; ++v31_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m1[(v14_lead + (v31_i1 * 6))]);
              r1[v31_i1] = v39_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v14_lead < 12) {
            #pragma unroll
            for (int32_t v46_i1 = 0; v46_i1 < 12; ++v46_i1) {
              float v54_data = __builtin_nontemporal_load(&glb_m3[(v14_lead + (v46_i1 * 12))]);
              r3[v46_i1] = v54_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          float v57_data = r1[0];
          float v58_data = r1[1];
          float v59_data = r1[2];
          float v60_data = r1[3];
          float v61_tp{};
          float v62_tp{};
          float v63_tp{};
          float v64_tp{};
          tensorforge::transpose4x4b32(v61_tp, v62_tp, v63_tp, v64_tp, v57_data, v58_data, v59_data, v60_data);
          tensorforge::VectorT<float, 4> v65_acc{};
          float v66_data = r0[0];
          float v67_data = r0[1];
          float v68_data = r0[2];
          float v69_data = r0[3];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v65_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v71_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v69_data, v72_acc, 2, 0, 0);
          float v74_data = r0[4];
          float v75_data = r0[5];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v73_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v78_acc, 2, 1, 0);
          r2[0] = (v79_acc[0]);
          r2[1] = (v79_acc[1]);
          r2[2] = (v79_acc[2]);
          r2[3] = (v79_acc[3]);
          float v84_data = r1[4];
          float v85_data = r1[5];
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          tensorforge::transpose4x4b32(v88_tp, v89_tp, v90_tp, v91_tp, v84_data, v85_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v92_acc{};
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v66_data, v92_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v67_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v68_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v69_data, v99_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v74_data, v100_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v75_data, v105_acc, 2, 1, 0);
          r2[4] = (v106_acc[0]);
          r2[5] = (v106_acc[1]);
          // wait(r3 = load{g>r}(glb_m3););
          float r4[6]{};
          // r4 = +(r3 * r2) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v110_data = r2[0];
          float v111_data = r2[1];
          float v112_data = r2[2];
          float v113_data = r2[3];
          float v114_tp{};
          float v115_tp{};
          float v116_tp{};
          float v117_tp{};
          tensorforge::transpose4x4b32(v114_tp, v115_tp, v116_tp, v117_tp, v110_data, v111_data, v112_data, v113_data);
          tensorforge::VectorT<float, 4> v118_acc{};
          float v119_data = r3[0];
          float v120_data = r3[1];
          float v121_data = r3[2];
          float v122_data = r3[3];
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v119_data, v118_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v120_data, v123_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v121_data, v124_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v122_data, v125_acc, 2, 0, 0);
          float v127_data = r3[4];
          float v128_data = r3[5];
          float v129_data = r3[6];
          float v130_data = r3[7];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v127_data, v126_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v128_data, v131_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v129_data, v132_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v130_data, v133_acc, 2, 1, 0);
          float v135_data = r3[8];
          float v136_data = r3[9];
          float v137_data = r3[10];
          float v138_data = r3[11];
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v135_data, v134_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v136_data, v139_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v137_data, v140_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v138_data, v141_acc, 2, 2, 0);
          r4[0] = (v142_acc[0]);
          r4[1] = (v142_acc[1]);
          r4[2] = (v142_acc[2]);
          r4[3] = (v142_acc[3]);
          float v147_data = r2[4];
          float v148_data = r2[5];
          float v151_tp{};
          float v152_tp{};
          float v153_tp{};
          float v154_tp{};
          tensorforge::transpose4x4b32(v151_tp, v152_tp, v153_tp, v154_tp, v147_data, v148_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v155_acc{};
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v119_data, v155_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v120_data, v160_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v121_data, v161_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v122_data, v162_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v127_data, v163_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v128_data, v168_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v129_data, v169_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v130_data, v170_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v135_data, v171_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v136_data, v176_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v153_tp, v137_data, v177_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v154_tp, v138_data, v178_acc, 2, 2, 0);
          r4[4] = (v179_acc[0]);
          r4[5] = (v179_acc[1]);
          // glb_m2 = store{r>g}(r4);
          if (v14_lead < 12) {
            #pragma unroll
            for (int32_t v186_i1 = 0; v186_i1 < 6; ++v186_i1) {
              float v188_data = r4[v186_i1];
              glb_m2[(v14_lead + (v186_i1 * 12))] = v188_data;
            }
          }
        }
      }
    }
  }
}

