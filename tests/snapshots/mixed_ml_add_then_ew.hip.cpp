// === base name ===
kernel_609dd06e89

// === header ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_609dd06e89, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_609dd06e89), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_609dd06e89, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // m4 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v16_lead = threadIdx.x % 64;
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 8; ++v18_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m0[(v16_lead + (v18_i1 * 8))]);
              r0[v18_i1] = v26_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m1);
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + (v33_i1 * 8))]);
              r1[v33_i1] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[8]{};
          // r3 = load{g>r}(glb_m2);
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v48_i1 = 0; v48_i1 < 8; ++v48_i1) {
              float v56_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v48_i1 * 8))]);
              r3[v48_i1] = v56_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v59_data = r1[0];
          float v60_data = r1[1];
          float v61_data = r1[2];
          float v62_data = r1[3];
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          float v66_tp{};
          tensorforge::transpose4x4b32(v63_tp, v64_tp, v65_tp, v66_tp, v59_data, v60_data, v61_data, v62_data);
          tensorforge::VectorT<float, 4> v67_acc{};
          float v68_data = r0[0];
          float v69_data = r0[1];
          float v70_data = r0[2];
          float v71_data = r0[3];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v67_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v69_data, v72_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v70_data, v73_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v74_acc, 4, 0, 0);
          float v76_data = r0[4];
          float v77_data = r0[5];
          float v78_data = r0[6];
          float v79_data = r0[7];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v75_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v77_data, v80_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v78_data, v81_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v82_acc, 4, 1, 0);
          r2[0] = (v83_acc[0]);
          r2[1] = (v83_acc[1]);
          r2[2] = (v83_acc[2]);
          r2[3] = (v83_acc[3]);
          float v88_data = r1[4];
          float v89_data = r1[5];
          float v90_data = r1[6];
          float v91_data = r1[7];
          float v92_tp{};
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          tensorforge::transpose4x4b32(v92_tp, v93_tp, v94_tp, v95_tp, v88_data, v89_data, v90_data, v91_data);
          tensorforge::VectorT<float, 4> v96_acc{};
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v96_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v101_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v102_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v103_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v104_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v109_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v110_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v111_acc, 4, 1, 0);
          r2[4] = (v112_acc[0]);
          r2[5] = (v112_acc[1]);
          r2[6] = (v112_acc[2]);
          r2[7] = (v112_acc[3]);
          float r4[8]{};
          // r4 = load{g>r}(glb_m3);
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v122_i1 = 0; v122_i1 < 8; ++v122_i1) {
              float v130_data = __builtin_nontemporal_load(&glb_m3[(v16_lead + (v122_i1 * 8))]);
              r4[v122_i1] = v130_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          // wait(r4 = load{g>r}(glb_m3););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir5[8]{};
          float v134_data = r4[0];
          float v135_data = r4[1];
          float v136_data = r4[2];
          float v137_data = r4[3];
          float v138_tp{};
          float v139_tp{};
          float v140_tp{};
          float v141_tp{};
          tensorforge::transpose4x4b32(v138_tp, v139_tp, v140_tp, v141_tp, v134_data, v135_data, v136_data, v137_data);
          tensorforge::VectorT<float, 4> v142_acc{};
          float v143_data = r3[0];
          float v144_data = r3[1];
          float v145_data = r3[2];
          float v146_data = r3[3];
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v143_data, v142_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v144_data, v147_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v145_data, v148_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v146_data, v149_acc, 4, 0, 0);
          float v151_data = r3[4];
          float v152_data = r3[5];
          float v153_data = r3[6];
          float v154_data = r3[7];
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v151_data, v150_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v152_data, v155_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v153_data, v156_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v154_data, v157_acc, 4, 1, 0);
          ir5[0] = (v158_acc[0]);
          ir5[1] = (v158_acc[1]);
          ir5[2] = (v158_acc[2]);
          ir5[3] = (v158_acc[3]);
          float v163_data = r4[4];
          float v164_data = r4[5];
          float v165_data = r4[6];
          float v166_data = r4[7];
          float v167_tp{};
          float v168_tp{};
          float v169_tp{};
          float v170_tp{};
          tensorforge::transpose4x4b32(v167_tp, v168_tp, v169_tp, v170_tp, v163_data, v164_data, v165_data, v166_data);
          tensorforge::VectorT<float, 4> v171_acc{};
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v167_tp, v143_data, v171_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v168_tp, v144_data, v176_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v169_tp, v145_data, v177_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v170_tp, v146_data, v178_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v167_tp, v151_data, v179_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v168_tp, v152_data, v184_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v169_tp, v153_data, v185_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v170_tp, v154_data, v186_acc, 4, 1, 0);
          ir5[4] = (v187_acc[0]);
          ir5[5] = (v187_acc[1]);
          ir5[6] = (v187_acc[2]);
          ir5[7] = (v187_acc[3]);
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v196_n1 = 0; v196_n1 < 8; ++v196_n1) {
              float v198_data = ir5[v196_n1];
              float v200_data = r2[v196_n1];
              r5[v196_n1] = (v200_data + v198_data);
            }
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v207_i1 = 0; v207_i1 < 8; ++v207_i1) {
              float v209_data = r5[v207_i1];
              int32_t v216_a = v16_lead + (v207_i1 * 8);
              s0[(v216_a ^ ((v216_a >> 5) & 31))] = v209_data;
            }
          }
          // glb_m4 = abs(s0)
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v224_k1 = 0; v224_k1 < 8; ++v224_k1) {
              int32_t v230_a = v224_k1 * 8;
              int32_t v231_a = v16_lead + v230_a;
              float v235_data = s0[(v231_a ^ ((v231_a >> 5) & 31))];
              glb_m4[(v16_lead + v230_a)] = (fabsf(v235_data));
            }
          }
        }
      }
    }
  }
}

