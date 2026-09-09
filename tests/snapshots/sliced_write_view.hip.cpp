// === base name ===
kernel_94d7c511463a37d1

// === header ===
void launcher_kernel_94d7c511463a37d1(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_94d7c511463a37d1(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_94d7c511463a37d1, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_94d7c511463a37d1), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_94d7c511463a37d1, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_94d7c511463a37d1(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
    // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v19_lead = v12_lead + (v13_i0 * 32);
            #pragma unroll
            for (int32_t v14_i1 = 10; v14_i1 < 13; ++v14_i1) {
              float v22_data = __builtin_nontemporal_load(&glb_m1[(v19_lead + (v14_i1 * 32))]);
              r0[(v13_i0 + (v14_i1 - 10))] = v22_data;
            }
          }
          float r1[1]{};
          // r1 = load{g>r}(glb_m2);
          if ((v12_lead >= 10) && (v12_lead < 13)) {
            #pragma unroll
            for (int32_t v32_i1 = 8; v32_i1 < 9; ++v32_i1) {
              float v40_data = __builtin_nontemporal_load(&glb_m2[(v12_lead + (v32_i1 * 13))]);
              r1[(v32_i1 - 8)] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float v44_data = r0[0];
          float v45_data = r0[1];
          float v46_data = r0[2];
          float v47_acc{};
          float v48_data = r1[0];
          float v49_bc = tensorforge::broadcast<32, 16, 0>(v48_data);
          tensorforge::fmacdpp16<10>(v47_acc, v49_bc, v44_data);
          tensorforge::fmacdpp16<11>(v47_acc, v49_bc, v45_data);
          tensorforge::fmacdpp16<12>(v47_acc, v49_bc, v46_data);
          r2[0] = v47_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v53_i0 = 0; v53_i0 < 1; ++v53_i0) {
            int32_t v61_lead = v12_lead + (v53_i0 * 32);
            #pragma unroll
            for (int32_t v54_i1 = 0; v54_i1 < 1; ++v54_i1) {
              float v56_data = r2[(v53_i0 + v54_i1)];
              glb_m0[(v61_lead + ((v54_i1 + 8) * 32))] = v56_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v69_i0 = 0; v69_i0 < 1; ++v69_i0) {
            int32_t v75_lead = v12_lead + (v69_i0 * 32);
            #pragma unroll
            for (int32_t v70_i1 = 0; v70_i1 < 13; ++v70_i1) {
              float v78_data = glb_m0[(v75_lead + (v70_i1 * 32))];
              r3[(v69_i0 + v70_i1)] = v78_data;
            }
          }
          float r4[13]{};
          // r4 = load{g>r}(glb_m4);
          if (v12_lead < 13) {
            #pragma unroll
            for (int32_t v85_i1 = 0; v85_i1 < 13; ++v85_i1) {
              float v93_data = __builtin_nontemporal_load(&glb_m4[(v12_lead + (v85_i1 * 13))]);
              r4[v85_i1] = v93_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v96_data = r4[0];
          float v97_data = r4[1];
          float v98_data = r4[2];
          float v99_data = r4[3];
          float v100_tp{};
          float v101_tp{};
          float v102_tp{};
          float v103_tp{};
          tensorforge::transpose4x4b32(v100_tp, v101_tp, v102_tp, v103_tp, v96_data, v97_data, v98_data, v99_data);
          tensorforge::VectorT<float, 4> v104_acc{};
          float v105_data = r3[0];
          float v106_data = r3[1];
          float v107_data = r3[2];
          float v108_data = r3[3];
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v105_data, v104_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v106_data, v109_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v107_data, v110_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v108_data, v111_acc, 3, 0, 0);
          float v113_data = r3[4];
          float v114_data = r3[5];
          float v115_data = r3[6];
          float v116_data = r3[7];
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v113_data, v112_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v114_data, v117_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v115_data, v118_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v116_data, v119_acc, 3, 1, 0);
          float v121_data = r3[8];
          float v122_data = r3[9];
          float v123_data = r3[10];
          float v124_data = r3[11];
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v121_data, v120_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v122_data, v125_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v102_tp, v123_data, v126_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v124_data, v127_acc, 3, 2, 0);
          float v129_data = r3[12];
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v129_data, v128_acc, 3, 3, 0);
          r5[0] = (v133_acc[0]);
          r5[1] = (v133_acc[1]);
          r5[2] = (v133_acc[2]);
          r5[3] = (v133_acc[3]);
          float v138_data = r4[4];
          float v139_data = r4[5];
          float v140_data = r4[6];
          float v141_data = r4[7];
          float v142_tp{};
          float v143_tp{};
          float v144_tp{};
          float v145_tp{};
          tensorforge::transpose4x4b32(v142_tp, v143_tp, v144_tp, v145_tp, v138_data, v139_data, v140_data, v141_data);
          tensorforge::VectorT<float, 4> v146_acc{};
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v105_data, v146_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v106_data, v151_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v107_data, v152_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v108_data, v153_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v113_data, v154_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v114_data, v159_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v115_data, v160_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v116_data, v161_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v121_data, v162_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v122_data, v167_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v123_data, v168_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v124_data, v169_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v129_data, v170_acc, 3, 3, 0);
          r5[4] = (v175_acc[0]);
          r5[5] = (v175_acc[1]);
          r5[6] = (v175_acc[2]);
          r5[7] = (v175_acc[3]);
          float v180_data = r4[8];
          float v181_data = r4[9];
          float v182_data = r4[10];
          float v183_data = r4[11];
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          float v187_tp{};
          tensorforge::transpose4x4b32(v184_tp, v185_tp, v186_tp, v187_tp, v180_data, v181_data, v182_data, v183_data);
          tensorforge::VectorT<float, 4> v188_acc{};
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v105_data, v188_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v106_data, v193_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v107_data, v194_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v108_data, v195_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v113_data, v196_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v114_data, v201_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v115_data, v202_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v116_data, v203_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v121_data, v204_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v122_data, v209_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v123_data, v210_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v124_data, v211_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v129_data, v212_acc, 3, 3, 0);
          r5[8] = (v217_acc[0]);
          r5[9] = (v217_acc[1]);
          r5[10] = (v217_acc[2]);
          r5[11] = (v217_acc[3]);
          float v235_acc{};
          float v236_data = r4[12];
          float v237_bc = tensorforge::broadcast<32, 16, 0>(v236_data);
          tensorforge::fmacdpp16<0>(v235_acc, v237_bc, v105_data);
          tensorforge::fmacdpp16<1>(v235_acc, v237_bc, v106_data);
          tensorforge::fmacdpp16<2>(v235_acc, v237_bc, v107_data);
          tensorforge::fmacdpp16<3>(v235_acc, v237_bc, v108_data);
          tensorforge::fmacdpp16<4>(v235_acc, v237_bc, v113_data);
          tensorforge::fmacdpp16<5>(v235_acc, v237_bc, v114_data);
          tensorforge::fmacdpp16<6>(v235_acc, v237_bc, v115_data);
          tensorforge::fmacdpp16<7>(v235_acc, v237_bc, v116_data);
          tensorforge::fmacdpp16<8>(v235_acc, v237_bc, v121_data);
          tensorforge::fmacdpp16<9>(v235_acc, v237_bc, v122_data);
          tensorforge::fmacdpp16<10>(v235_acc, v237_bc, v123_data);
          tensorforge::fmacdpp16<11>(v235_acc, v237_bc, v124_data);
          tensorforge::fmacdpp16<12>(v235_acc, v237_bc, v129_data);
          r5[12] = v235_acc;
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v241_i0 = 0; v241_i0 < 1; ++v241_i0) {
            int32_t v249_lead = v12_lead + (v241_i0 * 32);
            #pragma unroll
            for (int32_t v242_i1 = 0; v242_i1 < 13; ++v242_i1) {
              float v244_data = r5[(v241_i0 + v242_i1)];
              glb_m3[(v249_lead + (v242_i1 * 32))] = v244_data;
            }
          }
        }
      }
    }
  }
}

