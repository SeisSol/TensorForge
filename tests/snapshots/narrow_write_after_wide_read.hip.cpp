// === base name ===
kernel_9ba443dd99a9b6cf

// === header ===
void launcher_kernel_9ba443dd99a9b6cf(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_9ba443dd99a9b6cf(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9ba443dd99a9b6cf, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_9ba443dd99a9b6cf), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_9ba443dd99a9b6cf, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_9ba443dd99a9b6cf(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×12(32×12) {0..32}×{0..12} strided
    // m2 12×13(12×13) {0..12}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1]
    // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] += m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×13(12×13) {0..12}×{0..13} strided({0..12}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1]
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 156 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v19_lead = v12_lead + (v13_i0 * 32);
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 13; ++v14_i1) {
              float v22_data = glb_m0[(v19_lead + (v14_i1 * 32))];
              r0[(v13_i0 + v14_i1)] = v22_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
            int32_t v34_lead = v12_lead + (v28_i0 * 32);
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v34_lead + (v29_i1 * 32))]);
              r2[(v28_i0 + v29_i1)] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v43_data = r0[0];
          float v44_data = r1[0];
          r1[0] = (v44_data + v43_data);
          float v46_data = r0[1];
          float v47_data = r1[1];
          r1[1] = (v47_data + v46_data);
          float v49_data = r0[2];
          float v50_data = r1[2];
          r1[2] = (v50_data + v49_data);
          float v52_data = r0[3];
          float v53_data = r1[3];
          r1[3] = (v53_data + v52_data);
          float v55_data = r0[4];
          float v56_data = r1[4];
          r1[4] = (v56_data + v55_data);
          float v58_data = r0[5];
          float v59_data = r1[5];
          r1[5] = (v59_data + v58_data);
          float v61_data = r0[6];
          float v62_data = r1[6];
          r1[6] = (v62_data + v61_data);
          float v64_data = r0[7];
          float v65_data = r1[7];
          r1[7] = (v65_data + v64_data);
          float v67_data = r0[8];
          float v68_data = r1[8];
          r1[8] = (v68_data + v67_data);
          float v70_data = r0[9];
          float v71_data = r1[9];
          r1[9] = (v71_data + v70_data);
          float v73_data = r0[10];
          float v74_data = r1[10];
          r1[10] = (v74_data + v73_data);
          float v76_data = r0[11];
          float v77_data = r1[11];
          r1[11] = (v77_data + v76_data);
          float v79_data = r0[12];
          float v80_data = r1[12];
          r1[12] = (v80_data + v79_data);
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v87_i1 = 0; v87_i1 < 13; ++v87_i1) {
              float v95_data = __builtin_nontemporal_load(&glb_m2[(v12_lead + (v87_i1 * 12))]);
              r3[v87_i1] = v95_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir4[13]{};
          float v99_data = r3[0];
          float v100_data = r3[1];
          float v101_data = r3[2];
          float v102_data = r3[3];
          float v103_tp{};
          float v104_tp{};
          float v105_tp{};
          float v106_tp{};
          tensorforge::transpose4x4b32(v103_tp, v104_tp, v105_tp, v106_tp, v99_data, v100_data, v101_data, v102_data);
          tensorforge::VectorT<float, 4> v107_acc{};
          float v108_data = r2[0];
          float v109_data = r2[1];
          float v110_data = r2[2];
          float v111_data = r2[3];
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v108_data, v107_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v109_data, v112_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v110_data, v113_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v111_data, v114_acc, 3, 0, 0);
          float v116_data = r2[4];
          float v117_data = r2[5];
          float v118_data = r2[6];
          float v119_data = r2[7];
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v116_data, v115_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v117_data, v120_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v118_data, v121_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v119_data, v122_acc, 3, 1, 0);
          float v124_data = r2[8];
          float v125_data = r2[9];
          float v126_data = r2[10];
          float v127_data = r2[11];
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v124_data, v123_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v125_data, v128_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v126_data, v129_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v127_data, v130_acc, 3, 2, 0);
          ir4[0] = (v131_acc[0]);
          ir4[1] = (v131_acc[1]);
          ir4[2] = (v131_acc[2]);
          ir4[3] = (v131_acc[3]);
          float v136_data = r3[4];
          float v137_data = r3[5];
          float v138_data = r3[6];
          float v139_data = r3[7];
          float v140_tp{};
          float v141_tp{};
          float v142_tp{};
          float v143_tp{};
          tensorforge::transpose4x4b32(v140_tp, v141_tp, v142_tp, v143_tp, v136_data, v137_data, v138_data, v139_data);
          tensorforge::VectorT<float, 4> v144_acc{};
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v108_data, v144_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v109_data, v149_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v110_data, v150_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v111_data, v151_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v116_data, v152_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v117_data, v157_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v118_data, v158_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v119_data, v159_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v124_data, v160_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v125_data, v165_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v126_data, v166_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v127_data, v167_acc, 3, 2, 0);
          ir4[4] = (v168_acc[0]);
          ir4[5] = (v168_acc[1]);
          ir4[6] = (v168_acc[2]);
          ir4[7] = (v168_acc[3]);
          float v173_data = r3[8];
          float v174_data = r3[9];
          float v175_data = r3[10];
          float v176_data = r3[11];
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          float v180_tp{};
          tensorforge::transpose4x4b32(v177_tp, v178_tp, v179_tp, v180_tp, v173_data, v174_data, v175_data, v176_data);
          tensorforge::VectorT<float, 4> v181_acc{};
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v108_data, v181_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v109_data, v186_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v110_data, v187_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v111_data, v188_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v116_data, v189_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v117_data, v194_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v118_data, v195_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v119_data, v196_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v124_data, v197_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v125_data, v202_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v126_data, v203_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v127_data, v204_acc, 3, 2, 0);
          ir4[8] = (v205_acc[0]);
          ir4[9] = (v205_acc[1]);
          ir4[10] = (v205_acc[2]);
          ir4[11] = (v205_acc[3]);
          float v222_acc{};
          float v223_data = r3[12];
          float v224_bc = tensorforge::broadcast<32, 16, 0>(v223_data);
          tensorforge::fmacdpp16<0>(v222_acc, v224_bc, v108_data);
          tensorforge::fmacdpp16<1>(v222_acc, v224_bc, v109_data);
          tensorforge::fmacdpp16<2>(v222_acc, v224_bc, v110_data);
          tensorforge::fmacdpp16<3>(v222_acc, v224_bc, v111_data);
          tensorforge::fmacdpp16<4>(v222_acc, v224_bc, v116_data);
          tensorforge::fmacdpp16<5>(v222_acc, v224_bc, v117_data);
          tensorforge::fmacdpp16<6>(v222_acc, v224_bc, v118_data);
          tensorforge::fmacdpp16<7>(v222_acc, v224_bc, v119_data);
          tensorforge::fmacdpp16<8>(v222_acc, v224_bc, v124_data);
          tensorforge::fmacdpp16<9>(v222_acc, v224_bc, v125_data);
          tensorforge::fmacdpp16<10>(v222_acc, v224_bc, v126_data);
          tensorforge::fmacdpp16<11>(v222_acc, v224_bc, v127_data);
          ir4[12] = v222_acc;
          #pragma unroll
          for (int32_t v228_n0 = 0; v228_n0 < 1; ++v228_n0) {
            #pragma unroll
            for (int32_t v229_n1 = 0; v229_n1 < 13; ++v229_n1) {
              int32_t v230_a = v228_n0 + v229_n1;
              float v231_data = ir4[v230_a];
              float v233_data = r1[v230_a];
              r4[v230_a] = (v233_data + v231_data);
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          float v240_data = r4[4];
          float v241_data = r5[0];
          r5[0] = (v241_data + v240_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v246_i0 = 0; v246_i0 < 1; ++v246_i0) {
            int32_t v254_lead = v12_lead + (v246_i0 * 32);
            #pragma unroll
            for (int32_t v247_i1 = 0; v247_i1 < 1; ++v247_i1) {
              float v249_data = r5[(v246_i0 + v247_i1)];
              glb_m0[(v254_lead + ((v247_i1 + 4) * 32))] = v249_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v262_i0 = 0; v262_i0 < 1; ++v262_i0) {
            int32_t v268_lead = v12_lead + (v262_i0 * 32);
            #pragma unroll
            for (int32_t v263_i1 = 0; v263_i1 < 13; ++v263_i1) {
              float v271_data = glb_m0[(v268_lead + (v263_i1 * 32))];
              r6[(v262_i0 + v263_i1)] = v271_data;
            }
          }
          float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          if (v12_lead < 13) {
            #pragma unroll
            for (int32_t v278_i1 = 0; v278_i1 < 13; ++v278_i1) {
              float v286_data = __builtin_nontemporal_load(&glb_m4[(v12_lead + (v278_i1 * 13))]);
              r7[v278_i1] = v286_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v289_data = r7[0];
          float v290_data = r7[1];
          float v291_data = r7[2];
          float v292_data = r7[3];
          float v293_tp{};
          float v294_tp{};
          float v295_tp{};
          float v296_tp{};
          tensorforge::transpose4x4b32(v293_tp, v294_tp, v295_tp, v296_tp, v289_data, v290_data, v291_data, v292_data);
          tensorforge::VectorT<float, 4> v297_acc{};
          float v298_data = r6[0];
          float v299_data = r6[1];
          float v300_data = r6[2];
          float v301_data = r6[3];
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v298_data, v297_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v299_data, v302_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v300_data, v303_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v301_data, v304_acc, 3, 0, 0);
          float v306_data = r6[4];
          float v307_data = r6[5];
          float v308_data = r6[6];
          float v309_data = r6[7];
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v306_data, v305_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v307_data, v310_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v308_data, v311_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v309_data, v312_acc, 3, 1, 0);
          float v314_data = r6[8];
          float v315_data = r6[9];
          float v316_data = r6[10];
          float v317_data = r6[11];
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v314_data, v313_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v315_data, v318_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v295_tp, v316_data, v319_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v296_tp, v317_data, v320_acc, 3, 2, 0);
          float v322_data = r6[12];
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v322_data, v321_acc, 3, 3, 0);
          r8[0] = (v326_acc[0]);
          r8[1] = (v326_acc[1]);
          r8[2] = (v326_acc[2]);
          r8[3] = (v326_acc[3]);
          float v331_data = r7[4];
          float v332_data = r7[5];
          float v333_data = r7[6];
          float v334_data = r7[7];
          float v335_tp{};
          float v336_tp{};
          float v337_tp{};
          float v338_tp{};
          tensorforge::transpose4x4b32(v335_tp, v336_tp, v337_tp, v338_tp, v331_data, v332_data, v333_data, v334_data);
          tensorforge::VectorT<float, 4> v339_acc{};
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v298_data, v339_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v299_data, v344_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v300_data, v345_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v301_data, v346_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v306_data, v347_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v307_data, v352_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v308_data, v353_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v309_data, v354_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v314_data, v355_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v315_data, v360_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v316_data, v361_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v317_data, v362_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v322_data, v363_acc, 3, 3, 0);
          r8[4] = (v368_acc[0]);
          r8[5] = (v368_acc[1]);
          r8[6] = (v368_acc[2]);
          r8[7] = (v368_acc[3]);
          float v373_data = r7[8];
          float v374_data = r7[9];
          float v375_data = r7[10];
          float v376_data = r7[11];
          float v377_tp{};
          float v378_tp{};
          float v379_tp{};
          float v380_tp{};
          tensorforge::transpose4x4b32(v377_tp, v378_tp, v379_tp, v380_tp, v373_data, v374_data, v375_data, v376_data);
          tensorforge::VectorT<float, 4> v381_acc{};
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v298_data, v381_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v299_data, v386_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v300_data, v387_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v301_data, v388_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v306_data, v389_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v307_data, v394_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v308_data, v395_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v309_data, v396_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v314_data, v397_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v315_data, v402_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v316_data, v403_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v380_tp, v317_data, v404_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v322_data, v405_acc, 3, 3, 0);
          r8[8] = (v410_acc[0]);
          r8[9] = (v410_acc[1]);
          r8[10] = (v410_acc[2]);
          r8[11] = (v410_acc[3]);
          float v428_acc{};
          float v429_data = r7[12];
          float v430_bc = tensorforge::broadcast<32, 16, 0>(v429_data);
          tensorforge::fmacdpp16<0>(v428_acc, v430_bc, v298_data);
          tensorforge::fmacdpp16<1>(v428_acc, v430_bc, v299_data);
          tensorforge::fmacdpp16<2>(v428_acc, v430_bc, v300_data);
          tensorforge::fmacdpp16<3>(v428_acc, v430_bc, v301_data);
          tensorforge::fmacdpp16<4>(v428_acc, v430_bc, v306_data);
          tensorforge::fmacdpp16<5>(v428_acc, v430_bc, v307_data);
          tensorforge::fmacdpp16<6>(v428_acc, v430_bc, v308_data);
          tensorforge::fmacdpp16<7>(v428_acc, v430_bc, v309_data);
          tensorforge::fmacdpp16<8>(v428_acc, v430_bc, v314_data);
          tensorforge::fmacdpp16<9>(v428_acc, v430_bc, v315_data);
          tensorforge::fmacdpp16<10>(v428_acc, v430_bc, v316_data);
          tensorforge::fmacdpp16<11>(v428_acc, v430_bc, v317_data);
          tensorforge::fmacdpp16<12>(v428_acc, v430_bc, v322_data);
          r8[12] = v428_acc;
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v434_i0 = 0; v434_i0 < 1; ++v434_i0) {
            int32_t v442_lead = v12_lead + (v434_i0 * 32);
            #pragma unroll
            for (int32_t v435_i1 = 0; v435_i1 < 13; ++v435_i1) {
              float v437_data = r8[(v434_i0 + v435_i1)];
              glb_m3[(v442_lead + (v435_i1 * 32))] = v437_data;
            }
          }
        }
      }
    }
  }
}

