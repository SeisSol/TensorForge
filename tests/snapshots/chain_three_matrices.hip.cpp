// === base name ===
kernel_c558c9ba77e17efc

// === header ===
void launcher_kernel_c558c9ba77e17efc(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_c558c9ba77e17efc(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c558c9ba77e17efc, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_c558c9ba77e17efc), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_c558c9ba77e17efc, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_c558c9ba77e17efc(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 56×9(56×9) {0..56}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 56×9(56×9) {0..56}×{0..9} strided
    // m3 56×56(56×56) {0..56}×{0..56} strided
    // t0 56×9(56×9) {0..56}×{0..9} pointer_based({0..56}×{0..9})[0, 1] = m0 56×9(56×9) {0..56}×{0..9} strided({0..56}×{0..9})[0, -1]×m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]
    // m2 56×9(56×9) {0..56}×{0..9} strided({0..56}×{0..9})[0, 1] = m3 56×56(56×56) {0..56}×{0..56} strided({0..56}×{0..56})[0, -1]×t0 56×9(56×9) {0..56}×{0..9} pointer_based({0..56}×{0..9})[-1, 1]
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
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 504 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 504 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 3136 + 0 + m3_extraOffset];
          float r0[18]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v11_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v12_i0 = 0; v12_i0 < 1; ++v12_i0) {
            int32_t v18_lead = v11_lead + (v12_i0 * 32);
            #pragma unroll
            for (int32_t v13_i1 = 0; v13_i1 < 9; ++v13_i1) {
              float v21_data = __builtin_nontemporal_load(&glb_m0[(v18_lead + (v13_i1 * 56))]);
              r0[(v12_i0 + (v13_i1 * 2))] = v21_data;
            }
          }
          if (v11_lead < 24) {
            int32_t v30_lead = v11_lead + 32_i32;
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m0[(v30_lead + (v25_i1 * 56))]);
              r0[(1 + (v25_i1 * 2))] = v33_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m1);
          if (v11_lead < 9) {
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 9; ++v41_i1) {
              float v49_data = __builtin_nontemporal_load(&glb_m1[(v11_lead + (v41_i1 * 9))]);
              r1[v41_i1] = v49_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[112]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v55_i0 = 0; v55_i0 < 1; ++v55_i0) {
            int32_t v61_lead = v11_lead + (v55_i0 * 32);
            #pragma unroll
            for (int32_t v56_i1 = 0; v56_i1 < 56; ++v56_i1) {
              float v64_data = __builtin_nontemporal_load(&glb_m3[(v61_lead + (v56_i1 * 56))]);
              r3[(v55_i0 + (v56_i1 * 2))] = v64_data;
            }
          }
          if (v11_lead < 24) {
            int32_t v73_lead = v11_lead + 32_i32;
            #pragma unroll
            for (int32_t v68_i1 = 0; v68_i1 < 56; ++v68_i1) {
              float v76_data = __builtin_nontemporal_load(&glb_m3[(v73_lead + (v68_i1 * 56))]);
              r3[(1 + (v68_i1 * 2))] = v76_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[18]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 9)] [(0, 9)]
          float v80_data = r1[0];
          float v81_data = r1[1];
          float v82_data = r1[2];
          float v83_data = r1[3];
          float v84_tp{};
          float v85_tp{};
          float v86_tp{};
          float v87_tp{};
          tensorforge::transpose4x4b32(v84_tp, v85_tp, v86_tp, v87_tp, v80_data, v81_data, v82_data, v83_data);
          tensorforge::VectorT<float, 4> v88_acc{};
          float v89_data = r0[0];
          float v90_data = r0[2];
          float v91_data = r0[4];
          float v92_data = r0[6];
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v89_data, v88_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v90_data, v93_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v91_data, v94_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v92_data, v95_acc, 3, 0, 0);
          float v97_data = r0[8];
          float v98_data = r0[10];
          float v99_data = r0[12];
          float v100_data = r0[14];
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v97_data, v96_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v98_data, v101_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v99_data, v102_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v100_data, v103_acc, 3, 1, 0);
          float v105_data = r0[16];
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v105_data, v104_acc, 3, 2, 0);
          r2[0] = (v109_acc[0]);
          r2[2] = (v109_acc[1]);
          r2[4] = (v109_acc[2]);
          r2[6] = (v109_acc[3]);
          tensorforge::VectorT<float, 4> v114_acc{};
          float v115_data = r0[1];
          float v116_data = r0[3];
          float v117_data = r0[5];
          float v118_data = r0[7];
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v115_data, v114_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v116_data, v119_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v117_data, v120_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v118_data, v121_acc, 3, 0, 0);
          float v123_data = r0[9];
          float v124_data = r0[11];
          float v125_data = r0[13];
          float v126_data = r0[15];
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v123_data, v122_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v124_data, v127_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v125_data, v128_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v126_data, v129_acc, 3, 1, 0);
          float v131_data = r0[17];
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v131_data, v130_acc, 3, 2, 0);
          r2[1] = (v135_acc[0]);
          r2[3] = (v135_acc[1]);
          r2[5] = (v135_acc[2]);
          r2[7] = (v135_acc[3]);
          float v140_data = r1[4];
          float v141_data = r1[5];
          float v142_data = r1[6];
          float v143_data = r1[7];
          float v144_tp{};
          float v145_tp{};
          float v146_tp{};
          float v147_tp{};
          tensorforge::transpose4x4b32(v144_tp, v145_tp, v146_tp, v147_tp, v140_data, v141_data, v142_data, v143_data);
          tensorforge::VectorT<float, 4> v148_acc{};
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v89_data, v148_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v90_data, v153_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v91_data, v154_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v92_data, v155_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v97_data, v156_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v98_data, v161_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v99_data, v162_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v100_data, v163_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v105_data, v164_acc, 3, 2, 0);
          r2[8] = (v169_acc[0]);
          r2[10] = (v169_acc[1]);
          r2[12] = (v169_acc[2]);
          r2[14] = (v169_acc[3]);
          tensorforge::VectorT<float, 4> v174_acc{};
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v115_data, v174_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v116_data, v179_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v117_data, v180_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v118_data, v181_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v123_data, v182_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v124_data, v187_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v125_data, v188_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v126_data, v189_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v131_data, v190_acc, 3, 2, 0);
          r2[9] = (v195_acc[0]);
          r2[11] = (v195_acc[1]);
          r2[13] = (v195_acc[2]);
          r2[15] = (v195_acc[3]);
          float v218_acc{};
          float v219_acc{};
          float v220_data = r1[8];
          float v221_bc = tensorforge::broadcast<32, 16, 0>(v220_data);
          tensorforge::fmacdpp16<0>(v218_acc, v221_bc, v89_data);
          tensorforge::fmacdpp16<0>(v219_acc, v221_bc, v115_data);
          tensorforge::fmacdpp16<1>(v218_acc, v221_bc, v90_data);
          tensorforge::fmacdpp16<1>(v219_acc, v221_bc, v116_data);
          tensorforge::fmacdpp16<2>(v218_acc, v221_bc, v91_data);
          tensorforge::fmacdpp16<2>(v219_acc, v221_bc, v117_data);
          tensorforge::fmacdpp16<3>(v218_acc, v221_bc, v92_data);
          tensorforge::fmacdpp16<3>(v219_acc, v221_bc, v118_data);
          tensorforge::fmacdpp16<4>(v218_acc, v221_bc, v97_data);
          tensorforge::fmacdpp16<4>(v219_acc, v221_bc, v123_data);
          tensorforge::fmacdpp16<5>(v218_acc, v221_bc, v98_data);
          tensorforge::fmacdpp16<5>(v219_acc, v221_bc, v124_data);
          tensorforge::fmacdpp16<6>(v218_acc, v221_bc, v99_data);
          tensorforge::fmacdpp16<6>(v219_acc, v221_bc, v125_data);
          tensorforge::fmacdpp16<7>(v218_acc, v221_bc, v100_data);
          tensorforge::fmacdpp16<7>(v219_acc, v221_bc, v126_data);
          tensorforge::fmacdpp16<8>(v218_acc, v221_bc, v105_data);
          tensorforge::fmacdpp16<8>(v219_acc, v221_bc, v131_data);
          r2[16] = v218_acc;
          r2[17] = v219_acc;
          // wait(r3 = load{g>r}(glb_m3););
          float r4[18]{};
          // r4 = +(r3 * r2) + None
          // [(0, 56), (0, 9)] [(0, 56)]
          float v223_data = r2[0];
          float v224_data = r2[2];
          float v225_data = r2[4];
          float v226_data = r2[6];
          float v227_tp{};
          float v228_tp{};
          float v229_tp{};
          float v230_tp{};
          tensorforge::transpose4x4b32(v227_tp, v228_tp, v229_tp, v230_tp, v223_data, v224_data, v225_data, v226_data);
          float v231_data = r2[1];
          float v232_data = r2[3];
          float v233_data = r2[5];
          float v234_data = r2[7];
          float v235_tp{};
          float v236_tp{};
          float v237_tp{};
          float v238_tp{};
          tensorforge::transpose4x4b32(v235_tp, v236_tp, v237_tp, v238_tp, v231_data, v232_data, v233_data, v234_data);
          tensorforge::VectorT<float, 4> v239_acc{};
          float v240_data = r3[0];
          float v241_data = r3[2];
          float v242_data = r3[4];
          float v243_data = r3[6];
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v240_data, v239_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v241_data, v244_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v242_data, v245_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v243_data, v246_acc, 3, 0, 0);
          float v248_data = r3[8];
          float v249_data = r3[10];
          float v250_data = r3[12];
          float v251_data = r3[14];
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v248_data, v247_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v249_data, v252_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v250_data, v253_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v251_data, v254_acc, 3, 1, 0);
          float v256_data = r3[16];
          float v257_data = r3[18];
          float v258_data = r3[20];
          float v259_data = r3[22];
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v256_data, v255_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v257_data, v260_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v258_data, v261_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v259_data, v262_acc, 3, 2, 0);
          float v264_data = r3[24];
          float v265_data = r3[26];
          float v266_data = r3[28];
          float v267_data = r3[30];
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v264_data, v263_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v265_data, v268_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v266_data, v269_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v267_data, v270_acc, 3, 3, 0);
          float v272_data = r3[32];
          float v273_data = r3[34];
          float v274_data = r3[36];
          float v275_data = r3[38];
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v272_data, v271_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v273_data, v276_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v274_data, v277_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v275_data, v278_acc, 3, 4, 0);
          float v280_data = r3[40];
          float v281_data = r3[42];
          float v282_data = r3[44];
          float v283_data = r3[46];
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v280_data, v279_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v281_data, v284_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v282_data, v285_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v283_data, v286_acc, 3, 5, 0);
          float v288_data = r3[48];
          float v289_data = r3[50];
          float v290_data = r3[52];
          float v291_data = r3[54];
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v288_data, v287_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v289_data, v292_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v290_data, v293_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v291_data, v294_acc, 3, 6, 0);
          float v296_data = r3[56];
          float v297_data = r3[58];
          float v298_data = r3[60];
          float v299_data = r3[62];
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v296_data, v295_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v297_data, v300_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v298_data, v301_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v299_data, v302_acc, 3, 7, 0);
          float v304_data = r3[64];
          float v305_data = r3[66];
          float v306_data = r3[68];
          float v307_data = r3[70];
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v304_data, v303_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v305_data, v308_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v306_data, v309_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v307_data, v310_acc, 3, 0, 0);
          float v312_data = r3[72];
          float v313_data = r3[74];
          float v314_data = r3[76];
          float v315_data = r3[78];
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v312_data, v311_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v313_data, v316_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v314_data, v317_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v315_data, v318_acc, 3, 1, 0);
          float v320_data = r3[80];
          float v321_data = r3[82];
          float v322_data = r3[84];
          float v323_data = r3[86];
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v320_data, v319_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v321_data, v324_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v322_data, v325_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v323_data, v326_acc, 3, 2, 0);
          float v328_data = r3[88];
          float v329_data = r3[90];
          float v330_data = r3[92];
          float v331_data = r3[94];
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v328_data, v327_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v329_data, v332_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v330_data, v333_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v331_data, v334_acc, 3, 3, 0);
          float v336_data = r3[96];
          float v337_data = r3[98];
          float v338_data = r3[100];
          float v339_data = r3[102];
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v336_data, v335_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v337_data, v340_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v338_data, v341_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v339_data, v342_acc, 3, 4, 0);
          float v344_data = r3[104];
          float v345_data = r3[106];
          float v346_data = r3[108];
          float v347_data = r3[110];
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v344_data, v343_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v345_data, v348_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v346_data, v349_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v347_data, v350_acc, 3, 5, 0);
          r4[0] = (v351_acc[0]);
          r4[2] = (v351_acc[1]);
          r4[4] = (v351_acc[2]);
          r4[6] = (v351_acc[3]);
          tensorforge::VectorT<float, 4> v356_acc{};
          float v357_data = r3[1];
          float v358_data = r3[3];
          float v359_data = r3[5];
          float v360_data = r3[7];
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v357_data, v356_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v358_data, v361_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v359_data, v362_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v360_data, v363_acc, 3, 0, 0);
          float v365_data = r3[9];
          float v366_data = r3[11];
          float v367_data = r3[13];
          float v368_data = r3[15];
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v365_data, v364_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v366_data, v369_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v367_data, v370_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v368_data, v371_acc, 3, 1, 0);
          float v373_data = r3[17];
          float v374_data = r3[19];
          float v375_data = r3[21];
          float v376_data = r3[23];
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v373_data, v372_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v374_data, v377_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v375_data, v378_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v376_data, v379_acc, 3, 2, 0);
          float v381_data = r3[25];
          float v382_data = r3[27];
          float v383_data = r3[29];
          float v384_data = r3[31];
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v381_data, v380_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v382_data, v385_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v383_data, v386_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v384_data, v387_acc, 3, 3, 0);
          float v389_data = r3[33];
          float v390_data = r3[35];
          float v391_data = r3[37];
          float v392_data = r3[39];
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v389_data, v388_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v390_data, v393_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v391_data, v394_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v392_data, v395_acc, 3, 4, 0);
          float v397_data = r3[41];
          float v398_data = r3[43];
          float v399_data = r3[45];
          float v400_data = r3[47];
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v397_data, v396_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v398_data, v401_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v399_data, v402_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v400_data, v403_acc, 3, 5, 0);
          float v405_data = r3[49];
          float v406_data = r3[51];
          float v407_data = r3[53];
          float v408_data = r3[55];
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v405_data, v404_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v406_data, v409_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v407_data, v410_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v408_data, v411_acc, 3, 6, 0);
          float v413_data = r3[57];
          float v414_data = r3[59];
          float v415_data = r3[61];
          float v416_data = r3[63];
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v227_tp, v413_data, v412_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v414_data, v417_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v229_tp, v415_data, v418_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v416_data, v419_acc, 3, 7, 0);
          float v421_data = r3[65];
          float v422_data = r3[67];
          float v423_data = r3[69];
          float v424_data = r3[71];
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v421_data, v420_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v422_data, v425_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v423_data, v426_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v424_data, v427_acc, 3, 0, 0);
          float v429_data = r3[73];
          float v430_data = r3[75];
          float v431_data = r3[77];
          float v432_data = r3[79];
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v429_data, v428_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v430_data, v433_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v431_data, v434_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v432_data, v435_acc, 3, 1, 0);
          float v437_data = r3[81];
          float v438_data = r3[83];
          float v439_data = r3[85];
          float v440_data = r3[87];
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v437_data, v436_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v438_data, v441_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v439_data, v442_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v440_data, v443_acc, 3, 2, 0);
          float v445_data = r3[89];
          float v446_data = r3[91];
          float v447_data = r3[93];
          float v448_data = r3[95];
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v445_data, v444_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v446_data, v449_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v447_data, v450_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v448_data, v451_acc, 3, 3, 0);
          float v453_data = r3[97];
          float v454_data = r3[99];
          float v455_data = r3[101];
          float v456_data = r3[103];
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v453_data, v452_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v454_data, v457_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v455_data, v458_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v456_data, v459_acc, 3, 4, 0);
          float v461_data = r3[105];
          float v462_data = r3[107];
          float v463_data = r3[109];
          float v464_data = r3[111];
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v235_tp, v461_data, v460_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v462_data, v465_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v463_data, v466_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v464_data, v467_acc, 3, 5, 0);
          r4[1] = (v468_acc[0]);
          r4[3] = (v468_acc[1]);
          r4[5] = (v468_acc[2]);
          r4[7] = (v468_acc[3]);
          float v473_data = r2[8];
          float v474_data = r2[10];
          float v475_data = r2[12];
          float v476_data = r2[14];
          float v477_tp{};
          float v478_tp{};
          float v479_tp{};
          float v480_tp{};
          tensorforge::transpose4x4b32(v477_tp, v478_tp, v479_tp, v480_tp, v473_data, v474_data, v475_data, v476_data);
          float v481_data = r2[9];
          float v482_data = r2[11];
          float v483_data = r2[13];
          float v484_data = r2[15];
          float v485_tp{};
          float v486_tp{};
          float v487_tp{};
          float v488_tp{};
          tensorforge::transpose4x4b32(v485_tp, v486_tp, v487_tp, v488_tp, v481_data, v482_data, v483_data, v484_data);
          tensorforge::VectorT<float, 4> v489_acc{};
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v240_data, v489_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v241_data, v494_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v242_data, v495_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v243_data, v496_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v248_data, v497_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v249_data, v502_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v250_data, v503_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v251_data, v504_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v256_data, v505_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v257_data, v510_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v258_data, v511_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v259_data, v512_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v264_data, v513_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v265_data, v518_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v266_data, v519_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v267_data, v520_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v272_data, v521_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v273_data, v526_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v274_data, v527_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v529_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v275_data, v528_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v534_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v280_data, v529_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v281_data, v534_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v282_data, v535_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v283_data, v536_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v542_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v288_data, v537_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v289_data, v542_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v290_data, v543_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v545_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v291_data, v544_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v296_data, v545_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v297_data, v550_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v298_data, v551_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v299_data, v552_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v304_data, v553_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v305_data, v558_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v306_data, v559_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v561_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v307_data, v560_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v566_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v312_data, v561_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v313_data, v566_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v314_data, v567_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v569_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v315_data, v568_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v320_data, v569_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v321_data, v574_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v322_data, v575_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v577_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v323_data, v576_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v328_data, v577_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v329_data, v582_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v330_data, v583_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v585_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v331_data, v584_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v590_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v336_data, v585_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v591_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v337_data, v590_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v592_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v338_data, v591_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v593_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v339_data, v592_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v344_data, v593_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v599_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v345_data, v598_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v346_data, v599_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v601_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v347_data, v600_acc, 3, 5, 0);
          r4[8] = (v601_acc[0]);
          r4[10] = (v601_acc[1]);
          r4[12] = (v601_acc[2]);
          r4[14] = (v601_acc[3]);
          tensorforge::VectorT<float, 4> v606_acc{};
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v357_data, v606_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v358_data, v611_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v359_data, v612_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v360_data, v613_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v365_data, v614_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v366_data, v619_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v367_data, v620_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v368_data, v621_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v373_data, v622_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v374_data, v627_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v375_data, v628_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v376_data, v629_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v635_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v381_data, v630_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v636_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v382_data, v635_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v637_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v383_data, v636_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v638_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v384_data, v637_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v643_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v389_data, v638_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v644_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v390_data, v643_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v645_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v391_data, v644_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v646_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v392_data, v645_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v651_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v397_data, v646_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v652_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v398_data, v651_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v399_data, v652_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v400_data, v653_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v659_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v405_data, v654_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v660_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v406_data, v659_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v407_data, v660_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v662_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v408_data, v661_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v477_tp, v413_data, v662_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v668_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v414_data, v667_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v669_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v415_data, v668_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v670_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v416_data, v669_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v421_data, v670_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v422_data, v675_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v677_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v423_data, v676_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v678_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v424_data, v677_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v683_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v429_data, v678_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v684_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v430_data, v683_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v685_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v431_data, v684_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v686_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v432_data, v685_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v691_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v437_data, v686_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v692_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v438_data, v691_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v693_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v439_data, v692_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v694_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v440_data, v693_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v699_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v445_data, v694_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v700_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v446_data, v699_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v447_data, v700_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v448_data, v701_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v453_data, v702_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v454_data, v707_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v709_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v455_data, v708_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v710_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v456_data, v709_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v715_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v485_tp, v461_data, v710_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v716_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v486_tp, v462_data, v715_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v717_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v487_tp, v463_data, v716_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v718_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v488_tp, v464_data, v717_acc, 3, 5, 0);
          r4[9] = (v718_acc[0]);
          r4[11] = (v718_acc[1]);
          r4[13] = (v718_acc[2]);
          r4[15] = (v718_acc[3]);
          float v835_acc{};
          float v836_acc{};
          float v837_data = r2[16];
          float v838_data = r2[17];
          float v839_bc = tensorforge::broadcast<32, 16, 0>(v837_data);
          tensorforge::fmacdpp16<0>(v835_acc, v839_bc, v240_data);
          tensorforge::fmacdpp16<0>(v836_acc, v839_bc, v357_data);
          tensorforge::fmacdpp16<1>(v835_acc, v839_bc, v241_data);
          tensorforge::fmacdpp16<1>(v836_acc, v839_bc, v358_data);
          tensorforge::fmacdpp16<2>(v835_acc, v839_bc, v242_data);
          tensorforge::fmacdpp16<2>(v836_acc, v839_bc, v359_data);
          tensorforge::fmacdpp16<3>(v835_acc, v839_bc, v243_data);
          tensorforge::fmacdpp16<3>(v836_acc, v839_bc, v360_data);
          tensorforge::fmacdpp16<4>(v835_acc, v839_bc, v248_data);
          tensorforge::fmacdpp16<4>(v836_acc, v839_bc, v365_data);
          tensorforge::fmacdpp16<5>(v835_acc, v839_bc, v249_data);
          tensorforge::fmacdpp16<5>(v836_acc, v839_bc, v366_data);
          tensorforge::fmacdpp16<6>(v835_acc, v839_bc, v250_data);
          tensorforge::fmacdpp16<6>(v836_acc, v839_bc, v367_data);
          tensorforge::fmacdpp16<7>(v835_acc, v839_bc, v251_data);
          tensorforge::fmacdpp16<7>(v836_acc, v839_bc, v368_data);
          tensorforge::fmacdpp16<8>(v835_acc, v839_bc, v256_data);
          tensorforge::fmacdpp16<8>(v836_acc, v839_bc, v373_data);
          tensorforge::fmacdpp16<9>(v835_acc, v839_bc, v257_data);
          tensorforge::fmacdpp16<9>(v836_acc, v839_bc, v374_data);
          tensorforge::fmacdpp16<10>(v835_acc, v839_bc, v258_data);
          tensorforge::fmacdpp16<10>(v836_acc, v839_bc, v375_data);
          tensorforge::fmacdpp16<11>(v835_acc, v839_bc, v259_data);
          tensorforge::fmacdpp16<11>(v836_acc, v839_bc, v376_data);
          tensorforge::fmacdpp16<12>(v835_acc, v839_bc, v264_data);
          tensorforge::fmacdpp16<12>(v836_acc, v839_bc, v381_data);
          tensorforge::fmacdpp16<13>(v835_acc, v839_bc, v265_data);
          tensorforge::fmacdpp16<13>(v836_acc, v839_bc, v382_data);
          tensorforge::fmacdpp16<14>(v835_acc, v839_bc, v266_data);
          tensorforge::fmacdpp16<14>(v836_acc, v839_bc, v383_data);
          tensorforge::fmacdpp16<15>(v835_acc, v839_bc, v267_data);
          tensorforge::fmacdpp16<15>(v836_acc, v839_bc, v384_data);
          float v840_bc = tensorforge::broadcast<32, 16, 1>(v837_data);
          tensorforge::fmacdpp16<0>(v835_acc, v840_bc, v272_data);
          tensorforge::fmacdpp16<0>(v836_acc, v840_bc, v389_data);
          tensorforge::fmacdpp16<1>(v835_acc, v840_bc, v273_data);
          tensorforge::fmacdpp16<1>(v836_acc, v840_bc, v390_data);
          tensorforge::fmacdpp16<2>(v835_acc, v840_bc, v274_data);
          tensorforge::fmacdpp16<2>(v836_acc, v840_bc, v391_data);
          tensorforge::fmacdpp16<3>(v835_acc, v840_bc, v275_data);
          tensorforge::fmacdpp16<3>(v836_acc, v840_bc, v392_data);
          tensorforge::fmacdpp16<4>(v835_acc, v840_bc, v280_data);
          tensorforge::fmacdpp16<4>(v836_acc, v840_bc, v397_data);
          tensorforge::fmacdpp16<5>(v835_acc, v840_bc, v281_data);
          tensorforge::fmacdpp16<5>(v836_acc, v840_bc, v398_data);
          tensorforge::fmacdpp16<6>(v835_acc, v840_bc, v282_data);
          tensorforge::fmacdpp16<6>(v836_acc, v840_bc, v399_data);
          tensorforge::fmacdpp16<7>(v835_acc, v840_bc, v283_data);
          tensorforge::fmacdpp16<7>(v836_acc, v840_bc, v400_data);
          tensorforge::fmacdpp16<8>(v835_acc, v840_bc, v288_data);
          tensorforge::fmacdpp16<8>(v836_acc, v840_bc, v405_data);
          tensorforge::fmacdpp16<9>(v835_acc, v840_bc, v289_data);
          tensorforge::fmacdpp16<9>(v836_acc, v840_bc, v406_data);
          tensorforge::fmacdpp16<10>(v835_acc, v840_bc, v290_data);
          tensorforge::fmacdpp16<10>(v836_acc, v840_bc, v407_data);
          tensorforge::fmacdpp16<11>(v835_acc, v840_bc, v291_data);
          tensorforge::fmacdpp16<11>(v836_acc, v840_bc, v408_data);
          tensorforge::fmacdpp16<12>(v835_acc, v840_bc, v296_data);
          tensorforge::fmacdpp16<12>(v836_acc, v840_bc, v413_data);
          tensorforge::fmacdpp16<13>(v835_acc, v840_bc, v297_data);
          tensorforge::fmacdpp16<13>(v836_acc, v840_bc, v414_data);
          tensorforge::fmacdpp16<14>(v835_acc, v840_bc, v298_data);
          tensorforge::fmacdpp16<14>(v836_acc, v840_bc, v415_data);
          tensorforge::fmacdpp16<15>(v835_acc, v840_bc, v299_data);
          tensorforge::fmacdpp16<15>(v836_acc, v840_bc, v416_data);
          float v841_bc = tensorforge::broadcast<32, 16, 0>(v838_data);
          tensorforge::fmacdpp16<0>(v835_acc, v841_bc, v304_data);
          tensorforge::fmacdpp16<0>(v836_acc, v841_bc, v421_data);
          tensorforge::fmacdpp16<1>(v835_acc, v841_bc, v305_data);
          tensorforge::fmacdpp16<1>(v836_acc, v841_bc, v422_data);
          tensorforge::fmacdpp16<2>(v835_acc, v841_bc, v306_data);
          tensorforge::fmacdpp16<2>(v836_acc, v841_bc, v423_data);
          tensorforge::fmacdpp16<3>(v835_acc, v841_bc, v307_data);
          tensorforge::fmacdpp16<3>(v836_acc, v841_bc, v424_data);
          tensorforge::fmacdpp16<4>(v835_acc, v841_bc, v312_data);
          tensorforge::fmacdpp16<4>(v836_acc, v841_bc, v429_data);
          tensorforge::fmacdpp16<5>(v835_acc, v841_bc, v313_data);
          tensorforge::fmacdpp16<5>(v836_acc, v841_bc, v430_data);
          tensorforge::fmacdpp16<6>(v835_acc, v841_bc, v314_data);
          tensorforge::fmacdpp16<6>(v836_acc, v841_bc, v431_data);
          tensorforge::fmacdpp16<7>(v835_acc, v841_bc, v315_data);
          tensorforge::fmacdpp16<7>(v836_acc, v841_bc, v432_data);
          tensorforge::fmacdpp16<8>(v835_acc, v841_bc, v320_data);
          tensorforge::fmacdpp16<8>(v836_acc, v841_bc, v437_data);
          tensorforge::fmacdpp16<9>(v835_acc, v841_bc, v321_data);
          tensorforge::fmacdpp16<9>(v836_acc, v841_bc, v438_data);
          tensorforge::fmacdpp16<10>(v835_acc, v841_bc, v322_data);
          tensorforge::fmacdpp16<10>(v836_acc, v841_bc, v439_data);
          tensorforge::fmacdpp16<11>(v835_acc, v841_bc, v323_data);
          tensorforge::fmacdpp16<11>(v836_acc, v841_bc, v440_data);
          tensorforge::fmacdpp16<12>(v835_acc, v841_bc, v328_data);
          tensorforge::fmacdpp16<12>(v836_acc, v841_bc, v445_data);
          tensorforge::fmacdpp16<13>(v835_acc, v841_bc, v329_data);
          tensorforge::fmacdpp16<13>(v836_acc, v841_bc, v446_data);
          tensorforge::fmacdpp16<14>(v835_acc, v841_bc, v330_data);
          tensorforge::fmacdpp16<14>(v836_acc, v841_bc, v447_data);
          tensorforge::fmacdpp16<15>(v835_acc, v841_bc, v331_data);
          tensorforge::fmacdpp16<15>(v836_acc, v841_bc, v448_data);
          float v842_bc = tensorforge::broadcast<32, 16, 1>(v838_data);
          tensorforge::fmacdpp16<0>(v835_acc, v842_bc, v336_data);
          tensorforge::fmacdpp16<0>(v836_acc, v842_bc, v453_data);
          tensorforge::fmacdpp16<1>(v835_acc, v842_bc, v337_data);
          tensorforge::fmacdpp16<1>(v836_acc, v842_bc, v454_data);
          tensorforge::fmacdpp16<2>(v835_acc, v842_bc, v338_data);
          tensorforge::fmacdpp16<2>(v836_acc, v842_bc, v455_data);
          tensorforge::fmacdpp16<3>(v835_acc, v842_bc, v339_data);
          tensorforge::fmacdpp16<3>(v836_acc, v842_bc, v456_data);
          tensorforge::fmacdpp16<4>(v835_acc, v842_bc, v344_data);
          tensorforge::fmacdpp16<4>(v836_acc, v842_bc, v461_data);
          tensorforge::fmacdpp16<5>(v835_acc, v842_bc, v345_data);
          tensorforge::fmacdpp16<5>(v836_acc, v842_bc, v462_data);
          tensorforge::fmacdpp16<6>(v835_acc, v842_bc, v346_data);
          tensorforge::fmacdpp16<6>(v836_acc, v842_bc, v463_data);
          tensorforge::fmacdpp16<7>(v835_acc, v842_bc, v347_data);
          tensorforge::fmacdpp16<7>(v836_acc, v842_bc, v464_data);
          r4[16] = v835_acc;
          r4[17] = v836_acc;
          // glb_m2 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v846_i0 = 0; v846_i0 < 1; ++v846_i0) {
            int32_t v855_lead = v11_lead + (v846_i0 * 32);
            #pragma unroll
            for (int32_t v847_i1 = 0; v847_i1 < 9; ++v847_i1) {
              float v850_data = r4[(v846_i0 + (v847_i1 * 2))];
              glb_m2[(v855_lead + (v847_i1 * 56))] = v850_data;
            }
          }
          if (v11_lead < 24) {
            int32_t v867_lead = v11_lead + 32_i32;
            #pragma unroll
            for (int32_t v859_i1 = 0; v859_i1 < 9; ++v859_i1) {
              float v862_data = r4[(1 + (v859_i1 * 2))];
              glb_m2[(v867_lead + (v859_i1 * 56))] = v862_data;
            }
          }
        }
      }
    }
  }
}

