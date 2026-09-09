// === base name ===
kernel_82283a2aa0

// === header ===
void launcher_kernel_82283a2aa0(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_82283a2aa0(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_82283a2aa0, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_82283a2aa0), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_82283a2aa0, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_82283a2aa0(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×16(32×16) {0..32}×{0..16} strided
    // m1 32×12(32×12) {0..32}×{0..12} strided
    // m2 12×16(12×16) {0..12}×{0..16} strided
    // m3 32×12(32×12) {0..32}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 32×12(32×12) {0..32}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..16})[0, 1] = m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[-1, 1]
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m3 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m5 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 512 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 192 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 384 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 384 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v14_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
            int32_t v21_lead = v14_lead + (v15_i0 * 32);
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 12; ++v16_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v16_i1 * 32))]);
              r0[(v15_i0 + v16_i1)] = v24_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v14_lead < 12) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 16; ++v31_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m2[(v14_lead + (v31_i1 * 12))]);
              r1[v31_i1] = v39_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v45_i0 = 0; v45_i0 < 1; ++v45_i0) {
            int32_t v51_lead = v14_lead + (v45_i0 * 32);
            #pragma unroll
            for (int32_t v46_i1 = 0; v46_i1 < 12; ++v46_i1) {
              float v54_data = __builtin_nontemporal_load(&glb_m3[(v51_lead + (v46_i1 * 32))]);
              r3[(v45_i0 + v46_i1)] = v54_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
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
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v66_data, v65_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v67_data, v70_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v68_data, v71_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v69_data, v72_acc, 3, 0, 0);
          float v74_data = r0[4];
          float v75_data = r0[5];
          float v76_data = r0[6];
          float v77_data = r0[7];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v73_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v75_data, v78_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v76_data, v79_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v77_data, v80_acc, 3, 1, 0);
          float v82_data = r0[8];
          float v83_data = r0[9];
          float v84_data = r0[10];
          float v85_data = r0[11];
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v82_data, v81_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v83_data, v86_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v84_data, v87_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v85_data, v88_acc, 3, 2, 0);
          r2[0] = (v89_acc[0]);
          r2[1] = (v89_acc[1]);
          r2[2] = (v89_acc[2]);
          r2[3] = (v89_acc[3]);
          float v94_data = r1[4];
          float v95_data = r1[5];
          float v96_data = r1[6];
          float v97_data = r1[7];
          float v98_tp{};
          float v99_tp{};
          float v100_tp{};
          float v101_tp{};
          tensorforge::transpose4x4b32(v98_tp, v99_tp, v100_tp, v101_tp, v94_data, v95_data, v96_data, v97_data);
          tensorforge::VectorT<float, 4> v102_acc{};
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v66_data, v102_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v67_data, v107_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v68_data, v108_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v69_data, v109_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v74_data, v110_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v75_data, v115_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v76_data, v116_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v77_data, v117_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v98_tp, v82_data, v118_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v99_tp, v83_data, v123_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v100_tp, v84_data, v124_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v101_tp, v85_data, v125_acc, 3, 2, 0);
          r2[4] = (v126_acc[0]);
          r2[5] = (v126_acc[1]);
          r2[6] = (v126_acc[2]);
          r2[7] = (v126_acc[3]);
          float v131_data = r1[8];
          float v132_data = r1[9];
          float v133_data = r1[10];
          float v134_data = r1[11];
          float v135_tp{};
          float v136_tp{};
          float v137_tp{};
          float v138_tp{};
          tensorforge::transpose4x4b32(v135_tp, v136_tp, v137_tp, v138_tp, v131_data, v132_data, v133_data, v134_data);
          tensorforge::VectorT<float, 4> v139_acc{};
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v66_data, v139_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v67_data, v144_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v68_data, v145_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v69_data, v146_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v74_data, v147_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v75_data, v152_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v76_data, v153_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v77_data, v154_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v82_data, v155_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v83_data, v160_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v84_data, v161_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v85_data, v162_acc, 3, 2, 0);
          r2[8] = (v163_acc[0]);
          r2[9] = (v163_acc[1]);
          r2[10] = (v163_acc[2]);
          r2[11] = (v163_acc[3]);
          float v168_data = r1[12];
          float v169_data = r1[13];
          float v170_data = r1[14];
          float v171_data = r1[15];
          float v172_tp{};
          float v173_tp{};
          float v174_tp{};
          float v175_tp{};
          tensorforge::transpose4x4b32(v172_tp, v173_tp, v174_tp, v175_tp, v168_data, v169_data, v170_data, v171_data);
          tensorforge::VectorT<float, 4> v176_acc{};
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v66_data, v176_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v67_data, v181_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v68_data, v182_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v69_data, v183_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v74_data, v184_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v75_data, v189_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v76_data, v190_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v77_data, v191_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v82_data, v192_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v83_data, v197_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v84_data, v198_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v85_data, v199_acc, 3, 2, 0);
          r2[12] = (v200_acc[0]);
          r2[13] = (v200_acc[1]);
          r2[14] = (v200_acc[2]);
          r2[15] = (v200_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v208_i0 = 0; v208_i0 < 1; ++v208_i0) {
            int32_t v216_lead = v14_lead + (v208_i0 * 32);
            #pragma unroll
            for (int32_t v209_i1 = 0; v209_i1 < 16; ++v209_i1) {
              float v211_data = r2[(v208_i0 + v209_i1)];
              glb_m0[(v216_lead + (v209_i1 * 32))] = v211_data;
            }
          }
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          if (v14_lead < 12) {
            #pragma unroll
            for (int32_t v224_i1 = 0; v224_i1 < 8; ++v224_i1) {
              float v232_data = __builtin_nontemporal_load(&glb_m4[(v14_lead + (v224_i1 * 12))]);
              r4[v224_i1] = v232_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v238_i0 = 0; v238_i0 < 1; ++v238_i0) {
            int32_t v244_lead = v14_lead + (v238_i0 * 32);
            #pragma unroll
            for (int32_t v239_i1 = 0; v239_i1 < 12; ++v239_i1) {
              float v247_data = __builtin_nontemporal_load(&glb_m5[(v244_lead + (v239_i1 * 32))]);
              r6[(v238_i0 + v239_i1)] = v247_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v250_data = r4[0];
          float v251_data = r4[1];
          float v252_data = r4[2];
          float v253_data = r4[3];
          float v254_tp{};
          float v255_tp{};
          float v256_tp{};
          float v257_tp{};
          tensorforge::transpose4x4b32(v254_tp, v255_tp, v256_tp, v257_tp, v250_data, v251_data, v252_data, v253_data);
          tensorforge::VectorT<float, 4> v258_acc{};
          float v259_data = r3[0];
          float v260_data = r3[1];
          float v261_data = r3[2];
          float v262_data = r3[3];
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v259_data, v258_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v260_data, v263_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v261_data, v264_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v262_data, v265_acc, 3, 0, 0);
          float v267_data = r3[4];
          float v268_data = r3[5];
          float v269_data = r3[6];
          float v270_data = r3[7];
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v267_data, v266_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v268_data, v271_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v269_data, v272_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v270_data, v273_acc, 3, 1, 0);
          float v275_data = r3[8];
          float v276_data = r3[9];
          float v277_data = r3[10];
          float v278_data = r3[11];
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v254_tp, v275_data, v274_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v255_tp, v276_data, v279_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v277_data, v280_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v278_data, v281_acc, 3, 2, 0);
          r5[0] = (v282_acc[0]);
          r5[1] = (v282_acc[1]);
          r5[2] = (v282_acc[2]);
          r5[3] = (v282_acc[3]);
          float v287_data = r4[4];
          float v288_data = r4[5];
          float v289_data = r4[6];
          float v290_data = r4[7];
          float v291_tp{};
          float v292_tp{};
          float v293_tp{};
          float v294_tp{};
          tensorforge::transpose4x4b32(v291_tp, v292_tp, v293_tp, v294_tp, v287_data, v288_data, v289_data, v290_data);
          tensorforge::VectorT<float, 4> v295_acc{};
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v259_data, v295_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v260_data, v300_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v261_data, v301_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v262_data, v302_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v267_data, v303_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v268_data, v308_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v269_data, v309_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v270_data, v310_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v291_tp, v275_data, v311_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v292_tp, v276_data, v316_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v293_tp, v277_data, v317_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v294_tp, v278_data, v318_acc, 3, 2, 0);
          r5[4] = (v319_acc[0]);
          r5[5] = (v319_acc[1]);
          r5[6] = (v319_acc[2]);
          r5[7] = (v319_acc[3]);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v327_i0 = 0; v327_i0 < 1; ++v327_i0) {
            int32_t v335_lead = v14_lead + (v327_i0 * 32);
            #pragma unroll
            for (int32_t v328_i1 = 0; v328_i1 < 8; ++v328_i1) {
              float v330_data = r5[(v327_i0 + v328_i1)];
              int32_t v337_a = v335_lead + (v328_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v337_a], v330_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          if (v14_lead < 12) {
            #pragma unroll
            for (int32_t v343_i1 = 0; v343_i1 < 8; ++v343_i1) {
              float v351_data = __builtin_nontemporal_load(&glb_m6[(v14_lead + (v343_i1 * 12))]);
              r7[v343_i1] = v351_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v354_data = r7[0];
          float v355_data = r7[1];
          float v356_data = r7[2];
          float v357_data = r7[3];
          float v358_tp{};
          float v359_tp{};
          float v360_tp{};
          float v361_tp{};
          tensorforge::transpose4x4b32(v358_tp, v359_tp, v360_tp, v361_tp, v354_data, v355_data, v356_data, v357_data);
          tensorforge::VectorT<float, 4> v362_acc{};
          float v363_data = r6[0];
          float v364_data = r6[1];
          float v365_data = r6[2];
          float v366_data = r6[3];
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v363_data, v362_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v364_data, v367_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v365_data, v368_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v366_data, v369_acc, 3, 0, 0);
          float v371_data = r6[4];
          float v372_data = r6[5];
          float v373_data = r6[6];
          float v374_data = r6[7];
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v371_data, v370_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v372_data, v375_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v373_data, v376_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v374_data, v377_acc, 3, 1, 0);
          float v379_data = r6[8];
          float v380_data = r6[9];
          float v381_data = r6[10];
          float v382_data = r6[11];
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v358_tp, v379_data, v378_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v359_tp, v380_data, v383_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v360_tp, v381_data, v384_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v382_data, v385_acc, 3, 2, 0);
          r8[0] = (v386_acc[0]);
          r8[1] = (v386_acc[1]);
          r8[2] = (v386_acc[2]);
          r8[3] = (v386_acc[3]);
          float v391_data = r7[4];
          float v392_data = r7[5];
          float v393_data = r7[6];
          float v394_data = r7[7];
          float v395_tp{};
          float v396_tp{};
          float v397_tp{};
          float v398_tp{};
          tensorforge::transpose4x4b32(v395_tp, v396_tp, v397_tp, v398_tp, v391_data, v392_data, v393_data, v394_data);
          tensorforge::VectorT<float, 4> v399_acc{};
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v363_data, v399_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v364_data, v404_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v365_data, v405_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v366_data, v406_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v371_data, v407_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v372_data, v412_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v373_data, v413_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v374_data, v414_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v379_data, v415_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v380_data, v420_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v381_data, v421_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v382_data, v422_acc, 3, 2, 0);
          r8[4] = (v423_acc[0]);
          r8[5] = (v423_acc[1]);
          r8[6] = (v423_acc[2]);
          r8[7] = (v423_acc[3]);
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v431_i0 = 0; v431_i0 < 1; ++v431_i0) {
            int32_t v439_lead = v14_lead + (v431_i0 * 32);
            #pragma unroll
            for (int32_t v432_i1 = 0; v432_i1 < 8; ++v432_i1) {
              float v434_data = r8[(v431_i0 + v432_i1)];
              int32_t v442_a = v439_lead + ((v432_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v442_a], v434_data);
            }
          }
        }
      }
    }
  }
}

