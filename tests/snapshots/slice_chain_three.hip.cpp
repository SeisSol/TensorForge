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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 6; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m0[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[6]{};
          // r1 = load{g>r}(glb_m1);
          float v23_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v23_lin;
          float v24_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v24_lin;
          float v25_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v25_lin;
          float v26_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v26_lin;
          float v27_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v27_lin;
          float v28_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v28_lin;
          float v29_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v29_lin;
          float v30_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v30_lin;
          float v31_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v31_lin;
          float v32_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v32_lin;
          float v33_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v33_lin;
          float v34_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v34_lin;
          float v35_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v35_lin;
          float v36_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v36_lin;
          float v37_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v37_lin;
          float v38_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v38_lin;
          float v39_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v39_lin;
          float v40_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v40_lin;
          float v41_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v41_lin;
          float v42_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v42_lin;
          float v43_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v43_lin;
          float v44_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v44_lin;
          float v45_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v45_lin;
          float v46_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v46_lin;
          float v47_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v47_lin;
          float v48_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v48_lin;
          float v49_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v49_lin;
          float v50_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v50_lin;
          float v51_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v51_lin;
          float v52_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v52_lin;
          float v53_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v53_lin;
          float v54_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v54_lin;
          float v55_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v55_lin;
          float v56_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v56_lin;
          float v57_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v57_lin;
          float v58_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v58_lin;
          float v59_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v59_lin;
          float v60_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v60_lin;
          float v61_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v61_lin;
          float v62_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v62_lin;
          float v63_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v63_lin;
          float v64_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v64_lin;
          float v65_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v65_lin;
          float v66_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v66_lin;
          float v67_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v67_lin;
          float v68_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v68_lin;
          float v69_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v69_lin;
          float v70_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v70_lin;
          float v71_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v71_lin;
          float v72_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v72_lin;
          float v73_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v73_lin;
          float v74_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v74_lin;
          float v75_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v75_lin;
          float v76_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v76_lin;
          float v77_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v77_lin;
          float v78_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v78_lin;
          float v79_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v79_lin;
          float v80_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v80_lin;
          float v81_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v81_lin;
          float v82_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v82_lin;
          float v83_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v83_lin;
          float v84_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v84_lin;
          float v85_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v85_lin;
          float v86_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v86_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v92_i1 = 0; v92_i1 < 12; ++v92_i1) {
              int32_t v98_a = v92_i1 * 12;
              int32_t v99_a = v3_lead + v98_a;
              float v107_data = __builtin_nontemporal_load(&glb_m3[(v3_lead + v98_a)]);
              int32_t v108_a = 0 + v92_i1;
              r3[v108_a] = v107_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          auto& ir2 = r2;
          float v110_data = r1[0];
          float v111_data = r1[1];
          float v112_data = r1[2];
          float v113_data = r1[3];
          float v114_tp{};
          float v115_tp{};
          float v116_tp{};
          float v117_tp{};
          tensorforge::transpose4x4b32(v114_tp, v115_tp, v116_tp, v117_tp, v110_data, v111_data, v112_data, v113_data);
          tensorforge::VectorT<float, 4> v118_acc{};
          float v119_data = r0[0];
          float v120_data = r0[1];
          float v121_data = r0[2];
          float v122_data = r0[3];
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v119_data, v118_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v120_data, v123_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v121_data, v124_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v122_data, v125_acc, 2, 0, 0);
          float v127_data = r0[4];
          float v128_data = r0[5];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v127_data, v126_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v115_tp, v128_data, v131_acc, 2, 1, 0);
          ir2[0] = (v132_acc[0]);
          ir2[1] = (v132_acc[1]);
          ir2[2] = (v132_acc[2]);
          ir2[3] = (v132_acc[3]);
          float v137_data = r1[4];
          float v138_data = r1[5];
          float v141_tp{};
          float v142_tp{};
          float v143_tp{};
          float v144_tp{};
          tensorforge::transpose4x4b32(v141_tp, v142_tp, v143_tp, v144_tp, v137_data, v138_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v145_acc{};
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v119_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v120_data, v150_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v121_data, v151_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v122_data, v152_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v127_data, v153_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v128_data, v158_acc, 2, 1, 0);
          ir2[4] = (v159_acc[0]);
          ir2[5] = (v159_acc[1]);
          // wait(r3 = load{g>r}(glb_m3););
          float r4[6]{};
          // r4 = +(r3 * r2) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          auto& ir4 = r4;
          float v163_data = r3[0];
          float v164_data = r3[1];
          float v165_data = r3[2];
          float v166_data = r3[3];
          float v167_data = r3[4];
          float v168_data = r3[5];
          float v169_data = r3[6];
          float v170_data = r3[7];
          float v171_data = r3[8];
          float v172_data = r3[9];
          float v173_data = r3[10];
          float v174_data = r3[11];
          float v175_acc{};
          float v176_acc{};
          float v177_acc{};
          float v178_acc{};
          float v179_acc{};
          float v180_acc{};
          float v181_lin = r2[0];
          tensorforge::fmacdpp16<0>(v175_acc, v181_lin, v163_data);
          tensorforge::fmacdpp16<1>(v175_acc, v181_lin, v164_data);
          tensorforge::fmacdpp16<2>(v175_acc, v181_lin, v165_data);
          tensorforge::fmacdpp16<3>(v175_acc, v181_lin, v166_data);
          tensorforge::fmacdpp16<4>(v175_acc, v181_lin, v167_data);
          tensorforge::fmacdpp16<5>(v175_acc, v181_lin, v168_data);
          tensorforge::fmacdpp16<6>(v175_acc, v181_lin, v169_data);
          tensorforge::fmacdpp16<7>(v175_acc, v181_lin, v170_data);
          tensorforge::fmacdpp16<8>(v175_acc, v181_lin, v171_data);
          tensorforge::fmacdpp16<9>(v175_acc, v181_lin, v172_data);
          tensorforge::fmacdpp16<10>(v175_acc, v181_lin, v173_data);
          tensorforge::fmacdpp16<11>(v175_acc, v181_lin, v174_data);
          tensorforge::fmacdpp16<12>(v176_acc, v181_lin, v163_data);
          tensorforge::fmacdpp16<13>(v176_acc, v181_lin, v164_data);
          tensorforge::fmacdpp16<14>(v176_acc, v181_lin, v165_data);
          tensorforge::fmacdpp16<15>(v176_acc, v181_lin, v166_data);
          float v182_lin = r2[1];
          tensorforge::fmacdpp16<0>(v176_acc, v182_lin, v167_data);
          tensorforge::fmacdpp16<1>(v176_acc, v182_lin, v168_data);
          tensorforge::fmacdpp16<2>(v176_acc, v182_lin, v169_data);
          tensorforge::fmacdpp16<3>(v176_acc, v182_lin, v170_data);
          tensorforge::fmacdpp16<4>(v176_acc, v182_lin, v171_data);
          tensorforge::fmacdpp16<5>(v176_acc, v182_lin, v172_data);
          tensorforge::fmacdpp16<6>(v176_acc, v182_lin, v173_data);
          tensorforge::fmacdpp16<7>(v176_acc, v182_lin, v174_data);
          tensorforge::fmacdpp16<8>(v177_acc, v182_lin, v163_data);
          tensorforge::fmacdpp16<9>(v177_acc, v182_lin, v164_data);
          tensorforge::fmacdpp16<10>(v177_acc, v182_lin, v165_data);
          tensorforge::fmacdpp16<11>(v177_acc, v182_lin, v166_data);
          tensorforge::fmacdpp16<12>(v177_acc, v182_lin, v167_data);
          tensorforge::fmacdpp16<13>(v177_acc, v182_lin, v168_data);
          tensorforge::fmacdpp16<14>(v177_acc, v182_lin, v169_data);
          tensorforge::fmacdpp16<15>(v177_acc, v182_lin, v170_data);
          float v183_lin = r2[2];
          tensorforge::fmacdpp16<0>(v177_acc, v183_lin, v171_data);
          tensorforge::fmacdpp16<1>(v177_acc, v183_lin, v172_data);
          tensorforge::fmacdpp16<2>(v177_acc, v183_lin, v173_data);
          tensorforge::fmacdpp16<3>(v177_acc, v183_lin, v174_data);
          tensorforge::fmacdpp16<4>(v178_acc, v183_lin, v163_data);
          tensorforge::fmacdpp16<5>(v178_acc, v183_lin, v164_data);
          tensorforge::fmacdpp16<6>(v178_acc, v183_lin, v165_data);
          tensorforge::fmacdpp16<7>(v178_acc, v183_lin, v166_data);
          tensorforge::fmacdpp16<8>(v178_acc, v183_lin, v167_data);
          tensorforge::fmacdpp16<9>(v178_acc, v183_lin, v168_data);
          tensorforge::fmacdpp16<10>(v178_acc, v183_lin, v169_data);
          tensorforge::fmacdpp16<11>(v178_acc, v183_lin, v170_data);
          tensorforge::fmacdpp16<12>(v178_acc, v183_lin, v171_data);
          tensorforge::fmacdpp16<13>(v178_acc, v183_lin, v172_data);
          tensorforge::fmacdpp16<14>(v178_acc, v183_lin, v173_data);
          tensorforge::fmacdpp16<15>(v178_acc, v183_lin, v174_data);
          float v184_lin = r2[3];
          tensorforge::fmacdpp16<0>(v179_acc, v184_lin, v163_data);
          tensorforge::fmacdpp16<1>(v179_acc, v184_lin, v164_data);
          tensorforge::fmacdpp16<2>(v179_acc, v184_lin, v165_data);
          tensorforge::fmacdpp16<3>(v179_acc, v184_lin, v166_data);
          tensorforge::fmacdpp16<4>(v179_acc, v184_lin, v167_data);
          tensorforge::fmacdpp16<5>(v179_acc, v184_lin, v168_data);
          tensorforge::fmacdpp16<6>(v179_acc, v184_lin, v169_data);
          tensorforge::fmacdpp16<7>(v179_acc, v184_lin, v170_data);
          tensorforge::fmacdpp16<8>(v179_acc, v184_lin, v171_data);
          tensorforge::fmacdpp16<9>(v179_acc, v184_lin, v172_data);
          tensorforge::fmacdpp16<10>(v179_acc, v184_lin, v173_data);
          tensorforge::fmacdpp16<11>(v179_acc, v184_lin, v174_data);
          tensorforge::fmacdpp16<12>(v180_acc, v184_lin, v163_data);
          tensorforge::fmacdpp16<13>(v180_acc, v184_lin, v164_data);
          tensorforge::fmacdpp16<14>(v180_acc, v184_lin, v165_data);
          tensorforge::fmacdpp16<15>(v180_acc, v184_lin, v166_data);
          float v185_lin = r2[4];
          tensorforge::fmacdpp16<0>(v180_acc, v185_lin, v167_data);
          tensorforge::fmacdpp16<1>(v180_acc, v185_lin, v168_data);
          tensorforge::fmacdpp16<2>(v180_acc, v185_lin, v169_data);
          tensorforge::fmacdpp16<3>(v180_acc, v185_lin, v170_data);
          tensorforge::fmacdpp16<4>(v180_acc, v185_lin, v171_data);
          tensorforge::fmacdpp16<5>(v180_acc, v185_lin, v172_data);
          tensorforge::fmacdpp16<6>(v180_acc, v185_lin, v173_data);
          tensorforge::fmacdpp16<7>(v180_acc, v185_lin, v174_data);
          ir4[0] = v175_acc;
          ir4[1] = v176_acc;
          ir4[2] = v177_acc;
          ir4[3] = v178_acc;
          ir4[4] = v179_acc;
          ir4[5] = v180_acc;
          // glb_m2 = store{r>g}(r4);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v190_i1 = 0; v190_i1 < 6; ++v190_i1) {
              int32_t v191_a = 0 + v190_i1;
              float v193_data = r4[v190_i1];
              int32_t v200_a = v3_lead + (v190_i1 * 12);
              glb_m2[v200_a] = v193_data;
            }
          }
          ;
        }
      }
    }
  }
}

