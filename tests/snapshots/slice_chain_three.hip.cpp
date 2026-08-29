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
          float v27_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v27_lin;
          float v28_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v28_lin;
          float v29_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v29_lin;
          float v30_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v30_lin;
          float v31_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v31_lin;
          float v32_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v32_lin;
          float v33_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v33_lin;
          float v34_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v34_lin;
          float v35_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v35_lin;
          float v36_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v36_lin;
          float v37_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v37_lin;
          float v38_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v38_lin;
          float v39_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v39_lin;
          float v40_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v40_lin;
          float v41_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v41_lin;
          float v42_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v42_lin;
          float v43_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v43_lin;
          float v44_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v44_lin;
          float v45_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v45_lin;
          float v46_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v46_lin;
          float v47_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v47_lin;
          float v48_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v48_lin;
          float v49_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v49_lin;
          float v50_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v50_lin;
          float v51_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v51_lin;
          float v52_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v52_lin;
          float v53_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v53_lin;
          float v54_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v54_lin;
          float v55_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v55_lin;
          float v56_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v56_lin;
          float v57_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v57_lin;
          float v58_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v58_lin;
          float v59_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v59_lin;
          float v60_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v60_lin;
          float v61_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v61_lin;
          float v62_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v62_lin;
          float v63_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v63_lin;
          float v64_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v64_lin;
          float v65_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v65_lin;
          float v66_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v66_lin;
          float v67_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v67_lin;
          float v68_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v68_lin;
          float v69_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v69_lin;
          float v70_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v70_lin;
          float v71_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v71_lin;
          float v72_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v72_lin;
          float v73_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v73_lin;
          float v74_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v74_lin;
          float v75_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v75_lin;
          float v76_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v76_lin;
          float v77_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v77_lin;
          float v78_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v78_lin;
          float v79_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v79_lin;
          float v80_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v80_lin;
          float v81_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v81_lin;
          float v82_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v82_lin;
          float v83_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v83_lin;
          float v84_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v84_lin;
          float v85_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v85_lin;
          float v86_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v86_lin;
          float v87_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v87_lin;
          float v88_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v88_lin;
          float v89_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v89_lin;
          float v90_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v90_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v14_lead < 12) {
            #pragma unroll
            for (int32_t v96_i1 = 0; v96_i1 < 12; ++v96_i1) {
              float v104_data = __builtin_nontemporal_load(&glb_m3[(v14_lead + (v96_i1 * 12))]);
              r3[v96_i1] = v104_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          float v107_data = r1[0];
          float v108_data = r1[1];
          float v109_data = r1[2];
          float v110_data = r1[3];
          float v111_tp{};
          float v112_tp{};
          float v113_tp{};
          float v114_tp{};
          tensorforge::transpose4x4b32(v111_tp, v112_tp, v113_tp, v114_tp, v107_data, v108_data, v109_data, v110_data);
          tensorforge::VectorT<float, 4> v115_acc{};
          float v116_data = r0[0];
          float v117_data = r0[1];
          float v118_data = r0[2];
          float v119_data = r0[3];
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v116_data, v115_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v117_data, v120_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v118_data, v121_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v119_data, v122_acc, 2, 0, 0);
          float v124_data = r0[4];
          float v125_data = r0[5];
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v124_data, v123_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v125_data, v128_acc, 2, 1, 0);
          r2[0] = (v129_acc[0]);
          r2[1] = (v129_acc[1]);
          r2[2] = (v129_acc[2]);
          r2[3] = (v129_acc[3]);
          float v134_data = r1[4];
          float v135_data = r1[5];
          float v138_tp{};
          float v139_tp{};
          float v140_tp{};
          float v141_tp{};
          tensorforge::transpose4x4b32(v138_tp, v139_tp, v140_tp, v141_tp, v134_data, v135_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v142_acc{};
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v116_data, v142_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v117_data, v147_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v118_data, v148_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v119_data, v149_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v124_data, v150_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v125_data, v155_acc, 2, 1, 0);
          r2[4] = (v156_acc[0]);
          r2[5] = (v156_acc[1]);
          // wait(r3 = load{g>r}(glb_m3););
          float r4[6]{};
          // r4 = +(r3 * r2) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v160_data = r3[0];
          float v161_data = r3[1];
          float v162_data = r3[2];
          float v163_data = r3[3];
          float v164_data = r3[4];
          float v165_data = r3[5];
          float v166_data = r3[6];
          float v167_data = r3[7];
          float v168_data = r3[8];
          float v169_data = r3[9];
          float v170_data = r3[10];
          float v171_data = r3[11];
          float v172_acc{};
          float v173_acc{};
          float v174_acc{};
          float v175_acc{};
          float v176_acc{};
          float v177_acc{};
          float v178_lin = r2[0];
          tensorforge::fmacdpp16<0>(v172_acc, v178_lin, v160_data);
          tensorforge::fmacdpp16<1>(v172_acc, v178_lin, v161_data);
          tensorforge::fmacdpp16<2>(v172_acc, v178_lin, v162_data);
          tensorforge::fmacdpp16<3>(v172_acc, v178_lin, v163_data);
          tensorforge::fmacdpp16<4>(v172_acc, v178_lin, v164_data);
          tensorforge::fmacdpp16<5>(v172_acc, v178_lin, v165_data);
          tensorforge::fmacdpp16<6>(v172_acc, v178_lin, v166_data);
          tensorforge::fmacdpp16<7>(v172_acc, v178_lin, v167_data);
          tensorforge::fmacdpp16<8>(v172_acc, v178_lin, v168_data);
          tensorforge::fmacdpp16<9>(v172_acc, v178_lin, v169_data);
          tensorforge::fmacdpp16<10>(v172_acc, v178_lin, v170_data);
          tensorforge::fmacdpp16<11>(v172_acc, v178_lin, v171_data);
          tensorforge::fmacdpp16<12>(v173_acc, v178_lin, v160_data);
          tensorforge::fmacdpp16<13>(v173_acc, v178_lin, v161_data);
          tensorforge::fmacdpp16<14>(v173_acc, v178_lin, v162_data);
          tensorforge::fmacdpp16<15>(v173_acc, v178_lin, v163_data);
          float v179_lin = r2[1];
          tensorforge::fmacdpp16<0>(v173_acc, v179_lin, v164_data);
          tensorforge::fmacdpp16<1>(v173_acc, v179_lin, v165_data);
          tensorforge::fmacdpp16<2>(v173_acc, v179_lin, v166_data);
          tensorforge::fmacdpp16<3>(v173_acc, v179_lin, v167_data);
          tensorforge::fmacdpp16<4>(v173_acc, v179_lin, v168_data);
          tensorforge::fmacdpp16<5>(v173_acc, v179_lin, v169_data);
          tensorforge::fmacdpp16<6>(v173_acc, v179_lin, v170_data);
          tensorforge::fmacdpp16<7>(v173_acc, v179_lin, v171_data);
          tensorforge::fmacdpp16<8>(v174_acc, v179_lin, v160_data);
          tensorforge::fmacdpp16<9>(v174_acc, v179_lin, v161_data);
          tensorforge::fmacdpp16<10>(v174_acc, v179_lin, v162_data);
          tensorforge::fmacdpp16<11>(v174_acc, v179_lin, v163_data);
          tensorforge::fmacdpp16<12>(v174_acc, v179_lin, v164_data);
          tensorforge::fmacdpp16<13>(v174_acc, v179_lin, v165_data);
          tensorforge::fmacdpp16<14>(v174_acc, v179_lin, v166_data);
          tensorforge::fmacdpp16<15>(v174_acc, v179_lin, v167_data);
          float v180_lin = r2[2];
          tensorforge::fmacdpp16<0>(v174_acc, v180_lin, v168_data);
          tensorforge::fmacdpp16<1>(v174_acc, v180_lin, v169_data);
          tensorforge::fmacdpp16<2>(v174_acc, v180_lin, v170_data);
          tensorforge::fmacdpp16<3>(v174_acc, v180_lin, v171_data);
          tensorforge::fmacdpp16<4>(v175_acc, v180_lin, v160_data);
          tensorforge::fmacdpp16<5>(v175_acc, v180_lin, v161_data);
          tensorforge::fmacdpp16<6>(v175_acc, v180_lin, v162_data);
          tensorforge::fmacdpp16<7>(v175_acc, v180_lin, v163_data);
          tensorforge::fmacdpp16<8>(v175_acc, v180_lin, v164_data);
          tensorforge::fmacdpp16<9>(v175_acc, v180_lin, v165_data);
          tensorforge::fmacdpp16<10>(v175_acc, v180_lin, v166_data);
          tensorforge::fmacdpp16<11>(v175_acc, v180_lin, v167_data);
          tensorforge::fmacdpp16<12>(v175_acc, v180_lin, v168_data);
          tensorforge::fmacdpp16<13>(v175_acc, v180_lin, v169_data);
          tensorforge::fmacdpp16<14>(v175_acc, v180_lin, v170_data);
          tensorforge::fmacdpp16<15>(v175_acc, v180_lin, v171_data);
          float v181_lin = r2[3];
          tensorforge::fmacdpp16<0>(v176_acc, v181_lin, v160_data);
          tensorforge::fmacdpp16<1>(v176_acc, v181_lin, v161_data);
          tensorforge::fmacdpp16<2>(v176_acc, v181_lin, v162_data);
          tensorforge::fmacdpp16<3>(v176_acc, v181_lin, v163_data);
          tensorforge::fmacdpp16<4>(v176_acc, v181_lin, v164_data);
          tensorforge::fmacdpp16<5>(v176_acc, v181_lin, v165_data);
          tensorforge::fmacdpp16<6>(v176_acc, v181_lin, v166_data);
          tensorforge::fmacdpp16<7>(v176_acc, v181_lin, v167_data);
          tensorforge::fmacdpp16<8>(v176_acc, v181_lin, v168_data);
          tensorforge::fmacdpp16<9>(v176_acc, v181_lin, v169_data);
          tensorforge::fmacdpp16<10>(v176_acc, v181_lin, v170_data);
          tensorforge::fmacdpp16<11>(v176_acc, v181_lin, v171_data);
          tensorforge::fmacdpp16<12>(v177_acc, v181_lin, v160_data);
          tensorforge::fmacdpp16<13>(v177_acc, v181_lin, v161_data);
          tensorforge::fmacdpp16<14>(v177_acc, v181_lin, v162_data);
          tensorforge::fmacdpp16<15>(v177_acc, v181_lin, v163_data);
          float v182_lin = r2[4];
          tensorforge::fmacdpp16<0>(v177_acc, v182_lin, v164_data);
          tensorforge::fmacdpp16<1>(v177_acc, v182_lin, v165_data);
          tensorforge::fmacdpp16<2>(v177_acc, v182_lin, v166_data);
          tensorforge::fmacdpp16<3>(v177_acc, v182_lin, v167_data);
          tensorforge::fmacdpp16<4>(v177_acc, v182_lin, v168_data);
          tensorforge::fmacdpp16<5>(v177_acc, v182_lin, v169_data);
          tensorforge::fmacdpp16<6>(v177_acc, v182_lin, v170_data);
          tensorforge::fmacdpp16<7>(v177_acc, v182_lin, v171_data);
          r4[0] = v172_acc;
          r4[1] = v173_acc;
          r4[2] = v174_acc;
          r4[3] = v175_acc;
          r4[4] = v176_acc;
          r4[5] = v177_acc;
          // glb_m2 = store{r>g}(r4);
          if (v14_lead < 12) {
            #pragma unroll
            for (int32_t v187_i1 = 0; v187_i1 < 6; ++v187_i1) {
              float v189_data = r4[v187_i1];
              glb_m2[(v14_lead + (v187_i1 * 12))] = v189_data;
            }
          }
        }
      }
    }
  }
}

