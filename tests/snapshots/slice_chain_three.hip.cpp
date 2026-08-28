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
          int32_t v11_lead = threadIdx.x % 16;
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v13_i1 = 0; v13_i1 < 6; ++v13_i1) {
              float v21_data = __builtin_nontemporal_load(&glb_m0[(v11_lead + (v13_i1 * 12))]);
              r0[v13_i1] = v21_data;
            }
          }
          float r1[6]{};
          // r1 = load{g>r}(glb_m1);
          float v24_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v24_lin;
          float v25_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v25_lin;
          float v26_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v26_lin;
          float v27_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v27_lin;
          float v28_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v28_lin;
          float v29_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v29_lin;
          float v30_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v30_lin;
          float v31_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v31_lin;
          float v32_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v32_lin;
          float v33_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v33_lin;
          float v34_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v34_lin;
          float v35_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v35_lin;
          float v36_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v36_lin;
          float v37_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v37_lin;
          float v38_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v38_lin;
          float v39_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v39_lin;
          float v40_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v40_lin;
          float v41_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v41_lin;
          float v42_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v42_lin;
          float v43_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v43_lin;
          float v44_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v44_lin;
          float v45_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v45_lin;
          float v46_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v46_lin;
          float v47_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v47_lin;
          float v48_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v48_lin;
          float v49_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v49_lin;
          float v50_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v50_lin;
          float v51_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v51_lin;
          float v52_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v52_lin;
          float v53_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v53_lin;
          float v54_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v54_lin;
          float v55_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v55_lin;
          float v56_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v56_lin;
          float v57_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v57_lin;
          float v58_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v58_lin;
          float v59_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v59_lin;
          float v60_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v60_lin;
          float v61_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v61_lin;
          float v62_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v62_lin;
          float v63_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v63_lin;
          float v64_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v64_lin;
          float v65_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v65_lin;
          float v66_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v66_lin;
          float v67_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v67_lin;
          float v68_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v68_lin;
          float v69_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v69_lin;
          float v70_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v70_lin;
          float v71_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v71_lin;
          float v72_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v72_lin;
          float v73_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v73_lin;
          float v74_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v74_lin;
          float v75_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v75_lin;
          float v76_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v76_lin;
          float v77_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v77_lin;
          float v78_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v78_lin;
          float v79_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v79_lin;
          float v80_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v80_lin;
          float v81_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v81_lin;
          float v82_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v82_lin;
          float v83_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v83_lin;
          float v84_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v84_lin;
          float v85_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v85_lin;
          float v86_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v86_lin;
          float v87_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v87_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v93_i1 = 0; v93_i1 < 12; ++v93_i1) {
              float v101_data = __builtin_nontemporal_load(&glb_m3[(v11_lead + (v93_i1 * 12))]);
              r3[v93_i1] = v101_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          float v104_data = r1[0];
          float v105_data = r1[1];
          float v106_data = r1[2];
          float v107_data = r1[3];
          float v108_tp{};
          float v109_tp{};
          float v110_tp{};
          float v111_tp{};
          tensorforge::transpose4x4b32(v108_tp, v109_tp, v110_tp, v111_tp, v104_data, v105_data, v106_data, v107_data);
          tensorforge::VectorT<float, 4> v112_acc{};
          float v113_data = r0[0];
          float v114_data = r0[1];
          float v115_data = r0[2];
          float v116_data = r0[3];
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v113_data, v112_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v114_data, v117_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v115_data, v118_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v116_data, v119_acc, 2, 0, 0);
          float v121_data = r0[4];
          float v122_data = r0[5];
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v121_data, v120_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v122_data, v125_acc, 2, 1, 0);
          r2[0] = (v126_acc[0]);
          r2[1] = (v126_acc[1]);
          r2[2] = (v126_acc[2]);
          r2[3] = (v126_acc[3]);
          float v131_data = r1[4];
          float v132_data = r1[5];
          float v135_tp{};
          float v136_tp{};
          float v137_tp{};
          float v138_tp{};
          tensorforge::transpose4x4b32(v135_tp, v136_tp, v137_tp, v138_tp, v131_data, v132_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v139_acc{};
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v113_data, v139_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v114_data, v144_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v115_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v116_data, v146_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v121_data, v147_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v122_data, v152_acc, 2, 1, 0);
          r2[4] = (v153_acc[0]);
          r2[5] = (v153_acc[1]);
          // wait(r3 = load{g>r}(glb_m3););
          float r4[6]{};
          // r4 = +(r3 * r2) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v157_data = r3[0];
          float v158_data = r3[1];
          float v159_data = r3[2];
          float v160_data = r3[3];
          float v161_data = r3[4];
          float v162_data = r3[5];
          float v163_data = r3[6];
          float v164_data = r3[7];
          float v165_data = r3[8];
          float v166_data = r3[9];
          float v167_data = r3[10];
          float v168_data = r3[11];
          float v169_acc{};
          float v170_acc{};
          float v171_acc{};
          float v172_acc{};
          float v173_acc{};
          float v174_acc{};
          float v175_lin = r2[0];
          tensorforge::fmacdpp16<0>(v169_acc, v175_lin, v157_data);
          tensorforge::fmacdpp16<1>(v169_acc, v175_lin, v158_data);
          tensorforge::fmacdpp16<2>(v169_acc, v175_lin, v159_data);
          tensorforge::fmacdpp16<3>(v169_acc, v175_lin, v160_data);
          tensorforge::fmacdpp16<4>(v169_acc, v175_lin, v161_data);
          tensorforge::fmacdpp16<5>(v169_acc, v175_lin, v162_data);
          tensorforge::fmacdpp16<6>(v169_acc, v175_lin, v163_data);
          tensorforge::fmacdpp16<7>(v169_acc, v175_lin, v164_data);
          tensorforge::fmacdpp16<8>(v169_acc, v175_lin, v165_data);
          tensorforge::fmacdpp16<9>(v169_acc, v175_lin, v166_data);
          tensorforge::fmacdpp16<10>(v169_acc, v175_lin, v167_data);
          tensorforge::fmacdpp16<11>(v169_acc, v175_lin, v168_data);
          tensorforge::fmacdpp16<12>(v170_acc, v175_lin, v157_data);
          tensorforge::fmacdpp16<13>(v170_acc, v175_lin, v158_data);
          tensorforge::fmacdpp16<14>(v170_acc, v175_lin, v159_data);
          tensorforge::fmacdpp16<15>(v170_acc, v175_lin, v160_data);
          float v176_lin = r2[1];
          tensorforge::fmacdpp16<0>(v170_acc, v176_lin, v161_data);
          tensorforge::fmacdpp16<1>(v170_acc, v176_lin, v162_data);
          tensorforge::fmacdpp16<2>(v170_acc, v176_lin, v163_data);
          tensorforge::fmacdpp16<3>(v170_acc, v176_lin, v164_data);
          tensorforge::fmacdpp16<4>(v170_acc, v176_lin, v165_data);
          tensorforge::fmacdpp16<5>(v170_acc, v176_lin, v166_data);
          tensorforge::fmacdpp16<6>(v170_acc, v176_lin, v167_data);
          tensorforge::fmacdpp16<7>(v170_acc, v176_lin, v168_data);
          tensorforge::fmacdpp16<8>(v171_acc, v176_lin, v157_data);
          tensorforge::fmacdpp16<9>(v171_acc, v176_lin, v158_data);
          tensorforge::fmacdpp16<10>(v171_acc, v176_lin, v159_data);
          tensorforge::fmacdpp16<11>(v171_acc, v176_lin, v160_data);
          tensorforge::fmacdpp16<12>(v171_acc, v176_lin, v161_data);
          tensorforge::fmacdpp16<13>(v171_acc, v176_lin, v162_data);
          tensorforge::fmacdpp16<14>(v171_acc, v176_lin, v163_data);
          tensorforge::fmacdpp16<15>(v171_acc, v176_lin, v164_data);
          float v177_lin = r2[2];
          tensorforge::fmacdpp16<0>(v171_acc, v177_lin, v165_data);
          tensorforge::fmacdpp16<1>(v171_acc, v177_lin, v166_data);
          tensorforge::fmacdpp16<2>(v171_acc, v177_lin, v167_data);
          tensorforge::fmacdpp16<3>(v171_acc, v177_lin, v168_data);
          tensorforge::fmacdpp16<4>(v172_acc, v177_lin, v157_data);
          tensorforge::fmacdpp16<5>(v172_acc, v177_lin, v158_data);
          tensorforge::fmacdpp16<6>(v172_acc, v177_lin, v159_data);
          tensorforge::fmacdpp16<7>(v172_acc, v177_lin, v160_data);
          tensorforge::fmacdpp16<8>(v172_acc, v177_lin, v161_data);
          tensorforge::fmacdpp16<9>(v172_acc, v177_lin, v162_data);
          tensorforge::fmacdpp16<10>(v172_acc, v177_lin, v163_data);
          tensorforge::fmacdpp16<11>(v172_acc, v177_lin, v164_data);
          tensorforge::fmacdpp16<12>(v172_acc, v177_lin, v165_data);
          tensorforge::fmacdpp16<13>(v172_acc, v177_lin, v166_data);
          tensorforge::fmacdpp16<14>(v172_acc, v177_lin, v167_data);
          tensorforge::fmacdpp16<15>(v172_acc, v177_lin, v168_data);
          float v178_lin = r2[3];
          tensorforge::fmacdpp16<0>(v173_acc, v178_lin, v157_data);
          tensorforge::fmacdpp16<1>(v173_acc, v178_lin, v158_data);
          tensorforge::fmacdpp16<2>(v173_acc, v178_lin, v159_data);
          tensorforge::fmacdpp16<3>(v173_acc, v178_lin, v160_data);
          tensorforge::fmacdpp16<4>(v173_acc, v178_lin, v161_data);
          tensorforge::fmacdpp16<5>(v173_acc, v178_lin, v162_data);
          tensorforge::fmacdpp16<6>(v173_acc, v178_lin, v163_data);
          tensorforge::fmacdpp16<7>(v173_acc, v178_lin, v164_data);
          tensorforge::fmacdpp16<8>(v173_acc, v178_lin, v165_data);
          tensorforge::fmacdpp16<9>(v173_acc, v178_lin, v166_data);
          tensorforge::fmacdpp16<10>(v173_acc, v178_lin, v167_data);
          tensorforge::fmacdpp16<11>(v173_acc, v178_lin, v168_data);
          tensorforge::fmacdpp16<12>(v174_acc, v178_lin, v157_data);
          tensorforge::fmacdpp16<13>(v174_acc, v178_lin, v158_data);
          tensorforge::fmacdpp16<14>(v174_acc, v178_lin, v159_data);
          tensorforge::fmacdpp16<15>(v174_acc, v178_lin, v160_data);
          float v179_lin = r2[4];
          tensorforge::fmacdpp16<0>(v174_acc, v179_lin, v161_data);
          tensorforge::fmacdpp16<1>(v174_acc, v179_lin, v162_data);
          tensorforge::fmacdpp16<2>(v174_acc, v179_lin, v163_data);
          tensorforge::fmacdpp16<3>(v174_acc, v179_lin, v164_data);
          tensorforge::fmacdpp16<4>(v174_acc, v179_lin, v165_data);
          tensorforge::fmacdpp16<5>(v174_acc, v179_lin, v166_data);
          tensorforge::fmacdpp16<6>(v174_acc, v179_lin, v167_data);
          tensorforge::fmacdpp16<7>(v174_acc, v179_lin, v168_data);
          r4[0] = v169_acc;
          r4[1] = v170_acc;
          r4[2] = v171_acc;
          r4[3] = v172_acc;
          r4[4] = v173_acc;
          r4[5] = v174_acc;
          // glb_m2 = store{r>g}(r4);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v184_i1 = 0; v184_i1 < 6; ++v184_i1) {
              float v186_data = r4[v184_i1];
              glb_m2[(v11_lead + (v184_i1 * 12))] = v186_data;
            }
          }
        }
      }
    }
  }
}

