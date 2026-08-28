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
          int32_t v8_lead = threadIdx.x % 16;
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 6; ++v10_i1) {
              int32_t v16_a = v10_i1 * 12;
              int32_t v17_a = v8_lead + v16_a;
              float v25_data = __builtin_nontemporal_load(&glb_m0[(v8_lead + v16_a)]);
              r0[v10_i1] = v25_data;
            }
          }
          float r1[6]{};
          // r1 = load{g>r}(glb_m1);
          float v28_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v28_lin;
          float v29_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v29_lin;
          float v30_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v30_lin;
          float v31_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v31_lin;
          float v32_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v32_lin;
          float v33_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v33_lin;
          float v34_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v34_lin;
          float v35_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v35_lin;
          float v36_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v36_lin;
          float v37_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v37_lin;
          float v38_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v38_lin;
          float v39_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v39_lin;
          float v40_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v40_lin;
          float v41_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v41_lin;
          float v42_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v42_lin;
          float v43_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v43_lin;
          float v44_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v44_lin;
          float v45_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v45_lin;
          float v46_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v46_lin;
          float v47_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v47_lin;
          float v48_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v48_lin;
          float v49_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v49_lin;
          float v50_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v50_lin;
          float v51_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v51_lin;
          float v52_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v52_lin;
          float v53_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v53_lin;
          float v54_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v54_lin;
          float v55_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v55_lin;
          float v56_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v56_lin;
          float v57_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v57_lin;
          float v58_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v58_lin;
          float v59_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v59_lin;
          float v60_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v60_lin;
          float v61_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v61_lin;
          float v62_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v62_lin;
          float v63_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v63_lin;
          float v64_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v64_lin;
          float v65_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v65_lin;
          float v66_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v66_lin;
          float v67_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v67_lin;
          float v68_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v68_lin;
          float v69_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v69_lin;
          float v70_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v70_lin;
          float v71_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v71_lin;
          float v72_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v72_lin;
          float v73_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v73_lin;
          float v74_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v74_lin;
          float v75_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v75_lin;
          float v76_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v76_lin;
          float v77_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v77_lin;
          float v78_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v78_lin;
          float v79_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v79_lin;
          float v80_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v80_lin;
          float v81_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v81_lin;
          float v82_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v82_lin;
          float v83_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v83_lin;
          float v84_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v84_lin;
          float v85_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v85_lin;
          float v86_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v86_lin;
          float v87_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v87_lin;
          float v88_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v88_lin;
          float v89_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v89_lin;
          float v90_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v90_lin;
          float v91_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v91_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v97_i1 = 0; v97_i1 < 12; ++v97_i1) {
              int32_t v103_a = v97_i1 * 12;
              int32_t v104_a = v8_lead + v103_a;
              float v112_data = __builtin_nontemporal_load(&glb_m3[(v8_lead + v103_a)]);
              r3[v97_i1] = v112_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          float v115_data = r1[0];
          float v116_data = r1[1];
          float v117_data = r1[2];
          float v118_data = r1[3];
          float v119_tp{};
          float v120_tp{};
          float v121_tp{};
          float v122_tp{};
          tensorforge::transpose4x4b32(v119_tp, v120_tp, v121_tp, v122_tp, v115_data, v116_data, v117_data, v118_data);
          tensorforge::VectorT<float, 4> v123_acc{};
          float v124_data = r0[0];
          float v125_data = r0[1];
          float v126_data = r0[2];
          float v127_data = r0[3];
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v124_data, v123_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v125_data, v128_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v126_data, v129_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v127_data, v130_acc, 2, 0, 0);
          float v132_data = r0[4];
          float v133_data = r0[5];
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v132_data, v131_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v133_data, v136_acc, 2, 1, 0);
          r2[0] = (v137_acc[0]);
          r2[1] = (v137_acc[1]);
          r2[2] = (v137_acc[2]);
          r2[3] = (v137_acc[3]);
          float v142_data = r1[4];
          float v143_data = r1[5];
          float v146_tp{};
          float v147_tp{};
          float v148_tp{};
          float v149_tp{};
          tensorforge::transpose4x4b32(v146_tp, v147_tp, v148_tp, v149_tp, v142_data, v143_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v150_acc{};
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v124_data, v150_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v125_data, v155_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v126_data, v156_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v127_data, v157_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v132_data, v158_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v133_data, v163_acc, 2, 1, 0);
          r2[4] = (v164_acc[0]);
          r2[5] = (v164_acc[1]);
          // wait(r3 = load{g>r}(glb_m3););
          float r4[6]{};
          // r4 = +(r3 * r2) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v168_data = r3[0];
          float v169_data = r3[1];
          float v170_data = r3[2];
          float v171_data = r3[3];
          float v172_data = r3[4];
          float v173_data = r3[5];
          float v174_data = r3[6];
          float v175_data = r3[7];
          float v176_data = r3[8];
          float v177_data = r3[9];
          float v178_data = r3[10];
          float v179_data = r3[11];
          float v180_acc{};
          float v181_acc{};
          float v182_acc{};
          float v183_acc{};
          float v184_acc{};
          float v185_acc{};
          float v186_lin = r2[0];
          tensorforge::fmacdpp16<0>(v180_acc, v186_lin, v168_data);
          tensorforge::fmacdpp16<1>(v180_acc, v186_lin, v169_data);
          tensorforge::fmacdpp16<2>(v180_acc, v186_lin, v170_data);
          tensorforge::fmacdpp16<3>(v180_acc, v186_lin, v171_data);
          tensorforge::fmacdpp16<4>(v180_acc, v186_lin, v172_data);
          tensorforge::fmacdpp16<5>(v180_acc, v186_lin, v173_data);
          tensorforge::fmacdpp16<6>(v180_acc, v186_lin, v174_data);
          tensorforge::fmacdpp16<7>(v180_acc, v186_lin, v175_data);
          tensorforge::fmacdpp16<8>(v180_acc, v186_lin, v176_data);
          tensorforge::fmacdpp16<9>(v180_acc, v186_lin, v177_data);
          tensorforge::fmacdpp16<10>(v180_acc, v186_lin, v178_data);
          tensorforge::fmacdpp16<11>(v180_acc, v186_lin, v179_data);
          tensorforge::fmacdpp16<12>(v181_acc, v186_lin, v168_data);
          tensorforge::fmacdpp16<13>(v181_acc, v186_lin, v169_data);
          tensorforge::fmacdpp16<14>(v181_acc, v186_lin, v170_data);
          tensorforge::fmacdpp16<15>(v181_acc, v186_lin, v171_data);
          float v187_lin = r2[1];
          tensorforge::fmacdpp16<0>(v181_acc, v187_lin, v172_data);
          tensorforge::fmacdpp16<1>(v181_acc, v187_lin, v173_data);
          tensorforge::fmacdpp16<2>(v181_acc, v187_lin, v174_data);
          tensorforge::fmacdpp16<3>(v181_acc, v187_lin, v175_data);
          tensorforge::fmacdpp16<4>(v181_acc, v187_lin, v176_data);
          tensorforge::fmacdpp16<5>(v181_acc, v187_lin, v177_data);
          tensorforge::fmacdpp16<6>(v181_acc, v187_lin, v178_data);
          tensorforge::fmacdpp16<7>(v181_acc, v187_lin, v179_data);
          tensorforge::fmacdpp16<8>(v182_acc, v187_lin, v168_data);
          tensorforge::fmacdpp16<9>(v182_acc, v187_lin, v169_data);
          tensorforge::fmacdpp16<10>(v182_acc, v187_lin, v170_data);
          tensorforge::fmacdpp16<11>(v182_acc, v187_lin, v171_data);
          tensorforge::fmacdpp16<12>(v182_acc, v187_lin, v172_data);
          tensorforge::fmacdpp16<13>(v182_acc, v187_lin, v173_data);
          tensorforge::fmacdpp16<14>(v182_acc, v187_lin, v174_data);
          tensorforge::fmacdpp16<15>(v182_acc, v187_lin, v175_data);
          float v188_lin = r2[2];
          tensorforge::fmacdpp16<0>(v182_acc, v188_lin, v176_data);
          tensorforge::fmacdpp16<1>(v182_acc, v188_lin, v177_data);
          tensorforge::fmacdpp16<2>(v182_acc, v188_lin, v178_data);
          tensorforge::fmacdpp16<3>(v182_acc, v188_lin, v179_data);
          tensorforge::fmacdpp16<4>(v183_acc, v188_lin, v168_data);
          tensorforge::fmacdpp16<5>(v183_acc, v188_lin, v169_data);
          tensorforge::fmacdpp16<6>(v183_acc, v188_lin, v170_data);
          tensorforge::fmacdpp16<7>(v183_acc, v188_lin, v171_data);
          tensorforge::fmacdpp16<8>(v183_acc, v188_lin, v172_data);
          tensorforge::fmacdpp16<9>(v183_acc, v188_lin, v173_data);
          tensorforge::fmacdpp16<10>(v183_acc, v188_lin, v174_data);
          tensorforge::fmacdpp16<11>(v183_acc, v188_lin, v175_data);
          tensorforge::fmacdpp16<12>(v183_acc, v188_lin, v176_data);
          tensorforge::fmacdpp16<13>(v183_acc, v188_lin, v177_data);
          tensorforge::fmacdpp16<14>(v183_acc, v188_lin, v178_data);
          tensorforge::fmacdpp16<15>(v183_acc, v188_lin, v179_data);
          float v189_lin = r2[3];
          tensorforge::fmacdpp16<0>(v184_acc, v189_lin, v168_data);
          tensorforge::fmacdpp16<1>(v184_acc, v189_lin, v169_data);
          tensorforge::fmacdpp16<2>(v184_acc, v189_lin, v170_data);
          tensorforge::fmacdpp16<3>(v184_acc, v189_lin, v171_data);
          tensorforge::fmacdpp16<4>(v184_acc, v189_lin, v172_data);
          tensorforge::fmacdpp16<5>(v184_acc, v189_lin, v173_data);
          tensorforge::fmacdpp16<6>(v184_acc, v189_lin, v174_data);
          tensorforge::fmacdpp16<7>(v184_acc, v189_lin, v175_data);
          tensorforge::fmacdpp16<8>(v184_acc, v189_lin, v176_data);
          tensorforge::fmacdpp16<9>(v184_acc, v189_lin, v177_data);
          tensorforge::fmacdpp16<10>(v184_acc, v189_lin, v178_data);
          tensorforge::fmacdpp16<11>(v184_acc, v189_lin, v179_data);
          tensorforge::fmacdpp16<12>(v185_acc, v189_lin, v168_data);
          tensorforge::fmacdpp16<13>(v185_acc, v189_lin, v169_data);
          tensorforge::fmacdpp16<14>(v185_acc, v189_lin, v170_data);
          tensorforge::fmacdpp16<15>(v185_acc, v189_lin, v171_data);
          float v190_lin = r2[4];
          tensorforge::fmacdpp16<0>(v185_acc, v190_lin, v172_data);
          tensorforge::fmacdpp16<1>(v185_acc, v190_lin, v173_data);
          tensorforge::fmacdpp16<2>(v185_acc, v190_lin, v174_data);
          tensorforge::fmacdpp16<3>(v185_acc, v190_lin, v175_data);
          tensorforge::fmacdpp16<4>(v185_acc, v190_lin, v176_data);
          tensorforge::fmacdpp16<5>(v185_acc, v190_lin, v177_data);
          tensorforge::fmacdpp16<6>(v185_acc, v190_lin, v178_data);
          tensorforge::fmacdpp16<7>(v185_acc, v190_lin, v179_data);
          r4[0] = v180_acc;
          r4[1] = v181_acc;
          r4[2] = v182_acc;
          r4[3] = v183_acc;
          r4[4] = v184_acc;
          r4[5] = v185_acc;
          // glb_m2 = store{r>g}(r4);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v195_i1 = 0; v195_i1 < 6; ++v195_i1) {
              int32_t v196_a = 0 + v195_i1;
              float v198_data = r4[v195_i1];
              glb_m2[(v8_lead + (v195_i1 * 12))] = v198_data;
            }
          }
        }
      }
    }
  }
}

