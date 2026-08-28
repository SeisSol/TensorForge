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
              int32_t v19_a = v13_i1 * 12;
              int32_t v20_a = v11_lead + v19_a;
              float v28_data = __builtin_nontemporal_load(&glb_m0[(v11_lead + v19_a)]);
              r0[v13_i1] = v28_data;
            }
          }
          float r1[6]{};
          // r1 = load{g>r}(glb_m1);
          float v31_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v31_lin;
          float v32_lin = glb_m1[16 + threadIdx.x * 1];
          r1[1] = v32_lin;
          float v33_lin = glb_m1[32 + threadIdx.x * 1];
          r1[2] = v33_lin;
          float v34_lin = glb_m1[48 + threadIdx.x * 1];
          r1[3] = v34_lin;
          float v35_lin = glb_m1[64 + threadIdx.x * 1];
          r1[4] = v35_lin;
          float v36_lin = glb_m1[80 + threadIdx.x * 1];
          r1[5] = v36_lin;
          float v37_lin = glb_m1[96 + threadIdx.x * 1];
          r1[6] = v37_lin;
          float v38_lin = glb_m1[112 + threadIdx.x * 1];
          r1[7] = v38_lin;
          float v39_lin = glb_m1[128 + threadIdx.x * 1];
          r1[8] = v39_lin;
          float v40_lin = glb_m1[144 + threadIdx.x * 1];
          r1[9] = v40_lin;
          float v41_lin = glb_m1[160 + threadIdx.x * 1];
          r1[10] = v41_lin;
          float v42_lin = glb_m1[176 + threadIdx.x * 1];
          r1[11] = v42_lin;
          float v43_lin = glb_m1[192 + threadIdx.x * 1];
          r1[12] = v43_lin;
          float v44_lin = glb_m1[208 + threadIdx.x * 1];
          r1[13] = v44_lin;
          float v45_lin = glb_m1[224 + threadIdx.x * 1];
          r1[14] = v45_lin;
          float v46_lin = glb_m1[240 + threadIdx.x * 1];
          r1[15] = v46_lin;
          float v47_lin = glb_m1[256 + threadIdx.x * 1];
          r1[16] = v47_lin;
          float v48_lin = glb_m1[272 + threadIdx.x * 1];
          r1[17] = v48_lin;
          float v49_lin = glb_m1[288 + threadIdx.x * 1];
          r1[18] = v49_lin;
          float v50_lin = glb_m1[304 + threadIdx.x * 1];
          r1[19] = v50_lin;
          float v51_lin = glb_m1[320 + threadIdx.x * 1];
          r1[20] = v51_lin;
          float v52_lin = glb_m1[336 + threadIdx.x * 1];
          r1[21] = v52_lin;
          float v53_lin = glb_m1[352 + threadIdx.x * 1];
          r1[22] = v53_lin;
          float v54_lin = glb_m1[368 + threadIdx.x * 1];
          r1[23] = v54_lin;
          float v55_lin = glb_m1[384 + threadIdx.x * 1];
          r1[24] = v55_lin;
          float v56_lin = glb_m1[400 + threadIdx.x * 1];
          r1[25] = v56_lin;
          float v57_lin = glb_m1[416 + threadIdx.x * 1];
          r1[26] = v57_lin;
          float v58_lin = glb_m1[432 + threadIdx.x * 1];
          r1[27] = v58_lin;
          float v59_lin = glb_m1[448 + threadIdx.x * 1];
          r1[28] = v59_lin;
          float v60_lin = glb_m1[464 + threadIdx.x * 1];
          r1[29] = v60_lin;
          float v61_lin = glb_m1[480 + threadIdx.x * 1];
          r1[30] = v61_lin;
          float v62_lin = glb_m1[496 + threadIdx.x * 1];
          r1[31] = v62_lin;
          float v63_lin = glb_m1[512 + threadIdx.x * 1];
          r1[32] = v63_lin;
          float v64_lin = glb_m1[528 + threadIdx.x * 1];
          r1[33] = v64_lin;
          float v65_lin = glb_m1[544 + threadIdx.x * 1];
          r1[34] = v65_lin;
          float v66_lin = glb_m1[560 + threadIdx.x * 1];
          r1[35] = v66_lin;
          float v67_lin = glb_m1[576 + threadIdx.x * 1];
          r1[36] = v67_lin;
          float v68_lin = glb_m1[592 + threadIdx.x * 1];
          r1[37] = v68_lin;
          float v69_lin = glb_m1[608 + threadIdx.x * 1];
          r1[38] = v69_lin;
          float v70_lin = glb_m1[624 + threadIdx.x * 1];
          r1[39] = v70_lin;
          float v71_lin = glb_m1[640 + threadIdx.x * 1];
          r1[40] = v71_lin;
          float v72_lin = glb_m1[656 + threadIdx.x * 1];
          r1[41] = v72_lin;
          float v73_lin = glb_m1[672 + threadIdx.x * 1];
          r1[42] = v73_lin;
          float v74_lin = glb_m1[688 + threadIdx.x * 1];
          r1[43] = v74_lin;
          float v75_lin = glb_m1[704 + threadIdx.x * 1];
          r1[44] = v75_lin;
          float v76_lin = glb_m1[720 + threadIdx.x * 1];
          r1[45] = v76_lin;
          float v77_lin = glb_m1[736 + threadIdx.x * 1];
          r1[46] = v77_lin;
          float v78_lin = glb_m1[752 + threadIdx.x * 1];
          r1[47] = v78_lin;
          float v79_lin = glb_m1[768 + threadIdx.x * 1];
          r1[48] = v79_lin;
          float v80_lin = glb_m1[784 + threadIdx.x * 1];
          r1[49] = v80_lin;
          float v81_lin = glb_m1[800 + threadIdx.x * 1];
          r1[50] = v81_lin;
          float v82_lin = glb_m1[816 + threadIdx.x * 1];
          r1[51] = v82_lin;
          float v83_lin = glb_m1[832 + threadIdx.x * 1];
          r1[52] = v83_lin;
          float v84_lin = glb_m1[848 + threadIdx.x * 1];
          r1[53] = v84_lin;
          float v85_lin = glb_m1[864 + threadIdx.x * 1];
          r1[54] = v85_lin;
          float v86_lin = glb_m1[880 + threadIdx.x * 1];
          r1[55] = v86_lin;
          float v87_lin = glb_m1[896 + threadIdx.x * 1];
          r1[56] = v87_lin;
          float v88_lin = glb_m1[912 + threadIdx.x * 1];
          r1[57] = v88_lin;
          float v89_lin = glb_m1[928 + threadIdx.x * 1];
          r1[58] = v89_lin;
          float v90_lin = glb_m1[944 + threadIdx.x * 1];
          r1[59] = v90_lin;
          float v91_lin = glb_m1[960 + threadIdx.x * 1];
          r1[60] = v91_lin;
          float v92_lin = glb_m1[976 + threadIdx.x * 1];
          r1[61] = v92_lin;
          float v93_lin = glb_m1[992 + threadIdx.x * 1];
          r1[62] = v93_lin;
          float v94_lin = glb_m1[1008 + threadIdx.x * 1];
          r1[63] = v94_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v100_i1 = 0; v100_i1 < 12; ++v100_i1) {
              int32_t v106_a = v100_i1 * 12;
              int32_t v107_a = v11_lead + v106_a;
              float v115_data = __builtin_nontemporal_load(&glb_m3[(v11_lead + v106_a)]);
              r3[v100_i1] = v115_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          float v118_data = r1[0];
          float v119_data = r1[1];
          float v120_data = r1[2];
          float v121_data = r1[3];
          float v122_tp{};
          float v123_tp{};
          float v124_tp{};
          float v125_tp{};
          tensorforge::transpose4x4b32(v122_tp, v123_tp, v124_tp, v125_tp, v118_data, v119_data, v120_data, v121_data);
          tensorforge::VectorT<float, 4> v126_acc{};
          float v127_data = r0[0];
          float v128_data = r0[1];
          float v129_data = r0[2];
          float v130_data = r0[3];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v127_data, v126_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v128_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v129_data, v132_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v130_data, v133_acc, 2, 0, 0);
          float v135_data = r0[4];
          float v136_data = r0[5];
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v135_data, v134_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v136_data, v139_acc, 2, 1, 0);
          r2[0] = (v140_acc[0]);
          r2[1] = (v140_acc[1]);
          r2[2] = (v140_acc[2]);
          r2[3] = (v140_acc[3]);
          float v145_data = r1[4];
          float v146_data = r1[5];
          float v149_tp{};
          float v150_tp{};
          float v151_tp{};
          float v152_tp{};
          tensorforge::transpose4x4b32(v149_tp, v150_tp, v151_tp, v152_tp, v145_data, v146_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v153_acc{};
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v127_data, v153_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v128_data, v158_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v129_data, v159_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v130_data, v160_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v135_data, v161_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v136_data, v166_acc, 2, 1, 0);
          r2[4] = (v167_acc[0]);
          r2[5] = (v167_acc[1]);
          // wait(r3 = load{g>r}(glb_m3););
          float r4[6]{};
          // r4 = +(r3 * r2) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v171_data = r3[0];
          float v172_data = r3[1];
          float v173_data = r3[2];
          float v174_data = r3[3];
          float v175_data = r3[4];
          float v176_data = r3[5];
          float v177_data = r3[6];
          float v178_data = r3[7];
          float v179_data = r3[8];
          float v180_data = r3[9];
          float v181_data = r3[10];
          float v182_data = r3[11];
          float v183_acc{};
          float v184_acc{};
          float v185_acc{};
          float v186_acc{};
          float v187_acc{};
          float v188_acc{};
          float v189_lin = r2[0];
          tensorforge::fmacdpp16<0>(v183_acc, v189_lin, v171_data);
          tensorforge::fmacdpp16<1>(v183_acc, v189_lin, v172_data);
          tensorforge::fmacdpp16<2>(v183_acc, v189_lin, v173_data);
          tensorforge::fmacdpp16<3>(v183_acc, v189_lin, v174_data);
          tensorforge::fmacdpp16<4>(v183_acc, v189_lin, v175_data);
          tensorforge::fmacdpp16<5>(v183_acc, v189_lin, v176_data);
          tensorforge::fmacdpp16<6>(v183_acc, v189_lin, v177_data);
          tensorforge::fmacdpp16<7>(v183_acc, v189_lin, v178_data);
          tensorforge::fmacdpp16<8>(v183_acc, v189_lin, v179_data);
          tensorforge::fmacdpp16<9>(v183_acc, v189_lin, v180_data);
          tensorforge::fmacdpp16<10>(v183_acc, v189_lin, v181_data);
          tensorforge::fmacdpp16<11>(v183_acc, v189_lin, v182_data);
          tensorforge::fmacdpp16<12>(v184_acc, v189_lin, v171_data);
          tensorforge::fmacdpp16<13>(v184_acc, v189_lin, v172_data);
          tensorforge::fmacdpp16<14>(v184_acc, v189_lin, v173_data);
          tensorforge::fmacdpp16<15>(v184_acc, v189_lin, v174_data);
          float v190_lin = r2[1];
          tensorforge::fmacdpp16<0>(v184_acc, v190_lin, v175_data);
          tensorforge::fmacdpp16<1>(v184_acc, v190_lin, v176_data);
          tensorforge::fmacdpp16<2>(v184_acc, v190_lin, v177_data);
          tensorforge::fmacdpp16<3>(v184_acc, v190_lin, v178_data);
          tensorforge::fmacdpp16<4>(v184_acc, v190_lin, v179_data);
          tensorforge::fmacdpp16<5>(v184_acc, v190_lin, v180_data);
          tensorforge::fmacdpp16<6>(v184_acc, v190_lin, v181_data);
          tensorforge::fmacdpp16<7>(v184_acc, v190_lin, v182_data);
          tensorforge::fmacdpp16<8>(v185_acc, v190_lin, v171_data);
          tensorforge::fmacdpp16<9>(v185_acc, v190_lin, v172_data);
          tensorforge::fmacdpp16<10>(v185_acc, v190_lin, v173_data);
          tensorforge::fmacdpp16<11>(v185_acc, v190_lin, v174_data);
          tensorforge::fmacdpp16<12>(v185_acc, v190_lin, v175_data);
          tensorforge::fmacdpp16<13>(v185_acc, v190_lin, v176_data);
          tensorforge::fmacdpp16<14>(v185_acc, v190_lin, v177_data);
          tensorforge::fmacdpp16<15>(v185_acc, v190_lin, v178_data);
          float v191_lin = r2[2];
          tensorforge::fmacdpp16<0>(v185_acc, v191_lin, v179_data);
          tensorforge::fmacdpp16<1>(v185_acc, v191_lin, v180_data);
          tensorforge::fmacdpp16<2>(v185_acc, v191_lin, v181_data);
          tensorforge::fmacdpp16<3>(v185_acc, v191_lin, v182_data);
          tensorforge::fmacdpp16<4>(v186_acc, v191_lin, v171_data);
          tensorforge::fmacdpp16<5>(v186_acc, v191_lin, v172_data);
          tensorforge::fmacdpp16<6>(v186_acc, v191_lin, v173_data);
          tensorforge::fmacdpp16<7>(v186_acc, v191_lin, v174_data);
          tensorforge::fmacdpp16<8>(v186_acc, v191_lin, v175_data);
          tensorforge::fmacdpp16<9>(v186_acc, v191_lin, v176_data);
          tensorforge::fmacdpp16<10>(v186_acc, v191_lin, v177_data);
          tensorforge::fmacdpp16<11>(v186_acc, v191_lin, v178_data);
          tensorforge::fmacdpp16<12>(v186_acc, v191_lin, v179_data);
          tensorforge::fmacdpp16<13>(v186_acc, v191_lin, v180_data);
          tensorforge::fmacdpp16<14>(v186_acc, v191_lin, v181_data);
          tensorforge::fmacdpp16<15>(v186_acc, v191_lin, v182_data);
          float v192_lin = r2[3];
          tensorforge::fmacdpp16<0>(v187_acc, v192_lin, v171_data);
          tensorforge::fmacdpp16<1>(v187_acc, v192_lin, v172_data);
          tensorforge::fmacdpp16<2>(v187_acc, v192_lin, v173_data);
          tensorforge::fmacdpp16<3>(v187_acc, v192_lin, v174_data);
          tensorforge::fmacdpp16<4>(v187_acc, v192_lin, v175_data);
          tensorforge::fmacdpp16<5>(v187_acc, v192_lin, v176_data);
          tensorforge::fmacdpp16<6>(v187_acc, v192_lin, v177_data);
          tensorforge::fmacdpp16<7>(v187_acc, v192_lin, v178_data);
          tensorforge::fmacdpp16<8>(v187_acc, v192_lin, v179_data);
          tensorforge::fmacdpp16<9>(v187_acc, v192_lin, v180_data);
          tensorforge::fmacdpp16<10>(v187_acc, v192_lin, v181_data);
          tensorforge::fmacdpp16<11>(v187_acc, v192_lin, v182_data);
          tensorforge::fmacdpp16<12>(v188_acc, v192_lin, v171_data);
          tensorforge::fmacdpp16<13>(v188_acc, v192_lin, v172_data);
          tensorforge::fmacdpp16<14>(v188_acc, v192_lin, v173_data);
          tensorforge::fmacdpp16<15>(v188_acc, v192_lin, v174_data);
          float v193_lin = r2[4];
          tensorforge::fmacdpp16<0>(v188_acc, v193_lin, v175_data);
          tensorforge::fmacdpp16<1>(v188_acc, v193_lin, v176_data);
          tensorforge::fmacdpp16<2>(v188_acc, v193_lin, v177_data);
          tensorforge::fmacdpp16<3>(v188_acc, v193_lin, v178_data);
          tensorforge::fmacdpp16<4>(v188_acc, v193_lin, v179_data);
          tensorforge::fmacdpp16<5>(v188_acc, v193_lin, v180_data);
          tensorforge::fmacdpp16<6>(v188_acc, v193_lin, v181_data);
          tensorforge::fmacdpp16<7>(v188_acc, v193_lin, v182_data);
          r4[0] = v183_acc;
          r4[1] = v184_acc;
          r4[2] = v185_acc;
          r4[3] = v186_acc;
          r4[4] = v187_acc;
          r4[5] = v188_acc;
          // glb_m2 = store{r>g}(r4);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v198_i1 = 0; v198_i1 < 6; ++v198_i1) {
              int32_t v199_a = 0 + v198_i1;
              float v201_data = r4[v198_i1];
              glb_m2[(v11_lead + (v198_i1 * 12))] = v201_data;
            }
          }
        }
      }
    }
  }
}

