// === base name ===
kernel_ac369fbcce17954a

// === header ===
void launcher_kernel_ac369fbcce17954a(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ac369fbcce17954a(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ac369fbcce17954a, block.x * block.y * block.z, 64 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (64 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_ac369fbcce17954a, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (64 * sizeof(float)));
          blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
        }
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_ac369fbcce17954a), hipFuncAttributeMaxDynamicSharedMemorySize, 64 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_ac369fbcce17954a, grid, block, 64 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ac369fbcce17954a(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 64×13(64×13) {0..64}×{0..13} pointer_based
    // m1 6(6) {0..6} none
    // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
    // t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..64}×{0..13}×{0..6})[0, 1, 2] = m0 64×13(64×13) {0..64}×{0..13} pointer_based({0..64}×{0..13})[0, 1]×m1 6(6) {0..6} none({0..6})[2]
    // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based({0..15}×{0..1}×{0..6})[0, 1, 2] += t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..15}×{0..1}×{0..6})[0, 1, 2]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 64];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 6) {
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      }
      __syncthreads();
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v5_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[v5_batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v18_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 2; ++v19_i0) {
            int32_t v25_lead = v18_lead + (v19_i0 * 32);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 13; ++v20_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m0[(v25_lead + (v20_i1 * 64))]);
              r0[(v19_i0 + (v20_i1 * 2))] = v28_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v32_data = glb_m1[0];
          float v36_tp{};
          float v37_tp{};
          float v38_tp{};
          float v39_tp{};
          tensorforge::transpose4x4b32(v36_tp, v37_tp, v38_tp, v39_tp, v32_data, v32_data, v32_data, v32_data);
          tensorforge::VectorT<float, 4> v40_acc{};
          float v41_data = r0[0];
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v41_data, v40_acc, 3, 0, 0);
          r1[0] = (v45_acc[0]);
          r1[2] = (v45_acc[1]);
          r1[4] = (v45_acc[2]);
          r1[6] = (v45_acc[3]);
          tensorforge::VectorT<float, 4> v50_acc{};
          float v51_data = r0[1];
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v51_data, v50_acc, 3, 0, 0);
          r1[1] = (v55_acc[0]);
          r1[3] = (v55_acc[1]);
          r1[5] = (v55_acc[2]);
          r1[7] = (v55_acc[3]);
          float v64_tp{};
          float v65_tp{};
          float v66_tp{};
          float v67_tp{};
          tensorforge::transpose4x4b32(v64_tp, v65_tp, v66_tp, v67_tp, v32_data, v32_data, v32_data, v32_data);
          tensorforge::VectorT<float, 4> v68_acc{};
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v41_data, v68_acc, 3, 0, 0);
          r1[8] = (v73_acc[0]);
          r1[10] = (v73_acc[1]);
          r1[12] = (v73_acc[2]);
          r1[14] = (v73_acc[3]);
          tensorforge::VectorT<float, 4> v78_acc{};
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v51_data, v78_acc, 3, 0, 0);
          r1[9] = (v83_acc[0]);
          r1[11] = (v83_acc[1]);
          r1[13] = (v83_acc[2]);
          r1[15] = (v83_acc[3]);
          float v92_tp{};
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          tensorforge::transpose4x4b32(v92_tp, v93_tp, v94_tp, v95_tp, v32_data, v32_data, v32_data, v32_data);
          tensorforge::VectorT<float, 4> v96_acc{};
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v41_data, v96_acc, 3, 0, 0);
          r1[16] = (v101_acc[0]);
          r1[18] = (v101_acc[1]);
          r1[20] = (v101_acc[2]);
          r1[22] = (v101_acc[3]);
          tensorforge::VectorT<float, 4> v106_acc{};
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v51_data, v106_acc, 3, 0, 0);
          r1[17] = (v111_acc[0]);
          r1[19] = (v111_acc[1]);
          r1[21] = (v111_acc[2]);
          r1[23] = (v111_acc[3]);
          float v117_data = glb_m1[1];
          float v120_tp{};
          float v121_tp{};
          float v122_tp{};
          float v123_tp{};
          tensorforge::transpose4x4b32(v120_tp, v121_tp, v122_tp, v123_tp, v32_data, v117_data, v117_data, v117_data);
          tensorforge::VectorT<float, 4> v124_acc{};
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v41_data, v124_acc, 3, 0, 0);
          r1[24] = (v129_acc[0]);
          r1[26] = (v129_acc[1]);
          r1[28] = (v129_acc[2]);
          r1[30] = (v129_acc[3]);
          tensorforge::VectorT<float, 4> v134_acc{};
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v51_data, v134_acc, 3, 0, 0);
          r1[25] = (v139_acc[0]);
          r1[27] = (v139_acc[1]);
          r1[29] = (v139_acc[2]);
          r1[31] = (v139_acc[3]);
          float v148_tp{};
          float v149_tp{};
          float v150_tp{};
          float v151_tp{};
          tensorforge::transpose4x4b32(v148_tp, v149_tp, v150_tp, v151_tp, v117_data, v117_data, v117_data, v117_data);
          tensorforge::VectorT<float, 4> v152_acc{};
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v41_data, v152_acc, 3, 0, 0);
          r1[32] = (v157_acc[0]);
          r1[34] = (v157_acc[1]);
          r1[36] = (v157_acc[2]);
          r1[38] = (v157_acc[3]);
          tensorforge::VectorT<float, 4> v162_acc{};
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v51_data, v162_acc, 3, 0, 0);
          r1[33] = (v167_acc[0]);
          r1[35] = (v167_acc[1]);
          r1[37] = (v167_acc[2]);
          r1[39] = (v167_acc[3]);
          float v176_tp{};
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          tensorforge::transpose4x4b32(v176_tp, v177_tp, v178_tp, v179_tp, v117_data, v117_data, v117_data, v117_data);
          tensorforge::VectorT<float, 4> v180_acc{};
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v41_data, v180_acc, 3, 0, 0);
          r1[40] = (v185_acc[0]);
          r1[42] = (v185_acc[1]);
          r1[44] = (v185_acc[2]);
          r1[46] = (v185_acc[3]);
          tensorforge::VectorT<float, 4> v190_acc{};
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v51_data, v190_acc, 3, 0, 0);
          r1[41] = (v195_acc[0]);
          r1[43] = (v195_acc[1]);
          r1[45] = (v195_acc[2]);
          r1[47] = (v195_acc[3]);
          float v202_data = glb_m1[2];
          float v204_tp{};
          float v205_tp{};
          float v206_tp{};
          float v207_tp{};
          tensorforge::transpose4x4b32(v204_tp, v205_tp, v206_tp, v207_tp, v117_data, v117_data, v202_data, v202_data);
          tensorforge::VectorT<float, 4> v208_acc{};
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v41_data, v208_acc, 3, 0, 0);
          r1[48] = (v213_acc[0]);
          r1[50] = (v213_acc[1]);
          r1[52] = (v213_acc[2]);
          r1[54] = (v213_acc[3]);
          tensorforge::VectorT<float, 4> v218_acc{};
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v51_data, v218_acc, 3, 0, 0);
          r1[49] = (v223_acc[0]);
          r1[51] = (v223_acc[1]);
          r1[53] = (v223_acc[2]);
          r1[55] = (v223_acc[3]);
          float v232_tp{};
          float v233_tp{};
          float v234_tp{};
          float v235_tp{};
          tensorforge::transpose4x4b32(v232_tp, v233_tp, v234_tp, v235_tp, v202_data, v202_data, v202_data, v202_data);
          tensorforge::VectorT<float, 4> v236_acc{};
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v41_data, v236_acc, 3, 0, 0);
          r1[56] = (v241_acc[0]);
          r1[58] = (v241_acc[1]);
          r1[60] = (v241_acc[2]);
          r1[62] = (v241_acc[3]);
          tensorforge::VectorT<float, 4> v246_acc{};
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v51_data, v246_acc, 3, 0, 0);
          r1[57] = (v251_acc[0]);
          r1[59] = (v251_acc[1]);
          r1[61] = (v251_acc[2]);
          r1[63] = (v251_acc[3]);
          float v260_tp{};
          float v261_tp{};
          float v262_tp{};
          float v263_tp{};
          tensorforge::transpose4x4b32(v260_tp, v261_tp, v262_tp, v263_tp, v202_data, v202_data, v202_data, v202_data);
          tensorforge::VectorT<float, 4> v264_acc{};
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v41_data, v264_acc, 3, 0, 0);
          r1[64] = (v269_acc[0]);
          r1[66] = (v269_acc[1]);
          r1[68] = (v269_acc[2]);
          r1[70] = (v269_acc[3]);
          tensorforge::VectorT<float, 4> v274_acc{};
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v51_data, v274_acc, 3, 0, 0);
          r1[65] = (v279_acc[0]);
          r1[67] = (v279_acc[1]);
          r1[69] = (v279_acc[2]);
          r1[71] = (v279_acc[3]);
          float v287_data = glb_m1[3];
          float v288_tp{};
          float v289_tp{};
          float v290_tp{};
          float v291_tp{};
          tensorforge::transpose4x4b32(v288_tp, v289_tp, v290_tp, v291_tp, v202_data, v202_data, v202_data, v287_data);
          tensorforge::VectorT<float, 4> v292_acc{};
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v41_data, v292_acc, 3, 0, 0);
          r1[72] = (v297_acc[0]);
          r1[74] = (v297_acc[1]);
          r1[76] = (v297_acc[2]);
          r1[78] = (v297_acc[3]);
          tensorforge::VectorT<float, 4> v302_acc{};
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v51_data, v302_acc, 3, 0, 0);
          r1[73] = (v307_acc[0]);
          r1[75] = (v307_acc[1]);
          r1[77] = (v307_acc[2]);
          r1[79] = (v307_acc[3]);
          float v316_tp{};
          float v317_tp{};
          float v318_tp{};
          float v319_tp{};
          tensorforge::transpose4x4b32(v316_tp, v317_tp, v318_tp, v319_tp, v287_data, v287_data, v287_data, v287_data);
          tensorforge::VectorT<float, 4> v320_acc{};
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v41_data, v320_acc, 3, 0, 0);
          r1[80] = (v325_acc[0]);
          r1[82] = (v325_acc[1]);
          r1[84] = (v325_acc[2]);
          r1[86] = (v325_acc[3]);
          tensorforge::VectorT<float, 4> v330_acc{};
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v51_data, v330_acc, 3, 0, 0);
          r1[81] = (v335_acc[0]);
          r1[83] = (v335_acc[1]);
          r1[85] = (v335_acc[2]);
          r1[87] = (v335_acc[3]);
          float v344_tp{};
          float v345_tp{};
          float v346_tp{};
          float v347_tp{};
          tensorforge::transpose4x4b32(v344_tp, v345_tp, v346_tp, v347_tp, v287_data, v287_data, v287_data, v287_data);
          tensorforge::VectorT<float, 4> v348_acc{};
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v41_data, v348_acc, 3, 0, 0);
          r1[88] = (v353_acc[0]);
          r1[90] = (v353_acc[1]);
          r1[92] = (v353_acc[2]);
          r1[94] = (v353_acc[3]);
          tensorforge::VectorT<float, 4> v358_acc{};
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v344_tp, v51_data, v358_acc, 3, 0, 0);
          r1[89] = (v363_acc[0]);
          r1[91] = (v363_acc[1]);
          r1[93] = (v363_acc[2]);
          r1[95] = (v363_acc[3]);
          float v372_tp{};
          float v373_tp{};
          float v374_tp{};
          float v375_tp{};
          tensorforge::transpose4x4b32(v372_tp, v373_tp, v374_tp, v375_tp, v287_data, v287_data, v287_data, v287_data);
          tensorforge::VectorT<float, 4> v376_acc{};
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v41_data, v376_acc, 3, 0, 0);
          r1[96] = (v381_acc[0]);
          r1[98] = (v381_acc[1]);
          r1[100] = (v381_acc[2]);
          r1[102] = (v381_acc[3]);
          tensorforge::VectorT<float, 4> v386_acc{};
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v51_data, v386_acc, 3, 0, 0);
          r1[97] = (v391_acc[0]);
          r1[99] = (v391_acc[1]);
          r1[101] = (v391_acc[2]);
          r1[103] = (v391_acc[3]);
          float v396_data = glb_m1[4];
          float v400_tp{};
          float v401_tp{};
          float v402_tp{};
          float v403_tp{};
          tensorforge::transpose4x4b32(v400_tp, v401_tp, v402_tp, v403_tp, v396_data, v396_data, v396_data, v396_data);
          tensorforge::VectorT<float, 4> v404_acc{};
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v41_data, v404_acc, 3, 0, 0);
          r1[104] = (v409_acc[0]);
          r1[106] = (v409_acc[1]);
          r1[108] = (v409_acc[2]);
          r1[110] = (v409_acc[3]);
          tensorforge::VectorT<float, 4> v414_acc{};
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v400_tp, v51_data, v414_acc, 3, 0, 0);
          r1[105] = (v419_acc[0]);
          r1[107] = (v419_acc[1]);
          r1[109] = (v419_acc[2]);
          r1[111] = (v419_acc[3]);
          float v428_tp{};
          float v429_tp{};
          float v430_tp{};
          float v431_tp{};
          tensorforge::transpose4x4b32(v428_tp, v429_tp, v430_tp, v431_tp, v396_data, v396_data, v396_data, v396_data);
          tensorforge::VectorT<float, 4> v432_acc{};
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v41_data, v432_acc, 3, 0, 0);
          r1[112] = (v437_acc[0]);
          r1[114] = (v437_acc[1]);
          r1[116] = (v437_acc[2]);
          r1[118] = (v437_acc[3]);
          tensorforge::VectorT<float, 4> v442_acc{};
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v51_data, v442_acc, 3, 0, 0);
          r1[113] = (v447_acc[0]);
          r1[115] = (v447_acc[1]);
          r1[117] = (v447_acc[2]);
          r1[119] = (v447_acc[3]);
          float v456_tp{};
          float v457_tp{};
          float v458_tp{};
          float v459_tp{};
          tensorforge::transpose4x4b32(v456_tp, v457_tp, v458_tp, v459_tp, v396_data, v396_data, v396_data, v396_data);
          tensorforge::VectorT<float, 4> v460_acc{};
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v41_data, v460_acc, 3, 0, 0);
          r1[120] = (v465_acc[0]);
          r1[122] = (v465_acc[1]);
          r1[124] = (v465_acc[2]);
          r1[126] = (v465_acc[3]);
          tensorforge::VectorT<float, 4> v470_acc{};
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v51_data, v470_acc, 3, 0, 0);
          r1[121] = (v475_acc[0]);
          r1[123] = (v475_acc[1]);
          r1[125] = (v475_acc[2]);
          r1[127] = (v475_acc[3]);
          float v481_data = glb_m1[5];
          float v484_tp{};
          float v485_tp{};
          float v486_tp{};
          float v487_tp{};
          tensorforge::transpose4x4b32(v484_tp, v485_tp, v486_tp, v487_tp, v396_data, v481_data, v481_data, v481_data);
          tensorforge::VectorT<float, 4> v488_acc{};
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v41_data, v488_acc, 3, 0, 0);
          r1[128] = (v493_acc[0]);
          r1[130] = (v493_acc[1]);
          r1[132] = (v493_acc[2]);
          r1[134] = (v493_acc[3]);
          tensorforge::VectorT<float, 4> v498_acc{};
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v484_tp, v51_data, v498_acc, 3, 0, 0);
          r1[129] = (v503_acc[0]);
          r1[131] = (v503_acc[1]);
          r1[133] = (v503_acc[2]);
          r1[135] = (v503_acc[3]);
          float v512_tp{};
          float v513_tp{};
          float v514_tp{};
          float v515_tp{};
          tensorforge::transpose4x4b32(v512_tp, v513_tp, v514_tp, v515_tp, v481_data, v481_data, v481_data, v481_data);
          tensorforge::VectorT<float, 4> v516_acc{};
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v41_data, v516_acc, 3, 0, 0);
          r1[136] = (v521_acc[0]);
          r1[138] = (v521_acc[1]);
          r1[140] = (v521_acc[2]);
          r1[142] = (v521_acc[3]);
          tensorforge::VectorT<float, 4> v526_acc{};
          tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v512_tp, v51_data, v526_acc, 3, 0, 0);
          r1[137] = (v531_acc[0]);
          r1[139] = (v531_acc[1]);
          r1[141] = (v531_acc[2]);
          r1[143] = (v531_acc[3]);
          float v540_tp{};
          float v541_tp{};
          float v542_tp{};
          float v543_tp{};
          tensorforge::transpose4x4b32(v540_tp, v541_tp, v542_tp, v543_tp, v481_data, v481_data, v481_data, v481_data);
          tensorforge::VectorT<float, 4> v544_acc{};
          tensorforge::VectorT<float, 4> v549_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v540_tp, v41_data, v544_acc, 3, 0, 0);
          r1[144] = (v549_acc[0]);
          r1[146] = (v549_acc[1]);
          r1[148] = (v549_acc[2]);
          r1[150] = (v549_acc[3]);
          tensorforge::VectorT<float, 4> v554_acc{};
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v540_tp, v51_data, v554_acc, 3, 0, 0);
          r1[145] = (v559_acc[0]);
          r1[147] = (v559_acc[1]);
          r1[149] = (v559_acc[2]);
          r1[151] = (v559_acc[3]);
          float v568_tp{};
          float v569_tp{};
          float v570_tp{};
          float v571_tp{};
          tensorforge::transpose4x4b32(v568_tp, v569_tp, v570_tp, v571_tp, v481_data, v481_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v572_acc{};
          tensorforge::VectorT<float, 4> v577_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v568_tp, v41_data, v572_acc, 3, 0, 0);
          r1[152] = (v577_acc[0]);
          r1[154] = (v577_acc[1]);
          tensorforge::VectorT<float, 4> v580_acc{};
          tensorforge::VectorT<float, 4> v585_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v568_tp, v51_data, v580_acc, 3, 0, 0);
          r1[153] = (v585_acc[0]);
          r1[155] = (v585_acc[1]);
          float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          if (v18_lead >= 20) {
            float v593_data = r1[24];
            float v594_data = r2[0];
            r2[0] = (v594_data + v593_data);
            float v596_data = r1[50];
            float v597_data = r2[2];
            r2[2] = (v597_data + v596_data);
            float v599_data = r1[76];
            float v600_data = r2[4];
            r2[4] = (v600_data + v599_data);
            float v602_data = r1[102];
            float v603_data = r2[6];
            r2[6] = (v603_data + v602_data);
            float v605_data = r1[128];
            float v606_data = r2[8];
            r2[8] = (v606_data + v605_data);
            float v608_data = r1[154];
            float v609_data = r2[10];
            r2[10] = (v609_data + v608_data);
          }
          if (v18_lead < 3) {
            float v612_data = r1[25];
            float v613_data = r2[1];
            r2[1] = (v613_data + v612_data);
            float v615_data = r1[51];
            float v616_data = r2[3];
            r2[3] = (v616_data + v615_data);
            float v618_data = r1[77];
            float v619_data = r2[5];
            r2[5] = (v619_data + v618_data);
            float v621_data = r1[103];
            float v622_data = r2[7];
            r2[7] = (v622_data + v621_data);
            float v624_data = r1[129];
            float v625_data = r2[9];
            r2[9] = (v625_data + v624_data);
            float v627_data = r1[155];
            float v628_data = r2[11];
            r2[11] = (v628_data + v627_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v18_lead >= 20) {
            #pragma unroll
            for (int32_t v634_i1 = 0; v634_i1 < 1; ++v634_i1) {
              int32_t v636_a = v634_i1 * 2;
              int32_t v649_a = v18_lead + ((v634_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v635_i2 = 0; v635_i2 < 6; ++v635_i2) {
                float v640_data = r2[(v636_a + (v635_i2 * 2))];
                int32_t v650_a = v649_a + (v635_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v650_a], v640_data);
              }
            }
          }
          if (v18_lead < 3) {
            int32_t v663_lead = v18_lead + 32_i32;
            #pragma unroll
            for (int32_t v652_i1 = 0; v652_i1 < 1; ++v652_i1) {
              int32_t v656_a = 1 + (v652_i1 * 2);
              int32_t v667_a = v663_lead + ((v652_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v653_i2 = 0; v653_i2 < 6; ++v653_i2) {
                float v658_data = r2[(v656_a + (v653_i2 * 2))];
                int32_t v668_a = v667_a + (v653_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v668_a], v658_data);
              }
            }
          }
        }
      }
    }
  }
}

