// === base name ===
kernel_671a350836

// === header ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_671a350836, block.x * block.y * block.z, 64 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_671a350836), hipFuncAttributeMaxDynamicSharedMemorySize, 64 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_671a350836, grid, block, 64 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 64];
      float* tempShrMem = &localShrMem0[0];
      const float *const __restrict__ ptr_glb_m1 = &m1[0];
      float* __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 6) {
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      }
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v14_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v15_i0 = 0; v15_i0 < 2; ++v15_i0) {
            int32_t v21_lead = v14_lead + (v15_i0 * 32);
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 13; ++v16_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m0[(v21_lead + (v16_i1 * 64))]);
              r0[(v15_i0 + (v16_i1 * 2))] = v24_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v28_data = glb_m1[0];
          float v32_tp{};
          float v33_tp{};
          float v34_tp{};
          float v35_tp{};
          tensorforge::transpose4x4b32(v32_tp, v33_tp, v34_tp, v35_tp, v28_data, v28_data, v28_data, v28_data);
          tensorforge::VectorT<float, 4> v36_acc{};
          float v37_data = r0[0];
          tensorforge::VectorT<float, 4> v41_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v37_data, v36_acc, 3, 0, 0);
          r1[0] = (v41_acc[0]);
          r1[2] = (v41_acc[1]);
          r1[4] = (v41_acc[2]);
          r1[6] = (v41_acc[3]);
          tensorforge::VectorT<float, 4> v46_acc{};
          float v47_data = r0[1];
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v47_data, v46_acc, 3, 0, 0);
          r1[1] = (v51_acc[0]);
          r1[3] = (v51_acc[1]);
          r1[5] = (v51_acc[2]);
          r1[7] = (v51_acc[3]);
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          float v63_tp{};
          tensorforge::transpose4x4b32(v60_tp, v61_tp, v62_tp, v63_tp, v28_data, v28_data, v28_data, v28_data);
          tensorforge::VectorT<float, 4> v64_acc{};
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v37_data, v64_acc, 3, 0, 0);
          r1[8] = (v69_acc[0]);
          r1[10] = (v69_acc[1]);
          r1[12] = (v69_acc[2]);
          r1[14] = (v69_acc[3]);
          tensorforge::VectorT<float, 4> v74_acc{};
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v47_data, v74_acc, 3, 0, 0);
          r1[9] = (v79_acc[0]);
          r1[11] = (v79_acc[1]);
          r1[13] = (v79_acc[2]);
          r1[15] = (v79_acc[3]);
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          tensorforge::transpose4x4b32(v88_tp, v89_tp, v90_tp, v91_tp, v28_data, v28_data, v28_data, v28_data);
          tensorforge::VectorT<float, 4> v92_acc{};
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v37_data, v92_acc, 3, 0, 0);
          r1[16] = (v97_acc[0]);
          r1[18] = (v97_acc[1]);
          r1[20] = (v97_acc[2]);
          r1[22] = (v97_acc[3]);
          tensorforge::VectorT<float, 4> v102_acc{};
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v47_data, v102_acc, 3, 0, 0);
          r1[17] = (v107_acc[0]);
          r1[19] = (v107_acc[1]);
          r1[21] = (v107_acc[2]);
          r1[23] = (v107_acc[3]);
          float v113_data = glb_m1[1];
          float v116_tp{};
          float v117_tp{};
          float v118_tp{};
          float v119_tp{};
          tensorforge::transpose4x4b32(v116_tp, v117_tp, v118_tp, v119_tp, v28_data, v113_data, v113_data, v113_data);
          tensorforge::VectorT<float, 4> v120_acc{};
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v37_data, v120_acc, 3, 0, 0);
          r1[24] = (v125_acc[0]);
          r1[26] = (v125_acc[1]);
          r1[28] = (v125_acc[2]);
          r1[30] = (v125_acc[3]);
          tensorforge::VectorT<float, 4> v130_acc{};
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v47_data, v130_acc, 3, 0, 0);
          r1[25] = (v135_acc[0]);
          r1[27] = (v135_acc[1]);
          r1[29] = (v135_acc[2]);
          r1[31] = (v135_acc[3]);
          float v144_tp{};
          float v145_tp{};
          float v146_tp{};
          float v147_tp{};
          tensorforge::transpose4x4b32(v144_tp, v145_tp, v146_tp, v147_tp, v113_data, v113_data, v113_data, v113_data);
          tensorforge::VectorT<float, 4> v148_acc{};
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v37_data, v148_acc, 3, 0, 0);
          r1[32] = (v153_acc[0]);
          r1[34] = (v153_acc[1]);
          r1[36] = (v153_acc[2]);
          r1[38] = (v153_acc[3]);
          tensorforge::VectorT<float, 4> v158_acc{};
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v144_tp, v47_data, v158_acc, 3, 0, 0);
          r1[33] = (v163_acc[0]);
          r1[35] = (v163_acc[1]);
          r1[37] = (v163_acc[2]);
          r1[39] = (v163_acc[3]);
          float v172_tp{};
          float v173_tp{};
          float v174_tp{};
          float v175_tp{};
          tensorforge::transpose4x4b32(v172_tp, v173_tp, v174_tp, v175_tp, v113_data, v113_data, v113_data, v113_data);
          tensorforge::VectorT<float, 4> v176_acc{};
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v37_data, v176_acc, 3, 0, 0);
          r1[40] = (v181_acc[0]);
          r1[42] = (v181_acc[1]);
          r1[44] = (v181_acc[2]);
          r1[46] = (v181_acc[3]);
          tensorforge::VectorT<float, 4> v186_acc{};
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v47_data, v186_acc, 3, 0, 0);
          r1[41] = (v191_acc[0]);
          r1[43] = (v191_acc[1]);
          r1[45] = (v191_acc[2]);
          r1[47] = (v191_acc[3]);
          float v198_data = glb_m1[2];
          float v200_tp{};
          float v201_tp{};
          float v202_tp{};
          float v203_tp{};
          tensorforge::transpose4x4b32(v200_tp, v201_tp, v202_tp, v203_tp, v113_data, v113_data, v198_data, v198_data);
          tensorforge::VectorT<float, 4> v204_acc{};
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v37_data, v204_acc, 3, 0, 0);
          r1[48] = (v209_acc[0]);
          r1[50] = (v209_acc[1]);
          r1[52] = (v209_acc[2]);
          r1[54] = (v209_acc[3]);
          tensorforge::VectorT<float, 4> v214_acc{};
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v47_data, v214_acc, 3, 0, 0);
          r1[49] = (v219_acc[0]);
          r1[51] = (v219_acc[1]);
          r1[53] = (v219_acc[2]);
          r1[55] = (v219_acc[3]);
          float v228_tp{};
          float v229_tp{};
          float v230_tp{};
          float v231_tp{};
          tensorforge::transpose4x4b32(v228_tp, v229_tp, v230_tp, v231_tp, v198_data, v198_data, v198_data, v198_data);
          tensorforge::VectorT<float, 4> v232_acc{};
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v37_data, v232_acc, 3, 0, 0);
          r1[56] = (v237_acc[0]);
          r1[58] = (v237_acc[1]);
          r1[60] = (v237_acc[2]);
          r1[62] = (v237_acc[3]);
          tensorforge::VectorT<float, 4> v242_acc{};
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v228_tp, v47_data, v242_acc, 3, 0, 0);
          r1[57] = (v247_acc[0]);
          r1[59] = (v247_acc[1]);
          r1[61] = (v247_acc[2]);
          r1[63] = (v247_acc[3]);
          float v256_tp{};
          float v257_tp{};
          float v258_tp{};
          float v259_tp{};
          tensorforge::transpose4x4b32(v256_tp, v257_tp, v258_tp, v259_tp, v198_data, v198_data, v198_data, v198_data);
          tensorforge::VectorT<float, 4> v260_acc{};
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v37_data, v260_acc, 3, 0, 0);
          r1[64] = (v265_acc[0]);
          r1[66] = (v265_acc[1]);
          r1[68] = (v265_acc[2]);
          r1[70] = (v265_acc[3]);
          tensorforge::VectorT<float, 4> v270_acc{};
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v47_data, v270_acc, 3, 0, 0);
          r1[65] = (v275_acc[0]);
          r1[67] = (v275_acc[1]);
          r1[69] = (v275_acc[2]);
          r1[71] = (v275_acc[3]);
          float v283_data = glb_m1[3];
          float v284_tp{};
          float v285_tp{};
          float v286_tp{};
          float v287_tp{};
          tensorforge::transpose4x4b32(v284_tp, v285_tp, v286_tp, v287_tp, v198_data, v198_data, v198_data, v283_data);
          tensorforge::VectorT<float, 4> v288_acc{};
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v37_data, v288_acc, 3, 0, 0);
          r1[72] = (v293_acc[0]);
          r1[74] = (v293_acc[1]);
          r1[76] = (v293_acc[2]);
          r1[78] = (v293_acc[3]);
          tensorforge::VectorT<float, 4> v298_acc{};
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v284_tp, v47_data, v298_acc, 3, 0, 0);
          r1[73] = (v303_acc[0]);
          r1[75] = (v303_acc[1]);
          r1[77] = (v303_acc[2]);
          r1[79] = (v303_acc[3]);
          float v312_tp{};
          float v313_tp{};
          float v314_tp{};
          float v315_tp{};
          tensorforge::transpose4x4b32(v312_tp, v313_tp, v314_tp, v315_tp, v283_data, v283_data, v283_data, v283_data);
          tensorforge::VectorT<float, 4> v316_acc{};
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v37_data, v316_acc, 3, 0, 0);
          r1[80] = (v321_acc[0]);
          r1[82] = (v321_acc[1]);
          r1[84] = (v321_acc[2]);
          r1[86] = (v321_acc[3]);
          tensorforge::VectorT<float, 4> v326_acc{};
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v47_data, v326_acc, 3, 0, 0);
          r1[81] = (v331_acc[0]);
          r1[83] = (v331_acc[1]);
          r1[85] = (v331_acc[2]);
          r1[87] = (v331_acc[3]);
          float v340_tp{};
          float v341_tp{};
          float v342_tp{};
          float v343_tp{};
          tensorforge::transpose4x4b32(v340_tp, v341_tp, v342_tp, v343_tp, v283_data, v283_data, v283_data, v283_data);
          tensorforge::VectorT<float, 4> v344_acc{};
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v37_data, v344_acc, 3, 0, 0);
          r1[88] = (v349_acc[0]);
          r1[90] = (v349_acc[1]);
          r1[92] = (v349_acc[2]);
          r1[94] = (v349_acc[3]);
          tensorforge::VectorT<float, 4> v354_acc{};
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v47_data, v354_acc, 3, 0, 0);
          r1[89] = (v359_acc[0]);
          r1[91] = (v359_acc[1]);
          r1[93] = (v359_acc[2]);
          r1[95] = (v359_acc[3]);
          float v368_tp{};
          float v369_tp{};
          float v370_tp{};
          float v371_tp{};
          tensorforge::transpose4x4b32(v368_tp, v369_tp, v370_tp, v371_tp, v283_data, v283_data, v283_data, v283_data);
          tensorforge::VectorT<float, 4> v372_acc{};
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v37_data, v372_acc, 3, 0, 0);
          r1[96] = (v377_acc[0]);
          r1[98] = (v377_acc[1]);
          r1[100] = (v377_acc[2]);
          r1[102] = (v377_acc[3]);
          tensorforge::VectorT<float, 4> v382_acc{};
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v47_data, v382_acc, 3, 0, 0);
          r1[97] = (v387_acc[0]);
          r1[99] = (v387_acc[1]);
          r1[101] = (v387_acc[2]);
          r1[103] = (v387_acc[3]);
          float v392_data = glb_m1[4];
          float v396_tp{};
          float v397_tp{};
          float v398_tp{};
          float v399_tp{};
          tensorforge::transpose4x4b32(v396_tp, v397_tp, v398_tp, v399_tp, v392_data, v392_data, v392_data, v392_data);
          tensorforge::VectorT<float, 4> v400_acc{};
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v37_data, v400_acc, 3, 0, 0);
          r1[104] = (v405_acc[0]);
          r1[106] = (v405_acc[1]);
          r1[108] = (v405_acc[2]);
          r1[110] = (v405_acc[3]);
          tensorforge::VectorT<float, 4> v410_acc{};
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v47_data, v410_acc, 3, 0, 0);
          r1[105] = (v415_acc[0]);
          r1[107] = (v415_acc[1]);
          r1[109] = (v415_acc[2]);
          r1[111] = (v415_acc[3]);
          float v424_tp{};
          float v425_tp{};
          float v426_tp{};
          float v427_tp{};
          tensorforge::transpose4x4b32(v424_tp, v425_tp, v426_tp, v427_tp, v392_data, v392_data, v392_data, v392_data);
          tensorforge::VectorT<float, 4> v428_acc{};
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v37_data, v428_acc, 3, 0, 0);
          r1[112] = (v433_acc[0]);
          r1[114] = (v433_acc[1]);
          r1[116] = (v433_acc[2]);
          r1[118] = (v433_acc[3]);
          tensorforge::VectorT<float, 4> v438_acc{};
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v47_data, v438_acc, 3, 0, 0);
          r1[113] = (v443_acc[0]);
          r1[115] = (v443_acc[1]);
          r1[117] = (v443_acc[2]);
          r1[119] = (v443_acc[3]);
          float v452_tp{};
          float v453_tp{};
          float v454_tp{};
          float v455_tp{};
          tensorforge::transpose4x4b32(v452_tp, v453_tp, v454_tp, v455_tp, v392_data, v392_data, v392_data, v392_data);
          tensorforge::VectorT<float, 4> v456_acc{};
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v37_data, v456_acc, 3, 0, 0);
          r1[120] = (v461_acc[0]);
          r1[122] = (v461_acc[1]);
          r1[124] = (v461_acc[2]);
          r1[126] = (v461_acc[3]);
          tensorforge::VectorT<float, 4> v466_acc{};
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v452_tp, v47_data, v466_acc, 3, 0, 0);
          r1[121] = (v471_acc[0]);
          r1[123] = (v471_acc[1]);
          r1[125] = (v471_acc[2]);
          r1[127] = (v471_acc[3]);
          float v477_data = glb_m1[5];
          float v480_tp{};
          float v481_tp{};
          float v482_tp{};
          float v483_tp{};
          tensorforge::transpose4x4b32(v480_tp, v481_tp, v482_tp, v483_tp, v392_data, v477_data, v477_data, v477_data);
          tensorforge::VectorT<float, 4> v484_acc{};
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v37_data, v484_acc, 3, 0, 0);
          r1[128] = (v489_acc[0]);
          r1[130] = (v489_acc[1]);
          r1[132] = (v489_acc[2]);
          r1[134] = (v489_acc[3]);
          tensorforge::VectorT<float, 4> v494_acc{};
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v47_data, v494_acc, 3, 0, 0);
          r1[129] = (v499_acc[0]);
          r1[131] = (v499_acc[1]);
          r1[133] = (v499_acc[2]);
          r1[135] = (v499_acc[3]);
          float v508_tp{};
          float v509_tp{};
          float v510_tp{};
          float v511_tp{};
          tensorforge::transpose4x4b32(v508_tp, v509_tp, v510_tp, v511_tp, v477_data, v477_data, v477_data, v477_data);
          tensorforge::VectorT<float, 4> v512_acc{};
          tensorforge::VectorT<float, 4> v517_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v37_data, v512_acc, 3, 0, 0);
          r1[136] = (v517_acc[0]);
          r1[138] = (v517_acc[1]);
          r1[140] = (v517_acc[2]);
          r1[142] = (v517_acc[3]);
          tensorforge::VectorT<float, 4> v522_acc{};
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v508_tp, v47_data, v522_acc, 3, 0, 0);
          r1[137] = (v527_acc[0]);
          r1[139] = (v527_acc[1]);
          r1[141] = (v527_acc[2]);
          r1[143] = (v527_acc[3]);
          float v536_tp{};
          float v537_tp{};
          float v538_tp{};
          float v539_tp{};
          tensorforge::transpose4x4b32(v536_tp, v537_tp, v538_tp, v539_tp, v477_data, v477_data, v477_data, v477_data);
          tensorforge::VectorT<float, 4> v540_acc{};
          tensorforge::VectorT<float, 4> v545_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v536_tp, v37_data, v540_acc, 3, 0, 0);
          r1[144] = (v545_acc[0]);
          r1[146] = (v545_acc[1]);
          r1[148] = (v545_acc[2]);
          r1[150] = (v545_acc[3]);
          tensorforge::VectorT<float, 4> v550_acc{};
          tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v536_tp, v47_data, v550_acc, 3, 0, 0);
          r1[145] = (v555_acc[0]);
          r1[147] = (v555_acc[1]);
          r1[149] = (v555_acc[2]);
          r1[151] = (v555_acc[3]);
          float v564_tp{};
          float v565_tp{};
          float v566_tp{};
          float v567_tp{};
          tensorforge::transpose4x4b32(v564_tp, v565_tp, v566_tp, v567_tp, v477_data, v477_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v568_acc{};
          tensorforge::VectorT<float, 4> v573_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v564_tp, v37_data, v568_acc, 3, 0, 0);
          r1[152] = (v573_acc[0]);
          r1[154] = (v573_acc[1]);
          tensorforge::VectorT<float, 4> v576_acc{};
          tensorforge::VectorT<float, 4> v581_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v564_tp, v47_data, v576_acc, 3, 0, 0);
          r1[153] = (v581_acc[0]);
          r1[155] = (v581_acc[1]);
          float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          if (v14_lead >= 20) {
            float v589_data = r1[24];
            float v590_data = r2[0];
            r2[0] = (v590_data + v589_data);
            float v592_data = r1[50];
            float v593_data = r2[2];
            r2[2] = (v593_data + v592_data);
            float v595_data = r1[76];
            float v596_data = r2[4];
            r2[4] = (v596_data + v595_data);
            float v598_data = r1[102];
            float v599_data = r2[6];
            r2[6] = (v599_data + v598_data);
            float v601_data = r1[128];
            float v602_data = r2[8];
            r2[8] = (v602_data + v601_data);
            float v604_data = r1[154];
            float v605_data = r2[10];
            r2[10] = (v605_data + v604_data);
          }
          if (v14_lead < 3) {
            float v608_data = r1[25];
            float v609_data = r2[1];
            r2[1] = (v609_data + v608_data);
            float v611_data = r1[51];
            float v612_data = r2[3];
            r2[3] = (v612_data + v611_data);
            float v614_data = r1[77];
            float v615_data = r2[5];
            r2[5] = (v615_data + v614_data);
            float v617_data = r1[103];
            float v618_data = r2[7];
            r2[7] = (v618_data + v617_data);
            float v620_data = r1[129];
            float v621_data = r2[9];
            r2[9] = (v621_data + v620_data);
            float v623_data = r1[155];
            float v624_data = r2[11];
            r2[11] = (v624_data + v623_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v14_lead >= 20) {
            #pragma unroll
            for (int32_t v630_i1 = 0; v630_i1 < 1; ++v630_i1) {
              int32_t v632_a = v630_i1 * 2;
              int32_t v645_a = v14_lead + ((v630_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v631_i2 = 0; v631_i2 < 6; ++v631_i2) {
                float v636_data = r2[(v632_a + (v631_i2 * 2))];
                int32_t v646_a = v645_a + (v631_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v646_a], v636_data);
              }
            }
          }
          if (v14_lead < 3) {
            int32_t v659_lead = v14_lead + 32_i32;
            #pragma unroll
            for (int32_t v648_i1 = 0; v648_i1 < 1; ++v648_i1) {
              int32_t v652_a = 1 + (v648_i1 * 2);
              int32_t v663_a = v659_lead + ((v648_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v649_i2 = 0; v649_i2 < 6; ++v649_i2) {
                float v654_data = r2[(v652_a + (v649_i2 * 2))];
                int32_t v664_a = v663_a + (v649_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v664_a], v654_data);
              }
            }
          }
        }
      }
    }
  }
}

