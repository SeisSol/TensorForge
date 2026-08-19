// === base name ===
kernel_151d4e8604

// === header ===
void launcher_kernel_151d4e8604(float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_151d4e8604(float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_151d4e8604, block.x * block.y * block.z, 512 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_151d4e8604), hipFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_151d4e8604, grid, block, 512 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_151d4e8604(float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} none
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} none({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      float* tempShrMem = &localShrMem0[0];
      const float *const __restrict__ ptr_glb_m1 = &m1[0];
      float* __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          {
            // r0 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r0[0] = v0;
            float v16 = glb_m2[16 + threadIdx.x * 1];
            r0[1] = v16;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r0[2] = v32;
            float v48 = glb_m2[48 + threadIdx.x * 1];
            r0[3] = v48;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r0[4] = v64;
            float v80 = glb_m2[80 + threadIdx.x * 1];
            r0[5] = v80;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r0[6] = v96;
            float v112 = glb_m2[112 + threadIdx.x * 1];
            r0[7] = v112;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r0[8] = v128;
            float v144 = glb_m2[144 + threadIdx.x * 1];
            r0[9] = v144;
            float v160 = glb_m2[160 + threadIdx.x * 1];
            r0[10] = v160;
            float v176 = glb_m2[176 + threadIdx.x * 1];
            r0[11] = v176;
            float v192 = glb_m2[192 + threadIdx.x * 1];
            r0[12] = v192;
            float v208 = glb_m2[208 + threadIdx.x * 1];
            r0[13] = v208;
            float v224 = glb_m2[224 + threadIdx.x * 1];
            r0[14] = v224;
            float v240 = glb_m2[240 + threadIdx.x * 1];
            r0[15] = v240;
          }
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir1 = r1;
          float v0_data = r0[0];
          float v1_data = r0[1];
          float v2_data = r0[2];
          float v3_data = r0[3];
          float v4_tp{};
          float v5_tp{};
          float v6_tp{};
          float v7_tp{};
          tensorforge::transpose4x4b32(v4_tp, v5_tp, v6_tp, v7_tp, v0_data, v1_data, v2_data, v3_data);
          tensorforge::VectorT<float, 4> v8_acc{};
          int32_t v11_lane = threadIdx.x % 16;
          int32_t v14_a = v11_lane + 0;
          float v15_data = glb_m1[v14_a];
          int32_t v21_a = v11_lane + 16;
          float v22_data = glb_m1[v21_a];
          int32_t v28_a = v11_lane + 32;
          float v29_data = glb_m1[v28_a];
          int32_t v35_a = v11_lane + 48;
          float v36_data = glb_m1[v35_a];
          tensorforge::VectorT<float, 4> v37_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4_tp, v15_data, v8_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v38_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v5_tp, v22_data, v37_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v39_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v29_data, v38_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v40_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v36_data, v39_acc, 2, 0, 0);
          int32_t v46_a = v11_lane + 64;
          float v47_data = glb_m1[v46_a];
          int32_t v53_a = v11_lane + 80;
          float v54_data = glb_m1[v53_a];
          int32_t v60_a = v11_lane + 96;
          float v61_data = glb_m1[v60_a];
          int32_t v67_a = v11_lane + 112;
          float v68_data = glb_m1[v67_a];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4_tp, v47_data, v40_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v5_tp, v54_data, v69_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v61_data, v70_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v68_data, v71_acc, 2, 1, 0);
          int32_t v78_a = v11_lane + 128;
          float v79_data = glb_m1[v78_a];
          int32_t v85_a = v11_lane + 144;
          float v86_data = glb_m1[v85_a];
          int32_t v92_a = v11_lane + 160;
          float v93_data = glb_m1[v92_a];
          int32_t v99_a = v11_lane + 176;
          float v100_data = glb_m1[v99_a];
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4_tp, v79_data, v72_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v5_tp, v86_data, v101_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v93_data, v102_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v100_data, v103_acc, 2, 2, 0);
          int32_t v110_a = v11_lane + 192;
          float v111_data = glb_m1[v110_a];
          int32_t v117_a = v11_lane + 208;
          float v118_data = glb_m1[v117_a];
          int32_t v124_a = v11_lane + 224;
          float v125_data = glb_m1[v124_a];
          int32_t v131_a = v11_lane + 240;
          float v132_data = glb_m1[v131_a];
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4_tp, v111_data, v104_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v5_tp, v118_data, v133_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v125_data, v134_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v132_data, v135_acc, 2, 3, 0);
          ir1[0] = (v136_acc[0]);
          ir1[1] = (v136_acc[1]);
          ir1[2] = (v136_acc[2]);
          ir1[3] = (v136_acc[3]);
          float v141_data = r0[4];
          float v142_data = r0[5];
          float v143_data = r0[6];
          float v144_data = r0[7];
          float v145_tp{};
          float v146_tp{};
          float v147_tp{};
          float v148_tp{};
          tensorforge::transpose4x4b32(v145_tp, v146_tp, v147_tp, v148_tp, v141_data, v142_data, v143_data, v144_data);
          tensorforge::VectorT<float, 4> v149_acc{};
          int32_t v155_a = v11_lane + 0;
          float v156_data = glb_m1[v155_a];
          int32_t v162_a = v11_lane + 16;
          float v163_data = glb_m1[v162_a];
          int32_t v169_a = v11_lane + 32;
          float v170_data = glb_m1[v169_a];
          int32_t v176_a = v11_lane + 48;
          float v177_data = glb_m1[v176_a];
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v156_data, v149_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v163_data, v178_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v170_data, v179_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v177_data, v180_acc, 2, 0, 0);
          int32_t v187_a = v11_lane + 64;
          float v188_data = glb_m1[v187_a];
          int32_t v194_a = v11_lane + 80;
          float v195_data = glb_m1[v194_a];
          int32_t v201_a = v11_lane + 96;
          float v202_data = glb_m1[v201_a];
          int32_t v208_a = v11_lane + 112;
          float v209_data = glb_m1[v208_a];
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v188_data, v181_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v195_data, v210_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v202_data, v211_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v209_data, v212_acc, 2, 1, 0);
          int32_t v219_a = v11_lane + 128;
          float v220_data = glb_m1[v219_a];
          int32_t v226_a = v11_lane + 144;
          float v227_data = glb_m1[v226_a];
          int32_t v233_a = v11_lane + 160;
          float v234_data = glb_m1[v233_a];
          int32_t v240_a = v11_lane + 176;
          float v241_data = glb_m1[v240_a];
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v220_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v227_data, v242_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v234_data, v243_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v241_data, v244_acc, 2, 2, 0);
          int32_t v251_a = v11_lane + 192;
          float v252_data = glb_m1[v251_a];
          int32_t v258_a = v11_lane + 208;
          float v259_data = glb_m1[v258_a];
          int32_t v265_a = v11_lane + 224;
          float v266_data = glb_m1[v265_a];
          int32_t v272_a = v11_lane + 240;
          float v273_data = glb_m1[v272_a];
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v145_tp, v252_data, v245_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v259_data, v274_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v266_data, v275_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v273_data, v276_acc, 2, 3, 0);
          ir1[4] = (v277_acc[0]);
          ir1[5] = (v277_acc[1]);
          ir1[6] = (v277_acc[2]);
          ir1[7] = (v277_acc[3]);
          float v282_data = r0[8];
          float v283_data = r0[9];
          float v284_data = r0[10];
          float v285_data = r0[11];
          float v286_tp{};
          float v287_tp{};
          float v288_tp{};
          float v289_tp{};
          tensorforge::transpose4x4b32(v286_tp, v287_tp, v288_tp, v289_tp, v282_data, v283_data, v284_data, v285_data);
          tensorforge::VectorT<float, 4> v290_acc{};
          int32_t v296_a = v11_lane + 0;
          float v297_data = glb_m1[v296_a];
          int32_t v303_a = v11_lane + 16;
          float v304_data = glb_m1[v303_a];
          int32_t v310_a = v11_lane + 32;
          float v311_data = glb_m1[v310_a];
          int32_t v317_a = v11_lane + 48;
          float v318_data = glb_m1[v317_a];
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v297_data, v290_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v304_data, v319_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v311_data, v320_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v318_data, v321_acc, 2, 0, 0);
          int32_t v328_a = v11_lane + 64;
          float v329_data = glb_m1[v328_a];
          int32_t v335_a = v11_lane + 80;
          float v336_data = glb_m1[v335_a];
          int32_t v342_a = v11_lane + 96;
          float v343_data = glb_m1[v342_a];
          int32_t v349_a = v11_lane + 112;
          float v350_data = glb_m1[v349_a];
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v329_data, v322_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v336_data, v351_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v343_data, v352_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v350_data, v353_acc, 2, 1, 0);
          int32_t v360_a = v11_lane + 128;
          float v361_data = glb_m1[v360_a];
          int32_t v367_a = v11_lane + 144;
          float v368_data = glb_m1[v367_a];
          int32_t v374_a = v11_lane + 160;
          float v375_data = glb_m1[v374_a];
          int32_t v381_a = v11_lane + 176;
          float v382_data = glb_m1[v381_a];
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v361_data, v354_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v368_data, v383_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v375_data, v384_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v382_data, v385_acc, 2, 2, 0);
          int32_t v392_a = v11_lane + 192;
          float v393_data = glb_m1[v392_a];
          int32_t v399_a = v11_lane + 208;
          float v400_data = glb_m1[v399_a];
          int32_t v406_a = v11_lane + 224;
          float v407_data = glb_m1[v406_a];
          int32_t v413_a = v11_lane + 240;
          float v414_data = glb_m1[v413_a];
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v393_data, v386_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v400_data, v415_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v407_data, v416_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v414_data, v417_acc, 2, 3, 0);
          ir1[8] = (v418_acc[0]);
          ir1[9] = (v418_acc[1]);
          ir1[10] = (v418_acc[2]);
          ir1[11] = (v418_acc[3]);
          float v423_data = r0[12];
          float v424_data = r0[13];
          float v425_data = r0[14];
          float v426_data = r0[15];
          float v427_tp{};
          float v428_tp{};
          float v429_tp{};
          float v430_tp{};
          tensorforge::transpose4x4b32(v427_tp, v428_tp, v429_tp, v430_tp, v423_data, v424_data, v425_data, v426_data);
          tensorforge::VectorT<float, 4> v431_acc{};
          int32_t v437_a = v11_lane + 0;
          float v438_data = glb_m1[v437_a];
          int32_t v444_a = v11_lane + 16;
          float v445_data = glb_m1[v444_a];
          int32_t v451_a = v11_lane + 32;
          float v452_data = glb_m1[v451_a];
          int32_t v458_a = v11_lane + 48;
          float v459_data = glb_m1[v458_a];
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v427_tp, v438_data, v431_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v445_data, v460_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v452_data, v461_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v459_data, v462_acc, 2, 0, 0);
          int32_t v469_a = v11_lane + 64;
          float v470_data = glb_m1[v469_a];
          int32_t v476_a = v11_lane + 80;
          float v477_data = glb_m1[v476_a];
          int32_t v483_a = v11_lane + 96;
          float v484_data = glb_m1[v483_a];
          int32_t v490_a = v11_lane + 112;
          float v491_data = glb_m1[v490_a];
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v427_tp, v470_data, v463_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v477_data, v492_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v484_data, v493_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v491_data, v494_acc, 2, 1, 0);
          int32_t v501_a = v11_lane + 128;
          float v502_data = glb_m1[v501_a];
          int32_t v508_a = v11_lane + 144;
          float v509_data = glb_m1[v508_a];
          int32_t v515_a = v11_lane + 160;
          float v516_data = glb_m1[v515_a];
          int32_t v522_a = v11_lane + 176;
          float v523_data = glb_m1[v522_a];
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v427_tp, v502_data, v495_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v509_data, v524_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v516_data, v525_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v523_data, v526_acc, 2, 2, 0);
          int32_t v533_a = v11_lane + 192;
          float v534_data = glb_m1[v533_a];
          int32_t v540_a = v11_lane + 208;
          float v541_data = glb_m1[v540_a];
          int32_t v547_a = v11_lane + 224;
          float v548_data = glb_m1[v547_a];
          int32_t v554_a = v11_lane + 240;
          float v555_data = glb_m1[v554_a];
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v427_tp, v534_data, v527_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v541_data, v556_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v548_data, v557_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v555_data, v558_acc, 2, 3, 0);
          ir1[12] = (v559_acc[0]);
          ir1[13] = (v559_acc[1]);
          ir1[14] = (v559_acc[2]);
          ir1[15] = (v559_acc[3]);
          // glb_m0 = store{r>g}(r1);
          int32_t v566_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v567_i0 = 0; v567_i0 < 1; ++v567_i0) {
            int32_t v575_lead = v566_lead + (v567_i0 * 16);
            #pragma unroll
            for (int32_t v568_i1 = 0; v568_i1 < 16; ++v568_i1) {
              int32_t v569_a = v567_i0 + v568_i1;
              float v570_data = r1[v569_a];
              int32_t v577_a = v575_lead + (v568_i1 * 16);
              glb_m0[v577_a] = v570_data;
            }
          }
          ;
        }
      }
    }
  }
}

