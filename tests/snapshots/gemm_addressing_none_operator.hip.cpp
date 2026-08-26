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
          float v2_data = r0[0];
          float v3_data = r0[1];
          float v4_data = r0[2];
          float v5_data = r0[3];
          float v6_tp{};
          float v7_tp{};
          float v8_tp{};
          float v9_tp{};
          tensorforge::transpose4x4b32(v6_tp, v7_tp, v8_tp, v9_tp, v2_data, v3_data, v4_data, v5_data);
          tensorforge::VectorT<float, 4> v10_acc{};
          int32_t v13_lane = threadIdx.x % 16;
          int32_t v16_a = v13_lane + 0;
          float v23_data = glb_m1[v13_lane];
          int32_t v29_a = v13_lane + 16;
          float v36_data = glb_m1[(v13_lane + 16)];
          int32_t v42_a = v13_lane + 32;
          float v49_data = glb_m1[(v13_lane + 32)];
          int32_t v55_a = v13_lane + 48;
          float v62_data = glb_m1[(v13_lane + 48)];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v23_data, v10_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v36_data, v63_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v8_tp, v49_data, v64_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v9_tp, v62_data, v65_acc, 2, 0, 0);
          int32_t v72_a = v13_lane + 64;
          float v79_data = glb_m1[(v13_lane + 64)];
          int32_t v85_a = v13_lane + 80;
          float v92_data = glb_m1[(v13_lane + 80)];
          int32_t v98_a = v13_lane + 96;
          float v105_data = glb_m1[(v13_lane + 96)];
          int32_t v111_a = v13_lane + 112;
          float v118_data = glb_m1[(v13_lane + 112)];
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v79_data, v66_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v92_data, v119_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v8_tp, v105_data, v120_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v9_tp, v118_data, v121_acc, 2, 1, 0);
          int32_t v128_a = v13_lane + 128;
          float v135_data = glb_m1[(v13_lane + 128)];
          int32_t v141_a = v13_lane + 144;
          float v148_data = glb_m1[(v13_lane + 144)];
          int32_t v154_a = v13_lane + 160;
          float v161_data = glb_m1[(v13_lane + 160)];
          int32_t v167_a = v13_lane + 176;
          float v174_data = glb_m1[(v13_lane + 176)];
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v135_data, v122_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v148_data, v175_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v8_tp, v161_data, v176_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v9_tp, v174_data, v177_acc, 2, 2, 0);
          int32_t v184_a = v13_lane + 192;
          float v191_data = glb_m1[(v13_lane + 192)];
          int32_t v197_a = v13_lane + 208;
          float v204_data = glb_m1[(v13_lane + 208)];
          int32_t v210_a = v13_lane + 224;
          float v217_data = glb_m1[(v13_lane + 224)];
          int32_t v223_a = v13_lane + 240;
          float v230_data = glb_m1[(v13_lane + 240)];
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v191_data, v178_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v204_data, v231_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v8_tp, v217_data, v232_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v9_tp, v230_data, v233_acc, 2, 3, 0);
          ir1[0] = (v234_acc[0]);
          ir1[1] = (v234_acc[1]);
          ir1[2] = (v234_acc[2]);
          ir1[3] = (v234_acc[3]);
          float v239_data = r0[4];
          float v240_data = r0[5];
          float v241_data = r0[6];
          float v242_data = r0[7];
          float v243_tp{};
          float v244_tp{};
          float v245_tp{};
          float v246_tp{};
          tensorforge::transpose4x4b32(v243_tp, v244_tp, v245_tp, v246_tp, v239_data, v240_data, v241_data, v242_data);
          tensorforge::VectorT<float, 4> v247_acc{};
          int32_t v253_a = v13_lane + 0;
          float v260_data = glb_m1[v13_lane];
          int32_t v266_a = v13_lane + 16;
          float v273_data = glb_m1[(v13_lane + 16)];
          int32_t v279_a = v13_lane + 32;
          float v286_data = glb_m1[(v13_lane + 32)];
          int32_t v292_a = v13_lane + 48;
          float v299_data = glb_m1[(v13_lane + 48)];
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v260_data, v247_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v273_data, v300_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v286_data, v301_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v299_data, v302_acc, 2, 0, 0);
          int32_t v309_a = v13_lane + 64;
          float v316_data = glb_m1[(v13_lane + 64)];
          int32_t v322_a = v13_lane + 80;
          float v329_data = glb_m1[(v13_lane + 80)];
          int32_t v335_a = v13_lane + 96;
          float v342_data = glb_m1[(v13_lane + 96)];
          int32_t v348_a = v13_lane + 112;
          float v355_data = glb_m1[(v13_lane + 112)];
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v316_data, v303_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v329_data, v356_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v342_data, v357_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v355_data, v358_acc, 2, 1, 0);
          int32_t v365_a = v13_lane + 128;
          float v372_data = glb_m1[(v13_lane + 128)];
          int32_t v378_a = v13_lane + 144;
          float v385_data = glb_m1[(v13_lane + 144)];
          int32_t v391_a = v13_lane + 160;
          float v398_data = glb_m1[(v13_lane + 160)];
          int32_t v404_a = v13_lane + 176;
          float v411_data = glb_m1[(v13_lane + 176)];
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v372_data, v359_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v385_data, v412_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v398_data, v413_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v411_data, v414_acc, 2, 2, 0);
          int32_t v421_a = v13_lane + 192;
          float v428_data = glb_m1[(v13_lane + 192)];
          int32_t v434_a = v13_lane + 208;
          float v441_data = glb_m1[(v13_lane + 208)];
          int32_t v447_a = v13_lane + 224;
          float v454_data = glb_m1[(v13_lane + 224)];
          int32_t v460_a = v13_lane + 240;
          float v467_data = glb_m1[(v13_lane + 240)];
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v428_data, v415_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v441_data, v468_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v454_data, v469_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v467_data, v470_acc, 2, 3, 0);
          ir1[4] = (v471_acc[0]);
          ir1[5] = (v471_acc[1]);
          ir1[6] = (v471_acc[2]);
          ir1[7] = (v471_acc[3]);
          float v476_data = r0[8];
          float v477_data = r0[9];
          float v478_data = r0[10];
          float v479_data = r0[11];
          float v480_tp{};
          float v481_tp{};
          float v482_tp{};
          float v483_tp{};
          tensorforge::transpose4x4b32(v480_tp, v481_tp, v482_tp, v483_tp, v476_data, v477_data, v478_data, v479_data);
          tensorforge::VectorT<float, 4> v484_acc{};
          int32_t v490_a = v13_lane + 0;
          float v497_data = glb_m1[v13_lane];
          int32_t v503_a = v13_lane + 16;
          float v510_data = glb_m1[(v13_lane + 16)];
          int32_t v516_a = v13_lane + 32;
          float v523_data = glb_m1[(v13_lane + 32)];
          int32_t v529_a = v13_lane + 48;
          float v536_data = glb_m1[(v13_lane + 48)];
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v497_data, v484_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v510_data, v537_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v523_data, v538_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v536_data, v539_acc, 2, 0, 0);
          int32_t v546_a = v13_lane + 64;
          float v553_data = glb_m1[(v13_lane + 64)];
          int32_t v559_a = v13_lane + 80;
          float v566_data = glb_m1[(v13_lane + 80)];
          int32_t v572_a = v13_lane + 96;
          float v579_data = glb_m1[(v13_lane + 96)];
          int32_t v585_a = v13_lane + 112;
          float v592_data = glb_m1[(v13_lane + 112)];
          tensorforge::VectorT<float, 4> v593_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v553_data, v540_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v566_data, v593_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v595_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v579_data, v594_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v596_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v592_data, v595_acc, 2, 1, 0);
          int32_t v602_a = v13_lane + 128;
          float v609_data = glb_m1[(v13_lane + 128)];
          int32_t v615_a = v13_lane + 144;
          float v622_data = glb_m1[(v13_lane + 144)];
          int32_t v628_a = v13_lane + 160;
          float v635_data = glb_m1[(v13_lane + 160)];
          int32_t v641_a = v13_lane + 176;
          float v648_data = glb_m1[(v13_lane + 176)];
          tensorforge::VectorT<float, 4> v649_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v609_data, v596_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v622_data, v649_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v651_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v635_data, v650_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v652_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v648_data, v651_acc, 2, 2, 0);
          int32_t v658_a = v13_lane + 192;
          float v665_data = glb_m1[(v13_lane + 192)];
          int32_t v671_a = v13_lane + 208;
          float v678_data = glb_m1[(v13_lane + 208)];
          int32_t v684_a = v13_lane + 224;
          float v691_data = glb_m1[(v13_lane + 224)];
          int32_t v697_a = v13_lane + 240;
          float v704_data = glb_m1[(v13_lane + 240)];
          tensorforge::VectorT<float, 4> v705_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v665_data, v652_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v678_data, v705_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v707_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v482_tp, v691_data, v706_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v708_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v483_tp, v704_data, v707_acc, 2, 3, 0);
          ir1[8] = (v708_acc[0]);
          ir1[9] = (v708_acc[1]);
          ir1[10] = (v708_acc[2]);
          ir1[11] = (v708_acc[3]);
          float v713_data = r0[12];
          float v714_data = r0[13];
          float v715_data = r0[14];
          float v716_data = r0[15];
          float v717_tp{};
          float v718_tp{};
          float v719_tp{};
          float v720_tp{};
          tensorforge::transpose4x4b32(v717_tp, v718_tp, v719_tp, v720_tp, v713_data, v714_data, v715_data, v716_data);
          tensorforge::VectorT<float, 4> v721_acc{};
          int32_t v727_a = v13_lane + 0;
          float v734_data = glb_m1[v13_lane];
          int32_t v740_a = v13_lane + 16;
          float v747_data = glb_m1[(v13_lane + 16)];
          int32_t v753_a = v13_lane + 32;
          float v760_data = glb_m1[(v13_lane + 32)];
          int32_t v766_a = v13_lane + 48;
          float v773_data = glb_m1[(v13_lane + 48)];
          tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v717_tp, v734_data, v721_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v718_tp, v747_data, v774_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v776_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v719_tp, v760_data, v775_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v777_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v720_tp, v773_data, v776_acc, 2, 0, 0);
          int32_t v783_a = v13_lane + 64;
          float v790_data = glb_m1[(v13_lane + 64)];
          int32_t v796_a = v13_lane + 80;
          float v803_data = glb_m1[(v13_lane + 80)];
          int32_t v809_a = v13_lane + 96;
          float v816_data = glb_m1[(v13_lane + 96)];
          int32_t v822_a = v13_lane + 112;
          float v829_data = glb_m1[(v13_lane + 112)];
          tensorforge::VectorT<float, 4> v830_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v717_tp, v790_data, v777_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v831_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v718_tp, v803_data, v830_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v832_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v719_tp, v816_data, v831_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v833_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v720_tp, v829_data, v832_acc, 2, 1, 0);
          int32_t v839_a = v13_lane + 128;
          float v846_data = glb_m1[(v13_lane + 128)];
          int32_t v852_a = v13_lane + 144;
          float v859_data = glb_m1[(v13_lane + 144)];
          int32_t v865_a = v13_lane + 160;
          float v872_data = glb_m1[(v13_lane + 160)];
          int32_t v878_a = v13_lane + 176;
          float v885_data = glb_m1[(v13_lane + 176)];
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v717_tp, v846_data, v833_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v887_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v718_tp, v859_data, v886_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v888_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v719_tp, v872_data, v887_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v889_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v720_tp, v885_data, v888_acc, 2, 2, 0);
          int32_t v895_a = v13_lane + 192;
          float v902_data = glb_m1[(v13_lane + 192)];
          int32_t v908_a = v13_lane + 208;
          float v915_data = glb_m1[(v13_lane + 208)];
          int32_t v921_a = v13_lane + 224;
          float v928_data = glb_m1[(v13_lane + 224)];
          int32_t v934_a = v13_lane + 240;
          float v941_data = glb_m1[(v13_lane + 240)];
          tensorforge::VectorT<float, 4> v942_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v717_tp, v902_data, v889_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v943_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v718_tp, v915_data, v942_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v944_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v719_tp, v928_data, v943_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v945_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v720_tp, v941_data, v944_acc, 2, 3, 0);
          ir1[12] = (v945_acc[0]);
          ir1[13] = (v945_acc[1]);
          ir1[14] = (v945_acc[2]);
          ir1[15] = (v945_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v953_i0 = 0; v953_i0 < 1; ++v953_i0) {
            int32_t v962_lead = v13_lane + (v953_i0 * 16);
            #pragma unroll
            for (int32_t v954_i1 = 0; v954_i1 < 16; ++v954_i1) {
              int32_t v955_a = v953_i0 + v954_i1;
              float v957_data = r1[(v953_i0 + v954_i1)];
              int32_t v964_a = v962_lead + (v954_i1 * 16);
              glb_m0[v964_a] = v957_data;
            }
          }
          ;
        }
      }
    }
  }
}

