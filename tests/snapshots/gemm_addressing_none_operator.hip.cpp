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
          float v21_data = glb_m1[v11_lane];
          int32_t v27_a = v11_lane + 16;
          float v34_data = glb_m1[(v11_lane + 16)];
          int32_t v40_a = v11_lane + 32;
          float v47_data = glb_m1[(v11_lane + 32)];
          int32_t v53_a = v11_lane + 48;
          float v60_data = glb_m1[(v11_lane + 48)];
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4_tp, v21_data, v8_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v5_tp, v34_data, v61_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v47_data, v62_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v60_data, v63_acc, 2, 0, 0);
          int32_t v70_a = v11_lane + 64;
          float v77_data = glb_m1[(v11_lane + 64)];
          int32_t v83_a = v11_lane + 80;
          float v90_data = glb_m1[(v11_lane + 80)];
          int32_t v96_a = v11_lane + 96;
          float v103_data = glb_m1[(v11_lane + 96)];
          int32_t v109_a = v11_lane + 112;
          float v116_data = glb_m1[(v11_lane + 112)];
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4_tp, v77_data, v64_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v5_tp, v90_data, v117_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v103_data, v118_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v116_data, v119_acc, 2, 1, 0);
          int32_t v126_a = v11_lane + 128;
          float v133_data = glb_m1[(v11_lane + 128)];
          int32_t v139_a = v11_lane + 144;
          float v146_data = glb_m1[(v11_lane + 144)];
          int32_t v152_a = v11_lane + 160;
          float v159_data = glb_m1[(v11_lane + 160)];
          int32_t v165_a = v11_lane + 176;
          float v172_data = glb_m1[(v11_lane + 176)];
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4_tp, v133_data, v120_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v5_tp, v146_data, v173_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v159_data, v174_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v172_data, v175_acc, 2, 2, 0);
          int32_t v182_a = v11_lane + 192;
          float v189_data = glb_m1[(v11_lane + 192)];
          int32_t v195_a = v11_lane + 208;
          float v202_data = glb_m1[(v11_lane + 208)];
          int32_t v208_a = v11_lane + 224;
          float v215_data = glb_m1[(v11_lane + 224)];
          int32_t v221_a = v11_lane + 240;
          float v228_data = glb_m1[(v11_lane + 240)];
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v4_tp, v189_data, v176_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v5_tp, v202_data, v229_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v6_tp, v215_data, v230_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v7_tp, v228_data, v231_acc, 2, 3, 0);
          ir1[0] = (v232_acc[0]);
          ir1[1] = (v232_acc[1]);
          ir1[2] = (v232_acc[2]);
          ir1[3] = (v232_acc[3]);
          float v237_data = r0[4];
          float v238_data = r0[5];
          float v239_data = r0[6];
          float v240_data = r0[7];
          float v241_tp{};
          float v242_tp{};
          float v243_tp{};
          float v244_tp{};
          tensorforge::transpose4x4b32(v241_tp, v242_tp, v243_tp, v244_tp, v237_data, v238_data, v239_data, v240_data);
          tensorforge::VectorT<float, 4> v245_acc{};
          int32_t v251_a = v11_lane + 0;
          float v258_data = glb_m1[v11_lane];
          int32_t v264_a = v11_lane + 16;
          float v271_data = glb_m1[(v11_lane + 16)];
          int32_t v277_a = v11_lane + 32;
          float v284_data = glb_m1[(v11_lane + 32)];
          int32_t v290_a = v11_lane + 48;
          float v297_data = glb_m1[(v11_lane + 48)];
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v258_data, v245_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v271_data, v298_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v284_data, v299_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v297_data, v300_acc, 2, 0, 0);
          int32_t v307_a = v11_lane + 64;
          float v314_data = glb_m1[(v11_lane + 64)];
          int32_t v320_a = v11_lane + 80;
          float v327_data = glb_m1[(v11_lane + 80)];
          int32_t v333_a = v11_lane + 96;
          float v340_data = glb_m1[(v11_lane + 96)];
          int32_t v346_a = v11_lane + 112;
          float v353_data = glb_m1[(v11_lane + 112)];
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v314_data, v301_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v327_data, v354_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v340_data, v355_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v353_data, v356_acc, 2, 1, 0);
          int32_t v363_a = v11_lane + 128;
          float v370_data = glb_m1[(v11_lane + 128)];
          int32_t v376_a = v11_lane + 144;
          float v383_data = glb_m1[(v11_lane + 144)];
          int32_t v389_a = v11_lane + 160;
          float v396_data = glb_m1[(v11_lane + 160)];
          int32_t v402_a = v11_lane + 176;
          float v409_data = glb_m1[(v11_lane + 176)];
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v370_data, v357_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v383_data, v410_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v396_data, v411_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v409_data, v412_acc, 2, 2, 0);
          int32_t v419_a = v11_lane + 192;
          float v426_data = glb_m1[(v11_lane + 192)];
          int32_t v432_a = v11_lane + 208;
          float v439_data = glb_m1[(v11_lane + 208)];
          int32_t v445_a = v11_lane + 224;
          float v452_data = glb_m1[(v11_lane + 224)];
          int32_t v458_a = v11_lane + 240;
          float v465_data = glb_m1[(v11_lane + 240)];
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v426_data, v413_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v439_data, v466_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v452_data, v467_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v465_data, v468_acc, 2, 3, 0);
          ir1[4] = (v469_acc[0]);
          ir1[5] = (v469_acc[1]);
          ir1[6] = (v469_acc[2]);
          ir1[7] = (v469_acc[3]);
          float v474_data = r0[8];
          float v475_data = r0[9];
          float v476_data = r0[10];
          float v477_data = r0[11];
          float v478_tp{};
          float v479_tp{};
          float v480_tp{};
          float v481_tp{};
          tensorforge::transpose4x4b32(v478_tp, v479_tp, v480_tp, v481_tp, v474_data, v475_data, v476_data, v477_data);
          tensorforge::VectorT<float, 4> v482_acc{};
          int32_t v488_a = v11_lane + 0;
          float v495_data = glb_m1[v11_lane];
          int32_t v501_a = v11_lane + 16;
          float v508_data = glb_m1[(v11_lane + 16)];
          int32_t v514_a = v11_lane + 32;
          float v521_data = glb_m1[(v11_lane + 32)];
          int32_t v527_a = v11_lane + 48;
          float v534_data = glb_m1[(v11_lane + 48)];
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v495_data, v482_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v508_data, v535_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v521_data, v536_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v534_data, v537_acc, 2, 0, 0);
          int32_t v544_a = v11_lane + 64;
          float v551_data = glb_m1[(v11_lane + 64)];
          int32_t v557_a = v11_lane + 80;
          float v564_data = glb_m1[(v11_lane + 80)];
          int32_t v570_a = v11_lane + 96;
          float v577_data = glb_m1[(v11_lane + 96)];
          int32_t v583_a = v11_lane + 112;
          float v590_data = glb_m1[(v11_lane + 112)];
          tensorforge::VectorT<float, 4> v591_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v551_data, v538_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v592_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v564_data, v591_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v593_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v577_data, v592_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v590_data, v593_acc, 2, 1, 0);
          int32_t v600_a = v11_lane + 128;
          float v607_data = glb_m1[(v11_lane + 128)];
          int32_t v613_a = v11_lane + 144;
          float v620_data = glb_m1[(v11_lane + 144)];
          int32_t v626_a = v11_lane + 160;
          float v633_data = glb_m1[(v11_lane + 160)];
          int32_t v639_a = v11_lane + 176;
          float v646_data = glb_m1[(v11_lane + 176)];
          tensorforge::VectorT<float, 4> v647_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v607_data, v594_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v648_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v620_data, v647_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v649_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v633_data, v648_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v650_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v646_data, v649_acc, 2, 2, 0);
          int32_t v656_a = v11_lane + 192;
          float v663_data = glb_m1[(v11_lane + 192)];
          int32_t v669_a = v11_lane + 208;
          float v676_data = glb_m1[(v11_lane + 208)];
          int32_t v682_a = v11_lane + 224;
          float v689_data = glb_m1[(v11_lane + 224)];
          int32_t v695_a = v11_lane + 240;
          float v702_data = glb_m1[(v11_lane + 240)];
          tensorforge::VectorT<float, 4> v703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v478_tp, v663_data, v650_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v704_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v479_tp, v676_data, v703_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v705_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v480_tp, v689_data, v704_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v481_tp, v702_data, v705_acc, 2, 3, 0);
          ir1[8] = (v706_acc[0]);
          ir1[9] = (v706_acc[1]);
          ir1[10] = (v706_acc[2]);
          ir1[11] = (v706_acc[3]);
          float v711_data = r0[12];
          float v712_data = r0[13];
          float v713_data = r0[14];
          float v714_data = r0[15];
          float v715_tp{};
          float v716_tp{};
          float v717_tp{};
          float v718_tp{};
          tensorforge::transpose4x4b32(v715_tp, v716_tp, v717_tp, v718_tp, v711_data, v712_data, v713_data, v714_data);
          tensorforge::VectorT<float, 4> v719_acc{};
          int32_t v725_a = v11_lane + 0;
          float v732_data = glb_m1[v11_lane];
          int32_t v738_a = v11_lane + 16;
          float v745_data = glb_m1[(v11_lane + 16)];
          int32_t v751_a = v11_lane + 32;
          float v758_data = glb_m1[(v11_lane + 32)];
          int32_t v764_a = v11_lane + 48;
          float v771_data = glb_m1[(v11_lane + 48)];
          tensorforge::VectorT<float, 4> v772_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v715_tp, v732_data, v719_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v773_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v716_tp, v745_data, v772_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v774_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v717_tp, v758_data, v773_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v775_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v718_tp, v771_data, v774_acc, 2, 0, 0);
          int32_t v781_a = v11_lane + 64;
          float v788_data = glb_m1[(v11_lane + 64)];
          int32_t v794_a = v11_lane + 80;
          float v801_data = glb_m1[(v11_lane + 80)];
          int32_t v807_a = v11_lane + 96;
          float v814_data = glb_m1[(v11_lane + 96)];
          int32_t v820_a = v11_lane + 112;
          float v827_data = glb_m1[(v11_lane + 112)];
          tensorforge::VectorT<float, 4> v828_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v715_tp, v788_data, v775_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v829_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v716_tp, v801_data, v828_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v830_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v717_tp, v814_data, v829_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v831_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v718_tp, v827_data, v830_acc, 2, 1, 0);
          int32_t v837_a = v11_lane + 128;
          float v844_data = glb_m1[(v11_lane + 128)];
          int32_t v850_a = v11_lane + 144;
          float v857_data = glb_m1[(v11_lane + 144)];
          int32_t v863_a = v11_lane + 160;
          float v870_data = glb_m1[(v11_lane + 160)];
          int32_t v876_a = v11_lane + 176;
          float v883_data = glb_m1[(v11_lane + 176)];
          tensorforge::VectorT<float, 4> v884_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v715_tp, v844_data, v831_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v885_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v716_tp, v857_data, v884_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v717_tp, v870_data, v885_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v887_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v718_tp, v883_data, v886_acc, 2, 2, 0);
          int32_t v893_a = v11_lane + 192;
          float v900_data = glb_m1[(v11_lane + 192)];
          int32_t v906_a = v11_lane + 208;
          float v913_data = glb_m1[(v11_lane + 208)];
          int32_t v919_a = v11_lane + 224;
          float v926_data = glb_m1[(v11_lane + 224)];
          int32_t v932_a = v11_lane + 240;
          float v939_data = glb_m1[(v11_lane + 240)];
          tensorforge::VectorT<float, 4> v940_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v715_tp, v900_data, v887_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v941_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v716_tp, v913_data, v940_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v942_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v717_tp, v926_data, v941_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v943_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v718_tp, v939_data, v942_acc, 2, 3, 0);
          ir1[12] = (v943_acc[0]);
          ir1[13] = (v943_acc[1]);
          ir1[14] = (v943_acc[2]);
          ir1[15] = (v943_acc[3]);
          // glb_m0 = store{r>g}(r1);
          int32_t v950_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v951_i0 = 0; v951_i0 < 1; ++v951_i0) {
            int32_t v960_lead = v950_lead + (v951_i0 * 16);
            #pragma unroll
            for (int32_t v952_i1 = 0; v952_i1 < 16; ++v952_i1) {
              int32_t v953_a = v951_i0 + v952_i1;
              float v955_data = r1[(v951_i0 + v952_i1)];
              int32_t v962_a = v960_lead + (v952_i1 * 16);
              glb_m0[v962_a] = v955_data;
            }
          }
          ;
        }
      }
    }
  }
}

