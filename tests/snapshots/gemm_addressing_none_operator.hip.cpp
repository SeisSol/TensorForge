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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m2);
          float v9_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v9_lin;
          float v10_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v10_lin;
          float v11_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v11_lin;
          float v12_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v12_lin;
          float v13_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v13_lin;
          float v14_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v14_lin;
          float v15_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v15_lin;
          float v16_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v16_lin;
          float v17_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v17_lin;
          float v18_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v18_lin;
          float v19_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v19_lin;
          float v20_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v20_lin;
          float v21_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v21_lin;
          float v22_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v22_lin;
          float v23_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v23_lin;
          float v24_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v24_lin;
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v26_data = r0[0];
          float v27_data = r0[1];
          float v28_data = r0[2];
          float v29_data = r0[3];
          float v30_tp{};
          float v31_tp{};
          float v32_tp{};
          float v33_tp{};
          tensorforge::transpose4x4b32(v30_tp, v31_tp, v32_tp, v33_tp, v26_data, v27_data, v28_data, v29_data);
          tensorforge::VectorT<float, 4> v34_acc{};
          int32_t v37_lane = threadIdx.x % 16;
          int32_t v40_a = v37_lane + 0;
          float v47_data = glb_m1[v37_lane];
          int32_t v53_a = v37_lane + 16;
          float v60_data = glb_m1[(v37_lane + 16)];
          int32_t v66_a = v37_lane + 32;
          float v73_data = glb_m1[(v37_lane + 32)];
          int32_t v79_a = v37_lane + 48;
          float v86_data = glb_m1[(v37_lane + 48)];
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v47_data, v34_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v60_data, v87_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v73_data, v88_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v86_data, v89_acc, 2, 0, 0);
          int32_t v96_a = v37_lane + 64;
          float v103_data = glb_m1[(v37_lane + 64)];
          int32_t v109_a = v37_lane + 80;
          float v116_data = glb_m1[(v37_lane + 80)];
          int32_t v122_a = v37_lane + 96;
          float v129_data = glb_m1[(v37_lane + 96)];
          int32_t v135_a = v37_lane + 112;
          float v142_data = glb_m1[(v37_lane + 112)];
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v103_data, v90_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v116_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v129_data, v144_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v142_data, v145_acc, 2, 1, 0);
          int32_t v152_a = v37_lane + 128;
          float v159_data = glb_m1[(v37_lane + 128)];
          int32_t v165_a = v37_lane + 144;
          float v172_data = glb_m1[(v37_lane + 144)];
          int32_t v178_a = v37_lane + 160;
          float v185_data = glb_m1[(v37_lane + 160)];
          int32_t v191_a = v37_lane + 176;
          float v198_data = glb_m1[(v37_lane + 176)];
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v159_data, v146_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v172_data, v199_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v185_data, v200_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v198_data, v201_acc, 2, 2, 0);
          int32_t v208_a = v37_lane + 192;
          float v215_data = glb_m1[(v37_lane + 192)];
          int32_t v221_a = v37_lane + 208;
          float v228_data = glb_m1[(v37_lane + 208)];
          int32_t v234_a = v37_lane + 224;
          float v241_data = glb_m1[(v37_lane + 224)];
          int32_t v247_a = v37_lane + 240;
          float v254_data = glb_m1[(v37_lane + 240)];
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v215_data, v202_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v228_data, v255_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v241_data, v256_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v254_data, v257_acc, 2, 3, 0);
          r1[0] = (v258_acc[0]);
          r1[1] = (v258_acc[1]);
          r1[2] = (v258_acc[2]);
          r1[3] = (v258_acc[3]);
          float v263_data = r0[4];
          float v264_data = r0[5];
          float v265_data = r0[6];
          float v266_data = r0[7];
          float v267_tp{};
          float v268_tp{};
          float v269_tp{};
          float v270_tp{};
          tensorforge::transpose4x4b32(v267_tp, v268_tp, v269_tp, v270_tp, v263_data, v264_data, v265_data, v266_data);
          tensorforge::VectorT<float, 4> v271_acc{};
          int32_t v277_a = v37_lane + 0;
          float v284_data = glb_m1[v37_lane];
          int32_t v290_a = v37_lane + 16;
          float v297_data = glb_m1[(v37_lane + 16)];
          int32_t v303_a = v37_lane + 32;
          float v310_data = glb_m1[(v37_lane + 32)];
          int32_t v316_a = v37_lane + 48;
          float v323_data = glb_m1[(v37_lane + 48)];
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v284_data, v271_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v297_data, v324_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v310_data, v325_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v323_data, v326_acc, 2, 0, 0);
          int32_t v333_a = v37_lane + 64;
          float v340_data = glb_m1[(v37_lane + 64)];
          int32_t v346_a = v37_lane + 80;
          float v353_data = glb_m1[(v37_lane + 80)];
          int32_t v359_a = v37_lane + 96;
          float v366_data = glb_m1[(v37_lane + 96)];
          int32_t v372_a = v37_lane + 112;
          float v379_data = glb_m1[(v37_lane + 112)];
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v340_data, v327_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v353_data, v380_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v366_data, v381_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v379_data, v382_acc, 2, 1, 0);
          int32_t v389_a = v37_lane + 128;
          float v396_data = glb_m1[(v37_lane + 128)];
          int32_t v402_a = v37_lane + 144;
          float v409_data = glb_m1[(v37_lane + 144)];
          int32_t v415_a = v37_lane + 160;
          float v422_data = glb_m1[(v37_lane + 160)];
          int32_t v428_a = v37_lane + 176;
          float v435_data = glb_m1[(v37_lane + 176)];
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v396_data, v383_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v409_data, v436_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v422_data, v437_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v435_data, v438_acc, 2, 2, 0);
          int32_t v445_a = v37_lane + 192;
          float v452_data = glb_m1[(v37_lane + 192)];
          int32_t v458_a = v37_lane + 208;
          float v465_data = glb_m1[(v37_lane + 208)];
          int32_t v471_a = v37_lane + 224;
          float v478_data = glb_m1[(v37_lane + 224)];
          int32_t v484_a = v37_lane + 240;
          float v491_data = glb_m1[(v37_lane + 240)];
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v452_data, v439_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v465_data, v492_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v478_data, v493_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v491_data, v494_acc, 2, 3, 0);
          r1[4] = (v495_acc[0]);
          r1[5] = (v495_acc[1]);
          r1[6] = (v495_acc[2]);
          r1[7] = (v495_acc[3]);
          float v500_data = r0[8];
          float v501_data = r0[9];
          float v502_data = r0[10];
          float v503_data = r0[11];
          float v504_tp{};
          float v505_tp{};
          float v506_tp{};
          float v507_tp{};
          tensorforge::transpose4x4b32(v504_tp, v505_tp, v506_tp, v507_tp, v500_data, v501_data, v502_data, v503_data);
          tensorforge::VectorT<float, 4> v508_acc{};
          int32_t v514_a = v37_lane + 0;
          float v521_data = glb_m1[v37_lane];
          int32_t v527_a = v37_lane + 16;
          float v534_data = glb_m1[(v37_lane + 16)];
          int32_t v540_a = v37_lane + 32;
          float v547_data = glb_m1[(v37_lane + 32)];
          int32_t v553_a = v37_lane + 48;
          float v560_data = glb_m1[(v37_lane + 48)];
          tensorforge::VectorT<float, 4> v561_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v521_data, v508_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v505_tp, v534_data, v561_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v563_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v506_tp, v547_data, v562_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v564_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v507_tp, v560_data, v563_acc, 2, 0, 0);
          int32_t v570_a = v37_lane + 64;
          float v577_data = glb_m1[(v37_lane + 64)];
          int32_t v583_a = v37_lane + 80;
          float v590_data = glb_m1[(v37_lane + 80)];
          int32_t v596_a = v37_lane + 96;
          float v603_data = glb_m1[(v37_lane + 96)];
          int32_t v609_a = v37_lane + 112;
          float v616_data = glb_m1[(v37_lane + 112)];
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v577_data, v564_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v618_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v505_tp, v590_data, v617_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v506_tp, v603_data, v618_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v507_tp, v616_data, v619_acc, 2, 1, 0);
          int32_t v626_a = v37_lane + 128;
          float v633_data = glb_m1[(v37_lane + 128)];
          int32_t v639_a = v37_lane + 144;
          float v646_data = glb_m1[(v37_lane + 144)];
          int32_t v652_a = v37_lane + 160;
          float v659_data = glb_m1[(v37_lane + 160)];
          int32_t v665_a = v37_lane + 176;
          float v672_data = glb_m1[(v37_lane + 176)];
          tensorforge::VectorT<float, 4> v673_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v633_data, v620_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v674_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v505_tp, v646_data, v673_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v675_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v506_tp, v659_data, v674_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v676_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v507_tp, v672_data, v675_acc, 2, 2, 0);
          int32_t v682_a = v37_lane + 192;
          float v689_data = glb_m1[(v37_lane + 192)];
          int32_t v695_a = v37_lane + 208;
          float v702_data = glb_m1[(v37_lane + 208)];
          int32_t v708_a = v37_lane + 224;
          float v715_data = glb_m1[(v37_lane + 224)];
          int32_t v721_a = v37_lane + 240;
          float v728_data = glb_m1[(v37_lane + 240)];
          tensorforge::VectorT<float, 4> v729_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v689_data, v676_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v730_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v505_tp, v702_data, v729_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v731_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v506_tp, v715_data, v730_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v732_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v507_tp, v728_data, v731_acc, 2, 3, 0);
          r1[8] = (v732_acc[0]);
          r1[9] = (v732_acc[1]);
          r1[10] = (v732_acc[2]);
          r1[11] = (v732_acc[3]);
          float v737_data = r0[12];
          float v738_data = r0[13];
          float v739_data = r0[14];
          float v740_data = r0[15];
          float v741_tp{};
          float v742_tp{};
          float v743_tp{};
          float v744_tp{};
          tensorforge::transpose4x4b32(v741_tp, v742_tp, v743_tp, v744_tp, v737_data, v738_data, v739_data, v740_data);
          tensorforge::VectorT<float, 4> v745_acc{};
          int32_t v751_a = v37_lane + 0;
          float v758_data = glb_m1[v37_lane];
          int32_t v764_a = v37_lane + 16;
          float v771_data = glb_m1[(v37_lane + 16)];
          int32_t v777_a = v37_lane + 32;
          float v784_data = glb_m1[(v37_lane + 32)];
          int32_t v790_a = v37_lane + 48;
          float v797_data = glb_m1[(v37_lane + 48)];
          tensorforge::VectorT<float, 4> v798_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v758_data, v745_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v799_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v742_tp, v771_data, v798_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v800_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v743_tp, v784_data, v799_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v801_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v744_tp, v797_data, v800_acc, 2, 0, 0);
          int32_t v807_a = v37_lane + 64;
          float v814_data = glb_m1[(v37_lane + 64)];
          int32_t v820_a = v37_lane + 80;
          float v827_data = glb_m1[(v37_lane + 80)];
          int32_t v833_a = v37_lane + 96;
          float v840_data = glb_m1[(v37_lane + 96)];
          int32_t v846_a = v37_lane + 112;
          float v853_data = glb_m1[(v37_lane + 112)];
          tensorforge::VectorT<float, 4> v854_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v814_data, v801_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v855_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v742_tp, v827_data, v854_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v856_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v743_tp, v840_data, v855_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v857_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v744_tp, v853_data, v856_acc, 2, 1, 0);
          int32_t v863_a = v37_lane + 128;
          float v870_data = glb_m1[(v37_lane + 128)];
          int32_t v876_a = v37_lane + 144;
          float v883_data = glb_m1[(v37_lane + 144)];
          int32_t v889_a = v37_lane + 160;
          float v896_data = glb_m1[(v37_lane + 160)];
          int32_t v902_a = v37_lane + 176;
          float v909_data = glb_m1[(v37_lane + 176)];
          tensorforge::VectorT<float, 4> v910_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v870_data, v857_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v911_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v742_tp, v883_data, v910_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v912_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v743_tp, v896_data, v911_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v913_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v744_tp, v909_data, v912_acc, 2, 2, 0);
          int32_t v919_a = v37_lane + 192;
          float v926_data = glb_m1[(v37_lane + 192)];
          int32_t v932_a = v37_lane + 208;
          float v939_data = glb_m1[(v37_lane + 208)];
          int32_t v945_a = v37_lane + 224;
          float v952_data = glb_m1[(v37_lane + 224)];
          int32_t v958_a = v37_lane + 240;
          float v965_data = glb_m1[(v37_lane + 240)];
          tensorforge::VectorT<float, 4> v966_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v926_data, v913_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v967_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v742_tp, v939_data, v966_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v968_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v743_tp, v952_data, v967_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v969_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v744_tp, v965_data, v968_acc, 2, 3, 0);
          r1[12] = (v969_acc[0]);
          r1[13] = (v969_acc[1]);
          r1[14] = (v969_acc[2]);
          r1[15] = (v969_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v977_i0 = 0; v977_i0 < 1; ++v977_i0) {
            int32_t v986_lead = v37_lane + (v977_i0 * 16);
            #pragma unroll
            for (int32_t v978_i1 = 0; v978_i1 < 16; ++v978_i1) {
              int32_t v979_a = v977_i0 + v978_i1;
              float v981_data = r1[(v977_i0 + v978_i1)];
              glb_m0[(v986_lead + (v978_i1 * 16))] = v981_data;
            }
          }
        }
      }
    }
  }
}

