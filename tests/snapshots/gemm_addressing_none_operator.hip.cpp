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
          // r0 = load{g>r}(glb_m2);
          float v5_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v5_lin;
          float v6_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v6_lin;
          float v7_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v7_lin;
          float v8_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v8_lin;
          float v9_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v9_lin;
          float v10_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v10_lin;
          float v11_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v11_lin;
          float v12_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v12_lin;
          float v13_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v13_lin;
          float v14_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v14_lin;
          float v15_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v15_lin;
          float v16_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v16_lin;
          float v17_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v17_lin;
          float v18_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v18_lin;
          float v19_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v19_lin;
          float v20_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v20_lin;
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v22_data = r0[0];
          float v23_data = r0[1];
          float v24_data = r0[2];
          float v25_data = r0[3];
          float v26_tp{};
          float v27_tp{};
          float v28_tp{};
          float v29_tp{};
          tensorforge::transpose4x4b32(v26_tp, v27_tp, v28_tp, v29_tp, v22_data, v23_data, v24_data, v25_data);
          tensorforge::VectorT<float, 4> v30_acc{};
          int32_t v33_lane = threadIdx.x % 16;
          int32_t v36_a = v33_lane + 0;
          float v43_data = glb_m1[v33_lane];
          int32_t v49_a = v33_lane + 16;
          float v56_data = glb_m1[(v33_lane + 16)];
          int32_t v62_a = v33_lane + 32;
          float v69_data = glb_m1[(v33_lane + 32)];
          int32_t v75_a = v33_lane + 48;
          float v82_data = glb_m1[(v33_lane + 48)];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v43_data, v30_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v56_data, v83_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v69_data, v84_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v82_data, v85_acc, 2, 0, 0);
          int32_t v92_a = v33_lane + 64;
          float v99_data = glb_m1[(v33_lane + 64)];
          int32_t v105_a = v33_lane + 80;
          float v112_data = glb_m1[(v33_lane + 80)];
          int32_t v118_a = v33_lane + 96;
          float v125_data = glb_m1[(v33_lane + 96)];
          int32_t v131_a = v33_lane + 112;
          float v138_data = glb_m1[(v33_lane + 112)];
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v99_data, v86_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v112_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v125_data, v140_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v138_data, v141_acc, 2, 1, 0);
          int32_t v148_a = v33_lane + 128;
          float v155_data = glb_m1[(v33_lane + 128)];
          int32_t v161_a = v33_lane + 144;
          float v168_data = glb_m1[(v33_lane + 144)];
          int32_t v174_a = v33_lane + 160;
          float v181_data = glb_m1[(v33_lane + 160)];
          int32_t v187_a = v33_lane + 176;
          float v194_data = glb_m1[(v33_lane + 176)];
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v155_data, v142_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v168_data, v195_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v181_data, v196_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v194_data, v197_acc, 2, 2, 0);
          int32_t v204_a = v33_lane + 192;
          float v211_data = glb_m1[(v33_lane + 192)];
          int32_t v217_a = v33_lane + 208;
          float v224_data = glb_m1[(v33_lane + 208)];
          int32_t v230_a = v33_lane + 224;
          float v237_data = glb_m1[(v33_lane + 224)];
          int32_t v243_a = v33_lane + 240;
          float v250_data = glb_m1[(v33_lane + 240)];
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v211_data, v198_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v224_data, v251_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v237_data, v252_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v250_data, v253_acc, 2, 3, 0);
          r1[0] = (v254_acc[0]);
          r1[1] = (v254_acc[1]);
          r1[2] = (v254_acc[2]);
          r1[3] = (v254_acc[3]);
          float v259_data = r0[4];
          float v260_data = r0[5];
          float v261_data = r0[6];
          float v262_data = r0[7];
          float v263_tp{};
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          tensorforge::transpose4x4b32(v263_tp, v264_tp, v265_tp, v266_tp, v259_data, v260_data, v261_data, v262_data);
          tensorforge::VectorT<float, 4> v267_acc{};
          int32_t v273_a = v33_lane + 0;
          float v280_data = glb_m1[v33_lane];
          int32_t v286_a = v33_lane + 16;
          float v293_data = glb_m1[(v33_lane + 16)];
          int32_t v299_a = v33_lane + 32;
          float v306_data = glb_m1[(v33_lane + 32)];
          int32_t v312_a = v33_lane + 48;
          float v319_data = glb_m1[(v33_lane + 48)];
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v280_data, v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v293_data, v320_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v306_data, v321_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v319_data, v322_acc, 2, 0, 0);
          int32_t v329_a = v33_lane + 64;
          float v336_data = glb_m1[(v33_lane + 64)];
          int32_t v342_a = v33_lane + 80;
          float v349_data = glb_m1[(v33_lane + 80)];
          int32_t v355_a = v33_lane + 96;
          float v362_data = glb_m1[(v33_lane + 96)];
          int32_t v368_a = v33_lane + 112;
          float v375_data = glb_m1[(v33_lane + 112)];
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v336_data, v323_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v349_data, v376_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v362_data, v377_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v375_data, v378_acc, 2, 1, 0);
          int32_t v385_a = v33_lane + 128;
          float v392_data = glb_m1[(v33_lane + 128)];
          int32_t v398_a = v33_lane + 144;
          float v405_data = glb_m1[(v33_lane + 144)];
          int32_t v411_a = v33_lane + 160;
          float v418_data = glb_m1[(v33_lane + 160)];
          int32_t v424_a = v33_lane + 176;
          float v431_data = glb_m1[(v33_lane + 176)];
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v392_data, v379_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v405_data, v432_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v418_data, v433_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v431_data, v434_acc, 2, 2, 0);
          int32_t v441_a = v33_lane + 192;
          float v448_data = glb_m1[(v33_lane + 192)];
          int32_t v454_a = v33_lane + 208;
          float v461_data = glb_m1[(v33_lane + 208)];
          int32_t v467_a = v33_lane + 224;
          float v474_data = glb_m1[(v33_lane + 224)];
          int32_t v480_a = v33_lane + 240;
          float v487_data = glb_m1[(v33_lane + 240)];
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v448_data, v435_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v461_data, v488_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v474_data, v489_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v487_data, v490_acc, 2, 3, 0);
          r1[4] = (v491_acc[0]);
          r1[5] = (v491_acc[1]);
          r1[6] = (v491_acc[2]);
          r1[7] = (v491_acc[3]);
          float v496_data = r0[8];
          float v497_data = r0[9];
          float v498_data = r0[10];
          float v499_data = r0[11];
          float v500_tp{};
          float v501_tp{};
          float v502_tp{};
          float v503_tp{};
          tensorforge::transpose4x4b32(v500_tp, v501_tp, v502_tp, v503_tp, v496_data, v497_data, v498_data, v499_data);
          tensorforge::VectorT<float, 4> v504_acc{};
          int32_t v510_a = v33_lane + 0;
          float v517_data = glb_m1[v33_lane];
          int32_t v523_a = v33_lane + 16;
          float v530_data = glb_m1[(v33_lane + 16)];
          int32_t v536_a = v33_lane + 32;
          float v543_data = glb_m1[(v33_lane + 32)];
          int32_t v549_a = v33_lane + 48;
          float v556_data = glb_m1[(v33_lane + 48)];
          tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v517_data, v504_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v530_data, v557_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v543_data, v558_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v556_data, v559_acc, 2, 0, 0);
          int32_t v566_a = v33_lane + 64;
          float v573_data = glb_m1[(v33_lane + 64)];
          int32_t v579_a = v33_lane + 80;
          float v586_data = glb_m1[(v33_lane + 80)];
          int32_t v592_a = v33_lane + 96;
          float v599_data = glb_m1[(v33_lane + 96)];
          int32_t v605_a = v33_lane + 112;
          float v612_data = glb_m1[(v33_lane + 112)];
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v573_data, v560_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v586_data, v613_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v599_data, v614_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v612_data, v615_acc, 2, 1, 0);
          int32_t v622_a = v33_lane + 128;
          float v629_data = glb_m1[(v33_lane + 128)];
          int32_t v635_a = v33_lane + 144;
          float v642_data = glb_m1[(v33_lane + 144)];
          int32_t v648_a = v33_lane + 160;
          float v655_data = glb_m1[(v33_lane + 160)];
          int32_t v661_a = v33_lane + 176;
          float v668_data = glb_m1[(v33_lane + 176)];
          tensorforge::VectorT<float, 4> v669_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v629_data, v616_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v670_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v642_data, v669_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v671_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v655_data, v670_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v672_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v668_data, v671_acc, 2, 2, 0);
          int32_t v678_a = v33_lane + 192;
          float v685_data = glb_m1[(v33_lane + 192)];
          int32_t v691_a = v33_lane + 208;
          float v698_data = glb_m1[(v33_lane + 208)];
          int32_t v704_a = v33_lane + 224;
          float v711_data = glb_m1[(v33_lane + 224)];
          int32_t v717_a = v33_lane + 240;
          float v724_data = glb_m1[(v33_lane + 240)];
          tensorforge::VectorT<float, 4> v725_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v685_data, v672_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v726_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v698_data, v725_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v727_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v711_data, v726_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v728_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v724_data, v727_acc, 2, 3, 0);
          r1[8] = (v728_acc[0]);
          r1[9] = (v728_acc[1]);
          r1[10] = (v728_acc[2]);
          r1[11] = (v728_acc[3]);
          float v733_data = r0[12];
          float v734_data = r0[13];
          float v735_data = r0[14];
          float v736_data = r0[15];
          float v737_tp{};
          float v738_tp{};
          float v739_tp{};
          float v740_tp{};
          tensorforge::transpose4x4b32(v737_tp, v738_tp, v739_tp, v740_tp, v733_data, v734_data, v735_data, v736_data);
          tensorforge::VectorT<float, 4> v741_acc{};
          int32_t v747_a = v33_lane + 0;
          float v754_data = glb_m1[v33_lane];
          int32_t v760_a = v33_lane + 16;
          float v767_data = glb_m1[(v33_lane + 16)];
          int32_t v773_a = v33_lane + 32;
          float v780_data = glb_m1[(v33_lane + 32)];
          int32_t v786_a = v33_lane + 48;
          float v793_data = glb_m1[(v33_lane + 48)];
          tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v754_data, v741_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v795_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v767_data, v794_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v796_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v780_data, v795_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v797_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v793_data, v796_acc, 2, 0, 0);
          int32_t v803_a = v33_lane + 64;
          float v810_data = glb_m1[(v33_lane + 64)];
          int32_t v816_a = v33_lane + 80;
          float v823_data = glb_m1[(v33_lane + 80)];
          int32_t v829_a = v33_lane + 96;
          float v836_data = glb_m1[(v33_lane + 96)];
          int32_t v842_a = v33_lane + 112;
          float v849_data = glb_m1[(v33_lane + 112)];
          tensorforge::VectorT<float, 4> v850_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v810_data, v797_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v851_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v823_data, v850_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v852_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v836_data, v851_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v853_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v849_data, v852_acc, 2, 1, 0);
          int32_t v859_a = v33_lane + 128;
          float v866_data = glb_m1[(v33_lane + 128)];
          int32_t v872_a = v33_lane + 144;
          float v879_data = glb_m1[(v33_lane + 144)];
          int32_t v885_a = v33_lane + 160;
          float v892_data = glb_m1[(v33_lane + 160)];
          int32_t v898_a = v33_lane + 176;
          float v905_data = glb_m1[(v33_lane + 176)];
          tensorforge::VectorT<float, 4> v906_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v866_data, v853_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v907_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v879_data, v906_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v908_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v892_data, v907_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v909_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v905_data, v908_acc, 2, 2, 0);
          int32_t v915_a = v33_lane + 192;
          float v922_data = glb_m1[(v33_lane + 192)];
          int32_t v928_a = v33_lane + 208;
          float v935_data = glb_m1[(v33_lane + 208)];
          int32_t v941_a = v33_lane + 224;
          float v948_data = glb_m1[(v33_lane + 224)];
          int32_t v954_a = v33_lane + 240;
          float v961_data = glb_m1[(v33_lane + 240)];
          tensorforge::VectorT<float, 4> v962_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v922_data, v909_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v963_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v935_data, v962_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v964_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v948_data, v963_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v965_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v961_data, v964_acc, 2, 3, 0);
          r1[12] = (v965_acc[0]);
          r1[13] = (v965_acc[1]);
          r1[14] = (v965_acc[2]);
          r1[15] = (v965_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v973_i0 = 0; v973_i0 < 1; ++v973_i0) {
            int32_t v982_lead = v33_lane + (v973_i0 * 16);
            #pragma unroll
            for (int32_t v974_i1 = 0; v974_i1 < 16; ++v974_i1) {
              int32_t v975_a = v973_i0 + v974_i1;
              float v977_data = r1[(v973_i0 + v974_i1)];
              glb_m0[(v982_lead + (v974_i1 * 16))] = v977_data;
            }
          }
        }
      }
    }
  }
}

