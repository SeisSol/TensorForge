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
          float v6_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v6_lin;
          float v7_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v7_lin;
          float v8_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v8_lin;
          float v9_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v9_lin;
          float v10_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v10_lin;
          float v11_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v11_lin;
          float v12_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v12_lin;
          float v13_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v13_lin;
          float v14_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v14_lin;
          float v15_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v15_lin;
          float v16_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v16_lin;
          float v17_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v17_lin;
          float v18_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v18_lin;
          float v19_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v19_lin;
          float v20_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v20_lin;
          float v21_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v21_lin;
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v23_data = r0[0];
          float v24_data = r0[1];
          float v25_data = r0[2];
          float v26_data = r0[3];
          float v27_tp{};
          float v28_tp{};
          float v29_tp{};
          float v30_tp{};
          tensorforge::transpose4x4b32(v27_tp, v28_tp, v29_tp, v30_tp, v23_data, v24_data, v25_data, v26_data);
          tensorforge::VectorT<float, 4> v31_acc{};
          int32_t v34_lane = threadIdx.x % 16;
          int32_t v37_a = v34_lane + 0;
          float v44_data = glb_m1[v34_lane];
          int32_t v50_a = v34_lane + 16;
          float v57_data = glb_m1[(v34_lane + 16)];
          int32_t v63_a = v34_lane + 32;
          float v70_data = glb_m1[(v34_lane + 32)];
          int32_t v76_a = v34_lane + 48;
          float v83_data = glb_m1[(v34_lane + 48)];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v44_data, v31_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v57_data, v84_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v70_data, v85_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v83_data, v86_acc, 2, 0, 0);
          int32_t v93_a = v34_lane + 64;
          float v100_data = glb_m1[(v34_lane + 64)];
          int32_t v106_a = v34_lane + 80;
          float v113_data = glb_m1[(v34_lane + 80)];
          int32_t v119_a = v34_lane + 96;
          float v126_data = glb_m1[(v34_lane + 96)];
          int32_t v132_a = v34_lane + 112;
          float v139_data = glb_m1[(v34_lane + 112)];
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v100_data, v87_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v113_data, v140_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v126_data, v141_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v139_data, v142_acc, 2, 1, 0);
          int32_t v149_a = v34_lane + 128;
          float v156_data = glb_m1[(v34_lane + 128)];
          int32_t v162_a = v34_lane + 144;
          float v169_data = glb_m1[(v34_lane + 144)];
          int32_t v175_a = v34_lane + 160;
          float v182_data = glb_m1[(v34_lane + 160)];
          int32_t v188_a = v34_lane + 176;
          float v195_data = glb_m1[(v34_lane + 176)];
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v156_data, v143_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v169_data, v196_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v182_data, v197_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v195_data, v198_acc, 2, 2, 0);
          int32_t v205_a = v34_lane + 192;
          float v212_data = glb_m1[(v34_lane + 192)];
          int32_t v218_a = v34_lane + 208;
          float v225_data = glb_m1[(v34_lane + 208)];
          int32_t v231_a = v34_lane + 224;
          float v238_data = glb_m1[(v34_lane + 224)];
          int32_t v244_a = v34_lane + 240;
          float v251_data = glb_m1[(v34_lane + 240)];
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v212_data, v199_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v225_data, v252_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v238_data, v253_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v251_data, v254_acc, 2, 3, 0);
          r1[0] = (v255_acc[0]);
          r1[1] = (v255_acc[1]);
          r1[2] = (v255_acc[2]);
          r1[3] = (v255_acc[3]);
          float v260_data = r0[4];
          float v261_data = r0[5];
          float v262_data = r0[6];
          float v263_data = r0[7];
          float v264_tp{};
          float v265_tp{};
          float v266_tp{};
          float v267_tp{};
          tensorforge::transpose4x4b32(v264_tp, v265_tp, v266_tp, v267_tp, v260_data, v261_data, v262_data, v263_data);
          tensorforge::VectorT<float, 4> v268_acc{};
          int32_t v274_a = v34_lane + 0;
          float v281_data = glb_m1[v34_lane];
          int32_t v287_a = v34_lane + 16;
          float v294_data = glb_m1[(v34_lane + 16)];
          int32_t v300_a = v34_lane + 32;
          float v307_data = glb_m1[(v34_lane + 32)];
          int32_t v313_a = v34_lane + 48;
          float v320_data = glb_m1[(v34_lane + 48)];
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v281_data, v268_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v294_data, v321_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v307_data, v322_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v320_data, v323_acc, 2, 0, 0);
          int32_t v330_a = v34_lane + 64;
          float v337_data = glb_m1[(v34_lane + 64)];
          int32_t v343_a = v34_lane + 80;
          float v350_data = glb_m1[(v34_lane + 80)];
          int32_t v356_a = v34_lane + 96;
          float v363_data = glb_m1[(v34_lane + 96)];
          int32_t v369_a = v34_lane + 112;
          float v376_data = glb_m1[(v34_lane + 112)];
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v337_data, v324_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v350_data, v377_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v363_data, v378_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v376_data, v379_acc, 2, 1, 0);
          int32_t v386_a = v34_lane + 128;
          float v393_data = glb_m1[(v34_lane + 128)];
          int32_t v399_a = v34_lane + 144;
          float v406_data = glb_m1[(v34_lane + 144)];
          int32_t v412_a = v34_lane + 160;
          float v419_data = glb_m1[(v34_lane + 160)];
          int32_t v425_a = v34_lane + 176;
          float v432_data = glb_m1[(v34_lane + 176)];
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v393_data, v380_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v406_data, v433_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v419_data, v434_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v432_data, v435_acc, 2, 2, 0);
          int32_t v442_a = v34_lane + 192;
          float v449_data = glb_m1[(v34_lane + 192)];
          int32_t v455_a = v34_lane + 208;
          float v462_data = glb_m1[(v34_lane + 208)];
          int32_t v468_a = v34_lane + 224;
          float v475_data = glb_m1[(v34_lane + 224)];
          int32_t v481_a = v34_lane + 240;
          float v488_data = glb_m1[(v34_lane + 240)];
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v449_data, v436_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v462_data, v489_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v266_tp, v475_data, v490_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v488_data, v491_acc, 2, 3, 0);
          r1[4] = (v492_acc[0]);
          r1[5] = (v492_acc[1]);
          r1[6] = (v492_acc[2]);
          r1[7] = (v492_acc[3]);
          float v497_data = r0[8];
          float v498_data = r0[9];
          float v499_data = r0[10];
          float v500_data = r0[11];
          float v501_tp{};
          float v502_tp{};
          float v503_tp{};
          float v504_tp{};
          tensorforge::transpose4x4b32(v501_tp, v502_tp, v503_tp, v504_tp, v497_data, v498_data, v499_data, v500_data);
          tensorforge::VectorT<float, 4> v505_acc{};
          int32_t v511_a = v34_lane + 0;
          float v518_data = glb_m1[v34_lane];
          int32_t v524_a = v34_lane + 16;
          float v531_data = glb_m1[(v34_lane + 16)];
          int32_t v537_a = v34_lane + 32;
          float v544_data = glb_m1[(v34_lane + 32)];
          int32_t v550_a = v34_lane + 48;
          float v557_data = glb_m1[(v34_lane + 48)];
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v518_data, v505_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v531_data, v558_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v544_data, v559_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v561_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v557_data, v560_acc, 2, 0, 0);
          int32_t v567_a = v34_lane + 64;
          float v574_data = glb_m1[(v34_lane + 64)];
          int32_t v580_a = v34_lane + 80;
          float v587_data = glb_m1[(v34_lane + 80)];
          int32_t v593_a = v34_lane + 96;
          float v600_data = glb_m1[(v34_lane + 96)];
          int32_t v606_a = v34_lane + 112;
          float v613_data = glb_m1[(v34_lane + 112)];
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v574_data, v561_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v587_data, v614_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v600_data, v615_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v613_data, v616_acc, 2, 1, 0);
          int32_t v623_a = v34_lane + 128;
          float v630_data = glb_m1[(v34_lane + 128)];
          int32_t v636_a = v34_lane + 144;
          float v643_data = glb_m1[(v34_lane + 144)];
          int32_t v649_a = v34_lane + 160;
          float v656_data = glb_m1[(v34_lane + 160)];
          int32_t v662_a = v34_lane + 176;
          float v669_data = glb_m1[(v34_lane + 176)];
          tensorforge::VectorT<float, 4> v670_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v630_data, v617_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v671_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v643_data, v670_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v672_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v656_data, v671_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v673_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v669_data, v672_acc, 2, 2, 0);
          int32_t v679_a = v34_lane + 192;
          float v686_data = glb_m1[(v34_lane + 192)];
          int32_t v692_a = v34_lane + 208;
          float v699_data = glb_m1[(v34_lane + 208)];
          int32_t v705_a = v34_lane + 224;
          float v712_data = glb_m1[(v34_lane + 224)];
          int32_t v718_a = v34_lane + 240;
          float v725_data = glb_m1[(v34_lane + 240)];
          tensorforge::VectorT<float, 4> v726_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v501_tp, v686_data, v673_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v727_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v502_tp, v699_data, v726_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v728_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v503_tp, v712_data, v727_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v729_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v504_tp, v725_data, v728_acc, 2, 3, 0);
          r1[8] = (v729_acc[0]);
          r1[9] = (v729_acc[1]);
          r1[10] = (v729_acc[2]);
          r1[11] = (v729_acc[3]);
          float v734_data = r0[12];
          float v735_data = r0[13];
          float v736_data = r0[14];
          float v737_data = r0[15];
          float v738_tp{};
          float v739_tp{};
          float v740_tp{};
          float v741_tp{};
          tensorforge::transpose4x4b32(v738_tp, v739_tp, v740_tp, v741_tp, v734_data, v735_data, v736_data, v737_data);
          tensorforge::VectorT<float, 4> v742_acc{};
          int32_t v748_a = v34_lane + 0;
          float v755_data = glb_m1[v34_lane];
          int32_t v761_a = v34_lane + 16;
          float v768_data = glb_m1[(v34_lane + 16)];
          int32_t v774_a = v34_lane + 32;
          float v781_data = glb_m1[(v34_lane + 32)];
          int32_t v787_a = v34_lane + 48;
          float v794_data = glb_m1[(v34_lane + 48)];
          tensorforge::VectorT<float, 4> v795_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v755_data, v742_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v796_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v768_data, v795_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v797_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v781_data, v796_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v798_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v794_data, v797_acc, 2, 0, 0);
          int32_t v804_a = v34_lane + 64;
          float v811_data = glb_m1[(v34_lane + 64)];
          int32_t v817_a = v34_lane + 80;
          float v824_data = glb_m1[(v34_lane + 80)];
          int32_t v830_a = v34_lane + 96;
          float v837_data = glb_m1[(v34_lane + 96)];
          int32_t v843_a = v34_lane + 112;
          float v850_data = glb_m1[(v34_lane + 112)];
          tensorforge::VectorT<float, 4> v851_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v811_data, v798_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v852_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v824_data, v851_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v853_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v837_data, v852_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v854_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v850_data, v853_acc, 2, 1, 0);
          int32_t v860_a = v34_lane + 128;
          float v867_data = glb_m1[(v34_lane + 128)];
          int32_t v873_a = v34_lane + 144;
          float v880_data = glb_m1[(v34_lane + 144)];
          int32_t v886_a = v34_lane + 160;
          float v893_data = glb_m1[(v34_lane + 160)];
          int32_t v899_a = v34_lane + 176;
          float v906_data = glb_m1[(v34_lane + 176)];
          tensorforge::VectorT<float, 4> v907_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v867_data, v854_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v908_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v880_data, v907_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v909_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v893_data, v908_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v910_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v906_data, v909_acc, 2, 2, 0);
          int32_t v916_a = v34_lane + 192;
          float v923_data = glb_m1[(v34_lane + 192)];
          int32_t v929_a = v34_lane + 208;
          float v936_data = glb_m1[(v34_lane + 208)];
          int32_t v942_a = v34_lane + 224;
          float v949_data = glb_m1[(v34_lane + 224)];
          int32_t v955_a = v34_lane + 240;
          float v962_data = glb_m1[(v34_lane + 240)];
          tensorforge::VectorT<float, 4> v963_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v738_tp, v923_data, v910_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v964_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v739_tp, v936_data, v963_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v965_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v740_tp, v949_data, v964_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v966_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v741_tp, v962_data, v965_acc, 2, 3, 0);
          r1[12] = (v966_acc[0]);
          r1[13] = (v966_acc[1]);
          r1[14] = (v966_acc[2]);
          r1[15] = (v966_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v974_i0 = 0; v974_i0 < 1; ++v974_i0) {
            int32_t v983_lead = v34_lane + (v974_i0 * 16);
            #pragma unroll
            for (int32_t v975_i1 = 0; v975_i1 < 16; ++v975_i1) {
              int32_t v976_a = v974_i0 + v975_i1;
              float v978_data = r1[(v974_i0 + v975_i1)];
              glb_m0[(v983_lead + (v975_i1 * 16))] = v978_data;
            }
          }
        }
      }
    }
  }
}

