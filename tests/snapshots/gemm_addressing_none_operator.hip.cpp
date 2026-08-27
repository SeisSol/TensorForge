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
          float v2_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v2_lin;
          float v3_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v3_lin;
          float v4_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v4_lin;
          float v5_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v5_lin;
          float v6_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v6_lin;
          float v7_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v7_lin;
          float v8_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v8_lin;
          float v9_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v9_lin;
          float v10_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v10_lin;
          float v11_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v11_lin;
          float v12_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v12_lin;
          float v13_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v13_lin;
          float v14_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v14_lin;
          float v15_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v15_lin;
          float v16_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v16_lin;
          float v17_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v17_lin;
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir1 = r1;
          float v19_data = r0[0];
          float v20_data = r0[1];
          float v21_data = r0[2];
          float v22_data = r0[3];
          float v23_tp{};
          float v24_tp{};
          float v25_tp{};
          float v26_tp{};
          tensorforge::transpose4x4b32(v23_tp, v24_tp, v25_tp, v26_tp, v19_data, v20_data, v21_data, v22_data);
          tensorforge::VectorT<float, 4> v27_acc{};
          int32_t v30_lane = threadIdx.x % 16;
          int32_t v33_a = v30_lane + 0;
          float v40_data = glb_m1[v30_lane];
          int32_t v46_a = v30_lane + 16;
          float v53_data = glb_m1[(v30_lane + 16)];
          int32_t v59_a = v30_lane + 32;
          float v66_data = glb_m1[(v30_lane + 32)];
          int32_t v72_a = v30_lane + 48;
          float v79_data = glb_m1[(v30_lane + 48)];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v40_data, v27_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v24_tp, v53_data, v80_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v66_data, v81_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v79_data, v82_acc, 2, 0, 0);
          int32_t v89_a = v30_lane + 64;
          float v96_data = glb_m1[(v30_lane + 64)];
          int32_t v102_a = v30_lane + 80;
          float v109_data = glb_m1[(v30_lane + 80)];
          int32_t v115_a = v30_lane + 96;
          float v122_data = glb_m1[(v30_lane + 96)];
          int32_t v128_a = v30_lane + 112;
          float v135_data = glb_m1[(v30_lane + 112)];
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v96_data, v83_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v24_tp, v109_data, v136_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v122_data, v137_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v135_data, v138_acc, 2, 1, 0);
          int32_t v145_a = v30_lane + 128;
          float v152_data = glb_m1[(v30_lane + 128)];
          int32_t v158_a = v30_lane + 144;
          float v165_data = glb_m1[(v30_lane + 144)];
          int32_t v171_a = v30_lane + 160;
          float v178_data = glb_m1[(v30_lane + 160)];
          int32_t v184_a = v30_lane + 176;
          float v191_data = glb_m1[(v30_lane + 176)];
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v152_data, v139_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v24_tp, v165_data, v192_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v178_data, v193_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v191_data, v194_acc, 2, 2, 0);
          int32_t v201_a = v30_lane + 192;
          float v208_data = glb_m1[(v30_lane + 192)];
          int32_t v214_a = v30_lane + 208;
          float v221_data = glb_m1[(v30_lane + 208)];
          int32_t v227_a = v30_lane + 224;
          float v234_data = glb_m1[(v30_lane + 224)];
          int32_t v240_a = v30_lane + 240;
          float v247_data = glb_m1[(v30_lane + 240)];
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v208_data, v195_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v24_tp, v221_data, v248_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v234_data, v249_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v247_data, v250_acc, 2, 3, 0);
          ir1[0] = (v251_acc[0]);
          ir1[1] = (v251_acc[1]);
          ir1[2] = (v251_acc[2]);
          ir1[3] = (v251_acc[3]);
          float v256_data = r0[4];
          float v257_data = r0[5];
          float v258_data = r0[6];
          float v259_data = r0[7];
          float v260_tp{};
          float v261_tp{};
          float v262_tp{};
          float v263_tp{};
          tensorforge::transpose4x4b32(v260_tp, v261_tp, v262_tp, v263_tp, v256_data, v257_data, v258_data, v259_data);
          tensorforge::VectorT<float, 4> v264_acc{};
          int32_t v270_a = v30_lane + 0;
          float v277_data = glb_m1[v30_lane];
          int32_t v283_a = v30_lane + 16;
          float v290_data = glb_m1[(v30_lane + 16)];
          int32_t v296_a = v30_lane + 32;
          float v303_data = glb_m1[(v30_lane + 32)];
          int32_t v309_a = v30_lane + 48;
          float v316_data = glb_m1[(v30_lane + 48)];
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v277_data, v264_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v290_data, v317_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v303_data, v318_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v316_data, v319_acc, 2, 0, 0);
          int32_t v326_a = v30_lane + 64;
          float v333_data = glb_m1[(v30_lane + 64)];
          int32_t v339_a = v30_lane + 80;
          float v346_data = glb_m1[(v30_lane + 80)];
          int32_t v352_a = v30_lane + 96;
          float v359_data = glb_m1[(v30_lane + 96)];
          int32_t v365_a = v30_lane + 112;
          float v372_data = glb_m1[(v30_lane + 112)];
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v333_data, v320_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v346_data, v373_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v359_data, v374_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v372_data, v375_acc, 2, 1, 0);
          int32_t v382_a = v30_lane + 128;
          float v389_data = glb_m1[(v30_lane + 128)];
          int32_t v395_a = v30_lane + 144;
          float v402_data = glb_m1[(v30_lane + 144)];
          int32_t v408_a = v30_lane + 160;
          float v415_data = glb_m1[(v30_lane + 160)];
          int32_t v421_a = v30_lane + 176;
          float v428_data = glb_m1[(v30_lane + 176)];
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v389_data, v376_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v402_data, v429_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v415_data, v430_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v428_data, v431_acc, 2, 2, 0);
          int32_t v438_a = v30_lane + 192;
          float v445_data = glb_m1[(v30_lane + 192)];
          int32_t v451_a = v30_lane + 208;
          float v458_data = glb_m1[(v30_lane + 208)];
          int32_t v464_a = v30_lane + 224;
          float v471_data = glb_m1[(v30_lane + 224)];
          int32_t v477_a = v30_lane + 240;
          float v484_data = glb_m1[(v30_lane + 240)];
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v445_data, v432_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v458_data, v485_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v471_data, v486_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v484_data, v487_acc, 2, 3, 0);
          ir1[4] = (v488_acc[0]);
          ir1[5] = (v488_acc[1]);
          ir1[6] = (v488_acc[2]);
          ir1[7] = (v488_acc[3]);
          float v493_data = r0[8];
          float v494_data = r0[9];
          float v495_data = r0[10];
          float v496_data = r0[11];
          float v497_tp{};
          float v498_tp{};
          float v499_tp{};
          float v500_tp{};
          tensorforge::transpose4x4b32(v497_tp, v498_tp, v499_tp, v500_tp, v493_data, v494_data, v495_data, v496_data);
          tensorforge::VectorT<float, 4> v501_acc{};
          int32_t v507_a = v30_lane + 0;
          float v514_data = glb_m1[v30_lane];
          int32_t v520_a = v30_lane + 16;
          float v527_data = glb_m1[(v30_lane + 16)];
          int32_t v533_a = v30_lane + 32;
          float v540_data = glb_m1[(v30_lane + 32)];
          int32_t v546_a = v30_lane + 48;
          float v553_data = glb_m1[(v30_lane + 48)];
          tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v514_data, v501_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v527_data, v554_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v540_data, v555_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v553_data, v556_acc, 2, 0, 0);
          int32_t v563_a = v30_lane + 64;
          float v570_data = glb_m1[(v30_lane + 64)];
          int32_t v576_a = v30_lane + 80;
          float v583_data = glb_m1[(v30_lane + 80)];
          int32_t v589_a = v30_lane + 96;
          float v596_data = glb_m1[(v30_lane + 96)];
          int32_t v602_a = v30_lane + 112;
          float v609_data = glb_m1[(v30_lane + 112)];
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v570_data, v557_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v583_data, v610_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v596_data, v611_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v609_data, v612_acc, 2, 1, 0);
          int32_t v619_a = v30_lane + 128;
          float v626_data = glb_m1[(v30_lane + 128)];
          int32_t v632_a = v30_lane + 144;
          float v639_data = glb_m1[(v30_lane + 144)];
          int32_t v645_a = v30_lane + 160;
          float v652_data = glb_m1[(v30_lane + 160)];
          int32_t v658_a = v30_lane + 176;
          float v665_data = glb_m1[(v30_lane + 176)];
          tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v626_data, v613_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v639_data, v666_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v668_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v652_data, v667_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v669_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v665_data, v668_acc, 2, 2, 0);
          int32_t v675_a = v30_lane + 192;
          float v682_data = glb_m1[(v30_lane + 192)];
          int32_t v688_a = v30_lane + 208;
          float v695_data = glb_m1[(v30_lane + 208)];
          int32_t v701_a = v30_lane + 224;
          float v708_data = glb_m1[(v30_lane + 224)];
          int32_t v714_a = v30_lane + 240;
          float v721_data = glb_m1[(v30_lane + 240)];
          tensorforge::VectorT<float, 4> v722_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v682_data, v669_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v723_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v695_data, v722_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v724_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v708_data, v723_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v725_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v500_tp, v721_data, v724_acc, 2, 3, 0);
          ir1[8] = (v725_acc[0]);
          ir1[9] = (v725_acc[1]);
          ir1[10] = (v725_acc[2]);
          ir1[11] = (v725_acc[3]);
          float v730_data = r0[12];
          float v731_data = r0[13];
          float v732_data = r0[14];
          float v733_data = r0[15];
          float v734_tp{};
          float v735_tp{};
          float v736_tp{};
          float v737_tp{};
          tensorforge::transpose4x4b32(v734_tp, v735_tp, v736_tp, v737_tp, v730_data, v731_data, v732_data, v733_data);
          tensorforge::VectorT<float, 4> v738_acc{};
          int32_t v744_a = v30_lane + 0;
          float v751_data = glb_m1[v30_lane];
          int32_t v757_a = v30_lane + 16;
          float v764_data = glb_m1[(v30_lane + 16)];
          int32_t v770_a = v30_lane + 32;
          float v777_data = glb_m1[(v30_lane + 32)];
          int32_t v783_a = v30_lane + 48;
          float v790_data = glb_m1[(v30_lane + 48)];
          tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v751_data, v738_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v764_data, v791_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v793_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v777_data, v792_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v790_data, v793_acc, 2, 0, 0);
          int32_t v800_a = v30_lane + 64;
          float v807_data = glb_m1[(v30_lane + 64)];
          int32_t v813_a = v30_lane + 80;
          float v820_data = glb_m1[(v30_lane + 80)];
          int32_t v826_a = v30_lane + 96;
          float v833_data = glb_m1[(v30_lane + 96)];
          int32_t v839_a = v30_lane + 112;
          float v846_data = glb_m1[(v30_lane + 112)];
          tensorforge::VectorT<float, 4> v847_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v807_data, v794_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v848_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v820_data, v847_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v849_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v833_data, v848_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v850_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v846_data, v849_acc, 2, 1, 0);
          int32_t v856_a = v30_lane + 128;
          float v863_data = glb_m1[(v30_lane + 128)];
          int32_t v869_a = v30_lane + 144;
          float v876_data = glb_m1[(v30_lane + 144)];
          int32_t v882_a = v30_lane + 160;
          float v889_data = glb_m1[(v30_lane + 160)];
          int32_t v895_a = v30_lane + 176;
          float v902_data = glb_m1[(v30_lane + 176)];
          tensorforge::VectorT<float, 4> v903_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v863_data, v850_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v904_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v876_data, v903_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v905_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v889_data, v904_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v906_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v902_data, v905_acc, 2, 2, 0);
          int32_t v912_a = v30_lane + 192;
          float v919_data = glb_m1[(v30_lane + 192)];
          int32_t v925_a = v30_lane + 208;
          float v932_data = glb_m1[(v30_lane + 208)];
          int32_t v938_a = v30_lane + 224;
          float v945_data = glb_m1[(v30_lane + 224)];
          int32_t v951_a = v30_lane + 240;
          float v958_data = glb_m1[(v30_lane + 240)];
          tensorforge::VectorT<float, 4> v959_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v919_data, v906_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v960_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v932_data, v959_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v961_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v945_data, v960_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v962_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v958_data, v961_acc, 2, 3, 0);
          ir1[12] = (v962_acc[0]);
          ir1[13] = (v962_acc[1]);
          ir1[14] = (v962_acc[2]);
          ir1[15] = (v962_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v970_i0 = 0; v970_i0 < 1; ++v970_i0) {
            int32_t v979_lead = v30_lane + (v970_i0 * 16);
            #pragma unroll
            for (int32_t v971_i1 = 0; v971_i1 < 16; ++v971_i1) {
              int32_t v972_a = v970_i0 + v971_i1;
              float v974_data = r1[(v970_i0 + v971_i1)];
              int32_t v981_a = v979_lead + (v971_i1 * 16);
              glb_m0[v981_a] = v974_data;
            }
          }
          ;
        }
      }
    }
  }
}

