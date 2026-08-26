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
          float v1_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v1_lin;
          float v2_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v2_lin;
          float v3_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v3_lin;
          float v4_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v4_lin;
          float v5_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v5_lin;
          float v6_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v6_lin;
          float v7_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v7_lin;
          float v8_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v8_lin;
          float v9_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v9_lin;
          float v10_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v10_lin;
          float v11_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v11_lin;
          float v12_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v12_lin;
          float v13_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v13_lin;
          float v14_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v14_lin;
          float v15_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v15_lin;
          float v16_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v16_lin;
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir1 = r1;
          float v18_data = r0[0];
          float v19_data = r0[1];
          float v20_data = r0[2];
          float v21_data = r0[3];
          float v22_tp{};
          float v23_tp{};
          float v24_tp{};
          float v25_tp{};
          tensorforge::transpose4x4b32(v22_tp, v23_tp, v24_tp, v25_tp, v18_data, v19_data, v20_data, v21_data);
          tensorforge::VectorT<float, 4> v26_acc{};
          int32_t v29_lane = threadIdx.x % 16;
          int32_t v32_a = v29_lane + 0;
          float v39_data = glb_m1[v29_lane];
          int32_t v45_a = v29_lane + 16;
          float v52_data = glb_m1[(v29_lane + 16)];
          int32_t v58_a = v29_lane + 32;
          float v65_data = glb_m1[(v29_lane + 32)];
          int32_t v71_a = v29_lane + 48;
          float v78_data = glb_m1[(v29_lane + 48)];
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v22_tp, v39_data, v26_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v52_data, v79_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v24_tp, v65_data, v80_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v78_data, v81_acc, 2, 0, 0);
          int32_t v88_a = v29_lane + 64;
          float v95_data = glb_m1[(v29_lane + 64)];
          int32_t v101_a = v29_lane + 80;
          float v108_data = glb_m1[(v29_lane + 80)];
          int32_t v114_a = v29_lane + 96;
          float v121_data = glb_m1[(v29_lane + 96)];
          int32_t v127_a = v29_lane + 112;
          float v134_data = glb_m1[(v29_lane + 112)];
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v22_tp, v95_data, v82_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v108_data, v135_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v24_tp, v121_data, v136_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v134_data, v137_acc, 2, 1, 0);
          int32_t v144_a = v29_lane + 128;
          float v151_data = glb_m1[(v29_lane + 128)];
          int32_t v157_a = v29_lane + 144;
          float v164_data = glb_m1[(v29_lane + 144)];
          int32_t v170_a = v29_lane + 160;
          float v177_data = glb_m1[(v29_lane + 160)];
          int32_t v183_a = v29_lane + 176;
          float v190_data = glb_m1[(v29_lane + 176)];
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v22_tp, v151_data, v138_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v164_data, v191_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v24_tp, v177_data, v192_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v190_data, v193_acc, 2, 2, 0);
          int32_t v200_a = v29_lane + 192;
          float v207_data = glb_m1[(v29_lane + 192)];
          int32_t v213_a = v29_lane + 208;
          float v220_data = glb_m1[(v29_lane + 208)];
          int32_t v226_a = v29_lane + 224;
          float v233_data = glb_m1[(v29_lane + 224)];
          int32_t v239_a = v29_lane + 240;
          float v246_data = glb_m1[(v29_lane + 240)];
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v22_tp, v207_data, v194_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v220_data, v247_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v24_tp, v233_data, v248_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v25_tp, v246_data, v249_acc, 2, 3, 0);
          ir1[0] = (v250_acc[0]);
          ir1[1] = (v250_acc[1]);
          ir1[2] = (v250_acc[2]);
          ir1[3] = (v250_acc[3]);
          float v255_data = r0[4];
          float v256_data = r0[5];
          float v257_data = r0[6];
          float v258_data = r0[7];
          float v259_tp{};
          float v260_tp{};
          float v261_tp{};
          float v262_tp{};
          tensorforge::transpose4x4b32(v259_tp, v260_tp, v261_tp, v262_tp, v255_data, v256_data, v257_data, v258_data);
          tensorforge::VectorT<float, 4> v263_acc{};
          int32_t v269_a = v29_lane + 0;
          float v276_data = glb_m1[v29_lane];
          int32_t v282_a = v29_lane + 16;
          float v289_data = glb_m1[(v29_lane + 16)];
          int32_t v295_a = v29_lane + 32;
          float v302_data = glb_m1[(v29_lane + 32)];
          int32_t v308_a = v29_lane + 48;
          float v315_data = glb_m1[(v29_lane + 48)];
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v276_data, v263_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v289_data, v316_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v302_data, v317_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v315_data, v318_acc, 2, 0, 0);
          int32_t v325_a = v29_lane + 64;
          float v332_data = glb_m1[(v29_lane + 64)];
          int32_t v338_a = v29_lane + 80;
          float v345_data = glb_m1[(v29_lane + 80)];
          int32_t v351_a = v29_lane + 96;
          float v358_data = glb_m1[(v29_lane + 96)];
          int32_t v364_a = v29_lane + 112;
          float v371_data = glb_m1[(v29_lane + 112)];
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v332_data, v319_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v345_data, v372_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v374_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v358_data, v373_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v371_data, v374_acc, 2, 1, 0);
          int32_t v381_a = v29_lane + 128;
          float v388_data = glb_m1[(v29_lane + 128)];
          int32_t v394_a = v29_lane + 144;
          float v401_data = glb_m1[(v29_lane + 144)];
          int32_t v407_a = v29_lane + 160;
          float v414_data = glb_m1[(v29_lane + 160)];
          int32_t v420_a = v29_lane + 176;
          float v427_data = glb_m1[(v29_lane + 176)];
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v388_data, v375_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v401_data, v428_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v414_data, v429_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v427_data, v430_acc, 2, 2, 0);
          int32_t v437_a = v29_lane + 192;
          float v444_data = glb_m1[(v29_lane + 192)];
          int32_t v450_a = v29_lane + 208;
          float v457_data = glb_m1[(v29_lane + 208)];
          int32_t v463_a = v29_lane + 224;
          float v470_data = glb_m1[(v29_lane + 224)];
          int32_t v476_a = v29_lane + 240;
          float v483_data = glb_m1[(v29_lane + 240)];
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v444_data, v431_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v457_data, v484_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v470_data, v485_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v483_data, v486_acc, 2, 3, 0);
          ir1[4] = (v487_acc[0]);
          ir1[5] = (v487_acc[1]);
          ir1[6] = (v487_acc[2]);
          ir1[7] = (v487_acc[3]);
          float v492_data = r0[8];
          float v493_data = r0[9];
          float v494_data = r0[10];
          float v495_data = r0[11];
          float v496_tp{};
          float v497_tp{};
          float v498_tp{};
          float v499_tp{};
          tensorforge::transpose4x4b32(v496_tp, v497_tp, v498_tp, v499_tp, v492_data, v493_data, v494_data, v495_data);
          tensorforge::VectorT<float, 4> v500_acc{};
          int32_t v506_a = v29_lane + 0;
          float v513_data = glb_m1[v29_lane];
          int32_t v519_a = v29_lane + 16;
          float v526_data = glb_m1[(v29_lane + 16)];
          int32_t v532_a = v29_lane + 32;
          float v539_data = glb_m1[(v29_lane + 32)];
          int32_t v545_a = v29_lane + 48;
          float v552_data = glb_m1[(v29_lane + 48)];
          tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v513_data, v500_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v526_data, v553_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v539_data, v554_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v552_data, v555_acc, 2, 0, 0);
          int32_t v562_a = v29_lane + 64;
          float v569_data = glb_m1[(v29_lane + 64)];
          int32_t v575_a = v29_lane + 80;
          float v582_data = glb_m1[(v29_lane + 80)];
          int32_t v588_a = v29_lane + 96;
          float v595_data = glb_m1[(v29_lane + 96)];
          int32_t v601_a = v29_lane + 112;
          float v608_data = glb_m1[(v29_lane + 112)];
          tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v569_data, v556_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v610_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v582_data, v609_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v595_data, v610_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v608_data, v611_acc, 2, 1, 0);
          int32_t v618_a = v29_lane + 128;
          float v625_data = glb_m1[(v29_lane + 128)];
          int32_t v631_a = v29_lane + 144;
          float v638_data = glb_m1[(v29_lane + 144)];
          int32_t v644_a = v29_lane + 160;
          float v651_data = glb_m1[(v29_lane + 160)];
          int32_t v657_a = v29_lane + 176;
          float v664_data = glb_m1[(v29_lane + 176)];
          tensorforge::VectorT<float, 4> v665_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v625_data, v612_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v666_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v638_data, v665_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v667_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v651_data, v666_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v668_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v664_data, v667_acc, 2, 2, 0);
          int32_t v674_a = v29_lane + 192;
          float v681_data = glb_m1[(v29_lane + 192)];
          int32_t v687_a = v29_lane + 208;
          float v694_data = glb_m1[(v29_lane + 208)];
          int32_t v700_a = v29_lane + 224;
          float v707_data = glb_m1[(v29_lane + 224)];
          int32_t v713_a = v29_lane + 240;
          float v720_data = glb_m1[(v29_lane + 240)];
          tensorforge::VectorT<float, 4> v721_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v496_tp, v681_data, v668_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v722_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v497_tp, v694_data, v721_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v723_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v498_tp, v707_data, v722_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v724_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v499_tp, v720_data, v723_acc, 2, 3, 0);
          ir1[8] = (v724_acc[0]);
          ir1[9] = (v724_acc[1]);
          ir1[10] = (v724_acc[2]);
          ir1[11] = (v724_acc[3]);
          float v729_data = r0[12];
          float v730_data = r0[13];
          float v731_data = r0[14];
          float v732_data = r0[15];
          float v733_tp{};
          float v734_tp{};
          float v735_tp{};
          float v736_tp{};
          tensorforge::transpose4x4b32(v733_tp, v734_tp, v735_tp, v736_tp, v729_data, v730_data, v731_data, v732_data);
          tensorforge::VectorT<float, 4> v737_acc{};
          int32_t v743_a = v29_lane + 0;
          float v750_data = glb_m1[v29_lane];
          int32_t v756_a = v29_lane + 16;
          float v763_data = glb_m1[(v29_lane + 16)];
          int32_t v769_a = v29_lane + 32;
          float v776_data = glb_m1[(v29_lane + 32)];
          int32_t v782_a = v29_lane + 48;
          float v789_data = glb_m1[(v29_lane + 48)];
          tensorforge::VectorT<float, 4> v790_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v733_tp, v750_data, v737_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v791_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v763_data, v790_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v792_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v776_data, v791_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v793_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v789_data, v792_acc, 2, 0, 0);
          int32_t v799_a = v29_lane + 64;
          float v806_data = glb_m1[(v29_lane + 64)];
          int32_t v812_a = v29_lane + 80;
          float v819_data = glb_m1[(v29_lane + 80)];
          int32_t v825_a = v29_lane + 96;
          float v832_data = glb_m1[(v29_lane + 96)];
          int32_t v838_a = v29_lane + 112;
          float v845_data = glb_m1[(v29_lane + 112)];
          tensorforge::VectorT<float, 4> v846_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v733_tp, v806_data, v793_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v847_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v819_data, v846_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v848_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v832_data, v847_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v849_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v845_data, v848_acc, 2, 1, 0);
          int32_t v855_a = v29_lane + 128;
          float v862_data = glb_m1[(v29_lane + 128)];
          int32_t v868_a = v29_lane + 144;
          float v875_data = glb_m1[(v29_lane + 144)];
          int32_t v881_a = v29_lane + 160;
          float v888_data = glb_m1[(v29_lane + 160)];
          int32_t v894_a = v29_lane + 176;
          float v901_data = glb_m1[(v29_lane + 176)];
          tensorforge::VectorT<float, 4> v902_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v733_tp, v862_data, v849_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v903_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v875_data, v902_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v904_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v888_data, v903_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v905_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v901_data, v904_acc, 2, 2, 0);
          int32_t v911_a = v29_lane + 192;
          float v918_data = glb_m1[(v29_lane + 192)];
          int32_t v924_a = v29_lane + 208;
          float v931_data = glb_m1[(v29_lane + 208)];
          int32_t v937_a = v29_lane + 224;
          float v944_data = glb_m1[(v29_lane + 224)];
          int32_t v950_a = v29_lane + 240;
          float v957_data = glb_m1[(v29_lane + 240)];
          tensorforge::VectorT<float, 4> v958_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v733_tp, v918_data, v905_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v959_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v931_data, v958_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v960_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v944_data, v959_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v961_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v957_data, v960_acc, 2, 3, 0);
          ir1[12] = (v961_acc[0]);
          ir1[13] = (v961_acc[1]);
          ir1[14] = (v961_acc[2]);
          ir1[15] = (v961_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v969_i0 = 0; v969_i0 < 1; ++v969_i0) {
            int32_t v978_lead = v29_lane + (v969_i0 * 16);
            #pragma unroll
            for (int32_t v970_i1 = 0; v970_i1 < 16; ++v970_i1) {
              int32_t v971_a = v969_i0 + v970_i1;
              float v973_data = r1[(v969_i0 + v970_i1)];
              int32_t v980_a = v978_lead + (v970_i1 * 16);
              glb_m0[v980_a] = v973_data;
            }
          }
          ;
        }
      }
    }
  }
}

