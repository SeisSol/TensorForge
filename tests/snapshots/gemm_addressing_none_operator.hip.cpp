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
          float v12_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v12_lin;
          float v13_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v13_lin;
          float v14_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v14_lin;
          float v15_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v15_lin;
          float v16_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v16_lin;
          float v17_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v17_lin;
          float v18_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v18_lin;
          float v19_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v19_lin;
          float v20_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v20_lin;
          float v21_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v21_lin;
          float v22_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v22_lin;
          float v23_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v23_lin;
          float v24_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v24_lin;
          float v25_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v25_lin;
          float v26_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v26_lin;
          float v27_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v27_lin;
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v29_data = r0[0];
          float v30_data = r0[1];
          float v31_data = r0[2];
          float v32_data = r0[3];
          float v33_tp{};
          float v34_tp{};
          float v35_tp{};
          float v36_tp{};
          tensorforge::transpose4x4b32(v33_tp, v34_tp, v35_tp, v36_tp, v29_data, v30_data, v31_data, v32_data);
          tensorforge::VectorT<float, 4> v37_acc{};
          int32_t v40_lane = threadIdx.x % 16;
          float v44_data = glb_m1[v40_lane];
          float v51_data = glb_m1[(v40_lane + 16)];
          float v58_data = glb_m1[(v40_lane + 32)];
          float v65_data = glb_m1[(v40_lane + 48)];
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v44_data, v37_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v51_data, v66_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v58_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v65_data, v68_acc, 2, 0, 0);
          float v76_data = glb_m1[(v40_lane + 64)];
          float v83_data = glb_m1[(v40_lane + 80)];
          float v90_data = glb_m1[(v40_lane + 96)];
          float v97_data = glb_m1[(v40_lane + 112)];
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v76_data, v69_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v83_data, v98_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v90_data, v99_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v97_data, v100_acc, 2, 1, 0);
          float v108_data = glb_m1[(v40_lane + 128)];
          float v115_data = glb_m1[(v40_lane + 144)];
          float v122_data = glb_m1[(v40_lane + 160)];
          float v129_data = glb_m1[(v40_lane + 176)];
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v108_data, v101_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v115_data, v130_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v122_data, v131_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v129_data, v132_acc, 2, 2, 0);
          float v140_data = glb_m1[(v40_lane + 192)];
          float v147_data = glb_m1[(v40_lane + 208)];
          float v154_data = glb_m1[(v40_lane + 224)];
          float v161_data = glb_m1[(v40_lane + 240)];
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v140_data, v133_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v147_data, v162_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v154_data, v163_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v161_data, v164_acc, 2, 3, 0);
          r1[0] = (v165_acc[0]);
          r1[1] = (v165_acc[1]);
          r1[2] = (v165_acc[2]);
          r1[3] = (v165_acc[3]);
          float v170_data = r0[4];
          float v171_data = r0[5];
          float v172_data = r0[6];
          float v173_data = r0[7];
          float v174_tp{};
          float v175_tp{};
          float v176_tp{};
          float v177_tp{};
          tensorforge::transpose4x4b32(v174_tp, v175_tp, v176_tp, v177_tp, v170_data, v171_data, v172_data, v173_data);
          tensorforge::VectorT<float, 4> v178_acc{};
          float v185_data = glb_m1[v40_lane];
          float v192_data = glb_m1[(v40_lane + 16)];
          float v199_data = glb_m1[(v40_lane + 32)];
          float v206_data = glb_m1[(v40_lane + 48)];
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v185_data, v178_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v192_data, v207_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v199_data, v208_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v206_data, v209_acc, 2, 0, 0);
          float v217_data = glb_m1[(v40_lane + 64)];
          float v224_data = glb_m1[(v40_lane + 80)];
          float v231_data = glb_m1[(v40_lane + 96)];
          float v238_data = glb_m1[(v40_lane + 112)];
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v217_data, v210_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v224_data, v239_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v231_data, v240_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v238_data, v241_acc, 2, 1, 0);
          float v249_data = glb_m1[(v40_lane + 128)];
          float v256_data = glb_m1[(v40_lane + 144)];
          float v263_data = glb_m1[(v40_lane + 160)];
          float v270_data = glb_m1[(v40_lane + 176)];
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v249_data, v242_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v256_data, v271_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v263_data, v272_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v270_data, v273_acc, 2, 2, 0);
          float v281_data = glb_m1[(v40_lane + 192)];
          float v288_data = glb_m1[(v40_lane + 208)];
          float v295_data = glb_m1[(v40_lane + 224)];
          float v302_data = glb_m1[(v40_lane + 240)];
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v281_data, v274_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v288_data, v303_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v295_data, v304_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v302_data, v305_acc, 2, 3, 0);
          r1[4] = (v306_acc[0]);
          r1[5] = (v306_acc[1]);
          r1[6] = (v306_acc[2]);
          r1[7] = (v306_acc[3]);
          float v311_data = r0[8];
          float v312_data = r0[9];
          float v313_data = r0[10];
          float v314_data = r0[11];
          float v315_tp{};
          float v316_tp{};
          float v317_tp{};
          float v318_tp{};
          tensorforge::transpose4x4b32(v315_tp, v316_tp, v317_tp, v318_tp, v311_data, v312_data, v313_data, v314_data);
          tensorforge::VectorT<float, 4> v319_acc{};
          float v326_data = glb_m1[v40_lane];
          float v333_data = glb_m1[(v40_lane + 16)];
          float v340_data = glb_m1[(v40_lane + 32)];
          float v347_data = glb_m1[(v40_lane + 48)];
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v326_data, v319_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v333_data, v348_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v340_data, v349_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v347_data, v350_acc, 2, 0, 0);
          float v358_data = glb_m1[(v40_lane + 64)];
          float v365_data = glb_m1[(v40_lane + 80)];
          float v372_data = glb_m1[(v40_lane + 96)];
          float v379_data = glb_m1[(v40_lane + 112)];
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v358_data, v351_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v365_data, v380_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v372_data, v381_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v379_data, v382_acc, 2, 1, 0);
          float v390_data = glb_m1[(v40_lane + 128)];
          float v397_data = glb_m1[(v40_lane + 144)];
          float v404_data = glb_m1[(v40_lane + 160)];
          float v411_data = glb_m1[(v40_lane + 176)];
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v390_data, v383_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v397_data, v412_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v404_data, v413_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v411_data, v414_acc, 2, 2, 0);
          float v422_data = glb_m1[(v40_lane + 192)];
          float v429_data = glb_m1[(v40_lane + 208)];
          float v436_data = glb_m1[(v40_lane + 224)];
          float v443_data = glb_m1[(v40_lane + 240)];
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v422_data, v415_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v429_data, v444_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v436_data, v445_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v443_data, v446_acc, 2, 3, 0);
          r1[8] = (v447_acc[0]);
          r1[9] = (v447_acc[1]);
          r1[10] = (v447_acc[2]);
          r1[11] = (v447_acc[3]);
          float v452_data = r0[12];
          float v453_data = r0[13];
          float v454_data = r0[14];
          float v455_data = r0[15];
          float v456_tp{};
          float v457_tp{};
          float v458_tp{};
          float v459_tp{};
          tensorforge::transpose4x4b32(v456_tp, v457_tp, v458_tp, v459_tp, v452_data, v453_data, v454_data, v455_data);
          tensorforge::VectorT<float, 4> v460_acc{};
          float v467_data = glb_m1[v40_lane];
          float v474_data = glb_m1[(v40_lane + 16)];
          float v481_data = glb_m1[(v40_lane + 32)];
          float v488_data = glb_m1[(v40_lane + 48)];
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v467_data, v460_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v474_data, v489_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v481_data, v490_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v488_data, v491_acc, 2, 0, 0);
          float v499_data = glb_m1[(v40_lane + 64)];
          float v506_data = glb_m1[(v40_lane + 80)];
          float v513_data = glb_m1[(v40_lane + 96)];
          float v520_data = glb_m1[(v40_lane + 112)];
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v499_data, v492_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v506_data, v521_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v513_data, v522_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v520_data, v523_acc, 2, 1, 0);
          float v531_data = glb_m1[(v40_lane + 128)];
          float v538_data = glb_m1[(v40_lane + 144)];
          float v545_data = glb_m1[(v40_lane + 160)];
          float v552_data = glb_m1[(v40_lane + 176)];
          tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v531_data, v524_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v538_data, v553_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v545_data, v554_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v552_data, v555_acc, 2, 2, 0);
          float v563_data = glb_m1[(v40_lane + 192)];
          float v570_data = glb_m1[(v40_lane + 208)];
          float v577_data = glb_m1[(v40_lane + 224)];
          float v584_data = glb_m1[(v40_lane + 240)];
          tensorforge::VectorT<float, 4> v585_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v563_data, v556_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v570_data, v585_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v587_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v577_data, v586_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v588_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v584_data, v587_acc, 2, 3, 0);
          r1[12] = (v588_acc[0]);
          r1[13] = (v588_acc[1]);
          r1[14] = (v588_acc[2]);
          r1[15] = (v588_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v596_i0 = 0; v596_i0 < 1; ++v596_i0) {
            int32_t v604_lead = v40_lane + (v596_i0 * 16);
            #pragma unroll
            for (int32_t v597_i1 = 0; v597_i1 < 16; ++v597_i1) {
              float v599_data = r1[(v596_i0 + v597_i1)];
              glb_m0[(v604_lead + (v597_i1 * 16))] = v599_data;
            }
          }
        }
      }
    }
  }
}

