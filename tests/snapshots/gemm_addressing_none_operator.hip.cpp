// === base name ===
kernel_162acbbbcd8cffa6

// === header ===
void launcher_kernel_162acbbbcd8cffa6(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_162acbbbcd8cffa6(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_162acbbbcd8cffa6, block.x * block.y * block.z, 512 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_162acbbbcd8cffa6), hipFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_162acbbbcd8cffa6, grid, block, 512 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_162acbbbcd8cffa6(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
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
          int32_t v14_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
            int32_t v21_lead = v14_lead + (v15_i0 * 16);
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 16; ++v16_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m2[(v21_lead + (v16_i1 * 16))]);
              r0[(v15_i0 + v16_i1)] = v24_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v27_data = r0[0];
          float v28_data = r0[1];
          float v29_data = r0[2];
          float v30_data = r0[3];
          float v31_tp{};
          float v32_tp{};
          float v33_tp{};
          float v34_tp{};
          tensorforge::transpose4x4b32(v31_tp, v32_tp, v33_tp, v34_tp, v27_data, v28_data, v29_data, v30_data);
          tensorforge::VectorT<float, 4> v35_acc{};
          float v42_data = glb_m1[v14_lead];
          float v49_data = glb_m1[(v14_lead + 16)];
          float v56_data = glb_m1[(v14_lead + 32)];
          float v63_data = glb_m1[(v14_lead + 48)];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v42_data, v35_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v49_data, v64_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v56_data, v65_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v63_data, v66_acc, 2, 0, 0);
          float v74_data = glb_m1[(v14_lead + 64)];
          float v81_data = glb_m1[(v14_lead + 80)];
          float v88_data = glb_m1[(v14_lead + 96)];
          float v95_data = glb_m1[(v14_lead + 112)];
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v74_data, v67_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v81_data, v96_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v88_data, v97_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v95_data, v98_acc, 2, 1, 0);
          float v106_data = glb_m1[(v14_lead + 128)];
          float v113_data = glb_m1[(v14_lead + 144)];
          float v120_data = glb_m1[(v14_lead + 160)];
          float v127_data = glb_m1[(v14_lead + 176)];
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v106_data, v99_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v113_data, v128_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v120_data, v129_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v127_data, v130_acc, 2, 2, 0);
          float v138_data = glb_m1[(v14_lead + 192)];
          float v145_data = glb_m1[(v14_lead + 208)];
          float v152_data = glb_m1[(v14_lead + 224)];
          float v159_data = glb_m1[(v14_lead + 240)];
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v138_data, v131_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v145_data, v160_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v152_data, v161_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v159_data, v162_acc, 2, 3, 0);
          r1[0] = (v163_acc[0]);
          r1[1] = (v163_acc[1]);
          r1[2] = (v163_acc[2]);
          r1[3] = (v163_acc[3]);
          float v168_data = r0[4];
          float v169_data = r0[5];
          float v170_data = r0[6];
          float v171_data = r0[7];
          float v172_tp{};
          float v173_tp{};
          float v174_tp{};
          float v175_tp{};
          tensorforge::transpose4x4b32(v172_tp, v173_tp, v174_tp, v175_tp, v168_data, v169_data, v170_data, v171_data);
          tensorforge::VectorT<float, 4> v176_acc{};
          float v183_data = glb_m1[v14_lead];
          float v190_data = glb_m1[(v14_lead + 16)];
          float v197_data = glb_m1[(v14_lead + 32)];
          float v204_data = glb_m1[(v14_lead + 48)];
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v183_data, v176_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v190_data, v205_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v197_data, v206_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v204_data, v207_acc, 2, 0, 0);
          float v215_data = glb_m1[(v14_lead + 64)];
          float v222_data = glb_m1[(v14_lead + 80)];
          float v229_data = glb_m1[(v14_lead + 96)];
          float v236_data = glb_m1[(v14_lead + 112)];
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v215_data, v208_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v222_data, v237_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v229_data, v238_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v236_data, v239_acc, 2, 1, 0);
          float v247_data = glb_m1[(v14_lead + 128)];
          float v254_data = glb_m1[(v14_lead + 144)];
          float v261_data = glb_m1[(v14_lead + 160)];
          float v268_data = glb_m1[(v14_lead + 176)];
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v247_data, v240_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v254_data, v269_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v261_data, v270_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v268_data, v271_acc, 2, 2, 0);
          float v279_data = glb_m1[(v14_lead + 192)];
          float v286_data = glb_m1[(v14_lead + 208)];
          float v293_data = glb_m1[(v14_lead + 224)];
          float v300_data = glb_m1[(v14_lead + 240)];
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v279_data, v272_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v286_data, v301_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v293_data, v302_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v175_tp, v300_data, v303_acc, 2, 3, 0);
          r1[4] = (v304_acc[0]);
          r1[5] = (v304_acc[1]);
          r1[6] = (v304_acc[2]);
          r1[7] = (v304_acc[3]);
          float v309_data = r0[8];
          float v310_data = r0[9];
          float v311_data = r0[10];
          float v312_data = r0[11];
          float v313_tp{};
          float v314_tp{};
          float v315_tp{};
          float v316_tp{};
          tensorforge::transpose4x4b32(v313_tp, v314_tp, v315_tp, v316_tp, v309_data, v310_data, v311_data, v312_data);
          tensorforge::VectorT<float, 4> v317_acc{};
          float v324_data = glb_m1[v14_lead];
          float v331_data = glb_m1[(v14_lead + 16)];
          float v338_data = glb_m1[(v14_lead + 32)];
          float v345_data = glb_m1[(v14_lead + 48)];
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v324_data, v317_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v331_data, v346_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v338_data, v347_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v345_data, v348_acc, 2, 0, 0);
          float v356_data = glb_m1[(v14_lead + 64)];
          float v363_data = glb_m1[(v14_lead + 80)];
          float v370_data = glb_m1[(v14_lead + 96)];
          float v377_data = glb_m1[(v14_lead + 112)];
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v356_data, v349_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v363_data, v378_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v370_data, v379_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v377_data, v380_acc, 2, 1, 0);
          float v388_data = glb_m1[(v14_lead + 128)];
          float v395_data = glb_m1[(v14_lead + 144)];
          float v402_data = glb_m1[(v14_lead + 160)];
          float v409_data = glb_m1[(v14_lead + 176)];
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v388_data, v381_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v395_data, v410_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v402_data, v411_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v409_data, v412_acc, 2, 2, 0);
          float v420_data = glb_m1[(v14_lead + 192)];
          float v427_data = glb_m1[(v14_lead + 208)];
          float v434_data = glb_m1[(v14_lead + 224)];
          float v441_data = glb_m1[(v14_lead + 240)];
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v420_data, v413_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v427_data, v442_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v434_data, v443_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v316_tp, v441_data, v444_acc, 2, 3, 0);
          r1[8] = (v445_acc[0]);
          r1[9] = (v445_acc[1]);
          r1[10] = (v445_acc[2]);
          r1[11] = (v445_acc[3]);
          float v450_data = r0[12];
          float v451_data = r0[13];
          float v452_data = r0[14];
          float v453_data = r0[15];
          float v454_tp{};
          float v455_tp{};
          float v456_tp{};
          float v457_tp{};
          tensorforge::transpose4x4b32(v454_tp, v455_tp, v456_tp, v457_tp, v450_data, v451_data, v452_data, v453_data);
          tensorforge::VectorT<float, 4> v458_acc{};
          float v465_data = glb_m1[v14_lead];
          float v472_data = glb_m1[(v14_lead + 16)];
          float v479_data = glb_m1[(v14_lead + 32)];
          float v486_data = glb_m1[(v14_lead + 48)];
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v465_data, v458_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v472_data, v487_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v479_data, v488_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v486_data, v489_acc, 2, 0, 0);
          float v497_data = glb_m1[(v14_lead + 64)];
          float v504_data = glb_m1[(v14_lead + 80)];
          float v511_data = glb_m1[(v14_lead + 96)];
          float v518_data = glb_m1[(v14_lead + 112)];
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v497_data, v490_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v504_data, v519_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v511_data, v520_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v518_data, v521_acc, 2, 1, 0);
          float v529_data = glb_m1[(v14_lead + 128)];
          float v536_data = glb_m1[(v14_lead + 144)];
          float v543_data = glb_m1[(v14_lead + 160)];
          float v550_data = glb_m1[(v14_lead + 176)];
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v529_data, v522_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v536_data, v551_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v543_data, v552_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v550_data, v553_acc, 2, 2, 0);
          float v561_data = glb_m1[(v14_lead + 192)];
          float v568_data = glb_m1[(v14_lead + 208)];
          float v575_data = glb_m1[(v14_lead + 224)];
          float v582_data = glb_m1[(v14_lead + 240)];
          tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v561_data, v554_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v568_data, v583_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v585_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v575_data, v584_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v457_tp, v582_data, v585_acc, 2, 3, 0);
          r1[12] = (v586_acc[0]);
          r1[13] = (v586_acc[1]);
          r1[14] = (v586_acc[2]);
          r1[15] = (v586_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v594_i0 = 0; v594_i0 < 1; ++v594_i0) {
            int32_t v602_lead = v14_lead + (v594_i0 * 16);
            #pragma unroll
            for (int32_t v595_i1 = 0; v595_i1 < 16; ++v595_i1) {
              float v597_data = r1[(v594_i0 + v595_i1)];
              glb_m0[(v602_lead + (v595_i1 * 16))] = v597_data;
            }
          }
        }
      }
    }
  }
}

