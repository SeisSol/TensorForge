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
          float v41_data = glb_m1[v37_lane];
          float v48_data = glb_m1[(v37_lane + 16)];
          float v55_data = glb_m1[(v37_lane + 32)];
          float v62_data = glb_m1[(v37_lane + 48)];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v41_data, v34_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v48_data, v63_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v55_data, v64_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v62_data, v65_acc, 2, 0, 0);
          float v73_data = glb_m1[(v37_lane + 64)];
          float v80_data = glb_m1[(v37_lane + 80)];
          float v87_data = glb_m1[(v37_lane + 96)];
          float v94_data = glb_m1[(v37_lane + 112)];
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v73_data, v66_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v80_data, v95_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v87_data, v96_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v94_data, v97_acc, 2, 1, 0);
          float v105_data = glb_m1[(v37_lane + 128)];
          float v112_data = glb_m1[(v37_lane + 144)];
          float v119_data = glb_m1[(v37_lane + 160)];
          float v126_data = glb_m1[(v37_lane + 176)];
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v105_data, v98_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v112_data, v127_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v119_data, v128_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v126_data, v129_acc, 2, 2, 0);
          float v137_data = glb_m1[(v37_lane + 192)];
          float v144_data = glb_m1[(v37_lane + 208)];
          float v151_data = glb_m1[(v37_lane + 224)];
          float v158_data = glb_m1[(v37_lane + 240)];
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v137_data, v130_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v144_data, v159_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v151_data, v160_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v158_data, v161_acc, 2, 3, 0);
          r1[0] = (v162_acc[0]);
          r1[1] = (v162_acc[1]);
          r1[2] = (v162_acc[2]);
          r1[3] = (v162_acc[3]);
          float v167_data = r0[4];
          float v168_data = r0[5];
          float v169_data = r0[6];
          float v170_data = r0[7];
          float v171_tp{};
          float v172_tp{};
          float v173_tp{};
          float v174_tp{};
          tensorforge::transpose4x4b32(v171_tp, v172_tp, v173_tp, v174_tp, v167_data, v168_data, v169_data, v170_data);
          tensorforge::VectorT<float, 4> v175_acc{};
          float v182_data = glb_m1[v37_lane];
          float v189_data = glb_m1[(v37_lane + 16)];
          float v196_data = glb_m1[(v37_lane + 32)];
          float v203_data = glb_m1[(v37_lane + 48)];
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v182_data, v175_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v189_data, v204_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v196_data, v205_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v203_data, v206_acc, 2, 0, 0);
          float v214_data = glb_m1[(v37_lane + 64)];
          float v221_data = glb_m1[(v37_lane + 80)];
          float v228_data = glb_m1[(v37_lane + 96)];
          float v235_data = glb_m1[(v37_lane + 112)];
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v214_data, v207_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v221_data, v236_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v228_data, v237_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v235_data, v238_acc, 2, 1, 0);
          float v246_data = glb_m1[(v37_lane + 128)];
          float v253_data = glb_m1[(v37_lane + 144)];
          float v260_data = glb_m1[(v37_lane + 160)];
          float v267_data = glb_m1[(v37_lane + 176)];
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v246_data, v239_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v253_data, v268_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v260_data, v269_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v267_data, v270_acc, 2, 2, 0);
          float v278_data = glb_m1[(v37_lane + 192)];
          float v285_data = glb_m1[(v37_lane + 208)];
          float v292_data = glb_m1[(v37_lane + 224)];
          float v299_data = glb_m1[(v37_lane + 240)];
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v171_tp, v278_data, v271_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v172_tp, v285_data, v300_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v173_tp, v292_data, v301_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v174_tp, v299_data, v302_acc, 2, 3, 0);
          r1[4] = (v303_acc[0]);
          r1[5] = (v303_acc[1]);
          r1[6] = (v303_acc[2]);
          r1[7] = (v303_acc[3]);
          float v308_data = r0[8];
          float v309_data = r0[9];
          float v310_data = r0[10];
          float v311_data = r0[11];
          float v312_tp{};
          float v313_tp{};
          float v314_tp{};
          float v315_tp{};
          tensorforge::transpose4x4b32(v312_tp, v313_tp, v314_tp, v315_tp, v308_data, v309_data, v310_data, v311_data);
          tensorforge::VectorT<float, 4> v316_acc{};
          float v323_data = glb_m1[v37_lane];
          float v330_data = glb_m1[(v37_lane + 16)];
          float v337_data = glb_m1[(v37_lane + 32)];
          float v344_data = glb_m1[(v37_lane + 48)];
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v323_data, v316_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v330_data, v345_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v337_data, v346_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v344_data, v347_acc, 2, 0, 0);
          float v355_data = glb_m1[(v37_lane + 64)];
          float v362_data = glb_m1[(v37_lane + 80)];
          float v369_data = glb_m1[(v37_lane + 96)];
          float v376_data = glb_m1[(v37_lane + 112)];
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v355_data, v348_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v362_data, v377_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v369_data, v378_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v376_data, v379_acc, 2, 1, 0);
          float v387_data = glb_m1[(v37_lane + 128)];
          float v394_data = glb_m1[(v37_lane + 144)];
          float v401_data = glb_m1[(v37_lane + 160)];
          float v408_data = glb_m1[(v37_lane + 176)];
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v387_data, v380_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v394_data, v409_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v401_data, v410_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v408_data, v411_acc, 2, 2, 0);
          float v419_data = glb_m1[(v37_lane + 192)];
          float v426_data = glb_m1[(v37_lane + 208)];
          float v433_data = glb_m1[(v37_lane + 224)];
          float v440_data = glb_m1[(v37_lane + 240)];
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v312_tp, v419_data, v412_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v313_tp, v426_data, v441_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v314_tp, v433_data, v442_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v315_tp, v440_data, v443_acc, 2, 3, 0);
          r1[8] = (v444_acc[0]);
          r1[9] = (v444_acc[1]);
          r1[10] = (v444_acc[2]);
          r1[11] = (v444_acc[3]);
          float v449_data = r0[12];
          float v450_data = r0[13];
          float v451_data = r0[14];
          float v452_data = r0[15];
          float v453_tp{};
          float v454_tp{};
          float v455_tp{};
          float v456_tp{};
          tensorforge::transpose4x4b32(v453_tp, v454_tp, v455_tp, v456_tp, v449_data, v450_data, v451_data, v452_data);
          tensorforge::VectorT<float, 4> v457_acc{};
          float v464_data = glb_m1[v37_lane];
          float v471_data = glb_m1[(v37_lane + 16)];
          float v478_data = glb_m1[(v37_lane + 32)];
          float v485_data = glb_m1[(v37_lane + 48)];
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v464_data, v457_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v471_data, v486_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v478_data, v487_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v485_data, v488_acc, 2, 0, 0);
          float v496_data = glb_m1[(v37_lane + 64)];
          float v503_data = glb_m1[(v37_lane + 80)];
          float v510_data = glb_m1[(v37_lane + 96)];
          float v517_data = glb_m1[(v37_lane + 112)];
          tensorforge::VectorT<float, 4> v518_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v496_data, v489_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v503_data, v518_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v510_data, v519_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v517_data, v520_acc, 2, 1, 0);
          float v528_data = glb_m1[(v37_lane + 128)];
          float v535_data = glb_m1[(v37_lane + 144)];
          float v542_data = glb_m1[(v37_lane + 160)];
          float v549_data = glb_m1[(v37_lane + 176)];
          tensorforge::VectorT<float, 4> v550_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v528_data, v521_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v535_data, v550_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v542_data, v551_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v549_data, v552_acc, 2, 2, 0);
          float v560_data = glb_m1[(v37_lane + 192)];
          float v567_data = glb_m1[(v37_lane + 208)];
          float v574_data = glb_m1[(v37_lane + 224)];
          float v581_data = glb_m1[(v37_lane + 240)];
          tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v453_tp, v560_data, v553_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v454_tp, v567_data, v582_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v455_tp, v574_data, v583_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v585_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v456_tp, v581_data, v584_acc, 2, 3, 0);
          r1[12] = (v585_acc[0]);
          r1[13] = (v585_acc[1]);
          r1[14] = (v585_acc[2]);
          r1[15] = (v585_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v593_i0 = 0; v593_i0 < 1; ++v593_i0) {
            int32_t v601_lead = v37_lane + (v593_i0 * 16);
            #pragma unroll
            for (int32_t v594_i1 = 0; v594_i1 < 16; ++v594_i1) {
              float v596_data = r1[(v593_i0 + v594_i1)];
              glb_m0[(v601_lead + (v594_i1 * 16))] = v596_data;
            }
          }
        }
      }
    }
  }
}

