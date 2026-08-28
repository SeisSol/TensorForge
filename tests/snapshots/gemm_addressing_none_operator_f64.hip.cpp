// === base name ===
kernel_3ff25cfed1

// === header ===
void launcher_kernel_3ff25cfed1(double* m0, unsigned m0_extraOffset, const double* m1, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3ff25cfed1(double* m0, unsigned m0_extraOffset, const double* m1, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3ff25cfed1, block.x * block.y * block.z, 512 * sizeof(double)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_3ff25cfed1), hipFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(double)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_3ff25cfed1, grid, block, 512 * sizeof(double), stream,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3ff25cfed1(double* m0, unsigned m0_extraOffset, const double* m1, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      double* tempShrMem = &localShrMem0[0];
      const double *const __restrict__ ptr_glb_m1 = &m1[0];
      double* __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m2);
          double v9_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v9_lin;
          double v10_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v10_lin;
          double v11_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v11_lin;
          double v12_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v12_lin;
          double v13_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v13_lin;
          double v14_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v14_lin;
          double v15_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v15_lin;
          double v16_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v16_lin;
          double v17_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v17_lin;
          double v18_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v18_lin;
          double v19_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v19_lin;
          double v20_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v20_lin;
          double v21_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v21_lin;
          double v22_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v22_lin;
          double v23_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v23_lin;
          double v24_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v24_lin;
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          int32_t v28_lane = threadIdx.x % 16;
          int32_t v31_a = v28_lane + 0;
          double v38_data = glb_m1[v28_lane];
          int32_t v44_a = v28_lane + 16;
          double v51_data = glb_m1[(v28_lane + 16)];
          int32_t v57_a = v28_lane + 32;
          double v64_data = glb_m1[(v28_lane + 32)];
          int32_t v70_a = v28_lane + 48;
          double v77_data = glb_m1[(v28_lane + 48)];
          int32_t v83_a = v28_lane + 64;
          double v90_data = glb_m1[(v28_lane + 64)];
          int32_t v96_a = v28_lane + 80;
          double v103_data = glb_m1[(v28_lane + 80)];
          int32_t v109_a = v28_lane + 96;
          double v116_data = glb_m1[(v28_lane + 96)];
          int32_t v122_a = v28_lane + 112;
          double v129_data = glb_m1[(v28_lane + 112)];
          int32_t v135_a = v28_lane + 128;
          double v142_data = glb_m1[(v28_lane + 128)];
          int32_t v148_a = v28_lane + 144;
          double v155_data = glb_m1[(v28_lane + 144)];
          int32_t v161_a = v28_lane + 160;
          double v168_data = glb_m1[(v28_lane + 160)];
          int32_t v174_a = v28_lane + 176;
          double v181_data = glb_m1[(v28_lane + 176)];
          int32_t v187_a = v28_lane + 192;
          double v194_data = glb_m1[(v28_lane + 192)];
          int32_t v200_a = v28_lane + 208;
          double v207_data = glb_m1[(v28_lane + 208)];
          int32_t v213_a = v28_lane + 224;
          double v220_data = glb_m1[(v28_lane + 224)];
          int32_t v226_a = v28_lane + 240;
          double v233_data = glb_m1[(v28_lane + 240)];
          double v234_acc{};
          double v235_acc{};
          double v236_acc{};
          double v237_acc{};
          double v238_acc{};
          double v239_acc{};
          double v240_acc{};
          double v241_acc{};
          double v242_acc{};
          double v243_acc{};
          double v244_acc{};
          double v245_acc{};
          double v246_acc{};
          double v247_acc{};
          double v248_acc{};
          double v249_acc{};
          double v250_data = r0[0];
          double v251_data = r0[1];
          double v252_data = r0[2];
          double v253_data = r0[3];
          double v254_data = r0[4];
          double v255_data = r0[5];
          double v256_data = r0[6];
          double v257_data = r0[7];
          double v258_data = r0[8];
          double v259_data = r0[9];
          double v260_data = r0[10];
          double v261_data = r0[11];
          double v262_data = r0[12];
          double v263_data = r0[13];
          double v264_data = r0[14];
          double v265_data = r0[15];
          tensorforge::fmacdpp16<0>(v234_acc, v250_data, v38_data);
          tensorforge::fmacdpp16<1>(v234_acc, v250_data, v51_data);
          tensorforge::fmacdpp16<2>(v234_acc, v250_data, v64_data);
          tensorforge::fmacdpp16<3>(v234_acc, v250_data, v77_data);
          tensorforge::fmacdpp16<4>(v234_acc, v250_data, v90_data);
          tensorforge::fmacdpp16<5>(v234_acc, v250_data, v103_data);
          tensorforge::fmacdpp16<6>(v234_acc, v250_data, v116_data);
          tensorforge::fmacdpp16<7>(v234_acc, v250_data, v129_data);
          tensorforge::fmacdpp16<8>(v234_acc, v250_data, v142_data);
          tensorforge::fmacdpp16<9>(v234_acc, v250_data, v155_data);
          tensorforge::fmacdpp16<10>(v234_acc, v250_data, v168_data);
          tensorforge::fmacdpp16<11>(v234_acc, v250_data, v181_data);
          tensorforge::fmacdpp16<12>(v234_acc, v250_data, v194_data);
          tensorforge::fmacdpp16<13>(v234_acc, v250_data, v207_data);
          tensorforge::fmacdpp16<14>(v234_acc, v250_data, v220_data);
          tensorforge::fmacdpp16<15>(v234_acc, v250_data, v233_data);
          tensorforge::fmacdpp16<0>(v235_acc, v251_data, v38_data);
          tensorforge::fmacdpp16<1>(v235_acc, v251_data, v51_data);
          tensorforge::fmacdpp16<2>(v235_acc, v251_data, v64_data);
          tensorforge::fmacdpp16<3>(v235_acc, v251_data, v77_data);
          tensorforge::fmacdpp16<4>(v235_acc, v251_data, v90_data);
          tensorforge::fmacdpp16<5>(v235_acc, v251_data, v103_data);
          tensorforge::fmacdpp16<6>(v235_acc, v251_data, v116_data);
          tensorforge::fmacdpp16<7>(v235_acc, v251_data, v129_data);
          tensorforge::fmacdpp16<8>(v235_acc, v251_data, v142_data);
          tensorforge::fmacdpp16<9>(v235_acc, v251_data, v155_data);
          tensorforge::fmacdpp16<10>(v235_acc, v251_data, v168_data);
          tensorforge::fmacdpp16<11>(v235_acc, v251_data, v181_data);
          tensorforge::fmacdpp16<12>(v235_acc, v251_data, v194_data);
          tensorforge::fmacdpp16<13>(v235_acc, v251_data, v207_data);
          tensorforge::fmacdpp16<14>(v235_acc, v251_data, v220_data);
          tensorforge::fmacdpp16<15>(v235_acc, v251_data, v233_data);
          tensorforge::fmacdpp16<0>(v236_acc, v252_data, v38_data);
          tensorforge::fmacdpp16<1>(v236_acc, v252_data, v51_data);
          tensorforge::fmacdpp16<2>(v236_acc, v252_data, v64_data);
          tensorforge::fmacdpp16<3>(v236_acc, v252_data, v77_data);
          tensorforge::fmacdpp16<4>(v236_acc, v252_data, v90_data);
          tensorforge::fmacdpp16<5>(v236_acc, v252_data, v103_data);
          tensorforge::fmacdpp16<6>(v236_acc, v252_data, v116_data);
          tensorforge::fmacdpp16<7>(v236_acc, v252_data, v129_data);
          tensorforge::fmacdpp16<8>(v236_acc, v252_data, v142_data);
          tensorforge::fmacdpp16<9>(v236_acc, v252_data, v155_data);
          tensorforge::fmacdpp16<10>(v236_acc, v252_data, v168_data);
          tensorforge::fmacdpp16<11>(v236_acc, v252_data, v181_data);
          tensorforge::fmacdpp16<12>(v236_acc, v252_data, v194_data);
          tensorforge::fmacdpp16<13>(v236_acc, v252_data, v207_data);
          tensorforge::fmacdpp16<14>(v236_acc, v252_data, v220_data);
          tensorforge::fmacdpp16<15>(v236_acc, v252_data, v233_data);
          tensorforge::fmacdpp16<0>(v237_acc, v253_data, v38_data);
          tensorforge::fmacdpp16<1>(v237_acc, v253_data, v51_data);
          tensorforge::fmacdpp16<2>(v237_acc, v253_data, v64_data);
          tensorforge::fmacdpp16<3>(v237_acc, v253_data, v77_data);
          tensorforge::fmacdpp16<4>(v237_acc, v253_data, v90_data);
          tensorforge::fmacdpp16<5>(v237_acc, v253_data, v103_data);
          tensorforge::fmacdpp16<6>(v237_acc, v253_data, v116_data);
          tensorforge::fmacdpp16<7>(v237_acc, v253_data, v129_data);
          tensorforge::fmacdpp16<8>(v237_acc, v253_data, v142_data);
          tensorforge::fmacdpp16<9>(v237_acc, v253_data, v155_data);
          tensorforge::fmacdpp16<10>(v237_acc, v253_data, v168_data);
          tensorforge::fmacdpp16<11>(v237_acc, v253_data, v181_data);
          tensorforge::fmacdpp16<12>(v237_acc, v253_data, v194_data);
          tensorforge::fmacdpp16<13>(v237_acc, v253_data, v207_data);
          tensorforge::fmacdpp16<14>(v237_acc, v253_data, v220_data);
          tensorforge::fmacdpp16<15>(v237_acc, v253_data, v233_data);
          tensorforge::fmacdpp16<0>(v238_acc, v254_data, v38_data);
          tensorforge::fmacdpp16<1>(v238_acc, v254_data, v51_data);
          tensorforge::fmacdpp16<2>(v238_acc, v254_data, v64_data);
          tensorforge::fmacdpp16<3>(v238_acc, v254_data, v77_data);
          tensorforge::fmacdpp16<4>(v238_acc, v254_data, v90_data);
          tensorforge::fmacdpp16<5>(v238_acc, v254_data, v103_data);
          tensorforge::fmacdpp16<6>(v238_acc, v254_data, v116_data);
          tensorforge::fmacdpp16<7>(v238_acc, v254_data, v129_data);
          tensorforge::fmacdpp16<8>(v238_acc, v254_data, v142_data);
          tensorforge::fmacdpp16<9>(v238_acc, v254_data, v155_data);
          tensorforge::fmacdpp16<10>(v238_acc, v254_data, v168_data);
          tensorforge::fmacdpp16<11>(v238_acc, v254_data, v181_data);
          tensorforge::fmacdpp16<12>(v238_acc, v254_data, v194_data);
          tensorforge::fmacdpp16<13>(v238_acc, v254_data, v207_data);
          tensorforge::fmacdpp16<14>(v238_acc, v254_data, v220_data);
          tensorforge::fmacdpp16<15>(v238_acc, v254_data, v233_data);
          tensorforge::fmacdpp16<0>(v239_acc, v255_data, v38_data);
          tensorforge::fmacdpp16<1>(v239_acc, v255_data, v51_data);
          tensorforge::fmacdpp16<2>(v239_acc, v255_data, v64_data);
          tensorforge::fmacdpp16<3>(v239_acc, v255_data, v77_data);
          tensorforge::fmacdpp16<4>(v239_acc, v255_data, v90_data);
          tensorforge::fmacdpp16<5>(v239_acc, v255_data, v103_data);
          tensorforge::fmacdpp16<6>(v239_acc, v255_data, v116_data);
          tensorforge::fmacdpp16<7>(v239_acc, v255_data, v129_data);
          tensorforge::fmacdpp16<8>(v239_acc, v255_data, v142_data);
          tensorforge::fmacdpp16<9>(v239_acc, v255_data, v155_data);
          tensorforge::fmacdpp16<10>(v239_acc, v255_data, v168_data);
          tensorforge::fmacdpp16<11>(v239_acc, v255_data, v181_data);
          tensorforge::fmacdpp16<12>(v239_acc, v255_data, v194_data);
          tensorforge::fmacdpp16<13>(v239_acc, v255_data, v207_data);
          tensorforge::fmacdpp16<14>(v239_acc, v255_data, v220_data);
          tensorforge::fmacdpp16<15>(v239_acc, v255_data, v233_data);
          tensorforge::fmacdpp16<0>(v240_acc, v256_data, v38_data);
          tensorforge::fmacdpp16<1>(v240_acc, v256_data, v51_data);
          tensorforge::fmacdpp16<2>(v240_acc, v256_data, v64_data);
          tensorforge::fmacdpp16<3>(v240_acc, v256_data, v77_data);
          tensorforge::fmacdpp16<4>(v240_acc, v256_data, v90_data);
          tensorforge::fmacdpp16<5>(v240_acc, v256_data, v103_data);
          tensorforge::fmacdpp16<6>(v240_acc, v256_data, v116_data);
          tensorforge::fmacdpp16<7>(v240_acc, v256_data, v129_data);
          tensorforge::fmacdpp16<8>(v240_acc, v256_data, v142_data);
          tensorforge::fmacdpp16<9>(v240_acc, v256_data, v155_data);
          tensorforge::fmacdpp16<10>(v240_acc, v256_data, v168_data);
          tensorforge::fmacdpp16<11>(v240_acc, v256_data, v181_data);
          tensorforge::fmacdpp16<12>(v240_acc, v256_data, v194_data);
          tensorforge::fmacdpp16<13>(v240_acc, v256_data, v207_data);
          tensorforge::fmacdpp16<14>(v240_acc, v256_data, v220_data);
          tensorforge::fmacdpp16<15>(v240_acc, v256_data, v233_data);
          tensorforge::fmacdpp16<0>(v241_acc, v257_data, v38_data);
          tensorforge::fmacdpp16<1>(v241_acc, v257_data, v51_data);
          tensorforge::fmacdpp16<2>(v241_acc, v257_data, v64_data);
          tensorforge::fmacdpp16<3>(v241_acc, v257_data, v77_data);
          tensorforge::fmacdpp16<4>(v241_acc, v257_data, v90_data);
          tensorforge::fmacdpp16<5>(v241_acc, v257_data, v103_data);
          tensorforge::fmacdpp16<6>(v241_acc, v257_data, v116_data);
          tensorforge::fmacdpp16<7>(v241_acc, v257_data, v129_data);
          tensorforge::fmacdpp16<8>(v241_acc, v257_data, v142_data);
          tensorforge::fmacdpp16<9>(v241_acc, v257_data, v155_data);
          tensorforge::fmacdpp16<10>(v241_acc, v257_data, v168_data);
          tensorforge::fmacdpp16<11>(v241_acc, v257_data, v181_data);
          tensorforge::fmacdpp16<12>(v241_acc, v257_data, v194_data);
          tensorforge::fmacdpp16<13>(v241_acc, v257_data, v207_data);
          tensorforge::fmacdpp16<14>(v241_acc, v257_data, v220_data);
          tensorforge::fmacdpp16<15>(v241_acc, v257_data, v233_data);
          tensorforge::fmacdpp16<0>(v242_acc, v258_data, v38_data);
          tensorforge::fmacdpp16<1>(v242_acc, v258_data, v51_data);
          tensorforge::fmacdpp16<2>(v242_acc, v258_data, v64_data);
          tensorforge::fmacdpp16<3>(v242_acc, v258_data, v77_data);
          tensorforge::fmacdpp16<4>(v242_acc, v258_data, v90_data);
          tensorforge::fmacdpp16<5>(v242_acc, v258_data, v103_data);
          tensorforge::fmacdpp16<6>(v242_acc, v258_data, v116_data);
          tensorforge::fmacdpp16<7>(v242_acc, v258_data, v129_data);
          tensorforge::fmacdpp16<8>(v242_acc, v258_data, v142_data);
          tensorforge::fmacdpp16<9>(v242_acc, v258_data, v155_data);
          tensorforge::fmacdpp16<10>(v242_acc, v258_data, v168_data);
          tensorforge::fmacdpp16<11>(v242_acc, v258_data, v181_data);
          tensorforge::fmacdpp16<12>(v242_acc, v258_data, v194_data);
          tensorforge::fmacdpp16<13>(v242_acc, v258_data, v207_data);
          tensorforge::fmacdpp16<14>(v242_acc, v258_data, v220_data);
          tensorforge::fmacdpp16<15>(v242_acc, v258_data, v233_data);
          tensorforge::fmacdpp16<0>(v243_acc, v259_data, v38_data);
          tensorforge::fmacdpp16<1>(v243_acc, v259_data, v51_data);
          tensorforge::fmacdpp16<2>(v243_acc, v259_data, v64_data);
          tensorforge::fmacdpp16<3>(v243_acc, v259_data, v77_data);
          tensorforge::fmacdpp16<4>(v243_acc, v259_data, v90_data);
          tensorforge::fmacdpp16<5>(v243_acc, v259_data, v103_data);
          tensorforge::fmacdpp16<6>(v243_acc, v259_data, v116_data);
          tensorforge::fmacdpp16<7>(v243_acc, v259_data, v129_data);
          tensorforge::fmacdpp16<8>(v243_acc, v259_data, v142_data);
          tensorforge::fmacdpp16<9>(v243_acc, v259_data, v155_data);
          tensorforge::fmacdpp16<10>(v243_acc, v259_data, v168_data);
          tensorforge::fmacdpp16<11>(v243_acc, v259_data, v181_data);
          tensorforge::fmacdpp16<12>(v243_acc, v259_data, v194_data);
          tensorforge::fmacdpp16<13>(v243_acc, v259_data, v207_data);
          tensorforge::fmacdpp16<14>(v243_acc, v259_data, v220_data);
          tensorforge::fmacdpp16<15>(v243_acc, v259_data, v233_data);
          tensorforge::fmacdpp16<0>(v244_acc, v260_data, v38_data);
          tensorforge::fmacdpp16<1>(v244_acc, v260_data, v51_data);
          tensorforge::fmacdpp16<2>(v244_acc, v260_data, v64_data);
          tensorforge::fmacdpp16<3>(v244_acc, v260_data, v77_data);
          tensorforge::fmacdpp16<4>(v244_acc, v260_data, v90_data);
          tensorforge::fmacdpp16<5>(v244_acc, v260_data, v103_data);
          tensorforge::fmacdpp16<6>(v244_acc, v260_data, v116_data);
          tensorforge::fmacdpp16<7>(v244_acc, v260_data, v129_data);
          tensorforge::fmacdpp16<8>(v244_acc, v260_data, v142_data);
          tensorforge::fmacdpp16<9>(v244_acc, v260_data, v155_data);
          tensorforge::fmacdpp16<10>(v244_acc, v260_data, v168_data);
          tensorforge::fmacdpp16<11>(v244_acc, v260_data, v181_data);
          tensorforge::fmacdpp16<12>(v244_acc, v260_data, v194_data);
          tensorforge::fmacdpp16<13>(v244_acc, v260_data, v207_data);
          tensorforge::fmacdpp16<14>(v244_acc, v260_data, v220_data);
          tensorforge::fmacdpp16<15>(v244_acc, v260_data, v233_data);
          tensorforge::fmacdpp16<0>(v245_acc, v261_data, v38_data);
          tensorforge::fmacdpp16<1>(v245_acc, v261_data, v51_data);
          tensorforge::fmacdpp16<2>(v245_acc, v261_data, v64_data);
          tensorforge::fmacdpp16<3>(v245_acc, v261_data, v77_data);
          tensorforge::fmacdpp16<4>(v245_acc, v261_data, v90_data);
          tensorforge::fmacdpp16<5>(v245_acc, v261_data, v103_data);
          tensorforge::fmacdpp16<6>(v245_acc, v261_data, v116_data);
          tensorforge::fmacdpp16<7>(v245_acc, v261_data, v129_data);
          tensorforge::fmacdpp16<8>(v245_acc, v261_data, v142_data);
          tensorforge::fmacdpp16<9>(v245_acc, v261_data, v155_data);
          tensorforge::fmacdpp16<10>(v245_acc, v261_data, v168_data);
          tensorforge::fmacdpp16<11>(v245_acc, v261_data, v181_data);
          tensorforge::fmacdpp16<12>(v245_acc, v261_data, v194_data);
          tensorforge::fmacdpp16<13>(v245_acc, v261_data, v207_data);
          tensorforge::fmacdpp16<14>(v245_acc, v261_data, v220_data);
          tensorforge::fmacdpp16<15>(v245_acc, v261_data, v233_data);
          tensorforge::fmacdpp16<0>(v246_acc, v262_data, v38_data);
          tensorforge::fmacdpp16<1>(v246_acc, v262_data, v51_data);
          tensorforge::fmacdpp16<2>(v246_acc, v262_data, v64_data);
          tensorforge::fmacdpp16<3>(v246_acc, v262_data, v77_data);
          tensorforge::fmacdpp16<4>(v246_acc, v262_data, v90_data);
          tensorforge::fmacdpp16<5>(v246_acc, v262_data, v103_data);
          tensorforge::fmacdpp16<6>(v246_acc, v262_data, v116_data);
          tensorforge::fmacdpp16<7>(v246_acc, v262_data, v129_data);
          tensorforge::fmacdpp16<8>(v246_acc, v262_data, v142_data);
          tensorforge::fmacdpp16<9>(v246_acc, v262_data, v155_data);
          tensorforge::fmacdpp16<10>(v246_acc, v262_data, v168_data);
          tensorforge::fmacdpp16<11>(v246_acc, v262_data, v181_data);
          tensorforge::fmacdpp16<12>(v246_acc, v262_data, v194_data);
          tensorforge::fmacdpp16<13>(v246_acc, v262_data, v207_data);
          tensorforge::fmacdpp16<14>(v246_acc, v262_data, v220_data);
          tensorforge::fmacdpp16<15>(v246_acc, v262_data, v233_data);
          tensorforge::fmacdpp16<0>(v247_acc, v263_data, v38_data);
          tensorforge::fmacdpp16<1>(v247_acc, v263_data, v51_data);
          tensorforge::fmacdpp16<2>(v247_acc, v263_data, v64_data);
          tensorforge::fmacdpp16<3>(v247_acc, v263_data, v77_data);
          tensorforge::fmacdpp16<4>(v247_acc, v263_data, v90_data);
          tensorforge::fmacdpp16<5>(v247_acc, v263_data, v103_data);
          tensorforge::fmacdpp16<6>(v247_acc, v263_data, v116_data);
          tensorforge::fmacdpp16<7>(v247_acc, v263_data, v129_data);
          tensorforge::fmacdpp16<8>(v247_acc, v263_data, v142_data);
          tensorforge::fmacdpp16<9>(v247_acc, v263_data, v155_data);
          tensorforge::fmacdpp16<10>(v247_acc, v263_data, v168_data);
          tensorforge::fmacdpp16<11>(v247_acc, v263_data, v181_data);
          tensorforge::fmacdpp16<12>(v247_acc, v263_data, v194_data);
          tensorforge::fmacdpp16<13>(v247_acc, v263_data, v207_data);
          tensorforge::fmacdpp16<14>(v247_acc, v263_data, v220_data);
          tensorforge::fmacdpp16<15>(v247_acc, v263_data, v233_data);
          tensorforge::fmacdpp16<0>(v248_acc, v264_data, v38_data);
          tensorforge::fmacdpp16<1>(v248_acc, v264_data, v51_data);
          tensorforge::fmacdpp16<2>(v248_acc, v264_data, v64_data);
          tensorforge::fmacdpp16<3>(v248_acc, v264_data, v77_data);
          tensorforge::fmacdpp16<4>(v248_acc, v264_data, v90_data);
          tensorforge::fmacdpp16<5>(v248_acc, v264_data, v103_data);
          tensorforge::fmacdpp16<6>(v248_acc, v264_data, v116_data);
          tensorforge::fmacdpp16<7>(v248_acc, v264_data, v129_data);
          tensorforge::fmacdpp16<8>(v248_acc, v264_data, v142_data);
          tensorforge::fmacdpp16<9>(v248_acc, v264_data, v155_data);
          tensorforge::fmacdpp16<10>(v248_acc, v264_data, v168_data);
          tensorforge::fmacdpp16<11>(v248_acc, v264_data, v181_data);
          tensorforge::fmacdpp16<12>(v248_acc, v264_data, v194_data);
          tensorforge::fmacdpp16<13>(v248_acc, v264_data, v207_data);
          tensorforge::fmacdpp16<14>(v248_acc, v264_data, v220_data);
          tensorforge::fmacdpp16<15>(v248_acc, v264_data, v233_data);
          tensorforge::fmacdpp16<0>(v249_acc, v265_data, v38_data);
          tensorforge::fmacdpp16<1>(v249_acc, v265_data, v51_data);
          tensorforge::fmacdpp16<2>(v249_acc, v265_data, v64_data);
          tensorforge::fmacdpp16<3>(v249_acc, v265_data, v77_data);
          tensorforge::fmacdpp16<4>(v249_acc, v265_data, v90_data);
          tensorforge::fmacdpp16<5>(v249_acc, v265_data, v103_data);
          tensorforge::fmacdpp16<6>(v249_acc, v265_data, v116_data);
          tensorforge::fmacdpp16<7>(v249_acc, v265_data, v129_data);
          tensorforge::fmacdpp16<8>(v249_acc, v265_data, v142_data);
          tensorforge::fmacdpp16<9>(v249_acc, v265_data, v155_data);
          tensorforge::fmacdpp16<10>(v249_acc, v265_data, v168_data);
          tensorforge::fmacdpp16<11>(v249_acc, v265_data, v181_data);
          tensorforge::fmacdpp16<12>(v249_acc, v265_data, v194_data);
          tensorforge::fmacdpp16<13>(v249_acc, v265_data, v207_data);
          tensorforge::fmacdpp16<14>(v249_acc, v265_data, v220_data);
          tensorforge::fmacdpp16<15>(v249_acc, v265_data, v233_data);
          r1[0] = v234_acc;
          r1[1] = v235_acc;
          r1[2] = v236_acc;
          r1[3] = v237_acc;
          r1[4] = v238_acc;
          r1[5] = v239_acc;
          r1[6] = v240_acc;
          r1[7] = v241_acc;
          r1[8] = v242_acc;
          r1[9] = v243_acc;
          r1[10] = v244_acc;
          r1[11] = v245_acc;
          r1[12] = v246_acc;
          r1[13] = v247_acc;
          r1[14] = v248_acc;
          r1[15] = v249_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v269_i0 = 0; v269_i0 < 1; ++v269_i0) {
            int32_t v278_lead = v28_lane + (v269_i0 * 16);
            #pragma unroll
            for (int32_t v270_i1 = 0; v270_i1 < 16; ++v270_i1) {
              int32_t v271_a = v269_i0 + v270_i1;
              double v273_data = r1[(v269_i0 + v270_i1)];
              glb_m0[(v278_lead + (v270_i1 * 16))] = v273_data;
            }
          }
        }
      }
    }
  }
}

