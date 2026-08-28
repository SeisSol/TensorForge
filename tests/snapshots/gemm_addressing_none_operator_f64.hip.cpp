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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m2);
          double v6_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v6_lin;
          double v7_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v7_lin;
          double v8_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v8_lin;
          double v9_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v9_lin;
          double v10_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v10_lin;
          double v11_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v11_lin;
          double v12_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v12_lin;
          double v13_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v13_lin;
          double v14_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v14_lin;
          double v15_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v15_lin;
          double v16_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v16_lin;
          double v17_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v17_lin;
          double v18_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v18_lin;
          double v19_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v19_lin;
          double v20_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v20_lin;
          double v21_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v21_lin;
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          int32_t v25_lane = threadIdx.x % 16;
          int32_t v28_a = v25_lane + 0;
          double v35_data = glb_m1[v25_lane];
          int32_t v41_a = v25_lane + 16;
          double v48_data = glb_m1[(v25_lane + 16)];
          int32_t v54_a = v25_lane + 32;
          double v61_data = glb_m1[(v25_lane + 32)];
          int32_t v67_a = v25_lane + 48;
          double v74_data = glb_m1[(v25_lane + 48)];
          int32_t v80_a = v25_lane + 64;
          double v87_data = glb_m1[(v25_lane + 64)];
          int32_t v93_a = v25_lane + 80;
          double v100_data = glb_m1[(v25_lane + 80)];
          int32_t v106_a = v25_lane + 96;
          double v113_data = glb_m1[(v25_lane + 96)];
          int32_t v119_a = v25_lane + 112;
          double v126_data = glb_m1[(v25_lane + 112)];
          int32_t v132_a = v25_lane + 128;
          double v139_data = glb_m1[(v25_lane + 128)];
          int32_t v145_a = v25_lane + 144;
          double v152_data = glb_m1[(v25_lane + 144)];
          int32_t v158_a = v25_lane + 160;
          double v165_data = glb_m1[(v25_lane + 160)];
          int32_t v171_a = v25_lane + 176;
          double v178_data = glb_m1[(v25_lane + 176)];
          int32_t v184_a = v25_lane + 192;
          double v191_data = glb_m1[(v25_lane + 192)];
          int32_t v197_a = v25_lane + 208;
          double v204_data = glb_m1[(v25_lane + 208)];
          int32_t v210_a = v25_lane + 224;
          double v217_data = glb_m1[(v25_lane + 224)];
          int32_t v223_a = v25_lane + 240;
          double v230_data = glb_m1[(v25_lane + 240)];
          double v231_acc{};
          double v232_acc{};
          double v233_acc{};
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
          double v247_data = r0[0];
          double v248_data = r0[1];
          double v249_data = r0[2];
          double v250_data = r0[3];
          double v251_data = r0[4];
          double v252_data = r0[5];
          double v253_data = r0[6];
          double v254_data = r0[7];
          double v255_data = r0[8];
          double v256_data = r0[9];
          double v257_data = r0[10];
          double v258_data = r0[11];
          double v259_data = r0[12];
          double v260_data = r0[13];
          double v261_data = r0[14];
          double v262_data = r0[15];
          tensorforge::fmacdpp16<0>(v231_acc, v247_data, v35_data);
          tensorforge::fmacdpp16<1>(v231_acc, v247_data, v48_data);
          tensorforge::fmacdpp16<2>(v231_acc, v247_data, v61_data);
          tensorforge::fmacdpp16<3>(v231_acc, v247_data, v74_data);
          tensorforge::fmacdpp16<4>(v231_acc, v247_data, v87_data);
          tensorforge::fmacdpp16<5>(v231_acc, v247_data, v100_data);
          tensorforge::fmacdpp16<6>(v231_acc, v247_data, v113_data);
          tensorforge::fmacdpp16<7>(v231_acc, v247_data, v126_data);
          tensorforge::fmacdpp16<8>(v231_acc, v247_data, v139_data);
          tensorforge::fmacdpp16<9>(v231_acc, v247_data, v152_data);
          tensorforge::fmacdpp16<10>(v231_acc, v247_data, v165_data);
          tensorforge::fmacdpp16<11>(v231_acc, v247_data, v178_data);
          tensorforge::fmacdpp16<12>(v231_acc, v247_data, v191_data);
          tensorforge::fmacdpp16<13>(v231_acc, v247_data, v204_data);
          tensorforge::fmacdpp16<14>(v231_acc, v247_data, v217_data);
          tensorforge::fmacdpp16<15>(v231_acc, v247_data, v230_data);
          tensorforge::fmacdpp16<0>(v232_acc, v248_data, v35_data);
          tensorforge::fmacdpp16<1>(v232_acc, v248_data, v48_data);
          tensorforge::fmacdpp16<2>(v232_acc, v248_data, v61_data);
          tensorforge::fmacdpp16<3>(v232_acc, v248_data, v74_data);
          tensorforge::fmacdpp16<4>(v232_acc, v248_data, v87_data);
          tensorforge::fmacdpp16<5>(v232_acc, v248_data, v100_data);
          tensorforge::fmacdpp16<6>(v232_acc, v248_data, v113_data);
          tensorforge::fmacdpp16<7>(v232_acc, v248_data, v126_data);
          tensorforge::fmacdpp16<8>(v232_acc, v248_data, v139_data);
          tensorforge::fmacdpp16<9>(v232_acc, v248_data, v152_data);
          tensorforge::fmacdpp16<10>(v232_acc, v248_data, v165_data);
          tensorforge::fmacdpp16<11>(v232_acc, v248_data, v178_data);
          tensorforge::fmacdpp16<12>(v232_acc, v248_data, v191_data);
          tensorforge::fmacdpp16<13>(v232_acc, v248_data, v204_data);
          tensorforge::fmacdpp16<14>(v232_acc, v248_data, v217_data);
          tensorforge::fmacdpp16<15>(v232_acc, v248_data, v230_data);
          tensorforge::fmacdpp16<0>(v233_acc, v249_data, v35_data);
          tensorforge::fmacdpp16<1>(v233_acc, v249_data, v48_data);
          tensorforge::fmacdpp16<2>(v233_acc, v249_data, v61_data);
          tensorforge::fmacdpp16<3>(v233_acc, v249_data, v74_data);
          tensorforge::fmacdpp16<4>(v233_acc, v249_data, v87_data);
          tensorforge::fmacdpp16<5>(v233_acc, v249_data, v100_data);
          tensorforge::fmacdpp16<6>(v233_acc, v249_data, v113_data);
          tensorforge::fmacdpp16<7>(v233_acc, v249_data, v126_data);
          tensorforge::fmacdpp16<8>(v233_acc, v249_data, v139_data);
          tensorforge::fmacdpp16<9>(v233_acc, v249_data, v152_data);
          tensorforge::fmacdpp16<10>(v233_acc, v249_data, v165_data);
          tensorforge::fmacdpp16<11>(v233_acc, v249_data, v178_data);
          tensorforge::fmacdpp16<12>(v233_acc, v249_data, v191_data);
          tensorforge::fmacdpp16<13>(v233_acc, v249_data, v204_data);
          tensorforge::fmacdpp16<14>(v233_acc, v249_data, v217_data);
          tensorforge::fmacdpp16<15>(v233_acc, v249_data, v230_data);
          tensorforge::fmacdpp16<0>(v234_acc, v250_data, v35_data);
          tensorforge::fmacdpp16<1>(v234_acc, v250_data, v48_data);
          tensorforge::fmacdpp16<2>(v234_acc, v250_data, v61_data);
          tensorforge::fmacdpp16<3>(v234_acc, v250_data, v74_data);
          tensorforge::fmacdpp16<4>(v234_acc, v250_data, v87_data);
          tensorforge::fmacdpp16<5>(v234_acc, v250_data, v100_data);
          tensorforge::fmacdpp16<6>(v234_acc, v250_data, v113_data);
          tensorforge::fmacdpp16<7>(v234_acc, v250_data, v126_data);
          tensorforge::fmacdpp16<8>(v234_acc, v250_data, v139_data);
          tensorforge::fmacdpp16<9>(v234_acc, v250_data, v152_data);
          tensorforge::fmacdpp16<10>(v234_acc, v250_data, v165_data);
          tensorforge::fmacdpp16<11>(v234_acc, v250_data, v178_data);
          tensorforge::fmacdpp16<12>(v234_acc, v250_data, v191_data);
          tensorforge::fmacdpp16<13>(v234_acc, v250_data, v204_data);
          tensorforge::fmacdpp16<14>(v234_acc, v250_data, v217_data);
          tensorforge::fmacdpp16<15>(v234_acc, v250_data, v230_data);
          tensorforge::fmacdpp16<0>(v235_acc, v251_data, v35_data);
          tensorforge::fmacdpp16<1>(v235_acc, v251_data, v48_data);
          tensorforge::fmacdpp16<2>(v235_acc, v251_data, v61_data);
          tensorforge::fmacdpp16<3>(v235_acc, v251_data, v74_data);
          tensorforge::fmacdpp16<4>(v235_acc, v251_data, v87_data);
          tensorforge::fmacdpp16<5>(v235_acc, v251_data, v100_data);
          tensorforge::fmacdpp16<6>(v235_acc, v251_data, v113_data);
          tensorforge::fmacdpp16<7>(v235_acc, v251_data, v126_data);
          tensorforge::fmacdpp16<8>(v235_acc, v251_data, v139_data);
          tensorforge::fmacdpp16<9>(v235_acc, v251_data, v152_data);
          tensorforge::fmacdpp16<10>(v235_acc, v251_data, v165_data);
          tensorforge::fmacdpp16<11>(v235_acc, v251_data, v178_data);
          tensorforge::fmacdpp16<12>(v235_acc, v251_data, v191_data);
          tensorforge::fmacdpp16<13>(v235_acc, v251_data, v204_data);
          tensorforge::fmacdpp16<14>(v235_acc, v251_data, v217_data);
          tensorforge::fmacdpp16<15>(v235_acc, v251_data, v230_data);
          tensorforge::fmacdpp16<0>(v236_acc, v252_data, v35_data);
          tensorforge::fmacdpp16<1>(v236_acc, v252_data, v48_data);
          tensorforge::fmacdpp16<2>(v236_acc, v252_data, v61_data);
          tensorforge::fmacdpp16<3>(v236_acc, v252_data, v74_data);
          tensorforge::fmacdpp16<4>(v236_acc, v252_data, v87_data);
          tensorforge::fmacdpp16<5>(v236_acc, v252_data, v100_data);
          tensorforge::fmacdpp16<6>(v236_acc, v252_data, v113_data);
          tensorforge::fmacdpp16<7>(v236_acc, v252_data, v126_data);
          tensorforge::fmacdpp16<8>(v236_acc, v252_data, v139_data);
          tensorforge::fmacdpp16<9>(v236_acc, v252_data, v152_data);
          tensorforge::fmacdpp16<10>(v236_acc, v252_data, v165_data);
          tensorforge::fmacdpp16<11>(v236_acc, v252_data, v178_data);
          tensorforge::fmacdpp16<12>(v236_acc, v252_data, v191_data);
          tensorforge::fmacdpp16<13>(v236_acc, v252_data, v204_data);
          tensorforge::fmacdpp16<14>(v236_acc, v252_data, v217_data);
          tensorforge::fmacdpp16<15>(v236_acc, v252_data, v230_data);
          tensorforge::fmacdpp16<0>(v237_acc, v253_data, v35_data);
          tensorforge::fmacdpp16<1>(v237_acc, v253_data, v48_data);
          tensorforge::fmacdpp16<2>(v237_acc, v253_data, v61_data);
          tensorforge::fmacdpp16<3>(v237_acc, v253_data, v74_data);
          tensorforge::fmacdpp16<4>(v237_acc, v253_data, v87_data);
          tensorforge::fmacdpp16<5>(v237_acc, v253_data, v100_data);
          tensorforge::fmacdpp16<6>(v237_acc, v253_data, v113_data);
          tensorforge::fmacdpp16<7>(v237_acc, v253_data, v126_data);
          tensorforge::fmacdpp16<8>(v237_acc, v253_data, v139_data);
          tensorforge::fmacdpp16<9>(v237_acc, v253_data, v152_data);
          tensorforge::fmacdpp16<10>(v237_acc, v253_data, v165_data);
          tensorforge::fmacdpp16<11>(v237_acc, v253_data, v178_data);
          tensorforge::fmacdpp16<12>(v237_acc, v253_data, v191_data);
          tensorforge::fmacdpp16<13>(v237_acc, v253_data, v204_data);
          tensorforge::fmacdpp16<14>(v237_acc, v253_data, v217_data);
          tensorforge::fmacdpp16<15>(v237_acc, v253_data, v230_data);
          tensorforge::fmacdpp16<0>(v238_acc, v254_data, v35_data);
          tensorforge::fmacdpp16<1>(v238_acc, v254_data, v48_data);
          tensorforge::fmacdpp16<2>(v238_acc, v254_data, v61_data);
          tensorforge::fmacdpp16<3>(v238_acc, v254_data, v74_data);
          tensorforge::fmacdpp16<4>(v238_acc, v254_data, v87_data);
          tensorforge::fmacdpp16<5>(v238_acc, v254_data, v100_data);
          tensorforge::fmacdpp16<6>(v238_acc, v254_data, v113_data);
          tensorforge::fmacdpp16<7>(v238_acc, v254_data, v126_data);
          tensorforge::fmacdpp16<8>(v238_acc, v254_data, v139_data);
          tensorforge::fmacdpp16<9>(v238_acc, v254_data, v152_data);
          tensorforge::fmacdpp16<10>(v238_acc, v254_data, v165_data);
          tensorforge::fmacdpp16<11>(v238_acc, v254_data, v178_data);
          tensorforge::fmacdpp16<12>(v238_acc, v254_data, v191_data);
          tensorforge::fmacdpp16<13>(v238_acc, v254_data, v204_data);
          tensorforge::fmacdpp16<14>(v238_acc, v254_data, v217_data);
          tensorforge::fmacdpp16<15>(v238_acc, v254_data, v230_data);
          tensorforge::fmacdpp16<0>(v239_acc, v255_data, v35_data);
          tensorforge::fmacdpp16<1>(v239_acc, v255_data, v48_data);
          tensorforge::fmacdpp16<2>(v239_acc, v255_data, v61_data);
          tensorforge::fmacdpp16<3>(v239_acc, v255_data, v74_data);
          tensorforge::fmacdpp16<4>(v239_acc, v255_data, v87_data);
          tensorforge::fmacdpp16<5>(v239_acc, v255_data, v100_data);
          tensorforge::fmacdpp16<6>(v239_acc, v255_data, v113_data);
          tensorforge::fmacdpp16<7>(v239_acc, v255_data, v126_data);
          tensorforge::fmacdpp16<8>(v239_acc, v255_data, v139_data);
          tensorforge::fmacdpp16<9>(v239_acc, v255_data, v152_data);
          tensorforge::fmacdpp16<10>(v239_acc, v255_data, v165_data);
          tensorforge::fmacdpp16<11>(v239_acc, v255_data, v178_data);
          tensorforge::fmacdpp16<12>(v239_acc, v255_data, v191_data);
          tensorforge::fmacdpp16<13>(v239_acc, v255_data, v204_data);
          tensorforge::fmacdpp16<14>(v239_acc, v255_data, v217_data);
          tensorforge::fmacdpp16<15>(v239_acc, v255_data, v230_data);
          tensorforge::fmacdpp16<0>(v240_acc, v256_data, v35_data);
          tensorforge::fmacdpp16<1>(v240_acc, v256_data, v48_data);
          tensorforge::fmacdpp16<2>(v240_acc, v256_data, v61_data);
          tensorforge::fmacdpp16<3>(v240_acc, v256_data, v74_data);
          tensorforge::fmacdpp16<4>(v240_acc, v256_data, v87_data);
          tensorforge::fmacdpp16<5>(v240_acc, v256_data, v100_data);
          tensorforge::fmacdpp16<6>(v240_acc, v256_data, v113_data);
          tensorforge::fmacdpp16<7>(v240_acc, v256_data, v126_data);
          tensorforge::fmacdpp16<8>(v240_acc, v256_data, v139_data);
          tensorforge::fmacdpp16<9>(v240_acc, v256_data, v152_data);
          tensorforge::fmacdpp16<10>(v240_acc, v256_data, v165_data);
          tensorforge::fmacdpp16<11>(v240_acc, v256_data, v178_data);
          tensorforge::fmacdpp16<12>(v240_acc, v256_data, v191_data);
          tensorforge::fmacdpp16<13>(v240_acc, v256_data, v204_data);
          tensorforge::fmacdpp16<14>(v240_acc, v256_data, v217_data);
          tensorforge::fmacdpp16<15>(v240_acc, v256_data, v230_data);
          tensorforge::fmacdpp16<0>(v241_acc, v257_data, v35_data);
          tensorforge::fmacdpp16<1>(v241_acc, v257_data, v48_data);
          tensorforge::fmacdpp16<2>(v241_acc, v257_data, v61_data);
          tensorforge::fmacdpp16<3>(v241_acc, v257_data, v74_data);
          tensorforge::fmacdpp16<4>(v241_acc, v257_data, v87_data);
          tensorforge::fmacdpp16<5>(v241_acc, v257_data, v100_data);
          tensorforge::fmacdpp16<6>(v241_acc, v257_data, v113_data);
          tensorforge::fmacdpp16<7>(v241_acc, v257_data, v126_data);
          tensorforge::fmacdpp16<8>(v241_acc, v257_data, v139_data);
          tensorforge::fmacdpp16<9>(v241_acc, v257_data, v152_data);
          tensorforge::fmacdpp16<10>(v241_acc, v257_data, v165_data);
          tensorforge::fmacdpp16<11>(v241_acc, v257_data, v178_data);
          tensorforge::fmacdpp16<12>(v241_acc, v257_data, v191_data);
          tensorforge::fmacdpp16<13>(v241_acc, v257_data, v204_data);
          tensorforge::fmacdpp16<14>(v241_acc, v257_data, v217_data);
          tensorforge::fmacdpp16<15>(v241_acc, v257_data, v230_data);
          tensorforge::fmacdpp16<0>(v242_acc, v258_data, v35_data);
          tensorforge::fmacdpp16<1>(v242_acc, v258_data, v48_data);
          tensorforge::fmacdpp16<2>(v242_acc, v258_data, v61_data);
          tensorforge::fmacdpp16<3>(v242_acc, v258_data, v74_data);
          tensorforge::fmacdpp16<4>(v242_acc, v258_data, v87_data);
          tensorforge::fmacdpp16<5>(v242_acc, v258_data, v100_data);
          tensorforge::fmacdpp16<6>(v242_acc, v258_data, v113_data);
          tensorforge::fmacdpp16<7>(v242_acc, v258_data, v126_data);
          tensorforge::fmacdpp16<8>(v242_acc, v258_data, v139_data);
          tensorforge::fmacdpp16<9>(v242_acc, v258_data, v152_data);
          tensorforge::fmacdpp16<10>(v242_acc, v258_data, v165_data);
          tensorforge::fmacdpp16<11>(v242_acc, v258_data, v178_data);
          tensorforge::fmacdpp16<12>(v242_acc, v258_data, v191_data);
          tensorforge::fmacdpp16<13>(v242_acc, v258_data, v204_data);
          tensorforge::fmacdpp16<14>(v242_acc, v258_data, v217_data);
          tensorforge::fmacdpp16<15>(v242_acc, v258_data, v230_data);
          tensorforge::fmacdpp16<0>(v243_acc, v259_data, v35_data);
          tensorforge::fmacdpp16<1>(v243_acc, v259_data, v48_data);
          tensorforge::fmacdpp16<2>(v243_acc, v259_data, v61_data);
          tensorforge::fmacdpp16<3>(v243_acc, v259_data, v74_data);
          tensorforge::fmacdpp16<4>(v243_acc, v259_data, v87_data);
          tensorforge::fmacdpp16<5>(v243_acc, v259_data, v100_data);
          tensorforge::fmacdpp16<6>(v243_acc, v259_data, v113_data);
          tensorforge::fmacdpp16<7>(v243_acc, v259_data, v126_data);
          tensorforge::fmacdpp16<8>(v243_acc, v259_data, v139_data);
          tensorforge::fmacdpp16<9>(v243_acc, v259_data, v152_data);
          tensorforge::fmacdpp16<10>(v243_acc, v259_data, v165_data);
          tensorforge::fmacdpp16<11>(v243_acc, v259_data, v178_data);
          tensorforge::fmacdpp16<12>(v243_acc, v259_data, v191_data);
          tensorforge::fmacdpp16<13>(v243_acc, v259_data, v204_data);
          tensorforge::fmacdpp16<14>(v243_acc, v259_data, v217_data);
          tensorforge::fmacdpp16<15>(v243_acc, v259_data, v230_data);
          tensorforge::fmacdpp16<0>(v244_acc, v260_data, v35_data);
          tensorforge::fmacdpp16<1>(v244_acc, v260_data, v48_data);
          tensorforge::fmacdpp16<2>(v244_acc, v260_data, v61_data);
          tensorforge::fmacdpp16<3>(v244_acc, v260_data, v74_data);
          tensorforge::fmacdpp16<4>(v244_acc, v260_data, v87_data);
          tensorforge::fmacdpp16<5>(v244_acc, v260_data, v100_data);
          tensorforge::fmacdpp16<6>(v244_acc, v260_data, v113_data);
          tensorforge::fmacdpp16<7>(v244_acc, v260_data, v126_data);
          tensorforge::fmacdpp16<8>(v244_acc, v260_data, v139_data);
          tensorforge::fmacdpp16<9>(v244_acc, v260_data, v152_data);
          tensorforge::fmacdpp16<10>(v244_acc, v260_data, v165_data);
          tensorforge::fmacdpp16<11>(v244_acc, v260_data, v178_data);
          tensorforge::fmacdpp16<12>(v244_acc, v260_data, v191_data);
          tensorforge::fmacdpp16<13>(v244_acc, v260_data, v204_data);
          tensorforge::fmacdpp16<14>(v244_acc, v260_data, v217_data);
          tensorforge::fmacdpp16<15>(v244_acc, v260_data, v230_data);
          tensorforge::fmacdpp16<0>(v245_acc, v261_data, v35_data);
          tensorforge::fmacdpp16<1>(v245_acc, v261_data, v48_data);
          tensorforge::fmacdpp16<2>(v245_acc, v261_data, v61_data);
          tensorforge::fmacdpp16<3>(v245_acc, v261_data, v74_data);
          tensorforge::fmacdpp16<4>(v245_acc, v261_data, v87_data);
          tensorforge::fmacdpp16<5>(v245_acc, v261_data, v100_data);
          tensorforge::fmacdpp16<6>(v245_acc, v261_data, v113_data);
          tensorforge::fmacdpp16<7>(v245_acc, v261_data, v126_data);
          tensorforge::fmacdpp16<8>(v245_acc, v261_data, v139_data);
          tensorforge::fmacdpp16<9>(v245_acc, v261_data, v152_data);
          tensorforge::fmacdpp16<10>(v245_acc, v261_data, v165_data);
          tensorforge::fmacdpp16<11>(v245_acc, v261_data, v178_data);
          tensorforge::fmacdpp16<12>(v245_acc, v261_data, v191_data);
          tensorforge::fmacdpp16<13>(v245_acc, v261_data, v204_data);
          tensorforge::fmacdpp16<14>(v245_acc, v261_data, v217_data);
          tensorforge::fmacdpp16<15>(v245_acc, v261_data, v230_data);
          tensorforge::fmacdpp16<0>(v246_acc, v262_data, v35_data);
          tensorforge::fmacdpp16<1>(v246_acc, v262_data, v48_data);
          tensorforge::fmacdpp16<2>(v246_acc, v262_data, v61_data);
          tensorforge::fmacdpp16<3>(v246_acc, v262_data, v74_data);
          tensorforge::fmacdpp16<4>(v246_acc, v262_data, v87_data);
          tensorforge::fmacdpp16<5>(v246_acc, v262_data, v100_data);
          tensorforge::fmacdpp16<6>(v246_acc, v262_data, v113_data);
          tensorforge::fmacdpp16<7>(v246_acc, v262_data, v126_data);
          tensorforge::fmacdpp16<8>(v246_acc, v262_data, v139_data);
          tensorforge::fmacdpp16<9>(v246_acc, v262_data, v152_data);
          tensorforge::fmacdpp16<10>(v246_acc, v262_data, v165_data);
          tensorforge::fmacdpp16<11>(v246_acc, v262_data, v178_data);
          tensorforge::fmacdpp16<12>(v246_acc, v262_data, v191_data);
          tensorforge::fmacdpp16<13>(v246_acc, v262_data, v204_data);
          tensorforge::fmacdpp16<14>(v246_acc, v262_data, v217_data);
          tensorforge::fmacdpp16<15>(v246_acc, v262_data, v230_data);
          r1[0] = v231_acc;
          r1[1] = v232_acc;
          r1[2] = v233_acc;
          r1[3] = v234_acc;
          r1[4] = v235_acc;
          r1[5] = v236_acc;
          r1[6] = v237_acc;
          r1[7] = v238_acc;
          r1[8] = v239_acc;
          r1[9] = v240_acc;
          r1[10] = v241_acc;
          r1[11] = v242_acc;
          r1[12] = v243_acc;
          r1[13] = v244_acc;
          r1[14] = v245_acc;
          r1[15] = v246_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v266_i0 = 0; v266_i0 < 1; ++v266_i0) {
            int32_t v275_lead = v25_lane + (v266_i0 * 16);
            #pragma unroll
            for (int32_t v267_i1 = 0; v267_i1 < 16; ++v267_i1) {
              int32_t v268_a = v266_i0 + v267_i1;
              double v270_data = r1[(v266_i0 + v267_i1)];
              glb_m0[(v275_lead + (v267_i1 * 16))] = v270_data;
            }
          }
        }
      }
    }
  }
}

