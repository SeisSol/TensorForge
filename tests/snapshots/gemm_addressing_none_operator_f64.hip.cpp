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
          double v2_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v2_lin;
          double v3_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v3_lin;
          double v4_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v4_lin;
          double v5_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v5_lin;
          double v6_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v6_lin;
          double v7_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v7_lin;
          double v8_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v8_lin;
          double v9_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v9_lin;
          double v10_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v10_lin;
          double v11_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v11_lin;
          double v12_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v12_lin;
          double v13_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v13_lin;
          double v14_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v14_lin;
          double v15_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v15_lin;
          double v16_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v16_lin;
          double v17_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v17_lin;
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir1 = r1;
          int32_t v21_lane = threadIdx.x % 16;
          int32_t v24_a = v21_lane + 0;
          double v31_data = glb_m1[v21_lane];
          int32_t v37_a = v21_lane + 16;
          double v44_data = glb_m1[(v21_lane + 16)];
          int32_t v50_a = v21_lane + 32;
          double v57_data = glb_m1[(v21_lane + 32)];
          int32_t v63_a = v21_lane + 48;
          double v70_data = glb_m1[(v21_lane + 48)];
          int32_t v76_a = v21_lane + 64;
          double v83_data = glb_m1[(v21_lane + 64)];
          int32_t v89_a = v21_lane + 80;
          double v96_data = glb_m1[(v21_lane + 80)];
          int32_t v102_a = v21_lane + 96;
          double v109_data = glb_m1[(v21_lane + 96)];
          int32_t v115_a = v21_lane + 112;
          double v122_data = glb_m1[(v21_lane + 112)];
          int32_t v128_a = v21_lane + 128;
          double v135_data = glb_m1[(v21_lane + 128)];
          int32_t v141_a = v21_lane + 144;
          double v148_data = glb_m1[(v21_lane + 144)];
          int32_t v154_a = v21_lane + 160;
          double v161_data = glb_m1[(v21_lane + 160)];
          int32_t v167_a = v21_lane + 176;
          double v174_data = glb_m1[(v21_lane + 176)];
          int32_t v180_a = v21_lane + 192;
          double v187_data = glb_m1[(v21_lane + 192)];
          int32_t v193_a = v21_lane + 208;
          double v200_data = glb_m1[(v21_lane + 208)];
          int32_t v206_a = v21_lane + 224;
          double v213_data = glb_m1[(v21_lane + 224)];
          int32_t v219_a = v21_lane + 240;
          double v226_data = glb_m1[(v21_lane + 240)];
          double v227_acc{};
          double v228_acc{};
          double v229_acc{};
          double v230_acc{};
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
          double v243_data = r0[0];
          double v244_data = r0[1];
          double v245_data = r0[2];
          double v246_data = r0[3];
          double v247_data = r0[4];
          double v248_data = r0[5];
          double v249_data = r0[6];
          double v250_data = r0[7];
          double v251_data = r0[8];
          double v252_data = r0[9];
          double v253_data = r0[10];
          double v254_data = r0[11];
          double v255_data = r0[12];
          double v256_data = r0[13];
          double v257_data = r0[14];
          double v258_data = r0[15];
          tensorforge::fmacdpp16<0>(v227_acc, v243_data, v31_data);
          tensorforge::fmacdpp16<1>(v227_acc, v243_data, v44_data);
          tensorforge::fmacdpp16<2>(v227_acc, v243_data, v57_data);
          tensorforge::fmacdpp16<3>(v227_acc, v243_data, v70_data);
          tensorforge::fmacdpp16<4>(v227_acc, v243_data, v83_data);
          tensorforge::fmacdpp16<5>(v227_acc, v243_data, v96_data);
          tensorforge::fmacdpp16<6>(v227_acc, v243_data, v109_data);
          tensorforge::fmacdpp16<7>(v227_acc, v243_data, v122_data);
          tensorforge::fmacdpp16<8>(v227_acc, v243_data, v135_data);
          tensorforge::fmacdpp16<9>(v227_acc, v243_data, v148_data);
          tensorforge::fmacdpp16<10>(v227_acc, v243_data, v161_data);
          tensorforge::fmacdpp16<11>(v227_acc, v243_data, v174_data);
          tensorforge::fmacdpp16<12>(v227_acc, v243_data, v187_data);
          tensorforge::fmacdpp16<13>(v227_acc, v243_data, v200_data);
          tensorforge::fmacdpp16<14>(v227_acc, v243_data, v213_data);
          tensorforge::fmacdpp16<15>(v227_acc, v243_data, v226_data);
          tensorforge::fmacdpp16<0>(v228_acc, v244_data, v31_data);
          tensorforge::fmacdpp16<1>(v228_acc, v244_data, v44_data);
          tensorforge::fmacdpp16<2>(v228_acc, v244_data, v57_data);
          tensorforge::fmacdpp16<3>(v228_acc, v244_data, v70_data);
          tensorforge::fmacdpp16<4>(v228_acc, v244_data, v83_data);
          tensorforge::fmacdpp16<5>(v228_acc, v244_data, v96_data);
          tensorforge::fmacdpp16<6>(v228_acc, v244_data, v109_data);
          tensorforge::fmacdpp16<7>(v228_acc, v244_data, v122_data);
          tensorforge::fmacdpp16<8>(v228_acc, v244_data, v135_data);
          tensorforge::fmacdpp16<9>(v228_acc, v244_data, v148_data);
          tensorforge::fmacdpp16<10>(v228_acc, v244_data, v161_data);
          tensorforge::fmacdpp16<11>(v228_acc, v244_data, v174_data);
          tensorforge::fmacdpp16<12>(v228_acc, v244_data, v187_data);
          tensorforge::fmacdpp16<13>(v228_acc, v244_data, v200_data);
          tensorforge::fmacdpp16<14>(v228_acc, v244_data, v213_data);
          tensorforge::fmacdpp16<15>(v228_acc, v244_data, v226_data);
          tensorforge::fmacdpp16<0>(v229_acc, v245_data, v31_data);
          tensorforge::fmacdpp16<1>(v229_acc, v245_data, v44_data);
          tensorforge::fmacdpp16<2>(v229_acc, v245_data, v57_data);
          tensorforge::fmacdpp16<3>(v229_acc, v245_data, v70_data);
          tensorforge::fmacdpp16<4>(v229_acc, v245_data, v83_data);
          tensorforge::fmacdpp16<5>(v229_acc, v245_data, v96_data);
          tensorforge::fmacdpp16<6>(v229_acc, v245_data, v109_data);
          tensorforge::fmacdpp16<7>(v229_acc, v245_data, v122_data);
          tensorforge::fmacdpp16<8>(v229_acc, v245_data, v135_data);
          tensorforge::fmacdpp16<9>(v229_acc, v245_data, v148_data);
          tensorforge::fmacdpp16<10>(v229_acc, v245_data, v161_data);
          tensorforge::fmacdpp16<11>(v229_acc, v245_data, v174_data);
          tensorforge::fmacdpp16<12>(v229_acc, v245_data, v187_data);
          tensorforge::fmacdpp16<13>(v229_acc, v245_data, v200_data);
          tensorforge::fmacdpp16<14>(v229_acc, v245_data, v213_data);
          tensorforge::fmacdpp16<15>(v229_acc, v245_data, v226_data);
          tensorforge::fmacdpp16<0>(v230_acc, v246_data, v31_data);
          tensorforge::fmacdpp16<1>(v230_acc, v246_data, v44_data);
          tensorforge::fmacdpp16<2>(v230_acc, v246_data, v57_data);
          tensorforge::fmacdpp16<3>(v230_acc, v246_data, v70_data);
          tensorforge::fmacdpp16<4>(v230_acc, v246_data, v83_data);
          tensorforge::fmacdpp16<5>(v230_acc, v246_data, v96_data);
          tensorforge::fmacdpp16<6>(v230_acc, v246_data, v109_data);
          tensorforge::fmacdpp16<7>(v230_acc, v246_data, v122_data);
          tensorforge::fmacdpp16<8>(v230_acc, v246_data, v135_data);
          tensorforge::fmacdpp16<9>(v230_acc, v246_data, v148_data);
          tensorforge::fmacdpp16<10>(v230_acc, v246_data, v161_data);
          tensorforge::fmacdpp16<11>(v230_acc, v246_data, v174_data);
          tensorforge::fmacdpp16<12>(v230_acc, v246_data, v187_data);
          tensorforge::fmacdpp16<13>(v230_acc, v246_data, v200_data);
          tensorforge::fmacdpp16<14>(v230_acc, v246_data, v213_data);
          tensorforge::fmacdpp16<15>(v230_acc, v246_data, v226_data);
          tensorforge::fmacdpp16<0>(v231_acc, v247_data, v31_data);
          tensorforge::fmacdpp16<1>(v231_acc, v247_data, v44_data);
          tensorforge::fmacdpp16<2>(v231_acc, v247_data, v57_data);
          tensorforge::fmacdpp16<3>(v231_acc, v247_data, v70_data);
          tensorforge::fmacdpp16<4>(v231_acc, v247_data, v83_data);
          tensorforge::fmacdpp16<5>(v231_acc, v247_data, v96_data);
          tensorforge::fmacdpp16<6>(v231_acc, v247_data, v109_data);
          tensorforge::fmacdpp16<7>(v231_acc, v247_data, v122_data);
          tensorforge::fmacdpp16<8>(v231_acc, v247_data, v135_data);
          tensorforge::fmacdpp16<9>(v231_acc, v247_data, v148_data);
          tensorforge::fmacdpp16<10>(v231_acc, v247_data, v161_data);
          tensorforge::fmacdpp16<11>(v231_acc, v247_data, v174_data);
          tensorforge::fmacdpp16<12>(v231_acc, v247_data, v187_data);
          tensorforge::fmacdpp16<13>(v231_acc, v247_data, v200_data);
          tensorforge::fmacdpp16<14>(v231_acc, v247_data, v213_data);
          tensorforge::fmacdpp16<15>(v231_acc, v247_data, v226_data);
          tensorforge::fmacdpp16<0>(v232_acc, v248_data, v31_data);
          tensorforge::fmacdpp16<1>(v232_acc, v248_data, v44_data);
          tensorforge::fmacdpp16<2>(v232_acc, v248_data, v57_data);
          tensorforge::fmacdpp16<3>(v232_acc, v248_data, v70_data);
          tensorforge::fmacdpp16<4>(v232_acc, v248_data, v83_data);
          tensorforge::fmacdpp16<5>(v232_acc, v248_data, v96_data);
          tensorforge::fmacdpp16<6>(v232_acc, v248_data, v109_data);
          tensorforge::fmacdpp16<7>(v232_acc, v248_data, v122_data);
          tensorforge::fmacdpp16<8>(v232_acc, v248_data, v135_data);
          tensorforge::fmacdpp16<9>(v232_acc, v248_data, v148_data);
          tensorforge::fmacdpp16<10>(v232_acc, v248_data, v161_data);
          tensorforge::fmacdpp16<11>(v232_acc, v248_data, v174_data);
          tensorforge::fmacdpp16<12>(v232_acc, v248_data, v187_data);
          tensorforge::fmacdpp16<13>(v232_acc, v248_data, v200_data);
          tensorforge::fmacdpp16<14>(v232_acc, v248_data, v213_data);
          tensorforge::fmacdpp16<15>(v232_acc, v248_data, v226_data);
          tensorforge::fmacdpp16<0>(v233_acc, v249_data, v31_data);
          tensorforge::fmacdpp16<1>(v233_acc, v249_data, v44_data);
          tensorforge::fmacdpp16<2>(v233_acc, v249_data, v57_data);
          tensorforge::fmacdpp16<3>(v233_acc, v249_data, v70_data);
          tensorforge::fmacdpp16<4>(v233_acc, v249_data, v83_data);
          tensorforge::fmacdpp16<5>(v233_acc, v249_data, v96_data);
          tensorforge::fmacdpp16<6>(v233_acc, v249_data, v109_data);
          tensorforge::fmacdpp16<7>(v233_acc, v249_data, v122_data);
          tensorforge::fmacdpp16<8>(v233_acc, v249_data, v135_data);
          tensorforge::fmacdpp16<9>(v233_acc, v249_data, v148_data);
          tensorforge::fmacdpp16<10>(v233_acc, v249_data, v161_data);
          tensorforge::fmacdpp16<11>(v233_acc, v249_data, v174_data);
          tensorforge::fmacdpp16<12>(v233_acc, v249_data, v187_data);
          tensorforge::fmacdpp16<13>(v233_acc, v249_data, v200_data);
          tensorforge::fmacdpp16<14>(v233_acc, v249_data, v213_data);
          tensorforge::fmacdpp16<15>(v233_acc, v249_data, v226_data);
          tensorforge::fmacdpp16<0>(v234_acc, v250_data, v31_data);
          tensorforge::fmacdpp16<1>(v234_acc, v250_data, v44_data);
          tensorforge::fmacdpp16<2>(v234_acc, v250_data, v57_data);
          tensorforge::fmacdpp16<3>(v234_acc, v250_data, v70_data);
          tensorforge::fmacdpp16<4>(v234_acc, v250_data, v83_data);
          tensorforge::fmacdpp16<5>(v234_acc, v250_data, v96_data);
          tensorforge::fmacdpp16<6>(v234_acc, v250_data, v109_data);
          tensorforge::fmacdpp16<7>(v234_acc, v250_data, v122_data);
          tensorforge::fmacdpp16<8>(v234_acc, v250_data, v135_data);
          tensorforge::fmacdpp16<9>(v234_acc, v250_data, v148_data);
          tensorforge::fmacdpp16<10>(v234_acc, v250_data, v161_data);
          tensorforge::fmacdpp16<11>(v234_acc, v250_data, v174_data);
          tensorforge::fmacdpp16<12>(v234_acc, v250_data, v187_data);
          tensorforge::fmacdpp16<13>(v234_acc, v250_data, v200_data);
          tensorforge::fmacdpp16<14>(v234_acc, v250_data, v213_data);
          tensorforge::fmacdpp16<15>(v234_acc, v250_data, v226_data);
          tensorforge::fmacdpp16<0>(v235_acc, v251_data, v31_data);
          tensorforge::fmacdpp16<1>(v235_acc, v251_data, v44_data);
          tensorforge::fmacdpp16<2>(v235_acc, v251_data, v57_data);
          tensorforge::fmacdpp16<3>(v235_acc, v251_data, v70_data);
          tensorforge::fmacdpp16<4>(v235_acc, v251_data, v83_data);
          tensorforge::fmacdpp16<5>(v235_acc, v251_data, v96_data);
          tensorforge::fmacdpp16<6>(v235_acc, v251_data, v109_data);
          tensorforge::fmacdpp16<7>(v235_acc, v251_data, v122_data);
          tensorforge::fmacdpp16<8>(v235_acc, v251_data, v135_data);
          tensorforge::fmacdpp16<9>(v235_acc, v251_data, v148_data);
          tensorforge::fmacdpp16<10>(v235_acc, v251_data, v161_data);
          tensorforge::fmacdpp16<11>(v235_acc, v251_data, v174_data);
          tensorforge::fmacdpp16<12>(v235_acc, v251_data, v187_data);
          tensorforge::fmacdpp16<13>(v235_acc, v251_data, v200_data);
          tensorforge::fmacdpp16<14>(v235_acc, v251_data, v213_data);
          tensorforge::fmacdpp16<15>(v235_acc, v251_data, v226_data);
          tensorforge::fmacdpp16<0>(v236_acc, v252_data, v31_data);
          tensorforge::fmacdpp16<1>(v236_acc, v252_data, v44_data);
          tensorforge::fmacdpp16<2>(v236_acc, v252_data, v57_data);
          tensorforge::fmacdpp16<3>(v236_acc, v252_data, v70_data);
          tensorforge::fmacdpp16<4>(v236_acc, v252_data, v83_data);
          tensorforge::fmacdpp16<5>(v236_acc, v252_data, v96_data);
          tensorforge::fmacdpp16<6>(v236_acc, v252_data, v109_data);
          tensorforge::fmacdpp16<7>(v236_acc, v252_data, v122_data);
          tensorforge::fmacdpp16<8>(v236_acc, v252_data, v135_data);
          tensorforge::fmacdpp16<9>(v236_acc, v252_data, v148_data);
          tensorforge::fmacdpp16<10>(v236_acc, v252_data, v161_data);
          tensorforge::fmacdpp16<11>(v236_acc, v252_data, v174_data);
          tensorforge::fmacdpp16<12>(v236_acc, v252_data, v187_data);
          tensorforge::fmacdpp16<13>(v236_acc, v252_data, v200_data);
          tensorforge::fmacdpp16<14>(v236_acc, v252_data, v213_data);
          tensorforge::fmacdpp16<15>(v236_acc, v252_data, v226_data);
          tensorforge::fmacdpp16<0>(v237_acc, v253_data, v31_data);
          tensorforge::fmacdpp16<1>(v237_acc, v253_data, v44_data);
          tensorforge::fmacdpp16<2>(v237_acc, v253_data, v57_data);
          tensorforge::fmacdpp16<3>(v237_acc, v253_data, v70_data);
          tensorforge::fmacdpp16<4>(v237_acc, v253_data, v83_data);
          tensorforge::fmacdpp16<5>(v237_acc, v253_data, v96_data);
          tensorforge::fmacdpp16<6>(v237_acc, v253_data, v109_data);
          tensorforge::fmacdpp16<7>(v237_acc, v253_data, v122_data);
          tensorforge::fmacdpp16<8>(v237_acc, v253_data, v135_data);
          tensorforge::fmacdpp16<9>(v237_acc, v253_data, v148_data);
          tensorforge::fmacdpp16<10>(v237_acc, v253_data, v161_data);
          tensorforge::fmacdpp16<11>(v237_acc, v253_data, v174_data);
          tensorforge::fmacdpp16<12>(v237_acc, v253_data, v187_data);
          tensorforge::fmacdpp16<13>(v237_acc, v253_data, v200_data);
          tensorforge::fmacdpp16<14>(v237_acc, v253_data, v213_data);
          tensorforge::fmacdpp16<15>(v237_acc, v253_data, v226_data);
          tensorforge::fmacdpp16<0>(v238_acc, v254_data, v31_data);
          tensorforge::fmacdpp16<1>(v238_acc, v254_data, v44_data);
          tensorforge::fmacdpp16<2>(v238_acc, v254_data, v57_data);
          tensorforge::fmacdpp16<3>(v238_acc, v254_data, v70_data);
          tensorforge::fmacdpp16<4>(v238_acc, v254_data, v83_data);
          tensorforge::fmacdpp16<5>(v238_acc, v254_data, v96_data);
          tensorforge::fmacdpp16<6>(v238_acc, v254_data, v109_data);
          tensorforge::fmacdpp16<7>(v238_acc, v254_data, v122_data);
          tensorforge::fmacdpp16<8>(v238_acc, v254_data, v135_data);
          tensorforge::fmacdpp16<9>(v238_acc, v254_data, v148_data);
          tensorforge::fmacdpp16<10>(v238_acc, v254_data, v161_data);
          tensorforge::fmacdpp16<11>(v238_acc, v254_data, v174_data);
          tensorforge::fmacdpp16<12>(v238_acc, v254_data, v187_data);
          tensorforge::fmacdpp16<13>(v238_acc, v254_data, v200_data);
          tensorforge::fmacdpp16<14>(v238_acc, v254_data, v213_data);
          tensorforge::fmacdpp16<15>(v238_acc, v254_data, v226_data);
          tensorforge::fmacdpp16<0>(v239_acc, v255_data, v31_data);
          tensorforge::fmacdpp16<1>(v239_acc, v255_data, v44_data);
          tensorforge::fmacdpp16<2>(v239_acc, v255_data, v57_data);
          tensorforge::fmacdpp16<3>(v239_acc, v255_data, v70_data);
          tensorforge::fmacdpp16<4>(v239_acc, v255_data, v83_data);
          tensorforge::fmacdpp16<5>(v239_acc, v255_data, v96_data);
          tensorforge::fmacdpp16<6>(v239_acc, v255_data, v109_data);
          tensorforge::fmacdpp16<7>(v239_acc, v255_data, v122_data);
          tensorforge::fmacdpp16<8>(v239_acc, v255_data, v135_data);
          tensorforge::fmacdpp16<9>(v239_acc, v255_data, v148_data);
          tensorforge::fmacdpp16<10>(v239_acc, v255_data, v161_data);
          tensorforge::fmacdpp16<11>(v239_acc, v255_data, v174_data);
          tensorforge::fmacdpp16<12>(v239_acc, v255_data, v187_data);
          tensorforge::fmacdpp16<13>(v239_acc, v255_data, v200_data);
          tensorforge::fmacdpp16<14>(v239_acc, v255_data, v213_data);
          tensorforge::fmacdpp16<15>(v239_acc, v255_data, v226_data);
          tensorforge::fmacdpp16<0>(v240_acc, v256_data, v31_data);
          tensorforge::fmacdpp16<1>(v240_acc, v256_data, v44_data);
          tensorforge::fmacdpp16<2>(v240_acc, v256_data, v57_data);
          tensorforge::fmacdpp16<3>(v240_acc, v256_data, v70_data);
          tensorforge::fmacdpp16<4>(v240_acc, v256_data, v83_data);
          tensorforge::fmacdpp16<5>(v240_acc, v256_data, v96_data);
          tensorforge::fmacdpp16<6>(v240_acc, v256_data, v109_data);
          tensorforge::fmacdpp16<7>(v240_acc, v256_data, v122_data);
          tensorforge::fmacdpp16<8>(v240_acc, v256_data, v135_data);
          tensorforge::fmacdpp16<9>(v240_acc, v256_data, v148_data);
          tensorforge::fmacdpp16<10>(v240_acc, v256_data, v161_data);
          tensorforge::fmacdpp16<11>(v240_acc, v256_data, v174_data);
          tensorforge::fmacdpp16<12>(v240_acc, v256_data, v187_data);
          tensorforge::fmacdpp16<13>(v240_acc, v256_data, v200_data);
          tensorforge::fmacdpp16<14>(v240_acc, v256_data, v213_data);
          tensorforge::fmacdpp16<15>(v240_acc, v256_data, v226_data);
          tensorforge::fmacdpp16<0>(v241_acc, v257_data, v31_data);
          tensorforge::fmacdpp16<1>(v241_acc, v257_data, v44_data);
          tensorforge::fmacdpp16<2>(v241_acc, v257_data, v57_data);
          tensorforge::fmacdpp16<3>(v241_acc, v257_data, v70_data);
          tensorforge::fmacdpp16<4>(v241_acc, v257_data, v83_data);
          tensorforge::fmacdpp16<5>(v241_acc, v257_data, v96_data);
          tensorforge::fmacdpp16<6>(v241_acc, v257_data, v109_data);
          tensorforge::fmacdpp16<7>(v241_acc, v257_data, v122_data);
          tensorforge::fmacdpp16<8>(v241_acc, v257_data, v135_data);
          tensorforge::fmacdpp16<9>(v241_acc, v257_data, v148_data);
          tensorforge::fmacdpp16<10>(v241_acc, v257_data, v161_data);
          tensorforge::fmacdpp16<11>(v241_acc, v257_data, v174_data);
          tensorforge::fmacdpp16<12>(v241_acc, v257_data, v187_data);
          tensorforge::fmacdpp16<13>(v241_acc, v257_data, v200_data);
          tensorforge::fmacdpp16<14>(v241_acc, v257_data, v213_data);
          tensorforge::fmacdpp16<15>(v241_acc, v257_data, v226_data);
          tensorforge::fmacdpp16<0>(v242_acc, v258_data, v31_data);
          tensorforge::fmacdpp16<1>(v242_acc, v258_data, v44_data);
          tensorforge::fmacdpp16<2>(v242_acc, v258_data, v57_data);
          tensorforge::fmacdpp16<3>(v242_acc, v258_data, v70_data);
          tensorforge::fmacdpp16<4>(v242_acc, v258_data, v83_data);
          tensorforge::fmacdpp16<5>(v242_acc, v258_data, v96_data);
          tensorforge::fmacdpp16<6>(v242_acc, v258_data, v109_data);
          tensorforge::fmacdpp16<7>(v242_acc, v258_data, v122_data);
          tensorforge::fmacdpp16<8>(v242_acc, v258_data, v135_data);
          tensorforge::fmacdpp16<9>(v242_acc, v258_data, v148_data);
          tensorforge::fmacdpp16<10>(v242_acc, v258_data, v161_data);
          tensorforge::fmacdpp16<11>(v242_acc, v258_data, v174_data);
          tensorforge::fmacdpp16<12>(v242_acc, v258_data, v187_data);
          tensorforge::fmacdpp16<13>(v242_acc, v258_data, v200_data);
          tensorforge::fmacdpp16<14>(v242_acc, v258_data, v213_data);
          tensorforge::fmacdpp16<15>(v242_acc, v258_data, v226_data);
          ir1[0] = v227_acc;
          ir1[1] = v228_acc;
          ir1[2] = v229_acc;
          ir1[3] = v230_acc;
          ir1[4] = v231_acc;
          ir1[5] = v232_acc;
          ir1[6] = v233_acc;
          ir1[7] = v234_acc;
          ir1[8] = v235_acc;
          ir1[9] = v236_acc;
          ir1[10] = v237_acc;
          ir1[11] = v238_acc;
          ir1[12] = v239_acc;
          ir1[13] = v240_acc;
          ir1[14] = v241_acc;
          ir1[15] = v242_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v262_i0 = 0; v262_i0 < 1; ++v262_i0) {
            int32_t v271_lead = v21_lane + (v262_i0 * 16);
            #pragma unroll
            for (int32_t v263_i1 = 0; v263_i1 < 16; ++v263_i1) {
              int32_t v264_a = v262_i0 + v263_i1;
              double v266_data = r1[(v262_i0 + v263_i1)];
              int32_t v273_a = v271_lead + (v263_i1 * 16);
              glb_m0[v273_a] = v266_data;
            }
          }
          ;
        }
      }
    }
  }
}

