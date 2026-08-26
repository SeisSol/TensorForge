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
          double v1_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v1_lin;
          double v2_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v2_lin;
          double v3_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v3_lin;
          double v4_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v4_lin;
          double v5_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v5_lin;
          double v6_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v6_lin;
          double v7_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v7_lin;
          double v8_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v8_lin;
          double v9_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v9_lin;
          double v10_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v10_lin;
          double v11_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v11_lin;
          double v12_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v12_lin;
          double v13_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v13_lin;
          double v14_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v14_lin;
          double v15_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v15_lin;
          double v16_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v16_lin;
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir1 = r1;
          int32_t v20_lane = threadIdx.x % 16;
          int32_t v23_a = v20_lane + 0;
          double v30_data = glb_m1[v20_lane];
          int32_t v36_a = v20_lane + 16;
          double v43_data = glb_m1[(v20_lane + 16)];
          int32_t v49_a = v20_lane + 32;
          double v56_data = glb_m1[(v20_lane + 32)];
          int32_t v62_a = v20_lane + 48;
          double v69_data = glb_m1[(v20_lane + 48)];
          int32_t v75_a = v20_lane + 64;
          double v82_data = glb_m1[(v20_lane + 64)];
          int32_t v88_a = v20_lane + 80;
          double v95_data = glb_m1[(v20_lane + 80)];
          int32_t v101_a = v20_lane + 96;
          double v108_data = glb_m1[(v20_lane + 96)];
          int32_t v114_a = v20_lane + 112;
          double v121_data = glb_m1[(v20_lane + 112)];
          int32_t v127_a = v20_lane + 128;
          double v134_data = glb_m1[(v20_lane + 128)];
          int32_t v140_a = v20_lane + 144;
          double v147_data = glb_m1[(v20_lane + 144)];
          int32_t v153_a = v20_lane + 160;
          double v160_data = glb_m1[(v20_lane + 160)];
          int32_t v166_a = v20_lane + 176;
          double v173_data = glb_m1[(v20_lane + 176)];
          int32_t v179_a = v20_lane + 192;
          double v186_data = glb_m1[(v20_lane + 192)];
          int32_t v192_a = v20_lane + 208;
          double v199_data = glb_m1[(v20_lane + 208)];
          int32_t v205_a = v20_lane + 224;
          double v212_data = glb_m1[(v20_lane + 224)];
          int32_t v218_a = v20_lane + 240;
          double v225_data = glb_m1[(v20_lane + 240)];
          double v226_acc{};
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
          double v242_data = r0[0];
          double v243_data = r0[1];
          double v244_data = r0[2];
          double v245_data = r0[3];
          double v246_data = r0[4];
          double v247_data = r0[5];
          double v248_data = r0[6];
          double v249_data = r0[7];
          double v250_data = r0[8];
          double v251_data = r0[9];
          double v252_data = r0[10];
          double v253_data = r0[11];
          double v254_data = r0[12];
          double v255_data = r0[13];
          double v256_data = r0[14];
          double v257_data = r0[15];
          tensorforge::fmacdpp16<0>(v226_acc, v242_data, v30_data);
          tensorforge::fmacdpp16<1>(v226_acc, v242_data, v43_data);
          tensorforge::fmacdpp16<2>(v226_acc, v242_data, v56_data);
          tensorforge::fmacdpp16<3>(v226_acc, v242_data, v69_data);
          tensorforge::fmacdpp16<4>(v226_acc, v242_data, v82_data);
          tensorforge::fmacdpp16<5>(v226_acc, v242_data, v95_data);
          tensorforge::fmacdpp16<6>(v226_acc, v242_data, v108_data);
          tensorforge::fmacdpp16<7>(v226_acc, v242_data, v121_data);
          tensorforge::fmacdpp16<8>(v226_acc, v242_data, v134_data);
          tensorforge::fmacdpp16<9>(v226_acc, v242_data, v147_data);
          tensorforge::fmacdpp16<10>(v226_acc, v242_data, v160_data);
          tensorforge::fmacdpp16<11>(v226_acc, v242_data, v173_data);
          tensorforge::fmacdpp16<12>(v226_acc, v242_data, v186_data);
          tensorforge::fmacdpp16<13>(v226_acc, v242_data, v199_data);
          tensorforge::fmacdpp16<14>(v226_acc, v242_data, v212_data);
          tensorforge::fmacdpp16<15>(v226_acc, v242_data, v225_data);
          tensorforge::fmacdpp16<0>(v227_acc, v243_data, v30_data);
          tensorforge::fmacdpp16<1>(v227_acc, v243_data, v43_data);
          tensorforge::fmacdpp16<2>(v227_acc, v243_data, v56_data);
          tensorforge::fmacdpp16<3>(v227_acc, v243_data, v69_data);
          tensorforge::fmacdpp16<4>(v227_acc, v243_data, v82_data);
          tensorforge::fmacdpp16<5>(v227_acc, v243_data, v95_data);
          tensorforge::fmacdpp16<6>(v227_acc, v243_data, v108_data);
          tensorforge::fmacdpp16<7>(v227_acc, v243_data, v121_data);
          tensorforge::fmacdpp16<8>(v227_acc, v243_data, v134_data);
          tensorforge::fmacdpp16<9>(v227_acc, v243_data, v147_data);
          tensorforge::fmacdpp16<10>(v227_acc, v243_data, v160_data);
          tensorforge::fmacdpp16<11>(v227_acc, v243_data, v173_data);
          tensorforge::fmacdpp16<12>(v227_acc, v243_data, v186_data);
          tensorforge::fmacdpp16<13>(v227_acc, v243_data, v199_data);
          tensorforge::fmacdpp16<14>(v227_acc, v243_data, v212_data);
          tensorforge::fmacdpp16<15>(v227_acc, v243_data, v225_data);
          tensorforge::fmacdpp16<0>(v228_acc, v244_data, v30_data);
          tensorforge::fmacdpp16<1>(v228_acc, v244_data, v43_data);
          tensorforge::fmacdpp16<2>(v228_acc, v244_data, v56_data);
          tensorforge::fmacdpp16<3>(v228_acc, v244_data, v69_data);
          tensorforge::fmacdpp16<4>(v228_acc, v244_data, v82_data);
          tensorforge::fmacdpp16<5>(v228_acc, v244_data, v95_data);
          tensorforge::fmacdpp16<6>(v228_acc, v244_data, v108_data);
          tensorforge::fmacdpp16<7>(v228_acc, v244_data, v121_data);
          tensorforge::fmacdpp16<8>(v228_acc, v244_data, v134_data);
          tensorforge::fmacdpp16<9>(v228_acc, v244_data, v147_data);
          tensorforge::fmacdpp16<10>(v228_acc, v244_data, v160_data);
          tensorforge::fmacdpp16<11>(v228_acc, v244_data, v173_data);
          tensorforge::fmacdpp16<12>(v228_acc, v244_data, v186_data);
          tensorforge::fmacdpp16<13>(v228_acc, v244_data, v199_data);
          tensorforge::fmacdpp16<14>(v228_acc, v244_data, v212_data);
          tensorforge::fmacdpp16<15>(v228_acc, v244_data, v225_data);
          tensorforge::fmacdpp16<0>(v229_acc, v245_data, v30_data);
          tensorforge::fmacdpp16<1>(v229_acc, v245_data, v43_data);
          tensorforge::fmacdpp16<2>(v229_acc, v245_data, v56_data);
          tensorforge::fmacdpp16<3>(v229_acc, v245_data, v69_data);
          tensorforge::fmacdpp16<4>(v229_acc, v245_data, v82_data);
          tensorforge::fmacdpp16<5>(v229_acc, v245_data, v95_data);
          tensorforge::fmacdpp16<6>(v229_acc, v245_data, v108_data);
          tensorforge::fmacdpp16<7>(v229_acc, v245_data, v121_data);
          tensorforge::fmacdpp16<8>(v229_acc, v245_data, v134_data);
          tensorforge::fmacdpp16<9>(v229_acc, v245_data, v147_data);
          tensorforge::fmacdpp16<10>(v229_acc, v245_data, v160_data);
          tensorforge::fmacdpp16<11>(v229_acc, v245_data, v173_data);
          tensorforge::fmacdpp16<12>(v229_acc, v245_data, v186_data);
          tensorforge::fmacdpp16<13>(v229_acc, v245_data, v199_data);
          tensorforge::fmacdpp16<14>(v229_acc, v245_data, v212_data);
          tensorforge::fmacdpp16<15>(v229_acc, v245_data, v225_data);
          tensorforge::fmacdpp16<0>(v230_acc, v246_data, v30_data);
          tensorforge::fmacdpp16<1>(v230_acc, v246_data, v43_data);
          tensorforge::fmacdpp16<2>(v230_acc, v246_data, v56_data);
          tensorforge::fmacdpp16<3>(v230_acc, v246_data, v69_data);
          tensorforge::fmacdpp16<4>(v230_acc, v246_data, v82_data);
          tensorforge::fmacdpp16<5>(v230_acc, v246_data, v95_data);
          tensorforge::fmacdpp16<6>(v230_acc, v246_data, v108_data);
          tensorforge::fmacdpp16<7>(v230_acc, v246_data, v121_data);
          tensorforge::fmacdpp16<8>(v230_acc, v246_data, v134_data);
          tensorforge::fmacdpp16<9>(v230_acc, v246_data, v147_data);
          tensorforge::fmacdpp16<10>(v230_acc, v246_data, v160_data);
          tensorforge::fmacdpp16<11>(v230_acc, v246_data, v173_data);
          tensorforge::fmacdpp16<12>(v230_acc, v246_data, v186_data);
          tensorforge::fmacdpp16<13>(v230_acc, v246_data, v199_data);
          tensorforge::fmacdpp16<14>(v230_acc, v246_data, v212_data);
          tensorforge::fmacdpp16<15>(v230_acc, v246_data, v225_data);
          tensorforge::fmacdpp16<0>(v231_acc, v247_data, v30_data);
          tensorforge::fmacdpp16<1>(v231_acc, v247_data, v43_data);
          tensorforge::fmacdpp16<2>(v231_acc, v247_data, v56_data);
          tensorforge::fmacdpp16<3>(v231_acc, v247_data, v69_data);
          tensorforge::fmacdpp16<4>(v231_acc, v247_data, v82_data);
          tensorforge::fmacdpp16<5>(v231_acc, v247_data, v95_data);
          tensorforge::fmacdpp16<6>(v231_acc, v247_data, v108_data);
          tensorforge::fmacdpp16<7>(v231_acc, v247_data, v121_data);
          tensorforge::fmacdpp16<8>(v231_acc, v247_data, v134_data);
          tensorforge::fmacdpp16<9>(v231_acc, v247_data, v147_data);
          tensorforge::fmacdpp16<10>(v231_acc, v247_data, v160_data);
          tensorforge::fmacdpp16<11>(v231_acc, v247_data, v173_data);
          tensorforge::fmacdpp16<12>(v231_acc, v247_data, v186_data);
          tensorforge::fmacdpp16<13>(v231_acc, v247_data, v199_data);
          tensorforge::fmacdpp16<14>(v231_acc, v247_data, v212_data);
          tensorforge::fmacdpp16<15>(v231_acc, v247_data, v225_data);
          tensorforge::fmacdpp16<0>(v232_acc, v248_data, v30_data);
          tensorforge::fmacdpp16<1>(v232_acc, v248_data, v43_data);
          tensorforge::fmacdpp16<2>(v232_acc, v248_data, v56_data);
          tensorforge::fmacdpp16<3>(v232_acc, v248_data, v69_data);
          tensorforge::fmacdpp16<4>(v232_acc, v248_data, v82_data);
          tensorforge::fmacdpp16<5>(v232_acc, v248_data, v95_data);
          tensorforge::fmacdpp16<6>(v232_acc, v248_data, v108_data);
          tensorforge::fmacdpp16<7>(v232_acc, v248_data, v121_data);
          tensorforge::fmacdpp16<8>(v232_acc, v248_data, v134_data);
          tensorforge::fmacdpp16<9>(v232_acc, v248_data, v147_data);
          tensorforge::fmacdpp16<10>(v232_acc, v248_data, v160_data);
          tensorforge::fmacdpp16<11>(v232_acc, v248_data, v173_data);
          tensorforge::fmacdpp16<12>(v232_acc, v248_data, v186_data);
          tensorforge::fmacdpp16<13>(v232_acc, v248_data, v199_data);
          tensorforge::fmacdpp16<14>(v232_acc, v248_data, v212_data);
          tensorforge::fmacdpp16<15>(v232_acc, v248_data, v225_data);
          tensorforge::fmacdpp16<0>(v233_acc, v249_data, v30_data);
          tensorforge::fmacdpp16<1>(v233_acc, v249_data, v43_data);
          tensorforge::fmacdpp16<2>(v233_acc, v249_data, v56_data);
          tensorforge::fmacdpp16<3>(v233_acc, v249_data, v69_data);
          tensorforge::fmacdpp16<4>(v233_acc, v249_data, v82_data);
          tensorforge::fmacdpp16<5>(v233_acc, v249_data, v95_data);
          tensorforge::fmacdpp16<6>(v233_acc, v249_data, v108_data);
          tensorforge::fmacdpp16<7>(v233_acc, v249_data, v121_data);
          tensorforge::fmacdpp16<8>(v233_acc, v249_data, v134_data);
          tensorforge::fmacdpp16<9>(v233_acc, v249_data, v147_data);
          tensorforge::fmacdpp16<10>(v233_acc, v249_data, v160_data);
          tensorforge::fmacdpp16<11>(v233_acc, v249_data, v173_data);
          tensorforge::fmacdpp16<12>(v233_acc, v249_data, v186_data);
          tensorforge::fmacdpp16<13>(v233_acc, v249_data, v199_data);
          tensorforge::fmacdpp16<14>(v233_acc, v249_data, v212_data);
          tensorforge::fmacdpp16<15>(v233_acc, v249_data, v225_data);
          tensorforge::fmacdpp16<0>(v234_acc, v250_data, v30_data);
          tensorforge::fmacdpp16<1>(v234_acc, v250_data, v43_data);
          tensorforge::fmacdpp16<2>(v234_acc, v250_data, v56_data);
          tensorforge::fmacdpp16<3>(v234_acc, v250_data, v69_data);
          tensorforge::fmacdpp16<4>(v234_acc, v250_data, v82_data);
          tensorforge::fmacdpp16<5>(v234_acc, v250_data, v95_data);
          tensorforge::fmacdpp16<6>(v234_acc, v250_data, v108_data);
          tensorforge::fmacdpp16<7>(v234_acc, v250_data, v121_data);
          tensorforge::fmacdpp16<8>(v234_acc, v250_data, v134_data);
          tensorforge::fmacdpp16<9>(v234_acc, v250_data, v147_data);
          tensorforge::fmacdpp16<10>(v234_acc, v250_data, v160_data);
          tensorforge::fmacdpp16<11>(v234_acc, v250_data, v173_data);
          tensorforge::fmacdpp16<12>(v234_acc, v250_data, v186_data);
          tensorforge::fmacdpp16<13>(v234_acc, v250_data, v199_data);
          tensorforge::fmacdpp16<14>(v234_acc, v250_data, v212_data);
          tensorforge::fmacdpp16<15>(v234_acc, v250_data, v225_data);
          tensorforge::fmacdpp16<0>(v235_acc, v251_data, v30_data);
          tensorforge::fmacdpp16<1>(v235_acc, v251_data, v43_data);
          tensorforge::fmacdpp16<2>(v235_acc, v251_data, v56_data);
          tensorforge::fmacdpp16<3>(v235_acc, v251_data, v69_data);
          tensorforge::fmacdpp16<4>(v235_acc, v251_data, v82_data);
          tensorforge::fmacdpp16<5>(v235_acc, v251_data, v95_data);
          tensorforge::fmacdpp16<6>(v235_acc, v251_data, v108_data);
          tensorforge::fmacdpp16<7>(v235_acc, v251_data, v121_data);
          tensorforge::fmacdpp16<8>(v235_acc, v251_data, v134_data);
          tensorforge::fmacdpp16<9>(v235_acc, v251_data, v147_data);
          tensorforge::fmacdpp16<10>(v235_acc, v251_data, v160_data);
          tensorforge::fmacdpp16<11>(v235_acc, v251_data, v173_data);
          tensorforge::fmacdpp16<12>(v235_acc, v251_data, v186_data);
          tensorforge::fmacdpp16<13>(v235_acc, v251_data, v199_data);
          tensorforge::fmacdpp16<14>(v235_acc, v251_data, v212_data);
          tensorforge::fmacdpp16<15>(v235_acc, v251_data, v225_data);
          tensorforge::fmacdpp16<0>(v236_acc, v252_data, v30_data);
          tensorforge::fmacdpp16<1>(v236_acc, v252_data, v43_data);
          tensorforge::fmacdpp16<2>(v236_acc, v252_data, v56_data);
          tensorforge::fmacdpp16<3>(v236_acc, v252_data, v69_data);
          tensorforge::fmacdpp16<4>(v236_acc, v252_data, v82_data);
          tensorforge::fmacdpp16<5>(v236_acc, v252_data, v95_data);
          tensorforge::fmacdpp16<6>(v236_acc, v252_data, v108_data);
          tensorforge::fmacdpp16<7>(v236_acc, v252_data, v121_data);
          tensorforge::fmacdpp16<8>(v236_acc, v252_data, v134_data);
          tensorforge::fmacdpp16<9>(v236_acc, v252_data, v147_data);
          tensorforge::fmacdpp16<10>(v236_acc, v252_data, v160_data);
          tensorforge::fmacdpp16<11>(v236_acc, v252_data, v173_data);
          tensorforge::fmacdpp16<12>(v236_acc, v252_data, v186_data);
          tensorforge::fmacdpp16<13>(v236_acc, v252_data, v199_data);
          tensorforge::fmacdpp16<14>(v236_acc, v252_data, v212_data);
          tensorforge::fmacdpp16<15>(v236_acc, v252_data, v225_data);
          tensorforge::fmacdpp16<0>(v237_acc, v253_data, v30_data);
          tensorforge::fmacdpp16<1>(v237_acc, v253_data, v43_data);
          tensorforge::fmacdpp16<2>(v237_acc, v253_data, v56_data);
          tensorforge::fmacdpp16<3>(v237_acc, v253_data, v69_data);
          tensorforge::fmacdpp16<4>(v237_acc, v253_data, v82_data);
          tensorforge::fmacdpp16<5>(v237_acc, v253_data, v95_data);
          tensorforge::fmacdpp16<6>(v237_acc, v253_data, v108_data);
          tensorforge::fmacdpp16<7>(v237_acc, v253_data, v121_data);
          tensorforge::fmacdpp16<8>(v237_acc, v253_data, v134_data);
          tensorforge::fmacdpp16<9>(v237_acc, v253_data, v147_data);
          tensorforge::fmacdpp16<10>(v237_acc, v253_data, v160_data);
          tensorforge::fmacdpp16<11>(v237_acc, v253_data, v173_data);
          tensorforge::fmacdpp16<12>(v237_acc, v253_data, v186_data);
          tensorforge::fmacdpp16<13>(v237_acc, v253_data, v199_data);
          tensorforge::fmacdpp16<14>(v237_acc, v253_data, v212_data);
          tensorforge::fmacdpp16<15>(v237_acc, v253_data, v225_data);
          tensorforge::fmacdpp16<0>(v238_acc, v254_data, v30_data);
          tensorforge::fmacdpp16<1>(v238_acc, v254_data, v43_data);
          tensorforge::fmacdpp16<2>(v238_acc, v254_data, v56_data);
          tensorforge::fmacdpp16<3>(v238_acc, v254_data, v69_data);
          tensorforge::fmacdpp16<4>(v238_acc, v254_data, v82_data);
          tensorforge::fmacdpp16<5>(v238_acc, v254_data, v95_data);
          tensorforge::fmacdpp16<6>(v238_acc, v254_data, v108_data);
          tensorforge::fmacdpp16<7>(v238_acc, v254_data, v121_data);
          tensorforge::fmacdpp16<8>(v238_acc, v254_data, v134_data);
          tensorforge::fmacdpp16<9>(v238_acc, v254_data, v147_data);
          tensorforge::fmacdpp16<10>(v238_acc, v254_data, v160_data);
          tensorforge::fmacdpp16<11>(v238_acc, v254_data, v173_data);
          tensorforge::fmacdpp16<12>(v238_acc, v254_data, v186_data);
          tensorforge::fmacdpp16<13>(v238_acc, v254_data, v199_data);
          tensorforge::fmacdpp16<14>(v238_acc, v254_data, v212_data);
          tensorforge::fmacdpp16<15>(v238_acc, v254_data, v225_data);
          tensorforge::fmacdpp16<0>(v239_acc, v255_data, v30_data);
          tensorforge::fmacdpp16<1>(v239_acc, v255_data, v43_data);
          tensorforge::fmacdpp16<2>(v239_acc, v255_data, v56_data);
          tensorforge::fmacdpp16<3>(v239_acc, v255_data, v69_data);
          tensorforge::fmacdpp16<4>(v239_acc, v255_data, v82_data);
          tensorforge::fmacdpp16<5>(v239_acc, v255_data, v95_data);
          tensorforge::fmacdpp16<6>(v239_acc, v255_data, v108_data);
          tensorforge::fmacdpp16<7>(v239_acc, v255_data, v121_data);
          tensorforge::fmacdpp16<8>(v239_acc, v255_data, v134_data);
          tensorforge::fmacdpp16<9>(v239_acc, v255_data, v147_data);
          tensorforge::fmacdpp16<10>(v239_acc, v255_data, v160_data);
          tensorforge::fmacdpp16<11>(v239_acc, v255_data, v173_data);
          tensorforge::fmacdpp16<12>(v239_acc, v255_data, v186_data);
          tensorforge::fmacdpp16<13>(v239_acc, v255_data, v199_data);
          tensorforge::fmacdpp16<14>(v239_acc, v255_data, v212_data);
          tensorforge::fmacdpp16<15>(v239_acc, v255_data, v225_data);
          tensorforge::fmacdpp16<0>(v240_acc, v256_data, v30_data);
          tensorforge::fmacdpp16<1>(v240_acc, v256_data, v43_data);
          tensorforge::fmacdpp16<2>(v240_acc, v256_data, v56_data);
          tensorforge::fmacdpp16<3>(v240_acc, v256_data, v69_data);
          tensorforge::fmacdpp16<4>(v240_acc, v256_data, v82_data);
          tensorforge::fmacdpp16<5>(v240_acc, v256_data, v95_data);
          tensorforge::fmacdpp16<6>(v240_acc, v256_data, v108_data);
          tensorforge::fmacdpp16<7>(v240_acc, v256_data, v121_data);
          tensorforge::fmacdpp16<8>(v240_acc, v256_data, v134_data);
          tensorforge::fmacdpp16<9>(v240_acc, v256_data, v147_data);
          tensorforge::fmacdpp16<10>(v240_acc, v256_data, v160_data);
          tensorforge::fmacdpp16<11>(v240_acc, v256_data, v173_data);
          tensorforge::fmacdpp16<12>(v240_acc, v256_data, v186_data);
          tensorforge::fmacdpp16<13>(v240_acc, v256_data, v199_data);
          tensorforge::fmacdpp16<14>(v240_acc, v256_data, v212_data);
          tensorforge::fmacdpp16<15>(v240_acc, v256_data, v225_data);
          tensorforge::fmacdpp16<0>(v241_acc, v257_data, v30_data);
          tensorforge::fmacdpp16<1>(v241_acc, v257_data, v43_data);
          tensorforge::fmacdpp16<2>(v241_acc, v257_data, v56_data);
          tensorforge::fmacdpp16<3>(v241_acc, v257_data, v69_data);
          tensorforge::fmacdpp16<4>(v241_acc, v257_data, v82_data);
          tensorforge::fmacdpp16<5>(v241_acc, v257_data, v95_data);
          tensorforge::fmacdpp16<6>(v241_acc, v257_data, v108_data);
          tensorforge::fmacdpp16<7>(v241_acc, v257_data, v121_data);
          tensorforge::fmacdpp16<8>(v241_acc, v257_data, v134_data);
          tensorforge::fmacdpp16<9>(v241_acc, v257_data, v147_data);
          tensorforge::fmacdpp16<10>(v241_acc, v257_data, v160_data);
          tensorforge::fmacdpp16<11>(v241_acc, v257_data, v173_data);
          tensorforge::fmacdpp16<12>(v241_acc, v257_data, v186_data);
          tensorforge::fmacdpp16<13>(v241_acc, v257_data, v199_data);
          tensorforge::fmacdpp16<14>(v241_acc, v257_data, v212_data);
          tensorforge::fmacdpp16<15>(v241_acc, v257_data, v225_data);
          ir1[0] = v226_acc;
          ir1[1] = v227_acc;
          ir1[2] = v228_acc;
          ir1[3] = v229_acc;
          ir1[4] = v230_acc;
          ir1[5] = v231_acc;
          ir1[6] = v232_acc;
          ir1[7] = v233_acc;
          ir1[8] = v234_acc;
          ir1[9] = v235_acc;
          ir1[10] = v236_acc;
          ir1[11] = v237_acc;
          ir1[12] = v238_acc;
          ir1[13] = v239_acc;
          ir1[14] = v240_acc;
          ir1[15] = v241_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v261_i0 = 0; v261_i0 < 1; ++v261_i0) {
            int32_t v270_lead = v20_lane + (v261_i0 * 16);
            #pragma unroll
            for (int32_t v262_i1 = 0; v262_i1 < 16; ++v262_i1) {
              int32_t v263_a = v261_i0 + v262_i1;
              double v265_data = r1[(v261_i0 + v262_i1)];
              int32_t v272_a = v270_lead + (v262_i1 * 16);
              glb_m0[v272_a] = v265_data;
            }
          }
          ;
        }
      }
    }
  }
}

