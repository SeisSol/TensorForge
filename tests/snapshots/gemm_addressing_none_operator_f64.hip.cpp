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
          {
            // r0 = load{g>r}(glb_m2);
            double v0 = glb_m2[0 + threadIdx.x * 1];
            r0[0] = v0;
            double v16 = glb_m2[16 + threadIdx.x * 1];
            r0[1] = v16;
            double v32 = glb_m2[32 + threadIdx.x * 1];
            r0[2] = v32;
            double v48 = glb_m2[48 + threadIdx.x * 1];
            r0[3] = v48;
            double v64 = glb_m2[64 + threadIdx.x * 1];
            r0[4] = v64;
            double v80 = glb_m2[80 + threadIdx.x * 1];
            r0[5] = v80;
            double v96 = glb_m2[96 + threadIdx.x * 1];
            r0[6] = v96;
            double v112 = glb_m2[112 + threadIdx.x * 1];
            r0[7] = v112;
            double v128 = glb_m2[128 + threadIdx.x * 1];
            r0[8] = v128;
            double v144 = glb_m2[144 + threadIdx.x * 1];
            r0[9] = v144;
            double v160 = glb_m2[160 + threadIdx.x * 1];
            r0[10] = v160;
            double v176 = glb_m2[176 + threadIdx.x * 1];
            r0[11] = v176;
            double v192 = glb_m2[192 + threadIdx.x * 1];
            r0[12] = v192;
            double v208 = glb_m2[208 + threadIdx.x * 1];
            r0[13] = v208;
            double v224 = glb_m2[224 + threadIdx.x * 1];
            r0[14] = v224;
            double v240 = glb_m2[240 + threadIdx.x * 1];
            r0[15] = v240;
          }
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          auto& ir1 = r1;
          int32_t v2_lane = threadIdx.x % 16;
          int32_t v5_a = v2_lane + 0;
          double v12_data = glb_m1[v2_lane];
          int32_t v18_a = v2_lane + 16;
          double v25_data = glb_m1[(v2_lane + 16)];
          int32_t v31_a = v2_lane + 32;
          double v38_data = glb_m1[(v2_lane + 32)];
          int32_t v44_a = v2_lane + 48;
          double v51_data = glb_m1[(v2_lane + 48)];
          int32_t v57_a = v2_lane + 64;
          double v64_data = glb_m1[(v2_lane + 64)];
          int32_t v70_a = v2_lane + 80;
          double v77_data = glb_m1[(v2_lane + 80)];
          int32_t v83_a = v2_lane + 96;
          double v90_data = glb_m1[(v2_lane + 96)];
          int32_t v96_a = v2_lane + 112;
          double v103_data = glb_m1[(v2_lane + 112)];
          int32_t v109_a = v2_lane + 128;
          double v116_data = glb_m1[(v2_lane + 128)];
          int32_t v122_a = v2_lane + 144;
          double v129_data = glb_m1[(v2_lane + 144)];
          int32_t v135_a = v2_lane + 160;
          double v142_data = glb_m1[(v2_lane + 160)];
          int32_t v148_a = v2_lane + 176;
          double v155_data = glb_m1[(v2_lane + 176)];
          int32_t v161_a = v2_lane + 192;
          double v168_data = glb_m1[(v2_lane + 192)];
          int32_t v174_a = v2_lane + 208;
          double v181_data = glb_m1[(v2_lane + 208)];
          int32_t v187_a = v2_lane + 224;
          double v194_data = glb_m1[(v2_lane + 224)];
          int32_t v200_a = v2_lane + 240;
          double v207_data = glb_m1[(v2_lane + 240)];
          double v208_acc{};
          double v209_acc{};
          double v210_acc{};
          double v211_acc{};
          double v212_acc{};
          double v213_acc{};
          double v214_acc{};
          double v215_acc{};
          double v216_acc{};
          double v217_acc{};
          double v218_acc{};
          double v219_acc{};
          double v220_acc{};
          double v221_acc{};
          double v222_acc{};
          double v223_acc{};
          double v224_data = r0[0];
          double v225_data = r0[1];
          double v226_data = r0[2];
          double v227_data = r0[3];
          double v228_data = r0[4];
          double v229_data = r0[5];
          double v230_data = r0[6];
          double v231_data = r0[7];
          double v232_data = r0[8];
          double v233_data = r0[9];
          double v234_data = r0[10];
          double v235_data = r0[11];
          double v236_data = r0[12];
          double v237_data = r0[13];
          double v238_data = r0[14];
          double v239_data = r0[15];
          tensorforge::fmacdpp16<0>(v208_acc, v224_data, v12_data);
          tensorforge::fmacdpp16<1>(v208_acc, v224_data, v25_data);
          tensorforge::fmacdpp16<2>(v208_acc, v224_data, v38_data);
          tensorforge::fmacdpp16<3>(v208_acc, v224_data, v51_data);
          tensorforge::fmacdpp16<4>(v208_acc, v224_data, v64_data);
          tensorforge::fmacdpp16<5>(v208_acc, v224_data, v77_data);
          tensorforge::fmacdpp16<6>(v208_acc, v224_data, v90_data);
          tensorforge::fmacdpp16<7>(v208_acc, v224_data, v103_data);
          tensorforge::fmacdpp16<8>(v208_acc, v224_data, v116_data);
          tensorforge::fmacdpp16<9>(v208_acc, v224_data, v129_data);
          tensorforge::fmacdpp16<10>(v208_acc, v224_data, v142_data);
          tensorforge::fmacdpp16<11>(v208_acc, v224_data, v155_data);
          tensorforge::fmacdpp16<12>(v208_acc, v224_data, v168_data);
          tensorforge::fmacdpp16<13>(v208_acc, v224_data, v181_data);
          tensorforge::fmacdpp16<14>(v208_acc, v224_data, v194_data);
          tensorforge::fmacdpp16<15>(v208_acc, v224_data, v207_data);
          tensorforge::fmacdpp16<0>(v209_acc, v225_data, v12_data);
          tensorforge::fmacdpp16<1>(v209_acc, v225_data, v25_data);
          tensorforge::fmacdpp16<2>(v209_acc, v225_data, v38_data);
          tensorforge::fmacdpp16<3>(v209_acc, v225_data, v51_data);
          tensorforge::fmacdpp16<4>(v209_acc, v225_data, v64_data);
          tensorforge::fmacdpp16<5>(v209_acc, v225_data, v77_data);
          tensorforge::fmacdpp16<6>(v209_acc, v225_data, v90_data);
          tensorforge::fmacdpp16<7>(v209_acc, v225_data, v103_data);
          tensorforge::fmacdpp16<8>(v209_acc, v225_data, v116_data);
          tensorforge::fmacdpp16<9>(v209_acc, v225_data, v129_data);
          tensorforge::fmacdpp16<10>(v209_acc, v225_data, v142_data);
          tensorforge::fmacdpp16<11>(v209_acc, v225_data, v155_data);
          tensorforge::fmacdpp16<12>(v209_acc, v225_data, v168_data);
          tensorforge::fmacdpp16<13>(v209_acc, v225_data, v181_data);
          tensorforge::fmacdpp16<14>(v209_acc, v225_data, v194_data);
          tensorforge::fmacdpp16<15>(v209_acc, v225_data, v207_data);
          tensorforge::fmacdpp16<0>(v210_acc, v226_data, v12_data);
          tensorforge::fmacdpp16<1>(v210_acc, v226_data, v25_data);
          tensorforge::fmacdpp16<2>(v210_acc, v226_data, v38_data);
          tensorforge::fmacdpp16<3>(v210_acc, v226_data, v51_data);
          tensorforge::fmacdpp16<4>(v210_acc, v226_data, v64_data);
          tensorforge::fmacdpp16<5>(v210_acc, v226_data, v77_data);
          tensorforge::fmacdpp16<6>(v210_acc, v226_data, v90_data);
          tensorforge::fmacdpp16<7>(v210_acc, v226_data, v103_data);
          tensorforge::fmacdpp16<8>(v210_acc, v226_data, v116_data);
          tensorforge::fmacdpp16<9>(v210_acc, v226_data, v129_data);
          tensorforge::fmacdpp16<10>(v210_acc, v226_data, v142_data);
          tensorforge::fmacdpp16<11>(v210_acc, v226_data, v155_data);
          tensorforge::fmacdpp16<12>(v210_acc, v226_data, v168_data);
          tensorforge::fmacdpp16<13>(v210_acc, v226_data, v181_data);
          tensorforge::fmacdpp16<14>(v210_acc, v226_data, v194_data);
          tensorforge::fmacdpp16<15>(v210_acc, v226_data, v207_data);
          tensorforge::fmacdpp16<0>(v211_acc, v227_data, v12_data);
          tensorforge::fmacdpp16<1>(v211_acc, v227_data, v25_data);
          tensorforge::fmacdpp16<2>(v211_acc, v227_data, v38_data);
          tensorforge::fmacdpp16<3>(v211_acc, v227_data, v51_data);
          tensorforge::fmacdpp16<4>(v211_acc, v227_data, v64_data);
          tensorforge::fmacdpp16<5>(v211_acc, v227_data, v77_data);
          tensorforge::fmacdpp16<6>(v211_acc, v227_data, v90_data);
          tensorforge::fmacdpp16<7>(v211_acc, v227_data, v103_data);
          tensorforge::fmacdpp16<8>(v211_acc, v227_data, v116_data);
          tensorforge::fmacdpp16<9>(v211_acc, v227_data, v129_data);
          tensorforge::fmacdpp16<10>(v211_acc, v227_data, v142_data);
          tensorforge::fmacdpp16<11>(v211_acc, v227_data, v155_data);
          tensorforge::fmacdpp16<12>(v211_acc, v227_data, v168_data);
          tensorforge::fmacdpp16<13>(v211_acc, v227_data, v181_data);
          tensorforge::fmacdpp16<14>(v211_acc, v227_data, v194_data);
          tensorforge::fmacdpp16<15>(v211_acc, v227_data, v207_data);
          tensorforge::fmacdpp16<0>(v212_acc, v228_data, v12_data);
          tensorforge::fmacdpp16<1>(v212_acc, v228_data, v25_data);
          tensorforge::fmacdpp16<2>(v212_acc, v228_data, v38_data);
          tensorforge::fmacdpp16<3>(v212_acc, v228_data, v51_data);
          tensorforge::fmacdpp16<4>(v212_acc, v228_data, v64_data);
          tensorforge::fmacdpp16<5>(v212_acc, v228_data, v77_data);
          tensorforge::fmacdpp16<6>(v212_acc, v228_data, v90_data);
          tensorforge::fmacdpp16<7>(v212_acc, v228_data, v103_data);
          tensorforge::fmacdpp16<8>(v212_acc, v228_data, v116_data);
          tensorforge::fmacdpp16<9>(v212_acc, v228_data, v129_data);
          tensorforge::fmacdpp16<10>(v212_acc, v228_data, v142_data);
          tensorforge::fmacdpp16<11>(v212_acc, v228_data, v155_data);
          tensorforge::fmacdpp16<12>(v212_acc, v228_data, v168_data);
          tensorforge::fmacdpp16<13>(v212_acc, v228_data, v181_data);
          tensorforge::fmacdpp16<14>(v212_acc, v228_data, v194_data);
          tensorforge::fmacdpp16<15>(v212_acc, v228_data, v207_data);
          tensorforge::fmacdpp16<0>(v213_acc, v229_data, v12_data);
          tensorforge::fmacdpp16<1>(v213_acc, v229_data, v25_data);
          tensorforge::fmacdpp16<2>(v213_acc, v229_data, v38_data);
          tensorforge::fmacdpp16<3>(v213_acc, v229_data, v51_data);
          tensorforge::fmacdpp16<4>(v213_acc, v229_data, v64_data);
          tensorforge::fmacdpp16<5>(v213_acc, v229_data, v77_data);
          tensorforge::fmacdpp16<6>(v213_acc, v229_data, v90_data);
          tensorforge::fmacdpp16<7>(v213_acc, v229_data, v103_data);
          tensorforge::fmacdpp16<8>(v213_acc, v229_data, v116_data);
          tensorforge::fmacdpp16<9>(v213_acc, v229_data, v129_data);
          tensorforge::fmacdpp16<10>(v213_acc, v229_data, v142_data);
          tensorforge::fmacdpp16<11>(v213_acc, v229_data, v155_data);
          tensorforge::fmacdpp16<12>(v213_acc, v229_data, v168_data);
          tensorforge::fmacdpp16<13>(v213_acc, v229_data, v181_data);
          tensorforge::fmacdpp16<14>(v213_acc, v229_data, v194_data);
          tensorforge::fmacdpp16<15>(v213_acc, v229_data, v207_data);
          tensorforge::fmacdpp16<0>(v214_acc, v230_data, v12_data);
          tensorforge::fmacdpp16<1>(v214_acc, v230_data, v25_data);
          tensorforge::fmacdpp16<2>(v214_acc, v230_data, v38_data);
          tensorforge::fmacdpp16<3>(v214_acc, v230_data, v51_data);
          tensorforge::fmacdpp16<4>(v214_acc, v230_data, v64_data);
          tensorforge::fmacdpp16<5>(v214_acc, v230_data, v77_data);
          tensorforge::fmacdpp16<6>(v214_acc, v230_data, v90_data);
          tensorforge::fmacdpp16<7>(v214_acc, v230_data, v103_data);
          tensorforge::fmacdpp16<8>(v214_acc, v230_data, v116_data);
          tensorforge::fmacdpp16<9>(v214_acc, v230_data, v129_data);
          tensorforge::fmacdpp16<10>(v214_acc, v230_data, v142_data);
          tensorforge::fmacdpp16<11>(v214_acc, v230_data, v155_data);
          tensorforge::fmacdpp16<12>(v214_acc, v230_data, v168_data);
          tensorforge::fmacdpp16<13>(v214_acc, v230_data, v181_data);
          tensorforge::fmacdpp16<14>(v214_acc, v230_data, v194_data);
          tensorforge::fmacdpp16<15>(v214_acc, v230_data, v207_data);
          tensorforge::fmacdpp16<0>(v215_acc, v231_data, v12_data);
          tensorforge::fmacdpp16<1>(v215_acc, v231_data, v25_data);
          tensorforge::fmacdpp16<2>(v215_acc, v231_data, v38_data);
          tensorforge::fmacdpp16<3>(v215_acc, v231_data, v51_data);
          tensorforge::fmacdpp16<4>(v215_acc, v231_data, v64_data);
          tensorforge::fmacdpp16<5>(v215_acc, v231_data, v77_data);
          tensorforge::fmacdpp16<6>(v215_acc, v231_data, v90_data);
          tensorforge::fmacdpp16<7>(v215_acc, v231_data, v103_data);
          tensorforge::fmacdpp16<8>(v215_acc, v231_data, v116_data);
          tensorforge::fmacdpp16<9>(v215_acc, v231_data, v129_data);
          tensorforge::fmacdpp16<10>(v215_acc, v231_data, v142_data);
          tensorforge::fmacdpp16<11>(v215_acc, v231_data, v155_data);
          tensorforge::fmacdpp16<12>(v215_acc, v231_data, v168_data);
          tensorforge::fmacdpp16<13>(v215_acc, v231_data, v181_data);
          tensorforge::fmacdpp16<14>(v215_acc, v231_data, v194_data);
          tensorforge::fmacdpp16<15>(v215_acc, v231_data, v207_data);
          tensorforge::fmacdpp16<0>(v216_acc, v232_data, v12_data);
          tensorforge::fmacdpp16<1>(v216_acc, v232_data, v25_data);
          tensorforge::fmacdpp16<2>(v216_acc, v232_data, v38_data);
          tensorforge::fmacdpp16<3>(v216_acc, v232_data, v51_data);
          tensorforge::fmacdpp16<4>(v216_acc, v232_data, v64_data);
          tensorforge::fmacdpp16<5>(v216_acc, v232_data, v77_data);
          tensorforge::fmacdpp16<6>(v216_acc, v232_data, v90_data);
          tensorforge::fmacdpp16<7>(v216_acc, v232_data, v103_data);
          tensorforge::fmacdpp16<8>(v216_acc, v232_data, v116_data);
          tensorforge::fmacdpp16<9>(v216_acc, v232_data, v129_data);
          tensorforge::fmacdpp16<10>(v216_acc, v232_data, v142_data);
          tensorforge::fmacdpp16<11>(v216_acc, v232_data, v155_data);
          tensorforge::fmacdpp16<12>(v216_acc, v232_data, v168_data);
          tensorforge::fmacdpp16<13>(v216_acc, v232_data, v181_data);
          tensorforge::fmacdpp16<14>(v216_acc, v232_data, v194_data);
          tensorforge::fmacdpp16<15>(v216_acc, v232_data, v207_data);
          tensorforge::fmacdpp16<0>(v217_acc, v233_data, v12_data);
          tensorforge::fmacdpp16<1>(v217_acc, v233_data, v25_data);
          tensorforge::fmacdpp16<2>(v217_acc, v233_data, v38_data);
          tensorforge::fmacdpp16<3>(v217_acc, v233_data, v51_data);
          tensorforge::fmacdpp16<4>(v217_acc, v233_data, v64_data);
          tensorforge::fmacdpp16<5>(v217_acc, v233_data, v77_data);
          tensorforge::fmacdpp16<6>(v217_acc, v233_data, v90_data);
          tensorforge::fmacdpp16<7>(v217_acc, v233_data, v103_data);
          tensorforge::fmacdpp16<8>(v217_acc, v233_data, v116_data);
          tensorforge::fmacdpp16<9>(v217_acc, v233_data, v129_data);
          tensorforge::fmacdpp16<10>(v217_acc, v233_data, v142_data);
          tensorforge::fmacdpp16<11>(v217_acc, v233_data, v155_data);
          tensorforge::fmacdpp16<12>(v217_acc, v233_data, v168_data);
          tensorforge::fmacdpp16<13>(v217_acc, v233_data, v181_data);
          tensorforge::fmacdpp16<14>(v217_acc, v233_data, v194_data);
          tensorforge::fmacdpp16<15>(v217_acc, v233_data, v207_data);
          tensorforge::fmacdpp16<0>(v218_acc, v234_data, v12_data);
          tensorforge::fmacdpp16<1>(v218_acc, v234_data, v25_data);
          tensorforge::fmacdpp16<2>(v218_acc, v234_data, v38_data);
          tensorforge::fmacdpp16<3>(v218_acc, v234_data, v51_data);
          tensorforge::fmacdpp16<4>(v218_acc, v234_data, v64_data);
          tensorforge::fmacdpp16<5>(v218_acc, v234_data, v77_data);
          tensorforge::fmacdpp16<6>(v218_acc, v234_data, v90_data);
          tensorforge::fmacdpp16<7>(v218_acc, v234_data, v103_data);
          tensorforge::fmacdpp16<8>(v218_acc, v234_data, v116_data);
          tensorforge::fmacdpp16<9>(v218_acc, v234_data, v129_data);
          tensorforge::fmacdpp16<10>(v218_acc, v234_data, v142_data);
          tensorforge::fmacdpp16<11>(v218_acc, v234_data, v155_data);
          tensorforge::fmacdpp16<12>(v218_acc, v234_data, v168_data);
          tensorforge::fmacdpp16<13>(v218_acc, v234_data, v181_data);
          tensorforge::fmacdpp16<14>(v218_acc, v234_data, v194_data);
          tensorforge::fmacdpp16<15>(v218_acc, v234_data, v207_data);
          tensorforge::fmacdpp16<0>(v219_acc, v235_data, v12_data);
          tensorforge::fmacdpp16<1>(v219_acc, v235_data, v25_data);
          tensorforge::fmacdpp16<2>(v219_acc, v235_data, v38_data);
          tensorforge::fmacdpp16<3>(v219_acc, v235_data, v51_data);
          tensorforge::fmacdpp16<4>(v219_acc, v235_data, v64_data);
          tensorforge::fmacdpp16<5>(v219_acc, v235_data, v77_data);
          tensorforge::fmacdpp16<6>(v219_acc, v235_data, v90_data);
          tensorforge::fmacdpp16<7>(v219_acc, v235_data, v103_data);
          tensorforge::fmacdpp16<8>(v219_acc, v235_data, v116_data);
          tensorforge::fmacdpp16<9>(v219_acc, v235_data, v129_data);
          tensorforge::fmacdpp16<10>(v219_acc, v235_data, v142_data);
          tensorforge::fmacdpp16<11>(v219_acc, v235_data, v155_data);
          tensorforge::fmacdpp16<12>(v219_acc, v235_data, v168_data);
          tensorforge::fmacdpp16<13>(v219_acc, v235_data, v181_data);
          tensorforge::fmacdpp16<14>(v219_acc, v235_data, v194_data);
          tensorforge::fmacdpp16<15>(v219_acc, v235_data, v207_data);
          tensorforge::fmacdpp16<0>(v220_acc, v236_data, v12_data);
          tensorforge::fmacdpp16<1>(v220_acc, v236_data, v25_data);
          tensorforge::fmacdpp16<2>(v220_acc, v236_data, v38_data);
          tensorforge::fmacdpp16<3>(v220_acc, v236_data, v51_data);
          tensorforge::fmacdpp16<4>(v220_acc, v236_data, v64_data);
          tensorforge::fmacdpp16<5>(v220_acc, v236_data, v77_data);
          tensorforge::fmacdpp16<6>(v220_acc, v236_data, v90_data);
          tensorforge::fmacdpp16<7>(v220_acc, v236_data, v103_data);
          tensorforge::fmacdpp16<8>(v220_acc, v236_data, v116_data);
          tensorforge::fmacdpp16<9>(v220_acc, v236_data, v129_data);
          tensorforge::fmacdpp16<10>(v220_acc, v236_data, v142_data);
          tensorforge::fmacdpp16<11>(v220_acc, v236_data, v155_data);
          tensorforge::fmacdpp16<12>(v220_acc, v236_data, v168_data);
          tensorforge::fmacdpp16<13>(v220_acc, v236_data, v181_data);
          tensorforge::fmacdpp16<14>(v220_acc, v236_data, v194_data);
          tensorforge::fmacdpp16<15>(v220_acc, v236_data, v207_data);
          tensorforge::fmacdpp16<0>(v221_acc, v237_data, v12_data);
          tensorforge::fmacdpp16<1>(v221_acc, v237_data, v25_data);
          tensorforge::fmacdpp16<2>(v221_acc, v237_data, v38_data);
          tensorforge::fmacdpp16<3>(v221_acc, v237_data, v51_data);
          tensorforge::fmacdpp16<4>(v221_acc, v237_data, v64_data);
          tensorforge::fmacdpp16<5>(v221_acc, v237_data, v77_data);
          tensorforge::fmacdpp16<6>(v221_acc, v237_data, v90_data);
          tensorforge::fmacdpp16<7>(v221_acc, v237_data, v103_data);
          tensorforge::fmacdpp16<8>(v221_acc, v237_data, v116_data);
          tensorforge::fmacdpp16<9>(v221_acc, v237_data, v129_data);
          tensorforge::fmacdpp16<10>(v221_acc, v237_data, v142_data);
          tensorforge::fmacdpp16<11>(v221_acc, v237_data, v155_data);
          tensorforge::fmacdpp16<12>(v221_acc, v237_data, v168_data);
          tensorforge::fmacdpp16<13>(v221_acc, v237_data, v181_data);
          tensorforge::fmacdpp16<14>(v221_acc, v237_data, v194_data);
          tensorforge::fmacdpp16<15>(v221_acc, v237_data, v207_data);
          tensorforge::fmacdpp16<0>(v222_acc, v238_data, v12_data);
          tensorforge::fmacdpp16<1>(v222_acc, v238_data, v25_data);
          tensorforge::fmacdpp16<2>(v222_acc, v238_data, v38_data);
          tensorforge::fmacdpp16<3>(v222_acc, v238_data, v51_data);
          tensorforge::fmacdpp16<4>(v222_acc, v238_data, v64_data);
          tensorforge::fmacdpp16<5>(v222_acc, v238_data, v77_data);
          tensorforge::fmacdpp16<6>(v222_acc, v238_data, v90_data);
          tensorforge::fmacdpp16<7>(v222_acc, v238_data, v103_data);
          tensorforge::fmacdpp16<8>(v222_acc, v238_data, v116_data);
          tensorforge::fmacdpp16<9>(v222_acc, v238_data, v129_data);
          tensorforge::fmacdpp16<10>(v222_acc, v238_data, v142_data);
          tensorforge::fmacdpp16<11>(v222_acc, v238_data, v155_data);
          tensorforge::fmacdpp16<12>(v222_acc, v238_data, v168_data);
          tensorforge::fmacdpp16<13>(v222_acc, v238_data, v181_data);
          tensorforge::fmacdpp16<14>(v222_acc, v238_data, v194_data);
          tensorforge::fmacdpp16<15>(v222_acc, v238_data, v207_data);
          tensorforge::fmacdpp16<0>(v223_acc, v239_data, v12_data);
          tensorforge::fmacdpp16<1>(v223_acc, v239_data, v25_data);
          tensorforge::fmacdpp16<2>(v223_acc, v239_data, v38_data);
          tensorforge::fmacdpp16<3>(v223_acc, v239_data, v51_data);
          tensorforge::fmacdpp16<4>(v223_acc, v239_data, v64_data);
          tensorforge::fmacdpp16<5>(v223_acc, v239_data, v77_data);
          tensorforge::fmacdpp16<6>(v223_acc, v239_data, v90_data);
          tensorforge::fmacdpp16<7>(v223_acc, v239_data, v103_data);
          tensorforge::fmacdpp16<8>(v223_acc, v239_data, v116_data);
          tensorforge::fmacdpp16<9>(v223_acc, v239_data, v129_data);
          tensorforge::fmacdpp16<10>(v223_acc, v239_data, v142_data);
          tensorforge::fmacdpp16<11>(v223_acc, v239_data, v155_data);
          tensorforge::fmacdpp16<12>(v223_acc, v239_data, v168_data);
          tensorforge::fmacdpp16<13>(v223_acc, v239_data, v181_data);
          tensorforge::fmacdpp16<14>(v223_acc, v239_data, v194_data);
          tensorforge::fmacdpp16<15>(v223_acc, v239_data, v207_data);
          ir1[0] = v208_acc;
          ir1[1] = v209_acc;
          ir1[2] = v210_acc;
          ir1[3] = v211_acc;
          ir1[4] = v212_acc;
          ir1[5] = v213_acc;
          ir1[6] = v214_acc;
          ir1[7] = v215_acc;
          ir1[8] = v216_acc;
          ir1[9] = v217_acc;
          ir1[10] = v218_acc;
          ir1[11] = v219_acc;
          ir1[12] = v220_acc;
          ir1[13] = v221_acc;
          ir1[14] = v222_acc;
          ir1[15] = v223_acc;
          // glb_m0 = store{r>g}(r1);
          int32_t v242_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v243_i0 = 0; v243_i0 < 1; ++v243_i0) {
            int32_t v252_lead = v242_lead + (v243_i0 * 16);
            #pragma unroll
            for (int32_t v244_i1 = 0; v244_i1 < 16; ++v244_i1) {
              int32_t v245_a = v243_i0 + v244_i1;
              double v247_data = r1[(v243_i0 + v244_i1)];
              int32_t v254_a = v252_lead + (v244_i1 * 16);
              glb_m0[v254_a] = v247_data;
            }
          }
          ;
        }
      }
    }
  }
}

