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
          double v6_data = glb_m1[v5_a];
          int32_t v12_a = v2_lane + 16;
          double v13_data = glb_m1[v12_a];
          int32_t v19_a = v2_lane + 32;
          double v20_data = glb_m1[v19_a];
          int32_t v26_a = v2_lane + 48;
          double v27_data = glb_m1[v26_a];
          int32_t v33_a = v2_lane + 64;
          double v34_data = glb_m1[v33_a];
          int32_t v40_a = v2_lane + 80;
          double v41_data = glb_m1[v40_a];
          int32_t v47_a = v2_lane + 96;
          double v48_data = glb_m1[v47_a];
          int32_t v54_a = v2_lane + 112;
          double v55_data = glb_m1[v54_a];
          int32_t v61_a = v2_lane + 128;
          double v62_data = glb_m1[v61_a];
          int32_t v68_a = v2_lane + 144;
          double v69_data = glb_m1[v68_a];
          int32_t v75_a = v2_lane + 160;
          double v76_data = glb_m1[v75_a];
          int32_t v82_a = v2_lane + 176;
          double v83_data = glb_m1[v82_a];
          int32_t v89_a = v2_lane + 192;
          double v90_data = glb_m1[v89_a];
          int32_t v96_a = v2_lane + 208;
          double v97_data = glb_m1[v96_a];
          int32_t v103_a = v2_lane + 224;
          double v104_data = glb_m1[v103_a];
          int32_t v110_a = v2_lane + 240;
          double v111_data = glb_m1[v110_a];
          double v112_acc{};
          double v113_acc{};
          double v114_acc{};
          double v115_acc{};
          double v116_acc{};
          double v117_acc{};
          double v118_acc{};
          double v119_acc{};
          double v120_acc{};
          double v121_acc{};
          double v122_acc{};
          double v123_acc{};
          double v124_acc{};
          double v125_acc{};
          double v126_acc{};
          double v127_acc{};
          double v128_data = r0[0];
          double v129_data = r0[1];
          double v130_data = r0[2];
          double v131_data = r0[3];
          double v132_data = r0[4];
          double v133_data = r0[5];
          double v134_data = r0[6];
          double v135_data = r0[7];
          double v136_data = r0[8];
          double v137_data = r0[9];
          double v138_data = r0[10];
          double v139_data = r0[11];
          double v140_data = r0[12];
          double v141_data = r0[13];
          double v142_data = r0[14];
          double v143_data = r0[15];
          tensorforge::fmacdpp16<0>(v112_acc, v128_data, v6_data);
          tensorforge::fmacdpp16<1>(v112_acc, v128_data, v13_data);
          tensorforge::fmacdpp16<2>(v112_acc, v128_data, v20_data);
          tensorforge::fmacdpp16<3>(v112_acc, v128_data, v27_data);
          tensorforge::fmacdpp16<4>(v112_acc, v128_data, v34_data);
          tensorforge::fmacdpp16<5>(v112_acc, v128_data, v41_data);
          tensorforge::fmacdpp16<6>(v112_acc, v128_data, v48_data);
          tensorforge::fmacdpp16<7>(v112_acc, v128_data, v55_data);
          tensorforge::fmacdpp16<8>(v112_acc, v128_data, v62_data);
          tensorforge::fmacdpp16<9>(v112_acc, v128_data, v69_data);
          tensorforge::fmacdpp16<10>(v112_acc, v128_data, v76_data);
          tensorforge::fmacdpp16<11>(v112_acc, v128_data, v83_data);
          tensorforge::fmacdpp16<12>(v112_acc, v128_data, v90_data);
          tensorforge::fmacdpp16<13>(v112_acc, v128_data, v97_data);
          tensorforge::fmacdpp16<14>(v112_acc, v128_data, v104_data);
          tensorforge::fmacdpp16<15>(v112_acc, v128_data, v111_data);
          tensorforge::fmacdpp16<0>(v113_acc, v129_data, v6_data);
          tensorforge::fmacdpp16<1>(v113_acc, v129_data, v13_data);
          tensorforge::fmacdpp16<2>(v113_acc, v129_data, v20_data);
          tensorforge::fmacdpp16<3>(v113_acc, v129_data, v27_data);
          tensorforge::fmacdpp16<4>(v113_acc, v129_data, v34_data);
          tensorforge::fmacdpp16<5>(v113_acc, v129_data, v41_data);
          tensorforge::fmacdpp16<6>(v113_acc, v129_data, v48_data);
          tensorforge::fmacdpp16<7>(v113_acc, v129_data, v55_data);
          tensorforge::fmacdpp16<8>(v113_acc, v129_data, v62_data);
          tensorforge::fmacdpp16<9>(v113_acc, v129_data, v69_data);
          tensorforge::fmacdpp16<10>(v113_acc, v129_data, v76_data);
          tensorforge::fmacdpp16<11>(v113_acc, v129_data, v83_data);
          tensorforge::fmacdpp16<12>(v113_acc, v129_data, v90_data);
          tensorforge::fmacdpp16<13>(v113_acc, v129_data, v97_data);
          tensorforge::fmacdpp16<14>(v113_acc, v129_data, v104_data);
          tensorforge::fmacdpp16<15>(v113_acc, v129_data, v111_data);
          tensorforge::fmacdpp16<0>(v114_acc, v130_data, v6_data);
          tensorforge::fmacdpp16<1>(v114_acc, v130_data, v13_data);
          tensorforge::fmacdpp16<2>(v114_acc, v130_data, v20_data);
          tensorforge::fmacdpp16<3>(v114_acc, v130_data, v27_data);
          tensorforge::fmacdpp16<4>(v114_acc, v130_data, v34_data);
          tensorforge::fmacdpp16<5>(v114_acc, v130_data, v41_data);
          tensorforge::fmacdpp16<6>(v114_acc, v130_data, v48_data);
          tensorforge::fmacdpp16<7>(v114_acc, v130_data, v55_data);
          tensorforge::fmacdpp16<8>(v114_acc, v130_data, v62_data);
          tensorforge::fmacdpp16<9>(v114_acc, v130_data, v69_data);
          tensorforge::fmacdpp16<10>(v114_acc, v130_data, v76_data);
          tensorforge::fmacdpp16<11>(v114_acc, v130_data, v83_data);
          tensorforge::fmacdpp16<12>(v114_acc, v130_data, v90_data);
          tensorforge::fmacdpp16<13>(v114_acc, v130_data, v97_data);
          tensorforge::fmacdpp16<14>(v114_acc, v130_data, v104_data);
          tensorforge::fmacdpp16<15>(v114_acc, v130_data, v111_data);
          tensorforge::fmacdpp16<0>(v115_acc, v131_data, v6_data);
          tensorforge::fmacdpp16<1>(v115_acc, v131_data, v13_data);
          tensorforge::fmacdpp16<2>(v115_acc, v131_data, v20_data);
          tensorforge::fmacdpp16<3>(v115_acc, v131_data, v27_data);
          tensorforge::fmacdpp16<4>(v115_acc, v131_data, v34_data);
          tensorforge::fmacdpp16<5>(v115_acc, v131_data, v41_data);
          tensorforge::fmacdpp16<6>(v115_acc, v131_data, v48_data);
          tensorforge::fmacdpp16<7>(v115_acc, v131_data, v55_data);
          tensorforge::fmacdpp16<8>(v115_acc, v131_data, v62_data);
          tensorforge::fmacdpp16<9>(v115_acc, v131_data, v69_data);
          tensorforge::fmacdpp16<10>(v115_acc, v131_data, v76_data);
          tensorforge::fmacdpp16<11>(v115_acc, v131_data, v83_data);
          tensorforge::fmacdpp16<12>(v115_acc, v131_data, v90_data);
          tensorforge::fmacdpp16<13>(v115_acc, v131_data, v97_data);
          tensorforge::fmacdpp16<14>(v115_acc, v131_data, v104_data);
          tensorforge::fmacdpp16<15>(v115_acc, v131_data, v111_data);
          tensorforge::fmacdpp16<0>(v116_acc, v132_data, v6_data);
          tensorforge::fmacdpp16<1>(v116_acc, v132_data, v13_data);
          tensorforge::fmacdpp16<2>(v116_acc, v132_data, v20_data);
          tensorforge::fmacdpp16<3>(v116_acc, v132_data, v27_data);
          tensorforge::fmacdpp16<4>(v116_acc, v132_data, v34_data);
          tensorforge::fmacdpp16<5>(v116_acc, v132_data, v41_data);
          tensorforge::fmacdpp16<6>(v116_acc, v132_data, v48_data);
          tensorforge::fmacdpp16<7>(v116_acc, v132_data, v55_data);
          tensorforge::fmacdpp16<8>(v116_acc, v132_data, v62_data);
          tensorforge::fmacdpp16<9>(v116_acc, v132_data, v69_data);
          tensorforge::fmacdpp16<10>(v116_acc, v132_data, v76_data);
          tensorforge::fmacdpp16<11>(v116_acc, v132_data, v83_data);
          tensorforge::fmacdpp16<12>(v116_acc, v132_data, v90_data);
          tensorforge::fmacdpp16<13>(v116_acc, v132_data, v97_data);
          tensorforge::fmacdpp16<14>(v116_acc, v132_data, v104_data);
          tensorforge::fmacdpp16<15>(v116_acc, v132_data, v111_data);
          tensorforge::fmacdpp16<0>(v117_acc, v133_data, v6_data);
          tensorforge::fmacdpp16<1>(v117_acc, v133_data, v13_data);
          tensorforge::fmacdpp16<2>(v117_acc, v133_data, v20_data);
          tensorforge::fmacdpp16<3>(v117_acc, v133_data, v27_data);
          tensorforge::fmacdpp16<4>(v117_acc, v133_data, v34_data);
          tensorforge::fmacdpp16<5>(v117_acc, v133_data, v41_data);
          tensorforge::fmacdpp16<6>(v117_acc, v133_data, v48_data);
          tensorforge::fmacdpp16<7>(v117_acc, v133_data, v55_data);
          tensorforge::fmacdpp16<8>(v117_acc, v133_data, v62_data);
          tensorforge::fmacdpp16<9>(v117_acc, v133_data, v69_data);
          tensorforge::fmacdpp16<10>(v117_acc, v133_data, v76_data);
          tensorforge::fmacdpp16<11>(v117_acc, v133_data, v83_data);
          tensorforge::fmacdpp16<12>(v117_acc, v133_data, v90_data);
          tensorforge::fmacdpp16<13>(v117_acc, v133_data, v97_data);
          tensorforge::fmacdpp16<14>(v117_acc, v133_data, v104_data);
          tensorforge::fmacdpp16<15>(v117_acc, v133_data, v111_data);
          tensorforge::fmacdpp16<0>(v118_acc, v134_data, v6_data);
          tensorforge::fmacdpp16<1>(v118_acc, v134_data, v13_data);
          tensorforge::fmacdpp16<2>(v118_acc, v134_data, v20_data);
          tensorforge::fmacdpp16<3>(v118_acc, v134_data, v27_data);
          tensorforge::fmacdpp16<4>(v118_acc, v134_data, v34_data);
          tensorforge::fmacdpp16<5>(v118_acc, v134_data, v41_data);
          tensorforge::fmacdpp16<6>(v118_acc, v134_data, v48_data);
          tensorforge::fmacdpp16<7>(v118_acc, v134_data, v55_data);
          tensorforge::fmacdpp16<8>(v118_acc, v134_data, v62_data);
          tensorforge::fmacdpp16<9>(v118_acc, v134_data, v69_data);
          tensorforge::fmacdpp16<10>(v118_acc, v134_data, v76_data);
          tensorforge::fmacdpp16<11>(v118_acc, v134_data, v83_data);
          tensorforge::fmacdpp16<12>(v118_acc, v134_data, v90_data);
          tensorforge::fmacdpp16<13>(v118_acc, v134_data, v97_data);
          tensorforge::fmacdpp16<14>(v118_acc, v134_data, v104_data);
          tensorforge::fmacdpp16<15>(v118_acc, v134_data, v111_data);
          tensorforge::fmacdpp16<0>(v119_acc, v135_data, v6_data);
          tensorforge::fmacdpp16<1>(v119_acc, v135_data, v13_data);
          tensorforge::fmacdpp16<2>(v119_acc, v135_data, v20_data);
          tensorforge::fmacdpp16<3>(v119_acc, v135_data, v27_data);
          tensorforge::fmacdpp16<4>(v119_acc, v135_data, v34_data);
          tensorforge::fmacdpp16<5>(v119_acc, v135_data, v41_data);
          tensorforge::fmacdpp16<6>(v119_acc, v135_data, v48_data);
          tensorforge::fmacdpp16<7>(v119_acc, v135_data, v55_data);
          tensorforge::fmacdpp16<8>(v119_acc, v135_data, v62_data);
          tensorforge::fmacdpp16<9>(v119_acc, v135_data, v69_data);
          tensorforge::fmacdpp16<10>(v119_acc, v135_data, v76_data);
          tensorforge::fmacdpp16<11>(v119_acc, v135_data, v83_data);
          tensorforge::fmacdpp16<12>(v119_acc, v135_data, v90_data);
          tensorforge::fmacdpp16<13>(v119_acc, v135_data, v97_data);
          tensorforge::fmacdpp16<14>(v119_acc, v135_data, v104_data);
          tensorforge::fmacdpp16<15>(v119_acc, v135_data, v111_data);
          tensorforge::fmacdpp16<0>(v120_acc, v136_data, v6_data);
          tensorforge::fmacdpp16<1>(v120_acc, v136_data, v13_data);
          tensorforge::fmacdpp16<2>(v120_acc, v136_data, v20_data);
          tensorforge::fmacdpp16<3>(v120_acc, v136_data, v27_data);
          tensorforge::fmacdpp16<4>(v120_acc, v136_data, v34_data);
          tensorforge::fmacdpp16<5>(v120_acc, v136_data, v41_data);
          tensorforge::fmacdpp16<6>(v120_acc, v136_data, v48_data);
          tensorforge::fmacdpp16<7>(v120_acc, v136_data, v55_data);
          tensorforge::fmacdpp16<8>(v120_acc, v136_data, v62_data);
          tensorforge::fmacdpp16<9>(v120_acc, v136_data, v69_data);
          tensorforge::fmacdpp16<10>(v120_acc, v136_data, v76_data);
          tensorforge::fmacdpp16<11>(v120_acc, v136_data, v83_data);
          tensorforge::fmacdpp16<12>(v120_acc, v136_data, v90_data);
          tensorforge::fmacdpp16<13>(v120_acc, v136_data, v97_data);
          tensorforge::fmacdpp16<14>(v120_acc, v136_data, v104_data);
          tensorforge::fmacdpp16<15>(v120_acc, v136_data, v111_data);
          tensorforge::fmacdpp16<0>(v121_acc, v137_data, v6_data);
          tensorforge::fmacdpp16<1>(v121_acc, v137_data, v13_data);
          tensorforge::fmacdpp16<2>(v121_acc, v137_data, v20_data);
          tensorforge::fmacdpp16<3>(v121_acc, v137_data, v27_data);
          tensorforge::fmacdpp16<4>(v121_acc, v137_data, v34_data);
          tensorforge::fmacdpp16<5>(v121_acc, v137_data, v41_data);
          tensorforge::fmacdpp16<6>(v121_acc, v137_data, v48_data);
          tensorforge::fmacdpp16<7>(v121_acc, v137_data, v55_data);
          tensorforge::fmacdpp16<8>(v121_acc, v137_data, v62_data);
          tensorforge::fmacdpp16<9>(v121_acc, v137_data, v69_data);
          tensorforge::fmacdpp16<10>(v121_acc, v137_data, v76_data);
          tensorforge::fmacdpp16<11>(v121_acc, v137_data, v83_data);
          tensorforge::fmacdpp16<12>(v121_acc, v137_data, v90_data);
          tensorforge::fmacdpp16<13>(v121_acc, v137_data, v97_data);
          tensorforge::fmacdpp16<14>(v121_acc, v137_data, v104_data);
          tensorforge::fmacdpp16<15>(v121_acc, v137_data, v111_data);
          tensorforge::fmacdpp16<0>(v122_acc, v138_data, v6_data);
          tensorforge::fmacdpp16<1>(v122_acc, v138_data, v13_data);
          tensorforge::fmacdpp16<2>(v122_acc, v138_data, v20_data);
          tensorforge::fmacdpp16<3>(v122_acc, v138_data, v27_data);
          tensorforge::fmacdpp16<4>(v122_acc, v138_data, v34_data);
          tensorforge::fmacdpp16<5>(v122_acc, v138_data, v41_data);
          tensorforge::fmacdpp16<6>(v122_acc, v138_data, v48_data);
          tensorforge::fmacdpp16<7>(v122_acc, v138_data, v55_data);
          tensorforge::fmacdpp16<8>(v122_acc, v138_data, v62_data);
          tensorforge::fmacdpp16<9>(v122_acc, v138_data, v69_data);
          tensorforge::fmacdpp16<10>(v122_acc, v138_data, v76_data);
          tensorforge::fmacdpp16<11>(v122_acc, v138_data, v83_data);
          tensorforge::fmacdpp16<12>(v122_acc, v138_data, v90_data);
          tensorforge::fmacdpp16<13>(v122_acc, v138_data, v97_data);
          tensorforge::fmacdpp16<14>(v122_acc, v138_data, v104_data);
          tensorforge::fmacdpp16<15>(v122_acc, v138_data, v111_data);
          tensorforge::fmacdpp16<0>(v123_acc, v139_data, v6_data);
          tensorforge::fmacdpp16<1>(v123_acc, v139_data, v13_data);
          tensorforge::fmacdpp16<2>(v123_acc, v139_data, v20_data);
          tensorforge::fmacdpp16<3>(v123_acc, v139_data, v27_data);
          tensorforge::fmacdpp16<4>(v123_acc, v139_data, v34_data);
          tensorforge::fmacdpp16<5>(v123_acc, v139_data, v41_data);
          tensorforge::fmacdpp16<6>(v123_acc, v139_data, v48_data);
          tensorforge::fmacdpp16<7>(v123_acc, v139_data, v55_data);
          tensorforge::fmacdpp16<8>(v123_acc, v139_data, v62_data);
          tensorforge::fmacdpp16<9>(v123_acc, v139_data, v69_data);
          tensorforge::fmacdpp16<10>(v123_acc, v139_data, v76_data);
          tensorforge::fmacdpp16<11>(v123_acc, v139_data, v83_data);
          tensorforge::fmacdpp16<12>(v123_acc, v139_data, v90_data);
          tensorforge::fmacdpp16<13>(v123_acc, v139_data, v97_data);
          tensorforge::fmacdpp16<14>(v123_acc, v139_data, v104_data);
          tensorforge::fmacdpp16<15>(v123_acc, v139_data, v111_data);
          tensorforge::fmacdpp16<0>(v124_acc, v140_data, v6_data);
          tensorforge::fmacdpp16<1>(v124_acc, v140_data, v13_data);
          tensorforge::fmacdpp16<2>(v124_acc, v140_data, v20_data);
          tensorforge::fmacdpp16<3>(v124_acc, v140_data, v27_data);
          tensorforge::fmacdpp16<4>(v124_acc, v140_data, v34_data);
          tensorforge::fmacdpp16<5>(v124_acc, v140_data, v41_data);
          tensorforge::fmacdpp16<6>(v124_acc, v140_data, v48_data);
          tensorforge::fmacdpp16<7>(v124_acc, v140_data, v55_data);
          tensorforge::fmacdpp16<8>(v124_acc, v140_data, v62_data);
          tensorforge::fmacdpp16<9>(v124_acc, v140_data, v69_data);
          tensorforge::fmacdpp16<10>(v124_acc, v140_data, v76_data);
          tensorforge::fmacdpp16<11>(v124_acc, v140_data, v83_data);
          tensorforge::fmacdpp16<12>(v124_acc, v140_data, v90_data);
          tensorforge::fmacdpp16<13>(v124_acc, v140_data, v97_data);
          tensorforge::fmacdpp16<14>(v124_acc, v140_data, v104_data);
          tensorforge::fmacdpp16<15>(v124_acc, v140_data, v111_data);
          tensorforge::fmacdpp16<0>(v125_acc, v141_data, v6_data);
          tensorforge::fmacdpp16<1>(v125_acc, v141_data, v13_data);
          tensorforge::fmacdpp16<2>(v125_acc, v141_data, v20_data);
          tensorforge::fmacdpp16<3>(v125_acc, v141_data, v27_data);
          tensorforge::fmacdpp16<4>(v125_acc, v141_data, v34_data);
          tensorforge::fmacdpp16<5>(v125_acc, v141_data, v41_data);
          tensorforge::fmacdpp16<6>(v125_acc, v141_data, v48_data);
          tensorforge::fmacdpp16<7>(v125_acc, v141_data, v55_data);
          tensorforge::fmacdpp16<8>(v125_acc, v141_data, v62_data);
          tensorforge::fmacdpp16<9>(v125_acc, v141_data, v69_data);
          tensorforge::fmacdpp16<10>(v125_acc, v141_data, v76_data);
          tensorforge::fmacdpp16<11>(v125_acc, v141_data, v83_data);
          tensorforge::fmacdpp16<12>(v125_acc, v141_data, v90_data);
          tensorforge::fmacdpp16<13>(v125_acc, v141_data, v97_data);
          tensorforge::fmacdpp16<14>(v125_acc, v141_data, v104_data);
          tensorforge::fmacdpp16<15>(v125_acc, v141_data, v111_data);
          tensorforge::fmacdpp16<0>(v126_acc, v142_data, v6_data);
          tensorforge::fmacdpp16<1>(v126_acc, v142_data, v13_data);
          tensorforge::fmacdpp16<2>(v126_acc, v142_data, v20_data);
          tensorforge::fmacdpp16<3>(v126_acc, v142_data, v27_data);
          tensorforge::fmacdpp16<4>(v126_acc, v142_data, v34_data);
          tensorforge::fmacdpp16<5>(v126_acc, v142_data, v41_data);
          tensorforge::fmacdpp16<6>(v126_acc, v142_data, v48_data);
          tensorforge::fmacdpp16<7>(v126_acc, v142_data, v55_data);
          tensorforge::fmacdpp16<8>(v126_acc, v142_data, v62_data);
          tensorforge::fmacdpp16<9>(v126_acc, v142_data, v69_data);
          tensorforge::fmacdpp16<10>(v126_acc, v142_data, v76_data);
          tensorforge::fmacdpp16<11>(v126_acc, v142_data, v83_data);
          tensorforge::fmacdpp16<12>(v126_acc, v142_data, v90_data);
          tensorforge::fmacdpp16<13>(v126_acc, v142_data, v97_data);
          tensorforge::fmacdpp16<14>(v126_acc, v142_data, v104_data);
          tensorforge::fmacdpp16<15>(v126_acc, v142_data, v111_data);
          tensorforge::fmacdpp16<0>(v127_acc, v143_data, v6_data);
          tensorforge::fmacdpp16<1>(v127_acc, v143_data, v13_data);
          tensorforge::fmacdpp16<2>(v127_acc, v143_data, v20_data);
          tensorforge::fmacdpp16<3>(v127_acc, v143_data, v27_data);
          tensorforge::fmacdpp16<4>(v127_acc, v143_data, v34_data);
          tensorforge::fmacdpp16<5>(v127_acc, v143_data, v41_data);
          tensorforge::fmacdpp16<6>(v127_acc, v143_data, v48_data);
          tensorforge::fmacdpp16<7>(v127_acc, v143_data, v55_data);
          tensorforge::fmacdpp16<8>(v127_acc, v143_data, v62_data);
          tensorforge::fmacdpp16<9>(v127_acc, v143_data, v69_data);
          tensorforge::fmacdpp16<10>(v127_acc, v143_data, v76_data);
          tensorforge::fmacdpp16<11>(v127_acc, v143_data, v83_data);
          tensorforge::fmacdpp16<12>(v127_acc, v143_data, v90_data);
          tensorforge::fmacdpp16<13>(v127_acc, v143_data, v97_data);
          tensorforge::fmacdpp16<14>(v127_acc, v143_data, v104_data);
          tensorforge::fmacdpp16<15>(v127_acc, v143_data, v111_data);
          ir1[0] = v112_acc;
          ir1[1] = v113_acc;
          ir1[2] = v114_acc;
          ir1[3] = v115_acc;
          ir1[4] = v116_acc;
          ir1[5] = v117_acc;
          ir1[6] = v118_acc;
          ir1[7] = v119_acc;
          ir1[8] = v120_acc;
          ir1[9] = v121_acc;
          ir1[10] = v122_acc;
          ir1[11] = v123_acc;
          ir1[12] = v124_acc;
          ir1[13] = v125_acc;
          ir1[14] = v126_acc;
          ir1[15] = v127_acc;
          // glb_m0 = store{r>g}(r1);
          int32_t v146_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v147_i0 = 0; v147_i0 < 1; ++v147_i0) {
            int32_t v155_lead = v146_lead + (v147_i0 * 16);
            #pragma unroll
            for (int32_t v148_i1 = 0; v148_i1 < 16; ++v148_i1) {
              int32_t v149_a = v147_i0 + v148_i1;
              double v150_data = r1[v149_a];
              int32_t v157_a = v155_lead + (v148_i1 * 16);
              glb_m0[v157_a] = v150_data;
            }
          }
          ;
        }
      }
    }
  }
}

