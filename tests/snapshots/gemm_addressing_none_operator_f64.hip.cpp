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
          double v12_lin = glb_m2[0 + threadIdx.x * 1];
          r0[0] = v12_lin;
          double v13_lin = glb_m2[16 + threadIdx.x * 1];
          r0[1] = v13_lin;
          double v14_lin = glb_m2[32 + threadIdx.x * 1];
          r0[2] = v14_lin;
          double v15_lin = glb_m2[48 + threadIdx.x * 1];
          r0[3] = v15_lin;
          double v16_lin = glb_m2[64 + threadIdx.x * 1];
          r0[4] = v16_lin;
          double v17_lin = glb_m2[80 + threadIdx.x * 1];
          r0[5] = v17_lin;
          double v18_lin = glb_m2[96 + threadIdx.x * 1];
          r0[6] = v18_lin;
          double v19_lin = glb_m2[112 + threadIdx.x * 1];
          r0[7] = v19_lin;
          double v20_lin = glb_m2[128 + threadIdx.x * 1];
          r0[8] = v20_lin;
          double v21_lin = glb_m2[144 + threadIdx.x * 1];
          r0[9] = v21_lin;
          double v22_lin = glb_m2[160 + threadIdx.x * 1];
          r0[10] = v22_lin;
          double v23_lin = glb_m2[176 + threadIdx.x * 1];
          r0[11] = v23_lin;
          double v24_lin = glb_m2[192 + threadIdx.x * 1];
          r0[12] = v24_lin;
          double v25_lin = glb_m2[208 + threadIdx.x * 1];
          r0[13] = v25_lin;
          double v26_lin = glb_m2[224 + threadIdx.x * 1];
          r0[14] = v26_lin;
          double v27_lin = glb_m2[240 + threadIdx.x * 1];
          r0[15] = v27_lin;
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          int32_t v31_lane = threadIdx.x % 16;
          double v35_data = glb_m1[v31_lane];
          double v42_data = glb_m1[(v31_lane + 16)];
          double v49_data = glb_m1[(v31_lane + 32)];
          double v56_data = glb_m1[(v31_lane + 48)];
          double v63_data = glb_m1[(v31_lane + 64)];
          double v70_data = glb_m1[(v31_lane + 80)];
          double v77_data = glb_m1[(v31_lane + 96)];
          double v84_data = glb_m1[(v31_lane + 112)];
          double v91_data = glb_m1[(v31_lane + 128)];
          double v98_data = glb_m1[(v31_lane + 144)];
          double v105_data = glb_m1[(v31_lane + 160)];
          double v112_data = glb_m1[(v31_lane + 176)];
          double v119_data = glb_m1[(v31_lane + 192)];
          double v126_data = glb_m1[(v31_lane + 208)];
          double v133_data = glb_m1[(v31_lane + 224)];
          double v140_data = glb_m1[(v31_lane + 240)];
          double v141_acc{};
          double v142_acc{};
          double v143_acc{};
          double v144_acc{};
          double v145_acc{};
          double v146_acc{};
          double v147_acc{};
          double v148_acc{};
          double v149_acc{};
          double v150_acc{};
          double v151_acc{};
          double v152_acc{};
          double v153_acc{};
          double v154_acc{};
          double v155_acc{};
          double v156_acc{};
          double v157_data = r0[0];
          double v158_data = r0[1];
          double v159_data = r0[2];
          double v160_data = r0[3];
          double v161_data = r0[4];
          double v162_data = r0[5];
          double v163_data = r0[6];
          double v164_data = r0[7];
          double v165_data = r0[8];
          double v166_data = r0[9];
          double v167_data = r0[10];
          double v168_data = r0[11];
          double v169_data = r0[12];
          double v170_data = r0[13];
          double v171_data = r0[14];
          double v172_data = r0[15];
          tensorforge::fmacdpp16<0>(v141_acc, v157_data, v35_data);
          tensorforge::fmacdpp16<1>(v141_acc, v157_data, v42_data);
          tensorforge::fmacdpp16<2>(v141_acc, v157_data, v49_data);
          tensorforge::fmacdpp16<3>(v141_acc, v157_data, v56_data);
          tensorforge::fmacdpp16<4>(v141_acc, v157_data, v63_data);
          tensorforge::fmacdpp16<5>(v141_acc, v157_data, v70_data);
          tensorforge::fmacdpp16<6>(v141_acc, v157_data, v77_data);
          tensorforge::fmacdpp16<7>(v141_acc, v157_data, v84_data);
          tensorforge::fmacdpp16<8>(v141_acc, v157_data, v91_data);
          tensorforge::fmacdpp16<9>(v141_acc, v157_data, v98_data);
          tensorforge::fmacdpp16<10>(v141_acc, v157_data, v105_data);
          tensorforge::fmacdpp16<11>(v141_acc, v157_data, v112_data);
          tensorforge::fmacdpp16<12>(v141_acc, v157_data, v119_data);
          tensorforge::fmacdpp16<13>(v141_acc, v157_data, v126_data);
          tensorforge::fmacdpp16<14>(v141_acc, v157_data, v133_data);
          tensorforge::fmacdpp16<15>(v141_acc, v157_data, v140_data);
          tensorforge::fmacdpp16<0>(v142_acc, v158_data, v35_data);
          tensorforge::fmacdpp16<1>(v142_acc, v158_data, v42_data);
          tensorforge::fmacdpp16<2>(v142_acc, v158_data, v49_data);
          tensorforge::fmacdpp16<3>(v142_acc, v158_data, v56_data);
          tensorforge::fmacdpp16<4>(v142_acc, v158_data, v63_data);
          tensorforge::fmacdpp16<5>(v142_acc, v158_data, v70_data);
          tensorforge::fmacdpp16<6>(v142_acc, v158_data, v77_data);
          tensorforge::fmacdpp16<7>(v142_acc, v158_data, v84_data);
          tensorforge::fmacdpp16<8>(v142_acc, v158_data, v91_data);
          tensorforge::fmacdpp16<9>(v142_acc, v158_data, v98_data);
          tensorforge::fmacdpp16<10>(v142_acc, v158_data, v105_data);
          tensorforge::fmacdpp16<11>(v142_acc, v158_data, v112_data);
          tensorforge::fmacdpp16<12>(v142_acc, v158_data, v119_data);
          tensorforge::fmacdpp16<13>(v142_acc, v158_data, v126_data);
          tensorforge::fmacdpp16<14>(v142_acc, v158_data, v133_data);
          tensorforge::fmacdpp16<15>(v142_acc, v158_data, v140_data);
          tensorforge::fmacdpp16<0>(v143_acc, v159_data, v35_data);
          tensorforge::fmacdpp16<1>(v143_acc, v159_data, v42_data);
          tensorforge::fmacdpp16<2>(v143_acc, v159_data, v49_data);
          tensorforge::fmacdpp16<3>(v143_acc, v159_data, v56_data);
          tensorforge::fmacdpp16<4>(v143_acc, v159_data, v63_data);
          tensorforge::fmacdpp16<5>(v143_acc, v159_data, v70_data);
          tensorforge::fmacdpp16<6>(v143_acc, v159_data, v77_data);
          tensorforge::fmacdpp16<7>(v143_acc, v159_data, v84_data);
          tensorforge::fmacdpp16<8>(v143_acc, v159_data, v91_data);
          tensorforge::fmacdpp16<9>(v143_acc, v159_data, v98_data);
          tensorforge::fmacdpp16<10>(v143_acc, v159_data, v105_data);
          tensorforge::fmacdpp16<11>(v143_acc, v159_data, v112_data);
          tensorforge::fmacdpp16<12>(v143_acc, v159_data, v119_data);
          tensorforge::fmacdpp16<13>(v143_acc, v159_data, v126_data);
          tensorforge::fmacdpp16<14>(v143_acc, v159_data, v133_data);
          tensorforge::fmacdpp16<15>(v143_acc, v159_data, v140_data);
          tensorforge::fmacdpp16<0>(v144_acc, v160_data, v35_data);
          tensorforge::fmacdpp16<1>(v144_acc, v160_data, v42_data);
          tensorforge::fmacdpp16<2>(v144_acc, v160_data, v49_data);
          tensorforge::fmacdpp16<3>(v144_acc, v160_data, v56_data);
          tensorforge::fmacdpp16<4>(v144_acc, v160_data, v63_data);
          tensorforge::fmacdpp16<5>(v144_acc, v160_data, v70_data);
          tensorforge::fmacdpp16<6>(v144_acc, v160_data, v77_data);
          tensorforge::fmacdpp16<7>(v144_acc, v160_data, v84_data);
          tensorforge::fmacdpp16<8>(v144_acc, v160_data, v91_data);
          tensorforge::fmacdpp16<9>(v144_acc, v160_data, v98_data);
          tensorforge::fmacdpp16<10>(v144_acc, v160_data, v105_data);
          tensorforge::fmacdpp16<11>(v144_acc, v160_data, v112_data);
          tensorforge::fmacdpp16<12>(v144_acc, v160_data, v119_data);
          tensorforge::fmacdpp16<13>(v144_acc, v160_data, v126_data);
          tensorforge::fmacdpp16<14>(v144_acc, v160_data, v133_data);
          tensorforge::fmacdpp16<15>(v144_acc, v160_data, v140_data);
          tensorforge::fmacdpp16<0>(v145_acc, v161_data, v35_data);
          tensorforge::fmacdpp16<1>(v145_acc, v161_data, v42_data);
          tensorforge::fmacdpp16<2>(v145_acc, v161_data, v49_data);
          tensorforge::fmacdpp16<3>(v145_acc, v161_data, v56_data);
          tensorforge::fmacdpp16<4>(v145_acc, v161_data, v63_data);
          tensorforge::fmacdpp16<5>(v145_acc, v161_data, v70_data);
          tensorforge::fmacdpp16<6>(v145_acc, v161_data, v77_data);
          tensorforge::fmacdpp16<7>(v145_acc, v161_data, v84_data);
          tensorforge::fmacdpp16<8>(v145_acc, v161_data, v91_data);
          tensorforge::fmacdpp16<9>(v145_acc, v161_data, v98_data);
          tensorforge::fmacdpp16<10>(v145_acc, v161_data, v105_data);
          tensorforge::fmacdpp16<11>(v145_acc, v161_data, v112_data);
          tensorforge::fmacdpp16<12>(v145_acc, v161_data, v119_data);
          tensorforge::fmacdpp16<13>(v145_acc, v161_data, v126_data);
          tensorforge::fmacdpp16<14>(v145_acc, v161_data, v133_data);
          tensorforge::fmacdpp16<15>(v145_acc, v161_data, v140_data);
          tensorforge::fmacdpp16<0>(v146_acc, v162_data, v35_data);
          tensorforge::fmacdpp16<1>(v146_acc, v162_data, v42_data);
          tensorforge::fmacdpp16<2>(v146_acc, v162_data, v49_data);
          tensorforge::fmacdpp16<3>(v146_acc, v162_data, v56_data);
          tensorforge::fmacdpp16<4>(v146_acc, v162_data, v63_data);
          tensorforge::fmacdpp16<5>(v146_acc, v162_data, v70_data);
          tensorforge::fmacdpp16<6>(v146_acc, v162_data, v77_data);
          tensorforge::fmacdpp16<7>(v146_acc, v162_data, v84_data);
          tensorforge::fmacdpp16<8>(v146_acc, v162_data, v91_data);
          tensorforge::fmacdpp16<9>(v146_acc, v162_data, v98_data);
          tensorforge::fmacdpp16<10>(v146_acc, v162_data, v105_data);
          tensorforge::fmacdpp16<11>(v146_acc, v162_data, v112_data);
          tensorforge::fmacdpp16<12>(v146_acc, v162_data, v119_data);
          tensorforge::fmacdpp16<13>(v146_acc, v162_data, v126_data);
          tensorforge::fmacdpp16<14>(v146_acc, v162_data, v133_data);
          tensorforge::fmacdpp16<15>(v146_acc, v162_data, v140_data);
          tensorforge::fmacdpp16<0>(v147_acc, v163_data, v35_data);
          tensorforge::fmacdpp16<1>(v147_acc, v163_data, v42_data);
          tensorforge::fmacdpp16<2>(v147_acc, v163_data, v49_data);
          tensorforge::fmacdpp16<3>(v147_acc, v163_data, v56_data);
          tensorforge::fmacdpp16<4>(v147_acc, v163_data, v63_data);
          tensorforge::fmacdpp16<5>(v147_acc, v163_data, v70_data);
          tensorforge::fmacdpp16<6>(v147_acc, v163_data, v77_data);
          tensorforge::fmacdpp16<7>(v147_acc, v163_data, v84_data);
          tensorforge::fmacdpp16<8>(v147_acc, v163_data, v91_data);
          tensorforge::fmacdpp16<9>(v147_acc, v163_data, v98_data);
          tensorforge::fmacdpp16<10>(v147_acc, v163_data, v105_data);
          tensorforge::fmacdpp16<11>(v147_acc, v163_data, v112_data);
          tensorforge::fmacdpp16<12>(v147_acc, v163_data, v119_data);
          tensorforge::fmacdpp16<13>(v147_acc, v163_data, v126_data);
          tensorforge::fmacdpp16<14>(v147_acc, v163_data, v133_data);
          tensorforge::fmacdpp16<15>(v147_acc, v163_data, v140_data);
          tensorforge::fmacdpp16<0>(v148_acc, v164_data, v35_data);
          tensorforge::fmacdpp16<1>(v148_acc, v164_data, v42_data);
          tensorforge::fmacdpp16<2>(v148_acc, v164_data, v49_data);
          tensorforge::fmacdpp16<3>(v148_acc, v164_data, v56_data);
          tensorforge::fmacdpp16<4>(v148_acc, v164_data, v63_data);
          tensorforge::fmacdpp16<5>(v148_acc, v164_data, v70_data);
          tensorforge::fmacdpp16<6>(v148_acc, v164_data, v77_data);
          tensorforge::fmacdpp16<7>(v148_acc, v164_data, v84_data);
          tensorforge::fmacdpp16<8>(v148_acc, v164_data, v91_data);
          tensorforge::fmacdpp16<9>(v148_acc, v164_data, v98_data);
          tensorforge::fmacdpp16<10>(v148_acc, v164_data, v105_data);
          tensorforge::fmacdpp16<11>(v148_acc, v164_data, v112_data);
          tensorforge::fmacdpp16<12>(v148_acc, v164_data, v119_data);
          tensorforge::fmacdpp16<13>(v148_acc, v164_data, v126_data);
          tensorforge::fmacdpp16<14>(v148_acc, v164_data, v133_data);
          tensorforge::fmacdpp16<15>(v148_acc, v164_data, v140_data);
          tensorforge::fmacdpp16<0>(v149_acc, v165_data, v35_data);
          tensorforge::fmacdpp16<1>(v149_acc, v165_data, v42_data);
          tensorforge::fmacdpp16<2>(v149_acc, v165_data, v49_data);
          tensorforge::fmacdpp16<3>(v149_acc, v165_data, v56_data);
          tensorforge::fmacdpp16<4>(v149_acc, v165_data, v63_data);
          tensorforge::fmacdpp16<5>(v149_acc, v165_data, v70_data);
          tensorforge::fmacdpp16<6>(v149_acc, v165_data, v77_data);
          tensorforge::fmacdpp16<7>(v149_acc, v165_data, v84_data);
          tensorforge::fmacdpp16<8>(v149_acc, v165_data, v91_data);
          tensorforge::fmacdpp16<9>(v149_acc, v165_data, v98_data);
          tensorforge::fmacdpp16<10>(v149_acc, v165_data, v105_data);
          tensorforge::fmacdpp16<11>(v149_acc, v165_data, v112_data);
          tensorforge::fmacdpp16<12>(v149_acc, v165_data, v119_data);
          tensorforge::fmacdpp16<13>(v149_acc, v165_data, v126_data);
          tensorforge::fmacdpp16<14>(v149_acc, v165_data, v133_data);
          tensorforge::fmacdpp16<15>(v149_acc, v165_data, v140_data);
          tensorforge::fmacdpp16<0>(v150_acc, v166_data, v35_data);
          tensorforge::fmacdpp16<1>(v150_acc, v166_data, v42_data);
          tensorforge::fmacdpp16<2>(v150_acc, v166_data, v49_data);
          tensorforge::fmacdpp16<3>(v150_acc, v166_data, v56_data);
          tensorforge::fmacdpp16<4>(v150_acc, v166_data, v63_data);
          tensorforge::fmacdpp16<5>(v150_acc, v166_data, v70_data);
          tensorforge::fmacdpp16<6>(v150_acc, v166_data, v77_data);
          tensorforge::fmacdpp16<7>(v150_acc, v166_data, v84_data);
          tensorforge::fmacdpp16<8>(v150_acc, v166_data, v91_data);
          tensorforge::fmacdpp16<9>(v150_acc, v166_data, v98_data);
          tensorforge::fmacdpp16<10>(v150_acc, v166_data, v105_data);
          tensorforge::fmacdpp16<11>(v150_acc, v166_data, v112_data);
          tensorforge::fmacdpp16<12>(v150_acc, v166_data, v119_data);
          tensorforge::fmacdpp16<13>(v150_acc, v166_data, v126_data);
          tensorforge::fmacdpp16<14>(v150_acc, v166_data, v133_data);
          tensorforge::fmacdpp16<15>(v150_acc, v166_data, v140_data);
          tensorforge::fmacdpp16<0>(v151_acc, v167_data, v35_data);
          tensorforge::fmacdpp16<1>(v151_acc, v167_data, v42_data);
          tensorforge::fmacdpp16<2>(v151_acc, v167_data, v49_data);
          tensorforge::fmacdpp16<3>(v151_acc, v167_data, v56_data);
          tensorforge::fmacdpp16<4>(v151_acc, v167_data, v63_data);
          tensorforge::fmacdpp16<5>(v151_acc, v167_data, v70_data);
          tensorforge::fmacdpp16<6>(v151_acc, v167_data, v77_data);
          tensorforge::fmacdpp16<7>(v151_acc, v167_data, v84_data);
          tensorforge::fmacdpp16<8>(v151_acc, v167_data, v91_data);
          tensorforge::fmacdpp16<9>(v151_acc, v167_data, v98_data);
          tensorforge::fmacdpp16<10>(v151_acc, v167_data, v105_data);
          tensorforge::fmacdpp16<11>(v151_acc, v167_data, v112_data);
          tensorforge::fmacdpp16<12>(v151_acc, v167_data, v119_data);
          tensorforge::fmacdpp16<13>(v151_acc, v167_data, v126_data);
          tensorforge::fmacdpp16<14>(v151_acc, v167_data, v133_data);
          tensorforge::fmacdpp16<15>(v151_acc, v167_data, v140_data);
          tensorforge::fmacdpp16<0>(v152_acc, v168_data, v35_data);
          tensorforge::fmacdpp16<1>(v152_acc, v168_data, v42_data);
          tensorforge::fmacdpp16<2>(v152_acc, v168_data, v49_data);
          tensorforge::fmacdpp16<3>(v152_acc, v168_data, v56_data);
          tensorforge::fmacdpp16<4>(v152_acc, v168_data, v63_data);
          tensorforge::fmacdpp16<5>(v152_acc, v168_data, v70_data);
          tensorforge::fmacdpp16<6>(v152_acc, v168_data, v77_data);
          tensorforge::fmacdpp16<7>(v152_acc, v168_data, v84_data);
          tensorforge::fmacdpp16<8>(v152_acc, v168_data, v91_data);
          tensorforge::fmacdpp16<9>(v152_acc, v168_data, v98_data);
          tensorforge::fmacdpp16<10>(v152_acc, v168_data, v105_data);
          tensorforge::fmacdpp16<11>(v152_acc, v168_data, v112_data);
          tensorforge::fmacdpp16<12>(v152_acc, v168_data, v119_data);
          tensorforge::fmacdpp16<13>(v152_acc, v168_data, v126_data);
          tensorforge::fmacdpp16<14>(v152_acc, v168_data, v133_data);
          tensorforge::fmacdpp16<15>(v152_acc, v168_data, v140_data);
          tensorforge::fmacdpp16<0>(v153_acc, v169_data, v35_data);
          tensorforge::fmacdpp16<1>(v153_acc, v169_data, v42_data);
          tensorforge::fmacdpp16<2>(v153_acc, v169_data, v49_data);
          tensorforge::fmacdpp16<3>(v153_acc, v169_data, v56_data);
          tensorforge::fmacdpp16<4>(v153_acc, v169_data, v63_data);
          tensorforge::fmacdpp16<5>(v153_acc, v169_data, v70_data);
          tensorforge::fmacdpp16<6>(v153_acc, v169_data, v77_data);
          tensorforge::fmacdpp16<7>(v153_acc, v169_data, v84_data);
          tensorforge::fmacdpp16<8>(v153_acc, v169_data, v91_data);
          tensorforge::fmacdpp16<9>(v153_acc, v169_data, v98_data);
          tensorforge::fmacdpp16<10>(v153_acc, v169_data, v105_data);
          tensorforge::fmacdpp16<11>(v153_acc, v169_data, v112_data);
          tensorforge::fmacdpp16<12>(v153_acc, v169_data, v119_data);
          tensorforge::fmacdpp16<13>(v153_acc, v169_data, v126_data);
          tensorforge::fmacdpp16<14>(v153_acc, v169_data, v133_data);
          tensorforge::fmacdpp16<15>(v153_acc, v169_data, v140_data);
          tensorforge::fmacdpp16<0>(v154_acc, v170_data, v35_data);
          tensorforge::fmacdpp16<1>(v154_acc, v170_data, v42_data);
          tensorforge::fmacdpp16<2>(v154_acc, v170_data, v49_data);
          tensorforge::fmacdpp16<3>(v154_acc, v170_data, v56_data);
          tensorforge::fmacdpp16<4>(v154_acc, v170_data, v63_data);
          tensorforge::fmacdpp16<5>(v154_acc, v170_data, v70_data);
          tensorforge::fmacdpp16<6>(v154_acc, v170_data, v77_data);
          tensorforge::fmacdpp16<7>(v154_acc, v170_data, v84_data);
          tensorforge::fmacdpp16<8>(v154_acc, v170_data, v91_data);
          tensorforge::fmacdpp16<9>(v154_acc, v170_data, v98_data);
          tensorforge::fmacdpp16<10>(v154_acc, v170_data, v105_data);
          tensorforge::fmacdpp16<11>(v154_acc, v170_data, v112_data);
          tensorforge::fmacdpp16<12>(v154_acc, v170_data, v119_data);
          tensorforge::fmacdpp16<13>(v154_acc, v170_data, v126_data);
          tensorforge::fmacdpp16<14>(v154_acc, v170_data, v133_data);
          tensorforge::fmacdpp16<15>(v154_acc, v170_data, v140_data);
          tensorforge::fmacdpp16<0>(v155_acc, v171_data, v35_data);
          tensorforge::fmacdpp16<1>(v155_acc, v171_data, v42_data);
          tensorforge::fmacdpp16<2>(v155_acc, v171_data, v49_data);
          tensorforge::fmacdpp16<3>(v155_acc, v171_data, v56_data);
          tensorforge::fmacdpp16<4>(v155_acc, v171_data, v63_data);
          tensorforge::fmacdpp16<5>(v155_acc, v171_data, v70_data);
          tensorforge::fmacdpp16<6>(v155_acc, v171_data, v77_data);
          tensorforge::fmacdpp16<7>(v155_acc, v171_data, v84_data);
          tensorforge::fmacdpp16<8>(v155_acc, v171_data, v91_data);
          tensorforge::fmacdpp16<9>(v155_acc, v171_data, v98_data);
          tensorforge::fmacdpp16<10>(v155_acc, v171_data, v105_data);
          tensorforge::fmacdpp16<11>(v155_acc, v171_data, v112_data);
          tensorforge::fmacdpp16<12>(v155_acc, v171_data, v119_data);
          tensorforge::fmacdpp16<13>(v155_acc, v171_data, v126_data);
          tensorforge::fmacdpp16<14>(v155_acc, v171_data, v133_data);
          tensorforge::fmacdpp16<15>(v155_acc, v171_data, v140_data);
          tensorforge::fmacdpp16<0>(v156_acc, v172_data, v35_data);
          tensorforge::fmacdpp16<1>(v156_acc, v172_data, v42_data);
          tensorforge::fmacdpp16<2>(v156_acc, v172_data, v49_data);
          tensorforge::fmacdpp16<3>(v156_acc, v172_data, v56_data);
          tensorforge::fmacdpp16<4>(v156_acc, v172_data, v63_data);
          tensorforge::fmacdpp16<5>(v156_acc, v172_data, v70_data);
          tensorforge::fmacdpp16<6>(v156_acc, v172_data, v77_data);
          tensorforge::fmacdpp16<7>(v156_acc, v172_data, v84_data);
          tensorforge::fmacdpp16<8>(v156_acc, v172_data, v91_data);
          tensorforge::fmacdpp16<9>(v156_acc, v172_data, v98_data);
          tensorforge::fmacdpp16<10>(v156_acc, v172_data, v105_data);
          tensorforge::fmacdpp16<11>(v156_acc, v172_data, v112_data);
          tensorforge::fmacdpp16<12>(v156_acc, v172_data, v119_data);
          tensorforge::fmacdpp16<13>(v156_acc, v172_data, v126_data);
          tensorforge::fmacdpp16<14>(v156_acc, v172_data, v133_data);
          tensorforge::fmacdpp16<15>(v156_acc, v172_data, v140_data);
          r1[0] = v141_acc;
          r1[1] = v142_acc;
          r1[2] = v143_acc;
          r1[3] = v144_acc;
          r1[4] = v145_acc;
          r1[5] = v146_acc;
          r1[6] = v147_acc;
          r1[7] = v148_acc;
          r1[8] = v149_acc;
          r1[9] = v150_acc;
          r1[10] = v151_acc;
          r1[11] = v152_acc;
          r1[12] = v153_acc;
          r1[13] = v154_acc;
          r1[14] = v155_acc;
          r1[15] = v156_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v176_i0 = 0; v176_i0 < 1; ++v176_i0) {
            int32_t v184_lead = v31_lane + (v176_i0 * 16);
            #pragma unroll
            for (int32_t v177_i1 = 0; v177_i1 < 16; ++v177_i1) {
              double v179_data = r1[(v176_i0 + v177_i1)];
              glb_m0[(v184_lead + (v177_i1 * 16))] = v179_data;
            }
          }
        }
      }
    }
  }
}

