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
          double v32_data = glb_m1[v28_lane];
          double v39_data = glb_m1[(v28_lane + 16)];
          double v46_data = glb_m1[(v28_lane + 32)];
          double v53_data = glb_m1[(v28_lane + 48)];
          double v60_data = glb_m1[(v28_lane + 64)];
          double v67_data = glb_m1[(v28_lane + 80)];
          double v74_data = glb_m1[(v28_lane + 96)];
          double v81_data = glb_m1[(v28_lane + 112)];
          double v88_data = glb_m1[(v28_lane + 128)];
          double v95_data = glb_m1[(v28_lane + 144)];
          double v102_data = glb_m1[(v28_lane + 160)];
          double v109_data = glb_m1[(v28_lane + 176)];
          double v116_data = glb_m1[(v28_lane + 192)];
          double v123_data = glb_m1[(v28_lane + 208)];
          double v130_data = glb_m1[(v28_lane + 224)];
          double v137_data = glb_m1[(v28_lane + 240)];
          double v138_acc{};
          double v139_acc{};
          double v140_acc{};
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
          double v154_data = r0[0];
          double v155_data = r0[1];
          double v156_data = r0[2];
          double v157_data = r0[3];
          double v158_data = r0[4];
          double v159_data = r0[5];
          double v160_data = r0[6];
          double v161_data = r0[7];
          double v162_data = r0[8];
          double v163_data = r0[9];
          double v164_data = r0[10];
          double v165_data = r0[11];
          double v166_data = r0[12];
          double v167_data = r0[13];
          double v168_data = r0[14];
          double v169_data = r0[15];
          tensorforge::fmacdpp16<0>(v138_acc, v154_data, v32_data);
          tensorforge::fmacdpp16<1>(v138_acc, v154_data, v39_data);
          tensorforge::fmacdpp16<2>(v138_acc, v154_data, v46_data);
          tensorforge::fmacdpp16<3>(v138_acc, v154_data, v53_data);
          tensorforge::fmacdpp16<4>(v138_acc, v154_data, v60_data);
          tensorforge::fmacdpp16<5>(v138_acc, v154_data, v67_data);
          tensorforge::fmacdpp16<6>(v138_acc, v154_data, v74_data);
          tensorforge::fmacdpp16<7>(v138_acc, v154_data, v81_data);
          tensorforge::fmacdpp16<8>(v138_acc, v154_data, v88_data);
          tensorforge::fmacdpp16<9>(v138_acc, v154_data, v95_data);
          tensorforge::fmacdpp16<10>(v138_acc, v154_data, v102_data);
          tensorforge::fmacdpp16<11>(v138_acc, v154_data, v109_data);
          tensorforge::fmacdpp16<12>(v138_acc, v154_data, v116_data);
          tensorforge::fmacdpp16<13>(v138_acc, v154_data, v123_data);
          tensorforge::fmacdpp16<14>(v138_acc, v154_data, v130_data);
          tensorforge::fmacdpp16<15>(v138_acc, v154_data, v137_data);
          tensorforge::fmacdpp16<0>(v139_acc, v155_data, v32_data);
          tensorforge::fmacdpp16<1>(v139_acc, v155_data, v39_data);
          tensorforge::fmacdpp16<2>(v139_acc, v155_data, v46_data);
          tensorforge::fmacdpp16<3>(v139_acc, v155_data, v53_data);
          tensorforge::fmacdpp16<4>(v139_acc, v155_data, v60_data);
          tensorforge::fmacdpp16<5>(v139_acc, v155_data, v67_data);
          tensorforge::fmacdpp16<6>(v139_acc, v155_data, v74_data);
          tensorforge::fmacdpp16<7>(v139_acc, v155_data, v81_data);
          tensorforge::fmacdpp16<8>(v139_acc, v155_data, v88_data);
          tensorforge::fmacdpp16<9>(v139_acc, v155_data, v95_data);
          tensorforge::fmacdpp16<10>(v139_acc, v155_data, v102_data);
          tensorforge::fmacdpp16<11>(v139_acc, v155_data, v109_data);
          tensorforge::fmacdpp16<12>(v139_acc, v155_data, v116_data);
          tensorforge::fmacdpp16<13>(v139_acc, v155_data, v123_data);
          tensorforge::fmacdpp16<14>(v139_acc, v155_data, v130_data);
          tensorforge::fmacdpp16<15>(v139_acc, v155_data, v137_data);
          tensorforge::fmacdpp16<0>(v140_acc, v156_data, v32_data);
          tensorforge::fmacdpp16<1>(v140_acc, v156_data, v39_data);
          tensorforge::fmacdpp16<2>(v140_acc, v156_data, v46_data);
          tensorforge::fmacdpp16<3>(v140_acc, v156_data, v53_data);
          tensorforge::fmacdpp16<4>(v140_acc, v156_data, v60_data);
          tensorforge::fmacdpp16<5>(v140_acc, v156_data, v67_data);
          tensorforge::fmacdpp16<6>(v140_acc, v156_data, v74_data);
          tensorforge::fmacdpp16<7>(v140_acc, v156_data, v81_data);
          tensorforge::fmacdpp16<8>(v140_acc, v156_data, v88_data);
          tensorforge::fmacdpp16<9>(v140_acc, v156_data, v95_data);
          tensorforge::fmacdpp16<10>(v140_acc, v156_data, v102_data);
          tensorforge::fmacdpp16<11>(v140_acc, v156_data, v109_data);
          tensorforge::fmacdpp16<12>(v140_acc, v156_data, v116_data);
          tensorforge::fmacdpp16<13>(v140_acc, v156_data, v123_data);
          tensorforge::fmacdpp16<14>(v140_acc, v156_data, v130_data);
          tensorforge::fmacdpp16<15>(v140_acc, v156_data, v137_data);
          tensorforge::fmacdpp16<0>(v141_acc, v157_data, v32_data);
          tensorforge::fmacdpp16<1>(v141_acc, v157_data, v39_data);
          tensorforge::fmacdpp16<2>(v141_acc, v157_data, v46_data);
          tensorforge::fmacdpp16<3>(v141_acc, v157_data, v53_data);
          tensorforge::fmacdpp16<4>(v141_acc, v157_data, v60_data);
          tensorforge::fmacdpp16<5>(v141_acc, v157_data, v67_data);
          tensorforge::fmacdpp16<6>(v141_acc, v157_data, v74_data);
          tensorforge::fmacdpp16<7>(v141_acc, v157_data, v81_data);
          tensorforge::fmacdpp16<8>(v141_acc, v157_data, v88_data);
          tensorforge::fmacdpp16<9>(v141_acc, v157_data, v95_data);
          tensorforge::fmacdpp16<10>(v141_acc, v157_data, v102_data);
          tensorforge::fmacdpp16<11>(v141_acc, v157_data, v109_data);
          tensorforge::fmacdpp16<12>(v141_acc, v157_data, v116_data);
          tensorforge::fmacdpp16<13>(v141_acc, v157_data, v123_data);
          tensorforge::fmacdpp16<14>(v141_acc, v157_data, v130_data);
          tensorforge::fmacdpp16<15>(v141_acc, v157_data, v137_data);
          tensorforge::fmacdpp16<0>(v142_acc, v158_data, v32_data);
          tensorforge::fmacdpp16<1>(v142_acc, v158_data, v39_data);
          tensorforge::fmacdpp16<2>(v142_acc, v158_data, v46_data);
          tensorforge::fmacdpp16<3>(v142_acc, v158_data, v53_data);
          tensorforge::fmacdpp16<4>(v142_acc, v158_data, v60_data);
          tensorforge::fmacdpp16<5>(v142_acc, v158_data, v67_data);
          tensorforge::fmacdpp16<6>(v142_acc, v158_data, v74_data);
          tensorforge::fmacdpp16<7>(v142_acc, v158_data, v81_data);
          tensorforge::fmacdpp16<8>(v142_acc, v158_data, v88_data);
          tensorforge::fmacdpp16<9>(v142_acc, v158_data, v95_data);
          tensorforge::fmacdpp16<10>(v142_acc, v158_data, v102_data);
          tensorforge::fmacdpp16<11>(v142_acc, v158_data, v109_data);
          tensorforge::fmacdpp16<12>(v142_acc, v158_data, v116_data);
          tensorforge::fmacdpp16<13>(v142_acc, v158_data, v123_data);
          tensorforge::fmacdpp16<14>(v142_acc, v158_data, v130_data);
          tensorforge::fmacdpp16<15>(v142_acc, v158_data, v137_data);
          tensorforge::fmacdpp16<0>(v143_acc, v159_data, v32_data);
          tensorforge::fmacdpp16<1>(v143_acc, v159_data, v39_data);
          tensorforge::fmacdpp16<2>(v143_acc, v159_data, v46_data);
          tensorforge::fmacdpp16<3>(v143_acc, v159_data, v53_data);
          tensorforge::fmacdpp16<4>(v143_acc, v159_data, v60_data);
          tensorforge::fmacdpp16<5>(v143_acc, v159_data, v67_data);
          tensorforge::fmacdpp16<6>(v143_acc, v159_data, v74_data);
          tensorforge::fmacdpp16<7>(v143_acc, v159_data, v81_data);
          tensorforge::fmacdpp16<8>(v143_acc, v159_data, v88_data);
          tensorforge::fmacdpp16<9>(v143_acc, v159_data, v95_data);
          tensorforge::fmacdpp16<10>(v143_acc, v159_data, v102_data);
          tensorforge::fmacdpp16<11>(v143_acc, v159_data, v109_data);
          tensorforge::fmacdpp16<12>(v143_acc, v159_data, v116_data);
          tensorforge::fmacdpp16<13>(v143_acc, v159_data, v123_data);
          tensorforge::fmacdpp16<14>(v143_acc, v159_data, v130_data);
          tensorforge::fmacdpp16<15>(v143_acc, v159_data, v137_data);
          tensorforge::fmacdpp16<0>(v144_acc, v160_data, v32_data);
          tensorforge::fmacdpp16<1>(v144_acc, v160_data, v39_data);
          tensorforge::fmacdpp16<2>(v144_acc, v160_data, v46_data);
          tensorforge::fmacdpp16<3>(v144_acc, v160_data, v53_data);
          tensorforge::fmacdpp16<4>(v144_acc, v160_data, v60_data);
          tensorforge::fmacdpp16<5>(v144_acc, v160_data, v67_data);
          tensorforge::fmacdpp16<6>(v144_acc, v160_data, v74_data);
          tensorforge::fmacdpp16<7>(v144_acc, v160_data, v81_data);
          tensorforge::fmacdpp16<8>(v144_acc, v160_data, v88_data);
          tensorforge::fmacdpp16<9>(v144_acc, v160_data, v95_data);
          tensorforge::fmacdpp16<10>(v144_acc, v160_data, v102_data);
          tensorforge::fmacdpp16<11>(v144_acc, v160_data, v109_data);
          tensorforge::fmacdpp16<12>(v144_acc, v160_data, v116_data);
          tensorforge::fmacdpp16<13>(v144_acc, v160_data, v123_data);
          tensorforge::fmacdpp16<14>(v144_acc, v160_data, v130_data);
          tensorforge::fmacdpp16<15>(v144_acc, v160_data, v137_data);
          tensorforge::fmacdpp16<0>(v145_acc, v161_data, v32_data);
          tensorforge::fmacdpp16<1>(v145_acc, v161_data, v39_data);
          tensorforge::fmacdpp16<2>(v145_acc, v161_data, v46_data);
          tensorforge::fmacdpp16<3>(v145_acc, v161_data, v53_data);
          tensorforge::fmacdpp16<4>(v145_acc, v161_data, v60_data);
          tensorforge::fmacdpp16<5>(v145_acc, v161_data, v67_data);
          tensorforge::fmacdpp16<6>(v145_acc, v161_data, v74_data);
          tensorforge::fmacdpp16<7>(v145_acc, v161_data, v81_data);
          tensorforge::fmacdpp16<8>(v145_acc, v161_data, v88_data);
          tensorforge::fmacdpp16<9>(v145_acc, v161_data, v95_data);
          tensorforge::fmacdpp16<10>(v145_acc, v161_data, v102_data);
          tensorforge::fmacdpp16<11>(v145_acc, v161_data, v109_data);
          tensorforge::fmacdpp16<12>(v145_acc, v161_data, v116_data);
          tensorforge::fmacdpp16<13>(v145_acc, v161_data, v123_data);
          tensorforge::fmacdpp16<14>(v145_acc, v161_data, v130_data);
          tensorforge::fmacdpp16<15>(v145_acc, v161_data, v137_data);
          tensorforge::fmacdpp16<0>(v146_acc, v162_data, v32_data);
          tensorforge::fmacdpp16<1>(v146_acc, v162_data, v39_data);
          tensorforge::fmacdpp16<2>(v146_acc, v162_data, v46_data);
          tensorforge::fmacdpp16<3>(v146_acc, v162_data, v53_data);
          tensorforge::fmacdpp16<4>(v146_acc, v162_data, v60_data);
          tensorforge::fmacdpp16<5>(v146_acc, v162_data, v67_data);
          tensorforge::fmacdpp16<6>(v146_acc, v162_data, v74_data);
          tensorforge::fmacdpp16<7>(v146_acc, v162_data, v81_data);
          tensorforge::fmacdpp16<8>(v146_acc, v162_data, v88_data);
          tensorforge::fmacdpp16<9>(v146_acc, v162_data, v95_data);
          tensorforge::fmacdpp16<10>(v146_acc, v162_data, v102_data);
          tensorforge::fmacdpp16<11>(v146_acc, v162_data, v109_data);
          tensorforge::fmacdpp16<12>(v146_acc, v162_data, v116_data);
          tensorforge::fmacdpp16<13>(v146_acc, v162_data, v123_data);
          tensorforge::fmacdpp16<14>(v146_acc, v162_data, v130_data);
          tensorforge::fmacdpp16<15>(v146_acc, v162_data, v137_data);
          tensorforge::fmacdpp16<0>(v147_acc, v163_data, v32_data);
          tensorforge::fmacdpp16<1>(v147_acc, v163_data, v39_data);
          tensorforge::fmacdpp16<2>(v147_acc, v163_data, v46_data);
          tensorforge::fmacdpp16<3>(v147_acc, v163_data, v53_data);
          tensorforge::fmacdpp16<4>(v147_acc, v163_data, v60_data);
          tensorforge::fmacdpp16<5>(v147_acc, v163_data, v67_data);
          tensorforge::fmacdpp16<6>(v147_acc, v163_data, v74_data);
          tensorforge::fmacdpp16<7>(v147_acc, v163_data, v81_data);
          tensorforge::fmacdpp16<8>(v147_acc, v163_data, v88_data);
          tensorforge::fmacdpp16<9>(v147_acc, v163_data, v95_data);
          tensorforge::fmacdpp16<10>(v147_acc, v163_data, v102_data);
          tensorforge::fmacdpp16<11>(v147_acc, v163_data, v109_data);
          tensorforge::fmacdpp16<12>(v147_acc, v163_data, v116_data);
          tensorforge::fmacdpp16<13>(v147_acc, v163_data, v123_data);
          tensorforge::fmacdpp16<14>(v147_acc, v163_data, v130_data);
          tensorforge::fmacdpp16<15>(v147_acc, v163_data, v137_data);
          tensorforge::fmacdpp16<0>(v148_acc, v164_data, v32_data);
          tensorforge::fmacdpp16<1>(v148_acc, v164_data, v39_data);
          tensorforge::fmacdpp16<2>(v148_acc, v164_data, v46_data);
          tensorforge::fmacdpp16<3>(v148_acc, v164_data, v53_data);
          tensorforge::fmacdpp16<4>(v148_acc, v164_data, v60_data);
          tensorforge::fmacdpp16<5>(v148_acc, v164_data, v67_data);
          tensorforge::fmacdpp16<6>(v148_acc, v164_data, v74_data);
          tensorforge::fmacdpp16<7>(v148_acc, v164_data, v81_data);
          tensorforge::fmacdpp16<8>(v148_acc, v164_data, v88_data);
          tensorforge::fmacdpp16<9>(v148_acc, v164_data, v95_data);
          tensorforge::fmacdpp16<10>(v148_acc, v164_data, v102_data);
          tensorforge::fmacdpp16<11>(v148_acc, v164_data, v109_data);
          tensorforge::fmacdpp16<12>(v148_acc, v164_data, v116_data);
          tensorforge::fmacdpp16<13>(v148_acc, v164_data, v123_data);
          tensorforge::fmacdpp16<14>(v148_acc, v164_data, v130_data);
          tensorforge::fmacdpp16<15>(v148_acc, v164_data, v137_data);
          tensorforge::fmacdpp16<0>(v149_acc, v165_data, v32_data);
          tensorforge::fmacdpp16<1>(v149_acc, v165_data, v39_data);
          tensorforge::fmacdpp16<2>(v149_acc, v165_data, v46_data);
          tensorforge::fmacdpp16<3>(v149_acc, v165_data, v53_data);
          tensorforge::fmacdpp16<4>(v149_acc, v165_data, v60_data);
          tensorforge::fmacdpp16<5>(v149_acc, v165_data, v67_data);
          tensorforge::fmacdpp16<6>(v149_acc, v165_data, v74_data);
          tensorforge::fmacdpp16<7>(v149_acc, v165_data, v81_data);
          tensorforge::fmacdpp16<8>(v149_acc, v165_data, v88_data);
          tensorforge::fmacdpp16<9>(v149_acc, v165_data, v95_data);
          tensorforge::fmacdpp16<10>(v149_acc, v165_data, v102_data);
          tensorforge::fmacdpp16<11>(v149_acc, v165_data, v109_data);
          tensorforge::fmacdpp16<12>(v149_acc, v165_data, v116_data);
          tensorforge::fmacdpp16<13>(v149_acc, v165_data, v123_data);
          tensorforge::fmacdpp16<14>(v149_acc, v165_data, v130_data);
          tensorforge::fmacdpp16<15>(v149_acc, v165_data, v137_data);
          tensorforge::fmacdpp16<0>(v150_acc, v166_data, v32_data);
          tensorforge::fmacdpp16<1>(v150_acc, v166_data, v39_data);
          tensorforge::fmacdpp16<2>(v150_acc, v166_data, v46_data);
          tensorforge::fmacdpp16<3>(v150_acc, v166_data, v53_data);
          tensorforge::fmacdpp16<4>(v150_acc, v166_data, v60_data);
          tensorforge::fmacdpp16<5>(v150_acc, v166_data, v67_data);
          tensorforge::fmacdpp16<6>(v150_acc, v166_data, v74_data);
          tensorforge::fmacdpp16<7>(v150_acc, v166_data, v81_data);
          tensorforge::fmacdpp16<8>(v150_acc, v166_data, v88_data);
          tensorforge::fmacdpp16<9>(v150_acc, v166_data, v95_data);
          tensorforge::fmacdpp16<10>(v150_acc, v166_data, v102_data);
          tensorforge::fmacdpp16<11>(v150_acc, v166_data, v109_data);
          tensorforge::fmacdpp16<12>(v150_acc, v166_data, v116_data);
          tensorforge::fmacdpp16<13>(v150_acc, v166_data, v123_data);
          tensorforge::fmacdpp16<14>(v150_acc, v166_data, v130_data);
          tensorforge::fmacdpp16<15>(v150_acc, v166_data, v137_data);
          tensorforge::fmacdpp16<0>(v151_acc, v167_data, v32_data);
          tensorforge::fmacdpp16<1>(v151_acc, v167_data, v39_data);
          tensorforge::fmacdpp16<2>(v151_acc, v167_data, v46_data);
          tensorforge::fmacdpp16<3>(v151_acc, v167_data, v53_data);
          tensorforge::fmacdpp16<4>(v151_acc, v167_data, v60_data);
          tensorforge::fmacdpp16<5>(v151_acc, v167_data, v67_data);
          tensorforge::fmacdpp16<6>(v151_acc, v167_data, v74_data);
          tensorforge::fmacdpp16<7>(v151_acc, v167_data, v81_data);
          tensorforge::fmacdpp16<8>(v151_acc, v167_data, v88_data);
          tensorforge::fmacdpp16<9>(v151_acc, v167_data, v95_data);
          tensorforge::fmacdpp16<10>(v151_acc, v167_data, v102_data);
          tensorforge::fmacdpp16<11>(v151_acc, v167_data, v109_data);
          tensorforge::fmacdpp16<12>(v151_acc, v167_data, v116_data);
          tensorforge::fmacdpp16<13>(v151_acc, v167_data, v123_data);
          tensorforge::fmacdpp16<14>(v151_acc, v167_data, v130_data);
          tensorforge::fmacdpp16<15>(v151_acc, v167_data, v137_data);
          tensorforge::fmacdpp16<0>(v152_acc, v168_data, v32_data);
          tensorforge::fmacdpp16<1>(v152_acc, v168_data, v39_data);
          tensorforge::fmacdpp16<2>(v152_acc, v168_data, v46_data);
          tensorforge::fmacdpp16<3>(v152_acc, v168_data, v53_data);
          tensorforge::fmacdpp16<4>(v152_acc, v168_data, v60_data);
          tensorforge::fmacdpp16<5>(v152_acc, v168_data, v67_data);
          tensorforge::fmacdpp16<6>(v152_acc, v168_data, v74_data);
          tensorforge::fmacdpp16<7>(v152_acc, v168_data, v81_data);
          tensorforge::fmacdpp16<8>(v152_acc, v168_data, v88_data);
          tensorforge::fmacdpp16<9>(v152_acc, v168_data, v95_data);
          tensorforge::fmacdpp16<10>(v152_acc, v168_data, v102_data);
          tensorforge::fmacdpp16<11>(v152_acc, v168_data, v109_data);
          tensorforge::fmacdpp16<12>(v152_acc, v168_data, v116_data);
          tensorforge::fmacdpp16<13>(v152_acc, v168_data, v123_data);
          tensorforge::fmacdpp16<14>(v152_acc, v168_data, v130_data);
          tensorforge::fmacdpp16<15>(v152_acc, v168_data, v137_data);
          tensorforge::fmacdpp16<0>(v153_acc, v169_data, v32_data);
          tensorforge::fmacdpp16<1>(v153_acc, v169_data, v39_data);
          tensorforge::fmacdpp16<2>(v153_acc, v169_data, v46_data);
          tensorforge::fmacdpp16<3>(v153_acc, v169_data, v53_data);
          tensorforge::fmacdpp16<4>(v153_acc, v169_data, v60_data);
          tensorforge::fmacdpp16<5>(v153_acc, v169_data, v67_data);
          tensorforge::fmacdpp16<6>(v153_acc, v169_data, v74_data);
          tensorforge::fmacdpp16<7>(v153_acc, v169_data, v81_data);
          tensorforge::fmacdpp16<8>(v153_acc, v169_data, v88_data);
          tensorforge::fmacdpp16<9>(v153_acc, v169_data, v95_data);
          tensorforge::fmacdpp16<10>(v153_acc, v169_data, v102_data);
          tensorforge::fmacdpp16<11>(v153_acc, v169_data, v109_data);
          tensorforge::fmacdpp16<12>(v153_acc, v169_data, v116_data);
          tensorforge::fmacdpp16<13>(v153_acc, v169_data, v123_data);
          tensorforge::fmacdpp16<14>(v153_acc, v169_data, v130_data);
          tensorforge::fmacdpp16<15>(v153_acc, v169_data, v137_data);
          r1[0] = v138_acc;
          r1[1] = v139_acc;
          r1[2] = v140_acc;
          r1[3] = v141_acc;
          r1[4] = v142_acc;
          r1[5] = v143_acc;
          r1[6] = v144_acc;
          r1[7] = v145_acc;
          r1[8] = v146_acc;
          r1[9] = v147_acc;
          r1[10] = v148_acc;
          r1[11] = v149_acc;
          r1[12] = v150_acc;
          r1[13] = v151_acc;
          r1[14] = v152_acc;
          r1[15] = v153_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v173_i0 = 0; v173_i0 < 1; ++v173_i0) {
            int32_t v181_lead = v28_lane + (v173_i0 * 16);
            #pragma unroll
            for (int32_t v174_i1 = 0; v174_i1 < 16; ++v174_i1) {
              double v176_data = r1[(v173_i0 + v174_i1)];
              glb_m0[(v181_lead + (v174_i1 * 16))] = v176_data;
            }
          }
        }
      }
    }
  }
}

