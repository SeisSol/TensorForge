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
          int32_t v14_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
            int32_t v21_lead = v14_lead + (v15_i0 * 16);
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 16; ++v16_i1) {
              double v24_data = __builtin_nontemporal_load(&glb_m2[(v21_lead + (v16_i1 * 16))]);
              r0[(v15_i0 + v16_i1)] = v24_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double v33_data = glb_m1[v14_lead];
          double v40_data = glb_m1[(v14_lead + 16)];
          double v47_data = glb_m1[(v14_lead + 32)];
          double v54_data = glb_m1[(v14_lead + 48)];
          double v61_data = glb_m1[(v14_lead + 64)];
          double v68_data = glb_m1[(v14_lead + 80)];
          double v75_data = glb_m1[(v14_lead + 96)];
          double v82_data = glb_m1[(v14_lead + 112)];
          double v89_data = glb_m1[(v14_lead + 128)];
          double v96_data = glb_m1[(v14_lead + 144)];
          double v103_data = glb_m1[(v14_lead + 160)];
          double v110_data = glb_m1[(v14_lead + 176)];
          double v117_data = glb_m1[(v14_lead + 192)];
          double v124_data = glb_m1[(v14_lead + 208)];
          double v131_data = glb_m1[(v14_lead + 224)];
          double v138_data = glb_m1[(v14_lead + 240)];
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
          double v154_acc{};
          double v155_data = r0[0];
          double v156_data = r0[1];
          double v157_data = r0[2];
          double v158_data = r0[3];
          double v159_data = r0[4];
          double v160_data = r0[5];
          double v161_data = r0[6];
          double v162_data = r0[7];
          double v163_data = r0[8];
          double v164_data = r0[9];
          double v165_data = r0[10];
          double v166_data = r0[11];
          double v167_data = r0[12];
          double v168_data = r0[13];
          double v169_data = r0[14];
          double v170_data = r0[15];
          tensorforge::fmacdpp16<0>(v139_acc, v155_data, v33_data);
          tensorforge::fmacdpp16<1>(v139_acc, v155_data, v40_data);
          tensorforge::fmacdpp16<2>(v139_acc, v155_data, v47_data);
          tensorforge::fmacdpp16<3>(v139_acc, v155_data, v54_data);
          tensorforge::fmacdpp16<4>(v139_acc, v155_data, v61_data);
          tensorforge::fmacdpp16<5>(v139_acc, v155_data, v68_data);
          tensorforge::fmacdpp16<6>(v139_acc, v155_data, v75_data);
          tensorforge::fmacdpp16<7>(v139_acc, v155_data, v82_data);
          tensorforge::fmacdpp16<8>(v139_acc, v155_data, v89_data);
          tensorforge::fmacdpp16<9>(v139_acc, v155_data, v96_data);
          tensorforge::fmacdpp16<10>(v139_acc, v155_data, v103_data);
          tensorforge::fmacdpp16<11>(v139_acc, v155_data, v110_data);
          tensorforge::fmacdpp16<12>(v139_acc, v155_data, v117_data);
          tensorforge::fmacdpp16<13>(v139_acc, v155_data, v124_data);
          tensorforge::fmacdpp16<14>(v139_acc, v155_data, v131_data);
          tensorforge::fmacdpp16<15>(v139_acc, v155_data, v138_data);
          tensorforge::fmacdpp16<0>(v140_acc, v156_data, v33_data);
          tensorforge::fmacdpp16<1>(v140_acc, v156_data, v40_data);
          tensorforge::fmacdpp16<2>(v140_acc, v156_data, v47_data);
          tensorforge::fmacdpp16<3>(v140_acc, v156_data, v54_data);
          tensorforge::fmacdpp16<4>(v140_acc, v156_data, v61_data);
          tensorforge::fmacdpp16<5>(v140_acc, v156_data, v68_data);
          tensorforge::fmacdpp16<6>(v140_acc, v156_data, v75_data);
          tensorforge::fmacdpp16<7>(v140_acc, v156_data, v82_data);
          tensorforge::fmacdpp16<8>(v140_acc, v156_data, v89_data);
          tensorforge::fmacdpp16<9>(v140_acc, v156_data, v96_data);
          tensorforge::fmacdpp16<10>(v140_acc, v156_data, v103_data);
          tensorforge::fmacdpp16<11>(v140_acc, v156_data, v110_data);
          tensorforge::fmacdpp16<12>(v140_acc, v156_data, v117_data);
          tensorforge::fmacdpp16<13>(v140_acc, v156_data, v124_data);
          tensorforge::fmacdpp16<14>(v140_acc, v156_data, v131_data);
          tensorforge::fmacdpp16<15>(v140_acc, v156_data, v138_data);
          tensorforge::fmacdpp16<0>(v141_acc, v157_data, v33_data);
          tensorforge::fmacdpp16<1>(v141_acc, v157_data, v40_data);
          tensorforge::fmacdpp16<2>(v141_acc, v157_data, v47_data);
          tensorforge::fmacdpp16<3>(v141_acc, v157_data, v54_data);
          tensorforge::fmacdpp16<4>(v141_acc, v157_data, v61_data);
          tensorforge::fmacdpp16<5>(v141_acc, v157_data, v68_data);
          tensorforge::fmacdpp16<6>(v141_acc, v157_data, v75_data);
          tensorforge::fmacdpp16<7>(v141_acc, v157_data, v82_data);
          tensorforge::fmacdpp16<8>(v141_acc, v157_data, v89_data);
          tensorforge::fmacdpp16<9>(v141_acc, v157_data, v96_data);
          tensorforge::fmacdpp16<10>(v141_acc, v157_data, v103_data);
          tensorforge::fmacdpp16<11>(v141_acc, v157_data, v110_data);
          tensorforge::fmacdpp16<12>(v141_acc, v157_data, v117_data);
          tensorforge::fmacdpp16<13>(v141_acc, v157_data, v124_data);
          tensorforge::fmacdpp16<14>(v141_acc, v157_data, v131_data);
          tensorforge::fmacdpp16<15>(v141_acc, v157_data, v138_data);
          tensorforge::fmacdpp16<0>(v142_acc, v158_data, v33_data);
          tensorforge::fmacdpp16<1>(v142_acc, v158_data, v40_data);
          tensorforge::fmacdpp16<2>(v142_acc, v158_data, v47_data);
          tensorforge::fmacdpp16<3>(v142_acc, v158_data, v54_data);
          tensorforge::fmacdpp16<4>(v142_acc, v158_data, v61_data);
          tensorforge::fmacdpp16<5>(v142_acc, v158_data, v68_data);
          tensorforge::fmacdpp16<6>(v142_acc, v158_data, v75_data);
          tensorforge::fmacdpp16<7>(v142_acc, v158_data, v82_data);
          tensorforge::fmacdpp16<8>(v142_acc, v158_data, v89_data);
          tensorforge::fmacdpp16<9>(v142_acc, v158_data, v96_data);
          tensorforge::fmacdpp16<10>(v142_acc, v158_data, v103_data);
          tensorforge::fmacdpp16<11>(v142_acc, v158_data, v110_data);
          tensorforge::fmacdpp16<12>(v142_acc, v158_data, v117_data);
          tensorforge::fmacdpp16<13>(v142_acc, v158_data, v124_data);
          tensorforge::fmacdpp16<14>(v142_acc, v158_data, v131_data);
          tensorforge::fmacdpp16<15>(v142_acc, v158_data, v138_data);
          tensorforge::fmacdpp16<0>(v143_acc, v159_data, v33_data);
          tensorforge::fmacdpp16<1>(v143_acc, v159_data, v40_data);
          tensorforge::fmacdpp16<2>(v143_acc, v159_data, v47_data);
          tensorforge::fmacdpp16<3>(v143_acc, v159_data, v54_data);
          tensorforge::fmacdpp16<4>(v143_acc, v159_data, v61_data);
          tensorforge::fmacdpp16<5>(v143_acc, v159_data, v68_data);
          tensorforge::fmacdpp16<6>(v143_acc, v159_data, v75_data);
          tensorforge::fmacdpp16<7>(v143_acc, v159_data, v82_data);
          tensorforge::fmacdpp16<8>(v143_acc, v159_data, v89_data);
          tensorforge::fmacdpp16<9>(v143_acc, v159_data, v96_data);
          tensorforge::fmacdpp16<10>(v143_acc, v159_data, v103_data);
          tensorforge::fmacdpp16<11>(v143_acc, v159_data, v110_data);
          tensorforge::fmacdpp16<12>(v143_acc, v159_data, v117_data);
          tensorforge::fmacdpp16<13>(v143_acc, v159_data, v124_data);
          tensorforge::fmacdpp16<14>(v143_acc, v159_data, v131_data);
          tensorforge::fmacdpp16<15>(v143_acc, v159_data, v138_data);
          tensorforge::fmacdpp16<0>(v144_acc, v160_data, v33_data);
          tensorforge::fmacdpp16<1>(v144_acc, v160_data, v40_data);
          tensorforge::fmacdpp16<2>(v144_acc, v160_data, v47_data);
          tensorforge::fmacdpp16<3>(v144_acc, v160_data, v54_data);
          tensorforge::fmacdpp16<4>(v144_acc, v160_data, v61_data);
          tensorforge::fmacdpp16<5>(v144_acc, v160_data, v68_data);
          tensorforge::fmacdpp16<6>(v144_acc, v160_data, v75_data);
          tensorforge::fmacdpp16<7>(v144_acc, v160_data, v82_data);
          tensorforge::fmacdpp16<8>(v144_acc, v160_data, v89_data);
          tensorforge::fmacdpp16<9>(v144_acc, v160_data, v96_data);
          tensorforge::fmacdpp16<10>(v144_acc, v160_data, v103_data);
          tensorforge::fmacdpp16<11>(v144_acc, v160_data, v110_data);
          tensorforge::fmacdpp16<12>(v144_acc, v160_data, v117_data);
          tensorforge::fmacdpp16<13>(v144_acc, v160_data, v124_data);
          tensorforge::fmacdpp16<14>(v144_acc, v160_data, v131_data);
          tensorforge::fmacdpp16<15>(v144_acc, v160_data, v138_data);
          tensorforge::fmacdpp16<0>(v145_acc, v161_data, v33_data);
          tensorforge::fmacdpp16<1>(v145_acc, v161_data, v40_data);
          tensorforge::fmacdpp16<2>(v145_acc, v161_data, v47_data);
          tensorforge::fmacdpp16<3>(v145_acc, v161_data, v54_data);
          tensorforge::fmacdpp16<4>(v145_acc, v161_data, v61_data);
          tensorforge::fmacdpp16<5>(v145_acc, v161_data, v68_data);
          tensorforge::fmacdpp16<6>(v145_acc, v161_data, v75_data);
          tensorforge::fmacdpp16<7>(v145_acc, v161_data, v82_data);
          tensorforge::fmacdpp16<8>(v145_acc, v161_data, v89_data);
          tensorforge::fmacdpp16<9>(v145_acc, v161_data, v96_data);
          tensorforge::fmacdpp16<10>(v145_acc, v161_data, v103_data);
          tensorforge::fmacdpp16<11>(v145_acc, v161_data, v110_data);
          tensorforge::fmacdpp16<12>(v145_acc, v161_data, v117_data);
          tensorforge::fmacdpp16<13>(v145_acc, v161_data, v124_data);
          tensorforge::fmacdpp16<14>(v145_acc, v161_data, v131_data);
          tensorforge::fmacdpp16<15>(v145_acc, v161_data, v138_data);
          tensorforge::fmacdpp16<0>(v146_acc, v162_data, v33_data);
          tensorforge::fmacdpp16<1>(v146_acc, v162_data, v40_data);
          tensorforge::fmacdpp16<2>(v146_acc, v162_data, v47_data);
          tensorforge::fmacdpp16<3>(v146_acc, v162_data, v54_data);
          tensorforge::fmacdpp16<4>(v146_acc, v162_data, v61_data);
          tensorforge::fmacdpp16<5>(v146_acc, v162_data, v68_data);
          tensorforge::fmacdpp16<6>(v146_acc, v162_data, v75_data);
          tensorforge::fmacdpp16<7>(v146_acc, v162_data, v82_data);
          tensorforge::fmacdpp16<8>(v146_acc, v162_data, v89_data);
          tensorforge::fmacdpp16<9>(v146_acc, v162_data, v96_data);
          tensorforge::fmacdpp16<10>(v146_acc, v162_data, v103_data);
          tensorforge::fmacdpp16<11>(v146_acc, v162_data, v110_data);
          tensorforge::fmacdpp16<12>(v146_acc, v162_data, v117_data);
          tensorforge::fmacdpp16<13>(v146_acc, v162_data, v124_data);
          tensorforge::fmacdpp16<14>(v146_acc, v162_data, v131_data);
          tensorforge::fmacdpp16<15>(v146_acc, v162_data, v138_data);
          tensorforge::fmacdpp16<0>(v147_acc, v163_data, v33_data);
          tensorforge::fmacdpp16<1>(v147_acc, v163_data, v40_data);
          tensorforge::fmacdpp16<2>(v147_acc, v163_data, v47_data);
          tensorforge::fmacdpp16<3>(v147_acc, v163_data, v54_data);
          tensorforge::fmacdpp16<4>(v147_acc, v163_data, v61_data);
          tensorforge::fmacdpp16<5>(v147_acc, v163_data, v68_data);
          tensorforge::fmacdpp16<6>(v147_acc, v163_data, v75_data);
          tensorforge::fmacdpp16<7>(v147_acc, v163_data, v82_data);
          tensorforge::fmacdpp16<8>(v147_acc, v163_data, v89_data);
          tensorforge::fmacdpp16<9>(v147_acc, v163_data, v96_data);
          tensorforge::fmacdpp16<10>(v147_acc, v163_data, v103_data);
          tensorforge::fmacdpp16<11>(v147_acc, v163_data, v110_data);
          tensorforge::fmacdpp16<12>(v147_acc, v163_data, v117_data);
          tensorforge::fmacdpp16<13>(v147_acc, v163_data, v124_data);
          tensorforge::fmacdpp16<14>(v147_acc, v163_data, v131_data);
          tensorforge::fmacdpp16<15>(v147_acc, v163_data, v138_data);
          tensorforge::fmacdpp16<0>(v148_acc, v164_data, v33_data);
          tensorforge::fmacdpp16<1>(v148_acc, v164_data, v40_data);
          tensorforge::fmacdpp16<2>(v148_acc, v164_data, v47_data);
          tensorforge::fmacdpp16<3>(v148_acc, v164_data, v54_data);
          tensorforge::fmacdpp16<4>(v148_acc, v164_data, v61_data);
          tensorforge::fmacdpp16<5>(v148_acc, v164_data, v68_data);
          tensorforge::fmacdpp16<6>(v148_acc, v164_data, v75_data);
          tensorforge::fmacdpp16<7>(v148_acc, v164_data, v82_data);
          tensorforge::fmacdpp16<8>(v148_acc, v164_data, v89_data);
          tensorforge::fmacdpp16<9>(v148_acc, v164_data, v96_data);
          tensorforge::fmacdpp16<10>(v148_acc, v164_data, v103_data);
          tensorforge::fmacdpp16<11>(v148_acc, v164_data, v110_data);
          tensorforge::fmacdpp16<12>(v148_acc, v164_data, v117_data);
          tensorforge::fmacdpp16<13>(v148_acc, v164_data, v124_data);
          tensorforge::fmacdpp16<14>(v148_acc, v164_data, v131_data);
          tensorforge::fmacdpp16<15>(v148_acc, v164_data, v138_data);
          tensorforge::fmacdpp16<0>(v149_acc, v165_data, v33_data);
          tensorforge::fmacdpp16<1>(v149_acc, v165_data, v40_data);
          tensorforge::fmacdpp16<2>(v149_acc, v165_data, v47_data);
          tensorforge::fmacdpp16<3>(v149_acc, v165_data, v54_data);
          tensorforge::fmacdpp16<4>(v149_acc, v165_data, v61_data);
          tensorforge::fmacdpp16<5>(v149_acc, v165_data, v68_data);
          tensorforge::fmacdpp16<6>(v149_acc, v165_data, v75_data);
          tensorforge::fmacdpp16<7>(v149_acc, v165_data, v82_data);
          tensorforge::fmacdpp16<8>(v149_acc, v165_data, v89_data);
          tensorforge::fmacdpp16<9>(v149_acc, v165_data, v96_data);
          tensorforge::fmacdpp16<10>(v149_acc, v165_data, v103_data);
          tensorforge::fmacdpp16<11>(v149_acc, v165_data, v110_data);
          tensorforge::fmacdpp16<12>(v149_acc, v165_data, v117_data);
          tensorforge::fmacdpp16<13>(v149_acc, v165_data, v124_data);
          tensorforge::fmacdpp16<14>(v149_acc, v165_data, v131_data);
          tensorforge::fmacdpp16<15>(v149_acc, v165_data, v138_data);
          tensorforge::fmacdpp16<0>(v150_acc, v166_data, v33_data);
          tensorforge::fmacdpp16<1>(v150_acc, v166_data, v40_data);
          tensorforge::fmacdpp16<2>(v150_acc, v166_data, v47_data);
          tensorforge::fmacdpp16<3>(v150_acc, v166_data, v54_data);
          tensorforge::fmacdpp16<4>(v150_acc, v166_data, v61_data);
          tensorforge::fmacdpp16<5>(v150_acc, v166_data, v68_data);
          tensorforge::fmacdpp16<6>(v150_acc, v166_data, v75_data);
          tensorforge::fmacdpp16<7>(v150_acc, v166_data, v82_data);
          tensorforge::fmacdpp16<8>(v150_acc, v166_data, v89_data);
          tensorforge::fmacdpp16<9>(v150_acc, v166_data, v96_data);
          tensorforge::fmacdpp16<10>(v150_acc, v166_data, v103_data);
          tensorforge::fmacdpp16<11>(v150_acc, v166_data, v110_data);
          tensorforge::fmacdpp16<12>(v150_acc, v166_data, v117_data);
          tensorforge::fmacdpp16<13>(v150_acc, v166_data, v124_data);
          tensorforge::fmacdpp16<14>(v150_acc, v166_data, v131_data);
          tensorforge::fmacdpp16<15>(v150_acc, v166_data, v138_data);
          tensorforge::fmacdpp16<0>(v151_acc, v167_data, v33_data);
          tensorforge::fmacdpp16<1>(v151_acc, v167_data, v40_data);
          tensorforge::fmacdpp16<2>(v151_acc, v167_data, v47_data);
          tensorforge::fmacdpp16<3>(v151_acc, v167_data, v54_data);
          tensorforge::fmacdpp16<4>(v151_acc, v167_data, v61_data);
          tensorforge::fmacdpp16<5>(v151_acc, v167_data, v68_data);
          tensorforge::fmacdpp16<6>(v151_acc, v167_data, v75_data);
          tensorforge::fmacdpp16<7>(v151_acc, v167_data, v82_data);
          tensorforge::fmacdpp16<8>(v151_acc, v167_data, v89_data);
          tensorforge::fmacdpp16<9>(v151_acc, v167_data, v96_data);
          tensorforge::fmacdpp16<10>(v151_acc, v167_data, v103_data);
          tensorforge::fmacdpp16<11>(v151_acc, v167_data, v110_data);
          tensorforge::fmacdpp16<12>(v151_acc, v167_data, v117_data);
          tensorforge::fmacdpp16<13>(v151_acc, v167_data, v124_data);
          tensorforge::fmacdpp16<14>(v151_acc, v167_data, v131_data);
          tensorforge::fmacdpp16<15>(v151_acc, v167_data, v138_data);
          tensorforge::fmacdpp16<0>(v152_acc, v168_data, v33_data);
          tensorforge::fmacdpp16<1>(v152_acc, v168_data, v40_data);
          tensorforge::fmacdpp16<2>(v152_acc, v168_data, v47_data);
          tensorforge::fmacdpp16<3>(v152_acc, v168_data, v54_data);
          tensorforge::fmacdpp16<4>(v152_acc, v168_data, v61_data);
          tensorforge::fmacdpp16<5>(v152_acc, v168_data, v68_data);
          tensorforge::fmacdpp16<6>(v152_acc, v168_data, v75_data);
          tensorforge::fmacdpp16<7>(v152_acc, v168_data, v82_data);
          tensorforge::fmacdpp16<8>(v152_acc, v168_data, v89_data);
          tensorforge::fmacdpp16<9>(v152_acc, v168_data, v96_data);
          tensorforge::fmacdpp16<10>(v152_acc, v168_data, v103_data);
          tensorforge::fmacdpp16<11>(v152_acc, v168_data, v110_data);
          tensorforge::fmacdpp16<12>(v152_acc, v168_data, v117_data);
          tensorforge::fmacdpp16<13>(v152_acc, v168_data, v124_data);
          tensorforge::fmacdpp16<14>(v152_acc, v168_data, v131_data);
          tensorforge::fmacdpp16<15>(v152_acc, v168_data, v138_data);
          tensorforge::fmacdpp16<0>(v153_acc, v169_data, v33_data);
          tensorforge::fmacdpp16<1>(v153_acc, v169_data, v40_data);
          tensorforge::fmacdpp16<2>(v153_acc, v169_data, v47_data);
          tensorforge::fmacdpp16<3>(v153_acc, v169_data, v54_data);
          tensorforge::fmacdpp16<4>(v153_acc, v169_data, v61_data);
          tensorforge::fmacdpp16<5>(v153_acc, v169_data, v68_data);
          tensorforge::fmacdpp16<6>(v153_acc, v169_data, v75_data);
          tensorforge::fmacdpp16<7>(v153_acc, v169_data, v82_data);
          tensorforge::fmacdpp16<8>(v153_acc, v169_data, v89_data);
          tensorforge::fmacdpp16<9>(v153_acc, v169_data, v96_data);
          tensorforge::fmacdpp16<10>(v153_acc, v169_data, v103_data);
          tensorforge::fmacdpp16<11>(v153_acc, v169_data, v110_data);
          tensorforge::fmacdpp16<12>(v153_acc, v169_data, v117_data);
          tensorforge::fmacdpp16<13>(v153_acc, v169_data, v124_data);
          tensorforge::fmacdpp16<14>(v153_acc, v169_data, v131_data);
          tensorforge::fmacdpp16<15>(v153_acc, v169_data, v138_data);
          tensorforge::fmacdpp16<0>(v154_acc, v170_data, v33_data);
          tensorforge::fmacdpp16<1>(v154_acc, v170_data, v40_data);
          tensorforge::fmacdpp16<2>(v154_acc, v170_data, v47_data);
          tensorforge::fmacdpp16<3>(v154_acc, v170_data, v54_data);
          tensorforge::fmacdpp16<4>(v154_acc, v170_data, v61_data);
          tensorforge::fmacdpp16<5>(v154_acc, v170_data, v68_data);
          tensorforge::fmacdpp16<6>(v154_acc, v170_data, v75_data);
          tensorforge::fmacdpp16<7>(v154_acc, v170_data, v82_data);
          tensorforge::fmacdpp16<8>(v154_acc, v170_data, v89_data);
          tensorforge::fmacdpp16<9>(v154_acc, v170_data, v96_data);
          tensorforge::fmacdpp16<10>(v154_acc, v170_data, v103_data);
          tensorforge::fmacdpp16<11>(v154_acc, v170_data, v110_data);
          tensorforge::fmacdpp16<12>(v154_acc, v170_data, v117_data);
          tensorforge::fmacdpp16<13>(v154_acc, v170_data, v124_data);
          tensorforge::fmacdpp16<14>(v154_acc, v170_data, v131_data);
          tensorforge::fmacdpp16<15>(v154_acc, v170_data, v138_data);
          r1[0] = v139_acc;
          r1[1] = v140_acc;
          r1[2] = v141_acc;
          r1[3] = v142_acc;
          r1[4] = v143_acc;
          r1[5] = v144_acc;
          r1[6] = v145_acc;
          r1[7] = v146_acc;
          r1[8] = v147_acc;
          r1[9] = v148_acc;
          r1[10] = v149_acc;
          r1[11] = v150_acc;
          r1[12] = v151_acc;
          r1[13] = v152_acc;
          r1[14] = v153_acc;
          r1[15] = v154_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v174_i0 = 0; v174_i0 < 1; ++v174_i0) {
            int32_t v182_lead = v14_lead + (v174_i0 * 16);
            #pragma unroll
            for (int32_t v175_i1 = 0; v175_i1 < 16; ++v175_i1) {
              double v177_data = r1[(v174_i0 + v175_i1)];
              glb_m0[(v182_lead + (v175_i1 * 16))] = v177_data;
            }
          }
        }
      }
    }
  }
}

