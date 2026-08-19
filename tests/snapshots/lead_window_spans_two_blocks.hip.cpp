// === base name ===
kernel_671a350836

// === header ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_671a350836, block.x * block.y * block.z, 64 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_671a350836), hipFuncAttributeMaxDynamicSharedMemorySize, 64 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_671a350836, grid, block, 64 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 64×13(64×13) {0..64}×{0..13} pointer_based
    // m1 6(6) {0..6} none
    // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
    // t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..64}×{0..13}×{0..6})[0, 1, 2] = m0 64×13(64×13) {0..64}×{0..13} pointer_based({0..64}×{0..13})[0, 1]×m1 6(6) {0..6} none({0..6})[2]
    // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based({0..15}×{0..1}×{0..6})[0, 1, 2] += t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..15}×{0..1}×{0..6})[0, 1, 2]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 64];
      float* tempShrMem = &localShrMem0[0];
      const float *const __restrict__ ptr_glb_m1 = &m1[0];
      float* __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0])
      if ((threadIdx.x + threadIdx.y * blockDim.x) < 6) {
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      }
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 2; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 13; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 64);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m0[v11_a]);
              }
              int32_t v14_a = v3_i0 + (v4_i1 * 2);
              r0[v14_a] = v12_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          auto& ir1 = r1;
          float v15_data = r0[0];
          float v16_data = r0[1];
          float v17_acc{};
          float v18_acc{};
          float v19_acc{};
          float v20_acc{};
          float v21_acc{};
          float v22_acc{};
          float v23_acc{};
          float v24_acc{};
          float v25_acc{};
          float v26_acc{};
          float v27_acc{};
          float v28_acc{};
          float v29_acc{};
          float v30_acc{};
          float v31_acc{};
          float v32_acc{};
          float v33_acc{};
          float v34_acc{};
          float v35_acc{};
          float v36_acc{};
          float v37_acc{};
          float v38_acc{};
          float v39_acc{};
          float v40_acc{};
          float v41_acc{};
          float v42_acc{};
          float v43_acc{};
          float v44_acc{};
          float v45_acc{};
          float v46_acc{};
          float v47_acc{};
          float v48_acc{};
          float v49_acc{};
          float v50_acc{};
          float v51_acc{};
          float v52_acc{};
          float v53_acc{};
          float v54_acc{};
          float v55_acc{};
          float v56_acc{};
          float v57_acc{};
          float v58_acc{};
          float v59_acc{};
          float v60_acc{};
          float v61_acc{};
          float v62_acc{};
          float v63_acc{};
          float v64_acc{};
          float v65_acc{};
          float v66_acc{};
          float v67_acc{};
          float v68_acc{};
          float v69_acc{};
          float v70_acc{};
          float v71_acc{};
          float v72_acc{};
          float v73_acc{};
          float v74_acc{};
          float v75_acc{};
          float v76_acc{};
          float v77_acc{};
          float v78_acc{};
          float v79_acc{};
          float v80_acc{};
          float v81_acc{};
          float v82_acc{};
          float v83_acc{};
          float v84_acc{};
          float v85_acc{};
          float v86_acc{};
          float v87_acc{};
          float v88_acc{};
          float v89_acc{};
          float v90_acc{};
          float v91_acc{};
          float v92_acc{};
          float v93_acc{};
          float v94_acc{};
          float v95_acc{};
          float v96_acc{};
          float v97_acc{};
          float v98_acc{};
          float v99_acc{};
          float v100_acc{};
          float v101_acc{};
          float v102_acc{};
          float v103_acc{};
          float v104_acc{};
          float v105_acc{};
          float v106_acc{};
          float v107_acc{};
          float v108_acc{};
          float v109_acc{};
          float v110_acc{};
          float v111_acc{};
          float v112_acc{};
          float v113_acc{};
          float v114_acc{};
          float v115_acc{};
          float v116_acc{};
          float v117_acc{};
          float v118_acc{};
          float v119_acc{};
          float v120_acc{};
          float v121_acc{};
          float v122_acc{};
          float v123_acc{};
          float v124_acc{};
          float v125_acc{};
          float v126_acc{};
          float v127_acc{};
          float v128_acc{};
          float v129_acc{};
          float v130_acc{};
          float v131_acc{};
          float v132_acc{};
          float v133_acc{};
          float v134_acc{};
          float v135_acc{};
          float v136_acc{};
          float v137_acc{};
          float v138_acc{};
          float v139_acc{};
          float v140_acc{};
          float v141_acc{};
          float v142_acc{};
          float v143_acc{};
          float v144_acc{};
          float v145_acc{};
          float v146_acc{};
          float v147_acc{};
          float v148_acc{};
          float v149_acc{};
          float v150_acc{};
          float v151_acc{};
          float v152_acc{};
          float v153_acc{};
          float v154_acc{};
          float v155_acc{};
          float v156_acc{};
          float v157_acc{};
          float v158_acc{};
          float v159_acc{};
          float v160_acc{};
          float v161_acc{};
          float v162_acc{};
          float v163_acc{};
          float v164_acc{};
          float v165_acc{};
          float v166_acc{};
          float v167_acc{};
          float v168_acc{};
          float v169_acc{};
          float v170_acc{};
          float v171_acc{};
          float v172_acc{};
          float v173_lin = glb_m1[0 + threadIdx.x * 1];
          float v174_bc = tensorforge::broadcast<32, 16, 0>(v173_lin);
          tensorforge::fmacdpp16<0>(v17_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<0>(v18_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<1>(v19_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<1>(v20_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<2>(v21_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<2>(v22_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<3>(v23_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<3>(v24_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<4>(v25_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<4>(v26_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<5>(v27_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<5>(v28_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<6>(v29_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<6>(v30_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<7>(v31_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<7>(v32_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<8>(v33_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<8>(v34_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<9>(v35_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<9>(v36_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<10>(v37_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<10>(v38_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<11>(v39_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<11>(v40_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<12>(v41_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<12>(v42_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<13>(v43_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<13>(v44_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<14>(v45_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<14>(v46_acc, v174_bc, v16_data);
          tensorforge::fmacdpp16<15>(v47_acc, v174_bc, v15_data);
          tensorforge::fmacdpp16<15>(v48_acc, v174_bc, v16_data);
          float v175_bc = tensorforge::broadcast<32, 16, 1>(v173_lin);
          tensorforge::fmacdpp16<0>(v49_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<0>(v50_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<1>(v51_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<1>(v52_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<2>(v53_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<2>(v54_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<3>(v55_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<3>(v56_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<4>(v57_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<4>(v58_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<5>(v59_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<5>(v60_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<6>(v61_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<6>(v62_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<7>(v63_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<7>(v64_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<8>(v65_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<8>(v66_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<9>(v67_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<9>(v68_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<10>(v69_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<10>(v70_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<11>(v71_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<11>(v72_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<12>(v73_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<12>(v74_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<13>(v75_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<13>(v76_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<14>(v77_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<14>(v78_acc, v175_bc, v16_data);
          tensorforge::fmacdpp16<15>(v79_acc, v175_bc, v15_data);
          tensorforge::fmacdpp16<15>(v80_acc, v175_bc, v16_data);
          float v176_lin = glb_m1[32 + threadIdx.x * 1];
          float v177_bc = tensorforge::broadcast<32, 16, 0>(v176_lin);
          tensorforge::fmacdpp16<0>(v81_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<0>(v82_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<1>(v83_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<1>(v84_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<2>(v85_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<2>(v86_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<3>(v87_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<3>(v88_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<4>(v89_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<4>(v90_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<5>(v91_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<5>(v92_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<6>(v93_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<6>(v94_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<7>(v95_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<7>(v96_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<8>(v97_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<8>(v98_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<9>(v99_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<9>(v100_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<10>(v101_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<10>(v102_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<11>(v103_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<11>(v104_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<12>(v105_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<12>(v106_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<13>(v107_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<13>(v108_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<14>(v109_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<14>(v110_acc, v177_bc, v16_data);
          tensorforge::fmacdpp16<15>(v111_acc, v177_bc, v15_data);
          tensorforge::fmacdpp16<15>(v112_acc, v177_bc, v16_data);
          float v178_bc = tensorforge::broadcast<32, 16, 1>(v176_lin);
          tensorforge::fmacdpp16<0>(v113_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<0>(v114_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<1>(v115_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<1>(v116_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<2>(v117_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<2>(v118_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<3>(v119_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<3>(v120_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<4>(v121_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<4>(v122_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<5>(v123_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<5>(v124_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<6>(v125_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<6>(v126_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<7>(v127_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<7>(v128_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<8>(v129_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<8>(v130_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<9>(v131_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<9>(v132_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<10>(v133_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<10>(v134_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<11>(v135_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<11>(v136_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<12>(v137_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<12>(v138_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<13>(v139_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<13>(v140_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<14>(v141_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<14>(v142_acc, v178_bc, v16_data);
          tensorforge::fmacdpp16<15>(v143_acc, v178_bc, v15_data);
          tensorforge::fmacdpp16<15>(v144_acc, v178_bc, v16_data);
          float v179_lin = glb_m1[64 + threadIdx.x * 1];
          float v180_bc = tensorforge::broadcast<32, 16, 0>(v179_lin);
          tensorforge::fmacdpp16<0>(v145_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<0>(v146_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<1>(v147_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<1>(v148_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<2>(v149_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<2>(v150_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<3>(v151_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<3>(v152_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<4>(v153_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<4>(v154_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<5>(v155_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<5>(v156_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<6>(v157_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<6>(v158_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<7>(v159_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<7>(v160_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<8>(v161_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<8>(v162_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<9>(v163_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<9>(v164_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<10>(v165_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<10>(v166_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<11>(v167_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<11>(v168_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<12>(v169_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<12>(v170_acc, v180_bc, v16_data);
          tensorforge::fmacdpp16<13>(v171_acc, v180_bc, v15_data);
          tensorforge::fmacdpp16<13>(v172_acc, v180_bc, v16_data);
          ir1[0] = v17_acc;
          ir1[1] = v18_acc;
          ir1[2] = v19_acc;
          ir1[3] = v20_acc;
          ir1[4] = v21_acc;
          ir1[5] = v22_acc;
          ir1[6] = v23_acc;
          ir1[7] = v24_acc;
          ir1[8] = v25_acc;
          ir1[9] = v26_acc;
          ir1[10] = v27_acc;
          ir1[11] = v28_acc;
          ir1[12] = v29_acc;
          ir1[13] = v30_acc;
          ir1[14] = v31_acc;
          ir1[15] = v32_acc;
          ir1[16] = v33_acc;
          ir1[17] = v34_acc;
          ir1[18] = v35_acc;
          ir1[19] = v36_acc;
          ir1[20] = v37_acc;
          ir1[21] = v38_acc;
          ir1[22] = v39_acc;
          ir1[23] = v40_acc;
          ir1[24] = v41_acc;
          ir1[25] = v42_acc;
          ir1[26] = v43_acc;
          ir1[27] = v44_acc;
          ir1[28] = v45_acc;
          ir1[29] = v46_acc;
          ir1[30] = v47_acc;
          ir1[31] = v48_acc;
          ir1[32] = v49_acc;
          ir1[33] = v50_acc;
          ir1[34] = v51_acc;
          ir1[35] = v52_acc;
          ir1[36] = v53_acc;
          ir1[37] = v54_acc;
          ir1[38] = v55_acc;
          ir1[39] = v56_acc;
          ir1[40] = v57_acc;
          ir1[41] = v58_acc;
          ir1[42] = v59_acc;
          ir1[43] = v60_acc;
          ir1[44] = v61_acc;
          ir1[45] = v62_acc;
          ir1[46] = v63_acc;
          ir1[47] = v64_acc;
          ir1[48] = v65_acc;
          ir1[49] = v66_acc;
          ir1[50] = v67_acc;
          ir1[51] = v68_acc;
          ir1[52] = v69_acc;
          ir1[53] = v70_acc;
          ir1[54] = v71_acc;
          ir1[55] = v72_acc;
          ir1[56] = v73_acc;
          ir1[57] = v74_acc;
          ir1[58] = v75_acc;
          ir1[59] = v76_acc;
          ir1[60] = v77_acc;
          ir1[61] = v78_acc;
          ir1[62] = v79_acc;
          ir1[63] = v80_acc;
          ir1[64] = v81_acc;
          ir1[65] = v82_acc;
          ir1[66] = v83_acc;
          ir1[67] = v84_acc;
          ir1[68] = v85_acc;
          ir1[69] = v86_acc;
          ir1[70] = v87_acc;
          ir1[71] = v88_acc;
          ir1[72] = v89_acc;
          ir1[73] = v90_acc;
          ir1[74] = v91_acc;
          ir1[75] = v92_acc;
          ir1[76] = v93_acc;
          ir1[77] = v94_acc;
          ir1[78] = v95_acc;
          ir1[79] = v96_acc;
          ir1[80] = v97_acc;
          ir1[81] = v98_acc;
          ir1[82] = v99_acc;
          ir1[83] = v100_acc;
          ir1[84] = v101_acc;
          ir1[85] = v102_acc;
          ir1[86] = v103_acc;
          ir1[87] = v104_acc;
          ir1[88] = v105_acc;
          ir1[89] = v106_acc;
          ir1[90] = v107_acc;
          ir1[91] = v108_acc;
          ir1[92] = v109_acc;
          ir1[93] = v110_acc;
          ir1[94] = v111_acc;
          ir1[95] = v112_acc;
          ir1[96] = v113_acc;
          ir1[97] = v114_acc;
          ir1[98] = v115_acc;
          ir1[99] = v116_acc;
          ir1[100] = v117_acc;
          ir1[101] = v118_acc;
          ir1[102] = v119_acc;
          ir1[103] = v120_acc;
          ir1[104] = v121_acc;
          ir1[105] = v122_acc;
          ir1[106] = v123_acc;
          ir1[107] = v124_acc;
          ir1[108] = v125_acc;
          ir1[109] = v126_acc;
          ir1[110] = v127_acc;
          ir1[111] = v128_acc;
          ir1[112] = v129_acc;
          ir1[113] = v130_acc;
          ir1[114] = v131_acc;
          ir1[115] = v132_acc;
          ir1[116] = v133_acc;
          ir1[117] = v134_acc;
          ir1[118] = v135_acc;
          ir1[119] = v136_acc;
          ir1[120] = v137_acc;
          ir1[121] = v138_acc;
          ir1[122] = v139_acc;
          ir1[123] = v140_acc;
          ir1[124] = v141_acc;
          ir1[125] = v142_acc;
          ir1[126] = v143_acc;
          ir1[127] = v144_acc;
          ir1[128] = v145_acc;
          ir1[129] = v146_acc;
          ir1[130] = v147_acc;
          ir1[131] = v148_acc;
          ir1[132] = v149_acc;
          ir1[133] = v150_acc;
          ir1[134] = v151_acc;
          ir1[135] = v152_acc;
          ir1[136] = v153_acc;
          ir1[137] = v154_acc;
          ir1[138] = v155_acc;
          ir1[139] = v156_acc;
          ir1[140] = v157_acc;
          ir1[141] = v158_acc;
          ir1[142] = v159_acc;
          ir1[143] = v160_acc;
          ir1[144] = v161_acc;
          ir1[145] = v162_acc;
          ir1[146] = v163_acc;
          ir1[147] = v164_acc;
          ir1[148] = v165_acc;
          ir1[149] = v166_acc;
          ir1[150] = v167_acc;
          ir1[151] = v168_acc;
          ir1[152] = v169_acc;
          ir1[153] = v170_acc;
          ir1[154] = v171_acc;
          ir1[155] = v172_acc;
          float r2[156]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          auto& ir2 = r2;
          int32_t v183_lead = threadIdx.x % 32;
          if (v183_lead >= 20) {
            float v185_data = r1[24];
            float v186_data = ir2[0];
            ir2[0] = (v186_data + v185_data);
            float v188_data = r1[50];
            float v189_data = ir2[2];
            ir2[2] = (v189_data + v188_data);
            float v191_data = r1[76];
            float v192_data = ir2[4];
            ir2[4] = (v192_data + v191_data);
            float v194_data = r1[102];
            float v195_data = ir2[6];
            ir2[6] = (v195_data + v194_data);
            float v197_data = r1[128];
            float v198_data = ir2[8];
            ir2[8] = (v198_data + v197_data);
            float v200_data = r1[154];
            float v201_data = ir2[10];
            ir2[10] = (v201_data + v200_data);
          }
          if (v183_lead < 3) {
            float v204_data = r1[25];
            float v205_data = ir2[1];
            ir2[1] = (v205_data + v204_data);
            float v207_data = r1[51];
            float v208_data = ir2[3];
            ir2[3] = (v208_data + v207_data);
            float v210_data = r1[77];
            float v211_data = ir2[5];
            ir2[5] = (v211_data + v210_data);
            float v213_data = r1[103];
            float v214_data = ir2[7];
            ir2[7] = (v214_data + v213_data);
            float v216_data = r1[129];
            float v217_data = ir2[9];
            ir2[9] = (v217_data + v216_data);
            float v219_data = r1[155];
            float v220_data = ir2[11];
            ir2[11] = (v220_data + v219_data);
          }
          // glb_m2 = store{r>g}(r2);
          int32_t v224_lead = threadIdx.x % 32;
          if (v224_lead >= 20) {
            #pragma unroll
            for (int32_t v226_i1 = 0; v226_i1 < 1; ++v226_i1) {
              int32_t v228_a = v226_i1 * 2;
              int32_t v241_a = v224_lead + ((v226_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v227_i2 = 0; v227_i2 < 6; ++v227_i2) {
                int32_t v231_a = v228_a + (v227_i2 * 2);
                float v232_data = r2[v231_a];
                int32_t v242_a = v241_a + (v227_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v242_a], v232_data);
              }
            }
          }
          if (v224_lead < 3) {
            int32_t v255_lead = v224_lead + 32_i32;
            #pragma unroll
            for (int32_t v244_i1 = 0; v244_i1 < 1; ++v244_i1) {
              int32_t v248_a = 1 + (v244_i1 * 2);
              int32_t v259_a = v255_lead + ((v244_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v245_i2 = 0; v245_i2 < 6; ++v245_i2) {
                int32_t v249_a = v248_a + (v245_i2 * 2);
                float v250_data = r2[v249_a];
                int32_t v260_a = v259_a + (v245_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v260_a], v250_data);
              }
            }
          }
          ;
        }
      }
    }
  }
}

