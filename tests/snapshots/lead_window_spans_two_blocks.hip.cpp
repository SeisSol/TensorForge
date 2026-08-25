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
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 13; ++v4_i1) {
              int32_t v10_a = v4_i1 * 64;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m0[(v16_lead + v10_a)]);
              int32_t v21_a = v3_i0 + (v4_i1 * 2);
              r0[v21_a] = v19_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          auto& ir1 = r1;
          float v22_data = r0[0];
          float v23_data = r0[1];
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
          float v173_acc{};
          float v174_acc{};
          float v175_acc{};
          float v176_acc{};
          float v177_acc{};
          float v178_acc{};
          float v179_acc{};
          float v180_lin = glb_m1[0 + threadIdx.x * 1];
          float v181_bc = tensorforge::broadcast<32, 16, 0>(v180_lin);
          tensorforge::fmacdpp16<0>(v24_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<0>(v25_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<1>(v26_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<1>(v27_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<2>(v28_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<2>(v29_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<3>(v30_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<3>(v31_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<4>(v32_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<4>(v33_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<5>(v34_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<5>(v35_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<6>(v36_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<6>(v37_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<7>(v38_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<7>(v39_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<8>(v40_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<8>(v41_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<9>(v42_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<9>(v43_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<10>(v44_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<10>(v45_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<11>(v46_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<11>(v47_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<12>(v48_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<12>(v49_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<13>(v50_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<13>(v51_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<14>(v52_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<14>(v53_acc, v181_bc, v23_data);
          tensorforge::fmacdpp16<15>(v54_acc, v181_bc, v22_data);
          tensorforge::fmacdpp16<15>(v55_acc, v181_bc, v23_data);
          float v182_bc = tensorforge::broadcast<32, 16, 1>(v180_lin);
          tensorforge::fmacdpp16<0>(v56_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<0>(v57_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<1>(v58_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<1>(v59_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<2>(v60_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<2>(v61_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<3>(v62_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<3>(v63_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<4>(v64_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<4>(v65_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<5>(v66_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<5>(v67_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<6>(v68_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<6>(v69_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<7>(v70_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<7>(v71_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<8>(v72_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<8>(v73_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<9>(v74_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<9>(v75_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<10>(v76_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<10>(v77_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<11>(v78_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<11>(v79_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<12>(v80_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<12>(v81_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<13>(v82_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<13>(v83_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<14>(v84_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<14>(v85_acc, v182_bc, v23_data);
          tensorforge::fmacdpp16<15>(v86_acc, v182_bc, v22_data);
          tensorforge::fmacdpp16<15>(v87_acc, v182_bc, v23_data);
          float v183_lin = glb_m1[32 + threadIdx.x * 1];
          float v184_bc = tensorforge::broadcast<32, 16, 0>(v183_lin);
          tensorforge::fmacdpp16<0>(v88_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<0>(v89_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<1>(v90_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<1>(v91_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<2>(v92_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<2>(v93_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<3>(v94_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<3>(v95_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<4>(v96_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<4>(v97_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<5>(v98_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<5>(v99_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<6>(v100_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<6>(v101_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<7>(v102_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<7>(v103_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<8>(v104_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<8>(v105_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<9>(v106_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<9>(v107_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<10>(v108_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<10>(v109_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<11>(v110_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<11>(v111_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<12>(v112_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<12>(v113_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<13>(v114_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<13>(v115_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<14>(v116_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<14>(v117_acc, v184_bc, v23_data);
          tensorforge::fmacdpp16<15>(v118_acc, v184_bc, v22_data);
          tensorforge::fmacdpp16<15>(v119_acc, v184_bc, v23_data);
          float v185_bc = tensorforge::broadcast<32, 16, 1>(v183_lin);
          tensorforge::fmacdpp16<0>(v120_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<0>(v121_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<1>(v122_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<1>(v123_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<2>(v124_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<2>(v125_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<3>(v126_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<3>(v127_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<4>(v128_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<4>(v129_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<5>(v130_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<5>(v131_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<6>(v132_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<6>(v133_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<7>(v134_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<7>(v135_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<8>(v136_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<8>(v137_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<9>(v138_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<9>(v139_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<10>(v140_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<10>(v141_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<11>(v142_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<11>(v143_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<12>(v144_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<12>(v145_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<13>(v146_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<13>(v147_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<14>(v148_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<14>(v149_acc, v185_bc, v23_data);
          tensorforge::fmacdpp16<15>(v150_acc, v185_bc, v22_data);
          tensorforge::fmacdpp16<15>(v151_acc, v185_bc, v23_data);
          float v186_lin = glb_m1[64 + threadIdx.x * 1];
          float v187_bc = tensorforge::broadcast<32, 16, 0>(v186_lin);
          tensorforge::fmacdpp16<0>(v152_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<0>(v153_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<1>(v154_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<1>(v155_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<2>(v156_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<2>(v157_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<3>(v158_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<3>(v159_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<4>(v160_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<4>(v161_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<5>(v162_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<5>(v163_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<6>(v164_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<6>(v165_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<7>(v166_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<7>(v167_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<8>(v168_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<8>(v169_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<9>(v170_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<9>(v171_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<10>(v172_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<10>(v173_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<11>(v174_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<11>(v175_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<12>(v176_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<12>(v177_acc, v187_bc, v23_data);
          tensorforge::fmacdpp16<13>(v178_acc, v187_bc, v22_data);
          tensorforge::fmacdpp16<13>(v179_acc, v187_bc, v23_data);
          ir1[0] = v24_acc;
          ir1[1] = v25_acc;
          ir1[2] = v26_acc;
          ir1[3] = v27_acc;
          ir1[4] = v28_acc;
          ir1[5] = v29_acc;
          ir1[6] = v30_acc;
          ir1[7] = v31_acc;
          ir1[8] = v32_acc;
          ir1[9] = v33_acc;
          ir1[10] = v34_acc;
          ir1[11] = v35_acc;
          ir1[12] = v36_acc;
          ir1[13] = v37_acc;
          ir1[14] = v38_acc;
          ir1[15] = v39_acc;
          ir1[16] = v40_acc;
          ir1[17] = v41_acc;
          ir1[18] = v42_acc;
          ir1[19] = v43_acc;
          ir1[20] = v44_acc;
          ir1[21] = v45_acc;
          ir1[22] = v46_acc;
          ir1[23] = v47_acc;
          ir1[24] = v48_acc;
          ir1[25] = v49_acc;
          ir1[26] = v50_acc;
          ir1[27] = v51_acc;
          ir1[28] = v52_acc;
          ir1[29] = v53_acc;
          ir1[30] = v54_acc;
          ir1[31] = v55_acc;
          ir1[32] = v56_acc;
          ir1[33] = v57_acc;
          ir1[34] = v58_acc;
          ir1[35] = v59_acc;
          ir1[36] = v60_acc;
          ir1[37] = v61_acc;
          ir1[38] = v62_acc;
          ir1[39] = v63_acc;
          ir1[40] = v64_acc;
          ir1[41] = v65_acc;
          ir1[42] = v66_acc;
          ir1[43] = v67_acc;
          ir1[44] = v68_acc;
          ir1[45] = v69_acc;
          ir1[46] = v70_acc;
          ir1[47] = v71_acc;
          ir1[48] = v72_acc;
          ir1[49] = v73_acc;
          ir1[50] = v74_acc;
          ir1[51] = v75_acc;
          ir1[52] = v76_acc;
          ir1[53] = v77_acc;
          ir1[54] = v78_acc;
          ir1[55] = v79_acc;
          ir1[56] = v80_acc;
          ir1[57] = v81_acc;
          ir1[58] = v82_acc;
          ir1[59] = v83_acc;
          ir1[60] = v84_acc;
          ir1[61] = v85_acc;
          ir1[62] = v86_acc;
          ir1[63] = v87_acc;
          ir1[64] = v88_acc;
          ir1[65] = v89_acc;
          ir1[66] = v90_acc;
          ir1[67] = v91_acc;
          ir1[68] = v92_acc;
          ir1[69] = v93_acc;
          ir1[70] = v94_acc;
          ir1[71] = v95_acc;
          ir1[72] = v96_acc;
          ir1[73] = v97_acc;
          ir1[74] = v98_acc;
          ir1[75] = v99_acc;
          ir1[76] = v100_acc;
          ir1[77] = v101_acc;
          ir1[78] = v102_acc;
          ir1[79] = v103_acc;
          ir1[80] = v104_acc;
          ir1[81] = v105_acc;
          ir1[82] = v106_acc;
          ir1[83] = v107_acc;
          ir1[84] = v108_acc;
          ir1[85] = v109_acc;
          ir1[86] = v110_acc;
          ir1[87] = v111_acc;
          ir1[88] = v112_acc;
          ir1[89] = v113_acc;
          ir1[90] = v114_acc;
          ir1[91] = v115_acc;
          ir1[92] = v116_acc;
          ir1[93] = v117_acc;
          ir1[94] = v118_acc;
          ir1[95] = v119_acc;
          ir1[96] = v120_acc;
          ir1[97] = v121_acc;
          ir1[98] = v122_acc;
          ir1[99] = v123_acc;
          ir1[100] = v124_acc;
          ir1[101] = v125_acc;
          ir1[102] = v126_acc;
          ir1[103] = v127_acc;
          ir1[104] = v128_acc;
          ir1[105] = v129_acc;
          ir1[106] = v130_acc;
          ir1[107] = v131_acc;
          ir1[108] = v132_acc;
          ir1[109] = v133_acc;
          ir1[110] = v134_acc;
          ir1[111] = v135_acc;
          ir1[112] = v136_acc;
          ir1[113] = v137_acc;
          ir1[114] = v138_acc;
          ir1[115] = v139_acc;
          ir1[116] = v140_acc;
          ir1[117] = v141_acc;
          ir1[118] = v142_acc;
          ir1[119] = v143_acc;
          ir1[120] = v144_acc;
          ir1[121] = v145_acc;
          ir1[122] = v146_acc;
          ir1[123] = v147_acc;
          ir1[124] = v148_acc;
          ir1[125] = v149_acc;
          ir1[126] = v150_acc;
          ir1[127] = v151_acc;
          ir1[128] = v152_acc;
          ir1[129] = v153_acc;
          ir1[130] = v154_acc;
          ir1[131] = v155_acc;
          ir1[132] = v156_acc;
          ir1[133] = v157_acc;
          ir1[134] = v158_acc;
          ir1[135] = v159_acc;
          ir1[136] = v160_acc;
          ir1[137] = v161_acc;
          ir1[138] = v162_acc;
          ir1[139] = v163_acc;
          ir1[140] = v164_acc;
          ir1[141] = v165_acc;
          ir1[142] = v166_acc;
          ir1[143] = v167_acc;
          ir1[144] = v168_acc;
          ir1[145] = v169_acc;
          ir1[146] = v170_acc;
          ir1[147] = v171_acc;
          ir1[148] = v172_acc;
          ir1[149] = v173_acc;
          ir1[150] = v174_acc;
          ir1[151] = v175_acc;
          ir1[152] = v176_acc;
          ir1[153] = v177_acc;
          ir1[154] = v178_acc;
          ir1[155] = v179_acc;
          float r2[156]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          auto& ir2 = r2;
          int32_t v190_lead = threadIdx.x % 32;
          if (v190_lead >= 20) {
            float v192_data = r1[24];
            float v193_data = ir2[0];
            ir2[0] = (v193_data + v192_data);
            float v195_data = r1[50];
            float v196_data = ir2[2];
            ir2[2] = (v196_data + v195_data);
            float v198_data = r1[76];
            float v199_data = ir2[4];
            ir2[4] = (v199_data + v198_data);
            float v201_data = r1[102];
            float v202_data = ir2[6];
            ir2[6] = (v202_data + v201_data);
            float v204_data = r1[128];
            float v205_data = ir2[8];
            ir2[8] = (v205_data + v204_data);
            float v207_data = r1[154];
            float v208_data = ir2[10];
            ir2[10] = (v208_data + v207_data);
          }
          if (v190_lead < 3) {
            float v211_data = r1[25];
            float v212_data = ir2[1];
            ir2[1] = (v212_data + v211_data);
            float v214_data = r1[51];
            float v215_data = ir2[3];
            ir2[3] = (v215_data + v214_data);
            float v217_data = r1[77];
            float v218_data = ir2[5];
            ir2[5] = (v218_data + v217_data);
            float v220_data = r1[103];
            float v221_data = ir2[7];
            ir2[7] = (v221_data + v220_data);
            float v223_data = r1[129];
            float v224_data = ir2[9];
            ir2[9] = (v224_data + v223_data);
            float v226_data = r1[155];
            float v227_data = ir2[11];
            ir2[11] = (v227_data + v226_data);
          }
          // glb_m2 = store{r>g}(r2);
          int32_t v231_lead = threadIdx.x % 32;
          if (v231_lead >= 20) {
            #pragma unroll
            for (int32_t v233_i1 = 0; v233_i1 < 1; ++v233_i1) {
              int32_t v235_a = v233_i1 * 2;
              int32_t v252_a = v231_lead + ((v233_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v234_i2 = 0; v234_i2 < 6; ++v234_i2) {
                int32_t v236_a = v234_i2 * 2;
                int32_t v238_a = v235_a + v236_a;
                float v243_data = r2[(v235_a + v236_a)];
                int32_t v253_a = v252_a + (v234_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v253_a], v243_data);
              }
            }
          }
          if (v231_lead < 3) {
            int32_t v270_lead = v231_lead + 32_i32;
            #pragma unroll
            for (int32_t v255_i1 = 0; v255_i1 < 1; ++v255_i1) {
              int32_t v259_a = 1 + (v255_i1 * 2);
              int32_t v274_a = v270_lead + ((v255_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v256_i2 = 0; v256_i2 < 6; ++v256_i2) {
                int32_t v258_a = v256_i2 * 2;
                int32_t v260_a = v259_a + v258_a;
                float v265_data = r2[(v259_a + v258_a)];
                int32_t v275_a = v274_a + (v256_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v275_a], v265_data);
              }
            }
          }
          ;
        }
      }
    }
  }
}

