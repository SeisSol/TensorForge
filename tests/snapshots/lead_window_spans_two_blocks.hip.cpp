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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v11_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v12_i0 = 0; v12_i0 < 2; ++v12_i0) {
            int32_t v18_lead = v11_lead + (v12_i0 * 32);
            #pragma unroll
            for (int32_t v13_i1 = 0; v13_i1 < 13; ++v13_i1) {
              float v21_data = __builtin_nontemporal_load(&glb_m0[(v18_lead + (v13_i1 * 64))]);
              r0[(v12_i0 + (v13_i1 * 2))] = v21_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v25_data = r0[0];
          float v26_data = r0[1];
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
          float v180_acc{};
          float v181_acc{};
          float v182_acc{};
          float v183_lin = glb_m1[0 + threadIdx.x * 1];
          float v184_bc = tensorforge::broadcast<32, 16, 0>(v183_lin);
          tensorforge::fmacdpp16<0>(v27_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<0>(v28_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<1>(v29_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<1>(v30_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<2>(v31_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<2>(v32_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<3>(v33_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<3>(v34_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<4>(v35_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<4>(v36_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<5>(v37_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<5>(v38_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<6>(v39_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<6>(v40_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<7>(v41_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<7>(v42_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<8>(v43_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<8>(v44_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<9>(v45_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<9>(v46_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<10>(v47_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<10>(v48_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<11>(v49_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<11>(v50_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<12>(v51_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<12>(v52_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<13>(v53_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<13>(v54_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<14>(v55_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<14>(v56_acc, v184_bc, v26_data);
          tensorforge::fmacdpp16<15>(v57_acc, v184_bc, v25_data);
          tensorforge::fmacdpp16<15>(v58_acc, v184_bc, v26_data);
          float v185_bc = tensorforge::broadcast<32, 16, 1>(v183_lin);
          tensorforge::fmacdpp16<0>(v59_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<0>(v60_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<1>(v61_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<1>(v62_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<2>(v63_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<2>(v64_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<3>(v65_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<3>(v66_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<4>(v67_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<4>(v68_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<5>(v69_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<5>(v70_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<6>(v71_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<6>(v72_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<7>(v73_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<7>(v74_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<8>(v75_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<8>(v76_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<9>(v77_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<9>(v78_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<10>(v79_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<10>(v80_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<11>(v81_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<11>(v82_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<12>(v83_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<12>(v84_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<13>(v85_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<13>(v86_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<14>(v87_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<14>(v88_acc, v185_bc, v26_data);
          tensorforge::fmacdpp16<15>(v89_acc, v185_bc, v25_data);
          tensorforge::fmacdpp16<15>(v90_acc, v185_bc, v26_data);
          float v186_lin = glb_m1[32 + threadIdx.x * 1];
          float v187_bc = tensorforge::broadcast<32, 16, 0>(v186_lin);
          tensorforge::fmacdpp16<0>(v91_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<0>(v92_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<1>(v93_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<1>(v94_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<2>(v95_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<2>(v96_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<3>(v97_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<3>(v98_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<4>(v99_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<4>(v100_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<5>(v101_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<5>(v102_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<6>(v103_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<6>(v104_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<7>(v105_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<7>(v106_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<8>(v107_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<8>(v108_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<9>(v109_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<9>(v110_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<10>(v111_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<10>(v112_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<11>(v113_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<11>(v114_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<12>(v115_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<12>(v116_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<13>(v117_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<13>(v118_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<14>(v119_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<14>(v120_acc, v187_bc, v26_data);
          tensorforge::fmacdpp16<15>(v121_acc, v187_bc, v25_data);
          tensorforge::fmacdpp16<15>(v122_acc, v187_bc, v26_data);
          float v188_bc = tensorforge::broadcast<32, 16, 1>(v186_lin);
          tensorforge::fmacdpp16<0>(v123_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<0>(v124_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<1>(v125_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<1>(v126_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<2>(v127_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<2>(v128_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<3>(v129_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<3>(v130_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<4>(v131_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<4>(v132_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<5>(v133_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<5>(v134_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<6>(v135_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<6>(v136_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<7>(v137_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<7>(v138_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<8>(v139_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<8>(v140_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<9>(v141_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<9>(v142_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<10>(v143_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<10>(v144_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<11>(v145_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<11>(v146_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<12>(v147_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<12>(v148_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<13>(v149_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<13>(v150_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<14>(v151_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<14>(v152_acc, v188_bc, v26_data);
          tensorforge::fmacdpp16<15>(v153_acc, v188_bc, v25_data);
          tensorforge::fmacdpp16<15>(v154_acc, v188_bc, v26_data);
          float v189_lin = glb_m1[64 + threadIdx.x * 1];
          float v190_bc = tensorforge::broadcast<32, 16, 0>(v189_lin);
          tensorforge::fmacdpp16<0>(v155_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<0>(v156_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<1>(v157_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<1>(v158_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<2>(v159_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<2>(v160_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<3>(v161_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<3>(v162_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<4>(v163_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<4>(v164_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<5>(v165_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<5>(v166_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<6>(v167_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<6>(v168_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<7>(v169_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<7>(v170_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<8>(v171_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<8>(v172_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<9>(v173_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<9>(v174_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<10>(v175_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<10>(v176_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<11>(v177_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<11>(v178_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<12>(v179_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<12>(v180_acc, v190_bc, v26_data);
          tensorforge::fmacdpp16<13>(v181_acc, v190_bc, v25_data);
          tensorforge::fmacdpp16<13>(v182_acc, v190_bc, v26_data);
          r1[0] = v27_acc;
          r1[1] = v28_acc;
          r1[2] = v29_acc;
          r1[3] = v30_acc;
          r1[4] = v31_acc;
          r1[5] = v32_acc;
          r1[6] = v33_acc;
          r1[7] = v34_acc;
          r1[8] = v35_acc;
          r1[9] = v36_acc;
          r1[10] = v37_acc;
          r1[11] = v38_acc;
          r1[12] = v39_acc;
          r1[13] = v40_acc;
          r1[14] = v41_acc;
          r1[15] = v42_acc;
          r1[16] = v43_acc;
          r1[17] = v44_acc;
          r1[18] = v45_acc;
          r1[19] = v46_acc;
          r1[20] = v47_acc;
          r1[21] = v48_acc;
          r1[22] = v49_acc;
          r1[23] = v50_acc;
          r1[24] = v51_acc;
          r1[25] = v52_acc;
          r1[26] = v53_acc;
          r1[27] = v54_acc;
          r1[28] = v55_acc;
          r1[29] = v56_acc;
          r1[30] = v57_acc;
          r1[31] = v58_acc;
          r1[32] = v59_acc;
          r1[33] = v60_acc;
          r1[34] = v61_acc;
          r1[35] = v62_acc;
          r1[36] = v63_acc;
          r1[37] = v64_acc;
          r1[38] = v65_acc;
          r1[39] = v66_acc;
          r1[40] = v67_acc;
          r1[41] = v68_acc;
          r1[42] = v69_acc;
          r1[43] = v70_acc;
          r1[44] = v71_acc;
          r1[45] = v72_acc;
          r1[46] = v73_acc;
          r1[47] = v74_acc;
          r1[48] = v75_acc;
          r1[49] = v76_acc;
          r1[50] = v77_acc;
          r1[51] = v78_acc;
          r1[52] = v79_acc;
          r1[53] = v80_acc;
          r1[54] = v81_acc;
          r1[55] = v82_acc;
          r1[56] = v83_acc;
          r1[57] = v84_acc;
          r1[58] = v85_acc;
          r1[59] = v86_acc;
          r1[60] = v87_acc;
          r1[61] = v88_acc;
          r1[62] = v89_acc;
          r1[63] = v90_acc;
          r1[64] = v91_acc;
          r1[65] = v92_acc;
          r1[66] = v93_acc;
          r1[67] = v94_acc;
          r1[68] = v95_acc;
          r1[69] = v96_acc;
          r1[70] = v97_acc;
          r1[71] = v98_acc;
          r1[72] = v99_acc;
          r1[73] = v100_acc;
          r1[74] = v101_acc;
          r1[75] = v102_acc;
          r1[76] = v103_acc;
          r1[77] = v104_acc;
          r1[78] = v105_acc;
          r1[79] = v106_acc;
          r1[80] = v107_acc;
          r1[81] = v108_acc;
          r1[82] = v109_acc;
          r1[83] = v110_acc;
          r1[84] = v111_acc;
          r1[85] = v112_acc;
          r1[86] = v113_acc;
          r1[87] = v114_acc;
          r1[88] = v115_acc;
          r1[89] = v116_acc;
          r1[90] = v117_acc;
          r1[91] = v118_acc;
          r1[92] = v119_acc;
          r1[93] = v120_acc;
          r1[94] = v121_acc;
          r1[95] = v122_acc;
          r1[96] = v123_acc;
          r1[97] = v124_acc;
          r1[98] = v125_acc;
          r1[99] = v126_acc;
          r1[100] = v127_acc;
          r1[101] = v128_acc;
          r1[102] = v129_acc;
          r1[103] = v130_acc;
          r1[104] = v131_acc;
          r1[105] = v132_acc;
          r1[106] = v133_acc;
          r1[107] = v134_acc;
          r1[108] = v135_acc;
          r1[109] = v136_acc;
          r1[110] = v137_acc;
          r1[111] = v138_acc;
          r1[112] = v139_acc;
          r1[113] = v140_acc;
          r1[114] = v141_acc;
          r1[115] = v142_acc;
          r1[116] = v143_acc;
          r1[117] = v144_acc;
          r1[118] = v145_acc;
          r1[119] = v146_acc;
          r1[120] = v147_acc;
          r1[121] = v148_acc;
          r1[122] = v149_acc;
          r1[123] = v150_acc;
          r1[124] = v151_acc;
          r1[125] = v152_acc;
          r1[126] = v153_acc;
          r1[127] = v154_acc;
          r1[128] = v155_acc;
          r1[129] = v156_acc;
          r1[130] = v157_acc;
          r1[131] = v158_acc;
          r1[132] = v159_acc;
          r1[133] = v160_acc;
          r1[134] = v161_acc;
          r1[135] = v162_acc;
          r1[136] = v163_acc;
          r1[137] = v164_acc;
          r1[138] = v165_acc;
          r1[139] = v166_acc;
          r1[140] = v167_acc;
          r1[141] = v168_acc;
          r1[142] = v169_acc;
          r1[143] = v170_acc;
          r1[144] = v171_acc;
          r1[145] = v172_acc;
          r1[146] = v173_acc;
          r1[147] = v174_acc;
          r1[148] = v175_acc;
          r1[149] = v176_acc;
          r1[150] = v177_acc;
          r1[151] = v178_acc;
          r1[152] = v179_acc;
          r1[153] = v180_acc;
          r1[154] = v181_acc;
          r1[155] = v182_acc;
          float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          if (v11_lead >= 20) {
            float v196_data = r1[24];
            float v197_data = r2[0];
            r2[0] = (v197_data + v196_data);
            float v199_data = r1[50];
            float v200_data = r2[2];
            r2[2] = (v200_data + v199_data);
            float v202_data = r1[76];
            float v203_data = r2[4];
            r2[4] = (v203_data + v202_data);
            float v205_data = r1[102];
            float v206_data = r2[6];
            r2[6] = (v206_data + v205_data);
            float v208_data = r1[128];
            float v209_data = r2[8];
            r2[8] = (v209_data + v208_data);
            float v211_data = r1[154];
            float v212_data = r2[10];
            r2[10] = (v212_data + v211_data);
          }
          if (v11_lead < 3) {
            float v215_data = r1[25];
            float v216_data = r2[1];
            r2[1] = (v216_data + v215_data);
            float v218_data = r1[51];
            float v219_data = r2[3];
            r2[3] = (v219_data + v218_data);
            float v221_data = r1[77];
            float v222_data = r2[5];
            r2[5] = (v222_data + v221_data);
            float v224_data = r1[103];
            float v225_data = r2[7];
            r2[7] = (v225_data + v224_data);
            float v227_data = r1[129];
            float v228_data = r2[9];
            r2[9] = (v228_data + v227_data);
            float v230_data = r1[155];
            float v231_data = r2[11];
            r2[11] = (v231_data + v230_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v11_lead >= 20) {
            #pragma unroll
            for (int32_t v237_i1 = 0; v237_i1 < 1; ++v237_i1) {
              int32_t v239_a = v237_i1 * 2;
              int32_t v252_a = v11_lead + ((v237_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v238_i2 = 0; v238_i2 < 6; ++v238_i2) {
                float v243_data = r2[(v239_a + (v238_i2 * 2))];
                int32_t v253_a = v252_a + (v238_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v253_a], v243_data);
              }
            }
          }
          if (v11_lead < 3) {
            int32_t v266_lead = v11_lead + 32_i32;
            #pragma unroll
            for (int32_t v255_i1 = 0; v255_i1 < 1; ++v255_i1) {
              int32_t v259_a = 1 + (v255_i1 * 2);
              int32_t v270_a = v266_lead + ((v255_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v256_i2 = 0; v256_i2 < 6; ++v256_i2) {
                float v261_data = r2[(v259_a + (v256_i2 * 2))];
                int32_t v271_a = v270_a + (v256_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v271_a], v261_data);
              }
            }
          }
        }
      }
    }
  }
}

