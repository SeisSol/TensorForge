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
            int32_t v17_lead = v12_i0 * 32;
            int32_t v18_lead = v11_lead + v17_lead;
            int32_t v25_lead = v11_lead + v17_lead;
            #pragma unroll
            for (int32_t v13_i1 = 0; v13_i1 < 13; ++v13_i1) {
              int32_t v19_a = v13_i1 * 64;
              int32_t v20_a = v18_lead + v19_a;
              float v28_data = __builtin_nontemporal_load(&glb_m0[(v25_lead + v19_a)]);
              r0[(v12_i0 + (v13_i1 * 2))] = v28_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v32_data = r0[0];
          float v33_data = r0[1];
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
          float v183_acc{};
          float v184_acc{};
          float v185_acc{};
          float v186_acc{};
          float v187_acc{};
          float v188_acc{};
          float v189_acc{};
          float v190_lin = glb_m1[0 + threadIdx.x * 1];
          float v191_bc = tensorforge::broadcast<32, 16, 0>(v190_lin);
          tensorforge::fmacdpp16<0>(v34_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<0>(v35_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<1>(v36_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<1>(v37_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<2>(v38_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<2>(v39_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<3>(v40_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<3>(v41_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<4>(v42_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<4>(v43_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<5>(v44_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<5>(v45_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<6>(v46_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<6>(v47_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<7>(v48_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<7>(v49_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<8>(v50_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<8>(v51_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<9>(v52_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<9>(v53_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<10>(v54_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<10>(v55_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<11>(v56_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<11>(v57_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<12>(v58_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<12>(v59_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<13>(v60_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<13>(v61_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<14>(v62_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<14>(v63_acc, v191_bc, v33_data);
          tensorforge::fmacdpp16<15>(v64_acc, v191_bc, v32_data);
          tensorforge::fmacdpp16<15>(v65_acc, v191_bc, v33_data);
          float v192_bc = tensorforge::broadcast<32, 16, 1>(v190_lin);
          tensorforge::fmacdpp16<0>(v66_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<0>(v67_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<1>(v68_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<1>(v69_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<2>(v70_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<2>(v71_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<3>(v72_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<3>(v73_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<4>(v74_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<4>(v75_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<5>(v76_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<5>(v77_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<6>(v78_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<6>(v79_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<7>(v80_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<7>(v81_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<8>(v82_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<8>(v83_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<9>(v84_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<9>(v85_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<10>(v86_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<10>(v87_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<11>(v88_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<11>(v89_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<12>(v90_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<12>(v91_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<13>(v92_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<13>(v93_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<14>(v94_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<14>(v95_acc, v192_bc, v33_data);
          tensorforge::fmacdpp16<15>(v96_acc, v192_bc, v32_data);
          tensorforge::fmacdpp16<15>(v97_acc, v192_bc, v33_data);
          float v193_lin = glb_m1[32 + threadIdx.x * 1];
          float v194_bc = tensorforge::broadcast<32, 16, 0>(v193_lin);
          tensorforge::fmacdpp16<0>(v98_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<0>(v99_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<1>(v100_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<1>(v101_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<2>(v102_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<2>(v103_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<3>(v104_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<3>(v105_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<4>(v106_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<4>(v107_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<5>(v108_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<5>(v109_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<6>(v110_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<6>(v111_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<7>(v112_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<7>(v113_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<8>(v114_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<8>(v115_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<9>(v116_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<9>(v117_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<10>(v118_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<10>(v119_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<11>(v120_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<11>(v121_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<12>(v122_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<12>(v123_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<13>(v124_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<13>(v125_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<14>(v126_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<14>(v127_acc, v194_bc, v33_data);
          tensorforge::fmacdpp16<15>(v128_acc, v194_bc, v32_data);
          tensorforge::fmacdpp16<15>(v129_acc, v194_bc, v33_data);
          float v195_bc = tensorforge::broadcast<32, 16, 1>(v193_lin);
          tensorforge::fmacdpp16<0>(v130_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<0>(v131_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<1>(v132_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<1>(v133_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<2>(v134_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<2>(v135_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<3>(v136_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<3>(v137_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<4>(v138_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<4>(v139_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<5>(v140_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<5>(v141_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<6>(v142_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<6>(v143_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<7>(v144_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<7>(v145_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<8>(v146_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<8>(v147_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<9>(v148_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<9>(v149_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<10>(v150_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<10>(v151_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<11>(v152_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<11>(v153_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<12>(v154_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<12>(v155_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<13>(v156_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<13>(v157_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<14>(v158_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<14>(v159_acc, v195_bc, v33_data);
          tensorforge::fmacdpp16<15>(v160_acc, v195_bc, v32_data);
          tensorforge::fmacdpp16<15>(v161_acc, v195_bc, v33_data);
          float v196_lin = glb_m1[64 + threadIdx.x * 1];
          float v197_bc = tensorforge::broadcast<32, 16, 0>(v196_lin);
          tensorforge::fmacdpp16<0>(v162_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<0>(v163_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<1>(v164_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<1>(v165_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<2>(v166_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<2>(v167_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<3>(v168_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<3>(v169_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<4>(v170_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<4>(v171_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<5>(v172_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<5>(v173_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<6>(v174_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<6>(v175_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<7>(v176_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<7>(v177_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<8>(v178_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<8>(v179_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<9>(v180_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<9>(v181_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<10>(v182_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<10>(v183_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<11>(v184_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<11>(v185_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<12>(v186_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<12>(v187_acc, v197_bc, v33_data);
          tensorforge::fmacdpp16<13>(v188_acc, v197_bc, v32_data);
          tensorforge::fmacdpp16<13>(v189_acc, v197_bc, v33_data);
          r1[0] = v34_acc;
          r1[1] = v35_acc;
          r1[2] = v36_acc;
          r1[3] = v37_acc;
          r1[4] = v38_acc;
          r1[5] = v39_acc;
          r1[6] = v40_acc;
          r1[7] = v41_acc;
          r1[8] = v42_acc;
          r1[9] = v43_acc;
          r1[10] = v44_acc;
          r1[11] = v45_acc;
          r1[12] = v46_acc;
          r1[13] = v47_acc;
          r1[14] = v48_acc;
          r1[15] = v49_acc;
          r1[16] = v50_acc;
          r1[17] = v51_acc;
          r1[18] = v52_acc;
          r1[19] = v53_acc;
          r1[20] = v54_acc;
          r1[21] = v55_acc;
          r1[22] = v56_acc;
          r1[23] = v57_acc;
          r1[24] = v58_acc;
          r1[25] = v59_acc;
          r1[26] = v60_acc;
          r1[27] = v61_acc;
          r1[28] = v62_acc;
          r1[29] = v63_acc;
          r1[30] = v64_acc;
          r1[31] = v65_acc;
          r1[32] = v66_acc;
          r1[33] = v67_acc;
          r1[34] = v68_acc;
          r1[35] = v69_acc;
          r1[36] = v70_acc;
          r1[37] = v71_acc;
          r1[38] = v72_acc;
          r1[39] = v73_acc;
          r1[40] = v74_acc;
          r1[41] = v75_acc;
          r1[42] = v76_acc;
          r1[43] = v77_acc;
          r1[44] = v78_acc;
          r1[45] = v79_acc;
          r1[46] = v80_acc;
          r1[47] = v81_acc;
          r1[48] = v82_acc;
          r1[49] = v83_acc;
          r1[50] = v84_acc;
          r1[51] = v85_acc;
          r1[52] = v86_acc;
          r1[53] = v87_acc;
          r1[54] = v88_acc;
          r1[55] = v89_acc;
          r1[56] = v90_acc;
          r1[57] = v91_acc;
          r1[58] = v92_acc;
          r1[59] = v93_acc;
          r1[60] = v94_acc;
          r1[61] = v95_acc;
          r1[62] = v96_acc;
          r1[63] = v97_acc;
          r1[64] = v98_acc;
          r1[65] = v99_acc;
          r1[66] = v100_acc;
          r1[67] = v101_acc;
          r1[68] = v102_acc;
          r1[69] = v103_acc;
          r1[70] = v104_acc;
          r1[71] = v105_acc;
          r1[72] = v106_acc;
          r1[73] = v107_acc;
          r1[74] = v108_acc;
          r1[75] = v109_acc;
          r1[76] = v110_acc;
          r1[77] = v111_acc;
          r1[78] = v112_acc;
          r1[79] = v113_acc;
          r1[80] = v114_acc;
          r1[81] = v115_acc;
          r1[82] = v116_acc;
          r1[83] = v117_acc;
          r1[84] = v118_acc;
          r1[85] = v119_acc;
          r1[86] = v120_acc;
          r1[87] = v121_acc;
          r1[88] = v122_acc;
          r1[89] = v123_acc;
          r1[90] = v124_acc;
          r1[91] = v125_acc;
          r1[92] = v126_acc;
          r1[93] = v127_acc;
          r1[94] = v128_acc;
          r1[95] = v129_acc;
          r1[96] = v130_acc;
          r1[97] = v131_acc;
          r1[98] = v132_acc;
          r1[99] = v133_acc;
          r1[100] = v134_acc;
          r1[101] = v135_acc;
          r1[102] = v136_acc;
          r1[103] = v137_acc;
          r1[104] = v138_acc;
          r1[105] = v139_acc;
          r1[106] = v140_acc;
          r1[107] = v141_acc;
          r1[108] = v142_acc;
          r1[109] = v143_acc;
          r1[110] = v144_acc;
          r1[111] = v145_acc;
          r1[112] = v146_acc;
          r1[113] = v147_acc;
          r1[114] = v148_acc;
          r1[115] = v149_acc;
          r1[116] = v150_acc;
          r1[117] = v151_acc;
          r1[118] = v152_acc;
          r1[119] = v153_acc;
          r1[120] = v154_acc;
          r1[121] = v155_acc;
          r1[122] = v156_acc;
          r1[123] = v157_acc;
          r1[124] = v158_acc;
          r1[125] = v159_acc;
          r1[126] = v160_acc;
          r1[127] = v161_acc;
          r1[128] = v162_acc;
          r1[129] = v163_acc;
          r1[130] = v164_acc;
          r1[131] = v165_acc;
          r1[132] = v166_acc;
          r1[133] = v167_acc;
          r1[134] = v168_acc;
          r1[135] = v169_acc;
          r1[136] = v170_acc;
          r1[137] = v171_acc;
          r1[138] = v172_acc;
          r1[139] = v173_acc;
          r1[140] = v174_acc;
          r1[141] = v175_acc;
          r1[142] = v176_acc;
          r1[143] = v177_acc;
          r1[144] = v178_acc;
          r1[145] = v179_acc;
          r1[146] = v180_acc;
          r1[147] = v181_acc;
          r1[148] = v182_acc;
          r1[149] = v183_acc;
          r1[150] = v184_acc;
          r1[151] = v185_acc;
          r1[152] = v186_acc;
          r1[153] = v187_acc;
          r1[154] = v188_acc;
          r1[155] = v189_acc;
          float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          if (v11_lead >= 20) {
            float v203_data = r1[24];
            float v204_data = r2[0];
            r2[0] = (v204_data + v203_data);
            float v206_data = r1[50];
            float v207_data = r2[2];
            r2[2] = (v207_data + v206_data);
            float v209_data = r1[76];
            float v210_data = r2[4];
            r2[4] = (v210_data + v209_data);
            float v212_data = r1[102];
            float v213_data = r2[6];
            r2[6] = (v213_data + v212_data);
            float v215_data = r1[128];
            float v216_data = r2[8];
            r2[8] = (v216_data + v215_data);
            float v218_data = r1[154];
            float v219_data = r2[10];
            r2[10] = (v219_data + v218_data);
          }
          if (v11_lead < 3) {
            float v222_data = r1[25];
            float v223_data = r2[1];
            r2[1] = (v223_data + v222_data);
            float v225_data = r1[51];
            float v226_data = r2[3];
            r2[3] = (v226_data + v225_data);
            float v228_data = r1[77];
            float v229_data = r2[5];
            r2[5] = (v229_data + v228_data);
            float v231_data = r1[103];
            float v232_data = r2[7];
            r2[7] = (v232_data + v231_data);
            float v234_data = r1[129];
            float v235_data = r2[9];
            r2[9] = (v235_data + v234_data);
            float v237_data = r1[155];
            float v238_data = r2[11];
            r2[11] = (v238_data + v237_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v11_lead >= 20) {
            #pragma unroll
            for (int32_t v244_i1 = 0; v244_i1 < 1; ++v244_i1) {
              int32_t v246_a = v244_i1 * 2;
              int32_t v263_a = v11_lead + ((v244_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v245_i2 = 0; v245_i2 < 6; ++v245_i2) {
                int32_t v247_a = v245_i2 * 2;
                int32_t v249_a = v246_a + v247_a;
                float v254_data = r2[(v246_a + v247_a)];
                int32_t v264_a = v263_a + (v245_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v264_a], v254_data);
              }
            }
          }
          if (v11_lead < 3) {
            int32_t v281_lead = v11_lead + 32_i32;
            #pragma unroll
            for (int32_t v266_i1 = 0; v266_i1 < 1; ++v266_i1) {
              int32_t v270_a = 1 + (v266_i1 * 2);
              int32_t v285_a = v281_lead + ((v266_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v267_i2 = 0; v267_i2 < 6; ++v267_i2) {
                int32_t v269_a = v267_i2 * 2;
                int32_t v271_a = v270_a + v269_a;
                float v276_data = r2[(v270_a + v269_a)];
                int32_t v286_a = v285_a + (v267_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v286_a], v276_data);
              }
            }
          }
        }
      }
    }
  }
}

