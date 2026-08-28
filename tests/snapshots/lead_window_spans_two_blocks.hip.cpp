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
          alignas(16) float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v7_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 2; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 32;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 13; ++v9_i1) {
              int32_t v15_a = v9_i1 * 64;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __builtin_nontemporal_load(&glb_m0[(v21_lead + v15_a)]);
              int32_t v26_a = v8_i0 + (v9_i1 * 2);
              r0[v26_a] = v24_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          alignas(16) float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v28_data = r0[0];
          float v29_data = r0[1];
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
          float v183_acc{};
          float v184_acc{};
          float v185_acc{};
          float v186_lin = glb_m1[0 + threadIdx.x * 1];
          float v187_bc = tensorforge::broadcast<32, 16, 0>(v186_lin);
          tensorforge::fmacdpp16<0>(v30_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<0>(v31_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<1>(v32_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<1>(v33_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<2>(v34_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<2>(v35_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<3>(v36_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<3>(v37_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<4>(v38_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<4>(v39_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<5>(v40_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<5>(v41_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<6>(v42_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<6>(v43_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<7>(v44_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<7>(v45_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<8>(v46_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<8>(v47_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<9>(v48_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<9>(v49_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<10>(v50_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<10>(v51_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<11>(v52_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<11>(v53_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<12>(v54_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<12>(v55_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<13>(v56_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<13>(v57_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<14>(v58_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<14>(v59_acc, v187_bc, v29_data);
          tensorforge::fmacdpp16<15>(v60_acc, v187_bc, v28_data);
          tensorforge::fmacdpp16<15>(v61_acc, v187_bc, v29_data);
          float v188_bc = tensorforge::broadcast<32, 16, 1>(v186_lin);
          tensorforge::fmacdpp16<0>(v62_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<0>(v63_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<1>(v64_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<1>(v65_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<2>(v66_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<2>(v67_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<3>(v68_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<3>(v69_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<4>(v70_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<4>(v71_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<5>(v72_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<5>(v73_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<6>(v74_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<6>(v75_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<7>(v76_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<7>(v77_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<8>(v78_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<8>(v79_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<9>(v80_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<9>(v81_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<10>(v82_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<10>(v83_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<11>(v84_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<11>(v85_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<12>(v86_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<12>(v87_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<13>(v88_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<13>(v89_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<14>(v90_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<14>(v91_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<15>(v92_acc, v188_bc, v28_data);
          tensorforge::fmacdpp16<15>(v93_acc, v188_bc, v29_data);
          float v189_lin = glb_m1[32 + threadIdx.x * 1];
          float v190_bc = tensorforge::broadcast<32, 16, 0>(v189_lin);
          tensorforge::fmacdpp16<0>(v94_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<0>(v95_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<1>(v96_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<1>(v97_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<2>(v98_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<2>(v99_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<3>(v100_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<3>(v101_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<4>(v102_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<4>(v103_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<5>(v104_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<5>(v105_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<6>(v106_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<6>(v107_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<7>(v108_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<7>(v109_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<8>(v110_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<8>(v111_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<9>(v112_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<9>(v113_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<10>(v114_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<10>(v115_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<11>(v116_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<11>(v117_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<12>(v118_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<12>(v119_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<13>(v120_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<13>(v121_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<14>(v122_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<14>(v123_acc, v190_bc, v29_data);
          tensorforge::fmacdpp16<15>(v124_acc, v190_bc, v28_data);
          tensorforge::fmacdpp16<15>(v125_acc, v190_bc, v29_data);
          float v191_bc = tensorforge::broadcast<32, 16, 1>(v189_lin);
          tensorforge::fmacdpp16<0>(v126_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<0>(v127_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<1>(v128_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<1>(v129_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<2>(v130_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<2>(v131_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<3>(v132_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<3>(v133_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<4>(v134_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<4>(v135_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<5>(v136_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<5>(v137_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<6>(v138_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<6>(v139_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<7>(v140_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<7>(v141_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<8>(v142_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<8>(v143_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<9>(v144_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<9>(v145_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<10>(v146_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<10>(v147_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<11>(v148_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<11>(v149_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<12>(v150_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<12>(v151_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<13>(v152_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<13>(v153_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<14>(v154_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<14>(v155_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<15>(v156_acc, v191_bc, v28_data);
          tensorforge::fmacdpp16<15>(v157_acc, v191_bc, v29_data);
          float v192_lin = glb_m1[64 + threadIdx.x * 1];
          float v193_bc = tensorforge::broadcast<32, 16, 0>(v192_lin);
          tensorforge::fmacdpp16<0>(v158_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<0>(v159_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<1>(v160_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<1>(v161_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<2>(v162_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<2>(v163_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<3>(v164_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<3>(v165_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<4>(v166_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<4>(v167_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<5>(v168_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<5>(v169_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<6>(v170_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<6>(v171_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<7>(v172_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<7>(v173_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<8>(v174_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<8>(v175_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<9>(v176_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<9>(v177_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<10>(v178_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<10>(v179_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<11>(v180_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<11>(v181_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<12>(v182_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<12>(v183_acc, v193_bc, v29_data);
          tensorforge::fmacdpp16<13>(v184_acc, v193_bc, v28_data);
          tensorforge::fmacdpp16<13>(v185_acc, v193_bc, v29_data);
          r1[0] = v30_acc;
          r1[1] = v31_acc;
          r1[2] = v32_acc;
          r1[3] = v33_acc;
          r1[4] = v34_acc;
          r1[5] = v35_acc;
          r1[6] = v36_acc;
          r1[7] = v37_acc;
          r1[8] = v38_acc;
          r1[9] = v39_acc;
          r1[10] = v40_acc;
          r1[11] = v41_acc;
          r1[12] = v42_acc;
          r1[13] = v43_acc;
          r1[14] = v44_acc;
          r1[15] = v45_acc;
          r1[16] = v46_acc;
          r1[17] = v47_acc;
          r1[18] = v48_acc;
          r1[19] = v49_acc;
          r1[20] = v50_acc;
          r1[21] = v51_acc;
          r1[22] = v52_acc;
          r1[23] = v53_acc;
          r1[24] = v54_acc;
          r1[25] = v55_acc;
          r1[26] = v56_acc;
          r1[27] = v57_acc;
          r1[28] = v58_acc;
          r1[29] = v59_acc;
          r1[30] = v60_acc;
          r1[31] = v61_acc;
          r1[32] = v62_acc;
          r1[33] = v63_acc;
          r1[34] = v64_acc;
          r1[35] = v65_acc;
          r1[36] = v66_acc;
          r1[37] = v67_acc;
          r1[38] = v68_acc;
          r1[39] = v69_acc;
          r1[40] = v70_acc;
          r1[41] = v71_acc;
          r1[42] = v72_acc;
          r1[43] = v73_acc;
          r1[44] = v74_acc;
          r1[45] = v75_acc;
          r1[46] = v76_acc;
          r1[47] = v77_acc;
          r1[48] = v78_acc;
          r1[49] = v79_acc;
          r1[50] = v80_acc;
          r1[51] = v81_acc;
          r1[52] = v82_acc;
          r1[53] = v83_acc;
          r1[54] = v84_acc;
          r1[55] = v85_acc;
          r1[56] = v86_acc;
          r1[57] = v87_acc;
          r1[58] = v88_acc;
          r1[59] = v89_acc;
          r1[60] = v90_acc;
          r1[61] = v91_acc;
          r1[62] = v92_acc;
          r1[63] = v93_acc;
          r1[64] = v94_acc;
          r1[65] = v95_acc;
          r1[66] = v96_acc;
          r1[67] = v97_acc;
          r1[68] = v98_acc;
          r1[69] = v99_acc;
          r1[70] = v100_acc;
          r1[71] = v101_acc;
          r1[72] = v102_acc;
          r1[73] = v103_acc;
          r1[74] = v104_acc;
          r1[75] = v105_acc;
          r1[76] = v106_acc;
          r1[77] = v107_acc;
          r1[78] = v108_acc;
          r1[79] = v109_acc;
          r1[80] = v110_acc;
          r1[81] = v111_acc;
          r1[82] = v112_acc;
          r1[83] = v113_acc;
          r1[84] = v114_acc;
          r1[85] = v115_acc;
          r1[86] = v116_acc;
          r1[87] = v117_acc;
          r1[88] = v118_acc;
          r1[89] = v119_acc;
          r1[90] = v120_acc;
          r1[91] = v121_acc;
          r1[92] = v122_acc;
          r1[93] = v123_acc;
          r1[94] = v124_acc;
          r1[95] = v125_acc;
          r1[96] = v126_acc;
          r1[97] = v127_acc;
          r1[98] = v128_acc;
          r1[99] = v129_acc;
          r1[100] = v130_acc;
          r1[101] = v131_acc;
          r1[102] = v132_acc;
          r1[103] = v133_acc;
          r1[104] = v134_acc;
          r1[105] = v135_acc;
          r1[106] = v136_acc;
          r1[107] = v137_acc;
          r1[108] = v138_acc;
          r1[109] = v139_acc;
          r1[110] = v140_acc;
          r1[111] = v141_acc;
          r1[112] = v142_acc;
          r1[113] = v143_acc;
          r1[114] = v144_acc;
          r1[115] = v145_acc;
          r1[116] = v146_acc;
          r1[117] = v147_acc;
          r1[118] = v148_acc;
          r1[119] = v149_acc;
          r1[120] = v150_acc;
          r1[121] = v151_acc;
          r1[122] = v152_acc;
          r1[123] = v153_acc;
          r1[124] = v154_acc;
          r1[125] = v155_acc;
          r1[126] = v156_acc;
          r1[127] = v157_acc;
          r1[128] = v158_acc;
          r1[129] = v159_acc;
          r1[130] = v160_acc;
          r1[131] = v161_acc;
          r1[132] = v162_acc;
          r1[133] = v163_acc;
          r1[134] = v164_acc;
          r1[135] = v165_acc;
          r1[136] = v166_acc;
          r1[137] = v167_acc;
          r1[138] = v168_acc;
          r1[139] = v169_acc;
          r1[140] = v170_acc;
          r1[141] = v171_acc;
          r1[142] = v172_acc;
          r1[143] = v173_acc;
          r1[144] = v174_acc;
          r1[145] = v175_acc;
          r1[146] = v176_acc;
          r1[147] = v177_acc;
          r1[148] = v178_acc;
          r1[149] = v179_acc;
          r1[150] = v180_acc;
          r1[151] = v181_acc;
          r1[152] = v182_acc;
          r1[153] = v183_acc;
          r1[154] = v184_acc;
          r1[155] = v185_acc;
          alignas(16) float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          if (v7_lead >= 20) {
            float v199_data = r1[24];
            float v200_data = r2[0];
            r2[0] = (v200_data + v199_data);
            float v202_data = r1[50];
            float v203_data = r2[2];
            r2[2] = (v203_data + v202_data);
            float v205_data = r1[76];
            float v206_data = r2[4];
            r2[4] = (v206_data + v205_data);
            float v208_data = r1[102];
            float v209_data = r2[6];
            r2[6] = (v209_data + v208_data);
            float v211_data = r1[128];
            float v212_data = r2[8];
            r2[8] = (v212_data + v211_data);
            float v214_data = r1[154];
            float v215_data = r2[10];
            r2[10] = (v215_data + v214_data);
          }
          if (v7_lead < 3) {
            float v218_data = r1[25];
            float v219_data = r2[1];
            r2[1] = (v219_data + v218_data);
            float v221_data = r1[51];
            float v222_data = r2[3];
            r2[3] = (v222_data + v221_data);
            float v224_data = r1[77];
            float v225_data = r2[5];
            r2[5] = (v225_data + v224_data);
            float v227_data = r1[103];
            float v228_data = r2[7];
            r2[7] = (v228_data + v227_data);
            float v230_data = r1[129];
            float v231_data = r2[9];
            r2[9] = (v231_data + v230_data);
            float v233_data = r1[155];
            float v234_data = r2[11];
            r2[11] = (v234_data + v233_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v7_lead >= 20) {
            #pragma unroll
            for (int32_t v240_i1 = 0; v240_i1 < 1; ++v240_i1) {
              int32_t v242_a = v240_i1 * 2;
              int32_t v259_a = v7_lead + ((v240_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v241_i2 = 0; v241_i2 < 6; ++v241_i2) {
                int32_t v243_a = v241_i2 * 2;
                int32_t v245_a = v242_a + v243_a;
                float v250_data = r2[(v242_a + v243_a)];
                int32_t v260_a = v259_a + (v241_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v260_a], v250_data);
              }
            }
          }
          if (v7_lead < 3) {
            int32_t v277_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v262_i1 = 0; v262_i1 < 1; ++v262_i1) {
              int32_t v266_a = 1 + (v262_i1 * 2);
              int32_t v281_a = v277_lead + ((v262_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v263_i2 = 0; v263_i2 < 6; ++v263_i2) {
                int32_t v265_a = v263_i2 * 2;
                int32_t v267_a = v266_a + v265_a;
                float v272_data = r2[(v266_a + v265_a)];
                int32_t v282_a = v281_a + (v263_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v282_a], v272_data);
              }
            }
          }
        }
      }
    }
  }
}

