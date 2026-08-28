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
          int32_t v8_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v9_i0 = 0; v9_i0 < 2; ++v9_i0) {
            int32_t v14_lead = v9_i0 * 32;
            int32_t v15_lead = v8_lead + v14_lead;
            int32_t v22_lead = v8_lead + v14_lead;
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 13; ++v10_i1) {
              int32_t v16_a = v10_i1 * 64;
              int32_t v17_a = v15_lead + v16_a;
              float v25_data = __builtin_nontemporal_load(&glb_m0[(v22_lead + v16_a)]);
              r0[(v9_i0 + (v10_i1 * 2))] = v25_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v29_data = r0[0];
          float v30_data = r0[1];
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
          float v186_acc{};
          float v187_lin = glb_m1[0 + threadIdx.x * 1];
          float v188_bc = tensorforge::broadcast<32, 16, 0>(v187_lin);
          tensorforge::fmacdpp16<0>(v31_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<0>(v32_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<1>(v33_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<1>(v34_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<2>(v35_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<2>(v36_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<3>(v37_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<3>(v38_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<4>(v39_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<4>(v40_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<5>(v41_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<5>(v42_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<6>(v43_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<6>(v44_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<7>(v45_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<7>(v46_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<8>(v47_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<8>(v48_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<9>(v49_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<9>(v50_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<10>(v51_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<10>(v52_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<11>(v53_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<11>(v54_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<12>(v55_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<12>(v56_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<13>(v57_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<13>(v58_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<14>(v59_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<14>(v60_acc, v188_bc, v30_data);
          tensorforge::fmacdpp16<15>(v61_acc, v188_bc, v29_data);
          tensorforge::fmacdpp16<15>(v62_acc, v188_bc, v30_data);
          float v189_bc = tensorforge::broadcast<32, 16, 1>(v187_lin);
          tensorforge::fmacdpp16<0>(v63_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<0>(v64_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<1>(v65_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<1>(v66_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<2>(v67_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<2>(v68_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<3>(v69_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<3>(v70_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<4>(v71_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<4>(v72_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<5>(v73_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<5>(v74_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<6>(v75_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<6>(v76_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<7>(v77_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<7>(v78_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<8>(v79_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<8>(v80_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<9>(v81_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<9>(v82_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<10>(v83_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<10>(v84_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<11>(v85_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<11>(v86_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<12>(v87_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<12>(v88_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<13>(v89_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<13>(v90_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<14>(v91_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<14>(v92_acc, v189_bc, v30_data);
          tensorforge::fmacdpp16<15>(v93_acc, v189_bc, v29_data);
          tensorforge::fmacdpp16<15>(v94_acc, v189_bc, v30_data);
          float v190_lin = glb_m1[32 + threadIdx.x * 1];
          float v191_bc = tensorforge::broadcast<32, 16, 0>(v190_lin);
          tensorforge::fmacdpp16<0>(v95_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<0>(v96_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<1>(v97_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<1>(v98_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<2>(v99_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<2>(v100_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<3>(v101_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<3>(v102_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<4>(v103_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<4>(v104_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<5>(v105_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<5>(v106_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<6>(v107_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<6>(v108_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<7>(v109_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<7>(v110_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<8>(v111_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<8>(v112_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<9>(v113_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<9>(v114_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<10>(v115_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<10>(v116_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<11>(v117_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<11>(v118_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<12>(v119_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<12>(v120_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<13>(v121_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<13>(v122_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<14>(v123_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<14>(v124_acc, v191_bc, v30_data);
          tensorforge::fmacdpp16<15>(v125_acc, v191_bc, v29_data);
          tensorforge::fmacdpp16<15>(v126_acc, v191_bc, v30_data);
          float v192_bc = tensorforge::broadcast<32, 16, 1>(v190_lin);
          tensorforge::fmacdpp16<0>(v127_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<0>(v128_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<1>(v129_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<1>(v130_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<2>(v131_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<2>(v132_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<3>(v133_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<3>(v134_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<4>(v135_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<4>(v136_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<5>(v137_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<5>(v138_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<6>(v139_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<6>(v140_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<7>(v141_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<7>(v142_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<8>(v143_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<8>(v144_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<9>(v145_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<9>(v146_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<10>(v147_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<10>(v148_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<11>(v149_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<11>(v150_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<12>(v151_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<12>(v152_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<13>(v153_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<13>(v154_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<14>(v155_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<14>(v156_acc, v192_bc, v30_data);
          tensorforge::fmacdpp16<15>(v157_acc, v192_bc, v29_data);
          tensorforge::fmacdpp16<15>(v158_acc, v192_bc, v30_data);
          float v193_lin = glb_m1[64 + threadIdx.x * 1];
          float v194_bc = tensorforge::broadcast<32, 16, 0>(v193_lin);
          tensorforge::fmacdpp16<0>(v159_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<0>(v160_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<1>(v161_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<1>(v162_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<2>(v163_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<2>(v164_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<3>(v165_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<3>(v166_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<4>(v167_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<4>(v168_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<5>(v169_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<5>(v170_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<6>(v171_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<6>(v172_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<7>(v173_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<7>(v174_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<8>(v175_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<8>(v176_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<9>(v177_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<9>(v178_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<10>(v179_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<10>(v180_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<11>(v181_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<11>(v182_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<12>(v183_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<12>(v184_acc, v194_bc, v30_data);
          tensorforge::fmacdpp16<13>(v185_acc, v194_bc, v29_data);
          tensorforge::fmacdpp16<13>(v186_acc, v194_bc, v30_data);
          r1[0] = v31_acc;
          r1[1] = v32_acc;
          r1[2] = v33_acc;
          r1[3] = v34_acc;
          r1[4] = v35_acc;
          r1[5] = v36_acc;
          r1[6] = v37_acc;
          r1[7] = v38_acc;
          r1[8] = v39_acc;
          r1[9] = v40_acc;
          r1[10] = v41_acc;
          r1[11] = v42_acc;
          r1[12] = v43_acc;
          r1[13] = v44_acc;
          r1[14] = v45_acc;
          r1[15] = v46_acc;
          r1[16] = v47_acc;
          r1[17] = v48_acc;
          r1[18] = v49_acc;
          r1[19] = v50_acc;
          r1[20] = v51_acc;
          r1[21] = v52_acc;
          r1[22] = v53_acc;
          r1[23] = v54_acc;
          r1[24] = v55_acc;
          r1[25] = v56_acc;
          r1[26] = v57_acc;
          r1[27] = v58_acc;
          r1[28] = v59_acc;
          r1[29] = v60_acc;
          r1[30] = v61_acc;
          r1[31] = v62_acc;
          r1[32] = v63_acc;
          r1[33] = v64_acc;
          r1[34] = v65_acc;
          r1[35] = v66_acc;
          r1[36] = v67_acc;
          r1[37] = v68_acc;
          r1[38] = v69_acc;
          r1[39] = v70_acc;
          r1[40] = v71_acc;
          r1[41] = v72_acc;
          r1[42] = v73_acc;
          r1[43] = v74_acc;
          r1[44] = v75_acc;
          r1[45] = v76_acc;
          r1[46] = v77_acc;
          r1[47] = v78_acc;
          r1[48] = v79_acc;
          r1[49] = v80_acc;
          r1[50] = v81_acc;
          r1[51] = v82_acc;
          r1[52] = v83_acc;
          r1[53] = v84_acc;
          r1[54] = v85_acc;
          r1[55] = v86_acc;
          r1[56] = v87_acc;
          r1[57] = v88_acc;
          r1[58] = v89_acc;
          r1[59] = v90_acc;
          r1[60] = v91_acc;
          r1[61] = v92_acc;
          r1[62] = v93_acc;
          r1[63] = v94_acc;
          r1[64] = v95_acc;
          r1[65] = v96_acc;
          r1[66] = v97_acc;
          r1[67] = v98_acc;
          r1[68] = v99_acc;
          r1[69] = v100_acc;
          r1[70] = v101_acc;
          r1[71] = v102_acc;
          r1[72] = v103_acc;
          r1[73] = v104_acc;
          r1[74] = v105_acc;
          r1[75] = v106_acc;
          r1[76] = v107_acc;
          r1[77] = v108_acc;
          r1[78] = v109_acc;
          r1[79] = v110_acc;
          r1[80] = v111_acc;
          r1[81] = v112_acc;
          r1[82] = v113_acc;
          r1[83] = v114_acc;
          r1[84] = v115_acc;
          r1[85] = v116_acc;
          r1[86] = v117_acc;
          r1[87] = v118_acc;
          r1[88] = v119_acc;
          r1[89] = v120_acc;
          r1[90] = v121_acc;
          r1[91] = v122_acc;
          r1[92] = v123_acc;
          r1[93] = v124_acc;
          r1[94] = v125_acc;
          r1[95] = v126_acc;
          r1[96] = v127_acc;
          r1[97] = v128_acc;
          r1[98] = v129_acc;
          r1[99] = v130_acc;
          r1[100] = v131_acc;
          r1[101] = v132_acc;
          r1[102] = v133_acc;
          r1[103] = v134_acc;
          r1[104] = v135_acc;
          r1[105] = v136_acc;
          r1[106] = v137_acc;
          r1[107] = v138_acc;
          r1[108] = v139_acc;
          r1[109] = v140_acc;
          r1[110] = v141_acc;
          r1[111] = v142_acc;
          r1[112] = v143_acc;
          r1[113] = v144_acc;
          r1[114] = v145_acc;
          r1[115] = v146_acc;
          r1[116] = v147_acc;
          r1[117] = v148_acc;
          r1[118] = v149_acc;
          r1[119] = v150_acc;
          r1[120] = v151_acc;
          r1[121] = v152_acc;
          r1[122] = v153_acc;
          r1[123] = v154_acc;
          r1[124] = v155_acc;
          r1[125] = v156_acc;
          r1[126] = v157_acc;
          r1[127] = v158_acc;
          r1[128] = v159_acc;
          r1[129] = v160_acc;
          r1[130] = v161_acc;
          r1[131] = v162_acc;
          r1[132] = v163_acc;
          r1[133] = v164_acc;
          r1[134] = v165_acc;
          r1[135] = v166_acc;
          r1[136] = v167_acc;
          r1[137] = v168_acc;
          r1[138] = v169_acc;
          r1[139] = v170_acc;
          r1[140] = v171_acc;
          r1[141] = v172_acc;
          r1[142] = v173_acc;
          r1[143] = v174_acc;
          r1[144] = v175_acc;
          r1[145] = v176_acc;
          r1[146] = v177_acc;
          r1[147] = v178_acc;
          r1[148] = v179_acc;
          r1[149] = v180_acc;
          r1[150] = v181_acc;
          r1[151] = v182_acc;
          r1[152] = v183_acc;
          r1[153] = v184_acc;
          r1[154] = v185_acc;
          r1[155] = v186_acc;
          float r2[12]{};
          // r2 = +(r1) + None
          // [(20, 35), (0, 1), (0, 6)] []
          if (v8_lead >= 20) {
            float v200_data = r1[24];
            float v201_data = r2[0];
            r2[0] = (v201_data + v200_data);
            float v203_data = r1[50];
            float v204_data = r2[2];
            r2[2] = (v204_data + v203_data);
            float v206_data = r1[76];
            float v207_data = r2[4];
            r2[4] = (v207_data + v206_data);
            float v209_data = r1[102];
            float v210_data = r2[6];
            r2[6] = (v210_data + v209_data);
            float v212_data = r1[128];
            float v213_data = r2[8];
            r2[8] = (v213_data + v212_data);
            float v215_data = r1[154];
            float v216_data = r2[10];
            r2[10] = (v216_data + v215_data);
          }
          if (v8_lead < 3) {
            float v219_data = r1[25];
            float v220_data = r2[1];
            r2[1] = (v220_data + v219_data);
            float v222_data = r1[51];
            float v223_data = r2[3];
            r2[3] = (v223_data + v222_data);
            float v225_data = r1[77];
            float v226_data = r2[5];
            r2[5] = (v226_data + v225_data);
            float v228_data = r1[103];
            float v229_data = r2[7];
            r2[7] = (v229_data + v228_data);
            float v231_data = r1[129];
            float v232_data = r2[9];
            r2[9] = (v232_data + v231_data);
            float v234_data = r1[155];
            float v235_data = r2[11];
            r2[11] = (v235_data + v234_data);
          }
          // glb_m2 = store{r>g}(r2);
          if (v8_lead >= 20) {
            #pragma unroll
            for (int32_t v241_i1 = 0; v241_i1 < 1; ++v241_i1) {
              int32_t v243_a = v241_i1 * 2;
              int32_t v260_a = v8_lead + ((v241_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v242_i2 = 0; v242_i2 < 6; ++v242_i2) {
                int32_t v244_a = v242_i2 * 2;
                int32_t v246_a = v243_a + v244_a;
                float v251_data = r2[(v243_a + v244_a)];
                int32_t v261_a = v260_a + (v242_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v261_a], v251_data);
              }
            }
          }
          if (v8_lead < 3) {
            int32_t v278_lead = v8_lead + 32_i32;
            #pragma unroll
            for (int32_t v263_i1 = 0; v263_i1 < 1; ++v263_i1) {
              int32_t v267_a = 1 + (v263_i1 * 2);
              int32_t v282_a = v278_lead + ((v263_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v264_i2 = 0; v264_i2 < 6; ++v264_i2) {
                int32_t v266_a = v264_i2 * 2;
                int32_t v268_a = v267_a + v266_a;
                float v273_data = r2[(v267_a + v266_a)];
                int32_t v283_a = v282_a + (v264_i2 * 832);
                __builtin_amdgcn_global_atomic_fadd_f32(&glb_m2[v283_a], v273_data);
              }
            }
          }
        }
      }
    }
  }
}

