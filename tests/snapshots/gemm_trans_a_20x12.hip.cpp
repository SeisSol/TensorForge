// === base name ===
kernel_1dbad859f0c2b685

// === header ===
void launcher_kernel_1dbad859f0c2b685(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_1dbad859f0c2b685(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1dbad859f0c2b685, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_1dbad859f0c2b685), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_1dbad859f0c2b685, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_1dbad859f0c2b685(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 12×16(12×16) {0..12}×{0..16} strided
    // m1 20×12(20×12) {0..20}×{0..12} strided
    // m2 20×16(20×16) {0..20}×{0..16} strided
    // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 20×12(20×12) {0..20}×{0..12} strided({0..20}×{0..12})[-1, 0]×m2 20×16(20×16) {0..20}×{0..16} strided({0..20}×{0..16})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
          float r0[20]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v14_lead = threadIdx.x % 16;
          bool v15_g = v14_lead < 12;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 20; ++v11_i0) {
            if (v15_g) {
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v11_i0 + (v14_lead * 20))]);
              r0[v11_i0] = v23_data;
            }
          }
          float r1[32]{};
          // r1 = load{g>r}(glb_m2);
          int32_t v28_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
            int32_t v35_lead = v28_lead + (v29_i0 * 16);
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 16; ++v30_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m2[(v35_lead + (v30_i1 * 20))]);
              r1[(v29_i0 + (v30_i1 * 2))] = v38_data;
            }
          }
          if (v28_lead < 4) {
            int32_t v47_lead = v28_lead + 16_i32;
            #pragma unroll
            for (int32_t v42_i1 = 0; v42_i1 < 16; ++v42_i1) {
              float v50_data = __builtin_nontemporal_load(&glb_m2[(v47_lead + (v42_i1 * 20))]);
              r1[(1 + (v42_i1 * 2))] = v50_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v54_data = r1[0];
          float v55_data = r1[2];
          float v56_data = r1[4];
          float v57_data = r1[6];
          float v58_tp{};
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          tensorforge::transpose4x4b32(v58_tp, v59_tp, v60_tp, v61_tp, v54_data, v55_data, v56_data, v57_data);
          float v62_data = r1[1];
          float v63_data = r1[3];
          float v64_data = r1[5];
          float v65_data = r1[7];
          float v66_tp{};
          float v67_tp{};
          float v68_tp{};
          float v69_tp{};
          tensorforge::transpose4x4b32(v66_tp, v67_tp, v68_tp, v69_tp, v62_data, v63_data, v64_data, v65_data);
          tensorforge::VectorT<float, 4> v70_acc{};
          float v71_data = r0[0];
          float v72_data = r0[1];
          float v73_data = r0[2];
          float v74_data = r0[3];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v72_data, v75_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v73_data, v76_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v74_data, v77_acc, 2, 0, 0);
          float v79_data = r0[4];
          float v80_data = r0[5];
          float v81_data = r0[6];
          float v82_data = r0[7];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v78_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v80_data, v83_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v81_data, v84_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v82_data, v85_acc, 2, 1, 0);
          float v87_data = r0[8];
          float v88_data = r0[9];
          float v89_data = r0[10];
          float v90_data = r0[11];
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v87_data, v86_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v88_data, v91_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v89_data, v92_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v90_data, v93_acc, 2, 2, 0);
          float v95_data = r0[12];
          float v96_data = r0[13];
          float v97_data = r0[14];
          float v98_data = r0[15];
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v95_data, v94_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v96_data, v99_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v97_data, v100_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v98_data, v101_acc, 2, 3, 0);
          float v103_data = r0[16];
          float v104_data = r0[17];
          float v105_data = r0[18];
          float v106_data = r0[19];
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v103_data, v102_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v104_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v105_data, v108_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v106_data, v109_acc, 2, 0, 0);
          r2[0] = (v110_acc[0]);
          r2[1] = (v110_acc[1]);
          r2[2] = (v110_acc[2]);
          r2[3] = (v110_acc[3]);
          float v115_data = r1[8];
          float v116_data = r1[10];
          float v117_data = r1[12];
          float v118_data = r1[14];
          float v119_tp{};
          float v120_tp{};
          float v121_tp{};
          float v122_tp{};
          tensorforge::transpose4x4b32(v119_tp, v120_tp, v121_tp, v122_tp, v115_data, v116_data, v117_data, v118_data);
          float v123_data = r1[9];
          float v124_data = r1[11];
          float v125_data = r1[13];
          float v126_data = r1[15];
          float v127_tp{};
          float v128_tp{};
          float v129_tp{};
          float v130_tp{};
          tensorforge::transpose4x4b32(v127_tp, v128_tp, v129_tp, v130_tp, v123_data, v124_data, v125_data, v126_data);
          tensorforge::VectorT<float, 4> v131_acc{};
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v71_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v72_data, v136_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v73_data, v137_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v74_data, v138_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v79_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v80_data, v144_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v81_data, v145_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v82_data, v146_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v87_data, v147_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v88_data, v152_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v89_data, v153_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v90_data, v154_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v95_data, v155_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v96_data, v160_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v97_data, v161_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v98_data, v162_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v103_data, v163_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v128_tp, v104_data, v168_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v129_tp, v105_data, v169_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v106_data, v170_acc, 2, 0, 0);
          r2[4] = (v171_acc[0]);
          r2[5] = (v171_acc[1]);
          r2[6] = (v171_acc[2]);
          r2[7] = (v171_acc[3]);
          float v176_data = r1[16];
          float v177_data = r1[18];
          float v178_data = r1[20];
          float v179_data = r1[22];
          float v180_tp{};
          float v181_tp{};
          float v182_tp{};
          float v183_tp{};
          tensorforge::transpose4x4b32(v180_tp, v181_tp, v182_tp, v183_tp, v176_data, v177_data, v178_data, v179_data);
          float v184_data = r1[17];
          float v185_data = r1[19];
          float v186_data = r1[21];
          float v187_data = r1[23];
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          float v191_tp{};
          tensorforge::transpose4x4b32(v188_tp, v189_tp, v190_tp, v191_tp, v184_data, v185_data, v186_data, v187_data);
          tensorforge::VectorT<float, 4> v192_acc{};
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v71_data, v192_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v72_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v73_data, v198_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v74_data, v199_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v79_data, v200_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v80_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v81_data, v206_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v82_data, v207_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v87_data, v208_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v88_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v89_data, v214_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v90_data, v215_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v95_data, v216_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v96_data, v221_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v97_data, v222_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v98_data, v223_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v103_data, v224_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v104_data, v229_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v105_data, v230_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v106_data, v231_acc, 2, 0, 0);
          r2[8] = (v232_acc[0]);
          r2[9] = (v232_acc[1]);
          r2[10] = (v232_acc[2]);
          r2[11] = (v232_acc[3]);
          float v237_data = r1[24];
          float v238_data = r1[26];
          float v239_data = r1[28];
          float v240_data = r1[30];
          float v241_tp{};
          float v242_tp{};
          float v243_tp{};
          float v244_tp{};
          tensorforge::transpose4x4b32(v241_tp, v242_tp, v243_tp, v244_tp, v237_data, v238_data, v239_data, v240_data);
          float v245_data = r1[25];
          float v246_data = r1[27];
          float v247_data = r1[29];
          float v248_data = r1[31];
          float v249_tp{};
          float v250_tp{};
          float v251_tp{};
          float v252_tp{};
          tensorforge::transpose4x4b32(v249_tp, v250_tp, v251_tp, v252_tp, v245_data, v246_data, v247_data, v248_data);
          tensorforge::VectorT<float, 4> v253_acc{};
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v71_data, v253_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v72_data, v258_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v73_data, v259_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v74_data, v260_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v79_data, v261_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v80_data, v266_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v81_data, v267_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v82_data, v268_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v87_data, v269_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v88_data, v274_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v89_data, v275_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v90_data, v276_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v95_data, v277_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v96_data, v282_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v97_data, v283_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v98_data, v284_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v103_data, v285_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v250_tp, v104_data, v290_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v251_tp, v105_data, v291_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v252_tp, v106_data, v292_acc, 2, 0, 0);
          r2[12] = (v293_acc[0]);
          r2[13] = (v293_acc[1]);
          r2[14] = (v293_acc[2]);
          r2[15] = (v293_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v28_lead < 12) {
            #pragma unroll
            for (int32_t v302_i1 = 0; v302_i1 < 16; ++v302_i1) {
              float v304_data = r2[v302_i1];
              glb_m0[(v28_lead + (v302_i1 * 12))] = v304_data;
            }
          }
        }
      }
    }
  }
}

