// === base name ===
kernel_d08f36e369

// === header ===
void launcher_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d08f36e369, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d08f36e369), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_d08f36e369, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d08f36e369(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 56×18(56×18) {0..56}×{0..18} strided
    // m1 56×18(56×18) {0..56}×{0..18} strided
    // m2 18×18(18×18) {0..18}×{0..18} strided
    // m0 56×18(56×18) {0..56}×{0..18} strided({0..56}×{0..18})[0, 1] = m1 56×18(56×18) {0..56}×{0..18} strided({0..56}×{0..18})[0, -1]×m2 18×18(18×18) {0..18}×{0..18} strided({0..18}×{0..18})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 1008 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1008 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 324 + 0 + m2_extraOffset];
          float r0[36]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 32);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 18; ++v12_i1) {
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v12_i1 * 56))]);
              r0[(v11_i0 + (v12_i1 * 2))] = v20_data;
            }
          }
          if (v10_lead < 24) {
            int32_t v29_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 18; ++v24_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m1[(v29_lead + (v24_i1 * 56))]);
              r0[(1 + (v24_i1 * 2))] = v32_data;
            }
          }
          float r1[18]{};
          // r1 = load{g>r}(glb_m2);
          if (v10_lead < 18) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 18; ++v40_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m2[(v10_lead + (v40_i1 * 18))]);
              r1[v40_i1] = v48_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
          float v51_data = r1[0];
          float v52_data = r1[1];
          float v53_data = r1[2];
          float v54_data = r1[3];
          float v55_tp{};
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          tensorforge::transpose4x4b32(v55_tp, v56_tp, v57_tp, v58_tp, v51_data, v52_data, v53_data, v54_data);
          tensorforge::VectorT<float, 4> v59_acc{};
          float v60_data = r0[0];
          float v61_data = r0[2];
          float v62_data = r0[4];
          float v63_data = r0[6];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v59_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v64_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v65_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 3, 0, 0);
          float v68_data = r0[8];
          float v69_data = r0[10];
          float v70_data = r0[12];
          float v71_data = r0[14];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v67_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 3, 1, 0);
          float v76_data = r0[16];
          float v77_data = r0[18];
          float v78_data = r0[20];
          float v79_data = r0[22];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v75_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v77_data, v80_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v81_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 3, 2, 0);
          float v84_data = r0[24];
          float v85_data = r0[26];
          float v86_data = r0[28];
          float v87_data = r0[30];
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v84_data, v83_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v85_data, v88_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v86_data, v89_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v87_data, v90_acc, 3, 3, 0);
          float v92_data = r0[32];
          float v93_data = r0[34];
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v92_data, v91_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v93_data, v96_acc, 3, 4, 0);
          r2[0] = (v97_acc[0]);
          r2[2] = (v97_acc[1]);
          r2[4] = (v97_acc[2]);
          r2[6] = (v97_acc[3]);
          tensorforge::VectorT<float, 4> v102_acc{};
          float v103_data = r0[1];
          float v104_data = r0[3];
          float v105_data = r0[5];
          float v106_data = r0[7];
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v103_data, v102_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v104_data, v107_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v105_data, v108_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v106_data, v109_acc, 3, 0, 0);
          float v111_data = r0[9];
          float v112_data = r0[11];
          float v113_data = r0[13];
          float v114_data = r0[15];
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v111_data, v110_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v112_data, v115_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v113_data, v116_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v114_data, v117_acc, 3, 1, 0);
          float v119_data = r0[17];
          float v120_data = r0[19];
          float v121_data = r0[21];
          float v122_data = r0[23];
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v119_data, v118_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v120_data, v123_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v121_data, v124_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v122_data, v125_acc, 3, 2, 0);
          float v127_data = r0[25];
          float v128_data = r0[27];
          float v129_data = r0[29];
          float v130_data = r0[31];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v127_data, v126_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v128_data, v131_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v129_data, v132_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v130_data, v133_acc, 3, 3, 0);
          float v135_data = r0[33];
          float v136_data = r0[35];
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v135_data, v134_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v136_data, v139_acc, 3, 4, 0);
          r2[1] = (v140_acc[0]);
          r2[3] = (v140_acc[1]);
          r2[5] = (v140_acc[2]);
          r2[7] = (v140_acc[3]);
          float v145_data = r1[4];
          float v146_data = r1[5];
          float v147_data = r1[6];
          float v148_data = r1[7];
          float v149_tp{};
          float v150_tp{};
          float v151_tp{};
          float v152_tp{};
          tensorforge::transpose4x4b32(v149_tp, v150_tp, v151_tp, v152_tp, v145_data, v146_data, v147_data, v148_data);
          tensorforge::VectorT<float, 4> v153_acc{};
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v60_data, v153_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v61_data, v158_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v62_data, v159_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v63_data, v160_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v68_data, v161_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v69_data, v166_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v70_data, v167_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v71_data, v168_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v76_data, v169_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v77_data, v174_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v78_data, v175_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v79_data, v176_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v84_data, v177_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v85_data, v182_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v86_data, v183_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v87_data, v184_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v92_data, v185_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v93_data, v190_acc, 3, 4, 0);
          r2[8] = (v191_acc[0]);
          r2[10] = (v191_acc[1]);
          r2[12] = (v191_acc[2]);
          r2[14] = (v191_acc[3]);
          tensorforge::VectorT<float, 4> v196_acc{};
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v103_data, v196_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v104_data, v201_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v105_data, v202_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v106_data, v203_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v111_data, v204_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v112_data, v209_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v113_data, v210_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v114_data, v211_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v119_data, v212_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v120_data, v217_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v121_data, v218_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v122_data, v219_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v127_data, v220_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v128_data, v225_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v129_data, v226_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v130_data, v227_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v135_data, v228_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v136_data, v233_acc, 3, 4, 0);
          r2[9] = (v234_acc[0]);
          r2[11] = (v234_acc[1]);
          r2[13] = (v234_acc[2]);
          r2[15] = (v234_acc[3]);
          float v239_data = r1[8];
          float v240_data = r1[9];
          float v241_data = r1[10];
          float v242_data = r1[11];
          float v243_tp{};
          float v244_tp{};
          float v245_tp{};
          float v246_tp{};
          tensorforge::transpose4x4b32(v243_tp, v244_tp, v245_tp, v246_tp, v239_data, v240_data, v241_data, v242_data);
          tensorforge::VectorT<float, 4> v247_acc{};
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v60_data, v247_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v61_data, v252_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v62_data, v253_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v63_data, v254_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v68_data, v255_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v69_data, v260_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v70_data, v261_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v71_data, v262_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v76_data, v263_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v77_data, v268_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v78_data, v269_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v79_data, v270_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v84_data, v271_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v85_data, v276_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v86_data, v277_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v87_data, v278_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v92_data, v279_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v93_data, v284_acc, 3, 4, 0);
          r2[16] = (v285_acc[0]);
          r2[18] = (v285_acc[1]);
          r2[20] = (v285_acc[2]);
          r2[22] = (v285_acc[3]);
          tensorforge::VectorT<float, 4> v290_acc{};
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v103_data, v290_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v104_data, v295_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v105_data, v296_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v106_data, v297_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v111_data, v298_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v112_data, v303_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v113_data, v304_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v114_data, v305_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v119_data, v306_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v120_data, v311_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v121_data, v312_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v122_data, v313_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v127_data, v314_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v128_data, v319_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v245_tp, v129_data, v320_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v130_data, v321_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v135_data, v322_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v244_tp, v136_data, v327_acc, 3, 4, 0);
          r2[17] = (v328_acc[0]);
          r2[19] = (v328_acc[1]);
          r2[21] = (v328_acc[2]);
          r2[23] = (v328_acc[3]);
          float v333_data = r1[12];
          float v334_data = r1[13];
          float v335_data = r1[14];
          float v336_data = r1[15];
          float v337_tp{};
          float v338_tp{};
          float v339_tp{};
          float v340_tp{};
          tensorforge::transpose4x4b32(v337_tp, v338_tp, v339_tp, v340_tp, v333_data, v334_data, v335_data, v336_data);
          tensorforge::VectorT<float, 4> v341_acc{};
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v60_data, v341_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v61_data, v346_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v62_data, v347_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v63_data, v348_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v68_data, v349_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v69_data, v354_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v70_data, v355_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v71_data, v356_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v76_data, v357_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v363_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v77_data, v362_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v78_data, v363_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v79_data, v364_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v84_data, v365_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v85_data, v370_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v86_data, v371_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v87_data, v372_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v92_data, v373_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v93_data, v378_acc, 3, 4, 0);
          r2[24] = (v379_acc[0]);
          r2[26] = (v379_acc[1]);
          r2[28] = (v379_acc[2]);
          r2[30] = (v379_acc[3]);
          tensorforge::VectorT<float, 4> v384_acc{};
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v103_data, v384_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v104_data, v389_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v391_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v105_data, v390_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v106_data, v391_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v111_data, v392_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v112_data, v397_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v399_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v113_data, v398_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v114_data, v399_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v119_data, v400_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v120_data, v405_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v121_data, v406_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v122_data, v407_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v127_data, v408_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v128_data, v413_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v339_tp, v129_data, v414_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v340_tp, v130_data, v415_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v135_data, v416_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v338_tp, v136_data, v421_acc, 3, 4, 0);
          r2[25] = (v422_acc[0]);
          r2[27] = (v422_acc[1]);
          r2[29] = (v422_acc[2]);
          r2[31] = (v422_acc[3]);
          float v427_data = r1[16];
          float v428_data = r1[17];
          float v431_tp{};
          float v432_tp{};
          float v433_tp{};
          float v434_tp{};
          tensorforge::transpose4x4b32(v431_tp, v432_tp, v433_tp, v434_tp, v427_data, v428_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v435_acc{};
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v60_data, v435_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v61_data, v440_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v62_data, v441_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v63_data, v442_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v68_data, v443_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v69_data, v448_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v70_data, v449_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v71_data, v450_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v76_data, v451_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v77_data, v456_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v78_data, v457_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v79_data, v458_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v84_data, v459_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v465_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v85_data, v464_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v86_data, v465_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v87_data, v466_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v92_data, v467_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v473_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v93_data, v472_acc, 3, 4, 0);
          r2[32] = (v473_acc[0]);
          r2[34] = (v473_acc[1]);
          tensorforge::VectorT<float, 4> v476_acc{};
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v103_data, v476_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v104_data, v481_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v105_data, v482_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v106_data, v483_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v111_data, v484_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v112_data, v489_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v113_data, v490_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v114_data, v491_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v119_data, v492_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v120_data, v497_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v121_data, v498_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v122_data, v499_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v127_data, v500_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v128_data, v505_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v433_tp, v129_data, v506_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v434_tp, v130_data, v507_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v135_data, v508_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v432_tp, v136_data, v513_acc, 3, 4, 0);
          r2[33] = (v514_acc[0]);
          r2[35] = (v514_acc[1]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v520_i0 = 0; v520_i0 < 1; ++v520_i0) {
            int32_t v529_lead = v10_lead + (v520_i0 * 32);
            #pragma unroll
            for (int32_t v521_i1 = 0; v521_i1 < 18; ++v521_i1) {
              float v524_data = r2[(v520_i0 + (v521_i1 * 2))];
              glb_m0[(v529_lead + (v521_i1 * 56))] = v524_data;
            }
          }
          if (v10_lead < 24) {
            int32_t v541_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v533_i1 = 0; v533_i1 < 18; ++v533_i1) {
              float v536_data = r2[(1 + (v533_i1 * 2))];
              glb_m0[(v541_lead + (v533_i1 * 56))] = v536_data;
            }
          }
        }
      }
    }
  }
}

