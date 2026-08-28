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
          float v36_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v36_lin;
          float v37_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v37_lin;
          float v38_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v38_lin;
          float v39_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v39_lin;
          float v40_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v40_lin;
          float v41_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v41_lin;
          float v42_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v42_lin;
          float v43_lin = glb_m2[224 + threadIdx.x * 1];
          r1[7] = v43_lin;
          float v44_lin = glb_m2[256 + threadIdx.x * 1];
          r1[8] = v44_lin;
          float v45_lin = glb_m2[288 + threadIdx.x * 1];
          r1[9] = v45_lin;
          float v46_lin = glb_m2[320 + threadIdx.x * 1];
          r1[10] = v46_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[36]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 18)] [(0, 18)]
          float v48_data = r1[0];
          float v49_data = r1[1];
          float v50_data = r1[2];
          float v51_data = r1[3];
          float v52_tp{};
          float v53_tp{};
          float v54_tp{};
          float v55_tp{};
          tensorforge::transpose4x4b32(v52_tp, v53_tp, v54_tp, v55_tp, v48_data, v49_data, v50_data, v51_data);
          tensorforge::VectorT<float, 4> v56_acc{};
          float v57_data = r0[0];
          float v58_data = r0[2];
          float v59_data = r0[4];
          float v60_data = r0[6];
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v57_data, v56_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v58_data, v61_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v59_data, v62_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v63_acc, 3, 0, 0);
          float v65_data = r0[8];
          float v66_data = r0[10];
          float v67_data = r0[12];
          float v68_data = r0[14];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v64_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v69_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v67_data, v70_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v71_acc, 3, 1, 0);
          float v73_data = r0[16];
          float v74_data = r0[18];
          float v75_data = r0[20];
          float v76_data = r0[22];
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v72_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v74_data, v77_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v75_data, v78_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v79_acc, 3, 2, 0);
          float v81_data = r0[24];
          float v82_data = r0[26];
          float v83_data = r0[28];
          float v84_data = r0[30];
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v81_data, v80_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v82_data, v85_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v83_data, v86_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v84_data, v87_acc, 3, 3, 0);
          float v89_data = r0[32];
          float v90_data = r0[34];
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v89_data, v88_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v90_data, v93_acc, 3, 4, 0);
          r2[0] = (v94_acc[0]);
          r2[2] = (v94_acc[1]);
          r2[4] = (v94_acc[2]);
          r2[6] = (v94_acc[3]);
          tensorforge::VectorT<float, 4> v99_acc{};
          float v100_data = r0[1];
          float v101_data = r0[3];
          float v102_data = r0[5];
          float v103_data = r0[7];
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v100_data, v99_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v101_data, v104_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v102_data, v105_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v103_data, v106_acc, 3, 0, 0);
          float v108_data = r0[9];
          float v109_data = r0[11];
          float v110_data = r0[13];
          float v111_data = r0[15];
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v108_data, v107_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v109_data, v112_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v110_data, v113_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v111_data, v114_acc, 3, 1, 0);
          float v116_data = r0[17];
          float v117_data = r0[19];
          float v118_data = r0[21];
          float v119_data = r0[23];
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v116_data, v115_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v117_data, v120_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v118_data, v121_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v119_data, v122_acc, 3, 2, 0);
          float v124_data = r0[25];
          float v125_data = r0[27];
          float v126_data = r0[29];
          float v127_data = r0[31];
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v124_data, v123_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v125_data, v128_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v126_data, v129_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v127_data, v130_acc, 3, 3, 0);
          float v132_data = r0[33];
          float v133_data = r0[35];
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v132_data, v131_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v133_data, v136_acc, 3, 4, 0);
          r2[1] = (v137_acc[0]);
          r2[3] = (v137_acc[1]);
          r2[5] = (v137_acc[2]);
          r2[7] = (v137_acc[3]);
          float v142_data = r1[4];
          float v143_data = r1[5];
          float v144_data = r1[6];
          float v145_data = r1[7];
          float v146_tp{};
          float v147_tp{};
          float v148_tp{};
          float v149_tp{};
          tensorforge::transpose4x4b32(v146_tp, v147_tp, v148_tp, v149_tp, v142_data, v143_data, v144_data, v145_data);
          tensorforge::VectorT<float, 4> v150_acc{};
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v57_data, v150_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v58_data, v155_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v59_data, v156_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v60_data, v157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v65_data, v158_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v66_data, v163_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v67_data, v164_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v68_data, v165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v73_data, v166_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v74_data, v171_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v75_data, v172_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v76_data, v173_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v81_data, v174_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v82_data, v179_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v83_data, v180_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v84_data, v181_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v89_data, v182_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v90_data, v187_acc, 3, 4, 0);
          r2[8] = (v188_acc[0]);
          r2[10] = (v188_acc[1]);
          r2[12] = (v188_acc[2]);
          r2[14] = (v188_acc[3]);
          tensorforge::VectorT<float, 4> v193_acc{};
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v100_data, v193_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v101_data, v198_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v102_data, v199_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v103_data, v200_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v108_data, v201_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v109_data, v206_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v110_data, v207_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v111_data, v208_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v116_data, v209_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v117_data, v214_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v118_data, v215_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v119_data, v216_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v124_data, v217_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v125_data, v222_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v126_data, v223_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v127_data, v224_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v146_tp, v132_data, v225_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v147_tp, v133_data, v230_acc, 3, 4, 0);
          r2[9] = (v231_acc[0]);
          r2[11] = (v231_acc[1]);
          r2[13] = (v231_acc[2]);
          r2[15] = (v231_acc[3]);
          float v236_data = r1[8];
          float v237_data = r1[9];
          float v238_data = r1[10];
          float v239_data = r1[11];
          float v240_tp{};
          float v241_tp{};
          float v242_tp{};
          float v243_tp{};
          tensorforge::transpose4x4b32(v240_tp, v241_tp, v242_tp, v243_tp, v236_data, v237_data, v238_data, v239_data);
          tensorforge::VectorT<float, 4> v244_acc{};
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v57_data, v244_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v58_data, v249_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v59_data, v250_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v60_data, v251_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v65_data, v252_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v66_data, v257_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v67_data, v258_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v260_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v68_data, v259_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v73_data, v260_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v74_data, v265_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v75_data, v266_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v268_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v76_data, v267_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v81_data, v268_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v82_data, v273_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v83_data, v274_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v84_data, v275_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v89_data, v276_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v90_data, v281_acc, 3, 4, 0);
          r2[16] = (v282_acc[0]);
          r2[18] = (v282_acc[1]);
          r2[20] = (v282_acc[2]);
          r2[22] = (v282_acc[3]);
          tensorforge::VectorT<float, 4> v287_acc{};
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v100_data, v287_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v101_data, v292_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v102_data, v293_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v103_data, v294_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v108_data, v295_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v109_data, v300_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v110_data, v301_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v111_data, v302_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v116_data, v303_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v117_data, v308_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v118_data, v309_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v119_data, v310_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v124_data, v311_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v125_data, v316_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v242_tp, v126_data, v317_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v243_tp, v127_data, v318_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v132_data, v319_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v133_data, v324_acc, 3, 4, 0);
          r2[17] = (v325_acc[0]);
          r2[19] = (v325_acc[1]);
          r2[21] = (v325_acc[2]);
          r2[23] = (v325_acc[3]);
          float v330_data = r1[12];
          float v331_data = r1[13];
          float v332_data = r1[14];
          float v333_data = r1[15];
          float v334_tp{};
          float v335_tp{};
          float v336_tp{};
          float v337_tp{};
          tensorforge::transpose4x4b32(v334_tp, v335_tp, v336_tp, v337_tp, v330_data, v331_data, v332_data, v333_data);
          tensorforge::VectorT<float, 4> v338_acc{};
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v57_data, v338_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v58_data, v343_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v59_data, v344_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v60_data, v345_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v65_data, v346_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v66_data, v351_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v67_data, v352_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v68_data, v353_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v73_data, v354_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v74_data, v359_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v75_data, v360_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v362_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v76_data, v361_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v81_data, v362_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v368_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v82_data, v367_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v369_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v83_data, v368_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v84_data, v369_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v375_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v89_data, v370_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v90_data, v375_acc, 3, 4, 0);
          r2[24] = (v376_acc[0]);
          r2[26] = (v376_acc[1]);
          r2[28] = (v376_acc[2]);
          r2[30] = (v376_acc[3]);
          tensorforge::VectorT<float, 4> v381_acc{};
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v100_data, v381_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v101_data, v386_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v102_data, v387_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v103_data, v388_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v108_data, v389_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v109_data, v394_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v110_data, v395_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v111_data, v396_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v116_data, v397_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v117_data, v402_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v118_data, v403_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v119_data, v404_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v124_data, v405_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v125_data, v410_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v336_tp, v126_data, v411_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v337_tp, v127_data, v412_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v132_data, v413_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v335_tp, v133_data, v418_acc, 3, 4, 0);
          r2[25] = (v419_acc[0]);
          r2[27] = (v419_acc[1]);
          r2[29] = (v419_acc[2]);
          r2[31] = (v419_acc[3]);
          float v424_data = r1[16];
          float v425_data = r1[17];
          float v428_tp{};
          float v429_tp{};
          float v430_tp{};
          float v431_tp{};
          tensorforge::transpose4x4b32(v428_tp, v429_tp, v430_tp, v431_tp, v424_data, v425_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v432_acc{};
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v57_data, v432_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v58_data, v437_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v59_data, v438_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v60_data, v439_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v65_data, v440_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v66_data, v445_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v67_data, v446_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v68_data, v447_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v73_data, v448_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v74_data, v453_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v75_data, v454_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v76_data, v455_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v81_data, v456_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v82_data, v461_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v83_data, v462_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v84_data, v463_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v89_data, v464_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v90_data, v469_acc, 3, 4, 0);
          r2[32] = (v470_acc[0]);
          r2[34] = (v470_acc[1]);
          tensorforge::VectorT<float, 4> v473_acc{};
          tensorforge::VectorT<float, 4> v478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v100_data, v473_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v101_data, v478_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v102_data, v479_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v103_data, v480_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v108_data, v481_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v109_data, v486_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v110_data, v487_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v111_data, v488_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v116_data, v489_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v117_data, v494_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v118_data, v495_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v119_data, v496_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v124_data, v497_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v125_data, v502_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v430_tp, v126_data, v503_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v431_tp, v127_data, v504_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v428_tp, v132_data, v505_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v429_tp, v133_data, v510_acc, 3, 4, 0);
          r2[33] = (v511_acc[0]);
          r2[35] = (v511_acc[1]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v517_i0 = 0; v517_i0 < 1; ++v517_i0) {
            int32_t v526_lead = v10_lead + (v517_i0 * 32);
            #pragma unroll
            for (int32_t v518_i1 = 0; v518_i1 < 18; ++v518_i1) {
              float v521_data = r2[(v517_i0 + (v518_i1 * 2))];
              glb_m0[(v526_lead + (v518_i1 * 56))] = v521_data;
            }
          }
          if (v10_lead < 24) {
            int32_t v538_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v530_i1 = 0; v530_i1 < 18; ++v530_i1) {
              float v533_data = r2[(1 + (v530_i1 * 2))];
              glb_m0[(v538_lead + (v530_i1 * 56))] = v533_data;
            }
          }
        }
      }
    }
  }
}

