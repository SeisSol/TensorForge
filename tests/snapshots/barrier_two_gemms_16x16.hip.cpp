// === base name ===
kernel_9367114bd9

// === header ===
void launcher_kernel_9367114bd9(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 = nullptr, unsigned* flags1 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_9367114bd9(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9367114bd9, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  dim3 grid (gridsize, 1, 1);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_9367114bd9), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  
    auto args = tensorforge::argsPtrs( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  numElements1,  flags0 ,  flags1 );
    hipLaunchCooperativeKernel(kernel_kernel_9367114bd9, grid, block, args.data(), 0 * sizeof(float), stream);
  ;
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_9367114bd9(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m3 16×16(16×16) {0..16}×{0..16} strided
    // m4 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    // barrier
    // m3 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m4 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v12_lead = threadIdx.x % 32;
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
              float v22_data = __builtin_nontemporal_load(&glb_m1[(v12_lead + (v14_i1 * 16))]);
              r0[v14_i1] = v22_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 16; ++v29_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m2[(v12_lead + (v29_i1 * 16))]);
              r1[v29_i1] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v40_data = r1[0];
          float v41_data = r1[1];
          float v42_data = r1[2];
          float v43_data = r1[3];
          float v44_tp{};
          float v45_tp{};
          float v46_tp{};
          float v47_tp{};
          tensorforge::transpose4x4b32(v44_tp, v45_tp, v46_tp, v47_tp, v40_data, v41_data, v42_data, v43_data);
          tensorforge::VectorT<float, 4> v48_acc{};
          float v49_data = r0[0];
          float v50_data = r0[1];
          float v51_data = r0[2];
          float v52_data = r0[3];
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v48_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v53_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v54_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 3, 0, 0);
          float v57_data = r0[4];
          float v58_data = r0[5];
          float v59_data = r0[6];
          float v60_data = r0[7];
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v56_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v61_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v62_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 3, 1, 0);
          float v65_data = r0[8];
          float v66_data = r0[9];
          float v67_data = r0[10];
          float v68_data = r0[11];
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v65_data, v64_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v69_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v70_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v71_acc, 3, 2, 0);
          float v73_data = r0[12];
          float v74_data = r0[13];
          float v75_data = r0[14];
          float v76_data = r0[15];
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v73_data, v72_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v77_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v75_data, v78_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v76_data, v79_acc, 3, 3, 0);
          r2[0] = (v80_acc[0]);
          r2[1] = (v80_acc[1]);
          r2[2] = (v80_acc[2]);
          r2[3] = (v80_acc[3]);
          float v85_data = r1[4];
          float v86_data = r1[5];
          float v87_data = r1[6];
          float v88_data = r1[7];
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          float v92_tp{};
          tensorforge::transpose4x4b32(v89_tp, v90_tp, v91_tp, v92_tp, v85_data, v86_data, v87_data, v88_data);
          tensorforge::VectorT<float, 4> v93_acc{};
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v49_data, v93_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v50_data, v98_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v51_data, v99_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v52_data, v100_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v101_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v106_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v59_data, v107_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v108_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v109_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v114_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v67_data, v115_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v116_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v117_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v122_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v75_data, v123_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v124_acc, 3, 3, 0);
          r2[4] = (v125_acc[0]);
          r2[5] = (v125_acc[1]);
          r2[6] = (v125_acc[2]);
          r2[7] = (v125_acc[3]);
          float v130_data = r1[8];
          float v131_data = r1[9];
          float v132_data = r1[10];
          float v133_data = r1[11];
          float v134_tp{};
          float v135_tp{};
          float v136_tp{};
          float v137_tp{};
          tensorforge::transpose4x4b32(v134_tp, v135_tp, v136_tp, v137_tp, v130_data, v131_data, v132_data, v133_data);
          tensorforge::VectorT<float, 4> v138_acc{};
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v49_data, v138_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v50_data, v143_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v51_data, v144_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v52_data, v145_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v57_data, v146_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v58_data, v151_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v59_data, v152_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v60_data, v153_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v65_data, v154_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v66_data, v159_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v67_data, v160_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v68_data, v161_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v73_data, v162_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v74_data, v167_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v75_data, v168_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v76_data, v169_acc, 3, 3, 0);
          r2[8] = (v170_acc[0]);
          r2[9] = (v170_acc[1]);
          r2[10] = (v170_acc[2]);
          r2[11] = (v170_acc[3]);
          float v175_data = r1[12];
          float v176_data = r1[13];
          float v177_data = r1[14];
          float v178_data = r1[15];
          float v179_tp{};
          float v180_tp{};
          float v181_tp{};
          float v182_tp{};
          tensorforge::transpose4x4b32(v179_tp, v180_tp, v181_tp, v182_tp, v175_data, v176_data, v177_data, v178_data);
          tensorforge::VectorT<float, 4> v183_acc{};
          tensorforge::VectorT<float, 4> v188_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v49_data, v183_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v50_data, v188_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v51_data, v189_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v52_data, v190_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v57_data, v191_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v58_data, v196_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v59_data, v197_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v60_data, v198_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v65_data, v199_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v66_data, v204_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v67_data, v205_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v68_data, v206_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v73_data, v207_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v74_data, v212_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v75_data, v213_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v76_data, v214_acc, 3, 3, 0);
          r2[12] = (v215_acc[0]);
          r2[13] = (v215_acc[1]);
          r2[14] = (v215_acc[2]);
          r2[15] = (v215_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v224_i1 = 0; v224_i1 < 16; ++v224_i1) {
              float v226_data = r2[v224_i1];
              glb_m0[(v12_lead + (v224_i1 * 16))] = v226_data;
            }
          }
        }
      }
    }
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements1 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      cooperative_groups::this_grid().sync();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements1; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements1 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v246_lead = threadIdx.x % 32;
          if (v246_lead < 16) {
            #pragma unroll
            for (int32_t v248_i1 = 0; v248_i1 < 16; ++v248_i1) {
              float v256_data = __builtin_nontemporal_load(&glb_m0[(v246_lead + (v248_i1 * 16))]);
              r0[v248_i1] = v256_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m4);
          if (v246_lead < 16) {
            #pragma unroll
            for (int32_t v263_i1 = 0; v263_i1 < 16; ++v263_i1) {
              float v271_data = __builtin_nontemporal_load(&glb_m4[(v246_lead + (v263_i1 * 16))]);
              r1[v263_i1] = v271_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v274_data = r1[0];
          float v275_data = r1[1];
          float v276_data = r1[2];
          float v277_data = r1[3];
          float v278_tp{};
          float v279_tp{};
          float v280_tp{};
          float v281_tp{};
          tensorforge::transpose4x4b32(v278_tp, v279_tp, v280_tp, v281_tp, v274_data, v275_data, v276_data, v277_data);
          tensorforge::VectorT<float, 4> v282_acc{};
          float v283_data = r0[0];
          float v284_data = r0[1];
          float v285_data = r0[2];
          float v286_data = r0[3];
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v283_data, v282_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v284_data, v287_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v285_data, v288_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v286_data, v289_acc, 3, 0, 0);
          float v291_data = r0[4];
          float v292_data = r0[5];
          float v293_data = r0[6];
          float v294_data = r0[7];
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v291_data, v290_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v292_data, v295_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v293_data, v296_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v294_data, v297_acc, 3, 1, 0);
          float v299_data = r0[8];
          float v300_data = r0[9];
          float v301_data = r0[10];
          float v302_data = r0[11];
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v299_data, v298_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v300_data, v303_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v301_data, v304_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v302_data, v305_acc, 3, 2, 0);
          float v307_data = r0[12];
          float v308_data = r0[13];
          float v309_data = r0[14];
          float v310_data = r0[15];
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v278_tp, v307_data, v306_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v308_data, v311_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v309_data, v312_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v310_data, v313_acc, 3, 3, 0);
          r2[0] = (v314_acc[0]);
          r2[1] = (v314_acc[1]);
          r2[2] = (v314_acc[2]);
          r2[3] = (v314_acc[3]);
          float v319_data = r1[4];
          float v320_data = r1[5];
          float v321_data = r1[6];
          float v322_data = r1[7];
          float v323_tp{};
          float v324_tp{};
          float v325_tp{};
          float v326_tp{};
          tensorforge::transpose4x4b32(v323_tp, v324_tp, v325_tp, v326_tp, v319_data, v320_data, v321_data, v322_data);
          tensorforge::VectorT<float, 4> v327_acc{};
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v283_data, v327_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v284_data, v332_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v285_data, v333_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v286_data, v334_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v291_data, v335_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v292_data, v340_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v293_data, v341_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v294_data, v342_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v299_data, v343_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v300_data, v348_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v301_data, v349_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v302_data, v350_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v307_data, v351_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v308_data, v356_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v309_data, v357_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v310_data, v358_acc, 3, 3, 0);
          r2[4] = (v359_acc[0]);
          r2[5] = (v359_acc[1]);
          r2[6] = (v359_acc[2]);
          r2[7] = (v359_acc[3]);
          float v364_data = r1[8];
          float v365_data = r1[9];
          float v366_data = r1[10];
          float v367_data = r1[11];
          float v368_tp{};
          float v369_tp{};
          float v370_tp{};
          float v371_tp{};
          tensorforge::transpose4x4b32(v368_tp, v369_tp, v370_tp, v371_tp, v364_data, v365_data, v366_data, v367_data);
          tensorforge::VectorT<float, 4> v372_acc{};
          tensorforge::VectorT<float, 4> v377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v283_data, v372_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v369_tp, v284_data, v377_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v285_data, v378_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v286_data, v379_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v291_data, v380_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v369_tp, v292_data, v385_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v293_data, v386_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v294_data, v387_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v299_data, v388_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v369_tp, v300_data, v393_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v301_data, v394_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v302_data, v395_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v368_tp, v307_data, v396_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v369_tp, v308_data, v401_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v309_data, v402_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v310_data, v403_acc, 3, 3, 0);
          r2[8] = (v404_acc[0]);
          r2[9] = (v404_acc[1]);
          r2[10] = (v404_acc[2]);
          r2[11] = (v404_acc[3]);
          float v409_data = r1[12];
          float v410_data = r1[13];
          float v411_data = r1[14];
          float v412_data = r1[15];
          float v413_tp{};
          float v414_tp{};
          float v415_tp{};
          float v416_tp{};
          tensorforge::transpose4x4b32(v413_tp, v414_tp, v415_tp, v416_tp, v409_data, v410_data, v411_data, v412_data);
          tensorforge::VectorT<float, 4> v417_acc{};
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v283_data, v417_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v414_tp, v284_data, v422_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v285_data, v423_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v286_data, v424_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v291_data, v425_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v414_tp, v292_data, v430_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v293_data, v431_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v294_data, v432_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v299_data, v433_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v414_tp, v300_data, v438_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v301_data, v439_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v302_data, v440_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v413_tp, v307_data, v441_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v414_tp, v308_data, v446_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v309_data, v447_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v310_data, v448_acc, 3, 3, 0);
          r2[12] = (v449_acc[0]);
          r2[13] = (v449_acc[1]);
          r2[14] = (v449_acc[2]);
          r2[15] = (v449_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v246_lead < 16) {
            #pragma unroll
            for (int32_t v458_i1 = 0; v458_i1 < 16; ++v458_i1) {
              float v460_data = r2[v458_i1];
              glb_m3[(v246_lead + (v458_i1 * 16))] = v460_data;
            }
          }
        }
      }
    }
  }
}

