// === base name ===
kernel_a5ad08b73b

// === header ===
void launcher_kernel_a5ad08b73b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 = nullptr, unsigned* flags1 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a5ad08b73b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a5ad08b73b, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a5ad08b73b), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_a5ad08b73b, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  numElements0,  numElements1,  flags0 ,  flags1 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a5ad08b73b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m3 16×16(16×16) {0..16}×{0..16} strided
    // m4 16×16(16×16) {0..16}×{0..16} strided
    // m5 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    // fence
    // m3 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m4 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m5 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 32;
          if (v13_lead < 16) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v13_lead + (v15_i1 * 16))]);
              r0[v15_i1] = v23_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v13_lead < 16) {
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 16; ++v30_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m2[(v13_lead + (v30_i1 * 16))]);
              r1[v30_i1] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v41_data = r1[0];
          float v42_data = r1[1];
          float v43_data = r1[2];
          float v44_data = r1[3];
          float v45_tp{};
          float v46_tp{};
          float v47_tp{};
          float v48_tp{};
          tensorforge::transpose4x4b32(v45_tp, v46_tp, v47_tp, v48_tp, v41_data, v42_data, v43_data, v44_data);
          tensorforge::VectorT<float, 4> v49_acc{};
          float v50_data = r0[0];
          float v51_data = r0[1];
          float v52_data = r0[2];
          float v53_data = r0[3];
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v49_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v54_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 3, 0, 0);
          float v58_data = r0[4];
          float v59_data = r0[5];
          float v60_data = r0[6];
          float v61_data = r0[7];
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v57_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v62_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 3, 1, 0);
          float v66_data = r0[8];
          float v67_data = r0[9];
          float v68_data = r0[10];
          float v69_data = r0[11];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v65_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v70_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v71_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 3, 2, 0);
          float v74_data = r0[12];
          float v75_data = r0[13];
          float v76_data = r0[14];
          float v77_data = r0[15];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v73_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v75_data, v78_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v76_data, v79_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v80_acc, 3, 3, 0);
          r2[0] = (v81_acc[0]);
          r2[1] = (v81_acc[1]);
          r2[2] = (v81_acc[2]);
          r2[3] = (v81_acc[3]);
          float v86_data = r1[4];
          float v87_data = r1[5];
          float v88_data = r1[6];
          float v89_data = r1[7];
          float v90_tp{};
          float v91_tp{};
          float v92_tp{};
          float v93_tp{};
          tensorforge::transpose4x4b32(v90_tp, v91_tp, v92_tp, v93_tp, v86_data, v87_data, v88_data, v89_data);
          tensorforge::VectorT<float, 4> v94_acc{};
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v50_data, v94_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v51_data, v99_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v52_data, v100_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v53_data, v101_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v102_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v59_data, v107_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v108_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v109_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v110_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v67_data, v115_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v116_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v117_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v118_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v75_data, v123_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v124_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v125_acc, 3, 3, 0);
          r2[4] = (v126_acc[0]);
          r2[5] = (v126_acc[1]);
          r2[6] = (v126_acc[2]);
          r2[7] = (v126_acc[3]);
          float v131_data = r1[8];
          float v132_data = r1[9];
          float v133_data = r1[10];
          float v134_data = r1[11];
          float v135_tp{};
          float v136_tp{};
          float v137_tp{};
          float v138_tp{};
          tensorforge::transpose4x4b32(v135_tp, v136_tp, v137_tp, v138_tp, v131_data, v132_data, v133_data, v134_data);
          tensorforge::VectorT<float, 4> v139_acc{};
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v50_data, v139_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v51_data, v144_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v52_data, v145_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v53_data, v146_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v58_data, v147_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v59_data, v152_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v60_data, v153_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v61_data, v154_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v66_data, v155_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v67_data, v160_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v68_data, v161_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v69_data, v162_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v74_data, v163_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v75_data, v168_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v76_data, v169_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v77_data, v170_acc, 3, 3, 0);
          r2[8] = (v171_acc[0]);
          r2[9] = (v171_acc[1]);
          r2[10] = (v171_acc[2]);
          r2[11] = (v171_acc[3]);
          float v176_data = r1[12];
          float v177_data = r1[13];
          float v178_data = r1[14];
          float v179_data = r1[15];
          float v180_tp{};
          float v181_tp{};
          float v182_tp{};
          float v183_tp{};
          tensorforge::transpose4x4b32(v180_tp, v181_tp, v182_tp, v183_tp, v176_data, v177_data, v178_data, v179_data);
          tensorforge::VectorT<float, 4> v184_acc{};
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v50_data, v184_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v51_data, v189_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v52_data, v190_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v53_data, v191_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v58_data, v192_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v59_data, v197_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v60_data, v198_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v61_data, v199_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v66_data, v200_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v67_data, v205_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v68_data, v206_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v69_data, v207_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v74_data, v208_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v75_data, v213_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v76_data, v214_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v77_data, v215_acc, 3, 3, 0);
          r2[12] = (v216_acc[0]);
          r2[13] = (v216_acc[1]);
          r2[14] = (v216_acc[2]);
          r2[15] = (v216_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v13_lead < 16) {
            #pragma unroll
            for (int32_t v225_i1 = 0; v225_i1 < 16; ++v225_i1) {
              float v227_data = r2[v225_i1];
              glb_m0[(v13_lead + (v225_i1 * 16))] = v227_data;
            }
          }
        }
      }
    }
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x) + numElements0) % (gridDim.x * blockDim.y);
      const auto batchId1 = batchId_start < numElements1 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      __syncthreads();
      for (size_t batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x) + numElements0) % (gridDim.x * blockDim.y); batchId0 < numElements1; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements1 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 256 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m4);
          int32_t v248_lead = threadIdx.x % 32;
          if (v248_lead < 16) {
            #pragma unroll
            for (int32_t v250_i1 = 0; v250_i1 < 16; ++v250_i1) {
              float v258_data = __builtin_nontemporal_load(&glb_m4[(v248_lead + (v250_i1 * 16))]);
              r0[v250_i1] = v258_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m5);
          if (v248_lead < 16) {
            #pragma unroll
            for (int32_t v265_i1 = 0; v265_i1 < 16; ++v265_i1) {
              float v273_data = __builtin_nontemporal_load(&glb_m5[(v248_lead + (v265_i1 * 16))]);
              r1[v265_i1] = v273_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m4););
          // wait(r1 = load{g>r}(glb_m5););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v276_data = r1[0];
          float v277_data = r1[1];
          float v278_data = r1[2];
          float v279_data = r1[3];
          float v280_tp{};
          float v281_tp{};
          float v282_tp{};
          float v283_tp{};
          tensorforge::transpose4x4b32(v280_tp, v281_tp, v282_tp, v283_tp, v276_data, v277_data, v278_data, v279_data);
          tensorforge::VectorT<float, 4> v284_acc{};
          float v285_data = r0[0];
          float v286_data = r0[1];
          float v287_data = r0[2];
          float v288_data = r0[3];
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v285_data, v284_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v286_data, v289_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v287_data, v290_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v288_data, v291_acc, 3, 0, 0);
          float v293_data = r0[4];
          float v294_data = r0[5];
          float v295_data = r0[6];
          float v296_data = r0[7];
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v293_data, v292_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v294_data, v297_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v295_data, v298_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v296_data, v299_acc, 3, 1, 0);
          float v301_data = r0[8];
          float v302_data = r0[9];
          float v303_data = r0[10];
          float v304_data = r0[11];
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v301_data, v300_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v302_data, v305_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v303_data, v306_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v304_data, v307_acc, 3, 2, 0);
          float v309_data = r0[12];
          float v310_data = r0[13];
          float v311_data = r0[14];
          float v312_data = r0[15];
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v309_data, v308_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v310_data, v313_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v311_data, v314_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v283_tp, v312_data, v315_acc, 3, 3, 0);
          r2[0] = (v316_acc[0]);
          r2[1] = (v316_acc[1]);
          r2[2] = (v316_acc[2]);
          r2[3] = (v316_acc[3]);
          float v321_data = r1[4];
          float v322_data = r1[5];
          float v323_data = r1[6];
          float v324_data = r1[7];
          float v325_tp{};
          float v326_tp{};
          float v327_tp{};
          float v328_tp{};
          tensorforge::transpose4x4b32(v325_tp, v326_tp, v327_tp, v328_tp, v321_data, v322_data, v323_data, v324_data);
          tensorforge::VectorT<float, 4> v329_acc{};
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v285_data, v329_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v286_data, v334_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v287_data, v335_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v337_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v288_data, v336_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v293_data, v337_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v294_data, v342_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v295_data, v343_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v345_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v296_data, v344_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v301_data, v345_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v302_data, v350_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v303_data, v351_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v304_data, v352_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v325_tp, v309_data, v353_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v326_tp, v310_data, v358_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v360_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v327_tp, v311_data, v359_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v361_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v328_tp, v312_data, v360_acc, 3, 3, 0);
          r2[4] = (v361_acc[0]);
          r2[5] = (v361_acc[1]);
          r2[6] = (v361_acc[2]);
          r2[7] = (v361_acc[3]);
          float v366_data = r1[8];
          float v367_data = r1[9];
          float v368_data = r1[10];
          float v369_data = r1[11];
          float v370_tp{};
          float v371_tp{};
          float v372_tp{};
          float v373_tp{};
          tensorforge::transpose4x4b32(v370_tp, v371_tp, v372_tp, v373_tp, v366_data, v367_data, v368_data, v369_data);
          tensorforge::VectorT<float, 4> v374_acc{};
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v285_data, v374_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v286_data, v379_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v287_data, v380_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v288_data, v381_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v293_data, v382_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v294_data, v387_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v295_data, v388_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v390_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v296_data, v389_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v301_data, v390_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v302_data, v395_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v303_data, v396_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v398_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v304_data, v397_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v370_tp, v309_data, v398_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v371_tp, v310_data, v403_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v372_tp, v311_data, v404_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v373_tp, v312_data, v405_acc, 3, 3, 0);
          r2[8] = (v406_acc[0]);
          r2[9] = (v406_acc[1]);
          r2[10] = (v406_acc[2]);
          r2[11] = (v406_acc[3]);
          float v411_data = r1[12];
          float v412_data = r1[13];
          float v413_data = r1[14];
          float v414_data = r1[15];
          float v415_tp{};
          float v416_tp{};
          float v417_tp{};
          float v418_tp{};
          tensorforge::transpose4x4b32(v415_tp, v416_tp, v417_tp, v418_tp, v411_data, v412_data, v413_data, v414_data);
          tensorforge::VectorT<float, 4> v419_acc{};
          tensorforge::VectorT<float, 4> v424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v285_data, v419_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v286_data, v424_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v287_data, v425_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v288_data, v426_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v293_data, v427_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v294_data, v432_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v295_data, v433_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v296_data, v434_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v301_data, v435_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v302_data, v440_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v303_data, v441_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v304_data, v442_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v415_tp, v309_data, v443_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v416_tp, v310_data, v448_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v417_tp, v311_data, v449_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v418_tp, v312_data, v450_acc, 3, 3, 0);
          r2[12] = (v451_acc[0]);
          r2[13] = (v451_acc[1]);
          r2[14] = (v451_acc[2]);
          r2[15] = (v451_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v248_lead < 16) {
            #pragma unroll
            for (int32_t v460_i1 = 0; v460_i1 < 16; ++v460_i1) {
              float v462_data = r2[v460_i1];
              glb_m3[(v248_lead + (v460_i1 * 16))] = v462_data;
            }
          }
        }
      }
    }
  }
}

