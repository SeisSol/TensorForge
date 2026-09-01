// === base name ===
kernel_98b8c9eb8b

// === header ===
void launcher_kernel_98b8c9eb8b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_98b8c9eb8b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_98b8c9eb8b, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_98b8c9eb8b), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_98b8c9eb8b, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_98b8c9eb8b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      int32_t v12_lead = threadIdx.x % 16;
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
        const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
        const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
        float r0[16]{};
        // r0 = load{g>r}(glb_m1);
        #pragma unroll
        for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
          int32_t v19_lead = v12_lead + (v13_i0 * 16);
          #pragma unroll
          for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
            float v22_data = __builtin_nontemporal_load(&glb_m1[(v19_lead + (v14_i1 * 16))]);
            r0[(v13_i0 + v14_i1)] = v22_data;
          }
        }
        float r1[16]{};
        // r1 = load{g>r}(glb_m2);
        float v25_lin = glb_m2[0 + threadIdx.x * 1];
        r1[0] = v25_lin;
        float v26_lin = glb_m2[16 + threadIdx.x * 1];
        r1[1] = v26_lin;
        float v27_lin = glb_m2[32 + threadIdx.x * 1];
        r1[2] = v27_lin;
        float v28_lin = glb_m2[48 + threadIdx.x * 1];
        r1[3] = v28_lin;
        float v29_lin = glb_m2[64 + threadIdx.x * 1];
        r1[4] = v29_lin;
        float v30_lin = glb_m2[80 + threadIdx.x * 1];
        r1[5] = v30_lin;
        float v31_lin = glb_m2[96 + threadIdx.x * 1];
        r1[6] = v31_lin;
        float v32_lin = glb_m2[112 + threadIdx.x * 1];
        r1[7] = v32_lin;
        float v33_lin = glb_m2[128 + threadIdx.x * 1];
        r1[8] = v33_lin;
        float v34_lin = glb_m2[144 + threadIdx.x * 1];
        r1[9] = v34_lin;
        float v35_lin = glb_m2[160 + threadIdx.x * 1];
        r1[10] = v35_lin;
        float v36_lin = glb_m2[176 + threadIdx.x * 1];
        r1[11] = v36_lin;
        float v37_lin = glb_m2[192 + threadIdx.x * 1];
        r1[12] = v37_lin;
        float v38_lin = glb_m2[208 + threadIdx.x * 1];
        r1[13] = v38_lin;
        float v39_lin = glb_m2[224 + threadIdx.x * 1];
        r1[14] = v39_lin;
        float v40_lin = glb_m2[240 + threadIdx.x * 1];
        r1[15] = v40_lin;
        // wait(r0 = load{g>r}(glb_m1););
        // wait(r1 = load{g>r}(glb_m2););
        float r2[16]{};
        // r2 = +(r0 * r1) + None
        // [(0, 16), (0, 16)] [(0, 16)]
        float v42_data = r1[0];
        float v43_data = r1[1];
        float v44_data = r1[2];
        float v45_data = r1[3];
        float v46_tp{};
        float v47_tp{};
        float v48_tp{};
        float v49_tp{};
        tensorforge::transpose4x4b32(v46_tp, v47_tp, v48_tp, v49_tp, v42_data, v43_data, v44_data, v45_data);
        tensorforge::VectorT<float, 4> v50_acc{};
        float v51_data = r0[0];
        float v52_data = r0[1];
        float v53_data = r0[2];
        float v54_data = r0[3];
        tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v50_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
        float v59_data = r0[4];
        float v60_data = r0[5];
        float v61_data = r0[6];
        float v62_data = r0[7];
        tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v58_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
        float v67_data = r0[8];
        float v68_data = r0[9];
        float v69_data = r0[10];
        float v70_data = r0[11];
        tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v66_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v71_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 2, 0);
        float v75_data = r0[12];
        float v76_data = r0[13];
        float v77_data = r0[14];
        float v78_data = r0[15];
        tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v75_data, v74_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v76_data, v79_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v80_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v78_data, v81_acc, 2, 3, 0);
        r2[0] = (v82_acc[0]);
        r2[1] = (v82_acc[1]);
        r2[2] = (v82_acc[2]);
        r2[3] = (v82_acc[3]);
        float v87_data = r1[4];
        float v88_data = r1[5];
        float v89_data = r1[6];
        float v90_data = r1[7];
        float v91_tp{};
        float v92_tp{};
        float v93_tp{};
        float v94_tp{};
        tensorforge::transpose4x4b32(v91_tp, v92_tp, v93_tp, v94_tp, v87_data, v88_data, v89_data, v90_data);
        tensorforge::VectorT<float, 4> v95_acc{};
        tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v51_data, v95_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v52_data, v100_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v53_data, v101_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v54_data, v102_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v59_data, v103_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v108_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v109_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v110_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v67_data, v111_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v116_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v117_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v118_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v75_data, v119_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v124_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v125_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v126_acc, 2, 3, 0);
        r2[4] = (v127_acc[0]);
        r2[5] = (v127_acc[1]);
        r2[6] = (v127_acc[2]);
        r2[7] = (v127_acc[3]);
        float v132_data = r1[8];
        float v133_data = r1[9];
        float v134_data = r1[10];
        float v135_data = r1[11];
        float v136_tp{};
        float v137_tp{};
        float v138_tp{};
        float v139_tp{};
        tensorforge::transpose4x4b32(v136_tp, v137_tp, v138_tp, v139_tp, v132_data, v133_data, v134_data, v135_data);
        tensorforge::VectorT<float, 4> v140_acc{};
        tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v51_data, v140_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v52_data, v145_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v53_data, v146_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v54_data, v147_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v59_data, v148_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v60_data, v153_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v61_data, v154_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v62_data, v155_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v67_data, v156_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v68_data, v161_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v69_data, v162_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v70_data, v163_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v75_data, v164_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v76_data, v169_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v77_data, v170_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v78_data, v171_acc, 2, 3, 0);
        r2[8] = (v172_acc[0]);
        r2[9] = (v172_acc[1]);
        r2[10] = (v172_acc[2]);
        r2[11] = (v172_acc[3]);
        float v177_data = r1[12];
        float v178_data = r1[13];
        float v179_data = r1[14];
        float v180_data = r1[15];
        float v181_tp{};
        float v182_tp{};
        float v183_tp{};
        float v184_tp{};
        tensorforge::transpose4x4b32(v181_tp, v182_tp, v183_tp, v184_tp, v177_data, v178_data, v179_data, v180_data);
        tensorforge::VectorT<float, 4> v185_acc{};
        tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v51_data, v185_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v52_data, v190_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v53_data, v191_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v54_data, v192_acc, 2, 0, 0);
        tensorforge::VectorT<float, 4> v198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v59_data, v193_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v60_data, v198_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v61_data, v199_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v62_data, v200_acc, 2, 1, 0);
        tensorforge::VectorT<float, 4> v206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v67_data, v201_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v68_data, v206_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v69_data, v207_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v70_data, v208_acc, 2, 2, 0);
        tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v181_tp, v75_data, v209_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v182_tp, v76_data, v214_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v77_data, v215_acc, 2, 3, 0);
        tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v78_data, v216_acc, 2, 3, 0);
        r2[12] = (v217_acc[0]);
        r2[13] = (v217_acc[1]);
        r2[14] = (v217_acc[2]);
        r2[15] = (v217_acc[3]);
        // glb_m0 = store{r>g}(r2);
        #pragma unroll
        for (int32_t v225_i0 = 0; v225_i0 < 1; ++v225_i0) {
          int32_t v233_lead = v12_lead + (v225_i0 * 16);
          #pragma unroll
          for (int32_t v226_i1 = 0; v226_i1 < 16; ++v226_i1) {
            float v228_data = r2[(v225_i0 + v226_i1)];
            glb_m0[(v233_lead + (v226_i1 * 16))] = v228_data;
          }
        }
      }
    }
  }
}

