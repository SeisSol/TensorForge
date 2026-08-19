// === base name ===
kernel_e7f2438624

// === header ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e7f2438624, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e7f2438624), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_e7f2438624, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×16(12×16) {0..12}×{0..16} strided
    // m1 12×20(12×20) {0..12}×{0..20} strided
    // m2 16×20(16×20) {0..16}×{0..20} strided
    // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 12×20(12×20) {0..12}×{0..20} strided({0..12}×{0..20})[0, -1]×m2 16×20(16×20) {0..16}×{0..20} strided({0..16}×{0..20})[1, -1]
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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
          float r0[20]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 20; ++v4_i1) {
              int32_t v11_a = v2_lead + (v4_i1 * 12);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v13_a = 0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          float r1[20]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v16 = glb_m2[16 + threadIdx.x * 1];
            r1[1] = v16;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[2] = v32;
            float v48 = glb_m2[48 + threadIdx.x * 1];
            r1[3] = v48;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r1[4] = v64;
            float v80 = glb_m2[80 + threadIdx.x * 1];
            r1[5] = v80;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r1[6] = v96;
            float v112 = glb_m2[112 + threadIdx.x * 1];
            r1[7] = v112;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r1[8] = v128;
            float v144 = glb_m2[144 + threadIdx.x * 1];
            r1[9] = v144;
            float v160 = glb_m2[160 + threadIdx.x * 1];
            r1[10] = v160;
            float v176 = glb_m2[176 + threadIdx.x * 1];
            r1[11] = v176;
            float v192 = glb_m2[192 + threadIdx.x * 1];
            r1[12] = v192;
            float v208 = glb_m2[208 + threadIdx.x * 1];
            r1[13] = v208;
            float v224 = glb_m2[224 + threadIdx.x * 1];
            r1[14] = v224;
            float v240 = glb_m2[240 + threadIdx.x * 1];
            r1[15] = v240;
            float v256 = glb_m2[256 + threadIdx.x * 1];
            r1[16] = v256;
            float v272 = glb_m2[272 + threadIdx.x * 1];
            r1[17] = v272;
            float v288 = glb_m2[288 + threadIdx.x * 1];
            r1[18] = v288;
            float v304 = glb_m2[304 + threadIdx.x * 1];
            r1[19] = v304;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          auto& ir2 = r2;
          float v14_data = tensorforge::broadcast<16, 1, 0>(r1[0]);
          float v15_data = tensorforge::broadcast<16, 1, 1>(r1[0]);
          float v16_data = tensorforge::broadcast<16, 1, 2>(r1[0]);
          float v17_data = tensorforge::broadcast<16, 1, 3>(r1[0]);
          float v18_tp{};
          float v19_tp{};
          float v20_tp{};
          float v21_tp{};
          tensorforge::transpose4x4b32(v18_tp, v19_tp, v20_tp, v21_tp, v14_data, v15_data, v16_data, v17_data);
          float v22_data = tensorforge::broadcast<16, 1, 0>(r1[1]);
          float v23_data = tensorforge::broadcast<16, 1, 1>(r1[1]);
          float v24_data = tensorforge::broadcast<16, 1, 2>(r1[1]);
          float v25_data = tensorforge::broadcast<16, 1, 3>(r1[1]);
          float v26_tp{};
          float v27_tp{};
          float v28_tp{};
          float v29_tp{};
          tensorforge::transpose4x4b32(v26_tp, v27_tp, v28_tp, v29_tp, v22_data, v23_data, v24_data, v25_data);
          tensorforge::VectorT<float, 4> v30_acc{};
          float v31_data = r0[0];
          float v32_data = r0[1];
          float v33_data = r0[2];
          float v34_data = r0[3];
          tensorforge::VectorT<float, 4> v35_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v18_tp, v31_data, v30_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v36_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v19_tp, v32_data, v35_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v37_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v33_data, v36_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v38_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v34_data, v37_acc, 2, 0, 0);
          float v39_data = r0[4];
          float v40_data = r0[5];
          float v41_data = r0[6];
          float v42_data = r0[7];
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v18_tp, v39_data, v38_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v19_tp, v40_data, v43_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v41_data, v44_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v42_data, v45_acc, 2, 1, 0);
          float v47_data = r0[8];
          float v48_data = r0[9];
          float v49_data = r0[10];
          float v50_data = r0[11];
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v18_tp, v47_data, v46_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v19_tp, v48_data, v51_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v49_data, v52_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v50_data, v53_acc, 2, 2, 0);
          float v55_data = r0[12];
          float v56_data = r0[13];
          float v57_data = r0[14];
          float v58_data = r0[15];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v18_tp, v55_data, v54_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v19_tp, v56_data, v59_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v57_data, v60_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v58_data, v61_acc, 2, 3, 0);
          float v63_data = r0[16];
          float v64_data = r0[17];
          float v65_data = r0[18];
          float v66_data = r0[19];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v26_tp, v63_data, v62_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v27_tp, v64_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v28_tp, v65_data, v68_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v29_tp, v66_data, v69_acc, 2, 0, 0);
          ir2[0] = (v70_acc[0]);
          ir2[1] = (v70_acc[1]);
          ir2[2] = (v70_acc[2]);
          ir2[3] = (v70_acc[3]);
          float v75_data = tensorforge::broadcast<16, 1, 4>(r1[0]);
          float v76_data = tensorforge::broadcast<16, 1, 5>(r1[0]);
          float v77_data = tensorforge::broadcast<16, 1, 6>(r1[0]);
          float v78_data = tensorforge::broadcast<16, 1, 7>(r1[0]);
          float v79_tp{};
          float v80_tp{};
          float v81_tp{};
          float v82_tp{};
          tensorforge::transpose4x4b32(v79_tp, v80_tp, v81_tp, v82_tp, v75_data, v76_data, v77_data, v78_data);
          float v83_data = tensorforge::broadcast<16, 1, 4>(r1[1]);
          float v84_data = tensorforge::broadcast<16, 1, 5>(r1[1]);
          float v85_data = tensorforge::broadcast<16, 1, 6>(r1[1]);
          float v86_data = tensorforge::broadcast<16, 1, 7>(r1[1]);
          float v87_tp{};
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          tensorforge::transpose4x4b32(v87_tp, v88_tp, v89_tp, v90_tp, v83_data, v84_data, v85_data, v86_data);
          tensorforge::VectorT<float, 4> v91_acc{};
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v31_data, v91_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v32_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v33_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v34_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v39_data, v99_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v40_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v41_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v42_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v47_data, v107_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v48_data, v112_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v49_data, v113_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v50_data, v114_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v79_tp, v55_data, v115_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v56_data, v120_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v57_data, v121_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v58_data, v122_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v123_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v128_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v129_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v130_acc, 2, 0, 0);
          ir2[4] = (v131_acc[0]);
          ir2[5] = (v131_acc[1]);
          ir2[6] = (v131_acc[2]);
          ir2[7] = (v131_acc[3]);
          float v136_data = tensorforge::broadcast<16, 1, 8>(r1[0]);
          float v137_data = tensorforge::broadcast<16, 1, 9>(r1[0]);
          float v138_data = tensorforge::broadcast<16, 1, 10>(r1[0]);
          float v139_data = tensorforge::broadcast<16, 1, 11>(r1[0]);
          float v140_tp{};
          float v141_tp{};
          float v142_tp{};
          float v143_tp{};
          tensorforge::transpose4x4b32(v140_tp, v141_tp, v142_tp, v143_tp, v136_data, v137_data, v138_data, v139_data);
          float v144_data = tensorforge::broadcast<16, 1, 8>(r1[1]);
          float v145_data = tensorforge::broadcast<16, 1, 9>(r1[1]);
          float v146_data = tensorforge::broadcast<16, 1, 10>(r1[1]);
          float v147_data = tensorforge::broadcast<16, 1, 11>(r1[1]);
          float v148_tp{};
          float v149_tp{};
          float v150_tp{};
          float v151_tp{};
          tensorforge::transpose4x4b32(v148_tp, v149_tp, v150_tp, v151_tp, v144_data, v145_data, v146_data, v147_data);
          tensorforge::VectorT<float, 4> v152_acc{};
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v31_data, v152_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v32_data, v157_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v33_data, v158_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v34_data, v159_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v39_data, v160_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v40_data, v165_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v41_data, v166_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v42_data, v167_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v47_data, v168_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v48_data, v173_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v49_data, v174_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v50_data, v175_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v181_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v55_data, v176_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v56_data, v181_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v183_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v142_tp, v57_data, v182_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v143_tp, v58_data, v183_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v189_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v148_tp, v63_data, v184_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v190_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v64_data, v189_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v191_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v65_data, v190_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v66_data, v191_acc, 2, 0, 0);
          ir2[8] = (v192_acc[0]);
          ir2[9] = (v192_acc[1]);
          ir2[10] = (v192_acc[2]);
          ir2[11] = (v192_acc[3]);
          float v197_data = tensorforge::broadcast<16, 1, 12>(r1[0]);
          float v198_data = tensorforge::broadcast<16, 1, 13>(r1[0]);
          float v199_data = tensorforge::broadcast<16, 1, 14>(r1[0]);
          float v200_data = tensorforge::broadcast<16, 1, 15>(r1[0]);
          float v201_tp{};
          float v202_tp{};
          float v203_tp{};
          float v204_tp{};
          tensorforge::transpose4x4b32(v201_tp, v202_tp, v203_tp, v204_tp, v197_data, v198_data, v199_data, v200_data);
          float v205_data = tensorforge::broadcast<16, 1, 12>(r1[1]);
          float v206_data = tensorforge::broadcast<16, 1, 13>(r1[1]);
          float v207_data = tensorforge::broadcast<16, 1, 14>(r1[1]);
          float v208_data = tensorforge::broadcast<16, 1, 15>(r1[1]);
          float v209_tp{};
          float v210_tp{};
          float v211_tp{};
          float v212_tp{};
          tensorforge::transpose4x4b32(v209_tp, v210_tp, v211_tp, v212_tp, v205_data, v206_data, v207_data, v208_data);
          tensorforge::VectorT<float, 4> v213_acc{};
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v31_data, v213_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v32_data, v218_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v33_data, v219_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v34_data, v220_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v39_data, v221_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v40_data, v226_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v41_data, v227_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v42_data, v228_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v47_data, v229_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v48_data, v234_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v49_data, v235_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v50_data, v236_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v55_data, v237_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v56_data, v242_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v57_data, v243_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v58_data, v244_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v209_tp, v63_data, v245_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v210_tp, v64_data, v250_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v252_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v211_tp, v65_data, v251_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v212_tp, v66_data, v252_acc, 2, 0, 0);
          ir2[12] = (v253_acc[0]);
          ir2[13] = (v253_acc[1]);
          ir2[14] = (v253_acc[2]);
          ir2[15] = (v253_acc[3]);
          // glb_m0 = store{r>g}(r2);
          int32_t v260_lead = threadIdx.x % 16;
          if (v260_lead < 12) {
            #pragma unroll
            for (int32_t v262_i1 = 0; v262_i1 < 16; ++v262_i1) {
              int32_t v263_a = 0 + v262_i1;
              float v265_data = r2[v262_i1];
              int32_t v272_a = v260_lead + (v262_i1 * 12);
              glb_m0[v272_a] = v265_data;
            }
          }
          ;
        }
      }
    }
  }
}

