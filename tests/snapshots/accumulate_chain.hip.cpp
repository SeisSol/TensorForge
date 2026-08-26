// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8a03a3cd0d, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_8a03a3cd0d), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_8a03a3cd0d, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×12(12×12) {0..12}×{0..12} strided
    // m2 12×8(12×8) {0..12}×{0..8} strided
    // m3 12×12(12×12) {0..12}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 12×12(12×12) {0..12}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m7 12×12(12×12) {0..12}×{0..12} strided
    // m8 12×8(12×8) {0..12}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
          const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
          const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[8]{};
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
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              int32_t v34_a = v28_i1 * 12;
              int32_t v35_a = v3_lead + v34_a;
              float v43_data = __builtin_nontemporal_load(&glb_m3[(v3_lead + v34_a)]);
              int32_t v44_a = 0 + v28_i1;
              r3[v44_a] = v43_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          auto& ir2 = r2;
          float v46_data = r0[0];
          float v47_data = r0[1];
          float v48_data = r0[2];
          float v49_data = r0[3];
          float v50_data = r0[4];
          float v51_data = r0[5];
          float v52_data = r0[6];
          float v53_data = r0[7];
          float v54_data = r0[8];
          float v55_data = r0[9];
          float v56_data = r0[10];
          float v57_data = r0[11];
          float v58_acc{};
          float v59_acc{};
          float v60_acc{};
          float v61_acc{};
          float v62_acc{};
          float v63_acc{};
          float v64_acc{};
          float v65_acc{};
          float v66_lin = r1[0];
          tensorforge::fmacdpp16<0>(v58_acc, v66_lin, v46_data);
          tensorforge::fmacdpp16<1>(v58_acc, v66_lin, v47_data);
          tensorforge::fmacdpp16<2>(v58_acc, v66_lin, v48_data);
          tensorforge::fmacdpp16<3>(v58_acc, v66_lin, v49_data);
          tensorforge::fmacdpp16<4>(v58_acc, v66_lin, v50_data);
          tensorforge::fmacdpp16<5>(v58_acc, v66_lin, v51_data);
          tensorforge::fmacdpp16<6>(v58_acc, v66_lin, v52_data);
          tensorforge::fmacdpp16<7>(v58_acc, v66_lin, v53_data);
          tensorforge::fmacdpp16<8>(v58_acc, v66_lin, v54_data);
          tensorforge::fmacdpp16<9>(v58_acc, v66_lin, v55_data);
          tensorforge::fmacdpp16<10>(v58_acc, v66_lin, v56_data);
          tensorforge::fmacdpp16<11>(v58_acc, v66_lin, v57_data);
          tensorforge::fmacdpp16<12>(v59_acc, v66_lin, v46_data);
          tensorforge::fmacdpp16<13>(v59_acc, v66_lin, v47_data);
          tensorforge::fmacdpp16<14>(v59_acc, v66_lin, v48_data);
          tensorforge::fmacdpp16<15>(v59_acc, v66_lin, v49_data);
          float v67_lin = r1[1];
          tensorforge::fmacdpp16<0>(v59_acc, v67_lin, v50_data);
          tensorforge::fmacdpp16<1>(v59_acc, v67_lin, v51_data);
          tensorforge::fmacdpp16<2>(v59_acc, v67_lin, v52_data);
          tensorforge::fmacdpp16<3>(v59_acc, v67_lin, v53_data);
          tensorforge::fmacdpp16<4>(v59_acc, v67_lin, v54_data);
          tensorforge::fmacdpp16<5>(v59_acc, v67_lin, v55_data);
          tensorforge::fmacdpp16<6>(v59_acc, v67_lin, v56_data);
          tensorforge::fmacdpp16<7>(v59_acc, v67_lin, v57_data);
          tensorforge::fmacdpp16<8>(v60_acc, v67_lin, v46_data);
          tensorforge::fmacdpp16<9>(v60_acc, v67_lin, v47_data);
          tensorforge::fmacdpp16<10>(v60_acc, v67_lin, v48_data);
          tensorforge::fmacdpp16<11>(v60_acc, v67_lin, v49_data);
          tensorforge::fmacdpp16<12>(v60_acc, v67_lin, v50_data);
          tensorforge::fmacdpp16<13>(v60_acc, v67_lin, v51_data);
          tensorforge::fmacdpp16<14>(v60_acc, v67_lin, v52_data);
          tensorforge::fmacdpp16<15>(v60_acc, v67_lin, v53_data);
          float v68_lin = r1[2];
          tensorforge::fmacdpp16<0>(v60_acc, v68_lin, v54_data);
          tensorforge::fmacdpp16<1>(v60_acc, v68_lin, v55_data);
          tensorforge::fmacdpp16<2>(v60_acc, v68_lin, v56_data);
          tensorforge::fmacdpp16<3>(v60_acc, v68_lin, v57_data);
          tensorforge::fmacdpp16<4>(v61_acc, v68_lin, v46_data);
          tensorforge::fmacdpp16<5>(v61_acc, v68_lin, v47_data);
          tensorforge::fmacdpp16<6>(v61_acc, v68_lin, v48_data);
          tensorforge::fmacdpp16<7>(v61_acc, v68_lin, v49_data);
          tensorforge::fmacdpp16<8>(v61_acc, v68_lin, v50_data);
          tensorforge::fmacdpp16<9>(v61_acc, v68_lin, v51_data);
          tensorforge::fmacdpp16<10>(v61_acc, v68_lin, v52_data);
          tensorforge::fmacdpp16<11>(v61_acc, v68_lin, v53_data);
          tensorforge::fmacdpp16<12>(v61_acc, v68_lin, v54_data);
          tensorforge::fmacdpp16<13>(v61_acc, v68_lin, v55_data);
          tensorforge::fmacdpp16<14>(v61_acc, v68_lin, v56_data);
          tensorforge::fmacdpp16<15>(v61_acc, v68_lin, v57_data);
          float v69_lin = r1[3];
          tensorforge::fmacdpp16<0>(v62_acc, v69_lin, v46_data);
          tensorforge::fmacdpp16<1>(v62_acc, v69_lin, v47_data);
          tensorforge::fmacdpp16<2>(v62_acc, v69_lin, v48_data);
          tensorforge::fmacdpp16<3>(v62_acc, v69_lin, v49_data);
          tensorforge::fmacdpp16<4>(v62_acc, v69_lin, v50_data);
          tensorforge::fmacdpp16<5>(v62_acc, v69_lin, v51_data);
          tensorforge::fmacdpp16<6>(v62_acc, v69_lin, v52_data);
          tensorforge::fmacdpp16<7>(v62_acc, v69_lin, v53_data);
          tensorforge::fmacdpp16<8>(v62_acc, v69_lin, v54_data);
          tensorforge::fmacdpp16<9>(v62_acc, v69_lin, v55_data);
          tensorforge::fmacdpp16<10>(v62_acc, v69_lin, v56_data);
          tensorforge::fmacdpp16<11>(v62_acc, v69_lin, v57_data);
          tensorforge::fmacdpp16<12>(v63_acc, v69_lin, v46_data);
          tensorforge::fmacdpp16<13>(v63_acc, v69_lin, v47_data);
          tensorforge::fmacdpp16<14>(v63_acc, v69_lin, v48_data);
          tensorforge::fmacdpp16<15>(v63_acc, v69_lin, v49_data);
          float v70_lin = r1[4];
          tensorforge::fmacdpp16<0>(v63_acc, v70_lin, v50_data);
          tensorforge::fmacdpp16<1>(v63_acc, v70_lin, v51_data);
          tensorforge::fmacdpp16<2>(v63_acc, v70_lin, v52_data);
          tensorforge::fmacdpp16<3>(v63_acc, v70_lin, v53_data);
          tensorforge::fmacdpp16<4>(v63_acc, v70_lin, v54_data);
          tensorforge::fmacdpp16<5>(v63_acc, v70_lin, v55_data);
          tensorforge::fmacdpp16<6>(v63_acc, v70_lin, v56_data);
          tensorforge::fmacdpp16<7>(v63_acc, v70_lin, v57_data);
          tensorforge::fmacdpp16<8>(v64_acc, v70_lin, v46_data);
          tensorforge::fmacdpp16<9>(v64_acc, v70_lin, v47_data);
          tensorforge::fmacdpp16<10>(v64_acc, v70_lin, v48_data);
          tensorforge::fmacdpp16<11>(v64_acc, v70_lin, v49_data);
          tensorforge::fmacdpp16<12>(v64_acc, v70_lin, v50_data);
          tensorforge::fmacdpp16<13>(v64_acc, v70_lin, v51_data);
          tensorforge::fmacdpp16<14>(v64_acc, v70_lin, v52_data);
          tensorforge::fmacdpp16<15>(v64_acc, v70_lin, v53_data);
          float v71_lin = r1[5];
          tensorforge::fmacdpp16<0>(v64_acc, v71_lin, v54_data);
          tensorforge::fmacdpp16<1>(v64_acc, v71_lin, v55_data);
          tensorforge::fmacdpp16<2>(v64_acc, v71_lin, v56_data);
          tensorforge::fmacdpp16<3>(v64_acc, v71_lin, v57_data);
          tensorforge::fmacdpp16<4>(v65_acc, v71_lin, v46_data);
          tensorforge::fmacdpp16<5>(v65_acc, v71_lin, v47_data);
          tensorforge::fmacdpp16<6>(v65_acc, v71_lin, v48_data);
          tensorforge::fmacdpp16<7>(v65_acc, v71_lin, v49_data);
          tensorforge::fmacdpp16<8>(v65_acc, v71_lin, v50_data);
          tensorforge::fmacdpp16<9>(v65_acc, v71_lin, v51_data);
          tensorforge::fmacdpp16<10>(v65_acc, v71_lin, v52_data);
          tensorforge::fmacdpp16<11>(v65_acc, v71_lin, v53_data);
          tensorforge::fmacdpp16<12>(v65_acc, v71_lin, v54_data);
          tensorforge::fmacdpp16<13>(v65_acc, v71_lin, v55_data);
          tensorforge::fmacdpp16<14>(v65_acc, v71_lin, v56_data);
          tensorforge::fmacdpp16<15>(v65_acc, v71_lin, v57_data);
          ir2[0] = v58_acc;
          ir2[1] = v59_acc;
          ir2[2] = v60_acc;
          ir2[3] = v61_acc;
          ir2[4] = v62_acc;
          ir2[5] = v63_acc;
          ir2[6] = v64_acc;
          ir2[7] = v65_acc;
          float r4[8]{};
          {
            // r4 = load{g>r}(glb_m4);
            float v0 = glb_m4[0 + threadIdx.x * 1];
            r4[0] = v0;
            float v16 = glb_m4[16 + threadIdx.x * 1];
            r4[1] = v16;
            float v32 = glb_m4[32 + threadIdx.x * 1];
            r4[2] = v32;
            float v48 = glb_m4[48 + threadIdx.x * 1];
            r4[3] = v48;
            float v64 = glb_m4[64 + threadIdx.x * 1];
            r4[4] = v64;
            float v80 = glb_m4[80 + threadIdx.x * 1];
            r4[5] = v80;
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v78_i1 = 0; v78_i1 < 12; ++v78_i1) {
              int32_t v84_a = v78_i1 * 12;
              int32_t v85_a = v3_lead + v84_a;
              float v93_data = __builtin_nontemporal_load(&glb_m5[(v3_lead + v84_a)]);
              int32_t v94_a = 0 + v78_i1;
              r6[v94_a] = v93_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          {
            // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir5[8]{};
            float v96_data = r3[0];
            float v97_data = r3[1];
            float v98_data = r3[2];
            float v99_data = r3[3];
            float v100_data = r3[4];
            float v101_data = r3[5];
            float v102_data = r3[6];
            float v103_data = r3[7];
            float v104_data = r3[8];
            float v105_data = r3[9];
            float v106_data = r3[10];
            float v107_data = r3[11];
            float v108_acc{};
            float v109_acc{};
            float v110_acc{};
            float v111_acc{};
            float v112_acc{};
            float v113_acc{};
            float v114_acc{};
            float v115_acc{};
            float v116_lin = r4[0];
            tensorforge::fmacdpp16<0>(v108_acc, v116_lin, v96_data);
            tensorforge::fmacdpp16<1>(v108_acc, v116_lin, v97_data);
            tensorforge::fmacdpp16<2>(v108_acc, v116_lin, v98_data);
            tensorforge::fmacdpp16<3>(v108_acc, v116_lin, v99_data);
            tensorforge::fmacdpp16<4>(v108_acc, v116_lin, v100_data);
            tensorforge::fmacdpp16<5>(v108_acc, v116_lin, v101_data);
            tensorforge::fmacdpp16<6>(v108_acc, v116_lin, v102_data);
            tensorforge::fmacdpp16<7>(v108_acc, v116_lin, v103_data);
            tensorforge::fmacdpp16<8>(v108_acc, v116_lin, v104_data);
            tensorforge::fmacdpp16<9>(v108_acc, v116_lin, v105_data);
            tensorforge::fmacdpp16<10>(v108_acc, v116_lin, v106_data);
            tensorforge::fmacdpp16<11>(v108_acc, v116_lin, v107_data);
            tensorforge::fmacdpp16<12>(v109_acc, v116_lin, v96_data);
            tensorforge::fmacdpp16<13>(v109_acc, v116_lin, v97_data);
            tensorforge::fmacdpp16<14>(v109_acc, v116_lin, v98_data);
            tensorforge::fmacdpp16<15>(v109_acc, v116_lin, v99_data);
            float v117_lin = r4[1];
            tensorforge::fmacdpp16<0>(v109_acc, v117_lin, v100_data);
            tensorforge::fmacdpp16<1>(v109_acc, v117_lin, v101_data);
            tensorforge::fmacdpp16<2>(v109_acc, v117_lin, v102_data);
            tensorforge::fmacdpp16<3>(v109_acc, v117_lin, v103_data);
            tensorforge::fmacdpp16<4>(v109_acc, v117_lin, v104_data);
            tensorforge::fmacdpp16<5>(v109_acc, v117_lin, v105_data);
            tensorforge::fmacdpp16<6>(v109_acc, v117_lin, v106_data);
            tensorforge::fmacdpp16<7>(v109_acc, v117_lin, v107_data);
            tensorforge::fmacdpp16<8>(v110_acc, v117_lin, v96_data);
            tensorforge::fmacdpp16<9>(v110_acc, v117_lin, v97_data);
            tensorforge::fmacdpp16<10>(v110_acc, v117_lin, v98_data);
            tensorforge::fmacdpp16<11>(v110_acc, v117_lin, v99_data);
            tensorforge::fmacdpp16<12>(v110_acc, v117_lin, v100_data);
            tensorforge::fmacdpp16<13>(v110_acc, v117_lin, v101_data);
            tensorforge::fmacdpp16<14>(v110_acc, v117_lin, v102_data);
            tensorforge::fmacdpp16<15>(v110_acc, v117_lin, v103_data);
            float v118_lin = r4[2];
            tensorforge::fmacdpp16<0>(v110_acc, v118_lin, v104_data);
            tensorforge::fmacdpp16<1>(v110_acc, v118_lin, v105_data);
            tensorforge::fmacdpp16<2>(v110_acc, v118_lin, v106_data);
            tensorforge::fmacdpp16<3>(v110_acc, v118_lin, v107_data);
            tensorforge::fmacdpp16<4>(v111_acc, v118_lin, v96_data);
            tensorforge::fmacdpp16<5>(v111_acc, v118_lin, v97_data);
            tensorforge::fmacdpp16<6>(v111_acc, v118_lin, v98_data);
            tensorforge::fmacdpp16<7>(v111_acc, v118_lin, v99_data);
            tensorforge::fmacdpp16<8>(v111_acc, v118_lin, v100_data);
            tensorforge::fmacdpp16<9>(v111_acc, v118_lin, v101_data);
            tensorforge::fmacdpp16<10>(v111_acc, v118_lin, v102_data);
            tensorforge::fmacdpp16<11>(v111_acc, v118_lin, v103_data);
            tensorforge::fmacdpp16<12>(v111_acc, v118_lin, v104_data);
            tensorforge::fmacdpp16<13>(v111_acc, v118_lin, v105_data);
            tensorforge::fmacdpp16<14>(v111_acc, v118_lin, v106_data);
            tensorforge::fmacdpp16<15>(v111_acc, v118_lin, v107_data);
            float v119_lin = r4[3];
            tensorforge::fmacdpp16<0>(v112_acc, v119_lin, v96_data);
            tensorforge::fmacdpp16<1>(v112_acc, v119_lin, v97_data);
            tensorforge::fmacdpp16<2>(v112_acc, v119_lin, v98_data);
            tensorforge::fmacdpp16<3>(v112_acc, v119_lin, v99_data);
            tensorforge::fmacdpp16<4>(v112_acc, v119_lin, v100_data);
            tensorforge::fmacdpp16<5>(v112_acc, v119_lin, v101_data);
            tensorforge::fmacdpp16<6>(v112_acc, v119_lin, v102_data);
            tensorforge::fmacdpp16<7>(v112_acc, v119_lin, v103_data);
            tensorforge::fmacdpp16<8>(v112_acc, v119_lin, v104_data);
            tensorforge::fmacdpp16<9>(v112_acc, v119_lin, v105_data);
            tensorforge::fmacdpp16<10>(v112_acc, v119_lin, v106_data);
            tensorforge::fmacdpp16<11>(v112_acc, v119_lin, v107_data);
            tensorforge::fmacdpp16<12>(v113_acc, v119_lin, v96_data);
            tensorforge::fmacdpp16<13>(v113_acc, v119_lin, v97_data);
            tensorforge::fmacdpp16<14>(v113_acc, v119_lin, v98_data);
            tensorforge::fmacdpp16<15>(v113_acc, v119_lin, v99_data);
            float v120_lin = r4[4];
            tensorforge::fmacdpp16<0>(v113_acc, v120_lin, v100_data);
            tensorforge::fmacdpp16<1>(v113_acc, v120_lin, v101_data);
            tensorforge::fmacdpp16<2>(v113_acc, v120_lin, v102_data);
            tensorforge::fmacdpp16<3>(v113_acc, v120_lin, v103_data);
            tensorforge::fmacdpp16<4>(v113_acc, v120_lin, v104_data);
            tensorforge::fmacdpp16<5>(v113_acc, v120_lin, v105_data);
            tensorforge::fmacdpp16<6>(v113_acc, v120_lin, v106_data);
            tensorforge::fmacdpp16<7>(v113_acc, v120_lin, v107_data);
            tensorforge::fmacdpp16<8>(v114_acc, v120_lin, v96_data);
            tensorforge::fmacdpp16<9>(v114_acc, v120_lin, v97_data);
            tensorforge::fmacdpp16<10>(v114_acc, v120_lin, v98_data);
            tensorforge::fmacdpp16<11>(v114_acc, v120_lin, v99_data);
            tensorforge::fmacdpp16<12>(v114_acc, v120_lin, v100_data);
            tensorforge::fmacdpp16<13>(v114_acc, v120_lin, v101_data);
            tensorforge::fmacdpp16<14>(v114_acc, v120_lin, v102_data);
            tensorforge::fmacdpp16<15>(v114_acc, v120_lin, v103_data);
            float v121_lin = r4[5];
            tensorforge::fmacdpp16<0>(v114_acc, v121_lin, v104_data);
            tensorforge::fmacdpp16<1>(v114_acc, v121_lin, v105_data);
            tensorforge::fmacdpp16<2>(v114_acc, v121_lin, v106_data);
            tensorforge::fmacdpp16<3>(v114_acc, v121_lin, v107_data);
            tensorforge::fmacdpp16<4>(v115_acc, v121_lin, v96_data);
            tensorforge::fmacdpp16<5>(v115_acc, v121_lin, v97_data);
            tensorforge::fmacdpp16<6>(v115_acc, v121_lin, v98_data);
            tensorforge::fmacdpp16<7>(v115_acc, v121_lin, v99_data);
            tensorforge::fmacdpp16<8>(v115_acc, v121_lin, v100_data);
            tensorforge::fmacdpp16<9>(v115_acc, v121_lin, v101_data);
            tensorforge::fmacdpp16<10>(v115_acc, v121_lin, v102_data);
            tensorforge::fmacdpp16<11>(v115_acc, v121_lin, v103_data);
            tensorforge::fmacdpp16<12>(v115_acc, v121_lin, v104_data);
            tensorforge::fmacdpp16<13>(v115_acc, v121_lin, v105_data);
            tensorforge::fmacdpp16<14>(v115_acc, v121_lin, v106_data);
            tensorforge::fmacdpp16<15>(v115_acc, v121_lin, v107_data);
            ir5[0] = v108_acc;
            ir5[1] = v109_acc;
            ir5[2] = v110_acc;
            ir5[3] = v111_acc;
            ir5[4] = v112_acc;
            ir5[5] = v113_acc;
            ir5[6] = v114_acc;
            ir5[7] = v115_acc;
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v126_n1 = 0; v126_n1 < 8; ++v126_n1) {
                int32_t v127_a = 0 + v126_n1;
                float v129_data = ir5[v126_n1];
                int32_t v130_a = 0 + v126_n1;
                float v132_data = r2[v126_n1];
                int32_t v134_a = 0 + v126_n1;
                r5[v126_n1] = (v132_data + v129_data);
              }
            }
          }
          float r7[8]{};
          {
            // r7 = load{g>r}(glb_m6);
            float v0 = glb_m6[0 + threadIdx.x * 1];
            r7[0] = v0;
            float v16 = glb_m6[16 + threadIdx.x * 1];
            r7[1] = v16;
            float v32 = glb_m6[32 + threadIdx.x * 1];
            r7[2] = v32;
            float v48 = glb_m6[48 + threadIdx.x * 1];
            r7[3] = v48;
            float v64 = glb_m6[64 + threadIdx.x * 1];
            r7[4] = v64;
            float v80 = glb_m6[80 + threadIdx.x * 1];
            r7[5] = v80;
          }
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v142_i1 = 0; v142_i1 < 12; ++v142_i1) {
              int32_t v148_a = v142_i1 * 12;
              int32_t v149_a = v3_lead + v148_a;
              float v157_data = __builtin_nontemporal_load(&glb_m7[(v3_lead + v148_a)]);
              int32_t v158_a = 0 + v142_i1;
              r9[v158_a] = v157_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          {
            // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir8[8]{};
            float v160_data = r6[0];
            float v161_data = r6[1];
            float v162_data = r6[2];
            float v163_data = r6[3];
            float v164_data = r6[4];
            float v165_data = r6[5];
            float v166_data = r6[6];
            float v167_data = r6[7];
            float v168_data = r6[8];
            float v169_data = r6[9];
            float v170_data = r6[10];
            float v171_data = r6[11];
            float v172_acc{};
            float v173_acc{};
            float v174_acc{};
            float v175_acc{};
            float v176_acc{};
            float v177_acc{};
            float v178_acc{};
            float v179_acc{};
            float v180_lin = r7[0];
            tensorforge::fmacdpp16<0>(v172_acc, v180_lin, v160_data);
            tensorforge::fmacdpp16<1>(v172_acc, v180_lin, v161_data);
            tensorforge::fmacdpp16<2>(v172_acc, v180_lin, v162_data);
            tensorforge::fmacdpp16<3>(v172_acc, v180_lin, v163_data);
            tensorforge::fmacdpp16<4>(v172_acc, v180_lin, v164_data);
            tensorforge::fmacdpp16<5>(v172_acc, v180_lin, v165_data);
            tensorforge::fmacdpp16<6>(v172_acc, v180_lin, v166_data);
            tensorforge::fmacdpp16<7>(v172_acc, v180_lin, v167_data);
            tensorforge::fmacdpp16<8>(v172_acc, v180_lin, v168_data);
            tensorforge::fmacdpp16<9>(v172_acc, v180_lin, v169_data);
            tensorforge::fmacdpp16<10>(v172_acc, v180_lin, v170_data);
            tensorforge::fmacdpp16<11>(v172_acc, v180_lin, v171_data);
            tensorforge::fmacdpp16<12>(v173_acc, v180_lin, v160_data);
            tensorforge::fmacdpp16<13>(v173_acc, v180_lin, v161_data);
            tensorforge::fmacdpp16<14>(v173_acc, v180_lin, v162_data);
            tensorforge::fmacdpp16<15>(v173_acc, v180_lin, v163_data);
            float v181_lin = r7[1];
            tensorforge::fmacdpp16<0>(v173_acc, v181_lin, v164_data);
            tensorforge::fmacdpp16<1>(v173_acc, v181_lin, v165_data);
            tensorforge::fmacdpp16<2>(v173_acc, v181_lin, v166_data);
            tensorforge::fmacdpp16<3>(v173_acc, v181_lin, v167_data);
            tensorforge::fmacdpp16<4>(v173_acc, v181_lin, v168_data);
            tensorforge::fmacdpp16<5>(v173_acc, v181_lin, v169_data);
            tensorforge::fmacdpp16<6>(v173_acc, v181_lin, v170_data);
            tensorforge::fmacdpp16<7>(v173_acc, v181_lin, v171_data);
            tensorforge::fmacdpp16<8>(v174_acc, v181_lin, v160_data);
            tensorforge::fmacdpp16<9>(v174_acc, v181_lin, v161_data);
            tensorforge::fmacdpp16<10>(v174_acc, v181_lin, v162_data);
            tensorforge::fmacdpp16<11>(v174_acc, v181_lin, v163_data);
            tensorforge::fmacdpp16<12>(v174_acc, v181_lin, v164_data);
            tensorforge::fmacdpp16<13>(v174_acc, v181_lin, v165_data);
            tensorforge::fmacdpp16<14>(v174_acc, v181_lin, v166_data);
            tensorforge::fmacdpp16<15>(v174_acc, v181_lin, v167_data);
            float v182_lin = r7[2];
            tensorforge::fmacdpp16<0>(v174_acc, v182_lin, v168_data);
            tensorforge::fmacdpp16<1>(v174_acc, v182_lin, v169_data);
            tensorforge::fmacdpp16<2>(v174_acc, v182_lin, v170_data);
            tensorforge::fmacdpp16<3>(v174_acc, v182_lin, v171_data);
            tensorforge::fmacdpp16<4>(v175_acc, v182_lin, v160_data);
            tensorforge::fmacdpp16<5>(v175_acc, v182_lin, v161_data);
            tensorforge::fmacdpp16<6>(v175_acc, v182_lin, v162_data);
            tensorforge::fmacdpp16<7>(v175_acc, v182_lin, v163_data);
            tensorforge::fmacdpp16<8>(v175_acc, v182_lin, v164_data);
            tensorforge::fmacdpp16<9>(v175_acc, v182_lin, v165_data);
            tensorforge::fmacdpp16<10>(v175_acc, v182_lin, v166_data);
            tensorforge::fmacdpp16<11>(v175_acc, v182_lin, v167_data);
            tensorforge::fmacdpp16<12>(v175_acc, v182_lin, v168_data);
            tensorforge::fmacdpp16<13>(v175_acc, v182_lin, v169_data);
            tensorforge::fmacdpp16<14>(v175_acc, v182_lin, v170_data);
            tensorforge::fmacdpp16<15>(v175_acc, v182_lin, v171_data);
            float v183_lin = r7[3];
            tensorforge::fmacdpp16<0>(v176_acc, v183_lin, v160_data);
            tensorforge::fmacdpp16<1>(v176_acc, v183_lin, v161_data);
            tensorforge::fmacdpp16<2>(v176_acc, v183_lin, v162_data);
            tensorforge::fmacdpp16<3>(v176_acc, v183_lin, v163_data);
            tensorforge::fmacdpp16<4>(v176_acc, v183_lin, v164_data);
            tensorforge::fmacdpp16<5>(v176_acc, v183_lin, v165_data);
            tensorforge::fmacdpp16<6>(v176_acc, v183_lin, v166_data);
            tensorforge::fmacdpp16<7>(v176_acc, v183_lin, v167_data);
            tensorforge::fmacdpp16<8>(v176_acc, v183_lin, v168_data);
            tensorforge::fmacdpp16<9>(v176_acc, v183_lin, v169_data);
            tensorforge::fmacdpp16<10>(v176_acc, v183_lin, v170_data);
            tensorforge::fmacdpp16<11>(v176_acc, v183_lin, v171_data);
            tensorforge::fmacdpp16<12>(v177_acc, v183_lin, v160_data);
            tensorforge::fmacdpp16<13>(v177_acc, v183_lin, v161_data);
            tensorforge::fmacdpp16<14>(v177_acc, v183_lin, v162_data);
            tensorforge::fmacdpp16<15>(v177_acc, v183_lin, v163_data);
            float v184_lin = r7[4];
            tensorforge::fmacdpp16<0>(v177_acc, v184_lin, v164_data);
            tensorforge::fmacdpp16<1>(v177_acc, v184_lin, v165_data);
            tensorforge::fmacdpp16<2>(v177_acc, v184_lin, v166_data);
            tensorforge::fmacdpp16<3>(v177_acc, v184_lin, v167_data);
            tensorforge::fmacdpp16<4>(v177_acc, v184_lin, v168_data);
            tensorforge::fmacdpp16<5>(v177_acc, v184_lin, v169_data);
            tensorforge::fmacdpp16<6>(v177_acc, v184_lin, v170_data);
            tensorforge::fmacdpp16<7>(v177_acc, v184_lin, v171_data);
            tensorforge::fmacdpp16<8>(v178_acc, v184_lin, v160_data);
            tensorforge::fmacdpp16<9>(v178_acc, v184_lin, v161_data);
            tensorforge::fmacdpp16<10>(v178_acc, v184_lin, v162_data);
            tensorforge::fmacdpp16<11>(v178_acc, v184_lin, v163_data);
            tensorforge::fmacdpp16<12>(v178_acc, v184_lin, v164_data);
            tensorforge::fmacdpp16<13>(v178_acc, v184_lin, v165_data);
            tensorforge::fmacdpp16<14>(v178_acc, v184_lin, v166_data);
            tensorforge::fmacdpp16<15>(v178_acc, v184_lin, v167_data);
            float v185_lin = r7[5];
            tensorforge::fmacdpp16<0>(v178_acc, v185_lin, v168_data);
            tensorforge::fmacdpp16<1>(v178_acc, v185_lin, v169_data);
            tensorforge::fmacdpp16<2>(v178_acc, v185_lin, v170_data);
            tensorforge::fmacdpp16<3>(v178_acc, v185_lin, v171_data);
            tensorforge::fmacdpp16<4>(v179_acc, v185_lin, v160_data);
            tensorforge::fmacdpp16<5>(v179_acc, v185_lin, v161_data);
            tensorforge::fmacdpp16<6>(v179_acc, v185_lin, v162_data);
            tensorforge::fmacdpp16<7>(v179_acc, v185_lin, v163_data);
            tensorforge::fmacdpp16<8>(v179_acc, v185_lin, v164_data);
            tensorforge::fmacdpp16<9>(v179_acc, v185_lin, v165_data);
            tensorforge::fmacdpp16<10>(v179_acc, v185_lin, v166_data);
            tensorforge::fmacdpp16<11>(v179_acc, v185_lin, v167_data);
            tensorforge::fmacdpp16<12>(v179_acc, v185_lin, v168_data);
            tensorforge::fmacdpp16<13>(v179_acc, v185_lin, v169_data);
            tensorforge::fmacdpp16<14>(v179_acc, v185_lin, v170_data);
            tensorforge::fmacdpp16<15>(v179_acc, v185_lin, v171_data);
            ir8[0] = v172_acc;
            ir8[1] = v173_acc;
            ir8[2] = v174_acc;
            ir8[3] = v175_acc;
            ir8[4] = v176_acc;
            ir8[5] = v177_acc;
            ir8[6] = v178_acc;
            ir8[7] = v179_acc;
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v190_n1 = 0; v190_n1 < 8; ++v190_n1) {
                int32_t v191_a = 0 + v190_n1;
                float v193_data = ir8[v190_n1];
                int32_t v194_a = 0 + v190_n1;
                float v196_data = r5[v190_n1];
                int32_t v198_a = 0 + v190_n1;
                r8[v190_n1] = (v196_data + v193_data);
              }
            }
          }
          float r10[8]{};
          {
            // r10 = load{g>r}(glb_m8);
            float v0 = glb_m8[0 + threadIdx.x * 1];
            r10[0] = v0;
            float v16 = glb_m8[16 + threadIdx.x * 1];
            r10[1] = v16;
            float v32 = glb_m8[32 + threadIdx.x * 1];
            r10[2] = v32;
            float v48 = glb_m8[48 + threadIdx.x * 1];
            r10[3] = v48;
            float v64 = glb_m8[64 + threadIdx.x * 1];
            r10[4] = v64;
            float v80 = glb_m8[80 + threadIdx.x * 1];
            r10[5] = v80;
          }
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          {
            // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir11[8]{};
            float v202_data = r9[0];
            float v203_data = r9[1];
            float v204_data = r9[2];
            float v205_data = r9[3];
            float v206_data = r9[4];
            float v207_data = r9[5];
            float v208_data = r9[6];
            float v209_data = r9[7];
            float v210_data = r9[8];
            float v211_data = r9[9];
            float v212_data = r9[10];
            float v213_data = r9[11];
            float v214_acc{};
            float v215_acc{};
            float v216_acc{};
            float v217_acc{};
            float v218_acc{};
            float v219_acc{};
            float v220_acc{};
            float v221_acc{};
            float v222_lin = r10[0];
            tensorforge::fmacdpp16<0>(v214_acc, v222_lin, v202_data);
            tensorforge::fmacdpp16<1>(v214_acc, v222_lin, v203_data);
            tensorforge::fmacdpp16<2>(v214_acc, v222_lin, v204_data);
            tensorforge::fmacdpp16<3>(v214_acc, v222_lin, v205_data);
            tensorforge::fmacdpp16<4>(v214_acc, v222_lin, v206_data);
            tensorforge::fmacdpp16<5>(v214_acc, v222_lin, v207_data);
            tensorforge::fmacdpp16<6>(v214_acc, v222_lin, v208_data);
            tensorforge::fmacdpp16<7>(v214_acc, v222_lin, v209_data);
            tensorforge::fmacdpp16<8>(v214_acc, v222_lin, v210_data);
            tensorforge::fmacdpp16<9>(v214_acc, v222_lin, v211_data);
            tensorforge::fmacdpp16<10>(v214_acc, v222_lin, v212_data);
            tensorforge::fmacdpp16<11>(v214_acc, v222_lin, v213_data);
            tensorforge::fmacdpp16<12>(v215_acc, v222_lin, v202_data);
            tensorforge::fmacdpp16<13>(v215_acc, v222_lin, v203_data);
            tensorforge::fmacdpp16<14>(v215_acc, v222_lin, v204_data);
            tensorforge::fmacdpp16<15>(v215_acc, v222_lin, v205_data);
            float v223_lin = r10[1];
            tensorforge::fmacdpp16<0>(v215_acc, v223_lin, v206_data);
            tensorforge::fmacdpp16<1>(v215_acc, v223_lin, v207_data);
            tensorforge::fmacdpp16<2>(v215_acc, v223_lin, v208_data);
            tensorforge::fmacdpp16<3>(v215_acc, v223_lin, v209_data);
            tensorforge::fmacdpp16<4>(v215_acc, v223_lin, v210_data);
            tensorforge::fmacdpp16<5>(v215_acc, v223_lin, v211_data);
            tensorforge::fmacdpp16<6>(v215_acc, v223_lin, v212_data);
            tensorforge::fmacdpp16<7>(v215_acc, v223_lin, v213_data);
            tensorforge::fmacdpp16<8>(v216_acc, v223_lin, v202_data);
            tensorforge::fmacdpp16<9>(v216_acc, v223_lin, v203_data);
            tensorforge::fmacdpp16<10>(v216_acc, v223_lin, v204_data);
            tensorforge::fmacdpp16<11>(v216_acc, v223_lin, v205_data);
            tensorforge::fmacdpp16<12>(v216_acc, v223_lin, v206_data);
            tensorforge::fmacdpp16<13>(v216_acc, v223_lin, v207_data);
            tensorforge::fmacdpp16<14>(v216_acc, v223_lin, v208_data);
            tensorforge::fmacdpp16<15>(v216_acc, v223_lin, v209_data);
            float v224_lin = r10[2];
            tensorforge::fmacdpp16<0>(v216_acc, v224_lin, v210_data);
            tensorforge::fmacdpp16<1>(v216_acc, v224_lin, v211_data);
            tensorforge::fmacdpp16<2>(v216_acc, v224_lin, v212_data);
            tensorforge::fmacdpp16<3>(v216_acc, v224_lin, v213_data);
            tensorforge::fmacdpp16<4>(v217_acc, v224_lin, v202_data);
            tensorforge::fmacdpp16<5>(v217_acc, v224_lin, v203_data);
            tensorforge::fmacdpp16<6>(v217_acc, v224_lin, v204_data);
            tensorforge::fmacdpp16<7>(v217_acc, v224_lin, v205_data);
            tensorforge::fmacdpp16<8>(v217_acc, v224_lin, v206_data);
            tensorforge::fmacdpp16<9>(v217_acc, v224_lin, v207_data);
            tensorforge::fmacdpp16<10>(v217_acc, v224_lin, v208_data);
            tensorforge::fmacdpp16<11>(v217_acc, v224_lin, v209_data);
            tensorforge::fmacdpp16<12>(v217_acc, v224_lin, v210_data);
            tensorforge::fmacdpp16<13>(v217_acc, v224_lin, v211_data);
            tensorforge::fmacdpp16<14>(v217_acc, v224_lin, v212_data);
            tensorforge::fmacdpp16<15>(v217_acc, v224_lin, v213_data);
            float v225_lin = r10[3];
            tensorforge::fmacdpp16<0>(v218_acc, v225_lin, v202_data);
            tensorforge::fmacdpp16<1>(v218_acc, v225_lin, v203_data);
            tensorforge::fmacdpp16<2>(v218_acc, v225_lin, v204_data);
            tensorforge::fmacdpp16<3>(v218_acc, v225_lin, v205_data);
            tensorforge::fmacdpp16<4>(v218_acc, v225_lin, v206_data);
            tensorforge::fmacdpp16<5>(v218_acc, v225_lin, v207_data);
            tensorforge::fmacdpp16<6>(v218_acc, v225_lin, v208_data);
            tensorforge::fmacdpp16<7>(v218_acc, v225_lin, v209_data);
            tensorforge::fmacdpp16<8>(v218_acc, v225_lin, v210_data);
            tensorforge::fmacdpp16<9>(v218_acc, v225_lin, v211_data);
            tensorforge::fmacdpp16<10>(v218_acc, v225_lin, v212_data);
            tensorforge::fmacdpp16<11>(v218_acc, v225_lin, v213_data);
            tensorforge::fmacdpp16<12>(v219_acc, v225_lin, v202_data);
            tensorforge::fmacdpp16<13>(v219_acc, v225_lin, v203_data);
            tensorforge::fmacdpp16<14>(v219_acc, v225_lin, v204_data);
            tensorforge::fmacdpp16<15>(v219_acc, v225_lin, v205_data);
            float v226_lin = r10[4];
            tensorforge::fmacdpp16<0>(v219_acc, v226_lin, v206_data);
            tensorforge::fmacdpp16<1>(v219_acc, v226_lin, v207_data);
            tensorforge::fmacdpp16<2>(v219_acc, v226_lin, v208_data);
            tensorforge::fmacdpp16<3>(v219_acc, v226_lin, v209_data);
            tensorforge::fmacdpp16<4>(v219_acc, v226_lin, v210_data);
            tensorforge::fmacdpp16<5>(v219_acc, v226_lin, v211_data);
            tensorforge::fmacdpp16<6>(v219_acc, v226_lin, v212_data);
            tensorforge::fmacdpp16<7>(v219_acc, v226_lin, v213_data);
            tensorforge::fmacdpp16<8>(v220_acc, v226_lin, v202_data);
            tensorforge::fmacdpp16<9>(v220_acc, v226_lin, v203_data);
            tensorforge::fmacdpp16<10>(v220_acc, v226_lin, v204_data);
            tensorforge::fmacdpp16<11>(v220_acc, v226_lin, v205_data);
            tensorforge::fmacdpp16<12>(v220_acc, v226_lin, v206_data);
            tensorforge::fmacdpp16<13>(v220_acc, v226_lin, v207_data);
            tensorforge::fmacdpp16<14>(v220_acc, v226_lin, v208_data);
            tensorforge::fmacdpp16<15>(v220_acc, v226_lin, v209_data);
            float v227_lin = r10[5];
            tensorforge::fmacdpp16<0>(v220_acc, v227_lin, v210_data);
            tensorforge::fmacdpp16<1>(v220_acc, v227_lin, v211_data);
            tensorforge::fmacdpp16<2>(v220_acc, v227_lin, v212_data);
            tensorforge::fmacdpp16<3>(v220_acc, v227_lin, v213_data);
            tensorforge::fmacdpp16<4>(v221_acc, v227_lin, v202_data);
            tensorforge::fmacdpp16<5>(v221_acc, v227_lin, v203_data);
            tensorforge::fmacdpp16<6>(v221_acc, v227_lin, v204_data);
            tensorforge::fmacdpp16<7>(v221_acc, v227_lin, v205_data);
            tensorforge::fmacdpp16<8>(v221_acc, v227_lin, v206_data);
            tensorforge::fmacdpp16<9>(v221_acc, v227_lin, v207_data);
            tensorforge::fmacdpp16<10>(v221_acc, v227_lin, v208_data);
            tensorforge::fmacdpp16<11>(v221_acc, v227_lin, v209_data);
            tensorforge::fmacdpp16<12>(v221_acc, v227_lin, v210_data);
            tensorforge::fmacdpp16<13>(v221_acc, v227_lin, v211_data);
            tensorforge::fmacdpp16<14>(v221_acc, v227_lin, v212_data);
            tensorforge::fmacdpp16<15>(v221_acc, v227_lin, v213_data);
            ir11[0] = v214_acc;
            ir11[1] = v215_acc;
            ir11[2] = v216_acc;
            ir11[3] = v217_acc;
            ir11[4] = v218_acc;
            ir11[5] = v219_acc;
            ir11[6] = v220_acc;
            ir11[7] = v221_acc;
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v232_n1 = 0; v232_n1 < 8; ++v232_n1) {
                int32_t v233_a = 0 + v232_n1;
                float v235_data = ir11[v232_n1];
                int32_t v236_a = 0 + v232_n1;
                float v238_data = r8[v232_n1];
                int32_t v240_a = 0 + v232_n1;
                r11[v232_n1] = (v238_data + v235_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v246_i1 = 0; v246_i1 < 8; ++v246_i1) {
              int32_t v247_a = 0 + v246_i1;
              float v249_data = r11[v246_i1];
              int32_t v256_a = v3_lead + (v246_i1 * 12);
              glb_m0[v256_a] = v249_data;
            }
          }
          ;
        }
      }
    }
  }
}

