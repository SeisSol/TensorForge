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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 12; ++v4_i1) {
              int32_t v10_a = v4_i1 * 12;
              int32_t v11_a = v2_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
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
          int32_t v23_lead = threadIdx.x % 16;
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              int32_t v31_a = v25_i1 * 12;
              int32_t v32_a = v23_lead + v31_a;
              float v40_data = __builtin_nontemporal_load(&glb_m3[(v23_lead + v31_a)]);
              int32_t v41_a = 0 + v25_i1;
              r3[v41_a] = v40_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          auto& ir2 = r2;
          float v42_data = r0[0];
          float v43_data = r0[1];
          float v44_data = r0[2];
          float v45_data = r0[3];
          float v46_data = r0[4];
          float v47_data = r0[5];
          float v48_data = r0[6];
          float v49_data = r0[7];
          float v50_data = r0[8];
          float v51_data = r0[9];
          float v52_data = r0[10];
          float v53_data = r0[11];
          float v54_acc{};
          float v55_acc{};
          float v56_acc{};
          float v57_acc{};
          float v58_acc{};
          float v59_acc{};
          float v60_acc{};
          float v61_acc{};
          float v62_lin = r1[0];
          tensorforge::fmacdpp16<0>(v54_acc, v62_lin, v42_data);
          tensorforge::fmacdpp16<1>(v54_acc, v62_lin, v43_data);
          tensorforge::fmacdpp16<2>(v54_acc, v62_lin, v44_data);
          tensorforge::fmacdpp16<3>(v54_acc, v62_lin, v45_data);
          tensorforge::fmacdpp16<4>(v54_acc, v62_lin, v46_data);
          tensorforge::fmacdpp16<5>(v54_acc, v62_lin, v47_data);
          tensorforge::fmacdpp16<6>(v54_acc, v62_lin, v48_data);
          tensorforge::fmacdpp16<7>(v54_acc, v62_lin, v49_data);
          tensorforge::fmacdpp16<8>(v54_acc, v62_lin, v50_data);
          tensorforge::fmacdpp16<9>(v54_acc, v62_lin, v51_data);
          tensorforge::fmacdpp16<10>(v54_acc, v62_lin, v52_data);
          tensorforge::fmacdpp16<11>(v54_acc, v62_lin, v53_data);
          tensorforge::fmacdpp16<12>(v55_acc, v62_lin, v42_data);
          tensorforge::fmacdpp16<13>(v55_acc, v62_lin, v43_data);
          tensorforge::fmacdpp16<14>(v55_acc, v62_lin, v44_data);
          tensorforge::fmacdpp16<15>(v55_acc, v62_lin, v45_data);
          float v63_lin = r1[1];
          tensorforge::fmacdpp16<0>(v55_acc, v63_lin, v46_data);
          tensorforge::fmacdpp16<1>(v55_acc, v63_lin, v47_data);
          tensorforge::fmacdpp16<2>(v55_acc, v63_lin, v48_data);
          tensorforge::fmacdpp16<3>(v55_acc, v63_lin, v49_data);
          tensorforge::fmacdpp16<4>(v55_acc, v63_lin, v50_data);
          tensorforge::fmacdpp16<5>(v55_acc, v63_lin, v51_data);
          tensorforge::fmacdpp16<6>(v55_acc, v63_lin, v52_data);
          tensorforge::fmacdpp16<7>(v55_acc, v63_lin, v53_data);
          tensorforge::fmacdpp16<8>(v56_acc, v63_lin, v42_data);
          tensorforge::fmacdpp16<9>(v56_acc, v63_lin, v43_data);
          tensorforge::fmacdpp16<10>(v56_acc, v63_lin, v44_data);
          tensorforge::fmacdpp16<11>(v56_acc, v63_lin, v45_data);
          tensorforge::fmacdpp16<12>(v56_acc, v63_lin, v46_data);
          tensorforge::fmacdpp16<13>(v56_acc, v63_lin, v47_data);
          tensorforge::fmacdpp16<14>(v56_acc, v63_lin, v48_data);
          tensorforge::fmacdpp16<15>(v56_acc, v63_lin, v49_data);
          float v64_lin = r1[2];
          tensorforge::fmacdpp16<0>(v56_acc, v64_lin, v50_data);
          tensorforge::fmacdpp16<1>(v56_acc, v64_lin, v51_data);
          tensorforge::fmacdpp16<2>(v56_acc, v64_lin, v52_data);
          tensorforge::fmacdpp16<3>(v56_acc, v64_lin, v53_data);
          tensorforge::fmacdpp16<4>(v57_acc, v64_lin, v42_data);
          tensorforge::fmacdpp16<5>(v57_acc, v64_lin, v43_data);
          tensorforge::fmacdpp16<6>(v57_acc, v64_lin, v44_data);
          tensorforge::fmacdpp16<7>(v57_acc, v64_lin, v45_data);
          tensorforge::fmacdpp16<8>(v57_acc, v64_lin, v46_data);
          tensorforge::fmacdpp16<9>(v57_acc, v64_lin, v47_data);
          tensorforge::fmacdpp16<10>(v57_acc, v64_lin, v48_data);
          tensorforge::fmacdpp16<11>(v57_acc, v64_lin, v49_data);
          tensorforge::fmacdpp16<12>(v57_acc, v64_lin, v50_data);
          tensorforge::fmacdpp16<13>(v57_acc, v64_lin, v51_data);
          tensorforge::fmacdpp16<14>(v57_acc, v64_lin, v52_data);
          tensorforge::fmacdpp16<15>(v57_acc, v64_lin, v53_data);
          float v65_lin = r1[3];
          tensorforge::fmacdpp16<0>(v58_acc, v65_lin, v42_data);
          tensorforge::fmacdpp16<1>(v58_acc, v65_lin, v43_data);
          tensorforge::fmacdpp16<2>(v58_acc, v65_lin, v44_data);
          tensorforge::fmacdpp16<3>(v58_acc, v65_lin, v45_data);
          tensorforge::fmacdpp16<4>(v58_acc, v65_lin, v46_data);
          tensorforge::fmacdpp16<5>(v58_acc, v65_lin, v47_data);
          tensorforge::fmacdpp16<6>(v58_acc, v65_lin, v48_data);
          tensorforge::fmacdpp16<7>(v58_acc, v65_lin, v49_data);
          tensorforge::fmacdpp16<8>(v58_acc, v65_lin, v50_data);
          tensorforge::fmacdpp16<9>(v58_acc, v65_lin, v51_data);
          tensorforge::fmacdpp16<10>(v58_acc, v65_lin, v52_data);
          tensorforge::fmacdpp16<11>(v58_acc, v65_lin, v53_data);
          tensorforge::fmacdpp16<12>(v59_acc, v65_lin, v42_data);
          tensorforge::fmacdpp16<13>(v59_acc, v65_lin, v43_data);
          tensorforge::fmacdpp16<14>(v59_acc, v65_lin, v44_data);
          tensorforge::fmacdpp16<15>(v59_acc, v65_lin, v45_data);
          float v66_lin = r1[4];
          tensorforge::fmacdpp16<0>(v59_acc, v66_lin, v46_data);
          tensorforge::fmacdpp16<1>(v59_acc, v66_lin, v47_data);
          tensorforge::fmacdpp16<2>(v59_acc, v66_lin, v48_data);
          tensorforge::fmacdpp16<3>(v59_acc, v66_lin, v49_data);
          tensorforge::fmacdpp16<4>(v59_acc, v66_lin, v50_data);
          tensorforge::fmacdpp16<5>(v59_acc, v66_lin, v51_data);
          tensorforge::fmacdpp16<6>(v59_acc, v66_lin, v52_data);
          tensorforge::fmacdpp16<7>(v59_acc, v66_lin, v53_data);
          tensorforge::fmacdpp16<8>(v60_acc, v66_lin, v42_data);
          tensorforge::fmacdpp16<9>(v60_acc, v66_lin, v43_data);
          tensorforge::fmacdpp16<10>(v60_acc, v66_lin, v44_data);
          tensorforge::fmacdpp16<11>(v60_acc, v66_lin, v45_data);
          tensorforge::fmacdpp16<12>(v60_acc, v66_lin, v46_data);
          tensorforge::fmacdpp16<13>(v60_acc, v66_lin, v47_data);
          tensorforge::fmacdpp16<14>(v60_acc, v66_lin, v48_data);
          tensorforge::fmacdpp16<15>(v60_acc, v66_lin, v49_data);
          float v67_lin = r1[5];
          tensorforge::fmacdpp16<0>(v60_acc, v67_lin, v50_data);
          tensorforge::fmacdpp16<1>(v60_acc, v67_lin, v51_data);
          tensorforge::fmacdpp16<2>(v60_acc, v67_lin, v52_data);
          tensorforge::fmacdpp16<3>(v60_acc, v67_lin, v53_data);
          tensorforge::fmacdpp16<4>(v61_acc, v67_lin, v42_data);
          tensorforge::fmacdpp16<5>(v61_acc, v67_lin, v43_data);
          tensorforge::fmacdpp16<6>(v61_acc, v67_lin, v44_data);
          tensorforge::fmacdpp16<7>(v61_acc, v67_lin, v45_data);
          tensorforge::fmacdpp16<8>(v61_acc, v67_lin, v46_data);
          tensorforge::fmacdpp16<9>(v61_acc, v67_lin, v47_data);
          tensorforge::fmacdpp16<10>(v61_acc, v67_lin, v48_data);
          tensorforge::fmacdpp16<11>(v61_acc, v67_lin, v49_data);
          tensorforge::fmacdpp16<12>(v61_acc, v67_lin, v50_data);
          tensorforge::fmacdpp16<13>(v61_acc, v67_lin, v51_data);
          tensorforge::fmacdpp16<14>(v61_acc, v67_lin, v52_data);
          tensorforge::fmacdpp16<15>(v61_acc, v67_lin, v53_data);
          ir2[0] = v54_acc;
          ir2[1] = v55_acc;
          ir2[2] = v56_acc;
          ir2[3] = v57_acc;
          ir2[4] = v58_acc;
          ir2[5] = v59_acc;
          ir2[6] = v60_acc;
          ir2[7] = v61_acc;
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
          int32_t v70_lead = threadIdx.x % 16;
          if (v70_lead < 12) {
            #pragma unroll
            for (int32_t v72_i1 = 0; v72_i1 < 12; ++v72_i1) {
              int32_t v78_a = v72_i1 * 12;
              int32_t v79_a = v70_lead + v78_a;
              float v87_data = __builtin_nontemporal_load(&glb_m5[(v70_lead + v78_a)]);
              int32_t v88_a = 0 + v72_i1;
              r6[v88_a] = v87_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          {
            // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir5[8]{};
            float v89_data = r3[0];
            float v90_data = r3[1];
            float v91_data = r3[2];
            float v92_data = r3[3];
            float v93_data = r3[4];
            float v94_data = r3[5];
            float v95_data = r3[6];
            float v96_data = r3[7];
            float v97_data = r3[8];
            float v98_data = r3[9];
            float v99_data = r3[10];
            float v100_data = r3[11];
            float v101_acc{};
            float v102_acc{};
            float v103_acc{};
            float v104_acc{};
            float v105_acc{};
            float v106_acc{};
            float v107_acc{};
            float v108_acc{};
            float v109_lin = r4[0];
            tensorforge::fmacdpp16<0>(v101_acc, v109_lin, v89_data);
            tensorforge::fmacdpp16<1>(v101_acc, v109_lin, v90_data);
            tensorforge::fmacdpp16<2>(v101_acc, v109_lin, v91_data);
            tensorforge::fmacdpp16<3>(v101_acc, v109_lin, v92_data);
            tensorforge::fmacdpp16<4>(v101_acc, v109_lin, v93_data);
            tensorforge::fmacdpp16<5>(v101_acc, v109_lin, v94_data);
            tensorforge::fmacdpp16<6>(v101_acc, v109_lin, v95_data);
            tensorforge::fmacdpp16<7>(v101_acc, v109_lin, v96_data);
            tensorforge::fmacdpp16<8>(v101_acc, v109_lin, v97_data);
            tensorforge::fmacdpp16<9>(v101_acc, v109_lin, v98_data);
            tensorforge::fmacdpp16<10>(v101_acc, v109_lin, v99_data);
            tensorforge::fmacdpp16<11>(v101_acc, v109_lin, v100_data);
            tensorforge::fmacdpp16<12>(v102_acc, v109_lin, v89_data);
            tensorforge::fmacdpp16<13>(v102_acc, v109_lin, v90_data);
            tensorforge::fmacdpp16<14>(v102_acc, v109_lin, v91_data);
            tensorforge::fmacdpp16<15>(v102_acc, v109_lin, v92_data);
            float v110_lin = r4[1];
            tensorforge::fmacdpp16<0>(v102_acc, v110_lin, v93_data);
            tensorforge::fmacdpp16<1>(v102_acc, v110_lin, v94_data);
            tensorforge::fmacdpp16<2>(v102_acc, v110_lin, v95_data);
            tensorforge::fmacdpp16<3>(v102_acc, v110_lin, v96_data);
            tensorforge::fmacdpp16<4>(v102_acc, v110_lin, v97_data);
            tensorforge::fmacdpp16<5>(v102_acc, v110_lin, v98_data);
            tensorforge::fmacdpp16<6>(v102_acc, v110_lin, v99_data);
            tensorforge::fmacdpp16<7>(v102_acc, v110_lin, v100_data);
            tensorforge::fmacdpp16<8>(v103_acc, v110_lin, v89_data);
            tensorforge::fmacdpp16<9>(v103_acc, v110_lin, v90_data);
            tensorforge::fmacdpp16<10>(v103_acc, v110_lin, v91_data);
            tensorforge::fmacdpp16<11>(v103_acc, v110_lin, v92_data);
            tensorforge::fmacdpp16<12>(v103_acc, v110_lin, v93_data);
            tensorforge::fmacdpp16<13>(v103_acc, v110_lin, v94_data);
            tensorforge::fmacdpp16<14>(v103_acc, v110_lin, v95_data);
            tensorforge::fmacdpp16<15>(v103_acc, v110_lin, v96_data);
            float v111_lin = r4[2];
            tensorforge::fmacdpp16<0>(v103_acc, v111_lin, v97_data);
            tensorforge::fmacdpp16<1>(v103_acc, v111_lin, v98_data);
            tensorforge::fmacdpp16<2>(v103_acc, v111_lin, v99_data);
            tensorforge::fmacdpp16<3>(v103_acc, v111_lin, v100_data);
            tensorforge::fmacdpp16<4>(v104_acc, v111_lin, v89_data);
            tensorforge::fmacdpp16<5>(v104_acc, v111_lin, v90_data);
            tensorforge::fmacdpp16<6>(v104_acc, v111_lin, v91_data);
            tensorforge::fmacdpp16<7>(v104_acc, v111_lin, v92_data);
            tensorforge::fmacdpp16<8>(v104_acc, v111_lin, v93_data);
            tensorforge::fmacdpp16<9>(v104_acc, v111_lin, v94_data);
            tensorforge::fmacdpp16<10>(v104_acc, v111_lin, v95_data);
            tensorforge::fmacdpp16<11>(v104_acc, v111_lin, v96_data);
            tensorforge::fmacdpp16<12>(v104_acc, v111_lin, v97_data);
            tensorforge::fmacdpp16<13>(v104_acc, v111_lin, v98_data);
            tensorforge::fmacdpp16<14>(v104_acc, v111_lin, v99_data);
            tensorforge::fmacdpp16<15>(v104_acc, v111_lin, v100_data);
            float v112_lin = r4[3];
            tensorforge::fmacdpp16<0>(v105_acc, v112_lin, v89_data);
            tensorforge::fmacdpp16<1>(v105_acc, v112_lin, v90_data);
            tensorforge::fmacdpp16<2>(v105_acc, v112_lin, v91_data);
            tensorforge::fmacdpp16<3>(v105_acc, v112_lin, v92_data);
            tensorforge::fmacdpp16<4>(v105_acc, v112_lin, v93_data);
            tensorforge::fmacdpp16<5>(v105_acc, v112_lin, v94_data);
            tensorforge::fmacdpp16<6>(v105_acc, v112_lin, v95_data);
            tensorforge::fmacdpp16<7>(v105_acc, v112_lin, v96_data);
            tensorforge::fmacdpp16<8>(v105_acc, v112_lin, v97_data);
            tensorforge::fmacdpp16<9>(v105_acc, v112_lin, v98_data);
            tensorforge::fmacdpp16<10>(v105_acc, v112_lin, v99_data);
            tensorforge::fmacdpp16<11>(v105_acc, v112_lin, v100_data);
            tensorforge::fmacdpp16<12>(v106_acc, v112_lin, v89_data);
            tensorforge::fmacdpp16<13>(v106_acc, v112_lin, v90_data);
            tensorforge::fmacdpp16<14>(v106_acc, v112_lin, v91_data);
            tensorforge::fmacdpp16<15>(v106_acc, v112_lin, v92_data);
            float v113_lin = r4[4];
            tensorforge::fmacdpp16<0>(v106_acc, v113_lin, v93_data);
            tensorforge::fmacdpp16<1>(v106_acc, v113_lin, v94_data);
            tensorforge::fmacdpp16<2>(v106_acc, v113_lin, v95_data);
            tensorforge::fmacdpp16<3>(v106_acc, v113_lin, v96_data);
            tensorforge::fmacdpp16<4>(v106_acc, v113_lin, v97_data);
            tensorforge::fmacdpp16<5>(v106_acc, v113_lin, v98_data);
            tensorforge::fmacdpp16<6>(v106_acc, v113_lin, v99_data);
            tensorforge::fmacdpp16<7>(v106_acc, v113_lin, v100_data);
            tensorforge::fmacdpp16<8>(v107_acc, v113_lin, v89_data);
            tensorforge::fmacdpp16<9>(v107_acc, v113_lin, v90_data);
            tensorforge::fmacdpp16<10>(v107_acc, v113_lin, v91_data);
            tensorforge::fmacdpp16<11>(v107_acc, v113_lin, v92_data);
            tensorforge::fmacdpp16<12>(v107_acc, v113_lin, v93_data);
            tensorforge::fmacdpp16<13>(v107_acc, v113_lin, v94_data);
            tensorforge::fmacdpp16<14>(v107_acc, v113_lin, v95_data);
            tensorforge::fmacdpp16<15>(v107_acc, v113_lin, v96_data);
            float v114_lin = r4[5];
            tensorforge::fmacdpp16<0>(v107_acc, v114_lin, v97_data);
            tensorforge::fmacdpp16<1>(v107_acc, v114_lin, v98_data);
            tensorforge::fmacdpp16<2>(v107_acc, v114_lin, v99_data);
            tensorforge::fmacdpp16<3>(v107_acc, v114_lin, v100_data);
            tensorforge::fmacdpp16<4>(v108_acc, v114_lin, v89_data);
            tensorforge::fmacdpp16<5>(v108_acc, v114_lin, v90_data);
            tensorforge::fmacdpp16<6>(v108_acc, v114_lin, v91_data);
            tensorforge::fmacdpp16<7>(v108_acc, v114_lin, v92_data);
            tensorforge::fmacdpp16<8>(v108_acc, v114_lin, v93_data);
            tensorforge::fmacdpp16<9>(v108_acc, v114_lin, v94_data);
            tensorforge::fmacdpp16<10>(v108_acc, v114_lin, v95_data);
            tensorforge::fmacdpp16<11>(v108_acc, v114_lin, v96_data);
            tensorforge::fmacdpp16<12>(v108_acc, v114_lin, v97_data);
            tensorforge::fmacdpp16<13>(v108_acc, v114_lin, v98_data);
            tensorforge::fmacdpp16<14>(v108_acc, v114_lin, v99_data);
            tensorforge::fmacdpp16<15>(v108_acc, v114_lin, v100_data);
            ir5[0] = v101_acc;
            ir5[1] = v102_acc;
            ir5[2] = v103_acc;
            ir5[3] = v104_acc;
            ir5[4] = v105_acc;
            ir5[5] = v106_acc;
            ir5[6] = v107_acc;
            ir5[7] = v108_acc;
            if ((threadIdx.x % 16) < 12) {
              #pragma unroll
              for (int32_t v119_n1 = 0; v119_n1 < 8; ++v119_n1) {
                int32_t v120_a = 0 + v119_n1;
                float v122_data = ir5[v119_n1];
                int32_t v123_a = 0 + v119_n1;
                float v125_data = r2[v119_n1];
                int32_t v127_a = 0 + v119_n1;
                r5[v119_n1] = (v125_data + v122_data);
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
          int32_t v131_lead = threadIdx.x % 16;
          if (v131_lead < 12) {
            #pragma unroll
            for (int32_t v133_i1 = 0; v133_i1 < 12; ++v133_i1) {
              int32_t v139_a = v133_i1 * 12;
              int32_t v140_a = v131_lead + v139_a;
              float v148_data = __builtin_nontemporal_load(&glb_m7[(v131_lead + v139_a)]);
              int32_t v149_a = 0 + v133_i1;
              r9[v149_a] = v148_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          {
            // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir8[8]{};
            float v150_data = r6[0];
            float v151_data = r6[1];
            float v152_data = r6[2];
            float v153_data = r6[3];
            float v154_data = r6[4];
            float v155_data = r6[5];
            float v156_data = r6[6];
            float v157_data = r6[7];
            float v158_data = r6[8];
            float v159_data = r6[9];
            float v160_data = r6[10];
            float v161_data = r6[11];
            float v162_acc{};
            float v163_acc{};
            float v164_acc{};
            float v165_acc{};
            float v166_acc{};
            float v167_acc{};
            float v168_acc{};
            float v169_acc{};
            float v170_lin = r7[0];
            tensorforge::fmacdpp16<0>(v162_acc, v170_lin, v150_data);
            tensorforge::fmacdpp16<1>(v162_acc, v170_lin, v151_data);
            tensorforge::fmacdpp16<2>(v162_acc, v170_lin, v152_data);
            tensorforge::fmacdpp16<3>(v162_acc, v170_lin, v153_data);
            tensorforge::fmacdpp16<4>(v162_acc, v170_lin, v154_data);
            tensorforge::fmacdpp16<5>(v162_acc, v170_lin, v155_data);
            tensorforge::fmacdpp16<6>(v162_acc, v170_lin, v156_data);
            tensorforge::fmacdpp16<7>(v162_acc, v170_lin, v157_data);
            tensorforge::fmacdpp16<8>(v162_acc, v170_lin, v158_data);
            tensorforge::fmacdpp16<9>(v162_acc, v170_lin, v159_data);
            tensorforge::fmacdpp16<10>(v162_acc, v170_lin, v160_data);
            tensorforge::fmacdpp16<11>(v162_acc, v170_lin, v161_data);
            tensorforge::fmacdpp16<12>(v163_acc, v170_lin, v150_data);
            tensorforge::fmacdpp16<13>(v163_acc, v170_lin, v151_data);
            tensorforge::fmacdpp16<14>(v163_acc, v170_lin, v152_data);
            tensorforge::fmacdpp16<15>(v163_acc, v170_lin, v153_data);
            float v171_lin = r7[1];
            tensorforge::fmacdpp16<0>(v163_acc, v171_lin, v154_data);
            tensorforge::fmacdpp16<1>(v163_acc, v171_lin, v155_data);
            tensorforge::fmacdpp16<2>(v163_acc, v171_lin, v156_data);
            tensorforge::fmacdpp16<3>(v163_acc, v171_lin, v157_data);
            tensorforge::fmacdpp16<4>(v163_acc, v171_lin, v158_data);
            tensorforge::fmacdpp16<5>(v163_acc, v171_lin, v159_data);
            tensorforge::fmacdpp16<6>(v163_acc, v171_lin, v160_data);
            tensorforge::fmacdpp16<7>(v163_acc, v171_lin, v161_data);
            tensorforge::fmacdpp16<8>(v164_acc, v171_lin, v150_data);
            tensorforge::fmacdpp16<9>(v164_acc, v171_lin, v151_data);
            tensorforge::fmacdpp16<10>(v164_acc, v171_lin, v152_data);
            tensorforge::fmacdpp16<11>(v164_acc, v171_lin, v153_data);
            tensorforge::fmacdpp16<12>(v164_acc, v171_lin, v154_data);
            tensorforge::fmacdpp16<13>(v164_acc, v171_lin, v155_data);
            tensorforge::fmacdpp16<14>(v164_acc, v171_lin, v156_data);
            tensorforge::fmacdpp16<15>(v164_acc, v171_lin, v157_data);
            float v172_lin = r7[2];
            tensorforge::fmacdpp16<0>(v164_acc, v172_lin, v158_data);
            tensorforge::fmacdpp16<1>(v164_acc, v172_lin, v159_data);
            tensorforge::fmacdpp16<2>(v164_acc, v172_lin, v160_data);
            tensorforge::fmacdpp16<3>(v164_acc, v172_lin, v161_data);
            tensorforge::fmacdpp16<4>(v165_acc, v172_lin, v150_data);
            tensorforge::fmacdpp16<5>(v165_acc, v172_lin, v151_data);
            tensorforge::fmacdpp16<6>(v165_acc, v172_lin, v152_data);
            tensorforge::fmacdpp16<7>(v165_acc, v172_lin, v153_data);
            tensorforge::fmacdpp16<8>(v165_acc, v172_lin, v154_data);
            tensorforge::fmacdpp16<9>(v165_acc, v172_lin, v155_data);
            tensorforge::fmacdpp16<10>(v165_acc, v172_lin, v156_data);
            tensorforge::fmacdpp16<11>(v165_acc, v172_lin, v157_data);
            tensorforge::fmacdpp16<12>(v165_acc, v172_lin, v158_data);
            tensorforge::fmacdpp16<13>(v165_acc, v172_lin, v159_data);
            tensorforge::fmacdpp16<14>(v165_acc, v172_lin, v160_data);
            tensorforge::fmacdpp16<15>(v165_acc, v172_lin, v161_data);
            float v173_lin = r7[3];
            tensorforge::fmacdpp16<0>(v166_acc, v173_lin, v150_data);
            tensorforge::fmacdpp16<1>(v166_acc, v173_lin, v151_data);
            tensorforge::fmacdpp16<2>(v166_acc, v173_lin, v152_data);
            tensorforge::fmacdpp16<3>(v166_acc, v173_lin, v153_data);
            tensorforge::fmacdpp16<4>(v166_acc, v173_lin, v154_data);
            tensorforge::fmacdpp16<5>(v166_acc, v173_lin, v155_data);
            tensorforge::fmacdpp16<6>(v166_acc, v173_lin, v156_data);
            tensorforge::fmacdpp16<7>(v166_acc, v173_lin, v157_data);
            tensorforge::fmacdpp16<8>(v166_acc, v173_lin, v158_data);
            tensorforge::fmacdpp16<9>(v166_acc, v173_lin, v159_data);
            tensorforge::fmacdpp16<10>(v166_acc, v173_lin, v160_data);
            tensorforge::fmacdpp16<11>(v166_acc, v173_lin, v161_data);
            tensorforge::fmacdpp16<12>(v167_acc, v173_lin, v150_data);
            tensorforge::fmacdpp16<13>(v167_acc, v173_lin, v151_data);
            tensorforge::fmacdpp16<14>(v167_acc, v173_lin, v152_data);
            tensorforge::fmacdpp16<15>(v167_acc, v173_lin, v153_data);
            float v174_lin = r7[4];
            tensorforge::fmacdpp16<0>(v167_acc, v174_lin, v154_data);
            tensorforge::fmacdpp16<1>(v167_acc, v174_lin, v155_data);
            tensorforge::fmacdpp16<2>(v167_acc, v174_lin, v156_data);
            tensorforge::fmacdpp16<3>(v167_acc, v174_lin, v157_data);
            tensorforge::fmacdpp16<4>(v167_acc, v174_lin, v158_data);
            tensorforge::fmacdpp16<5>(v167_acc, v174_lin, v159_data);
            tensorforge::fmacdpp16<6>(v167_acc, v174_lin, v160_data);
            tensorforge::fmacdpp16<7>(v167_acc, v174_lin, v161_data);
            tensorforge::fmacdpp16<8>(v168_acc, v174_lin, v150_data);
            tensorforge::fmacdpp16<9>(v168_acc, v174_lin, v151_data);
            tensorforge::fmacdpp16<10>(v168_acc, v174_lin, v152_data);
            tensorforge::fmacdpp16<11>(v168_acc, v174_lin, v153_data);
            tensorforge::fmacdpp16<12>(v168_acc, v174_lin, v154_data);
            tensorforge::fmacdpp16<13>(v168_acc, v174_lin, v155_data);
            tensorforge::fmacdpp16<14>(v168_acc, v174_lin, v156_data);
            tensorforge::fmacdpp16<15>(v168_acc, v174_lin, v157_data);
            float v175_lin = r7[5];
            tensorforge::fmacdpp16<0>(v168_acc, v175_lin, v158_data);
            tensorforge::fmacdpp16<1>(v168_acc, v175_lin, v159_data);
            tensorforge::fmacdpp16<2>(v168_acc, v175_lin, v160_data);
            tensorforge::fmacdpp16<3>(v168_acc, v175_lin, v161_data);
            tensorforge::fmacdpp16<4>(v169_acc, v175_lin, v150_data);
            tensorforge::fmacdpp16<5>(v169_acc, v175_lin, v151_data);
            tensorforge::fmacdpp16<6>(v169_acc, v175_lin, v152_data);
            tensorforge::fmacdpp16<7>(v169_acc, v175_lin, v153_data);
            tensorforge::fmacdpp16<8>(v169_acc, v175_lin, v154_data);
            tensorforge::fmacdpp16<9>(v169_acc, v175_lin, v155_data);
            tensorforge::fmacdpp16<10>(v169_acc, v175_lin, v156_data);
            tensorforge::fmacdpp16<11>(v169_acc, v175_lin, v157_data);
            tensorforge::fmacdpp16<12>(v169_acc, v175_lin, v158_data);
            tensorforge::fmacdpp16<13>(v169_acc, v175_lin, v159_data);
            tensorforge::fmacdpp16<14>(v169_acc, v175_lin, v160_data);
            tensorforge::fmacdpp16<15>(v169_acc, v175_lin, v161_data);
            ir8[0] = v162_acc;
            ir8[1] = v163_acc;
            ir8[2] = v164_acc;
            ir8[3] = v165_acc;
            ir8[4] = v166_acc;
            ir8[5] = v167_acc;
            ir8[6] = v168_acc;
            ir8[7] = v169_acc;
            if ((threadIdx.x % 16) < 12) {
              #pragma unroll
              for (int32_t v180_n1 = 0; v180_n1 < 8; ++v180_n1) {
                int32_t v181_a = 0 + v180_n1;
                float v183_data = ir8[v180_n1];
                int32_t v184_a = 0 + v180_n1;
                float v186_data = r5[v180_n1];
                int32_t v188_a = 0 + v180_n1;
                r8[v180_n1] = (v186_data + v183_data);
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
            float v190_data = r9[0];
            float v191_data = r9[1];
            float v192_data = r9[2];
            float v193_data = r9[3];
            float v194_data = r9[4];
            float v195_data = r9[5];
            float v196_data = r9[6];
            float v197_data = r9[7];
            float v198_data = r9[8];
            float v199_data = r9[9];
            float v200_data = r9[10];
            float v201_data = r9[11];
            float v202_acc{};
            float v203_acc{};
            float v204_acc{};
            float v205_acc{};
            float v206_acc{};
            float v207_acc{};
            float v208_acc{};
            float v209_acc{};
            float v210_lin = r10[0];
            tensorforge::fmacdpp16<0>(v202_acc, v210_lin, v190_data);
            tensorforge::fmacdpp16<1>(v202_acc, v210_lin, v191_data);
            tensorforge::fmacdpp16<2>(v202_acc, v210_lin, v192_data);
            tensorforge::fmacdpp16<3>(v202_acc, v210_lin, v193_data);
            tensorforge::fmacdpp16<4>(v202_acc, v210_lin, v194_data);
            tensorforge::fmacdpp16<5>(v202_acc, v210_lin, v195_data);
            tensorforge::fmacdpp16<6>(v202_acc, v210_lin, v196_data);
            tensorforge::fmacdpp16<7>(v202_acc, v210_lin, v197_data);
            tensorforge::fmacdpp16<8>(v202_acc, v210_lin, v198_data);
            tensorforge::fmacdpp16<9>(v202_acc, v210_lin, v199_data);
            tensorforge::fmacdpp16<10>(v202_acc, v210_lin, v200_data);
            tensorforge::fmacdpp16<11>(v202_acc, v210_lin, v201_data);
            tensorforge::fmacdpp16<12>(v203_acc, v210_lin, v190_data);
            tensorforge::fmacdpp16<13>(v203_acc, v210_lin, v191_data);
            tensorforge::fmacdpp16<14>(v203_acc, v210_lin, v192_data);
            tensorforge::fmacdpp16<15>(v203_acc, v210_lin, v193_data);
            float v211_lin = r10[1];
            tensorforge::fmacdpp16<0>(v203_acc, v211_lin, v194_data);
            tensorforge::fmacdpp16<1>(v203_acc, v211_lin, v195_data);
            tensorforge::fmacdpp16<2>(v203_acc, v211_lin, v196_data);
            tensorforge::fmacdpp16<3>(v203_acc, v211_lin, v197_data);
            tensorforge::fmacdpp16<4>(v203_acc, v211_lin, v198_data);
            tensorforge::fmacdpp16<5>(v203_acc, v211_lin, v199_data);
            tensorforge::fmacdpp16<6>(v203_acc, v211_lin, v200_data);
            tensorforge::fmacdpp16<7>(v203_acc, v211_lin, v201_data);
            tensorforge::fmacdpp16<8>(v204_acc, v211_lin, v190_data);
            tensorforge::fmacdpp16<9>(v204_acc, v211_lin, v191_data);
            tensorforge::fmacdpp16<10>(v204_acc, v211_lin, v192_data);
            tensorforge::fmacdpp16<11>(v204_acc, v211_lin, v193_data);
            tensorforge::fmacdpp16<12>(v204_acc, v211_lin, v194_data);
            tensorforge::fmacdpp16<13>(v204_acc, v211_lin, v195_data);
            tensorforge::fmacdpp16<14>(v204_acc, v211_lin, v196_data);
            tensorforge::fmacdpp16<15>(v204_acc, v211_lin, v197_data);
            float v212_lin = r10[2];
            tensorforge::fmacdpp16<0>(v204_acc, v212_lin, v198_data);
            tensorforge::fmacdpp16<1>(v204_acc, v212_lin, v199_data);
            tensorforge::fmacdpp16<2>(v204_acc, v212_lin, v200_data);
            tensorforge::fmacdpp16<3>(v204_acc, v212_lin, v201_data);
            tensorforge::fmacdpp16<4>(v205_acc, v212_lin, v190_data);
            tensorforge::fmacdpp16<5>(v205_acc, v212_lin, v191_data);
            tensorforge::fmacdpp16<6>(v205_acc, v212_lin, v192_data);
            tensorforge::fmacdpp16<7>(v205_acc, v212_lin, v193_data);
            tensorforge::fmacdpp16<8>(v205_acc, v212_lin, v194_data);
            tensorforge::fmacdpp16<9>(v205_acc, v212_lin, v195_data);
            tensorforge::fmacdpp16<10>(v205_acc, v212_lin, v196_data);
            tensorforge::fmacdpp16<11>(v205_acc, v212_lin, v197_data);
            tensorforge::fmacdpp16<12>(v205_acc, v212_lin, v198_data);
            tensorforge::fmacdpp16<13>(v205_acc, v212_lin, v199_data);
            tensorforge::fmacdpp16<14>(v205_acc, v212_lin, v200_data);
            tensorforge::fmacdpp16<15>(v205_acc, v212_lin, v201_data);
            float v213_lin = r10[3];
            tensorforge::fmacdpp16<0>(v206_acc, v213_lin, v190_data);
            tensorforge::fmacdpp16<1>(v206_acc, v213_lin, v191_data);
            tensorforge::fmacdpp16<2>(v206_acc, v213_lin, v192_data);
            tensorforge::fmacdpp16<3>(v206_acc, v213_lin, v193_data);
            tensorforge::fmacdpp16<4>(v206_acc, v213_lin, v194_data);
            tensorforge::fmacdpp16<5>(v206_acc, v213_lin, v195_data);
            tensorforge::fmacdpp16<6>(v206_acc, v213_lin, v196_data);
            tensorforge::fmacdpp16<7>(v206_acc, v213_lin, v197_data);
            tensorforge::fmacdpp16<8>(v206_acc, v213_lin, v198_data);
            tensorforge::fmacdpp16<9>(v206_acc, v213_lin, v199_data);
            tensorforge::fmacdpp16<10>(v206_acc, v213_lin, v200_data);
            tensorforge::fmacdpp16<11>(v206_acc, v213_lin, v201_data);
            tensorforge::fmacdpp16<12>(v207_acc, v213_lin, v190_data);
            tensorforge::fmacdpp16<13>(v207_acc, v213_lin, v191_data);
            tensorforge::fmacdpp16<14>(v207_acc, v213_lin, v192_data);
            tensorforge::fmacdpp16<15>(v207_acc, v213_lin, v193_data);
            float v214_lin = r10[4];
            tensorforge::fmacdpp16<0>(v207_acc, v214_lin, v194_data);
            tensorforge::fmacdpp16<1>(v207_acc, v214_lin, v195_data);
            tensorforge::fmacdpp16<2>(v207_acc, v214_lin, v196_data);
            tensorforge::fmacdpp16<3>(v207_acc, v214_lin, v197_data);
            tensorforge::fmacdpp16<4>(v207_acc, v214_lin, v198_data);
            tensorforge::fmacdpp16<5>(v207_acc, v214_lin, v199_data);
            tensorforge::fmacdpp16<6>(v207_acc, v214_lin, v200_data);
            tensorforge::fmacdpp16<7>(v207_acc, v214_lin, v201_data);
            tensorforge::fmacdpp16<8>(v208_acc, v214_lin, v190_data);
            tensorforge::fmacdpp16<9>(v208_acc, v214_lin, v191_data);
            tensorforge::fmacdpp16<10>(v208_acc, v214_lin, v192_data);
            tensorforge::fmacdpp16<11>(v208_acc, v214_lin, v193_data);
            tensorforge::fmacdpp16<12>(v208_acc, v214_lin, v194_data);
            tensorforge::fmacdpp16<13>(v208_acc, v214_lin, v195_data);
            tensorforge::fmacdpp16<14>(v208_acc, v214_lin, v196_data);
            tensorforge::fmacdpp16<15>(v208_acc, v214_lin, v197_data);
            float v215_lin = r10[5];
            tensorforge::fmacdpp16<0>(v208_acc, v215_lin, v198_data);
            tensorforge::fmacdpp16<1>(v208_acc, v215_lin, v199_data);
            tensorforge::fmacdpp16<2>(v208_acc, v215_lin, v200_data);
            tensorforge::fmacdpp16<3>(v208_acc, v215_lin, v201_data);
            tensorforge::fmacdpp16<4>(v209_acc, v215_lin, v190_data);
            tensorforge::fmacdpp16<5>(v209_acc, v215_lin, v191_data);
            tensorforge::fmacdpp16<6>(v209_acc, v215_lin, v192_data);
            tensorforge::fmacdpp16<7>(v209_acc, v215_lin, v193_data);
            tensorforge::fmacdpp16<8>(v209_acc, v215_lin, v194_data);
            tensorforge::fmacdpp16<9>(v209_acc, v215_lin, v195_data);
            tensorforge::fmacdpp16<10>(v209_acc, v215_lin, v196_data);
            tensorforge::fmacdpp16<11>(v209_acc, v215_lin, v197_data);
            tensorforge::fmacdpp16<12>(v209_acc, v215_lin, v198_data);
            tensorforge::fmacdpp16<13>(v209_acc, v215_lin, v199_data);
            tensorforge::fmacdpp16<14>(v209_acc, v215_lin, v200_data);
            tensorforge::fmacdpp16<15>(v209_acc, v215_lin, v201_data);
            ir11[0] = v202_acc;
            ir11[1] = v203_acc;
            ir11[2] = v204_acc;
            ir11[3] = v205_acc;
            ir11[4] = v206_acc;
            ir11[5] = v207_acc;
            ir11[6] = v208_acc;
            ir11[7] = v209_acc;
            if ((threadIdx.x % 16) < 12) {
              #pragma unroll
              for (int32_t v220_n1 = 0; v220_n1 < 8; ++v220_n1) {
                int32_t v221_a = 0 + v220_n1;
                float v223_data = ir11[v220_n1];
                int32_t v224_a = 0 + v220_n1;
                float v226_data = r8[v220_n1];
                int32_t v228_a = 0 + v220_n1;
                r11[v220_n1] = (v226_data + v223_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r11);
          int32_t v232_lead = threadIdx.x % 16;
          if (v232_lead < 12) {
            #pragma unroll
            for (int32_t v234_i1 = 0; v234_i1 < 8; ++v234_i1) {
              int32_t v235_a = 0 + v234_i1;
              float v237_data = r11[v234_i1];
              int32_t v244_a = v232_lead + (v234_i1 * 12);
              glb_m0[v244_a] = v237_data;
            }
          }
          ;
        }
      }
    }
  }
}

