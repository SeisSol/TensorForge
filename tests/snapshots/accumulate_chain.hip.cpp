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
          int32_t v13_lead = threadIdx.x % 16;
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 12; ++v15_i1) {
              int32_t v21_a = v15_i1 * 12;
              int32_t v22_a = v13_lead + v21_a;
              float v30_data = __builtin_nontemporal_load(&glb_m1[(v13_lead + v21_a)]);
              r0[v15_i1] = v30_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v33_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v33_lin;
          float v34_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v34_lin;
          float v35_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v35_lin;
          float v36_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v36_lin;
          float v37_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v37_lin;
          float v38_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v38_lin;
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v44_i1 = 0; v44_i1 < 12; ++v44_i1) {
              int32_t v50_a = v44_i1 * 12;
              int32_t v51_a = v13_lead + v50_a;
              float v59_data = __builtin_nontemporal_load(&glb_m3[(v13_lead + v50_a)]);
              r3[v44_i1] = v59_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float v62_data = r0[0];
          float v63_data = r0[1];
          float v64_data = r0[2];
          float v65_data = r0[3];
          float v66_data = r0[4];
          float v67_data = r0[5];
          float v68_data = r0[6];
          float v69_data = r0[7];
          float v70_data = r0[8];
          float v71_data = r0[9];
          float v72_data = r0[10];
          float v73_data = r0[11];
          float v74_acc{};
          float v75_acc{};
          float v76_acc{};
          float v77_acc{};
          float v78_acc{};
          float v79_acc{};
          float v80_acc{};
          float v81_acc{};
          float v82_lin = r1[0];
          tensorforge::fmacdpp16<0>(v74_acc, v82_lin, v62_data);
          tensorforge::fmacdpp16<1>(v74_acc, v82_lin, v63_data);
          tensorforge::fmacdpp16<2>(v74_acc, v82_lin, v64_data);
          tensorforge::fmacdpp16<3>(v74_acc, v82_lin, v65_data);
          tensorforge::fmacdpp16<4>(v74_acc, v82_lin, v66_data);
          tensorforge::fmacdpp16<5>(v74_acc, v82_lin, v67_data);
          tensorforge::fmacdpp16<6>(v74_acc, v82_lin, v68_data);
          tensorforge::fmacdpp16<7>(v74_acc, v82_lin, v69_data);
          tensorforge::fmacdpp16<8>(v74_acc, v82_lin, v70_data);
          tensorforge::fmacdpp16<9>(v74_acc, v82_lin, v71_data);
          tensorforge::fmacdpp16<10>(v74_acc, v82_lin, v72_data);
          tensorforge::fmacdpp16<11>(v74_acc, v82_lin, v73_data);
          tensorforge::fmacdpp16<12>(v75_acc, v82_lin, v62_data);
          tensorforge::fmacdpp16<13>(v75_acc, v82_lin, v63_data);
          tensorforge::fmacdpp16<14>(v75_acc, v82_lin, v64_data);
          tensorforge::fmacdpp16<15>(v75_acc, v82_lin, v65_data);
          float v83_lin = r1[1];
          tensorforge::fmacdpp16<0>(v75_acc, v83_lin, v66_data);
          tensorforge::fmacdpp16<1>(v75_acc, v83_lin, v67_data);
          tensorforge::fmacdpp16<2>(v75_acc, v83_lin, v68_data);
          tensorforge::fmacdpp16<3>(v75_acc, v83_lin, v69_data);
          tensorforge::fmacdpp16<4>(v75_acc, v83_lin, v70_data);
          tensorforge::fmacdpp16<5>(v75_acc, v83_lin, v71_data);
          tensorforge::fmacdpp16<6>(v75_acc, v83_lin, v72_data);
          tensorforge::fmacdpp16<7>(v75_acc, v83_lin, v73_data);
          tensorforge::fmacdpp16<8>(v76_acc, v83_lin, v62_data);
          tensorforge::fmacdpp16<9>(v76_acc, v83_lin, v63_data);
          tensorforge::fmacdpp16<10>(v76_acc, v83_lin, v64_data);
          tensorforge::fmacdpp16<11>(v76_acc, v83_lin, v65_data);
          tensorforge::fmacdpp16<12>(v76_acc, v83_lin, v66_data);
          tensorforge::fmacdpp16<13>(v76_acc, v83_lin, v67_data);
          tensorforge::fmacdpp16<14>(v76_acc, v83_lin, v68_data);
          tensorforge::fmacdpp16<15>(v76_acc, v83_lin, v69_data);
          float v84_lin = r1[2];
          tensorforge::fmacdpp16<0>(v76_acc, v84_lin, v70_data);
          tensorforge::fmacdpp16<1>(v76_acc, v84_lin, v71_data);
          tensorforge::fmacdpp16<2>(v76_acc, v84_lin, v72_data);
          tensorforge::fmacdpp16<3>(v76_acc, v84_lin, v73_data);
          tensorforge::fmacdpp16<4>(v77_acc, v84_lin, v62_data);
          tensorforge::fmacdpp16<5>(v77_acc, v84_lin, v63_data);
          tensorforge::fmacdpp16<6>(v77_acc, v84_lin, v64_data);
          tensorforge::fmacdpp16<7>(v77_acc, v84_lin, v65_data);
          tensorforge::fmacdpp16<8>(v77_acc, v84_lin, v66_data);
          tensorforge::fmacdpp16<9>(v77_acc, v84_lin, v67_data);
          tensorforge::fmacdpp16<10>(v77_acc, v84_lin, v68_data);
          tensorforge::fmacdpp16<11>(v77_acc, v84_lin, v69_data);
          tensorforge::fmacdpp16<12>(v77_acc, v84_lin, v70_data);
          tensorforge::fmacdpp16<13>(v77_acc, v84_lin, v71_data);
          tensorforge::fmacdpp16<14>(v77_acc, v84_lin, v72_data);
          tensorforge::fmacdpp16<15>(v77_acc, v84_lin, v73_data);
          float v85_lin = r1[3];
          tensorforge::fmacdpp16<0>(v78_acc, v85_lin, v62_data);
          tensorforge::fmacdpp16<1>(v78_acc, v85_lin, v63_data);
          tensorforge::fmacdpp16<2>(v78_acc, v85_lin, v64_data);
          tensorforge::fmacdpp16<3>(v78_acc, v85_lin, v65_data);
          tensorforge::fmacdpp16<4>(v78_acc, v85_lin, v66_data);
          tensorforge::fmacdpp16<5>(v78_acc, v85_lin, v67_data);
          tensorforge::fmacdpp16<6>(v78_acc, v85_lin, v68_data);
          tensorforge::fmacdpp16<7>(v78_acc, v85_lin, v69_data);
          tensorforge::fmacdpp16<8>(v78_acc, v85_lin, v70_data);
          tensorforge::fmacdpp16<9>(v78_acc, v85_lin, v71_data);
          tensorforge::fmacdpp16<10>(v78_acc, v85_lin, v72_data);
          tensorforge::fmacdpp16<11>(v78_acc, v85_lin, v73_data);
          tensorforge::fmacdpp16<12>(v79_acc, v85_lin, v62_data);
          tensorforge::fmacdpp16<13>(v79_acc, v85_lin, v63_data);
          tensorforge::fmacdpp16<14>(v79_acc, v85_lin, v64_data);
          tensorforge::fmacdpp16<15>(v79_acc, v85_lin, v65_data);
          float v86_lin = r1[4];
          tensorforge::fmacdpp16<0>(v79_acc, v86_lin, v66_data);
          tensorforge::fmacdpp16<1>(v79_acc, v86_lin, v67_data);
          tensorforge::fmacdpp16<2>(v79_acc, v86_lin, v68_data);
          tensorforge::fmacdpp16<3>(v79_acc, v86_lin, v69_data);
          tensorforge::fmacdpp16<4>(v79_acc, v86_lin, v70_data);
          tensorforge::fmacdpp16<5>(v79_acc, v86_lin, v71_data);
          tensorforge::fmacdpp16<6>(v79_acc, v86_lin, v72_data);
          tensorforge::fmacdpp16<7>(v79_acc, v86_lin, v73_data);
          tensorforge::fmacdpp16<8>(v80_acc, v86_lin, v62_data);
          tensorforge::fmacdpp16<9>(v80_acc, v86_lin, v63_data);
          tensorforge::fmacdpp16<10>(v80_acc, v86_lin, v64_data);
          tensorforge::fmacdpp16<11>(v80_acc, v86_lin, v65_data);
          tensorforge::fmacdpp16<12>(v80_acc, v86_lin, v66_data);
          tensorforge::fmacdpp16<13>(v80_acc, v86_lin, v67_data);
          tensorforge::fmacdpp16<14>(v80_acc, v86_lin, v68_data);
          tensorforge::fmacdpp16<15>(v80_acc, v86_lin, v69_data);
          float v87_lin = r1[5];
          tensorforge::fmacdpp16<0>(v80_acc, v87_lin, v70_data);
          tensorforge::fmacdpp16<1>(v80_acc, v87_lin, v71_data);
          tensorforge::fmacdpp16<2>(v80_acc, v87_lin, v72_data);
          tensorforge::fmacdpp16<3>(v80_acc, v87_lin, v73_data);
          tensorforge::fmacdpp16<4>(v81_acc, v87_lin, v62_data);
          tensorforge::fmacdpp16<5>(v81_acc, v87_lin, v63_data);
          tensorforge::fmacdpp16<6>(v81_acc, v87_lin, v64_data);
          tensorforge::fmacdpp16<7>(v81_acc, v87_lin, v65_data);
          tensorforge::fmacdpp16<8>(v81_acc, v87_lin, v66_data);
          tensorforge::fmacdpp16<9>(v81_acc, v87_lin, v67_data);
          tensorforge::fmacdpp16<10>(v81_acc, v87_lin, v68_data);
          tensorforge::fmacdpp16<11>(v81_acc, v87_lin, v69_data);
          tensorforge::fmacdpp16<12>(v81_acc, v87_lin, v70_data);
          tensorforge::fmacdpp16<13>(v81_acc, v87_lin, v71_data);
          tensorforge::fmacdpp16<14>(v81_acc, v87_lin, v72_data);
          tensorforge::fmacdpp16<15>(v81_acc, v87_lin, v73_data);
          r2[0] = v74_acc;
          r2[1] = v75_acc;
          r2[2] = v76_acc;
          r2[3] = v77_acc;
          r2[4] = v78_acc;
          r2[5] = v79_acc;
          r2[6] = v80_acc;
          r2[7] = v81_acc;
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v89_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v89_lin;
          float v90_lin = glb_m4[16 + threadIdx.x * 1];
          r4[1] = v90_lin;
          float v91_lin = glb_m4[32 + threadIdx.x * 1];
          r4[2] = v91_lin;
          float v92_lin = glb_m4[48 + threadIdx.x * 1];
          r4[3] = v92_lin;
          float v93_lin = glb_m4[64 + threadIdx.x * 1];
          r4[4] = v93_lin;
          float v94_lin = glb_m4[80 + threadIdx.x * 1];
          r4[5] = v94_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v100_i1 = 0; v100_i1 < 12; ++v100_i1) {
              int32_t v106_a = v100_i1 * 12;
              int32_t v107_a = v13_lead + v106_a;
              float v115_data = __builtin_nontemporal_load(&glb_m5[(v13_lead + v106_a)]);
              r6[v100_i1] = v115_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v119_data = r3[0];
          float v120_data = r3[1];
          float v121_data = r3[2];
          float v122_data = r3[3];
          float v123_data = r3[4];
          float v124_data = r3[5];
          float v125_data = r3[6];
          float v126_data = r3[7];
          float v127_data = r3[8];
          float v128_data = r3[9];
          float v129_data = r3[10];
          float v130_data = r3[11];
          float v131_acc{};
          float v132_acc{};
          float v133_acc{};
          float v134_acc{};
          float v135_acc{};
          float v136_acc{};
          float v137_acc{};
          float v138_acc{};
          float v139_lin = r4[0];
          tensorforge::fmacdpp16<0>(v131_acc, v139_lin, v119_data);
          tensorforge::fmacdpp16<1>(v131_acc, v139_lin, v120_data);
          tensorforge::fmacdpp16<2>(v131_acc, v139_lin, v121_data);
          tensorforge::fmacdpp16<3>(v131_acc, v139_lin, v122_data);
          tensorforge::fmacdpp16<4>(v131_acc, v139_lin, v123_data);
          tensorforge::fmacdpp16<5>(v131_acc, v139_lin, v124_data);
          tensorforge::fmacdpp16<6>(v131_acc, v139_lin, v125_data);
          tensorforge::fmacdpp16<7>(v131_acc, v139_lin, v126_data);
          tensorforge::fmacdpp16<8>(v131_acc, v139_lin, v127_data);
          tensorforge::fmacdpp16<9>(v131_acc, v139_lin, v128_data);
          tensorforge::fmacdpp16<10>(v131_acc, v139_lin, v129_data);
          tensorforge::fmacdpp16<11>(v131_acc, v139_lin, v130_data);
          tensorforge::fmacdpp16<12>(v132_acc, v139_lin, v119_data);
          tensorforge::fmacdpp16<13>(v132_acc, v139_lin, v120_data);
          tensorforge::fmacdpp16<14>(v132_acc, v139_lin, v121_data);
          tensorforge::fmacdpp16<15>(v132_acc, v139_lin, v122_data);
          float v140_lin = r4[1];
          tensorforge::fmacdpp16<0>(v132_acc, v140_lin, v123_data);
          tensorforge::fmacdpp16<1>(v132_acc, v140_lin, v124_data);
          tensorforge::fmacdpp16<2>(v132_acc, v140_lin, v125_data);
          tensorforge::fmacdpp16<3>(v132_acc, v140_lin, v126_data);
          tensorforge::fmacdpp16<4>(v132_acc, v140_lin, v127_data);
          tensorforge::fmacdpp16<5>(v132_acc, v140_lin, v128_data);
          tensorforge::fmacdpp16<6>(v132_acc, v140_lin, v129_data);
          tensorforge::fmacdpp16<7>(v132_acc, v140_lin, v130_data);
          tensorforge::fmacdpp16<8>(v133_acc, v140_lin, v119_data);
          tensorforge::fmacdpp16<9>(v133_acc, v140_lin, v120_data);
          tensorforge::fmacdpp16<10>(v133_acc, v140_lin, v121_data);
          tensorforge::fmacdpp16<11>(v133_acc, v140_lin, v122_data);
          tensorforge::fmacdpp16<12>(v133_acc, v140_lin, v123_data);
          tensorforge::fmacdpp16<13>(v133_acc, v140_lin, v124_data);
          tensorforge::fmacdpp16<14>(v133_acc, v140_lin, v125_data);
          tensorforge::fmacdpp16<15>(v133_acc, v140_lin, v126_data);
          float v141_lin = r4[2];
          tensorforge::fmacdpp16<0>(v133_acc, v141_lin, v127_data);
          tensorforge::fmacdpp16<1>(v133_acc, v141_lin, v128_data);
          tensorforge::fmacdpp16<2>(v133_acc, v141_lin, v129_data);
          tensorforge::fmacdpp16<3>(v133_acc, v141_lin, v130_data);
          tensorforge::fmacdpp16<4>(v134_acc, v141_lin, v119_data);
          tensorforge::fmacdpp16<5>(v134_acc, v141_lin, v120_data);
          tensorforge::fmacdpp16<6>(v134_acc, v141_lin, v121_data);
          tensorforge::fmacdpp16<7>(v134_acc, v141_lin, v122_data);
          tensorforge::fmacdpp16<8>(v134_acc, v141_lin, v123_data);
          tensorforge::fmacdpp16<9>(v134_acc, v141_lin, v124_data);
          tensorforge::fmacdpp16<10>(v134_acc, v141_lin, v125_data);
          tensorforge::fmacdpp16<11>(v134_acc, v141_lin, v126_data);
          tensorforge::fmacdpp16<12>(v134_acc, v141_lin, v127_data);
          tensorforge::fmacdpp16<13>(v134_acc, v141_lin, v128_data);
          tensorforge::fmacdpp16<14>(v134_acc, v141_lin, v129_data);
          tensorforge::fmacdpp16<15>(v134_acc, v141_lin, v130_data);
          float v142_lin = r4[3];
          tensorforge::fmacdpp16<0>(v135_acc, v142_lin, v119_data);
          tensorforge::fmacdpp16<1>(v135_acc, v142_lin, v120_data);
          tensorforge::fmacdpp16<2>(v135_acc, v142_lin, v121_data);
          tensorforge::fmacdpp16<3>(v135_acc, v142_lin, v122_data);
          tensorforge::fmacdpp16<4>(v135_acc, v142_lin, v123_data);
          tensorforge::fmacdpp16<5>(v135_acc, v142_lin, v124_data);
          tensorforge::fmacdpp16<6>(v135_acc, v142_lin, v125_data);
          tensorforge::fmacdpp16<7>(v135_acc, v142_lin, v126_data);
          tensorforge::fmacdpp16<8>(v135_acc, v142_lin, v127_data);
          tensorforge::fmacdpp16<9>(v135_acc, v142_lin, v128_data);
          tensorforge::fmacdpp16<10>(v135_acc, v142_lin, v129_data);
          tensorforge::fmacdpp16<11>(v135_acc, v142_lin, v130_data);
          tensorforge::fmacdpp16<12>(v136_acc, v142_lin, v119_data);
          tensorforge::fmacdpp16<13>(v136_acc, v142_lin, v120_data);
          tensorforge::fmacdpp16<14>(v136_acc, v142_lin, v121_data);
          tensorforge::fmacdpp16<15>(v136_acc, v142_lin, v122_data);
          float v143_lin = r4[4];
          tensorforge::fmacdpp16<0>(v136_acc, v143_lin, v123_data);
          tensorforge::fmacdpp16<1>(v136_acc, v143_lin, v124_data);
          tensorforge::fmacdpp16<2>(v136_acc, v143_lin, v125_data);
          tensorforge::fmacdpp16<3>(v136_acc, v143_lin, v126_data);
          tensorforge::fmacdpp16<4>(v136_acc, v143_lin, v127_data);
          tensorforge::fmacdpp16<5>(v136_acc, v143_lin, v128_data);
          tensorforge::fmacdpp16<6>(v136_acc, v143_lin, v129_data);
          tensorforge::fmacdpp16<7>(v136_acc, v143_lin, v130_data);
          tensorforge::fmacdpp16<8>(v137_acc, v143_lin, v119_data);
          tensorforge::fmacdpp16<9>(v137_acc, v143_lin, v120_data);
          tensorforge::fmacdpp16<10>(v137_acc, v143_lin, v121_data);
          tensorforge::fmacdpp16<11>(v137_acc, v143_lin, v122_data);
          tensorforge::fmacdpp16<12>(v137_acc, v143_lin, v123_data);
          tensorforge::fmacdpp16<13>(v137_acc, v143_lin, v124_data);
          tensorforge::fmacdpp16<14>(v137_acc, v143_lin, v125_data);
          tensorforge::fmacdpp16<15>(v137_acc, v143_lin, v126_data);
          float v144_lin = r4[5];
          tensorforge::fmacdpp16<0>(v137_acc, v144_lin, v127_data);
          tensorforge::fmacdpp16<1>(v137_acc, v144_lin, v128_data);
          tensorforge::fmacdpp16<2>(v137_acc, v144_lin, v129_data);
          tensorforge::fmacdpp16<3>(v137_acc, v144_lin, v130_data);
          tensorforge::fmacdpp16<4>(v138_acc, v144_lin, v119_data);
          tensorforge::fmacdpp16<5>(v138_acc, v144_lin, v120_data);
          tensorforge::fmacdpp16<6>(v138_acc, v144_lin, v121_data);
          tensorforge::fmacdpp16<7>(v138_acc, v144_lin, v122_data);
          tensorforge::fmacdpp16<8>(v138_acc, v144_lin, v123_data);
          tensorforge::fmacdpp16<9>(v138_acc, v144_lin, v124_data);
          tensorforge::fmacdpp16<10>(v138_acc, v144_lin, v125_data);
          tensorforge::fmacdpp16<11>(v138_acc, v144_lin, v126_data);
          tensorforge::fmacdpp16<12>(v138_acc, v144_lin, v127_data);
          tensorforge::fmacdpp16<13>(v138_acc, v144_lin, v128_data);
          tensorforge::fmacdpp16<14>(v138_acc, v144_lin, v129_data);
          tensorforge::fmacdpp16<15>(v138_acc, v144_lin, v130_data);
          ir5[0] = v131_acc;
          ir5[1] = v132_acc;
          ir5[2] = v133_acc;
          ir5[3] = v134_acc;
          ir5[4] = v135_acc;
          ir5[5] = v136_acc;
          ir5[6] = v137_acc;
          ir5[7] = v138_acc;
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v149_n1 = 0; v149_n1 < 8; ++v149_n1) {
              int32_t v150_a = 0 + v149_n1;
              float v152_data = ir5[v149_n1];
              int32_t v153_a = 0 + v149_n1;
              float v155_data = r2[v149_n1];
              r5[v149_n1] = (v155_data + v152_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v159_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v159_lin;
          float v160_lin = glb_m6[16 + threadIdx.x * 1];
          r7[1] = v160_lin;
          float v161_lin = glb_m6[32 + threadIdx.x * 1];
          r7[2] = v161_lin;
          float v162_lin = glb_m6[48 + threadIdx.x * 1];
          r7[3] = v162_lin;
          float v163_lin = glb_m6[64 + threadIdx.x * 1];
          r7[4] = v163_lin;
          float v164_lin = glb_m6[80 + threadIdx.x * 1];
          r7[5] = v164_lin;
          // wait(r6 = load{g>r}(glb_m5););
          float r9[12]{};
          // r9 = load{g>r}(glb_m7);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v170_i1 = 0; v170_i1 < 12; ++v170_i1) {
              int32_t v176_a = v170_i1 * 12;
              int32_t v177_a = v13_lead + v176_a;
              float v185_data = __builtin_nontemporal_load(&glb_m7[(v13_lead + v176_a)]);
              r9[v170_i1] = v185_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir8[8]{};
          float v189_data = r6[0];
          float v190_data = r6[1];
          float v191_data = r6[2];
          float v192_data = r6[3];
          float v193_data = r6[4];
          float v194_data = r6[5];
          float v195_data = r6[6];
          float v196_data = r6[7];
          float v197_data = r6[8];
          float v198_data = r6[9];
          float v199_data = r6[10];
          float v200_data = r6[11];
          float v201_acc{};
          float v202_acc{};
          float v203_acc{};
          float v204_acc{};
          float v205_acc{};
          float v206_acc{};
          float v207_acc{};
          float v208_acc{};
          float v209_lin = r7[0];
          tensorforge::fmacdpp16<0>(v201_acc, v209_lin, v189_data);
          tensorforge::fmacdpp16<1>(v201_acc, v209_lin, v190_data);
          tensorforge::fmacdpp16<2>(v201_acc, v209_lin, v191_data);
          tensorforge::fmacdpp16<3>(v201_acc, v209_lin, v192_data);
          tensorforge::fmacdpp16<4>(v201_acc, v209_lin, v193_data);
          tensorforge::fmacdpp16<5>(v201_acc, v209_lin, v194_data);
          tensorforge::fmacdpp16<6>(v201_acc, v209_lin, v195_data);
          tensorforge::fmacdpp16<7>(v201_acc, v209_lin, v196_data);
          tensorforge::fmacdpp16<8>(v201_acc, v209_lin, v197_data);
          tensorforge::fmacdpp16<9>(v201_acc, v209_lin, v198_data);
          tensorforge::fmacdpp16<10>(v201_acc, v209_lin, v199_data);
          tensorforge::fmacdpp16<11>(v201_acc, v209_lin, v200_data);
          tensorforge::fmacdpp16<12>(v202_acc, v209_lin, v189_data);
          tensorforge::fmacdpp16<13>(v202_acc, v209_lin, v190_data);
          tensorforge::fmacdpp16<14>(v202_acc, v209_lin, v191_data);
          tensorforge::fmacdpp16<15>(v202_acc, v209_lin, v192_data);
          float v210_lin = r7[1];
          tensorforge::fmacdpp16<0>(v202_acc, v210_lin, v193_data);
          tensorforge::fmacdpp16<1>(v202_acc, v210_lin, v194_data);
          tensorforge::fmacdpp16<2>(v202_acc, v210_lin, v195_data);
          tensorforge::fmacdpp16<3>(v202_acc, v210_lin, v196_data);
          tensorforge::fmacdpp16<4>(v202_acc, v210_lin, v197_data);
          tensorforge::fmacdpp16<5>(v202_acc, v210_lin, v198_data);
          tensorforge::fmacdpp16<6>(v202_acc, v210_lin, v199_data);
          tensorforge::fmacdpp16<7>(v202_acc, v210_lin, v200_data);
          tensorforge::fmacdpp16<8>(v203_acc, v210_lin, v189_data);
          tensorforge::fmacdpp16<9>(v203_acc, v210_lin, v190_data);
          tensorforge::fmacdpp16<10>(v203_acc, v210_lin, v191_data);
          tensorforge::fmacdpp16<11>(v203_acc, v210_lin, v192_data);
          tensorforge::fmacdpp16<12>(v203_acc, v210_lin, v193_data);
          tensorforge::fmacdpp16<13>(v203_acc, v210_lin, v194_data);
          tensorforge::fmacdpp16<14>(v203_acc, v210_lin, v195_data);
          tensorforge::fmacdpp16<15>(v203_acc, v210_lin, v196_data);
          float v211_lin = r7[2];
          tensorforge::fmacdpp16<0>(v203_acc, v211_lin, v197_data);
          tensorforge::fmacdpp16<1>(v203_acc, v211_lin, v198_data);
          tensorforge::fmacdpp16<2>(v203_acc, v211_lin, v199_data);
          tensorforge::fmacdpp16<3>(v203_acc, v211_lin, v200_data);
          tensorforge::fmacdpp16<4>(v204_acc, v211_lin, v189_data);
          tensorforge::fmacdpp16<5>(v204_acc, v211_lin, v190_data);
          tensorforge::fmacdpp16<6>(v204_acc, v211_lin, v191_data);
          tensorforge::fmacdpp16<7>(v204_acc, v211_lin, v192_data);
          tensorforge::fmacdpp16<8>(v204_acc, v211_lin, v193_data);
          tensorforge::fmacdpp16<9>(v204_acc, v211_lin, v194_data);
          tensorforge::fmacdpp16<10>(v204_acc, v211_lin, v195_data);
          tensorforge::fmacdpp16<11>(v204_acc, v211_lin, v196_data);
          tensorforge::fmacdpp16<12>(v204_acc, v211_lin, v197_data);
          tensorforge::fmacdpp16<13>(v204_acc, v211_lin, v198_data);
          tensorforge::fmacdpp16<14>(v204_acc, v211_lin, v199_data);
          tensorforge::fmacdpp16<15>(v204_acc, v211_lin, v200_data);
          float v212_lin = r7[3];
          tensorforge::fmacdpp16<0>(v205_acc, v212_lin, v189_data);
          tensorforge::fmacdpp16<1>(v205_acc, v212_lin, v190_data);
          tensorforge::fmacdpp16<2>(v205_acc, v212_lin, v191_data);
          tensorforge::fmacdpp16<3>(v205_acc, v212_lin, v192_data);
          tensorforge::fmacdpp16<4>(v205_acc, v212_lin, v193_data);
          tensorforge::fmacdpp16<5>(v205_acc, v212_lin, v194_data);
          tensorforge::fmacdpp16<6>(v205_acc, v212_lin, v195_data);
          tensorforge::fmacdpp16<7>(v205_acc, v212_lin, v196_data);
          tensorforge::fmacdpp16<8>(v205_acc, v212_lin, v197_data);
          tensorforge::fmacdpp16<9>(v205_acc, v212_lin, v198_data);
          tensorforge::fmacdpp16<10>(v205_acc, v212_lin, v199_data);
          tensorforge::fmacdpp16<11>(v205_acc, v212_lin, v200_data);
          tensorforge::fmacdpp16<12>(v206_acc, v212_lin, v189_data);
          tensorforge::fmacdpp16<13>(v206_acc, v212_lin, v190_data);
          tensorforge::fmacdpp16<14>(v206_acc, v212_lin, v191_data);
          tensorforge::fmacdpp16<15>(v206_acc, v212_lin, v192_data);
          float v213_lin = r7[4];
          tensorforge::fmacdpp16<0>(v206_acc, v213_lin, v193_data);
          tensorforge::fmacdpp16<1>(v206_acc, v213_lin, v194_data);
          tensorforge::fmacdpp16<2>(v206_acc, v213_lin, v195_data);
          tensorforge::fmacdpp16<3>(v206_acc, v213_lin, v196_data);
          tensorforge::fmacdpp16<4>(v206_acc, v213_lin, v197_data);
          tensorforge::fmacdpp16<5>(v206_acc, v213_lin, v198_data);
          tensorforge::fmacdpp16<6>(v206_acc, v213_lin, v199_data);
          tensorforge::fmacdpp16<7>(v206_acc, v213_lin, v200_data);
          tensorforge::fmacdpp16<8>(v207_acc, v213_lin, v189_data);
          tensorforge::fmacdpp16<9>(v207_acc, v213_lin, v190_data);
          tensorforge::fmacdpp16<10>(v207_acc, v213_lin, v191_data);
          tensorforge::fmacdpp16<11>(v207_acc, v213_lin, v192_data);
          tensorforge::fmacdpp16<12>(v207_acc, v213_lin, v193_data);
          tensorforge::fmacdpp16<13>(v207_acc, v213_lin, v194_data);
          tensorforge::fmacdpp16<14>(v207_acc, v213_lin, v195_data);
          tensorforge::fmacdpp16<15>(v207_acc, v213_lin, v196_data);
          float v214_lin = r7[5];
          tensorforge::fmacdpp16<0>(v207_acc, v214_lin, v197_data);
          tensorforge::fmacdpp16<1>(v207_acc, v214_lin, v198_data);
          tensorforge::fmacdpp16<2>(v207_acc, v214_lin, v199_data);
          tensorforge::fmacdpp16<3>(v207_acc, v214_lin, v200_data);
          tensorforge::fmacdpp16<4>(v208_acc, v214_lin, v189_data);
          tensorforge::fmacdpp16<5>(v208_acc, v214_lin, v190_data);
          tensorforge::fmacdpp16<6>(v208_acc, v214_lin, v191_data);
          tensorforge::fmacdpp16<7>(v208_acc, v214_lin, v192_data);
          tensorforge::fmacdpp16<8>(v208_acc, v214_lin, v193_data);
          tensorforge::fmacdpp16<9>(v208_acc, v214_lin, v194_data);
          tensorforge::fmacdpp16<10>(v208_acc, v214_lin, v195_data);
          tensorforge::fmacdpp16<11>(v208_acc, v214_lin, v196_data);
          tensorforge::fmacdpp16<12>(v208_acc, v214_lin, v197_data);
          tensorforge::fmacdpp16<13>(v208_acc, v214_lin, v198_data);
          tensorforge::fmacdpp16<14>(v208_acc, v214_lin, v199_data);
          tensorforge::fmacdpp16<15>(v208_acc, v214_lin, v200_data);
          ir8[0] = v201_acc;
          ir8[1] = v202_acc;
          ir8[2] = v203_acc;
          ir8[3] = v204_acc;
          ir8[4] = v205_acc;
          ir8[5] = v206_acc;
          ir8[6] = v207_acc;
          ir8[7] = v208_acc;
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v219_n1 = 0; v219_n1 < 8; ++v219_n1) {
              int32_t v220_a = 0 + v219_n1;
              float v222_data = ir8[v219_n1];
              int32_t v223_a = 0 + v219_n1;
              float v225_data = r5[v219_n1];
              r8[v219_n1] = (v225_data + v222_data);
            }
          }
          float r10[8]{};
          // r10 = load{g>r}(glb_m8);
          float v229_lin = glb_m8[0 + threadIdx.x * 1];
          r10[0] = v229_lin;
          float v230_lin = glb_m8[16 + threadIdx.x * 1];
          r10[1] = v230_lin;
          float v231_lin = glb_m8[32 + threadIdx.x * 1];
          r10[2] = v231_lin;
          float v232_lin = glb_m8[48 + threadIdx.x * 1];
          r10[3] = v232_lin;
          float v233_lin = glb_m8[64 + threadIdx.x * 1];
          r10[4] = v233_lin;
          float v234_lin = glb_m8[80 + threadIdx.x * 1];
          r10[5] = v234_lin;
          // wait(r9 = load{g>r}(glb_m7););
          // wait(r10 = load{g>r}(glb_m8););
          float r11[8]{};
          // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir11[8]{};
          float v237_data = r9[0];
          float v238_data = r9[1];
          float v239_data = r9[2];
          float v240_data = r9[3];
          float v241_data = r9[4];
          float v242_data = r9[5];
          float v243_data = r9[6];
          float v244_data = r9[7];
          float v245_data = r9[8];
          float v246_data = r9[9];
          float v247_data = r9[10];
          float v248_data = r9[11];
          float v249_acc{};
          float v250_acc{};
          float v251_acc{};
          float v252_acc{};
          float v253_acc{};
          float v254_acc{};
          float v255_acc{};
          float v256_acc{};
          float v257_lin = r10[0];
          tensorforge::fmacdpp16<0>(v249_acc, v257_lin, v237_data);
          tensorforge::fmacdpp16<1>(v249_acc, v257_lin, v238_data);
          tensorforge::fmacdpp16<2>(v249_acc, v257_lin, v239_data);
          tensorforge::fmacdpp16<3>(v249_acc, v257_lin, v240_data);
          tensorforge::fmacdpp16<4>(v249_acc, v257_lin, v241_data);
          tensorforge::fmacdpp16<5>(v249_acc, v257_lin, v242_data);
          tensorforge::fmacdpp16<6>(v249_acc, v257_lin, v243_data);
          tensorforge::fmacdpp16<7>(v249_acc, v257_lin, v244_data);
          tensorforge::fmacdpp16<8>(v249_acc, v257_lin, v245_data);
          tensorforge::fmacdpp16<9>(v249_acc, v257_lin, v246_data);
          tensorforge::fmacdpp16<10>(v249_acc, v257_lin, v247_data);
          tensorforge::fmacdpp16<11>(v249_acc, v257_lin, v248_data);
          tensorforge::fmacdpp16<12>(v250_acc, v257_lin, v237_data);
          tensorforge::fmacdpp16<13>(v250_acc, v257_lin, v238_data);
          tensorforge::fmacdpp16<14>(v250_acc, v257_lin, v239_data);
          tensorforge::fmacdpp16<15>(v250_acc, v257_lin, v240_data);
          float v258_lin = r10[1];
          tensorforge::fmacdpp16<0>(v250_acc, v258_lin, v241_data);
          tensorforge::fmacdpp16<1>(v250_acc, v258_lin, v242_data);
          tensorforge::fmacdpp16<2>(v250_acc, v258_lin, v243_data);
          tensorforge::fmacdpp16<3>(v250_acc, v258_lin, v244_data);
          tensorforge::fmacdpp16<4>(v250_acc, v258_lin, v245_data);
          tensorforge::fmacdpp16<5>(v250_acc, v258_lin, v246_data);
          tensorforge::fmacdpp16<6>(v250_acc, v258_lin, v247_data);
          tensorforge::fmacdpp16<7>(v250_acc, v258_lin, v248_data);
          tensorforge::fmacdpp16<8>(v251_acc, v258_lin, v237_data);
          tensorforge::fmacdpp16<9>(v251_acc, v258_lin, v238_data);
          tensorforge::fmacdpp16<10>(v251_acc, v258_lin, v239_data);
          tensorforge::fmacdpp16<11>(v251_acc, v258_lin, v240_data);
          tensorforge::fmacdpp16<12>(v251_acc, v258_lin, v241_data);
          tensorforge::fmacdpp16<13>(v251_acc, v258_lin, v242_data);
          tensorforge::fmacdpp16<14>(v251_acc, v258_lin, v243_data);
          tensorforge::fmacdpp16<15>(v251_acc, v258_lin, v244_data);
          float v259_lin = r10[2];
          tensorforge::fmacdpp16<0>(v251_acc, v259_lin, v245_data);
          tensorforge::fmacdpp16<1>(v251_acc, v259_lin, v246_data);
          tensorforge::fmacdpp16<2>(v251_acc, v259_lin, v247_data);
          tensorforge::fmacdpp16<3>(v251_acc, v259_lin, v248_data);
          tensorforge::fmacdpp16<4>(v252_acc, v259_lin, v237_data);
          tensorforge::fmacdpp16<5>(v252_acc, v259_lin, v238_data);
          tensorforge::fmacdpp16<6>(v252_acc, v259_lin, v239_data);
          tensorforge::fmacdpp16<7>(v252_acc, v259_lin, v240_data);
          tensorforge::fmacdpp16<8>(v252_acc, v259_lin, v241_data);
          tensorforge::fmacdpp16<9>(v252_acc, v259_lin, v242_data);
          tensorforge::fmacdpp16<10>(v252_acc, v259_lin, v243_data);
          tensorforge::fmacdpp16<11>(v252_acc, v259_lin, v244_data);
          tensorforge::fmacdpp16<12>(v252_acc, v259_lin, v245_data);
          tensorforge::fmacdpp16<13>(v252_acc, v259_lin, v246_data);
          tensorforge::fmacdpp16<14>(v252_acc, v259_lin, v247_data);
          tensorforge::fmacdpp16<15>(v252_acc, v259_lin, v248_data);
          float v260_lin = r10[3];
          tensorforge::fmacdpp16<0>(v253_acc, v260_lin, v237_data);
          tensorforge::fmacdpp16<1>(v253_acc, v260_lin, v238_data);
          tensorforge::fmacdpp16<2>(v253_acc, v260_lin, v239_data);
          tensorforge::fmacdpp16<3>(v253_acc, v260_lin, v240_data);
          tensorforge::fmacdpp16<4>(v253_acc, v260_lin, v241_data);
          tensorforge::fmacdpp16<5>(v253_acc, v260_lin, v242_data);
          tensorforge::fmacdpp16<6>(v253_acc, v260_lin, v243_data);
          tensorforge::fmacdpp16<7>(v253_acc, v260_lin, v244_data);
          tensorforge::fmacdpp16<8>(v253_acc, v260_lin, v245_data);
          tensorforge::fmacdpp16<9>(v253_acc, v260_lin, v246_data);
          tensorforge::fmacdpp16<10>(v253_acc, v260_lin, v247_data);
          tensorforge::fmacdpp16<11>(v253_acc, v260_lin, v248_data);
          tensorforge::fmacdpp16<12>(v254_acc, v260_lin, v237_data);
          tensorforge::fmacdpp16<13>(v254_acc, v260_lin, v238_data);
          tensorforge::fmacdpp16<14>(v254_acc, v260_lin, v239_data);
          tensorforge::fmacdpp16<15>(v254_acc, v260_lin, v240_data);
          float v261_lin = r10[4];
          tensorforge::fmacdpp16<0>(v254_acc, v261_lin, v241_data);
          tensorforge::fmacdpp16<1>(v254_acc, v261_lin, v242_data);
          tensorforge::fmacdpp16<2>(v254_acc, v261_lin, v243_data);
          tensorforge::fmacdpp16<3>(v254_acc, v261_lin, v244_data);
          tensorforge::fmacdpp16<4>(v254_acc, v261_lin, v245_data);
          tensorforge::fmacdpp16<5>(v254_acc, v261_lin, v246_data);
          tensorforge::fmacdpp16<6>(v254_acc, v261_lin, v247_data);
          tensorforge::fmacdpp16<7>(v254_acc, v261_lin, v248_data);
          tensorforge::fmacdpp16<8>(v255_acc, v261_lin, v237_data);
          tensorforge::fmacdpp16<9>(v255_acc, v261_lin, v238_data);
          tensorforge::fmacdpp16<10>(v255_acc, v261_lin, v239_data);
          tensorforge::fmacdpp16<11>(v255_acc, v261_lin, v240_data);
          tensorforge::fmacdpp16<12>(v255_acc, v261_lin, v241_data);
          tensorforge::fmacdpp16<13>(v255_acc, v261_lin, v242_data);
          tensorforge::fmacdpp16<14>(v255_acc, v261_lin, v243_data);
          tensorforge::fmacdpp16<15>(v255_acc, v261_lin, v244_data);
          float v262_lin = r10[5];
          tensorforge::fmacdpp16<0>(v255_acc, v262_lin, v245_data);
          tensorforge::fmacdpp16<1>(v255_acc, v262_lin, v246_data);
          tensorforge::fmacdpp16<2>(v255_acc, v262_lin, v247_data);
          tensorforge::fmacdpp16<3>(v255_acc, v262_lin, v248_data);
          tensorforge::fmacdpp16<4>(v256_acc, v262_lin, v237_data);
          tensorforge::fmacdpp16<5>(v256_acc, v262_lin, v238_data);
          tensorforge::fmacdpp16<6>(v256_acc, v262_lin, v239_data);
          tensorforge::fmacdpp16<7>(v256_acc, v262_lin, v240_data);
          tensorforge::fmacdpp16<8>(v256_acc, v262_lin, v241_data);
          tensorforge::fmacdpp16<9>(v256_acc, v262_lin, v242_data);
          tensorforge::fmacdpp16<10>(v256_acc, v262_lin, v243_data);
          tensorforge::fmacdpp16<11>(v256_acc, v262_lin, v244_data);
          tensorforge::fmacdpp16<12>(v256_acc, v262_lin, v245_data);
          tensorforge::fmacdpp16<13>(v256_acc, v262_lin, v246_data);
          tensorforge::fmacdpp16<14>(v256_acc, v262_lin, v247_data);
          tensorforge::fmacdpp16<15>(v256_acc, v262_lin, v248_data);
          ir11[0] = v249_acc;
          ir11[1] = v250_acc;
          ir11[2] = v251_acc;
          ir11[3] = v252_acc;
          ir11[4] = v253_acc;
          ir11[5] = v254_acc;
          ir11[6] = v255_acc;
          ir11[7] = v256_acc;
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v267_n1 = 0; v267_n1 < 8; ++v267_n1) {
              int32_t v268_a = 0 + v267_n1;
              float v270_data = ir11[v267_n1];
              int32_t v271_a = 0 + v267_n1;
              float v273_data = r8[v267_n1];
              r11[v267_n1] = (v273_data + v270_data);
            }
          }
          // glb_m0 = store{r>g}(r11);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v280_i1 = 0; v280_i1 < 8; ++v280_i1) {
              int32_t v281_a = 0 + v280_i1;
              float v283_data = r11[v280_i1];
              glb_m0[(v13_lead + (v280_i1 * 12))] = v283_data;
            }
          }
        }
      }
    }
  }
}

