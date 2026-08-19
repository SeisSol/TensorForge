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
              int32_t v11_a = v2_lead + (v4_i1 * 12);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v13_a = 0 + v4_i1;
              r0[v13_a] = v12_data;
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
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              int32_t v25_a = v16_lead + (v18_i1 * 12);
              float v26_data;
              {
                v26_data = __builtin_nontemporal_load(&glb_m3[v25_a]);
              }
              int32_t v27_a = 0 + v18_i1;
              r3[v27_a] = v26_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          auto& ir2 = r2;
          float v28_data = r0[0];
          float v29_data = r0[1];
          float v30_data = r0[2];
          float v31_data = r0[3];
          float v32_data = r0[4];
          float v33_data = r0[5];
          float v34_data = r0[6];
          float v35_data = r0[7];
          float v36_data = r0[8];
          float v37_data = r0[9];
          float v38_data = r0[10];
          float v39_data = r0[11];
          float v40_acc{};
          float v41_acc{};
          float v42_acc{};
          float v43_acc{};
          float v44_acc{};
          float v45_acc{};
          float v46_acc{};
          float v47_acc{};
          float v48_lin = r1[0];
          tensorforge::fmacdpp16<0>(v40_acc, v48_lin, v28_data);
          tensorforge::fmacdpp16<1>(v40_acc, v48_lin, v29_data);
          tensorforge::fmacdpp16<2>(v40_acc, v48_lin, v30_data);
          tensorforge::fmacdpp16<3>(v40_acc, v48_lin, v31_data);
          tensorforge::fmacdpp16<4>(v40_acc, v48_lin, v32_data);
          tensorforge::fmacdpp16<5>(v40_acc, v48_lin, v33_data);
          tensorforge::fmacdpp16<6>(v40_acc, v48_lin, v34_data);
          tensorforge::fmacdpp16<7>(v40_acc, v48_lin, v35_data);
          tensorforge::fmacdpp16<8>(v40_acc, v48_lin, v36_data);
          tensorforge::fmacdpp16<9>(v40_acc, v48_lin, v37_data);
          tensorforge::fmacdpp16<10>(v40_acc, v48_lin, v38_data);
          tensorforge::fmacdpp16<11>(v40_acc, v48_lin, v39_data);
          tensorforge::fmacdpp16<12>(v41_acc, v48_lin, v28_data);
          tensorforge::fmacdpp16<13>(v41_acc, v48_lin, v29_data);
          tensorforge::fmacdpp16<14>(v41_acc, v48_lin, v30_data);
          tensorforge::fmacdpp16<15>(v41_acc, v48_lin, v31_data);
          float v49_lin = r1[1];
          tensorforge::fmacdpp16<0>(v41_acc, v49_lin, v32_data);
          tensorforge::fmacdpp16<1>(v41_acc, v49_lin, v33_data);
          tensorforge::fmacdpp16<2>(v41_acc, v49_lin, v34_data);
          tensorforge::fmacdpp16<3>(v41_acc, v49_lin, v35_data);
          tensorforge::fmacdpp16<4>(v41_acc, v49_lin, v36_data);
          tensorforge::fmacdpp16<5>(v41_acc, v49_lin, v37_data);
          tensorforge::fmacdpp16<6>(v41_acc, v49_lin, v38_data);
          tensorforge::fmacdpp16<7>(v41_acc, v49_lin, v39_data);
          tensorforge::fmacdpp16<8>(v42_acc, v49_lin, v28_data);
          tensorforge::fmacdpp16<9>(v42_acc, v49_lin, v29_data);
          tensorforge::fmacdpp16<10>(v42_acc, v49_lin, v30_data);
          tensorforge::fmacdpp16<11>(v42_acc, v49_lin, v31_data);
          tensorforge::fmacdpp16<12>(v42_acc, v49_lin, v32_data);
          tensorforge::fmacdpp16<13>(v42_acc, v49_lin, v33_data);
          tensorforge::fmacdpp16<14>(v42_acc, v49_lin, v34_data);
          tensorforge::fmacdpp16<15>(v42_acc, v49_lin, v35_data);
          float v50_lin = r1[2];
          tensorforge::fmacdpp16<0>(v42_acc, v50_lin, v36_data);
          tensorforge::fmacdpp16<1>(v42_acc, v50_lin, v37_data);
          tensorforge::fmacdpp16<2>(v42_acc, v50_lin, v38_data);
          tensorforge::fmacdpp16<3>(v42_acc, v50_lin, v39_data);
          tensorforge::fmacdpp16<4>(v43_acc, v50_lin, v28_data);
          tensorforge::fmacdpp16<5>(v43_acc, v50_lin, v29_data);
          tensorforge::fmacdpp16<6>(v43_acc, v50_lin, v30_data);
          tensorforge::fmacdpp16<7>(v43_acc, v50_lin, v31_data);
          tensorforge::fmacdpp16<8>(v43_acc, v50_lin, v32_data);
          tensorforge::fmacdpp16<9>(v43_acc, v50_lin, v33_data);
          tensorforge::fmacdpp16<10>(v43_acc, v50_lin, v34_data);
          tensorforge::fmacdpp16<11>(v43_acc, v50_lin, v35_data);
          tensorforge::fmacdpp16<12>(v43_acc, v50_lin, v36_data);
          tensorforge::fmacdpp16<13>(v43_acc, v50_lin, v37_data);
          tensorforge::fmacdpp16<14>(v43_acc, v50_lin, v38_data);
          tensorforge::fmacdpp16<15>(v43_acc, v50_lin, v39_data);
          float v51_lin = r1[3];
          tensorforge::fmacdpp16<0>(v44_acc, v51_lin, v28_data);
          tensorforge::fmacdpp16<1>(v44_acc, v51_lin, v29_data);
          tensorforge::fmacdpp16<2>(v44_acc, v51_lin, v30_data);
          tensorforge::fmacdpp16<3>(v44_acc, v51_lin, v31_data);
          tensorforge::fmacdpp16<4>(v44_acc, v51_lin, v32_data);
          tensorforge::fmacdpp16<5>(v44_acc, v51_lin, v33_data);
          tensorforge::fmacdpp16<6>(v44_acc, v51_lin, v34_data);
          tensorforge::fmacdpp16<7>(v44_acc, v51_lin, v35_data);
          tensorforge::fmacdpp16<8>(v44_acc, v51_lin, v36_data);
          tensorforge::fmacdpp16<9>(v44_acc, v51_lin, v37_data);
          tensorforge::fmacdpp16<10>(v44_acc, v51_lin, v38_data);
          tensorforge::fmacdpp16<11>(v44_acc, v51_lin, v39_data);
          tensorforge::fmacdpp16<12>(v45_acc, v51_lin, v28_data);
          tensorforge::fmacdpp16<13>(v45_acc, v51_lin, v29_data);
          tensorforge::fmacdpp16<14>(v45_acc, v51_lin, v30_data);
          tensorforge::fmacdpp16<15>(v45_acc, v51_lin, v31_data);
          float v52_lin = r1[4];
          tensorforge::fmacdpp16<0>(v45_acc, v52_lin, v32_data);
          tensorforge::fmacdpp16<1>(v45_acc, v52_lin, v33_data);
          tensorforge::fmacdpp16<2>(v45_acc, v52_lin, v34_data);
          tensorforge::fmacdpp16<3>(v45_acc, v52_lin, v35_data);
          tensorforge::fmacdpp16<4>(v45_acc, v52_lin, v36_data);
          tensorforge::fmacdpp16<5>(v45_acc, v52_lin, v37_data);
          tensorforge::fmacdpp16<6>(v45_acc, v52_lin, v38_data);
          tensorforge::fmacdpp16<7>(v45_acc, v52_lin, v39_data);
          tensorforge::fmacdpp16<8>(v46_acc, v52_lin, v28_data);
          tensorforge::fmacdpp16<9>(v46_acc, v52_lin, v29_data);
          tensorforge::fmacdpp16<10>(v46_acc, v52_lin, v30_data);
          tensorforge::fmacdpp16<11>(v46_acc, v52_lin, v31_data);
          tensorforge::fmacdpp16<12>(v46_acc, v52_lin, v32_data);
          tensorforge::fmacdpp16<13>(v46_acc, v52_lin, v33_data);
          tensorforge::fmacdpp16<14>(v46_acc, v52_lin, v34_data);
          tensorforge::fmacdpp16<15>(v46_acc, v52_lin, v35_data);
          float v53_lin = r1[5];
          tensorforge::fmacdpp16<0>(v46_acc, v53_lin, v36_data);
          tensorforge::fmacdpp16<1>(v46_acc, v53_lin, v37_data);
          tensorforge::fmacdpp16<2>(v46_acc, v53_lin, v38_data);
          tensorforge::fmacdpp16<3>(v46_acc, v53_lin, v39_data);
          tensorforge::fmacdpp16<4>(v47_acc, v53_lin, v28_data);
          tensorforge::fmacdpp16<5>(v47_acc, v53_lin, v29_data);
          tensorforge::fmacdpp16<6>(v47_acc, v53_lin, v30_data);
          tensorforge::fmacdpp16<7>(v47_acc, v53_lin, v31_data);
          tensorforge::fmacdpp16<8>(v47_acc, v53_lin, v32_data);
          tensorforge::fmacdpp16<9>(v47_acc, v53_lin, v33_data);
          tensorforge::fmacdpp16<10>(v47_acc, v53_lin, v34_data);
          tensorforge::fmacdpp16<11>(v47_acc, v53_lin, v35_data);
          tensorforge::fmacdpp16<12>(v47_acc, v53_lin, v36_data);
          tensorforge::fmacdpp16<13>(v47_acc, v53_lin, v37_data);
          tensorforge::fmacdpp16<14>(v47_acc, v53_lin, v38_data);
          tensorforge::fmacdpp16<15>(v47_acc, v53_lin, v39_data);
          ir2[0] = v40_acc;
          ir2[1] = v41_acc;
          ir2[2] = v42_acc;
          ir2[3] = v43_acc;
          ir2[4] = v44_acc;
          ir2[5] = v45_acc;
          ir2[6] = v46_acc;
          ir2[7] = v47_acc;
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
          int32_t v56_lead = threadIdx.x % 16;
          if (v56_lead < 12) {
            #pragma unroll
            for (int32_t v58_i1 = 0; v58_i1 < 12; ++v58_i1) {
              int32_t v65_a = v56_lead + (v58_i1 * 12);
              float v66_data;
              {
                v66_data = __builtin_nontemporal_load(&glb_m5[v65_a]);
              }
              int32_t v67_a = 0 + v58_i1;
              r6[v67_a] = v66_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          {
            // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir5[8]{};
            float v68_data = r3[0];
            float v69_data = r3[1];
            float v70_data = r3[2];
            float v71_data = r3[3];
            float v72_data = r3[4];
            float v73_data = r3[5];
            float v74_data = r3[6];
            float v75_data = r3[7];
            float v76_data = r3[8];
            float v77_data = r3[9];
            float v78_data = r3[10];
            float v79_data = r3[11];
            float v80_acc{};
            float v81_acc{};
            float v82_acc{};
            float v83_acc{};
            float v84_acc{};
            float v85_acc{};
            float v86_acc{};
            float v87_acc{};
            float v88_lin = r4[0];
            tensorforge::fmacdpp16<0>(v80_acc, v88_lin, v68_data);
            tensorforge::fmacdpp16<1>(v80_acc, v88_lin, v69_data);
            tensorforge::fmacdpp16<2>(v80_acc, v88_lin, v70_data);
            tensorforge::fmacdpp16<3>(v80_acc, v88_lin, v71_data);
            tensorforge::fmacdpp16<4>(v80_acc, v88_lin, v72_data);
            tensorforge::fmacdpp16<5>(v80_acc, v88_lin, v73_data);
            tensorforge::fmacdpp16<6>(v80_acc, v88_lin, v74_data);
            tensorforge::fmacdpp16<7>(v80_acc, v88_lin, v75_data);
            tensorforge::fmacdpp16<8>(v80_acc, v88_lin, v76_data);
            tensorforge::fmacdpp16<9>(v80_acc, v88_lin, v77_data);
            tensorforge::fmacdpp16<10>(v80_acc, v88_lin, v78_data);
            tensorforge::fmacdpp16<11>(v80_acc, v88_lin, v79_data);
            tensorforge::fmacdpp16<12>(v81_acc, v88_lin, v68_data);
            tensorforge::fmacdpp16<13>(v81_acc, v88_lin, v69_data);
            tensorforge::fmacdpp16<14>(v81_acc, v88_lin, v70_data);
            tensorforge::fmacdpp16<15>(v81_acc, v88_lin, v71_data);
            float v89_lin = r4[1];
            tensorforge::fmacdpp16<0>(v81_acc, v89_lin, v72_data);
            tensorforge::fmacdpp16<1>(v81_acc, v89_lin, v73_data);
            tensorforge::fmacdpp16<2>(v81_acc, v89_lin, v74_data);
            tensorforge::fmacdpp16<3>(v81_acc, v89_lin, v75_data);
            tensorforge::fmacdpp16<4>(v81_acc, v89_lin, v76_data);
            tensorforge::fmacdpp16<5>(v81_acc, v89_lin, v77_data);
            tensorforge::fmacdpp16<6>(v81_acc, v89_lin, v78_data);
            tensorforge::fmacdpp16<7>(v81_acc, v89_lin, v79_data);
            tensorforge::fmacdpp16<8>(v82_acc, v89_lin, v68_data);
            tensorforge::fmacdpp16<9>(v82_acc, v89_lin, v69_data);
            tensorforge::fmacdpp16<10>(v82_acc, v89_lin, v70_data);
            tensorforge::fmacdpp16<11>(v82_acc, v89_lin, v71_data);
            tensorforge::fmacdpp16<12>(v82_acc, v89_lin, v72_data);
            tensorforge::fmacdpp16<13>(v82_acc, v89_lin, v73_data);
            tensorforge::fmacdpp16<14>(v82_acc, v89_lin, v74_data);
            tensorforge::fmacdpp16<15>(v82_acc, v89_lin, v75_data);
            float v90_lin = r4[2];
            tensorforge::fmacdpp16<0>(v82_acc, v90_lin, v76_data);
            tensorforge::fmacdpp16<1>(v82_acc, v90_lin, v77_data);
            tensorforge::fmacdpp16<2>(v82_acc, v90_lin, v78_data);
            tensorforge::fmacdpp16<3>(v82_acc, v90_lin, v79_data);
            tensorforge::fmacdpp16<4>(v83_acc, v90_lin, v68_data);
            tensorforge::fmacdpp16<5>(v83_acc, v90_lin, v69_data);
            tensorforge::fmacdpp16<6>(v83_acc, v90_lin, v70_data);
            tensorforge::fmacdpp16<7>(v83_acc, v90_lin, v71_data);
            tensorforge::fmacdpp16<8>(v83_acc, v90_lin, v72_data);
            tensorforge::fmacdpp16<9>(v83_acc, v90_lin, v73_data);
            tensorforge::fmacdpp16<10>(v83_acc, v90_lin, v74_data);
            tensorforge::fmacdpp16<11>(v83_acc, v90_lin, v75_data);
            tensorforge::fmacdpp16<12>(v83_acc, v90_lin, v76_data);
            tensorforge::fmacdpp16<13>(v83_acc, v90_lin, v77_data);
            tensorforge::fmacdpp16<14>(v83_acc, v90_lin, v78_data);
            tensorforge::fmacdpp16<15>(v83_acc, v90_lin, v79_data);
            float v91_lin = r4[3];
            tensorforge::fmacdpp16<0>(v84_acc, v91_lin, v68_data);
            tensorforge::fmacdpp16<1>(v84_acc, v91_lin, v69_data);
            tensorforge::fmacdpp16<2>(v84_acc, v91_lin, v70_data);
            tensorforge::fmacdpp16<3>(v84_acc, v91_lin, v71_data);
            tensorforge::fmacdpp16<4>(v84_acc, v91_lin, v72_data);
            tensorforge::fmacdpp16<5>(v84_acc, v91_lin, v73_data);
            tensorforge::fmacdpp16<6>(v84_acc, v91_lin, v74_data);
            tensorforge::fmacdpp16<7>(v84_acc, v91_lin, v75_data);
            tensorforge::fmacdpp16<8>(v84_acc, v91_lin, v76_data);
            tensorforge::fmacdpp16<9>(v84_acc, v91_lin, v77_data);
            tensorforge::fmacdpp16<10>(v84_acc, v91_lin, v78_data);
            tensorforge::fmacdpp16<11>(v84_acc, v91_lin, v79_data);
            tensorforge::fmacdpp16<12>(v85_acc, v91_lin, v68_data);
            tensorforge::fmacdpp16<13>(v85_acc, v91_lin, v69_data);
            tensorforge::fmacdpp16<14>(v85_acc, v91_lin, v70_data);
            tensorforge::fmacdpp16<15>(v85_acc, v91_lin, v71_data);
            float v92_lin = r4[4];
            tensorforge::fmacdpp16<0>(v85_acc, v92_lin, v72_data);
            tensorforge::fmacdpp16<1>(v85_acc, v92_lin, v73_data);
            tensorforge::fmacdpp16<2>(v85_acc, v92_lin, v74_data);
            tensorforge::fmacdpp16<3>(v85_acc, v92_lin, v75_data);
            tensorforge::fmacdpp16<4>(v85_acc, v92_lin, v76_data);
            tensorforge::fmacdpp16<5>(v85_acc, v92_lin, v77_data);
            tensorforge::fmacdpp16<6>(v85_acc, v92_lin, v78_data);
            tensorforge::fmacdpp16<7>(v85_acc, v92_lin, v79_data);
            tensorforge::fmacdpp16<8>(v86_acc, v92_lin, v68_data);
            tensorforge::fmacdpp16<9>(v86_acc, v92_lin, v69_data);
            tensorforge::fmacdpp16<10>(v86_acc, v92_lin, v70_data);
            tensorforge::fmacdpp16<11>(v86_acc, v92_lin, v71_data);
            tensorforge::fmacdpp16<12>(v86_acc, v92_lin, v72_data);
            tensorforge::fmacdpp16<13>(v86_acc, v92_lin, v73_data);
            tensorforge::fmacdpp16<14>(v86_acc, v92_lin, v74_data);
            tensorforge::fmacdpp16<15>(v86_acc, v92_lin, v75_data);
            float v93_lin = r4[5];
            tensorforge::fmacdpp16<0>(v86_acc, v93_lin, v76_data);
            tensorforge::fmacdpp16<1>(v86_acc, v93_lin, v77_data);
            tensorforge::fmacdpp16<2>(v86_acc, v93_lin, v78_data);
            tensorforge::fmacdpp16<3>(v86_acc, v93_lin, v79_data);
            tensorforge::fmacdpp16<4>(v87_acc, v93_lin, v68_data);
            tensorforge::fmacdpp16<5>(v87_acc, v93_lin, v69_data);
            tensorforge::fmacdpp16<6>(v87_acc, v93_lin, v70_data);
            tensorforge::fmacdpp16<7>(v87_acc, v93_lin, v71_data);
            tensorforge::fmacdpp16<8>(v87_acc, v93_lin, v72_data);
            tensorforge::fmacdpp16<9>(v87_acc, v93_lin, v73_data);
            tensorforge::fmacdpp16<10>(v87_acc, v93_lin, v74_data);
            tensorforge::fmacdpp16<11>(v87_acc, v93_lin, v75_data);
            tensorforge::fmacdpp16<12>(v87_acc, v93_lin, v76_data);
            tensorforge::fmacdpp16<13>(v87_acc, v93_lin, v77_data);
            tensorforge::fmacdpp16<14>(v87_acc, v93_lin, v78_data);
            tensorforge::fmacdpp16<15>(v87_acc, v93_lin, v79_data);
            ir5[0] = v80_acc;
            ir5[1] = v81_acc;
            ir5[2] = v82_acc;
            ir5[3] = v83_acc;
            ir5[4] = v84_acc;
            ir5[5] = v85_acc;
            ir5[6] = v86_acc;
            ir5[7] = v87_acc;
            if ((threadIdx.x % 16) < 12) {
              #pragma unroll
              for (int32_t v98_n1 = 0; v98_n1 < 8; ++v98_n1) {
                int32_t v99_a = 0 + v98_n1;
                float v101_data = ir5[v98_n1];
                int32_t v102_a = 0 + v98_n1;
                float v104_data = r2[v98_n1];
                int32_t v106_a = 0 + v98_n1;
                r5[v98_n1] = (v104_data + v101_data);
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
          int32_t v110_lead = threadIdx.x % 16;
          if (v110_lead < 12) {
            #pragma unroll
            for (int32_t v112_i1 = 0; v112_i1 < 12; ++v112_i1) {
              int32_t v119_a = v110_lead + (v112_i1 * 12);
              float v120_data;
              {
                v120_data = __builtin_nontemporal_load(&glb_m7[v119_a]);
              }
              int32_t v121_a = 0 + v112_i1;
              r9[v121_a] = v120_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          {
            // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 12)]
            float ir8[8]{};
            float v122_data = r6[0];
            float v123_data = r6[1];
            float v124_data = r6[2];
            float v125_data = r6[3];
            float v126_data = r6[4];
            float v127_data = r6[5];
            float v128_data = r6[6];
            float v129_data = r6[7];
            float v130_data = r6[8];
            float v131_data = r6[9];
            float v132_data = r6[10];
            float v133_data = r6[11];
            float v134_acc{};
            float v135_acc{};
            float v136_acc{};
            float v137_acc{};
            float v138_acc{};
            float v139_acc{};
            float v140_acc{};
            float v141_acc{};
            float v142_lin = r7[0];
            tensorforge::fmacdpp16<0>(v134_acc, v142_lin, v122_data);
            tensorforge::fmacdpp16<1>(v134_acc, v142_lin, v123_data);
            tensorforge::fmacdpp16<2>(v134_acc, v142_lin, v124_data);
            tensorforge::fmacdpp16<3>(v134_acc, v142_lin, v125_data);
            tensorforge::fmacdpp16<4>(v134_acc, v142_lin, v126_data);
            tensorforge::fmacdpp16<5>(v134_acc, v142_lin, v127_data);
            tensorforge::fmacdpp16<6>(v134_acc, v142_lin, v128_data);
            tensorforge::fmacdpp16<7>(v134_acc, v142_lin, v129_data);
            tensorforge::fmacdpp16<8>(v134_acc, v142_lin, v130_data);
            tensorforge::fmacdpp16<9>(v134_acc, v142_lin, v131_data);
            tensorforge::fmacdpp16<10>(v134_acc, v142_lin, v132_data);
            tensorforge::fmacdpp16<11>(v134_acc, v142_lin, v133_data);
            tensorforge::fmacdpp16<12>(v135_acc, v142_lin, v122_data);
            tensorforge::fmacdpp16<13>(v135_acc, v142_lin, v123_data);
            tensorforge::fmacdpp16<14>(v135_acc, v142_lin, v124_data);
            tensorforge::fmacdpp16<15>(v135_acc, v142_lin, v125_data);
            float v143_lin = r7[1];
            tensorforge::fmacdpp16<0>(v135_acc, v143_lin, v126_data);
            tensorforge::fmacdpp16<1>(v135_acc, v143_lin, v127_data);
            tensorforge::fmacdpp16<2>(v135_acc, v143_lin, v128_data);
            tensorforge::fmacdpp16<3>(v135_acc, v143_lin, v129_data);
            tensorforge::fmacdpp16<4>(v135_acc, v143_lin, v130_data);
            tensorforge::fmacdpp16<5>(v135_acc, v143_lin, v131_data);
            tensorforge::fmacdpp16<6>(v135_acc, v143_lin, v132_data);
            tensorforge::fmacdpp16<7>(v135_acc, v143_lin, v133_data);
            tensorforge::fmacdpp16<8>(v136_acc, v143_lin, v122_data);
            tensorforge::fmacdpp16<9>(v136_acc, v143_lin, v123_data);
            tensorforge::fmacdpp16<10>(v136_acc, v143_lin, v124_data);
            tensorforge::fmacdpp16<11>(v136_acc, v143_lin, v125_data);
            tensorforge::fmacdpp16<12>(v136_acc, v143_lin, v126_data);
            tensorforge::fmacdpp16<13>(v136_acc, v143_lin, v127_data);
            tensorforge::fmacdpp16<14>(v136_acc, v143_lin, v128_data);
            tensorforge::fmacdpp16<15>(v136_acc, v143_lin, v129_data);
            float v144_lin = r7[2];
            tensorforge::fmacdpp16<0>(v136_acc, v144_lin, v130_data);
            tensorforge::fmacdpp16<1>(v136_acc, v144_lin, v131_data);
            tensorforge::fmacdpp16<2>(v136_acc, v144_lin, v132_data);
            tensorforge::fmacdpp16<3>(v136_acc, v144_lin, v133_data);
            tensorforge::fmacdpp16<4>(v137_acc, v144_lin, v122_data);
            tensorforge::fmacdpp16<5>(v137_acc, v144_lin, v123_data);
            tensorforge::fmacdpp16<6>(v137_acc, v144_lin, v124_data);
            tensorforge::fmacdpp16<7>(v137_acc, v144_lin, v125_data);
            tensorforge::fmacdpp16<8>(v137_acc, v144_lin, v126_data);
            tensorforge::fmacdpp16<9>(v137_acc, v144_lin, v127_data);
            tensorforge::fmacdpp16<10>(v137_acc, v144_lin, v128_data);
            tensorforge::fmacdpp16<11>(v137_acc, v144_lin, v129_data);
            tensorforge::fmacdpp16<12>(v137_acc, v144_lin, v130_data);
            tensorforge::fmacdpp16<13>(v137_acc, v144_lin, v131_data);
            tensorforge::fmacdpp16<14>(v137_acc, v144_lin, v132_data);
            tensorforge::fmacdpp16<15>(v137_acc, v144_lin, v133_data);
            float v145_lin = r7[3];
            tensorforge::fmacdpp16<0>(v138_acc, v145_lin, v122_data);
            tensorforge::fmacdpp16<1>(v138_acc, v145_lin, v123_data);
            tensorforge::fmacdpp16<2>(v138_acc, v145_lin, v124_data);
            tensorforge::fmacdpp16<3>(v138_acc, v145_lin, v125_data);
            tensorforge::fmacdpp16<4>(v138_acc, v145_lin, v126_data);
            tensorforge::fmacdpp16<5>(v138_acc, v145_lin, v127_data);
            tensorforge::fmacdpp16<6>(v138_acc, v145_lin, v128_data);
            tensorforge::fmacdpp16<7>(v138_acc, v145_lin, v129_data);
            tensorforge::fmacdpp16<8>(v138_acc, v145_lin, v130_data);
            tensorforge::fmacdpp16<9>(v138_acc, v145_lin, v131_data);
            tensorforge::fmacdpp16<10>(v138_acc, v145_lin, v132_data);
            tensorforge::fmacdpp16<11>(v138_acc, v145_lin, v133_data);
            tensorforge::fmacdpp16<12>(v139_acc, v145_lin, v122_data);
            tensorforge::fmacdpp16<13>(v139_acc, v145_lin, v123_data);
            tensorforge::fmacdpp16<14>(v139_acc, v145_lin, v124_data);
            tensorforge::fmacdpp16<15>(v139_acc, v145_lin, v125_data);
            float v146_lin = r7[4];
            tensorforge::fmacdpp16<0>(v139_acc, v146_lin, v126_data);
            tensorforge::fmacdpp16<1>(v139_acc, v146_lin, v127_data);
            tensorforge::fmacdpp16<2>(v139_acc, v146_lin, v128_data);
            tensorforge::fmacdpp16<3>(v139_acc, v146_lin, v129_data);
            tensorforge::fmacdpp16<4>(v139_acc, v146_lin, v130_data);
            tensorforge::fmacdpp16<5>(v139_acc, v146_lin, v131_data);
            tensorforge::fmacdpp16<6>(v139_acc, v146_lin, v132_data);
            tensorforge::fmacdpp16<7>(v139_acc, v146_lin, v133_data);
            tensorforge::fmacdpp16<8>(v140_acc, v146_lin, v122_data);
            tensorforge::fmacdpp16<9>(v140_acc, v146_lin, v123_data);
            tensorforge::fmacdpp16<10>(v140_acc, v146_lin, v124_data);
            tensorforge::fmacdpp16<11>(v140_acc, v146_lin, v125_data);
            tensorforge::fmacdpp16<12>(v140_acc, v146_lin, v126_data);
            tensorforge::fmacdpp16<13>(v140_acc, v146_lin, v127_data);
            tensorforge::fmacdpp16<14>(v140_acc, v146_lin, v128_data);
            tensorforge::fmacdpp16<15>(v140_acc, v146_lin, v129_data);
            float v147_lin = r7[5];
            tensorforge::fmacdpp16<0>(v140_acc, v147_lin, v130_data);
            tensorforge::fmacdpp16<1>(v140_acc, v147_lin, v131_data);
            tensorforge::fmacdpp16<2>(v140_acc, v147_lin, v132_data);
            tensorforge::fmacdpp16<3>(v140_acc, v147_lin, v133_data);
            tensorforge::fmacdpp16<4>(v141_acc, v147_lin, v122_data);
            tensorforge::fmacdpp16<5>(v141_acc, v147_lin, v123_data);
            tensorforge::fmacdpp16<6>(v141_acc, v147_lin, v124_data);
            tensorforge::fmacdpp16<7>(v141_acc, v147_lin, v125_data);
            tensorforge::fmacdpp16<8>(v141_acc, v147_lin, v126_data);
            tensorforge::fmacdpp16<9>(v141_acc, v147_lin, v127_data);
            tensorforge::fmacdpp16<10>(v141_acc, v147_lin, v128_data);
            tensorforge::fmacdpp16<11>(v141_acc, v147_lin, v129_data);
            tensorforge::fmacdpp16<12>(v141_acc, v147_lin, v130_data);
            tensorforge::fmacdpp16<13>(v141_acc, v147_lin, v131_data);
            tensorforge::fmacdpp16<14>(v141_acc, v147_lin, v132_data);
            tensorforge::fmacdpp16<15>(v141_acc, v147_lin, v133_data);
            ir8[0] = v134_acc;
            ir8[1] = v135_acc;
            ir8[2] = v136_acc;
            ir8[3] = v137_acc;
            ir8[4] = v138_acc;
            ir8[5] = v139_acc;
            ir8[6] = v140_acc;
            ir8[7] = v141_acc;
            if ((threadIdx.x % 16) < 12) {
              #pragma unroll
              for (int32_t v152_n1 = 0; v152_n1 < 8; ++v152_n1) {
                int32_t v153_a = 0 + v152_n1;
                float v155_data = ir8[v152_n1];
                int32_t v156_a = 0 + v152_n1;
                float v158_data = r5[v152_n1];
                int32_t v160_a = 0 + v152_n1;
                r8[v152_n1] = (v158_data + v155_data);
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
            float v162_data = r9[0];
            float v163_data = r9[1];
            float v164_data = r9[2];
            float v165_data = r9[3];
            float v166_data = r9[4];
            float v167_data = r9[5];
            float v168_data = r9[6];
            float v169_data = r9[7];
            float v170_data = r9[8];
            float v171_data = r9[9];
            float v172_data = r9[10];
            float v173_data = r9[11];
            float v174_acc{};
            float v175_acc{};
            float v176_acc{};
            float v177_acc{};
            float v178_acc{};
            float v179_acc{};
            float v180_acc{};
            float v181_acc{};
            float v182_lin = r10[0];
            tensorforge::fmacdpp16<0>(v174_acc, v182_lin, v162_data);
            tensorforge::fmacdpp16<1>(v174_acc, v182_lin, v163_data);
            tensorforge::fmacdpp16<2>(v174_acc, v182_lin, v164_data);
            tensorforge::fmacdpp16<3>(v174_acc, v182_lin, v165_data);
            tensorforge::fmacdpp16<4>(v174_acc, v182_lin, v166_data);
            tensorforge::fmacdpp16<5>(v174_acc, v182_lin, v167_data);
            tensorforge::fmacdpp16<6>(v174_acc, v182_lin, v168_data);
            tensorforge::fmacdpp16<7>(v174_acc, v182_lin, v169_data);
            tensorforge::fmacdpp16<8>(v174_acc, v182_lin, v170_data);
            tensorforge::fmacdpp16<9>(v174_acc, v182_lin, v171_data);
            tensorforge::fmacdpp16<10>(v174_acc, v182_lin, v172_data);
            tensorforge::fmacdpp16<11>(v174_acc, v182_lin, v173_data);
            tensorforge::fmacdpp16<12>(v175_acc, v182_lin, v162_data);
            tensorforge::fmacdpp16<13>(v175_acc, v182_lin, v163_data);
            tensorforge::fmacdpp16<14>(v175_acc, v182_lin, v164_data);
            tensorforge::fmacdpp16<15>(v175_acc, v182_lin, v165_data);
            float v183_lin = r10[1];
            tensorforge::fmacdpp16<0>(v175_acc, v183_lin, v166_data);
            tensorforge::fmacdpp16<1>(v175_acc, v183_lin, v167_data);
            tensorforge::fmacdpp16<2>(v175_acc, v183_lin, v168_data);
            tensorforge::fmacdpp16<3>(v175_acc, v183_lin, v169_data);
            tensorforge::fmacdpp16<4>(v175_acc, v183_lin, v170_data);
            tensorforge::fmacdpp16<5>(v175_acc, v183_lin, v171_data);
            tensorforge::fmacdpp16<6>(v175_acc, v183_lin, v172_data);
            tensorforge::fmacdpp16<7>(v175_acc, v183_lin, v173_data);
            tensorforge::fmacdpp16<8>(v176_acc, v183_lin, v162_data);
            tensorforge::fmacdpp16<9>(v176_acc, v183_lin, v163_data);
            tensorforge::fmacdpp16<10>(v176_acc, v183_lin, v164_data);
            tensorforge::fmacdpp16<11>(v176_acc, v183_lin, v165_data);
            tensorforge::fmacdpp16<12>(v176_acc, v183_lin, v166_data);
            tensorforge::fmacdpp16<13>(v176_acc, v183_lin, v167_data);
            tensorforge::fmacdpp16<14>(v176_acc, v183_lin, v168_data);
            tensorforge::fmacdpp16<15>(v176_acc, v183_lin, v169_data);
            float v184_lin = r10[2];
            tensorforge::fmacdpp16<0>(v176_acc, v184_lin, v170_data);
            tensorforge::fmacdpp16<1>(v176_acc, v184_lin, v171_data);
            tensorforge::fmacdpp16<2>(v176_acc, v184_lin, v172_data);
            tensorforge::fmacdpp16<3>(v176_acc, v184_lin, v173_data);
            tensorforge::fmacdpp16<4>(v177_acc, v184_lin, v162_data);
            tensorforge::fmacdpp16<5>(v177_acc, v184_lin, v163_data);
            tensorforge::fmacdpp16<6>(v177_acc, v184_lin, v164_data);
            tensorforge::fmacdpp16<7>(v177_acc, v184_lin, v165_data);
            tensorforge::fmacdpp16<8>(v177_acc, v184_lin, v166_data);
            tensorforge::fmacdpp16<9>(v177_acc, v184_lin, v167_data);
            tensorforge::fmacdpp16<10>(v177_acc, v184_lin, v168_data);
            tensorforge::fmacdpp16<11>(v177_acc, v184_lin, v169_data);
            tensorforge::fmacdpp16<12>(v177_acc, v184_lin, v170_data);
            tensorforge::fmacdpp16<13>(v177_acc, v184_lin, v171_data);
            tensorforge::fmacdpp16<14>(v177_acc, v184_lin, v172_data);
            tensorforge::fmacdpp16<15>(v177_acc, v184_lin, v173_data);
            float v185_lin = r10[3];
            tensorforge::fmacdpp16<0>(v178_acc, v185_lin, v162_data);
            tensorforge::fmacdpp16<1>(v178_acc, v185_lin, v163_data);
            tensorforge::fmacdpp16<2>(v178_acc, v185_lin, v164_data);
            tensorforge::fmacdpp16<3>(v178_acc, v185_lin, v165_data);
            tensorforge::fmacdpp16<4>(v178_acc, v185_lin, v166_data);
            tensorforge::fmacdpp16<5>(v178_acc, v185_lin, v167_data);
            tensorforge::fmacdpp16<6>(v178_acc, v185_lin, v168_data);
            tensorforge::fmacdpp16<7>(v178_acc, v185_lin, v169_data);
            tensorforge::fmacdpp16<8>(v178_acc, v185_lin, v170_data);
            tensorforge::fmacdpp16<9>(v178_acc, v185_lin, v171_data);
            tensorforge::fmacdpp16<10>(v178_acc, v185_lin, v172_data);
            tensorforge::fmacdpp16<11>(v178_acc, v185_lin, v173_data);
            tensorforge::fmacdpp16<12>(v179_acc, v185_lin, v162_data);
            tensorforge::fmacdpp16<13>(v179_acc, v185_lin, v163_data);
            tensorforge::fmacdpp16<14>(v179_acc, v185_lin, v164_data);
            tensorforge::fmacdpp16<15>(v179_acc, v185_lin, v165_data);
            float v186_lin = r10[4];
            tensorforge::fmacdpp16<0>(v179_acc, v186_lin, v166_data);
            tensorforge::fmacdpp16<1>(v179_acc, v186_lin, v167_data);
            tensorforge::fmacdpp16<2>(v179_acc, v186_lin, v168_data);
            tensorforge::fmacdpp16<3>(v179_acc, v186_lin, v169_data);
            tensorforge::fmacdpp16<4>(v179_acc, v186_lin, v170_data);
            tensorforge::fmacdpp16<5>(v179_acc, v186_lin, v171_data);
            tensorforge::fmacdpp16<6>(v179_acc, v186_lin, v172_data);
            tensorforge::fmacdpp16<7>(v179_acc, v186_lin, v173_data);
            tensorforge::fmacdpp16<8>(v180_acc, v186_lin, v162_data);
            tensorforge::fmacdpp16<9>(v180_acc, v186_lin, v163_data);
            tensorforge::fmacdpp16<10>(v180_acc, v186_lin, v164_data);
            tensorforge::fmacdpp16<11>(v180_acc, v186_lin, v165_data);
            tensorforge::fmacdpp16<12>(v180_acc, v186_lin, v166_data);
            tensorforge::fmacdpp16<13>(v180_acc, v186_lin, v167_data);
            tensorforge::fmacdpp16<14>(v180_acc, v186_lin, v168_data);
            tensorforge::fmacdpp16<15>(v180_acc, v186_lin, v169_data);
            float v187_lin = r10[5];
            tensorforge::fmacdpp16<0>(v180_acc, v187_lin, v170_data);
            tensorforge::fmacdpp16<1>(v180_acc, v187_lin, v171_data);
            tensorforge::fmacdpp16<2>(v180_acc, v187_lin, v172_data);
            tensorforge::fmacdpp16<3>(v180_acc, v187_lin, v173_data);
            tensorforge::fmacdpp16<4>(v181_acc, v187_lin, v162_data);
            tensorforge::fmacdpp16<5>(v181_acc, v187_lin, v163_data);
            tensorforge::fmacdpp16<6>(v181_acc, v187_lin, v164_data);
            tensorforge::fmacdpp16<7>(v181_acc, v187_lin, v165_data);
            tensorforge::fmacdpp16<8>(v181_acc, v187_lin, v166_data);
            tensorforge::fmacdpp16<9>(v181_acc, v187_lin, v167_data);
            tensorforge::fmacdpp16<10>(v181_acc, v187_lin, v168_data);
            tensorforge::fmacdpp16<11>(v181_acc, v187_lin, v169_data);
            tensorforge::fmacdpp16<12>(v181_acc, v187_lin, v170_data);
            tensorforge::fmacdpp16<13>(v181_acc, v187_lin, v171_data);
            tensorforge::fmacdpp16<14>(v181_acc, v187_lin, v172_data);
            tensorforge::fmacdpp16<15>(v181_acc, v187_lin, v173_data);
            ir11[0] = v174_acc;
            ir11[1] = v175_acc;
            ir11[2] = v176_acc;
            ir11[3] = v177_acc;
            ir11[4] = v178_acc;
            ir11[5] = v179_acc;
            ir11[6] = v180_acc;
            ir11[7] = v181_acc;
            if ((threadIdx.x % 16) < 12) {
              #pragma unroll
              for (int32_t v192_n1 = 0; v192_n1 < 8; ++v192_n1) {
                int32_t v193_a = 0 + v192_n1;
                float v195_data = ir11[v192_n1];
                int32_t v196_a = 0 + v192_n1;
                float v198_data = r8[v192_n1];
                int32_t v200_a = 0 + v192_n1;
                r11[v192_n1] = (v198_data + v195_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r11);
          int32_t v204_lead = threadIdx.x % 16;
          if (v204_lead < 12) {
            #pragma unroll
            for (int32_t v206_i1 = 0; v206_i1 < 8; ++v206_i1) {
              int32_t v207_a = 0 + v206_i1;
              float v209_data = r11[v206_i1];
              int32_t v216_a = v204_lead + (v206_i1 * 12);
              glb_m0[v216_a] = v209_data;
            }
          }
          ;
        }
      }
    }
  }
}

