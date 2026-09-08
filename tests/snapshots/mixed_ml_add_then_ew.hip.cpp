// === base name ===
kernel_609dd06e89

// === header ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_609dd06e89, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_609dd06e89), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_609dd06e89, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // m4 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // C = abs(TMP)
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      __syncthreads();
      float* __restrict__ s0 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v16_lead = threadIdx.x % 64;
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 8; ++v18_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m0[(v16_lead + (v18_i1 * 8))]);
              r0[v18_i1] = v26_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m1);
          float v29_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v29_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[8]{};
          // r3 = load{g>r}(glb_m2);
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
              float v43_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v35_i1 * 8))]);
              r3[v35_i1] = v43_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v46_data = r0[0];
          float v47_data = r0[1];
          float v48_data = r0[2];
          float v49_data = r0[3];
          float v50_data = r0[4];
          float v51_data = r0[5];
          float v52_data = r0[6];
          float v53_data = r0[7];
          float v54_acc{};
          float v55_acc{};
          float v56_acc{};
          float v57_acc{};
          float v58_acc{};
          float v59_acc{};
          float v60_acc{};
          float v61_acc{};
          float v62_lin = r1[0];
          float v63_bc = tensorforge::broadcast<64, 16, 0>(v62_lin);
          tensorforge::fmacdpp16<0>(v54_acc, v63_bc, v46_data);
          tensorforge::fmacdpp16<1>(v54_acc, v63_bc, v47_data);
          tensorforge::fmacdpp16<2>(v54_acc, v63_bc, v48_data);
          tensorforge::fmacdpp16<3>(v54_acc, v63_bc, v49_data);
          tensorforge::fmacdpp16<4>(v54_acc, v63_bc, v50_data);
          tensorforge::fmacdpp16<5>(v54_acc, v63_bc, v51_data);
          tensorforge::fmacdpp16<6>(v54_acc, v63_bc, v52_data);
          tensorforge::fmacdpp16<7>(v54_acc, v63_bc, v53_data);
          tensorforge::fmacdpp16<8>(v55_acc, v63_bc, v46_data);
          tensorforge::fmacdpp16<9>(v55_acc, v63_bc, v47_data);
          tensorforge::fmacdpp16<10>(v55_acc, v63_bc, v48_data);
          tensorforge::fmacdpp16<11>(v55_acc, v63_bc, v49_data);
          tensorforge::fmacdpp16<12>(v55_acc, v63_bc, v50_data);
          tensorforge::fmacdpp16<13>(v55_acc, v63_bc, v51_data);
          tensorforge::fmacdpp16<14>(v55_acc, v63_bc, v52_data);
          tensorforge::fmacdpp16<15>(v55_acc, v63_bc, v53_data);
          float v64_bc = tensorforge::broadcast<64, 16, 1>(v62_lin);
          tensorforge::fmacdpp16<0>(v56_acc, v64_bc, v46_data);
          tensorforge::fmacdpp16<1>(v56_acc, v64_bc, v47_data);
          tensorforge::fmacdpp16<2>(v56_acc, v64_bc, v48_data);
          tensorforge::fmacdpp16<3>(v56_acc, v64_bc, v49_data);
          tensorforge::fmacdpp16<4>(v56_acc, v64_bc, v50_data);
          tensorforge::fmacdpp16<5>(v56_acc, v64_bc, v51_data);
          tensorforge::fmacdpp16<6>(v56_acc, v64_bc, v52_data);
          tensorforge::fmacdpp16<7>(v56_acc, v64_bc, v53_data);
          tensorforge::fmacdpp16<8>(v57_acc, v64_bc, v46_data);
          tensorforge::fmacdpp16<9>(v57_acc, v64_bc, v47_data);
          tensorforge::fmacdpp16<10>(v57_acc, v64_bc, v48_data);
          tensorforge::fmacdpp16<11>(v57_acc, v64_bc, v49_data);
          tensorforge::fmacdpp16<12>(v57_acc, v64_bc, v50_data);
          tensorforge::fmacdpp16<13>(v57_acc, v64_bc, v51_data);
          tensorforge::fmacdpp16<14>(v57_acc, v64_bc, v52_data);
          tensorforge::fmacdpp16<15>(v57_acc, v64_bc, v53_data);
          float v65_bc = tensorforge::broadcast<64, 16, 2>(v62_lin);
          tensorforge::fmacdpp16<0>(v58_acc, v65_bc, v46_data);
          tensorforge::fmacdpp16<1>(v58_acc, v65_bc, v47_data);
          tensorforge::fmacdpp16<2>(v58_acc, v65_bc, v48_data);
          tensorforge::fmacdpp16<3>(v58_acc, v65_bc, v49_data);
          tensorforge::fmacdpp16<4>(v58_acc, v65_bc, v50_data);
          tensorforge::fmacdpp16<5>(v58_acc, v65_bc, v51_data);
          tensorforge::fmacdpp16<6>(v58_acc, v65_bc, v52_data);
          tensorforge::fmacdpp16<7>(v58_acc, v65_bc, v53_data);
          tensorforge::fmacdpp16<8>(v59_acc, v65_bc, v46_data);
          tensorforge::fmacdpp16<9>(v59_acc, v65_bc, v47_data);
          tensorforge::fmacdpp16<10>(v59_acc, v65_bc, v48_data);
          tensorforge::fmacdpp16<11>(v59_acc, v65_bc, v49_data);
          tensorforge::fmacdpp16<12>(v59_acc, v65_bc, v50_data);
          tensorforge::fmacdpp16<13>(v59_acc, v65_bc, v51_data);
          tensorforge::fmacdpp16<14>(v59_acc, v65_bc, v52_data);
          tensorforge::fmacdpp16<15>(v59_acc, v65_bc, v53_data);
          float v66_bc = tensorforge::broadcast<64, 16, 3>(v62_lin);
          tensorforge::fmacdpp16<0>(v60_acc, v66_bc, v46_data);
          tensorforge::fmacdpp16<1>(v60_acc, v66_bc, v47_data);
          tensorforge::fmacdpp16<2>(v60_acc, v66_bc, v48_data);
          tensorforge::fmacdpp16<3>(v60_acc, v66_bc, v49_data);
          tensorforge::fmacdpp16<4>(v60_acc, v66_bc, v50_data);
          tensorforge::fmacdpp16<5>(v60_acc, v66_bc, v51_data);
          tensorforge::fmacdpp16<6>(v60_acc, v66_bc, v52_data);
          tensorforge::fmacdpp16<7>(v60_acc, v66_bc, v53_data);
          tensorforge::fmacdpp16<8>(v61_acc, v66_bc, v46_data);
          tensorforge::fmacdpp16<9>(v61_acc, v66_bc, v47_data);
          tensorforge::fmacdpp16<10>(v61_acc, v66_bc, v48_data);
          tensorforge::fmacdpp16<11>(v61_acc, v66_bc, v49_data);
          tensorforge::fmacdpp16<12>(v61_acc, v66_bc, v50_data);
          tensorforge::fmacdpp16<13>(v61_acc, v66_bc, v51_data);
          tensorforge::fmacdpp16<14>(v61_acc, v66_bc, v52_data);
          tensorforge::fmacdpp16<15>(v61_acc, v66_bc, v53_data);
          r2[0] = v54_acc;
          r2[1] = v55_acc;
          r2[2] = v56_acc;
          r2[3] = v57_acc;
          r2[4] = v58_acc;
          r2[5] = v59_acc;
          r2[6] = v60_acc;
          r2[7] = v61_acc;
          float r4[8]{};
          // r4 = load{g>r}(glb_m3);
          float v68_lin = glb_m3[0 + threadIdx.x * 1];
          r4[0] = v68_lin;
          // wait(r3 = load{g>r}(glb_m2););
          // wait(r4 = load{g>r}(glb_m3););
          float r5[8]{};
          // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir5[8]{};
          float v71_data = r3[0];
          float v72_data = r3[1];
          float v73_data = r3[2];
          float v74_data = r3[3];
          float v75_data = r3[4];
          float v76_data = r3[5];
          float v77_data = r3[6];
          float v78_data = r3[7];
          float v79_acc{};
          float v80_acc{};
          float v81_acc{};
          float v82_acc{};
          float v83_acc{};
          float v84_acc{};
          float v85_acc{};
          float v86_acc{};
          float v87_lin = r4[0];
          float v88_bc = tensorforge::broadcast<64, 16, 0>(v87_lin);
          tensorforge::fmacdpp16<0>(v79_acc, v88_bc, v71_data);
          tensorforge::fmacdpp16<1>(v79_acc, v88_bc, v72_data);
          tensorforge::fmacdpp16<2>(v79_acc, v88_bc, v73_data);
          tensorforge::fmacdpp16<3>(v79_acc, v88_bc, v74_data);
          tensorforge::fmacdpp16<4>(v79_acc, v88_bc, v75_data);
          tensorforge::fmacdpp16<5>(v79_acc, v88_bc, v76_data);
          tensorforge::fmacdpp16<6>(v79_acc, v88_bc, v77_data);
          tensorforge::fmacdpp16<7>(v79_acc, v88_bc, v78_data);
          tensorforge::fmacdpp16<8>(v80_acc, v88_bc, v71_data);
          tensorforge::fmacdpp16<9>(v80_acc, v88_bc, v72_data);
          tensorforge::fmacdpp16<10>(v80_acc, v88_bc, v73_data);
          tensorforge::fmacdpp16<11>(v80_acc, v88_bc, v74_data);
          tensorforge::fmacdpp16<12>(v80_acc, v88_bc, v75_data);
          tensorforge::fmacdpp16<13>(v80_acc, v88_bc, v76_data);
          tensorforge::fmacdpp16<14>(v80_acc, v88_bc, v77_data);
          tensorforge::fmacdpp16<15>(v80_acc, v88_bc, v78_data);
          float v89_bc = tensorforge::broadcast<64, 16, 1>(v87_lin);
          tensorforge::fmacdpp16<0>(v81_acc, v89_bc, v71_data);
          tensorforge::fmacdpp16<1>(v81_acc, v89_bc, v72_data);
          tensorforge::fmacdpp16<2>(v81_acc, v89_bc, v73_data);
          tensorforge::fmacdpp16<3>(v81_acc, v89_bc, v74_data);
          tensorforge::fmacdpp16<4>(v81_acc, v89_bc, v75_data);
          tensorforge::fmacdpp16<5>(v81_acc, v89_bc, v76_data);
          tensorforge::fmacdpp16<6>(v81_acc, v89_bc, v77_data);
          tensorforge::fmacdpp16<7>(v81_acc, v89_bc, v78_data);
          tensorforge::fmacdpp16<8>(v82_acc, v89_bc, v71_data);
          tensorforge::fmacdpp16<9>(v82_acc, v89_bc, v72_data);
          tensorforge::fmacdpp16<10>(v82_acc, v89_bc, v73_data);
          tensorforge::fmacdpp16<11>(v82_acc, v89_bc, v74_data);
          tensorforge::fmacdpp16<12>(v82_acc, v89_bc, v75_data);
          tensorforge::fmacdpp16<13>(v82_acc, v89_bc, v76_data);
          tensorforge::fmacdpp16<14>(v82_acc, v89_bc, v77_data);
          tensorforge::fmacdpp16<15>(v82_acc, v89_bc, v78_data);
          float v90_bc = tensorforge::broadcast<64, 16, 2>(v87_lin);
          tensorforge::fmacdpp16<0>(v83_acc, v90_bc, v71_data);
          tensorforge::fmacdpp16<1>(v83_acc, v90_bc, v72_data);
          tensorforge::fmacdpp16<2>(v83_acc, v90_bc, v73_data);
          tensorforge::fmacdpp16<3>(v83_acc, v90_bc, v74_data);
          tensorforge::fmacdpp16<4>(v83_acc, v90_bc, v75_data);
          tensorforge::fmacdpp16<5>(v83_acc, v90_bc, v76_data);
          tensorforge::fmacdpp16<6>(v83_acc, v90_bc, v77_data);
          tensorforge::fmacdpp16<7>(v83_acc, v90_bc, v78_data);
          tensorforge::fmacdpp16<8>(v84_acc, v90_bc, v71_data);
          tensorforge::fmacdpp16<9>(v84_acc, v90_bc, v72_data);
          tensorforge::fmacdpp16<10>(v84_acc, v90_bc, v73_data);
          tensorforge::fmacdpp16<11>(v84_acc, v90_bc, v74_data);
          tensorforge::fmacdpp16<12>(v84_acc, v90_bc, v75_data);
          tensorforge::fmacdpp16<13>(v84_acc, v90_bc, v76_data);
          tensorforge::fmacdpp16<14>(v84_acc, v90_bc, v77_data);
          tensorforge::fmacdpp16<15>(v84_acc, v90_bc, v78_data);
          float v91_bc = tensorforge::broadcast<64, 16, 3>(v87_lin);
          tensorforge::fmacdpp16<0>(v85_acc, v91_bc, v71_data);
          tensorforge::fmacdpp16<1>(v85_acc, v91_bc, v72_data);
          tensorforge::fmacdpp16<2>(v85_acc, v91_bc, v73_data);
          tensorforge::fmacdpp16<3>(v85_acc, v91_bc, v74_data);
          tensorforge::fmacdpp16<4>(v85_acc, v91_bc, v75_data);
          tensorforge::fmacdpp16<5>(v85_acc, v91_bc, v76_data);
          tensorforge::fmacdpp16<6>(v85_acc, v91_bc, v77_data);
          tensorforge::fmacdpp16<7>(v85_acc, v91_bc, v78_data);
          tensorforge::fmacdpp16<8>(v86_acc, v91_bc, v71_data);
          tensorforge::fmacdpp16<9>(v86_acc, v91_bc, v72_data);
          tensorforge::fmacdpp16<10>(v86_acc, v91_bc, v73_data);
          tensorforge::fmacdpp16<11>(v86_acc, v91_bc, v74_data);
          tensorforge::fmacdpp16<12>(v86_acc, v91_bc, v75_data);
          tensorforge::fmacdpp16<13>(v86_acc, v91_bc, v76_data);
          tensorforge::fmacdpp16<14>(v86_acc, v91_bc, v77_data);
          tensorforge::fmacdpp16<15>(v86_acc, v91_bc, v78_data);
          ir5[0] = v79_acc;
          ir5[1] = v80_acc;
          ir5[2] = v81_acc;
          ir5[3] = v82_acc;
          ir5[4] = v83_acc;
          ir5[5] = v84_acc;
          ir5[6] = v85_acc;
          ir5[7] = v86_acc;
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v96_n1 = 0; v96_n1 < 8; ++v96_n1) {
              float v98_data = ir5[v96_n1];
              float v100_data = r2[v96_n1];
              r5[v96_n1] = (v100_data + v98_data);
            }
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v107_i1 = 0; v107_i1 < 8; ++v107_i1) {
              float v109_data = r5[v107_i1];
              int32_t v116_a = v16_lead + (v107_i1 * 8);
              s0[(v116_a ^ ((v116_a >> 5) & 31))] = v109_data;
            }
          }
          // glb_m4 = abs(s0)
          if (v16_lead < 8) {
            #pragma unroll
            for (int32_t v124_k1 = 0; v124_k1 < 8; ++v124_k1) {
              int32_t v130_a = v124_k1 * 8;
              int32_t v131_a = v16_lead + v130_a;
              float v135_data = s0[(v131_a ^ ((v131_a >> 5) & 31))];
              glb_m4[(v16_lead + v130_a)] = (fabsf(v135_data));
            }
          }
        }
      }
    }
  }
}

