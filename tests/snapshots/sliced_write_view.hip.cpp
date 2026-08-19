// === base name ===
kernel_939857c66e

// === header ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_939857c66e, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_939857c66e), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_939857c66e, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
    // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 10; v4_i1 < 13; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v14_a = v3_i0 + (v4_i1 - 10);
              r0[v14_a] = v12_data;
            }
          }
          float r1[13]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[1] = v32;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r1[2] = v64;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r1[3] = v96;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r1[4] = v128;
            float v160 = glb_m2[160 + threadIdx.x * 1];
            r1[5] = v160;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          auto& ir2 = r2;
          float v15_data = r0[0];
          float v16_data = r0[1];
          float v17_data = r0[2];
          float v18_acc{};
          float v19_lin = r1[0];
          float v20_bc = tensorforge::broadcast<32, 16, 0>(v19_lin);
          tensorforge::fmacdpp16<0>(v18_acc, v20_bc, v15_data);
          tensorforge::fmacdpp16<1>(v18_acc, v20_bc, v16_data);
          tensorforge::fmacdpp16<2>(v18_acc, v20_bc, v17_data);
          ir2[0] = v18_acc;
          // glb_m0 = store{r>g}(r2);
          int32_t v23_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v33_lead = v23_lead + (v24_i0 * 32);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 1; ++v25_i1) {
              int32_t v26_a = v24_i0 + v25_i1;
              float v28_data = r2[(v24_i0 + v25_i1)];
              int32_t v36_a = v33_lead + ((v25_i1 + 8) * 32);
              glb_m0[v36_a] = v28_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          int32_t v39_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v40_i0 = 0; v40_i0 < 1; ++v40_i0) {
            int32_t v46_lead = v39_lead + (v40_i0 * 32);
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 13; ++v41_i1) {
              int32_t v48_a = v46_lead + (v41_i1 * 32);
              float v49_data;
              {
                v49_data = glb_m0[v48_a];
              }
              int32_t v50_a = v40_i0 + v41_i1;
              r3[v50_a] = v49_data;
            }
          }
          float r4[13]{};
          {
            // r4 = load{g>r}(glb_m4);
            float v0 = glb_m4[0 + threadIdx.x * 1];
            r4[0] = v0;
            float v32 = glb_m4[32 + threadIdx.x * 1];
            r4[1] = v32;
            float v64 = glb_m4[64 + threadIdx.x * 1];
            r4[2] = v64;
            float v96 = glb_m4[96 + threadIdx.x * 1];
            r4[3] = v96;
            float v128 = glb_m4[128 + threadIdx.x * 1];
            r4[4] = v128;
            float v160 = glb_m4[160 + threadIdx.x * 1];
            r4[5] = v160;
          }
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          auto& ir5 = r5;
          float v51_data = r3[0];
          float v52_data = r3[1];
          float v53_data = r3[2];
          float v54_data = r3[3];
          float v55_data = r3[4];
          float v56_data = r3[5];
          float v57_data = r3[6];
          float v58_data = r3[7];
          float v59_data = r3[8];
          float v60_data = r3[9];
          float v61_data = r3[10];
          float v62_data = r3[11];
          float v63_data = r3[12];
          float v64_acc{};
          float v65_acc{};
          float v66_acc{};
          float v67_acc{};
          float v68_acc{};
          float v69_acc{};
          float v70_acc{};
          float v71_acc{};
          float v72_acc{};
          float v73_acc{};
          float v74_acc{};
          float v75_acc{};
          float v76_acc{};
          float v77_lin = r4[0];
          float v78_bc = tensorforge::broadcast<32, 16, 0>(v77_lin);
          tensorforge::fmacdpp16<0>(v64_acc, v78_bc, v51_data);
          tensorforge::fmacdpp16<1>(v64_acc, v78_bc, v52_data);
          tensorforge::fmacdpp16<2>(v64_acc, v78_bc, v53_data);
          tensorforge::fmacdpp16<3>(v64_acc, v78_bc, v54_data);
          tensorforge::fmacdpp16<4>(v64_acc, v78_bc, v55_data);
          tensorforge::fmacdpp16<5>(v64_acc, v78_bc, v56_data);
          tensorforge::fmacdpp16<6>(v64_acc, v78_bc, v57_data);
          tensorforge::fmacdpp16<7>(v64_acc, v78_bc, v58_data);
          tensorforge::fmacdpp16<8>(v64_acc, v78_bc, v59_data);
          tensorforge::fmacdpp16<9>(v64_acc, v78_bc, v60_data);
          tensorforge::fmacdpp16<10>(v64_acc, v78_bc, v61_data);
          tensorforge::fmacdpp16<11>(v64_acc, v78_bc, v62_data);
          tensorforge::fmacdpp16<12>(v64_acc, v78_bc, v63_data);
          tensorforge::fmacdpp16<13>(v65_acc, v78_bc, v51_data);
          tensorforge::fmacdpp16<14>(v65_acc, v78_bc, v52_data);
          tensorforge::fmacdpp16<15>(v65_acc, v78_bc, v53_data);
          float v79_bc = tensorforge::broadcast<32, 16, 1>(v77_lin);
          tensorforge::fmacdpp16<0>(v65_acc, v79_bc, v54_data);
          tensorforge::fmacdpp16<1>(v65_acc, v79_bc, v55_data);
          tensorforge::fmacdpp16<2>(v65_acc, v79_bc, v56_data);
          tensorforge::fmacdpp16<3>(v65_acc, v79_bc, v57_data);
          tensorforge::fmacdpp16<4>(v65_acc, v79_bc, v58_data);
          tensorforge::fmacdpp16<5>(v65_acc, v79_bc, v59_data);
          tensorforge::fmacdpp16<6>(v65_acc, v79_bc, v60_data);
          tensorforge::fmacdpp16<7>(v65_acc, v79_bc, v61_data);
          tensorforge::fmacdpp16<8>(v65_acc, v79_bc, v62_data);
          tensorforge::fmacdpp16<9>(v65_acc, v79_bc, v63_data);
          tensorforge::fmacdpp16<10>(v66_acc, v79_bc, v51_data);
          tensorforge::fmacdpp16<11>(v66_acc, v79_bc, v52_data);
          tensorforge::fmacdpp16<12>(v66_acc, v79_bc, v53_data);
          tensorforge::fmacdpp16<13>(v66_acc, v79_bc, v54_data);
          tensorforge::fmacdpp16<14>(v66_acc, v79_bc, v55_data);
          tensorforge::fmacdpp16<15>(v66_acc, v79_bc, v56_data);
          float v80_lin = r4[1];
          float v81_bc = tensorforge::broadcast<32, 16, 0>(v80_lin);
          tensorforge::fmacdpp16<0>(v66_acc, v81_bc, v57_data);
          tensorforge::fmacdpp16<1>(v66_acc, v81_bc, v58_data);
          tensorforge::fmacdpp16<2>(v66_acc, v81_bc, v59_data);
          tensorforge::fmacdpp16<3>(v66_acc, v81_bc, v60_data);
          tensorforge::fmacdpp16<4>(v66_acc, v81_bc, v61_data);
          tensorforge::fmacdpp16<5>(v66_acc, v81_bc, v62_data);
          tensorforge::fmacdpp16<6>(v66_acc, v81_bc, v63_data);
          tensorforge::fmacdpp16<7>(v67_acc, v81_bc, v51_data);
          tensorforge::fmacdpp16<8>(v67_acc, v81_bc, v52_data);
          tensorforge::fmacdpp16<9>(v67_acc, v81_bc, v53_data);
          tensorforge::fmacdpp16<10>(v67_acc, v81_bc, v54_data);
          tensorforge::fmacdpp16<11>(v67_acc, v81_bc, v55_data);
          tensorforge::fmacdpp16<12>(v67_acc, v81_bc, v56_data);
          tensorforge::fmacdpp16<13>(v67_acc, v81_bc, v57_data);
          tensorforge::fmacdpp16<14>(v67_acc, v81_bc, v58_data);
          tensorforge::fmacdpp16<15>(v67_acc, v81_bc, v59_data);
          float v82_bc = tensorforge::broadcast<32, 16, 1>(v80_lin);
          tensorforge::fmacdpp16<0>(v67_acc, v82_bc, v60_data);
          tensorforge::fmacdpp16<1>(v67_acc, v82_bc, v61_data);
          tensorforge::fmacdpp16<2>(v67_acc, v82_bc, v62_data);
          tensorforge::fmacdpp16<3>(v67_acc, v82_bc, v63_data);
          tensorforge::fmacdpp16<4>(v68_acc, v82_bc, v51_data);
          tensorforge::fmacdpp16<5>(v68_acc, v82_bc, v52_data);
          tensorforge::fmacdpp16<6>(v68_acc, v82_bc, v53_data);
          tensorforge::fmacdpp16<7>(v68_acc, v82_bc, v54_data);
          tensorforge::fmacdpp16<8>(v68_acc, v82_bc, v55_data);
          tensorforge::fmacdpp16<9>(v68_acc, v82_bc, v56_data);
          tensorforge::fmacdpp16<10>(v68_acc, v82_bc, v57_data);
          tensorforge::fmacdpp16<11>(v68_acc, v82_bc, v58_data);
          tensorforge::fmacdpp16<12>(v68_acc, v82_bc, v59_data);
          tensorforge::fmacdpp16<13>(v68_acc, v82_bc, v60_data);
          tensorforge::fmacdpp16<14>(v68_acc, v82_bc, v61_data);
          tensorforge::fmacdpp16<15>(v68_acc, v82_bc, v62_data);
          float v83_lin = r4[2];
          float v84_bc = tensorforge::broadcast<32, 16, 0>(v83_lin);
          tensorforge::fmacdpp16<0>(v68_acc, v84_bc, v63_data);
          tensorforge::fmacdpp16<1>(v69_acc, v84_bc, v51_data);
          tensorforge::fmacdpp16<2>(v69_acc, v84_bc, v52_data);
          tensorforge::fmacdpp16<3>(v69_acc, v84_bc, v53_data);
          tensorforge::fmacdpp16<4>(v69_acc, v84_bc, v54_data);
          tensorforge::fmacdpp16<5>(v69_acc, v84_bc, v55_data);
          tensorforge::fmacdpp16<6>(v69_acc, v84_bc, v56_data);
          tensorforge::fmacdpp16<7>(v69_acc, v84_bc, v57_data);
          tensorforge::fmacdpp16<8>(v69_acc, v84_bc, v58_data);
          tensorforge::fmacdpp16<9>(v69_acc, v84_bc, v59_data);
          tensorforge::fmacdpp16<10>(v69_acc, v84_bc, v60_data);
          tensorforge::fmacdpp16<11>(v69_acc, v84_bc, v61_data);
          tensorforge::fmacdpp16<12>(v69_acc, v84_bc, v62_data);
          tensorforge::fmacdpp16<13>(v69_acc, v84_bc, v63_data);
          tensorforge::fmacdpp16<14>(v70_acc, v84_bc, v51_data);
          tensorforge::fmacdpp16<15>(v70_acc, v84_bc, v52_data);
          float v85_bc = tensorforge::broadcast<32, 16, 1>(v83_lin);
          tensorforge::fmacdpp16<0>(v70_acc, v85_bc, v53_data);
          tensorforge::fmacdpp16<1>(v70_acc, v85_bc, v54_data);
          tensorforge::fmacdpp16<2>(v70_acc, v85_bc, v55_data);
          tensorforge::fmacdpp16<3>(v70_acc, v85_bc, v56_data);
          tensorforge::fmacdpp16<4>(v70_acc, v85_bc, v57_data);
          tensorforge::fmacdpp16<5>(v70_acc, v85_bc, v58_data);
          tensorforge::fmacdpp16<6>(v70_acc, v85_bc, v59_data);
          tensorforge::fmacdpp16<7>(v70_acc, v85_bc, v60_data);
          tensorforge::fmacdpp16<8>(v70_acc, v85_bc, v61_data);
          tensorforge::fmacdpp16<9>(v70_acc, v85_bc, v62_data);
          tensorforge::fmacdpp16<10>(v70_acc, v85_bc, v63_data);
          tensorforge::fmacdpp16<11>(v71_acc, v85_bc, v51_data);
          tensorforge::fmacdpp16<12>(v71_acc, v85_bc, v52_data);
          tensorforge::fmacdpp16<13>(v71_acc, v85_bc, v53_data);
          tensorforge::fmacdpp16<14>(v71_acc, v85_bc, v54_data);
          tensorforge::fmacdpp16<15>(v71_acc, v85_bc, v55_data);
          float v86_lin = r4[3];
          float v87_bc = tensorforge::broadcast<32, 16, 0>(v86_lin);
          tensorforge::fmacdpp16<0>(v71_acc, v87_bc, v56_data);
          tensorforge::fmacdpp16<1>(v71_acc, v87_bc, v57_data);
          tensorforge::fmacdpp16<2>(v71_acc, v87_bc, v58_data);
          tensorforge::fmacdpp16<3>(v71_acc, v87_bc, v59_data);
          tensorforge::fmacdpp16<4>(v71_acc, v87_bc, v60_data);
          tensorforge::fmacdpp16<5>(v71_acc, v87_bc, v61_data);
          tensorforge::fmacdpp16<6>(v71_acc, v87_bc, v62_data);
          tensorforge::fmacdpp16<7>(v71_acc, v87_bc, v63_data);
          tensorforge::fmacdpp16<8>(v72_acc, v87_bc, v51_data);
          tensorforge::fmacdpp16<9>(v72_acc, v87_bc, v52_data);
          tensorforge::fmacdpp16<10>(v72_acc, v87_bc, v53_data);
          tensorforge::fmacdpp16<11>(v72_acc, v87_bc, v54_data);
          tensorforge::fmacdpp16<12>(v72_acc, v87_bc, v55_data);
          tensorforge::fmacdpp16<13>(v72_acc, v87_bc, v56_data);
          tensorforge::fmacdpp16<14>(v72_acc, v87_bc, v57_data);
          tensorforge::fmacdpp16<15>(v72_acc, v87_bc, v58_data);
          float v88_bc = tensorforge::broadcast<32, 16, 1>(v86_lin);
          tensorforge::fmacdpp16<0>(v72_acc, v88_bc, v59_data);
          tensorforge::fmacdpp16<1>(v72_acc, v88_bc, v60_data);
          tensorforge::fmacdpp16<2>(v72_acc, v88_bc, v61_data);
          tensorforge::fmacdpp16<3>(v72_acc, v88_bc, v62_data);
          tensorforge::fmacdpp16<4>(v72_acc, v88_bc, v63_data);
          tensorforge::fmacdpp16<5>(v73_acc, v88_bc, v51_data);
          tensorforge::fmacdpp16<6>(v73_acc, v88_bc, v52_data);
          tensorforge::fmacdpp16<7>(v73_acc, v88_bc, v53_data);
          tensorforge::fmacdpp16<8>(v73_acc, v88_bc, v54_data);
          tensorforge::fmacdpp16<9>(v73_acc, v88_bc, v55_data);
          tensorforge::fmacdpp16<10>(v73_acc, v88_bc, v56_data);
          tensorforge::fmacdpp16<11>(v73_acc, v88_bc, v57_data);
          tensorforge::fmacdpp16<12>(v73_acc, v88_bc, v58_data);
          tensorforge::fmacdpp16<13>(v73_acc, v88_bc, v59_data);
          tensorforge::fmacdpp16<14>(v73_acc, v88_bc, v60_data);
          tensorforge::fmacdpp16<15>(v73_acc, v88_bc, v61_data);
          float v89_lin = r4[4];
          float v90_bc = tensorforge::broadcast<32, 16, 0>(v89_lin);
          tensorforge::fmacdpp16<0>(v73_acc, v90_bc, v62_data);
          tensorforge::fmacdpp16<1>(v73_acc, v90_bc, v63_data);
          tensorforge::fmacdpp16<2>(v74_acc, v90_bc, v51_data);
          tensorforge::fmacdpp16<3>(v74_acc, v90_bc, v52_data);
          tensorforge::fmacdpp16<4>(v74_acc, v90_bc, v53_data);
          tensorforge::fmacdpp16<5>(v74_acc, v90_bc, v54_data);
          tensorforge::fmacdpp16<6>(v74_acc, v90_bc, v55_data);
          tensorforge::fmacdpp16<7>(v74_acc, v90_bc, v56_data);
          tensorforge::fmacdpp16<8>(v74_acc, v90_bc, v57_data);
          tensorforge::fmacdpp16<9>(v74_acc, v90_bc, v58_data);
          tensorforge::fmacdpp16<10>(v74_acc, v90_bc, v59_data);
          tensorforge::fmacdpp16<11>(v74_acc, v90_bc, v60_data);
          tensorforge::fmacdpp16<12>(v74_acc, v90_bc, v61_data);
          tensorforge::fmacdpp16<13>(v74_acc, v90_bc, v62_data);
          tensorforge::fmacdpp16<14>(v74_acc, v90_bc, v63_data);
          tensorforge::fmacdpp16<15>(v75_acc, v90_bc, v51_data);
          float v91_bc = tensorforge::broadcast<32, 16, 1>(v89_lin);
          tensorforge::fmacdpp16<0>(v75_acc, v91_bc, v52_data);
          tensorforge::fmacdpp16<1>(v75_acc, v91_bc, v53_data);
          tensorforge::fmacdpp16<2>(v75_acc, v91_bc, v54_data);
          tensorforge::fmacdpp16<3>(v75_acc, v91_bc, v55_data);
          tensorforge::fmacdpp16<4>(v75_acc, v91_bc, v56_data);
          tensorforge::fmacdpp16<5>(v75_acc, v91_bc, v57_data);
          tensorforge::fmacdpp16<6>(v75_acc, v91_bc, v58_data);
          tensorforge::fmacdpp16<7>(v75_acc, v91_bc, v59_data);
          tensorforge::fmacdpp16<8>(v75_acc, v91_bc, v60_data);
          tensorforge::fmacdpp16<9>(v75_acc, v91_bc, v61_data);
          tensorforge::fmacdpp16<10>(v75_acc, v91_bc, v62_data);
          tensorforge::fmacdpp16<11>(v75_acc, v91_bc, v63_data);
          tensorforge::fmacdpp16<12>(v76_acc, v91_bc, v51_data);
          tensorforge::fmacdpp16<13>(v76_acc, v91_bc, v52_data);
          tensorforge::fmacdpp16<14>(v76_acc, v91_bc, v53_data);
          tensorforge::fmacdpp16<15>(v76_acc, v91_bc, v54_data);
          float v92_lin = r4[5];
          float v93_bc = tensorforge::broadcast<32, 16, 0>(v92_lin);
          tensorforge::fmacdpp16<0>(v76_acc, v93_bc, v55_data);
          tensorforge::fmacdpp16<1>(v76_acc, v93_bc, v56_data);
          tensorforge::fmacdpp16<2>(v76_acc, v93_bc, v57_data);
          tensorforge::fmacdpp16<3>(v76_acc, v93_bc, v58_data);
          tensorforge::fmacdpp16<4>(v76_acc, v93_bc, v59_data);
          tensorforge::fmacdpp16<5>(v76_acc, v93_bc, v60_data);
          tensorforge::fmacdpp16<6>(v76_acc, v93_bc, v61_data);
          tensorforge::fmacdpp16<7>(v76_acc, v93_bc, v62_data);
          tensorforge::fmacdpp16<8>(v76_acc, v93_bc, v63_data);
          ir5[0] = v64_acc;
          ir5[1] = v65_acc;
          ir5[2] = v66_acc;
          ir5[3] = v67_acc;
          ir5[4] = v68_acc;
          ir5[5] = v69_acc;
          ir5[6] = v70_acc;
          ir5[7] = v71_acc;
          ir5[8] = v72_acc;
          ir5[9] = v73_acc;
          ir5[10] = v74_acc;
          ir5[11] = v75_acc;
          ir5[12] = v76_acc;
          // glb_m3 = store{r>g}(r5);
          int32_t v96_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v97_i0 = 0; v97_i0 < 1; ++v97_i0) {
            int32_t v106_lead = v96_lead + (v97_i0 * 32);
            #pragma unroll
            for (int32_t v98_i1 = 0; v98_i1 < 13; ++v98_i1) {
              int32_t v99_a = v97_i0 + v98_i1;
              float v101_data = r5[(v97_i0 + v98_i1)];
              int32_t v108_a = v106_lead + (v98_i1 * 32);
              glb_m3[v108_a] = v101_data;
            }
          }
          ;
        }
      }
    }
  }
}

