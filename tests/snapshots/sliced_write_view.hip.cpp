// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
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
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 10; v4_i1 < 13; ++v4_i1) {
              int32_t v10_a = v4_i1 * 32;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + v10_a)]);
              int32_t v21_a = v3_i0 + (v4_i1 - 10);
              r0[v21_a] = v19_data;
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
          float v22_data = r0[0];
          float v23_data = r0[1];
          float v24_data = r0[2];
          float v25_acc{};
          float v26_lin = r1[0];
          float v27_bc = tensorforge::broadcast<32, 16, 0>(v26_lin);
          tensorforge::fmacdpp16<0>(v25_acc, v27_bc, v22_data);
          tensorforge::fmacdpp16<1>(v25_acc, v27_bc, v23_data);
          tensorforge::fmacdpp16<2>(v25_acc, v27_bc, v24_data);
          ir2[0] = v25_acc;
          // glb_m0 = store{r>g}(r2);
          int32_t v30_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
            int32_t v40_lead = v30_lead + (v31_i0 * 32);
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 1; ++v32_i1) {
              int32_t v33_a = v31_i0 + v32_i1;
              float v35_data = r2[(v31_i0 + v32_i1)];
              int32_t v43_a = v40_lead + ((v32_i1 + 8) * 32);
              glb_m0[v43_a] = v35_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          int32_t v46_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v47_i0 = 0; v47_i0 < 1; ++v47_i0) {
            int32_t v52_lead = v47_i0 * 32;
            int32_t v53_lead = v46_lead + v52_lead;
            int32_t v60_lead = v46_lead + v52_lead;
            #pragma unroll
            for (int32_t v48_i1 = 0; v48_i1 < 13; ++v48_i1) {
              int32_t v54_a = v48_i1 * 32;
              int32_t v55_a = v53_lead + v54_a;
              float v63_data = glb_m0[(v60_lead + v54_a)];
              int32_t v64_a = v47_i0 + v48_i1;
              r3[v64_a] = v63_data;
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
          float v65_data = r3[0];
          float v66_data = r3[1];
          float v67_data = r3[2];
          float v68_data = r3[3];
          float v69_data = r3[4];
          float v70_data = r3[5];
          float v71_data = r3[6];
          float v72_data = r3[7];
          float v73_data = r3[8];
          float v74_data = r3[9];
          float v75_data = r3[10];
          float v76_data = r3[11];
          float v77_data = r3[12];
          float v78_acc{};
          float v79_acc{};
          float v80_acc{};
          float v81_acc{};
          float v82_acc{};
          float v83_acc{};
          float v84_acc{};
          float v85_acc{};
          float v86_acc{};
          float v87_acc{};
          float v88_acc{};
          float v89_acc{};
          float v90_acc{};
          float v91_lin = r4[0];
          float v92_bc = tensorforge::broadcast<32, 16, 0>(v91_lin);
          tensorforge::fmacdpp16<0>(v78_acc, v92_bc, v65_data);
          tensorforge::fmacdpp16<1>(v78_acc, v92_bc, v66_data);
          tensorforge::fmacdpp16<2>(v78_acc, v92_bc, v67_data);
          tensorforge::fmacdpp16<3>(v78_acc, v92_bc, v68_data);
          tensorforge::fmacdpp16<4>(v78_acc, v92_bc, v69_data);
          tensorforge::fmacdpp16<5>(v78_acc, v92_bc, v70_data);
          tensorforge::fmacdpp16<6>(v78_acc, v92_bc, v71_data);
          tensorforge::fmacdpp16<7>(v78_acc, v92_bc, v72_data);
          tensorforge::fmacdpp16<8>(v78_acc, v92_bc, v73_data);
          tensorforge::fmacdpp16<9>(v78_acc, v92_bc, v74_data);
          tensorforge::fmacdpp16<10>(v78_acc, v92_bc, v75_data);
          tensorforge::fmacdpp16<11>(v78_acc, v92_bc, v76_data);
          tensorforge::fmacdpp16<12>(v78_acc, v92_bc, v77_data);
          tensorforge::fmacdpp16<13>(v79_acc, v92_bc, v65_data);
          tensorforge::fmacdpp16<14>(v79_acc, v92_bc, v66_data);
          tensorforge::fmacdpp16<15>(v79_acc, v92_bc, v67_data);
          float v93_bc = tensorforge::broadcast<32, 16, 1>(v91_lin);
          tensorforge::fmacdpp16<0>(v79_acc, v93_bc, v68_data);
          tensorforge::fmacdpp16<1>(v79_acc, v93_bc, v69_data);
          tensorforge::fmacdpp16<2>(v79_acc, v93_bc, v70_data);
          tensorforge::fmacdpp16<3>(v79_acc, v93_bc, v71_data);
          tensorforge::fmacdpp16<4>(v79_acc, v93_bc, v72_data);
          tensorforge::fmacdpp16<5>(v79_acc, v93_bc, v73_data);
          tensorforge::fmacdpp16<6>(v79_acc, v93_bc, v74_data);
          tensorforge::fmacdpp16<7>(v79_acc, v93_bc, v75_data);
          tensorforge::fmacdpp16<8>(v79_acc, v93_bc, v76_data);
          tensorforge::fmacdpp16<9>(v79_acc, v93_bc, v77_data);
          tensorforge::fmacdpp16<10>(v80_acc, v93_bc, v65_data);
          tensorforge::fmacdpp16<11>(v80_acc, v93_bc, v66_data);
          tensorforge::fmacdpp16<12>(v80_acc, v93_bc, v67_data);
          tensorforge::fmacdpp16<13>(v80_acc, v93_bc, v68_data);
          tensorforge::fmacdpp16<14>(v80_acc, v93_bc, v69_data);
          tensorforge::fmacdpp16<15>(v80_acc, v93_bc, v70_data);
          float v94_lin = r4[1];
          float v95_bc = tensorforge::broadcast<32, 16, 0>(v94_lin);
          tensorforge::fmacdpp16<0>(v80_acc, v95_bc, v71_data);
          tensorforge::fmacdpp16<1>(v80_acc, v95_bc, v72_data);
          tensorforge::fmacdpp16<2>(v80_acc, v95_bc, v73_data);
          tensorforge::fmacdpp16<3>(v80_acc, v95_bc, v74_data);
          tensorforge::fmacdpp16<4>(v80_acc, v95_bc, v75_data);
          tensorforge::fmacdpp16<5>(v80_acc, v95_bc, v76_data);
          tensorforge::fmacdpp16<6>(v80_acc, v95_bc, v77_data);
          tensorforge::fmacdpp16<7>(v81_acc, v95_bc, v65_data);
          tensorforge::fmacdpp16<8>(v81_acc, v95_bc, v66_data);
          tensorforge::fmacdpp16<9>(v81_acc, v95_bc, v67_data);
          tensorforge::fmacdpp16<10>(v81_acc, v95_bc, v68_data);
          tensorforge::fmacdpp16<11>(v81_acc, v95_bc, v69_data);
          tensorforge::fmacdpp16<12>(v81_acc, v95_bc, v70_data);
          tensorforge::fmacdpp16<13>(v81_acc, v95_bc, v71_data);
          tensorforge::fmacdpp16<14>(v81_acc, v95_bc, v72_data);
          tensorforge::fmacdpp16<15>(v81_acc, v95_bc, v73_data);
          float v96_bc = tensorforge::broadcast<32, 16, 1>(v94_lin);
          tensorforge::fmacdpp16<0>(v81_acc, v96_bc, v74_data);
          tensorforge::fmacdpp16<1>(v81_acc, v96_bc, v75_data);
          tensorforge::fmacdpp16<2>(v81_acc, v96_bc, v76_data);
          tensorforge::fmacdpp16<3>(v81_acc, v96_bc, v77_data);
          tensorforge::fmacdpp16<4>(v82_acc, v96_bc, v65_data);
          tensorforge::fmacdpp16<5>(v82_acc, v96_bc, v66_data);
          tensorforge::fmacdpp16<6>(v82_acc, v96_bc, v67_data);
          tensorforge::fmacdpp16<7>(v82_acc, v96_bc, v68_data);
          tensorforge::fmacdpp16<8>(v82_acc, v96_bc, v69_data);
          tensorforge::fmacdpp16<9>(v82_acc, v96_bc, v70_data);
          tensorforge::fmacdpp16<10>(v82_acc, v96_bc, v71_data);
          tensorforge::fmacdpp16<11>(v82_acc, v96_bc, v72_data);
          tensorforge::fmacdpp16<12>(v82_acc, v96_bc, v73_data);
          tensorforge::fmacdpp16<13>(v82_acc, v96_bc, v74_data);
          tensorforge::fmacdpp16<14>(v82_acc, v96_bc, v75_data);
          tensorforge::fmacdpp16<15>(v82_acc, v96_bc, v76_data);
          float v97_lin = r4[2];
          float v98_bc = tensorforge::broadcast<32, 16, 0>(v97_lin);
          tensorforge::fmacdpp16<0>(v82_acc, v98_bc, v77_data);
          tensorforge::fmacdpp16<1>(v83_acc, v98_bc, v65_data);
          tensorforge::fmacdpp16<2>(v83_acc, v98_bc, v66_data);
          tensorforge::fmacdpp16<3>(v83_acc, v98_bc, v67_data);
          tensorforge::fmacdpp16<4>(v83_acc, v98_bc, v68_data);
          tensorforge::fmacdpp16<5>(v83_acc, v98_bc, v69_data);
          tensorforge::fmacdpp16<6>(v83_acc, v98_bc, v70_data);
          tensorforge::fmacdpp16<7>(v83_acc, v98_bc, v71_data);
          tensorforge::fmacdpp16<8>(v83_acc, v98_bc, v72_data);
          tensorforge::fmacdpp16<9>(v83_acc, v98_bc, v73_data);
          tensorforge::fmacdpp16<10>(v83_acc, v98_bc, v74_data);
          tensorforge::fmacdpp16<11>(v83_acc, v98_bc, v75_data);
          tensorforge::fmacdpp16<12>(v83_acc, v98_bc, v76_data);
          tensorforge::fmacdpp16<13>(v83_acc, v98_bc, v77_data);
          tensorforge::fmacdpp16<14>(v84_acc, v98_bc, v65_data);
          tensorforge::fmacdpp16<15>(v84_acc, v98_bc, v66_data);
          float v99_bc = tensorforge::broadcast<32, 16, 1>(v97_lin);
          tensorforge::fmacdpp16<0>(v84_acc, v99_bc, v67_data);
          tensorforge::fmacdpp16<1>(v84_acc, v99_bc, v68_data);
          tensorforge::fmacdpp16<2>(v84_acc, v99_bc, v69_data);
          tensorforge::fmacdpp16<3>(v84_acc, v99_bc, v70_data);
          tensorforge::fmacdpp16<4>(v84_acc, v99_bc, v71_data);
          tensorforge::fmacdpp16<5>(v84_acc, v99_bc, v72_data);
          tensorforge::fmacdpp16<6>(v84_acc, v99_bc, v73_data);
          tensorforge::fmacdpp16<7>(v84_acc, v99_bc, v74_data);
          tensorforge::fmacdpp16<8>(v84_acc, v99_bc, v75_data);
          tensorforge::fmacdpp16<9>(v84_acc, v99_bc, v76_data);
          tensorforge::fmacdpp16<10>(v84_acc, v99_bc, v77_data);
          tensorforge::fmacdpp16<11>(v85_acc, v99_bc, v65_data);
          tensorforge::fmacdpp16<12>(v85_acc, v99_bc, v66_data);
          tensorforge::fmacdpp16<13>(v85_acc, v99_bc, v67_data);
          tensorforge::fmacdpp16<14>(v85_acc, v99_bc, v68_data);
          tensorforge::fmacdpp16<15>(v85_acc, v99_bc, v69_data);
          float v100_lin = r4[3];
          float v101_bc = tensorforge::broadcast<32, 16, 0>(v100_lin);
          tensorforge::fmacdpp16<0>(v85_acc, v101_bc, v70_data);
          tensorforge::fmacdpp16<1>(v85_acc, v101_bc, v71_data);
          tensorforge::fmacdpp16<2>(v85_acc, v101_bc, v72_data);
          tensorforge::fmacdpp16<3>(v85_acc, v101_bc, v73_data);
          tensorforge::fmacdpp16<4>(v85_acc, v101_bc, v74_data);
          tensorforge::fmacdpp16<5>(v85_acc, v101_bc, v75_data);
          tensorforge::fmacdpp16<6>(v85_acc, v101_bc, v76_data);
          tensorforge::fmacdpp16<7>(v85_acc, v101_bc, v77_data);
          tensorforge::fmacdpp16<8>(v86_acc, v101_bc, v65_data);
          tensorforge::fmacdpp16<9>(v86_acc, v101_bc, v66_data);
          tensorforge::fmacdpp16<10>(v86_acc, v101_bc, v67_data);
          tensorforge::fmacdpp16<11>(v86_acc, v101_bc, v68_data);
          tensorforge::fmacdpp16<12>(v86_acc, v101_bc, v69_data);
          tensorforge::fmacdpp16<13>(v86_acc, v101_bc, v70_data);
          tensorforge::fmacdpp16<14>(v86_acc, v101_bc, v71_data);
          tensorforge::fmacdpp16<15>(v86_acc, v101_bc, v72_data);
          float v102_bc = tensorforge::broadcast<32, 16, 1>(v100_lin);
          tensorforge::fmacdpp16<0>(v86_acc, v102_bc, v73_data);
          tensorforge::fmacdpp16<1>(v86_acc, v102_bc, v74_data);
          tensorforge::fmacdpp16<2>(v86_acc, v102_bc, v75_data);
          tensorforge::fmacdpp16<3>(v86_acc, v102_bc, v76_data);
          tensorforge::fmacdpp16<4>(v86_acc, v102_bc, v77_data);
          tensorforge::fmacdpp16<5>(v87_acc, v102_bc, v65_data);
          tensorforge::fmacdpp16<6>(v87_acc, v102_bc, v66_data);
          tensorforge::fmacdpp16<7>(v87_acc, v102_bc, v67_data);
          tensorforge::fmacdpp16<8>(v87_acc, v102_bc, v68_data);
          tensorforge::fmacdpp16<9>(v87_acc, v102_bc, v69_data);
          tensorforge::fmacdpp16<10>(v87_acc, v102_bc, v70_data);
          tensorforge::fmacdpp16<11>(v87_acc, v102_bc, v71_data);
          tensorforge::fmacdpp16<12>(v87_acc, v102_bc, v72_data);
          tensorforge::fmacdpp16<13>(v87_acc, v102_bc, v73_data);
          tensorforge::fmacdpp16<14>(v87_acc, v102_bc, v74_data);
          tensorforge::fmacdpp16<15>(v87_acc, v102_bc, v75_data);
          float v103_lin = r4[4];
          float v104_bc = tensorforge::broadcast<32, 16, 0>(v103_lin);
          tensorforge::fmacdpp16<0>(v87_acc, v104_bc, v76_data);
          tensorforge::fmacdpp16<1>(v87_acc, v104_bc, v77_data);
          tensorforge::fmacdpp16<2>(v88_acc, v104_bc, v65_data);
          tensorforge::fmacdpp16<3>(v88_acc, v104_bc, v66_data);
          tensorforge::fmacdpp16<4>(v88_acc, v104_bc, v67_data);
          tensorforge::fmacdpp16<5>(v88_acc, v104_bc, v68_data);
          tensorforge::fmacdpp16<6>(v88_acc, v104_bc, v69_data);
          tensorforge::fmacdpp16<7>(v88_acc, v104_bc, v70_data);
          tensorforge::fmacdpp16<8>(v88_acc, v104_bc, v71_data);
          tensorforge::fmacdpp16<9>(v88_acc, v104_bc, v72_data);
          tensorforge::fmacdpp16<10>(v88_acc, v104_bc, v73_data);
          tensorforge::fmacdpp16<11>(v88_acc, v104_bc, v74_data);
          tensorforge::fmacdpp16<12>(v88_acc, v104_bc, v75_data);
          tensorforge::fmacdpp16<13>(v88_acc, v104_bc, v76_data);
          tensorforge::fmacdpp16<14>(v88_acc, v104_bc, v77_data);
          tensorforge::fmacdpp16<15>(v89_acc, v104_bc, v65_data);
          float v105_bc = tensorforge::broadcast<32, 16, 1>(v103_lin);
          tensorforge::fmacdpp16<0>(v89_acc, v105_bc, v66_data);
          tensorforge::fmacdpp16<1>(v89_acc, v105_bc, v67_data);
          tensorforge::fmacdpp16<2>(v89_acc, v105_bc, v68_data);
          tensorforge::fmacdpp16<3>(v89_acc, v105_bc, v69_data);
          tensorforge::fmacdpp16<4>(v89_acc, v105_bc, v70_data);
          tensorforge::fmacdpp16<5>(v89_acc, v105_bc, v71_data);
          tensorforge::fmacdpp16<6>(v89_acc, v105_bc, v72_data);
          tensorforge::fmacdpp16<7>(v89_acc, v105_bc, v73_data);
          tensorforge::fmacdpp16<8>(v89_acc, v105_bc, v74_data);
          tensorforge::fmacdpp16<9>(v89_acc, v105_bc, v75_data);
          tensorforge::fmacdpp16<10>(v89_acc, v105_bc, v76_data);
          tensorforge::fmacdpp16<11>(v89_acc, v105_bc, v77_data);
          tensorforge::fmacdpp16<12>(v90_acc, v105_bc, v65_data);
          tensorforge::fmacdpp16<13>(v90_acc, v105_bc, v66_data);
          tensorforge::fmacdpp16<14>(v90_acc, v105_bc, v67_data);
          tensorforge::fmacdpp16<15>(v90_acc, v105_bc, v68_data);
          float v106_lin = r4[5];
          float v107_bc = tensorforge::broadcast<32, 16, 0>(v106_lin);
          tensorforge::fmacdpp16<0>(v90_acc, v107_bc, v69_data);
          tensorforge::fmacdpp16<1>(v90_acc, v107_bc, v70_data);
          tensorforge::fmacdpp16<2>(v90_acc, v107_bc, v71_data);
          tensorforge::fmacdpp16<3>(v90_acc, v107_bc, v72_data);
          tensorforge::fmacdpp16<4>(v90_acc, v107_bc, v73_data);
          tensorforge::fmacdpp16<5>(v90_acc, v107_bc, v74_data);
          tensorforge::fmacdpp16<6>(v90_acc, v107_bc, v75_data);
          tensorforge::fmacdpp16<7>(v90_acc, v107_bc, v76_data);
          tensorforge::fmacdpp16<8>(v90_acc, v107_bc, v77_data);
          ir5[0] = v78_acc;
          ir5[1] = v79_acc;
          ir5[2] = v80_acc;
          ir5[3] = v81_acc;
          ir5[4] = v82_acc;
          ir5[5] = v83_acc;
          ir5[6] = v84_acc;
          ir5[7] = v85_acc;
          ir5[8] = v86_acc;
          ir5[9] = v87_acc;
          ir5[10] = v88_acc;
          ir5[11] = v89_acc;
          ir5[12] = v90_acc;
          // glb_m3 = store{r>g}(r5);
          int32_t v110_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v111_i0 = 0; v111_i0 < 1; ++v111_i0) {
            int32_t v120_lead = v110_lead + (v111_i0 * 32);
            #pragma unroll
            for (int32_t v112_i1 = 0; v112_i1 < 13; ++v112_i1) {
              int32_t v113_a = v111_i0 + v112_i1;
              float v115_data = r5[(v111_i0 + v112_i1)];
              int32_t v122_a = v120_lead + (v112_i1 * 32);
              glb_m3[v122_a] = v115_data;
            }
          }
          ;
        }
      }
    }
  }
}

