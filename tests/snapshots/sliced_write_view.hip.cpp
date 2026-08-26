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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 10; v5_i1 < 13; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + v11_a)]);
              int32_t v22_a = v4_i0 + (v5_i1 - 10);
              r0[v22_a] = v20_data;
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
          float v25_data = r0[0];
          float v26_data = r0[1];
          float v27_data = r0[2];
          float v28_acc{};
          float v29_lin = r1[0];
          float v30_bc = tensorforge::broadcast<32, 16, 0>(v29_lin);
          tensorforge::fmacdpp16<0>(v28_acc, v30_bc, v25_data);
          tensorforge::fmacdpp16<1>(v28_acc, v30_bc, v26_data);
          tensorforge::fmacdpp16<2>(v28_acc, v30_bc, v27_data);
          ir2[0] = v28_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
            int32_t v43_lead = v3_lead + (v34_i0 * 32);
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 1; ++v35_i1) {
              int32_t v36_a = v34_i0 + v35_i1;
              float v38_data = r2[(v34_i0 + v35_i1)];
              int32_t v46_a = v43_lead + ((v35_i1 + 8) * 32);
              glb_m0[v46_a] = v38_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v51_i0 = 0; v51_i0 < 1; ++v51_i0) {
            int32_t v56_lead = v51_i0 * 32;
            int32_t v57_lead = v3_lead + v56_lead;
            int32_t v64_lead = v3_lead + v56_lead;
            #pragma unroll
            for (int32_t v52_i1 = 0; v52_i1 < 13; ++v52_i1) {
              int32_t v58_a = v52_i1 * 32;
              int32_t v59_a = v57_lead + v58_a;
              float v67_data = glb_m0[(v64_lead + v58_a)];
              int32_t v68_a = v51_i0 + v52_i1;
              r3[v68_a] = v67_data;
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
          float v71_data = r3[0];
          float v72_data = r3[1];
          float v73_data = r3[2];
          float v74_data = r3[3];
          float v75_data = r3[4];
          float v76_data = r3[5];
          float v77_data = r3[6];
          float v78_data = r3[7];
          float v79_data = r3[8];
          float v80_data = r3[9];
          float v81_data = r3[10];
          float v82_data = r3[11];
          float v83_data = r3[12];
          float v84_acc{};
          float v85_acc{};
          float v86_acc{};
          float v87_acc{};
          float v88_acc{};
          float v89_acc{};
          float v90_acc{};
          float v91_acc{};
          float v92_acc{};
          float v93_acc{};
          float v94_acc{};
          float v95_acc{};
          float v96_acc{};
          float v97_lin = r4[0];
          float v98_bc = tensorforge::broadcast<32, 16, 0>(v97_lin);
          tensorforge::fmacdpp16<0>(v84_acc, v98_bc, v71_data);
          tensorforge::fmacdpp16<1>(v84_acc, v98_bc, v72_data);
          tensorforge::fmacdpp16<2>(v84_acc, v98_bc, v73_data);
          tensorforge::fmacdpp16<3>(v84_acc, v98_bc, v74_data);
          tensorforge::fmacdpp16<4>(v84_acc, v98_bc, v75_data);
          tensorforge::fmacdpp16<5>(v84_acc, v98_bc, v76_data);
          tensorforge::fmacdpp16<6>(v84_acc, v98_bc, v77_data);
          tensorforge::fmacdpp16<7>(v84_acc, v98_bc, v78_data);
          tensorforge::fmacdpp16<8>(v84_acc, v98_bc, v79_data);
          tensorforge::fmacdpp16<9>(v84_acc, v98_bc, v80_data);
          tensorforge::fmacdpp16<10>(v84_acc, v98_bc, v81_data);
          tensorforge::fmacdpp16<11>(v84_acc, v98_bc, v82_data);
          tensorforge::fmacdpp16<12>(v84_acc, v98_bc, v83_data);
          tensorforge::fmacdpp16<13>(v85_acc, v98_bc, v71_data);
          tensorforge::fmacdpp16<14>(v85_acc, v98_bc, v72_data);
          tensorforge::fmacdpp16<15>(v85_acc, v98_bc, v73_data);
          float v99_bc = tensorforge::broadcast<32, 16, 1>(v97_lin);
          tensorforge::fmacdpp16<0>(v85_acc, v99_bc, v74_data);
          tensorforge::fmacdpp16<1>(v85_acc, v99_bc, v75_data);
          tensorforge::fmacdpp16<2>(v85_acc, v99_bc, v76_data);
          tensorforge::fmacdpp16<3>(v85_acc, v99_bc, v77_data);
          tensorforge::fmacdpp16<4>(v85_acc, v99_bc, v78_data);
          tensorforge::fmacdpp16<5>(v85_acc, v99_bc, v79_data);
          tensorforge::fmacdpp16<6>(v85_acc, v99_bc, v80_data);
          tensorforge::fmacdpp16<7>(v85_acc, v99_bc, v81_data);
          tensorforge::fmacdpp16<8>(v85_acc, v99_bc, v82_data);
          tensorforge::fmacdpp16<9>(v85_acc, v99_bc, v83_data);
          tensorforge::fmacdpp16<10>(v86_acc, v99_bc, v71_data);
          tensorforge::fmacdpp16<11>(v86_acc, v99_bc, v72_data);
          tensorforge::fmacdpp16<12>(v86_acc, v99_bc, v73_data);
          tensorforge::fmacdpp16<13>(v86_acc, v99_bc, v74_data);
          tensorforge::fmacdpp16<14>(v86_acc, v99_bc, v75_data);
          tensorforge::fmacdpp16<15>(v86_acc, v99_bc, v76_data);
          float v100_lin = r4[1];
          float v101_bc = tensorforge::broadcast<32, 16, 0>(v100_lin);
          tensorforge::fmacdpp16<0>(v86_acc, v101_bc, v77_data);
          tensorforge::fmacdpp16<1>(v86_acc, v101_bc, v78_data);
          tensorforge::fmacdpp16<2>(v86_acc, v101_bc, v79_data);
          tensorforge::fmacdpp16<3>(v86_acc, v101_bc, v80_data);
          tensorforge::fmacdpp16<4>(v86_acc, v101_bc, v81_data);
          tensorforge::fmacdpp16<5>(v86_acc, v101_bc, v82_data);
          tensorforge::fmacdpp16<6>(v86_acc, v101_bc, v83_data);
          tensorforge::fmacdpp16<7>(v87_acc, v101_bc, v71_data);
          tensorforge::fmacdpp16<8>(v87_acc, v101_bc, v72_data);
          tensorforge::fmacdpp16<9>(v87_acc, v101_bc, v73_data);
          tensorforge::fmacdpp16<10>(v87_acc, v101_bc, v74_data);
          tensorforge::fmacdpp16<11>(v87_acc, v101_bc, v75_data);
          tensorforge::fmacdpp16<12>(v87_acc, v101_bc, v76_data);
          tensorforge::fmacdpp16<13>(v87_acc, v101_bc, v77_data);
          tensorforge::fmacdpp16<14>(v87_acc, v101_bc, v78_data);
          tensorforge::fmacdpp16<15>(v87_acc, v101_bc, v79_data);
          float v102_bc = tensorforge::broadcast<32, 16, 1>(v100_lin);
          tensorforge::fmacdpp16<0>(v87_acc, v102_bc, v80_data);
          tensorforge::fmacdpp16<1>(v87_acc, v102_bc, v81_data);
          tensorforge::fmacdpp16<2>(v87_acc, v102_bc, v82_data);
          tensorforge::fmacdpp16<3>(v87_acc, v102_bc, v83_data);
          tensorforge::fmacdpp16<4>(v88_acc, v102_bc, v71_data);
          tensorforge::fmacdpp16<5>(v88_acc, v102_bc, v72_data);
          tensorforge::fmacdpp16<6>(v88_acc, v102_bc, v73_data);
          tensorforge::fmacdpp16<7>(v88_acc, v102_bc, v74_data);
          tensorforge::fmacdpp16<8>(v88_acc, v102_bc, v75_data);
          tensorforge::fmacdpp16<9>(v88_acc, v102_bc, v76_data);
          tensorforge::fmacdpp16<10>(v88_acc, v102_bc, v77_data);
          tensorforge::fmacdpp16<11>(v88_acc, v102_bc, v78_data);
          tensorforge::fmacdpp16<12>(v88_acc, v102_bc, v79_data);
          tensorforge::fmacdpp16<13>(v88_acc, v102_bc, v80_data);
          tensorforge::fmacdpp16<14>(v88_acc, v102_bc, v81_data);
          tensorforge::fmacdpp16<15>(v88_acc, v102_bc, v82_data);
          float v103_lin = r4[2];
          float v104_bc = tensorforge::broadcast<32, 16, 0>(v103_lin);
          tensorforge::fmacdpp16<0>(v88_acc, v104_bc, v83_data);
          tensorforge::fmacdpp16<1>(v89_acc, v104_bc, v71_data);
          tensorforge::fmacdpp16<2>(v89_acc, v104_bc, v72_data);
          tensorforge::fmacdpp16<3>(v89_acc, v104_bc, v73_data);
          tensorforge::fmacdpp16<4>(v89_acc, v104_bc, v74_data);
          tensorforge::fmacdpp16<5>(v89_acc, v104_bc, v75_data);
          tensorforge::fmacdpp16<6>(v89_acc, v104_bc, v76_data);
          tensorforge::fmacdpp16<7>(v89_acc, v104_bc, v77_data);
          tensorforge::fmacdpp16<8>(v89_acc, v104_bc, v78_data);
          tensorforge::fmacdpp16<9>(v89_acc, v104_bc, v79_data);
          tensorforge::fmacdpp16<10>(v89_acc, v104_bc, v80_data);
          tensorforge::fmacdpp16<11>(v89_acc, v104_bc, v81_data);
          tensorforge::fmacdpp16<12>(v89_acc, v104_bc, v82_data);
          tensorforge::fmacdpp16<13>(v89_acc, v104_bc, v83_data);
          tensorforge::fmacdpp16<14>(v90_acc, v104_bc, v71_data);
          tensorforge::fmacdpp16<15>(v90_acc, v104_bc, v72_data);
          float v105_bc = tensorforge::broadcast<32, 16, 1>(v103_lin);
          tensorforge::fmacdpp16<0>(v90_acc, v105_bc, v73_data);
          tensorforge::fmacdpp16<1>(v90_acc, v105_bc, v74_data);
          tensorforge::fmacdpp16<2>(v90_acc, v105_bc, v75_data);
          tensorforge::fmacdpp16<3>(v90_acc, v105_bc, v76_data);
          tensorforge::fmacdpp16<4>(v90_acc, v105_bc, v77_data);
          tensorforge::fmacdpp16<5>(v90_acc, v105_bc, v78_data);
          tensorforge::fmacdpp16<6>(v90_acc, v105_bc, v79_data);
          tensorforge::fmacdpp16<7>(v90_acc, v105_bc, v80_data);
          tensorforge::fmacdpp16<8>(v90_acc, v105_bc, v81_data);
          tensorforge::fmacdpp16<9>(v90_acc, v105_bc, v82_data);
          tensorforge::fmacdpp16<10>(v90_acc, v105_bc, v83_data);
          tensorforge::fmacdpp16<11>(v91_acc, v105_bc, v71_data);
          tensorforge::fmacdpp16<12>(v91_acc, v105_bc, v72_data);
          tensorforge::fmacdpp16<13>(v91_acc, v105_bc, v73_data);
          tensorforge::fmacdpp16<14>(v91_acc, v105_bc, v74_data);
          tensorforge::fmacdpp16<15>(v91_acc, v105_bc, v75_data);
          float v106_lin = r4[3];
          float v107_bc = tensorforge::broadcast<32, 16, 0>(v106_lin);
          tensorforge::fmacdpp16<0>(v91_acc, v107_bc, v76_data);
          tensorforge::fmacdpp16<1>(v91_acc, v107_bc, v77_data);
          tensorforge::fmacdpp16<2>(v91_acc, v107_bc, v78_data);
          tensorforge::fmacdpp16<3>(v91_acc, v107_bc, v79_data);
          tensorforge::fmacdpp16<4>(v91_acc, v107_bc, v80_data);
          tensorforge::fmacdpp16<5>(v91_acc, v107_bc, v81_data);
          tensorforge::fmacdpp16<6>(v91_acc, v107_bc, v82_data);
          tensorforge::fmacdpp16<7>(v91_acc, v107_bc, v83_data);
          tensorforge::fmacdpp16<8>(v92_acc, v107_bc, v71_data);
          tensorforge::fmacdpp16<9>(v92_acc, v107_bc, v72_data);
          tensorforge::fmacdpp16<10>(v92_acc, v107_bc, v73_data);
          tensorforge::fmacdpp16<11>(v92_acc, v107_bc, v74_data);
          tensorforge::fmacdpp16<12>(v92_acc, v107_bc, v75_data);
          tensorforge::fmacdpp16<13>(v92_acc, v107_bc, v76_data);
          tensorforge::fmacdpp16<14>(v92_acc, v107_bc, v77_data);
          tensorforge::fmacdpp16<15>(v92_acc, v107_bc, v78_data);
          float v108_bc = tensorforge::broadcast<32, 16, 1>(v106_lin);
          tensorforge::fmacdpp16<0>(v92_acc, v108_bc, v79_data);
          tensorforge::fmacdpp16<1>(v92_acc, v108_bc, v80_data);
          tensorforge::fmacdpp16<2>(v92_acc, v108_bc, v81_data);
          tensorforge::fmacdpp16<3>(v92_acc, v108_bc, v82_data);
          tensorforge::fmacdpp16<4>(v92_acc, v108_bc, v83_data);
          tensorforge::fmacdpp16<5>(v93_acc, v108_bc, v71_data);
          tensorforge::fmacdpp16<6>(v93_acc, v108_bc, v72_data);
          tensorforge::fmacdpp16<7>(v93_acc, v108_bc, v73_data);
          tensorforge::fmacdpp16<8>(v93_acc, v108_bc, v74_data);
          tensorforge::fmacdpp16<9>(v93_acc, v108_bc, v75_data);
          tensorforge::fmacdpp16<10>(v93_acc, v108_bc, v76_data);
          tensorforge::fmacdpp16<11>(v93_acc, v108_bc, v77_data);
          tensorforge::fmacdpp16<12>(v93_acc, v108_bc, v78_data);
          tensorforge::fmacdpp16<13>(v93_acc, v108_bc, v79_data);
          tensorforge::fmacdpp16<14>(v93_acc, v108_bc, v80_data);
          tensorforge::fmacdpp16<15>(v93_acc, v108_bc, v81_data);
          float v109_lin = r4[4];
          float v110_bc = tensorforge::broadcast<32, 16, 0>(v109_lin);
          tensorforge::fmacdpp16<0>(v93_acc, v110_bc, v82_data);
          tensorforge::fmacdpp16<1>(v93_acc, v110_bc, v83_data);
          tensorforge::fmacdpp16<2>(v94_acc, v110_bc, v71_data);
          tensorforge::fmacdpp16<3>(v94_acc, v110_bc, v72_data);
          tensorforge::fmacdpp16<4>(v94_acc, v110_bc, v73_data);
          tensorforge::fmacdpp16<5>(v94_acc, v110_bc, v74_data);
          tensorforge::fmacdpp16<6>(v94_acc, v110_bc, v75_data);
          tensorforge::fmacdpp16<7>(v94_acc, v110_bc, v76_data);
          tensorforge::fmacdpp16<8>(v94_acc, v110_bc, v77_data);
          tensorforge::fmacdpp16<9>(v94_acc, v110_bc, v78_data);
          tensorforge::fmacdpp16<10>(v94_acc, v110_bc, v79_data);
          tensorforge::fmacdpp16<11>(v94_acc, v110_bc, v80_data);
          tensorforge::fmacdpp16<12>(v94_acc, v110_bc, v81_data);
          tensorforge::fmacdpp16<13>(v94_acc, v110_bc, v82_data);
          tensorforge::fmacdpp16<14>(v94_acc, v110_bc, v83_data);
          tensorforge::fmacdpp16<15>(v95_acc, v110_bc, v71_data);
          float v111_bc = tensorforge::broadcast<32, 16, 1>(v109_lin);
          tensorforge::fmacdpp16<0>(v95_acc, v111_bc, v72_data);
          tensorforge::fmacdpp16<1>(v95_acc, v111_bc, v73_data);
          tensorforge::fmacdpp16<2>(v95_acc, v111_bc, v74_data);
          tensorforge::fmacdpp16<3>(v95_acc, v111_bc, v75_data);
          tensorforge::fmacdpp16<4>(v95_acc, v111_bc, v76_data);
          tensorforge::fmacdpp16<5>(v95_acc, v111_bc, v77_data);
          tensorforge::fmacdpp16<6>(v95_acc, v111_bc, v78_data);
          tensorforge::fmacdpp16<7>(v95_acc, v111_bc, v79_data);
          tensorforge::fmacdpp16<8>(v95_acc, v111_bc, v80_data);
          tensorforge::fmacdpp16<9>(v95_acc, v111_bc, v81_data);
          tensorforge::fmacdpp16<10>(v95_acc, v111_bc, v82_data);
          tensorforge::fmacdpp16<11>(v95_acc, v111_bc, v83_data);
          tensorforge::fmacdpp16<12>(v96_acc, v111_bc, v71_data);
          tensorforge::fmacdpp16<13>(v96_acc, v111_bc, v72_data);
          tensorforge::fmacdpp16<14>(v96_acc, v111_bc, v73_data);
          tensorforge::fmacdpp16<15>(v96_acc, v111_bc, v74_data);
          float v112_lin = r4[5];
          float v113_bc = tensorforge::broadcast<32, 16, 0>(v112_lin);
          tensorforge::fmacdpp16<0>(v96_acc, v113_bc, v75_data);
          tensorforge::fmacdpp16<1>(v96_acc, v113_bc, v76_data);
          tensorforge::fmacdpp16<2>(v96_acc, v113_bc, v77_data);
          tensorforge::fmacdpp16<3>(v96_acc, v113_bc, v78_data);
          tensorforge::fmacdpp16<4>(v96_acc, v113_bc, v79_data);
          tensorforge::fmacdpp16<5>(v96_acc, v113_bc, v80_data);
          tensorforge::fmacdpp16<6>(v96_acc, v113_bc, v81_data);
          tensorforge::fmacdpp16<7>(v96_acc, v113_bc, v82_data);
          tensorforge::fmacdpp16<8>(v96_acc, v113_bc, v83_data);
          ir5[0] = v84_acc;
          ir5[1] = v85_acc;
          ir5[2] = v86_acc;
          ir5[3] = v87_acc;
          ir5[4] = v88_acc;
          ir5[5] = v89_acc;
          ir5[6] = v90_acc;
          ir5[7] = v91_acc;
          ir5[8] = v92_acc;
          ir5[9] = v93_acc;
          ir5[10] = v94_acc;
          ir5[11] = v95_acc;
          ir5[12] = v96_acc;
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v117_i0 = 0; v117_i0 < 1; ++v117_i0) {
            int32_t v126_lead = v3_lead + (v117_i0 * 32);
            #pragma unroll
            for (int32_t v118_i1 = 0; v118_i1 < 13; ++v118_i1) {
              int32_t v119_a = v117_i0 + v118_i1;
              float v121_data = r5[(v117_i0 + v118_i1)];
              int32_t v128_a = v126_lead + (v118_i1 * 32);
              glb_m3[v128_a] = v121_data;
            }
          }
          ;
        }
      }
    }
  }
}

