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
          // r1 = load{g>r}(glb_m2);
          float v24_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v24_lin;
          float v25_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v25_lin;
          float v26_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v26_lin;
          float v27_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v27_lin;
          float v28_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v28_lin;
          float v29_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v29_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          auto& ir2 = r2;
          float v31_data = r0[0];
          float v32_data = r0[1];
          float v33_data = r0[2];
          float v34_acc{};
          float v35_lin = r1[0];
          float v36_bc = tensorforge::broadcast<32, 16, 0>(v35_lin);
          tensorforge::fmacdpp16<0>(v34_acc, v36_bc, v31_data);
          tensorforge::fmacdpp16<1>(v34_acc, v36_bc, v32_data);
          tensorforge::fmacdpp16<2>(v34_acc, v36_bc, v33_data);
          ir2[0] = v34_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v40_i0 = 0; v40_i0 < 1; ++v40_i0) {
            int32_t v49_lead = v3_lead + (v40_i0 * 32);
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 1; ++v41_i1) {
              int32_t v42_a = v40_i0 + v41_i1;
              float v44_data = r2[(v40_i0 + v41_i1)];
              int32_t v52_a = v49_lead + ((v41_i1 + 8) * 32);
              glb_m0[v52_a] = v44_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v57_i0 = 0; v57_i0 < 1; ++v57_i0) {
            int32_t v62_lead = v57_i0 * 32;
            int32_t v63_lead = v3_lead + v62_lead;
            int32_t v70_lead = v3_lead + v62_lead;
            #pragma unroll
            for (int32_t v58_i1 = 0; v58_i1 < 13; ++v58_i1) {
              int32_t v64_a = v58_i1 * 32;
              int32_t v65_a = v63_lead + v64_a;
              float v73_data = glb_m0[(v70_lead + v64_a)];
              int32_t v74_a = v57_i0 + v58_i1;
              r3[v74_a] = v73_data;
            }
          }
          float r4[13]{};
          // r4 = load{g>r}(glb_m4);
          float v76_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v76_lin;
          float v77_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v77_lin;
          float v78_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v78_lin;
          float v79_lin = glb_m4[96 + threadIdx.x * 1];
          r4[3] = v79_lin;
          float v80_lin = glb_m4[128 + threadIdx.x * 1];
          r4[4] = v80_lin;
          float v81_lin = glb_m4[160 + threadIdx.x * 1];
          r4[5] = v81_lin;
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          auto& ir5 = r5;
          float v83_data = r3[0];
          float v84_data = r3[1];
          float v85_data = r3[2];
          float v86_data = r3[3];
          float v87_data = r3[4];
          float v88_data = r3[5];
          float v89_data = r3[6];
          float v90_data = r3[7];
          float v91_data = r3[8];
          float v92_data = r3[9];
          float v93_data = r3[10];
          float v94_data = r3[11];
          float v95_data = r3[12];
          float v96_acc{};
          float v97_acc{};
          float v98_acc{};
          float v99_acc{};
          float v100_acc{};
          float v101_acc{};
          float v102_acc{};
          float v103_acc{};
          float v104_acc{};
          float v105_acc{};
          float v106_acc{};
          float v107_acc{};
          float v108_acc{};
          float v109_lin = r4[0];
          float v110_bc = tensorforge::broadcast<32, 16, 0>(v109_lin);
          tensorforge::fmacdpp16<0>(v96_acc, v110_bc, v83_data);
          tensorforge::fmacdpp16<1>(v96_acc, v110_bc, v84_data);
          tensorforge::fmacdpp16<2>(v96_acc, v110_bc, v85_data);
          tensorforge::fmacdpp16<3>(v96_acc, v110_bc, v86_data);
          tensorforge::fmacdpp16<4>(v96_acc, v110_bc, v87_data);
          tensorforge::fmacdpp16<5>(v96_acc, v110_bc, v88_data);
          tensorforge::fmacdpp16<6>(v96_acc, v110_bc, v89_data);
          tensorforge::fmacdpp16<7>(v96_acc, v110_bc, v90_data);
          tensorforge::fmacdpp16<8>(v96_acc, v110_bc, v91_data);
          tensorforge::fmacdpp16<9>(v96_acc, v110_bc, v92_data);
          tensorforge::fmacdpp16<10>(v96_acc, v110_bc, v93_data);
          tensorforge::fmacdpp16<11>(v96_acc, v110_bc, v94_data);
          tensorforge::fmacdpp16<12>(v96_acc, v110_bc, v95_data);
          tensorforge::fmacdpp16<13>(v97_acc, v110_bc, v83_data);
          tensorforge::fmacdpp16<14>(v97_acc, v110_bc, v84_data);
          tensorforge::fmacdpp16<15>(v97_acc, v110_bc, v85_data);
          float v111_bc = tensorforge::broadcast<32, 16, 1>(v109_lin);
          tensorforge::fmacdpp16<0>(v97_acc, v111_bc, v86_data);
          tensorforge::fmacdpp16<1>(v97_acc, v111_bc, v87_data);
          tensorforge::fmacdpp16<2>(v97_acc, v111_bc, v88_data);
          tensorforge::fmacdpp16<3>(v97_acc, v111_bc, v89_data);
          tensorforge::fmacdpp16<4>(v97_acc, v111_bc, v90_data);
          tensorforge::fmacdpp16<5>(v97_acc, v111_bc, v91_data);
          tensorforge::fmacdpp16<6>(v97_acc, v111_bc, v92_data);
          tensorforge::fmacdpp16<7>(v97_acc, v111_bc, v93_data);
          tensorforge::fmacdpp16<8>(v97_acc, v111_bc, v94_data);
          tensorforge::fmacdpp16<9>(v97_acc, v111_bc, v95_data);
          tensorforge::fmacdpp16<10>(v98_acc, v111_bc, v83_data);
          tensorforge::fmacdpp16<11>(v98_acc, v111_bc, v84_data);
          tensorforge::fmacdpp16<12>(v98_acc, v111_bc, v85_data);
          tensorforge::fmacdpp16<13>(v98_acc, v111_bc, v86_data);
          tensorforge::fmacdpp16<14>(v98_acc, v111_bc, v87_data);
          tensorforge::fmacdpp16<15>(v98_acc, v111_bc, v88_data);
          float v112_lin = r4[1];
          float v113_bc = tensorforge::broadcast<32, 16, 0>(v112_lin);
          tensorforge::fmacdpp16<0>(v98_acc, v113_bc, v89_data);
          tensorforge::fmacdpp16<1>(v98_acc, v113_bc, v90_data);
          tensorforge::fmacdpp16<2>(v98_acc, v113_bc, v91_data);
          tensorforge::fmacdpp16<3>(v98_acc, v113_bc, v92_data);
          tensorforge::fmacdpp16<4>(v98_acc, v113_bc, v93_data);
          tensorforge::fmacdpp16<5>(v98_acc, v113_bc, v94_data);
          tensorforge::fmacdpp16<6>(v98_acc, v113_bc, v95_data);
          tensorforge::fmacdpp16<7>(v99_acc, v113_bc, v83_data);
          tensorforge::fmacdpp16<8>(v99_acc, v113_bc, v84_data);
          tensorforge::fmacdpp16<9>(v99_acc, v113_bc, v85_data);
          tensorforge::fmacdpp16<10>(v99_acc, v113_bc, v86_data);
          tensorforge::fmacdpp16<11>(v99_acc, v113_bc, v87_data);
          tensorforge::fmacdpp16<12>(v99_acc, v113_bc, v88_data);
          tensorforge::fmacdpp16<13>(v99_acc, v113_bc, v89_data);
          tensorforge::fmacdpp16<14>(v99_acc, v113_bc, v90_data);
          tensorforge::fmacdpp16<15>(v99_acc, v113_bc, v91_data);
          float v114_bc = tensorforge::broadcast<32, 16, 1>(v112_lin);
          tensorforge::fmacdpp16<0>(v99_acc, v114_bc, v92_data);
          tensorforge::fmacdpp16<1>(v99_acc, v114_bc, v93_data);
          tensorforge::fmacdpp16<2>(v99_acc, v114_bc, v94_data);
          tensorforge::fmacdpp16<3>(v99_acc, v114_bc, v95_data);
          tensorforge::fmacdpp16<4>(v100_acc, v114_bc, v83_data);
          tensorforge::fmacdpp16<5>(v100_acc, v114_bc, v84_data);
          tensorforge::fmacdpp16<6>(v100_acc, v114_bc, v85_data);
          tensorforge::fmacdpp16<7>(v100_acc, v114_bc, v86_data);
          tensorforge::fmacdpp16<8>(v100_acc, v114_bc, v87_data);
          tensorforge::fmacdpp16<9>(v100_acc, v114_bc, v88_data);
          tensorforge::fmacdpp16<10>(v100_acc, v114_bc, v89_data);
          tensorforge::fmacdpp16<11>(v100_acc, v114_bc, v90_data);
          tensorforge::fmacdpp16<12>(v100_acc, v114_bc, v91_data);
          tensorforge::fmacdpp16<13>(v100_acc, v114_bc, v92_data);
          tensorforge::fmacdpp16<14>(v100_acc, v114_bc, v93_data);
          tensorforge::fmacdpp16<15>(v100_acc, v114_bc, v94_data);
          float v115_lin = r4[2];
          float v116_bc = tensorforge::broadcast<32, 16, 0>(v115_lin);
          tensorforge::fmacdpp16<0>(v100_acc, v116_bc, v95_data);
          tensorforge::fmacdpp16<1>(v101_acc, v116_bc, v83_data);
          tensorforge::fmacdpp16<2>(v101_acc, v116_bc, v84_data);
          tensorforge::fmacdpp16<3>(v101_acc, v116_bc, v85_data);
          tensorforge::fmacdpp16<4>(v101_acc, v116_bc, v86_data);
          tensorforge::fmacdpp16<5>(v101_acc, v116_bc, v87_data);
          tensorforge::fmacdpp16<6>(v101_acc, v116_bc, v88_data);
          tensorforge::fmacdpp16<7>(v101_acc, v116_bc, v89_data);
          tensorforge::fmacdpp16<8>(v101_acc, v116_bc, v90_data);
          tensorforge::fmacdpp16<9>(v101_acc, v116_bc, v91_data);
          tensorforge::fmacdpp16<10>(v101_acc, v116_bc, v92_data);
          tensorforge::fmacdpp16<11>(v101_acc, v116_bc, v93_data);
          tensorforge::fmacdpp16<12>(v101_acc, v116_bc, v94_data);
          tensorforge::fmacdpp16<13>(v101_acc, v116_bc, v95_data);
          tensorforge::fmacdpp16<14>(v102_acc, v116_bc, v83_data);
          tensorforge::fmacdpp16<15>(v102_acc, v116_bc, v84_data);
          float v117_bc = tensorforge::broadcast<32, 16, 1>(v115_lin);
          tensorforge::fmacdpp16<0>(v102_acc, v117_bc, v85_data);
          tensorforge::fmacdpp16<1>(v102_acc, v117_bc, v86_data);
          tensorforge::fmacdpp16<2>(v102_acc, v117_bc, v87_data);
          tensorforge::fmacdpp16<3>(v102_acc, v117_bc, v88_data);
          tensorforge::fmacdpp16<4>(v102_acc, v117_bc, v89_data);
          tensorforge::fmacdpp16<5>(v102_acc, v117_bc, v90_data);
          tensorforge::fmacdpp16<6>(v102_acc, v117_bc, v91_data);
          tensorforge::fmacdpp16<7>(v102_acc, v117_bc, v92_data);
          tensorforge::fmacdpp16<8>(v102_acc, v117_bc, v93_data);
          tensorforge::fmacdpp16<9>(v102_acc, v117_bc, v94_data);
          tensorforge::fmacdpp16<10>(v102_acc, v117_bc, v95_data);
          tensorforge::fmacdpp16<11>(v103_acc, v117_bc, v83_data);
          tensorforge::fmacdpp16<12>(v103_acc, v117_bc, v84_data);
          tensorforge::fmacdpp16<13>(v103_acc, v117_bc, v85_data);
          tensorforge::fmacdpp16<14>(v103_acc, v117_bc, v86_data);
          tensorforge::fmacdpp16<15>(v103_acc, v117_bc, v87_data);
          float v118_lin = r4[3];
          float v119_bc = tensorforge::broadcast<32, 16, 0>(v118_lin);
          tensorforge::fmacdpp16<0>(v103_acc, v119_bc, v88_data);
          tensorforge::fmacdpp16<1>(v103_acc, v119_bc, v89_data);
          tensorforge::fmacdpp16<2>(v103_acc, v119_bc, v90_data);
          tensorforge::fmacdpp16<3>(v103_acc, v119_bc, v91_data);
          tensorforge::fmacdpp16<4>(v103_acc, v119_bc, v92_data);
          tensorforge::fmacdpp16<5>(v103_acc, v119_bc, v93_data);
          tensorforge::fmacdpp16<6>(v103_acc, v119_bc, v94_data);
          tensorforge::fmacdpp16<7>(v103_acc, v119_bc, v95_data);
          tensorforge::fmacdpp16<8>(v104_acc, v119_bc, v83_data);
          tensorforge::fmacdpp16<9>(v104_acc, v119_bc, v84_data);
          tensorforge::fmacdpp16<10>(v104_acc, v119_bc, v85_data);
          tensorforge::fmacdpp16<11>(v104_acc, v119_bc, v86_data);
          tensorforge::fmacdpp16<12>(v104_acc, v119_bc, v87_data);
          tensorforge::fmacdpp16<13>(v104_acc, v119_bc, v88_data);
          tensorforge::fmacdpp16<14>(v104_acc, v119_bc, v89_data);
          tensorforge::fmacdpp16<15>(v104_acc, v119_bc, v90_data);
          float v120_bc = tensorforge::broadcast<32, 16, 1>(v118_lin);
          tensorforge::fmacdpp16<0>(v104_acc, v120_bc, v91_data);
          tensorforge::fmacdpp16<1>(v104_acc, v120_bc, v92_data);
          tensorforge::fmacdpp16<2>(v104_acc, v120_bc, v93_data);
          tensorforge::fmacdpp16<3>(v104_acc, v120_bc, v94_data);
          tensorforge::fmacdpp16<4>(v104_acc, v120_bc, v95_data);
          tensorforge::fmacdpp16<5>(v105_acc, v120_bc, v83_data);
          tensorforge::fmacdpp16<6>(v105_acc, v120_bc, v84_data);
          tensorforge::fmacdpp16<7>(v105_acc, v120_bc, v85_data);
          tensorforge::fmacdpp16<8>(v105_acc, v120_bc, v86_data);
          tensorforge::fmacdpp16<9>(v105_acc, v120_bc, v87_data);
          tensorforge::fmacdpp16<10>(v105_acc, v120_bc, v88_data);
          tensorforge::fmacdpp16<11>(v105_acc, v120_bc, v89_data);
          tensorforge::fmacdpp16<12>(v105_acc, v120_bc, v90_data);
          tensorforge::fmacdpp16<13>(v105_acc, v120_bc, v91_data);
          tensorforge::fmacdpp16<14>(v105_acc, v120_bc, v92_data);
          tensorforge::fmacdpp16<15>(v105_acc, v120_bc, v93_data);
          float v121_lin = r4[4];
          float v122_bc = tensorforge::broadcast<32, 16, 0>(v121_lin);
          tensorforge::fmacdpp16<0>(v105_acc, v122_bc, v94_data);
          tensorforge::fmacdpp16<1>(v105_acc, v122_bc, v95_data);
          tensorforge::fmacdpp16<2>(v106_acc, v122_bc, v83_data);
          tensorforge::fmacdpp16<3>(v106_acc, v122_bc, v84_data);
          tensorforge::fmacdpp16<4>(v106_acc, v122_bc, v85_data);
          tensorforge::fmacdpp16<5>(v106_acc, v122_bc, v86_data);
          tensorforge::fmacdpp16<6>(v106_acc, v122_bc, v87_data);
          tensorforge::fmacdpp16<7>(v106_acc, v122_bc, v88_data);
          tensorforge::fmacdpp16<8>(v106_acc, v122_bc, v89_data);
          tensorforge::fmacdpp16<9>(v106_acc, v122_bc, v90_data);
          tensorforge::fmacdpp16<10>(v106_acc, v122_bc, v91_data);
          tensorforge::fmacdpp16<11>(v106_acc, v122_bc, v92_data);
          tensorforge::fmacdpp16<12>(v106_acc, v122_bc, v93_data);
          tensorforge::fmacdpp16<13>(v106_acc, v122_bc, v94_data);
          tensorforge::fmacdpp16<14>(v106_acc, v122_bc, v95_data);
          tensorforge::fmacdpp16<15>(v107_acc, v122_bc, v83_data);
          float v123_bc = tensorforge::broadcast<32, 16, 1>(v121_lin);
          tensorforge::fmacdpp16<0>(v107_acc, v123_bc, v84_data);
          tensorforge::fmacdpp16<1>(v107_acc, v123_bc, v85_data);
          tensorforge::fmacdpp16<2>(v107_acc, v123_bc, v86_data);
          tensorforge::fmacdpp16<3>(v107_acc, v123_bc, v87_data);
          tensorforge::fmacdpp16<4>(v107_acc, v123_bc, v88_data);
          tensorforge::fmacdpp16<5>(v107_acc, v123_bc, v89_data);
          tensorforge::fmacdpp16<6>(v107_acc, v123_bc, v90_data);
          tensorforge::fmacdpp16<7>(v107_acc, v123_bc, v91_data);
          tensorforge::fmacdpp16<8>(v107_acc, v123_bc, v92_data);
          tensorforge::fmacdpp16<9>(v107_acc, v123_bc, v93_data);
          tensorforge::fmacdpp16<10>(v107_acc, v123_bc, v94_data);
          tensorforge::fmacdpp16<11>(v107_acc, v123_bc, v95_data);
          tensorforge::fmacdpp16<12>(v108_acc, v123_bc, v83_data);
          tensorforge::fmacdpp16<13>(v108_acc, v123_bc, v84_data);
          tensorforge::fmacdpp16<14>(v108_acc, v123_bc, v85_data);
          tensorforge::fmacdpp16<15>(v108_acc, v123_bc, v86_data);
          float v124_lin = r4[5];
          float v125_bc = tensorforge::broadcast<32, 16, 0>(v124_lin);
          tensorforge::fmacdpp16<0>(v108_acc, v125_bc, v87_data);
          tensorforge::fmacdpp16<1>(v108_acc, v125_bc, v88_data);
          tensorforge::fmacdpp16<2>(v108_acc, v125_bc, v89_data);
          tensorforge::fmacdpp16<3>(v108_acc, v125_bc, v90_data);
          tensorforge::fmacdpp16<4>(v108_acc, v125_bc, v91_data);
          tensorforge::fmacdpp16<5>(v108_acc, v125_bc, v92_data);
          tensorforge::fmacdpp16<6>(v108_acc, v125_bc, v93_data);
          tensorforge::fmacdpp16<7>(v108_acc, v125_bc, v94_data);
          tensorforge::fmacdpp16<8>(v108_acc, v125_bc, v95_data);
          ir5[0] = v96_acc;
          ir5[1] = v97_acc;
          ir5[2] = v98_acc;
          ir5[3] = v99_acc;
          ir5[4] = v100_acc;
          ir5[5] = v101_acc;
          ir5[6] = v102_acc;
          ir5[7] = v103_acc;
          ir5[8] = v104_acc;
          ir5[9] = v105_acc;
          ir5[10] = v106_acc;
          ir5[11] = v107_acc;
          ir5[12] = v108_acc;
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v129_i0 = 0; v129_i0 < 1; ++v129_i0) {
            int32_t v138_lead = v3_lead + (v129_i0 * 32);
            #pragma unroll
            for (int32_t v130_i1 = 0; v130_i1 < 13; ++v130_i1) {
              int32_t v131_a = v129_i0 + v130_i1;
              float v133_data = r5[(v129_i0 + v130_i1)];
              int32_t v140_a = v138_lead + (v130_i1 * 32);
              glb_m3[v140_a] = v133_data;
            }
          }
          ;
        }
      }
    }
  }
}

