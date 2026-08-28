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
          int32_t v8_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
            int32_t v14_lead = v9_i0 * 32;
            int32_t v15_lead = v8_lead + v14_lead;
            int32_t v22_lead = v8_lead + v14_lead;
            #pragma unroll
            for (int32_t v10_i1 = 10; v10_i1 < 13; ++v10_i1) {
              int32_t v16_a = v10_i1 * 32;
              int32_t v17_a = v15_lead + v16_a;
              float v25_data = __builtin_nontemporal_load(&glb_m1[(v22_lead + v16_a)]);
              int32_t v27_a = v9_i0 + (v10_i1 - 10);
              r0[v27_a] = v25_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          float v29_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v29_lin;
          float v30_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v30_lin;
          float v31_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v31_lin;
          float v32_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v32_lin;
          float v33_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v33_lin;
          float v34_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v34_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float v36_data = r0[0];
          float v37_data = r0[1];
          float v38_data = r0[2];
          float v39_acc{};
          float v40_lin = r1[0];
          float v41_bc = tensorforge::broadcast<32, 16, 0>(v40_lin);
          tensorforge::fmacdpp16<0>(v39_acc, v41_bc, v36_data);
          tensorforge::fmacdpp16<1>(v39_acc, v41_bc, v37_data);
          tensorforge::fmacdpp16<2>(v39_acc, v41_bc, v38_data);
          r2[0] = v39_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v45_i0 = 0; v45_i0 < 1; ++v45_i0) {
            int32_t v54_lead = v8_lead + (v45_i0 * 32);
            #pragma unroll
            for (int32_t v46_i1 = 0; v46_i1 < 1; ++v46_i1) {
              int32_t v47_a = v45_i0 + v46_i1;
              float v49_data = r2[(v45_i0 + v46_i1)];
              glb_m0[(v54_lead + ((v46_i1 + 8) * 32))] = v49_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v62_i0 = 0; v62_i0 < 1; ++v62_i0) {
            int32_t v67_lead = v62_i0 * 32;
            int32_t v68_lead = v8_lead + v67_lead;
            int32_t v75_lead = v8_lead + v67_lead;
            #pragma unroll
            for (int32_t v63_i1 = 0; v63_i1 < 13; ++v63_i1) {
              int32_t v69_a = v63_i1 * 32;
              int32_t v70_a = v68_lead + v69_a;
              float v78_data = glb_m0[(v75_lead + v69_a)];
              int32_t v79_a = v62_i0 + v63_i1;
              r3[v79_a] = v78_data;
            }
          }
          float r4[13]{};
          // r4 = load{g>r}(glb_m4);
          float v81_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v81_lin;
          float v82_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v82_lin;
          float v83_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v83_lin;
          float v84_lin = glb_m4[96 + threadIdx.x * 1];
          r4[3] = v84_lin;
          float v85_lin = glb_m4[128 + threadIdx.x * 1];
          r4[4] = v85_lin;
          float v86_lin = glb_m4[160 + threadIdx.x * 1];
          r4[5] = v86_lin;
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v88_data = r3[0];
          float v89_data = r3[1];
          float v90_data = r3[2];
          float v91_data = r3[3];
          float v92_data = r3[4];
          float v93_data = r3[5];
          float v94_data = r3[6];
          float v95_data = r3[7];
          float v96_data = r3[8];
          float v97_data = r3[9];
          float v98_data = r3[10];
          float v99_data = r3[11];
          float v100_data = r3[12];
          float v101_acc{};
          float v102_acc{};
          float v103_acc{};
          float v104_acc{};
          float v105_acc{};
          float v106_acc{};
          float v107_acc{};
          float v108_acc{};
          float v109_acc{};
          float v110_acc{};
          float v111_acc{};
          float v112_acc{};
          float v113_acc{};
          float v114_lin = r4[0];
          float v115_bc = tensorforge::broadcast<32, 16, 0>(v114_lin);
          tensorforge::fmacdpp16<0>(v101_acc, v115_bc, v88_data);
          tensorforge::fmacdpp16<1>(v101_acc, v115_bc, v89_data);
          tensorforge::fmacdpp16<2>(v101_acc, v115_bc, v90_data);
          tensorforge::fmacdpp16<3>(v101_acc, v115_bc, v91_data);
          tensorforge::fmacdpp16<4>(v101_acc, v115_bc, v92_data);
          tensorforge::fmacdpp16<5>(v101_acc, v115_bc, v93_data);
          tensorforge::fmacdpp16<6>(v101_acc, v115_bc, v94_data);
          tensorforge::fmacdpp16<7>(v101_acc, v115_bc, v95_data);
          tensorforge::fmacdpp16<8>(v101_acc, v115_bc, v96_data);
          tensorforge::fmacdpp16<9>(v101_acc, v115_bc, v97_data);
          tensorforge::fmacdpp16<10>(v101_acc, v115_bc, v98_data);
          tensorforge::fmacdpp16<11>(v101_acc, v115_bc, v99_data);
          tensorforge::fmacdpp16<12>(v101_acc, v115_bc, v100_data);
          tensorforge::fmacdpp16<13>(v102_acc, v115_bc, v88_data);
          tensorforge::fmacdpp16<14>(v102_acc, v115_bc, v89_data);
          tensorforge::fmacdpp16<15>(v102_acc, v115_bc, v90_data);
          float v116_bc = tensorforge::broadcast<32, 16, 1>(v114_lin);
          tensorforge::fmacdpp16<0>(v102_acc, v116_bc, v91_data);
          tensorforge::fmacdpp16<1>(v102_acc, v116_bc, v92_data);
          tensorforge::fmacdpp16<2>(v102_acc, v116_bc, v93_data);
          tensorforge::fmacdpp16<3>(v102_acc, v116_bc, v94_data);
          tensorforge::fmacdpp16<4>(v102_acc, v116_bc, v95_data);
          tensorforge::fmacdpp16<5>(v102_acc, v116_bc, v96_data);
          tensorforge::fmacdpp16<6>(v102_acc, v116_bc, v97_data);
          tensorforge::fmacdpp16<7>(v102_acc, v116_bc, v98_data);
          tensorforge::fmacdpp16<8>(v102_acc, v116_bc, v99_data);
          tensorforge::fmacdpp16<9>(v102_acc, v116_bc, v100_data);
          tensorforge::fmacdpp16<10>(v103_acc, v116_bc, v88_data);
          tensorforge::fmacdpp16<11>(v103_acc, v116_bc, v89_data);
          tensorforge::fmacdpp16<12>(v103_acc, v116_bc, v90_data);
          tensorforge::fmacdpp16<13>(v103_acc, v116_bc, v91_data);
          tensorforge::fmacdpp16<14>(v103_acc, v116_bc, v92_data);
          tensorforge::fmacdpp16<15>(v103_acc, v116_bc, v93_data);
          float v117_lin = r4[1];
          float v118_bc = tensorforge::broadcast<32, 16, 0>(v117_lin);
          tensorforge::fmacdpp16<0>(v103_acc, v118_bc, v94_data);
          tensorforge::fmacdpp16<1>(v103_acc, v118_bc, v95_data);
          tensorforge::fmacdpp16<2>(v103_acc, v118_bc, v96_data);
          tensorforge::fmacdpp16<3>(v103_acc, v118_bc, v97_data);
          tensorforge::fmacdpp16<4>(v103_acc, v118_bc, v98_data);
          tensorforge::fmacdpp16<5>(v103_acc, v118_bc, v99_data);
          tensorforge::fmacdpp16<6>(v103_acc, v118_bc, v100_data);
          tensorforge::fmacdpp16<7>(v104_acc, v118_bc, v88_data);
          tensorforge::fmacdpp16<8>(v104_acc, v118_bc, v89_data);
          tensorforge::fmacdpp16<9>(v104_acc, v118_bc, v90_data);
          tensorforge::fmacdpp16<10>(v104_acc, v118_bc, v91_data);
          tensorforge::fmacdpp16<11>(v104_acc, v118_bc, v92_data);
          tensorforge::fmacdpp16<12>(v104_acc, v118_bc, v93_data);
          tensorforge::fmacdpp16<13>(v104_acc, v118_bc, v94_data);
          tensorforge::fmacdpp16<14>(v104_acc, v118_bc, v95_data);
          tensorforge::fmacdpp16<15>(v104_acc, v118_bc, v96_data);
          float v119_bc = tensorforge::broadcast<32, 16, 1>(v117_lin);
          tensorforge::fmacdpp16<0>(v104_acc, v119_bc, v97_data);
          tensorforge::fmacdpp16<1>(v104_acc, v119_bc, v98_data);
          tensorforge::fmacdpp16<2>(v104_acc, v119_bc, v99_data);
          tensorforge::fmacdpp16<3>(v104_acc, v119_bc, v100_data);
          tensorforge::fmacdpp16<4>(v105_acc, v119_bc, v88_data);
          tensorforge::fmacdpp16<5>(v105_acc, v119_bc, v89_data);
          tensorforge::fmacdpp16<6>(v105_acc, v119_bc, v90_data);
          tensorforge::fmacdpp16<7>(v105_acc, v119_bc, v91_data);
          tensorforge::fmacdpp16<8>(v105_acc, v119_bc, v92_data);
          tensorforge::fmacdpp16<9>(v105_acc, v119_bc, v93_data);
          tensorforge::fmacdpp16<10>(v105_acc, v119_bc, v94_data);
          tensorforge::fmacdpp16<11>(v105_acc, v119_bc, v95_data);
          tensorforge::fmacdpp16<12>(v105_acc, v119_bc, v96_data);
          tensorforge::fmacdpp16<13>(v105_acc, v119_bc, v97_data);
          tensorforge::fmacdpp16<14>(v105_acc, v119_bc, v98_data);
          tensorforge::fmacdpp16<15>(v105_acc, v119_bc, v99_data);
          float v120_lin = r4[2];
          float v121_bc = tensorforge::broadcast<32, 16, 0>(v120_lin);
          tensorforge::fmacdpp16<0>(v105_acc, v121_bc, v100_data);
          tensorforge::fmacdpp16<1>(v106_acc, v121_bc, v88_data);
          tensorforge::fmacdpp16<2>(v106_acc, v121_bc, v89_data);
          tensorforge::fmacdpp16<3>(v106_acc, v121_bc, v90_data);
          tensorforge::fmacdpp16<4>(v106_acc, v121_bc, v91_data);
          tensorforge::fmacdpp16<5>(v106_acc, v121_bc, v92_data);
          tensorforge::fmacdpp16<6>(v106_acc, v121_bc, v93_data);
          tensorforge::fmacdpp16<7>(v106_acc, v121_bc, v94_data);
          tensorforge::fmacdpp16<8>(v106_acc, v121_bc, v95_data);
          tensorforge::fmacdpp16<9>(v106_acc, v121_bc, v96_data);
          tensorforge::fmacdpp16<10>(v106_acc, v121_bc, v97_data);
          tensorforge::fmacdpp16<11>(v106_acc, v121_bc, v98_data);
          tensorforge::fmacdpp16<12>(v106_acc, v121_bc, v99_data);
          tensorforge::fmacdpp16<13>(v106_acc, v121_bc, v100_data);
          tensorforge::fmacdpp16<14>(v107_acc, v121_bc, v88_data);
          tensorforge::fmacdpp16<15>(v107_acc, v121_bc, v89_data);
          float v122_bc = tensorforge::broadcast<32, 16, 1>(v120_lin);
          tensorforge::fmacdpp16<0>(v107_acc, v122_bc, v90_data);
          tensorforge::fmacdpp16<1>(v107_acc, v122_bc, v91_data);
          tensorforge::fmacdpp16<2>(v107_acc, v122_bc, v92_data);
          tensorforge::fmacdpp16<3>(v107_acc, v122_bc, v93_data);
          tensorforge::fmacdpp16<4>(v107_acc, v122_bc, v94_data);
          tensorforge::fmacdpp16<5>(v107_acc, v122_bc, v95_data);
          tensorforge::fmacdpp16<6>(v107_acc, v122_bc, v96_data);
          tensorforge::fmacdpp16<7>(v107_acc, v122_bc, v97_data);
          tensorforge::fmacdpp16<8>(v107_acc, v122_bc, v98_data);
          tensorforge::fmacdpp16<9>(v107_acc, v122_bc, v99_data);
          tensorforge::fmacdpp16<10>(v107_acc, v122_bc, v100_data);
          tensorforge::fmacdpp16<11>(v108_acc, v122_bc, v88_data);
          tensorforge::fmacdpp16<12>(v108_acc, v122_bc, v89_data);
          tensorforge::fmacdpp16<13>(v108_acc, v122_bc, v90_data);
          tensorforge::fmacdpp16<14>(v108_acc, v122_bc, v91_data);
          tensorforge::fmacdpp16<15>(v108_acc, v122_bc, v92_data);
          float v123_lin = r4[3];
          float v124_bc = tensorforge::broadcast<32, 16, 0>(v123_lin);
          tensorforge::fmacdpp16<0>(v108_acc, v124_bc, v93_data);
          tensorforge::fmacdpp16<1>(v108_acc, v124_bc, v94_data);
          tensorforge::fmacdpp16<2>(v108_acc, v124_bc, v95_data);
          tensorforge::fmacdpp16<3>(v108_acc, v124_bc, v96_data);
          tensorforge::fmacdpp16<4>(v108_acc, v124_bc, v97_data);
          tensorforge::fmacdpp16<5>(v108_acc, v124_bc, v98_data);
          tensorforge::fmacdpp16<6>(v108_acc, v124_bc, v99_data);
          tensorforge::fmacdpp16<7>(v108_acc, v124_bc, v100_data);
          tensorforge::fmacdpp16<8>(v109_acc, v124_bc, v88_data);
          tensorforge::fmacdpp16<9>(v109_acc, v124_bc, v89_data);
          tensorforge::fmacdpp16<10>(v109_acc, v124_bc, v90_data);
          tensorforge::fmacdpp16<11>(v109_acc, v124_bc, v91_data);
          tensorforge::fmacdpp16<12>(v109_acc, v124_bc, v92_data);
          tensorforge::fmacdpp16<13>(v109_acc, v124_bc, v93_data);
          tensorforge::fmacdpp16<14>(v109_acc, v124_bc, v94_data);
          tensorforge::fmacdpp16<15>(v109_acc, v124_bc, v95_data);
          float v125_bc = tensorforge::broadcast<32, 16, 1>(v123_lin);
          tensorforge::fmacdpp16<0>(v109_acc, v125_bc, v96_data);
          tensorforge::fmacdpp16<1>(v109_acc, v125_bc, v97_data);
          tensorforge::fmacdpp16<2>(v109_acc, v125_bc, v98_data);
          tensorforge::fmacdpp16<3>(v109_acc, v125_bc, v99_data);
          tensorforge::fmacdpp16<4>(v109_acc, v125_bc, v100_data);
          tensorforge::fmacdpp16<5>(v110_acc, v125_bc, v88_data);
          tensorforge::fmacdpp16<6>(v110_acc, v125_bc, v89_data);
          tensorforge::fmacdpp16<7>(v110_acc, v125_bc, v90_data);
          tensorforge::fmacdpp16<8>(v110_acc, v125_bc, v91_data);
          tensorforge::fmacdpp16<9>(v110_acc, v125_bc, v92_data);
          tensorforge::fmacdpp16<10>(v110_acc, v125_bc, v93_data);
          tensorforge::fmacdpp16<11>(v110_acc, v125_bc, v94_data);
          tensorforge::fmacdpp16<12>(v110_acc, v125_bc, v95_data);
          tensorforge::fmacdpp16<13>(v110_acc, v125_bc, v96_data);
          tensorforge::fmacdpp16<14>(v110_acc, v125_bc, v97_data);
          tensorforge::fmacdpp16<15>(v110_acc, v125_bc, v98_data);
          float v126_lin = r4[4];
          float v127_bc = tensorforge::broadcast<32, 16, 0>(v126_lin);
          tensorforge::fmacdpp16<0>(v110_acc, v127_bc, v99_data);
          tensorforge::fmacdpp16<1>(v110_acc, v127_bc, v100_data);
          tensorforge::fmacdpp16<2>(v111_acc, v127_bc, v88_data);
          tensorforge::fmacdpp16<3>(v111_acc, v127_bc, v89_data);
          tensorforge::fmacdpp16<4>(v111_acc, v127_bc, v90_data);
          tensorforge::fmacdpp16<5>(v111_acc, v127_bc, v91_data);
          tensorforge::fmacdpp16<6>(v111_acc, v127_bc, v92_data);
          tensorforge::fmacdpp16<7>(v111_acc, v127_bc, v93_data);
          tensorforge::fmacdpp16<8>(v111_acc, v127_bc, v94_data);
          tensorforge::fmacdpp16<9>(v111_acc, v127_bc, v95_data);
          tensorforge::fmacdpp16<10>(v111_acc, v127_bc, v96_data);
          tensorforge::fmacdpp16<11>(v111_acc, v127_bc, v97_data);
          tensorforge::fmacdpp16<12>(v111_acc, v127_bc, v98_data);
          tensorforge::fmacdpp16<13>(v111_acc, v127_bc, v99_data);
          tensorforge::fmacdpp16<14>(v111_acc, v127_bc, v100_data);
          tensorforge::fmacdpp16<15>(v112_acc, v127_bc, v88_data);
          float v128_bc = tensorforge::broadcast<32, 16, 1>(v126_lin);
          tensorforge::fmacdpp16<0>(v112_acc, v128_bc, v89_data);
          tensorforge::fmacdpp16<1>(v112_acc, v128_bc, v90_data);
          tensorforge::fmacdpp16<2>(v112_acc, v128_bc, v91_data);
          tensorforge::fmacdpp16<3>(v112_acc, v128_bc, v92_data);
          tensorforge::fmacdpp16<4>(v112_acc, v128_bc, v93_data);
          tensorforge::fmacdpp16<5>(v112_acc, v128_bc, v94_data);
          tensorforge::fmacdpp16<6>(v112_acc, v128_bc, v95_data);
          tensorforge::fmacdpp16<7>(v112_acc, v128_bc, v96_data);
          tensorforge::fmacdpp16<8>(v112_acc, v128_bc, v97_data);
          tensorforge::fmacdpp16<9>(v112_acc, v128_bc, v98_data);
          tensorforge::fmacdpp16<10>(v112_acc, v128_bc, v99_data);
          tensorforge::fmacdpp16<11>(v112_acc, v128_bc, v100_data);
          tensorforge::fmacdpp16<12>(v113_acc, v128_bc, v88_data);
          tensorforge::fmacdpp16<13>(v113_acc, v128_bc, v89_data);
          tensorforge::fmacdpp16<14>(v113_acc, v128_bc, v90_data);
          tensorforge::fmacdpp16<15>(v113_acc, v128_bc, v91_data);
          float v129_lin = r4[5];
          float v130_bc = tensorforge::broadcast<32, 16, 0>(v129_lin);
          tensorforge::fmacdpp16<0>(v113_acc, v130_bc, v92_data);
          tensorforge::fmacdpp16<1>(v113_acc, v130_bc, v93_data);
          tensorforge::fmacdpp16<2>(v113_acc, v130_bc, v94_data);
          tensorforge::fmacdpp16<3>(v113_acc, v130_bc, v95_data);
          tensorforge::fmacdpp16<4>(v113_acc, v130_bc, v96_data);
          tensorforge::fmacdpp16<5>(v113_acc, v130_bc, v97_data);
          tensorforge::fmacdpp16<6>(v113_acc, v130_bc, v98_data);
          tensorforge::fmacdpp16<7>(v113_acc, v130_bc, v99_data);
          tensorforge::fmacdpp16<8>(v113_acc, v130_bc, v100_data);
          r5[0] = v101_acc;
          r5[1] = v102_acc;
          r5[2] = v103_acc;
          r5[3] = v104_acc;
          r5[4] = v105_acc;
          r5[5] = v106_acc;
          r5[6] = v107_acc;
          r5[7] = v108_acc;
          r5[8] = v109_acc;
          r5[9] = v110_acc;
          r5[10] = v111_acc;
          r5[11] = v112_acc;
          r5[12] = v113_acc;
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v134_i0 = 0; v134_i0 < 1; ++v134_i0) {
            int32_t v143_lead = v8_lead + (v134_i0 * 32);
            #pragma unroll
            for (int32_t v135_i1 = 0; v135_i1 < 13; ++v135_i1) {
              int32_t v136_a = v134_i0 + v135_i1;
              float v138_data = r5[(v134_i0 + v135_i1)];
              glb_m3[(v143_lead + (v135_i1 * 32))] = v138_data;
            }
          }
        }
      }
    }
  }
}

