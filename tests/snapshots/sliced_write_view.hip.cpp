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
          int32_t v9_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v10_i0 = 0; v10_i0 < 1; ++v10_i0) {
            int32_t v15_lead = v10_i0 * 32;
            int32_t v16_lead = v9_lead + v15_lead;
            int32_t v23_lead = v9_lead + v15_lead;
            #pragma unroll
            for (int32_t v11_i1 = 10; v11_i1 < 13; ++v11_i1) {
              int32_t v17_a = v11_i1 * 32;
              int32_t v18_a = v16_lead + v17_a;
              float v26_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + v17_a)]);
              r0[(v10_i0 + (v11_i1 - 10))] = v26_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          float v30_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v30_lin;
          float v31_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v31_lin;
          float v32_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v32_lin;
          float v33_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v33_lin;
          float v34_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v34_lin;
          float v35_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v35_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float v37_data = r0[0];
          float v38_data = r0[1];
          float v39_data = r0[2];
          float v40_acc{};
          float v41_lin = r1[0];
          float v42_bc = tensorforge::broadcast<32, 16, 0>(v41_lin);
          tensorforge::fmacdpp16<0>(v40_acc, v42_bc, v37_data);
          tensorforge::fmacdpp16<1>(v40_acc, v42_bc, v38_data);
          tensorforge::fmacdpp16<2>(v40_acc, v42_bc, v39_data);
          r2[0] = v40_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v46_i0 = 0; v46_i0 < 1; ++v46_i0) {
            int32_t v55_lead = v9_lead + (v46_i0 * 32);
            #pragma unroll
            for (int32_t v47_i1 = 0; v47_i1 < 1; ++v47_i1) {
              int32_t v48_a = v46_i0 + v47_i1;
              float v50_data = r2[(v46_i0 + v47_i1)];
              glb_m0[(v55_lead + ((v47_i1 + 8) * 32))] = v50_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v63_i0 = 0; v63_i0 < 1; ++v63_i0) {
            int32_t v68_lead = v63_i0 * 32;
            int32_t v69_lead = v9_lead + v68_lead;
            int32_t v76_lead = v9_lead + v68_lead;
            #pragma unroll
            for (int32_t v64_i1 = 0; v64_i1 < 13; ++v64_i1) {
              int32_t v70_a = v64_i1 * 32;
              int32_t v71_a = v69_lead + v70_a;
              float v79_data = glb_m0[(v76_lead + v70_a)];
              r3[(v63_i0 + v64_i1)] = v79_data;
            }
          }
          float r4[13]{};
          // r4 = load{g>r}(glb_m4);
          float v82_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v82_lin;
          float v83_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v83_lin;
          float v84_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v84_lin;
          float v85_lin = glb_m4[96 + threadIdx.x * 1];
          r4[3] = v85_lin;
          float v86_lin = glb_m4[128 + threadIdx.x * 1];
          r4[4] = v86_lin;
          float v87_lin = glb_m4[160 + threadIdx.x * 1];
          r4[5] = v87_lin;
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
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
          float v101_data = r3[12];
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
          float v114_acc{};
          float v115_lin = r4[0];
          float v116_bc = tensorforge::broadcast<32, 16, 0>(v115_lin);
          tensorforge::fmacdpp16<0>(v102_acc, v116_bc, v89_data);
          tensorforge::fmacdpp16<1>(v102_acc, v116_bc, v90_data);
          tensorforge::fmacdpp16<2>(v102_acc, v116_bc, v91_data);
          tensorforge::fmacdpp16<3>(v102_acc, v116_bc, v92_data);
          tensorforge::fmacdpp16<4>(v102_acc, v116_bc, v93_data);
          tensorforge::fmacdpp16<5>(v102_acc, v116_bc, v94_data);
          tensorforge::fmacdpp16<6>(v102_acc, v116_bc, v95_data);
          tensorforge::fmacdpp16<7>(v102_acc, v116_bc, v96_data);
          tensorforge::fmacdpp16<8>(v102_acc, v116_bc, v97_data);
          tensorforge::fmacdpp16<9>(v102_acc, v116_bc, v98_data);
          tensorforge::fmacdpp16<10>(v102_acc, v116_bc, v99_data);
          tensorforge::fmacdpp16<11>(v102_acc, v116_bc, v100_data);
          tensorforge::fmacdpp16<12>(v102_acc, v116_bc, v101_data);
          tensorforge::fmacdpp16<13>(v103_acc, v116_bc, v89_data);
          tensorforge::fmacdpp16<14>(v103_acc, v116_bc, v90_data);
          tensorforge::fmacdpp16<15>(v103_acc, v116_bc, v91_data);
          float v117_bc = tensorforge::broadcast<32, 16, 1>(v115_lin);
          tensorforge::fmacdpp16<0>(v103_acc, v117_bc, v92_data);
          tensorforge::fmacdpp16<1>(v103_acc, v117_bc, v93_data);
          tensorforge::fmacdpp16<2>(v103_acc, v117_bc, v94_data);
          tensorforge::fmacdpp16<3>(v103_acc, v117_bc, v95_data);
          tensorforge::fmacdpp16<4>(v103_acc, v117_bc, v96_data);
          tensorforge::fmacdpp16<5>(v103_acc, v117_bc, v97_data);
          tensorforge::fmacdpp16<6>(v103_acc, v117_bc, v98_data);
          tensorforge::fmacdpp16<7>(v103_acc, v117_bc, v99_data);
          tensorforge::fmacdpp16<8>(v103_acc, v117_bc, v100_data);
          tensorforge::fmacdpp16<9>(v103_acc, v117_bc, v101_data);
          tensorforge::fmacdpp16<10>(v104_acc, v117_bc, v89_data);
          tensorforge::fmacdpp16<11>(v104_acc, v117_bc, v90_data);
          tensorforge::fmacdpp16<12>(v104_acc, v117_bc, v91_data);
          tensorforge::fmacdpp16<13>(v104_acc, v117_bc, v92_data);
          tensorforge::fmacdpp16<14>(v104_acc, v117_bc, v93_data);
          tensorforge::fmacdpp16<15>(v104_acc, v117_bc, v94_data);
          float v118_lin = r4[1];
          float v119_bc = tensorforge::broadcast<32, 16, 0>(v118_lin);
          tensorforge::fmacdpp16<0>(v104_acc, v119_bc, v95_data);
          tensorforge::fmacdpp16<1>(v104_acc, v119_bc, v96_data);
          tensorforge::fmacdpp16<2>(v104_acc, v119_bc, v97_data);
          tensorforge::fmacdpp16<3>(v104_acc, v119_bc, v98_data);
          tensorforge::fmacdpp16<4>(v104_acc, v119_bc, v99_data);
          tensorforge::fmacdpp16<5>(v104_acc, v119_bc, v100_data);
          tensorforge::fmacdpp16<6>(v104_acc, v119_bc, v101_data);
          tensorforge::fmacdpp16<7>(v105_acc, v119_bc, v89_data);
          tensorforge::fmacdpp16<8>(v105_acc, v119_bc, v90_data);
          tensorforge::fmacdpp16<9>(v105_acc, v119_bc, v91_data);
          tensorforge::fmacdpp16<10>(v105_acc, v119_bc, v92_data);
          tensorforge::fmacdpp16<11>(v105_acc, v119_bc, v93_data);
          tensorforge::fmacdpp16<12>(v105_acc, v119_bc, v94_data);
          tensorforge::fmacdpp16<13>(v105_acc, v119_bc, v95_data);
          tensorforge::fmacdpp16<14>(v105_acc, v119_bc, v96_data);
          tensorforge::fmacdpp16<15>(v105_acc, v119_bc, v97_data);
          float v120_bc = tensorforge::broadcast<32, 16, 1>(v118_lin);
          tensorforge::fmacdpp16<0>(v105_acc, v120_bc, v98_data);
          tensorforge::fmacdpp16<1>(v105_acc, v120_bc, v99_data);
          tensorforge::fmacdpp16<2>(v105_acc, v120_bc, v100_data);
          tensorforge::fmacdpp16<3>(v105_acc, v120_bc, v101_data);
          tensorforge::fmacdpp16<4>(v106_acc, v120_bc, v89_data);
          tensorforge::fmacdpp16<5>(v106_acc, v120_bc, v90_data);
          tensorforge::fmacdpp16<6>(v106_acc, v120_bc, v91_data);
          tensorforge::fmacdpp16<7>(v106_acc, v120_bc, v92_data);
          tensorforge::fmacdpp16<8>(v106_acc, v120_bc, v93_data);
          tensorforge::fmacdpp16<9>(v106_acc, v120_bc, v94_data);
          tensorforge::fmacdpp16<10>(v106_acc, v120_bc, v95_data);
          tensorforge::fmacdpp16<11>(v106_acc, v120_bc, v96_data);
          tensorforge::fmacdpp16<12>(v106_acc, v120_bc, v97_data);
          tensorforge::fmacdpp16<13>(v106_acc, v120_bc, v98_data);
          tensorforge::fmacdpp16<14>(v106_acc, v120_bc, v99_data);
          tensorforge::fmacdpp16<15>(v106_acc, v120_bc, v100_data);
          float v121_lin = r4[2];
          float v122_bc = tensorforge::broadcast<32, 16, 0>(v121_lin);
          tensorforge::fmacdpp16<0>(v106_acc, v122_bc, v101_data);
          tensorforge::fmacdpp16<1>(v107_acc, v122_bc, v89_data);
          tensorforge::fmacdpp16<2>(v107_acc, v122_bc, v90_data);
          tensorforge::fmacdpp16<3>(v107_acc, v122_bc, v91_data);
          tensorforge::fmacdpp16<4>(v107_acc, v122_bc, v92_data);
          tensorforge::fmacdpp16<5>(v107_acc, v122_bc, v93_data);
          tensorforge::fmacdpp16<6>(v107_acc, v122_bc, v94_data);
          tensorforge::fmacdpp16<7>(v107_acc, v122_bc, v95_data);
          tensorforge::fmacdpp16<8>(v107_acc, v122_bc, v96_data);
          tensorforge::fmacdpp16<9>(v107_acc, v122_bc, v97_data);
          tensorforge::fmacdpp16<10>(v107_acc, v122_bc, v98_data);
          tensorforge::fmacdpp16<11>(v107_acc, v122_bc, v99_data);
          tensorforge::fmacdpp16<12>(v107_acc, v122_bc, v100_data);
          tensorforge::fmacdpp16<13>(v107_acc, v122_bc, v101_data);
          tensorforge::fmacdpp16<14>(v108_acc, v122_bc, v89_data);
          tensorforge::fmacdpp16<15>(v108_acc, v122_bc, v90_data);
          float v123_bc = tensorforge::broadcast<32, 16, 1>(v121_lin);
          tensorforge::fmacdpp16<0>(v108_acc, v123_bc, v91_data);
          tensorforge::fmacdpp16<1>(v108_acc, v123_bc, v92_data);
          tensorforge::fmacdpp16<2>(v108_acc, v123_bc, v93_data);
          tensorforge::fmacdpp16<3>(v108_acc, v123_bc, v94_data);
          tensorforge::fmacdpp16<4>(v108_acc, v123_bc, v95_data);
          tensorforge::fmacdpp16<5>(v108_acc, v123_bc, v96_data);
          tensorforge::fmacdpp16<6>(v108_acc, v123_bc, v97_data);
          tensorforge::fmacdpp16<7>(v108_acc, v123_bc, v98_data);
          tensorforge::fmacdpp16<8>(v108_acc, v123_bc, v99_data);
          tensorforge::fmacdpp16<9>(v108_acc, v123_bc, v100_data);
          tensorforge::fmacdpp16<10>(v108_acc, v123_bc, v101_data);
          tensorforge::fmacdpp16<11>(v109_acc, v123_bc, v89_data);
          tensorforge::fmacdpp16<12>(v109_acc, v123_bc, v90_data);
          tensorforge::fmacdpp16<13>(v109_acc, v123_bc, v91_data);
          tensorforge::fmacdpp16<14>(v109_acc, v123_bc, v92_data);
          tensorforge::fmacdpp16<15>(v109_acc, v123_bc, v93_data);
          float v124_lin = r4[3];
          float v125_bc = tensorforge::broadcast<32, 16, 0>(v124_lin);
          tensorforge::fmacdpp16<0>(v109_acc, v125_bc, v94_data);
          tensorforge::fmacdpp16<1>(v109_acc, v125_bc, v95_data);
          tensorforge::fmacdpp16<2>(v109_acc, v125_bc, v96_data);
          tensorforge::fmacdpp16<3>(v109_acc, v125_bc, v97_data);
          tensorforge::fmacdpp16<4>(v109_acc, v125_bc, v98_data);
          tensorforge::fmacdpp16<5>(v109_acc, v125_bc, v99_data);
          tensorforge::fmacdpp16<6>(v109_acc, v125_bc, v100_data);
          tensorforge::fmacdpp16<7>(v109_acc, v125_bc, v101_data);
          tensorforge::fmacdpp16<8>(v110_acc, v125_bc, v89_data);
          tensorforge::fmacdpp16<9>(v110_acc, v125_bc, v90_data);
          tensorforge::fmacdpp16<10>(v110_acc, v125_bc, v91_data);
          tensorforge::fmacdpp16<11>(v110_acc, v125_bc, v92_data);
          tensorforge::fmacdpp16<12>(v110_acc, v125_bc, v93_data);
          tensorforge::fmacdpp16<13>(v110_acc, v125_bc, v94_data);
          tensorforge::fmacdpp16<14>(v110_acc, v125_bc, v95_data);
          tensorforge::fmacdpp16<15>(v110_acc, v125_bc, v96_data);
          float v126_bc = tensorforge::broadcast<32, 16, 1>(v124_lin);
          tensorforge::fmacdpp16<0>(v110_acc, v126_bc, v97_data);
          tensorforge::fmacdpp16<1>(v110_acc, v126_bc, v98_data);
          tensorforge::fmacdpp16<2>(v110_acc, v126_bc, v99_data);
          tensorforge::fmacdpp16<3>(v110_acc, v126_bc, v100_data);
          tensorforge::fmacdpp16<4>(v110_acc, v126_bc, v101_data);
          tensorforge::fmacdpp16<5>(v111_acc, v126_bc, v89_data);
          tensorforge::fmacdpp16<6>(v111_acc, v126_bc, v90_data);
          tensorforge::fmacdpp16<7>(v111_acc, v126_bc, v91_data);
          tensorforge::fmacdpp16<8>(v111_acc, v126_bc, v92_data);
          tensorforge::fmacdpp16<9>(v111_acc, v126_bc, v93_data);
          tensorforge::fmacdpp16<10>(v111_acc, v126_bc, v94_data);
          tensorforge::fmacdpp16<11>(v111_acc, v126_bc, v95_data);
          tensorforge::fmacdpp16<12>(v111_acc, v126_bc, v96_data);
          tensorforge::fmacdpp16<13>(v111_acc, v126_bc, v97_data);
          tensorforge::fmacdpp16<14>(v111_acc, v126_bc, v98_data);
          tensorforge::fmacdpp16<15>(v111_acc, v126_bc, v99_data);
          float v127_lin = r4[4];
          float v128_bc = tensorforge::broadcast<32, 16, 0>(v127_lin);
          tensorforge::fmacdpp16<0>(v111_acc, v128_bc, v100_data);
          tensorforge::fmacdpp16<1>(v111_acc, v128_bc, v101_data);
          tensorforge::fmacdpp16<2>(v112_acc, v128_bc, v89_data);
          tensorforge::fmacdpp16<3>(v112_acc, v128_bc, v90_data);
          tensorforge::fmacdpp16<4>(v112_acc, v128_bc, v91_data);
          tensorforge::fmacdpp16<5>(v112_acc, v128_bc, v92_data);
          tensorforge::fmacdpp16<6>(v112_acc, v128_bc, v93_data);
          tensorforge::fmacdpp16<7>(v112_acc, v128_bc, v94_data);
          tensorforge::fmacdpp16<8>(v112_acc, v128_bc, v95_data);
          tensorforge::fmacdpp16<9>(v112_acc, v128_bc, v96_data);
          tensorforge::fmacdpp16<10>(v112_acc, v128_bc, v97_data);
          tensorforge::fmacdpp16<11>(v112_acc, v128_bc, v98_data);
          tensorforge::fmacdpp16<12>(v112_acc, v128_bc, v99_data);
          tensorforge::fmacdpp16<13>(v112_acc, v128_bc, v100_data);
          tensorforge::fmacdpp16<14>(v112_acc, v128_bc, v101_data);
          tensorforge::fmacdpp16<15>(v113_acc, v128_bc, v89_data);
          float v129_bc = tensorforge::broadcast<32, 16, 1>(v127_lin);
          tensorforge::fmacdpp16<0>(v113_acc, v129_bc, v90_data);
          tensorforge::fmacdpp16<1>(v113_acc, v129_bc, v91_data);
          tensorforge::fmacdpp16<2>(v113_acc, v129_bc, v92_data);
          tensorforge::fmacdpp16<3>(v113_acc, v129_bc, v93_data);
          tensorforge::fmacdpp16<4>(v113_acc, v129_bc, v94_data);
          tensorforge::fmacdpp16<5>(v113_acc, v129_bc, v95_data);
          tensorforge::fmacdpp16<6>(v113_acc, v129_bc, v96_data);
          tensorforge::fmacdpp16<7>(v113_acc, v129_bc, v97_data);
          tensorforge::fmacdpp16<8>(v113_acc, v129_bc, v98_data);
          tensorforge::fmacdpp16<9>(v113_acc, v129_bc, v99_data);
          tensorforge::fmacdpp16<10>(v113_acc, v129_bc, v100_data);
          tensorforge::fmacdpp16<11>(v113_acc, v129_bc, v101_data);
          tensorforge::fmacdpp16<12>(v114_acc, v129_bc, v89_data);
          tensorforge::fmacdpp16<13>(v114_acc, v129_bc, v90_data);
          tensorforge::fmacdpp16<14>(v114_acc, v129_bc, v91_data);
          tensorforge::fmacdpp16<15>(v114_acc, v129_bc, v92_data);
          float v130_lin = r4[5];
          float v131_bc = tensorforge::broadcast<32, 16, 0>(v130_lin);
          tensorforge::fmacdpp16<0>(v114_acc, v131_bc, v93_data);
          tensorforge::fmacdpp16<1>(v114_acc, v131_bc, v94_data);
          tensorforge::fmacdpp16<2>(v114_acc, v131_bc, v95_data);
          tensorforge::fmacdpp16<3>(v114_acc, v131_bc, v96_data);
          tensorforge::fmacdpp16<4>(v114_acc, v131_bc, v97_data);
          tensorforge::fmacdpp16<5>(v114_acc, v131_bc, v98_data);
          tensorforge::fmacdpp16<6>(v114_acc, v131_bc, v99_data);
          tensorforge::fmacdpp16<7>(v114_acc, v131_bc, v100_data);
          tensorforge::fmacdpp16<8>(v114_acc, v131_bc, v101_data);
          r5[0] = v102_acc;
          r5[1] = v103_acc;
          r5[2] = v104_acc;
          r5[3] = v105_acc;
          r5[4] = v106_acc;
          r5[5] = v107_acc;
          r5[6] = v108_acc;
          r5[7] = v109_acc;
          r5[8] = v110_acc;
          r5[9] = v111_acc;
          r5[10] = v112_acc;
          r5[11] = v113_acc;
          r5[12] = v114_acc;
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v135_i0 = 0; v135_i0 < 1; ++v135_i0) {
            int32_t v144_lead = v9_lead + (v135_i0 * 32);
            #pragma unroll
            for (int32_t v136_i1 = 0; v136_i1 < 13; ++v136_i1) {
              int32_t v137_a = v135_i0 + v136_i1;
              float v139_data = r5[(v135_i0 + v136_i1)];
              glb_m3[(v144_lead + (v136_i1 * 32))] = v139_data;
            }
          }
        }
      }
    }
  }
}

