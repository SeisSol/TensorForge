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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v18_lead = v13_i0 * 32;
            int32_t v19_lead = v12_lead + v18_lead;
            int32_t v26_lead = v12_lead + v18_lead;
            #pragma unroll
            for (int32_t v14_i1 = 10; v14_i1 < 13; ++v14_i1) {
              int32_t v20_a = v14_i1 * 32;
              int32_t v21_a = v19_lead + v20_a;
              float v29_data = __builtin_nontemporal_load(&glb_m1[(v26_lead + v20_a)]);
              r0[(v13_i0 + (v14_i1 - 10))] = v29_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          float v33_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v33_lin;
          float v34_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v34_lin;
          float v35_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v35_lin;
          float v36_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v36_lin;
          float v37_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v37_lin;
          float v38_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v38_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float v40_data = r0[0];
          float v41_data = r0[1];
          float v42_data = r0[2];
          float v43_acc{};
          float v44_lin = r1[0];
          float v45_bc = tensorforge::broadcast<32, 16, 0>(v44_lin);
          tensorforge::fmacdpp16<0>(v43_acc, v45_bc, v40_data);
          tensorforge::fmacdpp16<1>(v43_acc, v45_bc, v41_data);
          tensorforge::fmacdpp16<2>(v43_acc, v45_bc, v42_data);
          r2[0] = v43_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v49_i0 = 0; v49_i0 < 1; ++v49_i0) {
            int32_t v58_lead = v12_lead + (v49_i0 * 32);
            #pragma unroll
            for (int32_t v50_i1 = 0; v50_i1 < 1; ++v50_i1) {
              int32_t v51_a = v49_i0 + v50_i1;
              float v53_data = r2[(v49_i0 + v50_i1)];
              glb_m0[(v58_lead + ((v50_i1 + 8) * 32))] = v53_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v66_i0 = 0; v66_i0 < 1; ++v66_i0) {
            int32_t v71_lead = v66_i0 * 32;
            int32_t v72_lead = v12_lead + v71_lead;
            int32_t v79_lead = v12_lead + v71_lead;
            #pragma unroll
            for (int32_t v67_i1 = 0; v67_i1 < 13; ++v67_i1) {
              int32_t v73_a = v67_i1 * 32;
              int32_t v74_a = v72_lead + v73_a;
              float v82_data = glb_m0[(v79_lead + v73_a)];
              r3[(v66_i0 + v67_i1)] = v82_data;
            }
          }
          float r4[13]{};
          // r4 = load{g>r}(glb_m4);
          float v85_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v85_lin;
          float v86_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v86_lin;
          float v87_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v87_lin;
          float v88_lin = glb_m4[96 + threadIdx.x * 1];
          r4[3] = v88_lin;
          float v89_lin = glb_m4[128 + threadIdx.x * 1];
          r4[4] = v89_lin;
          float v90_lin = glb_m4[160 + threadIdx.x * 1];
          r4[5] = v90_lin;
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v92_data = r3[0];
          float v93_data = r3[1];
          float v94_data = r3[2];
          float v95_data = r3[3];
          float v96_data = r3[4];
          float v97_data = r3[5];
          float v98_data = r3[6];
          float v99_data = r3[7];
          float v100_data = r3[8];
          float v101_data = r3[9];
          float v102_data = r3[10];
          float v103_data = r3[11];
          float v104_data = r3[12];
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
          float v115_acc{};
          float v116_acc{};
          float v117_acc{};
          float v118_lin = r4[0];
          float v119_bc = tensorforge::broadcast<32, 16, 0>(v118_lin);
          tensorforge::fmacdpp16<0>(v105_acc, v119_bc, v92_data);
          tensorforge::fmacdpp16<1>(v105_acc, v119_bc, v93_data);
          tensorforge::fmacdpp16<2>(v105_acc, v119_bc, v94_data);
          tensorforge::fmacdpp16<3>(v105_acc, v119_bc, v95_data);
          tensorforge::fmacdpp16<4>(v105_acc, v119_bc, v96_data);
          tensorforge::fmacdpp16<5>(v105_acc, v119_bc, v97_data);
          tensorforge::fmacdpp16<6>(v105_acc, v119_bc, v98_data);
          tensorforge::fmacdpp16<7>(v105_acc, v119_bc, v99_data);
          tensorforge::fmacdpp16<8>(v105_acc, v119_bc, v100_data);
          tensorforge::fmacdpp16<9>(v105_acc, v119_bc, v101_data);
          tensorforge::fmacdpp16<10>(v105_acc, v119_bc, v102_data);
          tensorforge::fmacdpp16<11>(v105_acc, v119_bc, v103_data);
          tensorforge::fmacdpp16<12>(v105_acc, v119_bc, v104_data);
          tensorforge::fmacdpp16<13>(v106_acc, v119_bc, v92_data);
          tensorforge::fmacdpp16<14>(v106_acc, v119_bc, v93_data);
          tensorforge::fmacdpp16<15>(v106_acc, v119_bc, v94_data);
          float v120_bc = tensorforge::broadcast<32, 16, 1>(v118_lin);
          tensorforge::fmacdpp16<0>(v106_acc, v120_bc, v95_data);
          tensorforge::fmacdpp16<1>(v106_acc, v120_bc, v96_data);
          tensorforge::fmacdpp16<2>(v106_acc, v120_bc, v97_data);
          tensorforge::fmacdpp16<3>(v106_acc, v120_bc, v98_data);
          tensorforge::fmacdpp16<4>(v106_acc, v120_bc, v99_data);
          tensorforge::fmacdpp16<5>(v106_acc, v120_bc, v100_data);
          tensorforge::fmacdpp16<6>(v106_acc, v120_bc, v101_data);
          tensorforge::fmacdpp16<7>(v106_acc, v120_bc, v102_data);
          tensorforge::fmacdpp16<8>(v106_acc, v120_bc, v103_data);
          tensorforge::fmacdpp16<9>(v106_acc, v120_bc, v104_data);
          tensorforge::fmacdpp16<10>(v107_acc, v120_bc, v92_data);
          tensorforge::fmacdpp16<11>(v107_acc, v120_bc, v93_data);
          tensorforge::fmacdpp16<12>(v107_acc, v120_bc, v94_data);
          tensorforge::fmacdpp16<13>(v107_acc, v120_bc, v95_data);
          tensorforge::fmacdpp16<14>(v107_acc, v120_bc, v96_data);
          tensorforge::fmacdpp16<15>(v107_acc, v120_bc, v97_data);
          float v121_lin = r4[1];
          float v122_bc = tensorforge::broadcast<32, 16, 0>(v121_lin);
          tensorforge::fmacdpp16<0>(v107_acc, v122_bc, v98_data);
          tensorforge::fmacdpp16<1>(v107_acc, v122_bc, v99_data);
          tensorforge::fmacdpp16<2>(v107_acc, v122_bc, v100_data);
          tensorforge::fmacdpp16<3>(v107_acc, v122_bc, v101_data);
          tensorforge::fmacdpp16<4>(v107_acc, v122_bc, v102_data);
          tensorforge::fmacdpp16<5>(v107_acc, v122_bc, v103_data);
          tensorforge::fmacdpp16<6>(v107_acc, v122_bc, v104_data);
          tensorforge::fmacdpp16<7>(v108_acc, v122_bc, v92_data);
          tensorforge::fmacdpp16<8>(v108_acc, v122_bc, v93_data);
          tensorforge::fmacdpp16<9>(v108_acc, v122_bc, v94_data);
          tensorforge::fmacdpp16<10>(v108_acc, v122_bc, v95_data);
          tensorforge::fmacdpp16<11>(v108_acc, v122_bc, v96_data);
          tensorforge::fmacdpp16<12>(v108_acc, v122_bc, v97_data);
          tensorforge::fmacdpp16<13>(v108_acc, v122_bc, v98_data);
          tensorforge::fmacdpp16<14>(v108_acc, v122_bc, v99_data);
          tensorforge::fmacdpp16<15>(v108_acc, v122_bc, v100_data);
          float v123_bc = tensorforge::broadcast<32, 16, 1>(v121_lin);
          tensorforge::fmacdpp16<0>(v108_acc, v123_bc, v101_data);
          tensorforge::fmacdpp16<1>(v108_acc, v123_bc, v102_data);
          tensorforge::fmacdpp16<2>(v108_acc, v123_bc, v103_data);
          tensorforge::fmacdpp16<3>(v108_acc, v123_bc, v104_data);
          tensorforge::fmacdpp16<4>(v109_acc, v123_bc, v92_data);
          tensorforge::fmacdpp16<5>(v109_acc, v123_bc, v93_data);
          tensorforge::fmacdpp16<6>(v109_acc, v123_bc, v94_data);
          tensorforge::fmacdpp16<7>(v109_acc, v123_bc, v95_data);
          tensorforge::fmacdpp16<8>(v109_acc, v123_bc, v96_data);
          tensorforge::fmacdpp16<9>(v109_acc, v123_bc, v97_data);
          tensorforge::fmacdpp16<10>(v109_acc, v123_bc, v98_data);
          tensorforge::fmacdpp16<11>(v109_acc, v123_bc, v99_data);
          tensorforge::fmacdpp16<12>(v109_acc, v123_bc, v100_data);
          tensorforge::fmacdpp16<13>(v109_acc, v123_bc, v101_data);
          tensorforge::fmacdpp16<14>(v109_acc, v123_bc, v102_data);
          tensorforge::fmacdpp16<15>(v109_acc, v123_bc, v103_data);
          float v124_lin = r4[2];
          float v125_bc = tensorforge::broadcast<32, 16, 0>(v124_lin);
          tensorforge::fmacdpp16<0>(v109_acc, v125_bc, v104_data);
          tensorforge::fmacdpp16<1>(v110_acc, v125_bc, v92_data);
          tensorforge::fmacdpp16<2>(v110_acc, v125_bc, v93_data);
          tensorforge::fmacdpp16<3>(v110_acc, v125_bc, v94_data);
          tensorforge::fmacdpp16<4>(v110_acc, v125_bc, v95_data);
          tensorforge::fmacdpp16<5>(v110_acc, v125_bc, v96_data);
          tensorforge::fmacdpp16<6>(v110_acc, v125_bc, v97_data);
          tensorforge::fmacdpp16<7>(v110_acc, v125_bc, v98_data);
          tensorforge::fmacdpp16<8>(v110_acc, v125_bc, v99_data);
          tensorforge::fmacdpp16<9>(v110_acc, v125_bc, v100_data);
          tensorforge::fmacdpp16<10>(v110_acc, v125_bc, v101_data);
          tensorforge::fmacdpp16<11>(v110_acc, v125_bc, v102_data);
          tensorforge::fmacdpp16<12>(v110_acc, v125_bc, v103_data);
          tensorforge::fmacdpp16<13>(v110_acc, v125_bc, v104_data);
          tensorforge::fmacdpp16<14>(v111_acc, v125_bc, v92_data);
          tensorforge::fmacdpp16<15>(v111_acc, v125_bc, v93_data);
          float v126_bc = tensorforge::broadcast<32, 16, 1>(v124_lin);
          tensorforge::fmacdpp16<0>(v111_acc, v126_bc, v94_data);
          tensorforge::fmacdpp16<1>(v111_acc, v126_bc, v95_data);
          tensorforge::fmacdpp16<2>(v111_acc, v126_bc, v96_data);
          tensorforge::fmacdpp16<3>(v111_acc, v126_bc, v97_data);
          tensorforge::fmacdpp16<4>(v111_acc, v126_bc, v98_data);
          tensorforge::fmacdpp16<5>(v111_acc, v126_bc, v99_data);
          tensorforge::fmacdpp16<6>(v111_acc, v126_bc, v100_data);
          tensorforge::fmacdpp16<7>(v111_acc, v126_bc, v101_data);
          tensorforge::fmacdpp16<8>(v111_acc, v126_bc, v102_data);
          tensorforge::fmacdpp16<9>(v111_acc, v126_bc, v103_data);
          tensorforge::fmacdpp16<10>(v111_acc, v126_bc, v104_data);
          tensorforge::fmacdpp16<11>(v112_acc, v126_bc, v92_data);
          tensorforge::fmacdpp16<12>(v112_acc, v126_bc, v93_data);
          tensorforge::fmacdpp16<13>(v112_acc, v126_bc, v94_data);
          tensorforge::fmacdpp16<14>(v112_acc, v126_bc, v95_data);
          tensorforge::fmacdpp16<15>(v112_acc, v126_bc, v96_data);
          float v127_lin = r4[3];
          float v128_bc = tensorforge::broadcast<32, 16, 0>(v127_lin);
          tensorforge::fmacdpp16<0>(v112_acc, v128_bc, v97_data);
          tensorforge::fmacdpp16<1>(v112_acc, v128_bc, v98_data);
          tensorforge::fmacdpp16<2>(v112_acc, v128_bc, v99_data);
          tensorforge::fmacdpp16<3>(v112_acc, v128_bc, v100_data);
          tensorforge::fmacdpp16<4>(v112_acc, v128_bc, v101_data);
          tensorforge::fmacdpp16<5>(v112_acc, v128_bc, v102_data);
          tensorforge::fmacdpp16<6>(v112_acc, v128_bc, v103_data);
          tensorforge::fmacdpp16<7>(v112_acc, v128_bc, v104_data);
          tensorforge::fmacdpp16<8>(v113_acc, v128_bc, v92_data);
          tensorforge::fmacdpp16<9>(v113_acc, v128_bc, v93_data);
          tensorforge::fmacdpp16<10>(v113_acc, v128_bc, v94_data);
          tensorforge::fmacdpp16<11>(v113_acc, v128_bc, v95_data);
          tensorforge::fmacdpp16<12>(v113_acc, v128_bc, v96_data);
          tensorforge::fmacdpp16<13>(v113_acc, v128_bc, v97_data);
          tensorforge::fmacdpp16<14>(v113_acc, v128_bc, v98_data);
          tensorforge::fmacdpp16<15>(v113_acc, v128_bc, v99_data);
          float v129_bc = tensorforge::broadcast<32, 16, 1>(v127_lin);
          tensorforge::fmacdpp16<0>(v113_acc, v129_bc, v100_data);
          tensorforge::fmacdpp16<1>(v113_acc, v129_bc, v101_data);
          tensorforge::fmacdpp16<2>(v113_acc, v129_bc, v102_data);
          tensorforge::fmacdpp16<3>(v113_acc, v129_bc, v103_data);
          tensorforge::fmacdpp16<4>(v113_acc, v129_bc, v104_data);
          tensorforge::fmacdpp16<5>(v114_acc, v129_bc, v92_data);
          tensorforge::fmacdpp16<6>(v114_acc, v129_bc, v93_data);
          tensorforge::fmacdpp16<7>(v114_acc, v129_bc, v94_data);
          tensorforge::fmacdpp16<8>(v114_acc, v129_bc, v95_data);
          tensorforge::fmacdpp16<9>(v114_acc, v129_bc, v96_data);
          tensorforge::fmacdpp16<10>(v114_acc, v129_bc, v97_data);
          tensorforge::fmacdpp16<11>(v114_acc, v129_bc, v98_data);
          tensorforge::fmacdpp16<12>(v114_acc, v129_bc, v99_data);
          tensorforge::fmacdpp16<13>(v114_acc, v129_bc, v100_data);
          tensorforge::fmacdpp16<14>(v114_acc, v129_bc, v101_data);
          tensorforge::fmacdpp16<15>(v114_acc, v129_bc, v102_data);
          float v130_lin = r4[4];
          float v131_bc = tensorforge::broadcast<32, 16, 0>(v130_lin);
          tensorforge::fmacdpp16<0>(v114_acc, v131_bc, v103_data);
          tensorforge::fmacdpp16<1>(v114_acc, v131_bc, v104_data);
          tensorforge::fmacdpp16<2>(v115_acc, v131_bc, v92_data);
          tensorforge::fmacdpp16<3>(v115_acc, v131_bc, v93_data);
          tensorforge::fmacdpp16<4>(v115_acc, v131_bc, v94_data);
          tensorforge::fmacdpp16<5>(v115_acc, v131_bc, v95_data);
          tensorforge::fmacdpp16<6>(v115_acc, v131_bc, v96_data);
          tensorforge::fmacdpp16<7>(v115_acc, v131_bc, v97_data);
          tensorforge::fmacdpp16<8>(v115_acc, v131_bc, v98_data);
          tensorforge::fmacdpp16<9>(v115_acc, v131_bc, v99_data);
          tensorforge::fmacdpp16<10>(v115_acc, v131_bc, v100_data);
          tensorforge::fmacdpp16<11>(v115_acc, v131_bc, v101_data);
          tensorforge::fmacdpp16<12>(v115_acc, v131_bc, v102_data);
          tensorforge::fmacdpp16<13>(v115_acc, v131_bc, v103_data);
          tensorforge::fmacdpp16<14>(v115_acc, v131_bc, v104_data);
          tensorforge::fmacdpp16<15>(v116_acc, v131_bc, v92_data);
          float v132_bc = tensorforge::broadcast<32, 16, 1>(v130_lin);
          tensorforge::fmacdpp16<0>(v116_acc, v132_bc, v93_data);
          tensorforge::fmacdpp16<1>(v116_acc, v132_bc, v94_data);
          tensorforge::fmacdpp16<2>(v116_acc, v132_bc, v95_data);
          tensorforge::fmacdpp16<3>(v116_acc, v132_bc, v96_data);
          tensorforge::fmacdpp16<4>(v116_acc, v132_bc, v97_data);
          tensorforge::fmacdpp16<5>(v116_acc, v132_bc, v98_data);
          tensorforge::fmacdpp16<6>(v116_acc, v132_bc, v99_data);
          tensorforge::fmacdpp16<7>(v116_acc, v132_bc, v100_data);
          tensorforge::fmacdpp16<8>(v116_acc, v132_bc, v101_data);
          tensorforge::fmacdpp16<9>(v116_acc, v132_bc, v102_data);
          tensorforge::fmacdpp16<10>(v116_acc, v132_bc, v103_data);
          tensorforge::fmacdpp16<11>(v116_acc, v132_bc, v104_data);
          tensorforge::fmacdpp16<12>(v117_acc, v132_bc, v92_data);
          tensorforge::fmacdpp16<13>(v117_acc, v132_bc, v93_data);
          tensorforge::fmacdpp16<14>(v117_acc, v132_bc, v94_data);
          tensorforge::fmacdpp16<15>(v117_acc, v132_bc, v95_data);
          float v133_lin = r4[5];
          float v134_bc = tensorforge::broadcast<32, 16, 0>(v133_lin);
          tensorforge::fmacdpp16<0>(v117_acc, v134_bc, v96_data);
          tensorforge::fmacdpp16<1>(v117_acc, v134_bc, v97_data);
          tensorforge::fmacdpp16<2>(v117_acc, v134_bc, v98_data);
          tensorforge::fmacdpp16<3>(v117_acc, v134_bc, v99_data);
          tensorforge::fmacdpp16<4>(v117_acc, v134_bc, v100_data);
          tensorforge::fmacdpp16<5>(v117_acc, v134_bc, v101_data);
          tensorforge::fmacdpp16<6>(v117_acc, v134_bc, v102_data);
          tensorforge::fmacdpp16<7>(v117_acc, v134_bc, v103_data);
          tensorforge::fmacdpp16<8>(v117_acc, v134_bc, v104_data);
          r5[0] = v105_acc;
          r5[1] = v106_acc;
          r5[2] = v107_acc;
          r5[3] = v108_acc;
          r5[4] = v109_acc;
          r5[5] = v110_acc;
          r5[6] = v111_acc;
          r5[7] = v112_acc;
          r5[8] = v113_acc;
          r5[9] = v114_acc;
          r5[10] = v115_acc;
          r5[11] = v116_acc;
          r5[12] = v117_acc;
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v138_i0 = 0; v138_i0 < 1; ++v138_i0) {
            int32_t v147_lead = v12_lead + (v138_i0 * 32);
            #pragma unroll
            for (int32_t v139_i1 = 0; v139_i1 < 13; ++v139_i1) {
              int32_t v140_a = v138_i0 + v139_i1;
              float v142_data = r5[(v138_i0 + v139_i1)];
              glb_m3[(v147_lead + (v139_i1 * 32))] = v142_data;
            }
          }
        }
      }
    }
  }
}

