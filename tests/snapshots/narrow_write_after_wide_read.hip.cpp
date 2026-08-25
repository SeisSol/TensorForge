// === base name ===
kernel_154580c330

// === header ===
void launcher_kernel_154580c330(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_154580c330(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_154580c330, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_154580c330), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_154580c330, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_154580c330(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×12(32×12) {0..32}×{0..12} strided
    // m2 12×13(12×13) {0..12}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1]
    // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] += m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×13(12×13) {0..12}×{0..13} strided({0..12}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1]
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 156 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 13; ++v4_i1) {
              int32_t v10_a = v4_i1 * 32;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = glb_m0[(v16_lead + v10_a)];
              int32_t v20_a = v3_i0 + v4_i1;
              r0[v20_a] = v19_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          int32_t v23_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v29_lead = v24_i0 * 32;
            int32_t v30_lead = v23_lead + v29_lead;
            int32_t v37_lead = v23_lead + v29_lead;
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              int32_t v31_a = v25_i1 * 32;
              int32_t v32_a = v30_lead + v31_a;
              float v40_data = __builtin_nontemporal_load(&glb_m1[(v37_lead + v31_a)]);
              int32_t v41_a = v24_i0 + v25_i1;
              r2[v41_a] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          auto& ir1 = r1;
          float v45_data = r0[0];
          float v46_data = ir1[0];
          ir1[0] = (v46_data + v45_data);
          float v48_data = r0[1];
          float v49_data = ir1[1];
          ir1[1] = (v49_data + v48_data);
          float v51_data = r0[2];
          float v52_data = ir1[2];
          ir1[2] = (v52_data + v51_data);
          float v54_data = r0[3];
          float v55_data = ir1[3];
          ir1[3] = (v55_data + v54_data);
          float v57_data = r0[4];
          float v58_data = ir1[4];
          ir1[4] = (v58_data + v57_data);
          float v60_data = r0[5];
          float v61_data = ir1[5];
          ir1[5] = (v61_data + v60_data);
          float v63_data = r0[6];
          float v64_data = ir1[6];
          ir1[6] = (v64_data + v63_data);
          float v66_data = r0[7];
          float v67_data = ir1[7];
          ir1[7] = (v67_data + v66_data);
          float v69_data = r0[8];
          float v70_data = ir1[8];
          ir1[8] = (v70_data + v69_data);
          float v72_data = r0[9];
          float v73_data = ir1[9];
          ir1[9] = (v73_data + v72_data);
          float v75_data = r0[10];
          float v76_data = ir1[10];
          ir1[10] = (v76_data + v75_data);
          float v78_data = r0[11];
          float v79_data = ir1[11];
          ir1[11] = (v79_data + v78_data);
          float v81_data = r0[12];
          float v82_data = ir1[12];
          ir1[12] = (v82_data + v81_data);
          float r3[13]{};
          {
            // r3 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r3[0] = v0;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r3[1] = v32;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r3[2] = v64;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r3[3] = v96;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r3[4] = v128;
          }
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          {
            // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 32), (0, 13)] [(0, 12)]
            float ir4[13]{};
            float v84_data = r2[0];
            float v85_data = r2[1];
            float v86_data = r2[2];
            float v87_data = r2[3];
            float v88_data = r2[4];
            float v89_data = r2[5];
            float v90_data = r2[6];
            float v91_data = r2[7];
            float v92_data = r2[8];
            float v93_data = r2[9];
            float v94_data = r2[10];
            float v95_data = r2[11];
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
            float v109_lin = r3[0];
            float v110_bc = tensorforge::broadcast<32, 16, 0>(v109_lin);
            tensorforge::fmacdpp16<0>(v96_acc, v110_bc, v84_data);
            tensorforge::fmacdpp16<1>(v96_acc, v110_bc, v85_data);
            tensorforge::fmacdpp16<2>(v96_acc, v110_bc, v86_data);
            tensorforge::fmacdpp16<3>(v96_acc, v110_bc, v87_data);
            tensorforge::fmacdpp16<4>(v96_acc, v110_bc, v88_data);
            tensorforge::fmacdpp16<5>(v96_acc, v110_bc, v89_data);
            tensorforge::fmacdpp16<6>(v96_acc, v110_bc, v90_data);
            tensorforge::fmacdpp16<7>(v96_acc, v110_bc, v91_data);
            tensorforge::fmacdpp16<8>(v96_acc, v110_bc, v92_data);
            tensorforge::fmacdpp16<9>(v96_acc, v110_bc, v93_data);
            tensorforge::fmacdpp16<10>(v96_acc, v110_bc, v94_data);
            tensorforge::fmacdpp16<11>(v96_acc, v110_bc, v95_data);
            tensorforge::fmacdpp16<12>(v97_acc, v110_bc, v84_data);
            tensorforge::fmacdpp16<13>(v97_acc, v110_bc, v85_data);
            tensorforge::fmacdpp16<14>(v97_acc, v110_bc, v86_data);
            tensorforge::fmacdpp16<15>(v97_acc, v110_bc, v87_data);
            float v111_bc = tensorforge::broadcast<32, 16, 1>(v109_lin);
            tensorforge::fmacdpp16<0>(v97_acc, v111_bc, v88_data);
            tensorforge::fmacdpp16<1>(v97_acc, v111_bc, v89_data);
            tensorforge::fmacdpp16<2>(v97_acc, v111_bc, v90_data);
            tensorforge::fmacdpp16<3>(v97_acc, v111_bc, v91_data);
            tensorforge::fmacdpp16<4>(v97_acc, v111_bc, v92_data);
            tensorforge::fmacdpp16<5>(v97_acc, v111_bc, v93_data);
            tensorforge::fmacdpp16<6>(v97_acc, v111_bc, v94_data);
            tensorforge::fmacdpp16<7>(v97_acc, v111_bc, v95_data);
            tensorforge::fmacdpp16<8>(v98_acc, v111_bc, v84_data);
            tensorforge::fmacdpp16<9>(v98_acc, v111_bc, v85_data);
            tensorforge::fmacdpp16<10>(v98_acc, v111_bc, v86_data);
            tensorforge::fmacdpp16<11>(v98_acc, v111_bc, v87_data);
            tensorforge::fmacdpp16<12>(v98_acc, v111_bc, v88_data);
            tensorforge::fmacdpp16<13>(v98_acc, v111_bc, v89_data);
            tensorforge::fmacdpp16<14>(v98_acc, v111_bc, v90_data);
            tensorforge::fmacdpp16<15>(v98_acc, v111_bc, v91_data);
            float v112_lin = r3[1];
            float v113_bc = tensorforge::broadcast<32, 16, 0>(v112_lin);
            tensorforge::fmacdpp16<0>(v98_acc, v113_bc, v92_data);
            tensorforge::fmacdpp16<1>(v98_acc, v113_bc, v93_data);
            tensorforge::fmacdpp16<2>(v98_acc, v113_bc, v94_data);
            tensorforge::fmacdpp16<3>(v98_acc, v113_bc, v95_data);
            tensorforge::fmacdpp16<4>(v99_acc, v113_bc, v84_data);
            tensorforge::fmacdpp16<5>(v99_acc, v113_bc, v85_data);
            tensorforge::fmacdpp16<6>(v99_acc, v113_bc, v86_data);
            tensorforge::fmacdpp16<7>(v99_acc, v113_bc, v87_data);
            tensorforge::fmacdpp16<8>(v99_acc, v113_bc, v88_data);
            tensorforge::fmacdpp16<9>(v99_acc, v113_bc, v89_data);
            tensorforge::fmacdpp16<10>(v99_acc, v113_bc, v90_data);
            tensorforge::fmacdpp16<11>(v99_acc, v113_bc, v91_data);
            tensorforge::fmacdpp16<12>(v99_acc, v113_bc, v92_data);
            tensorforge::fmacdpp16<13>(v99_acc, v113_bc, v93_data);
            tensorforge::fmacdpp16<14>(v99_acc, v113_bc, v94_data);
            tensorforge::fmacdpp16<15>(v99_acc, v113_bc, v95_data);
            float v114_bc = tensorforge::broadcast<32, 16, 1>(v112_lin);
            tensorforge::fmacdpp16<0>(v100_acc, v114_bc, v84_data);
            tensorforge::fmacdpp16<1>(v100_acc, v114_bc, v85_data);
            tensorforge::fmacdpp16<2>(v100_acc, v114_bc, v86_data);
            tensorforge::fmacdpp16<3>(v100_acc, v114_bc, v87_data);
            tensorforge::fmacdpp16<4>(v100_acc, v114_bc, v88_data);
            tensorforge::fmacdpp16<5>(v100_acc, v114_bc, v89_data);
            tensorforge::fmacdpp16<6>(v100_acc, v114_bc, v90_data);
            tensorforge::fmacdpp16<7>(v100_acc, v114_bc, v91_data);
            tensorforge::fmacdpp16<8>(v100_acc, v114_bc, v92_data);
            tensorforge::fmacdpp16<9>(v100_acc, v114_bc, v93_data);
            tensorforge::fmacdpp16<10>(v100_acc, v114_bc, v94_data);
            tensorforge::fmacdpp16<11>(v100_acc, v114_bc, v95_data);
            tensorforge::fmacdpp16<12>(v101_acc, v114_bc, v84_data);
            tensorforge::fmacdpp16<13>(v101_acc, v114_bc, v85_data);
            tensorforge::fmacdpp16<14>(v101_acc, v114_bc, v86_data);
            tensorforge::fmacdpp16<15>(v101_acc, v114_bc, v87_data);
            float v115_lin = r3[2];
            float v116_bc = tensorforge::broadcast<32, 16, 0>(v115_lin);
            tensorforge::fmacdpp16<0>(v101_acc, v116_bc, v88_data);
            tensorforge::fmacdpp16<1>(v101_acc, v116_bc, v89_data);
            tensorforge::fmacdpp16<2>(v101_acc, v116_bc, v90_data);
            tensorforge::fmacdpp16<3>(v101_acc, v116_bc, v91_data);
            tensorforge::fmacdpp16<4>(v101_acc, v116_bc, v92_data);
            tensorforge::fmacdpp16<5>(v101_acc, v116_bc, v93_data);
            tensorforge::fmacdpp16<6>(v101_acc, v116_bc, v94_data);
            tensorforge::fmacdpp16<7>(v101_acc, v116_bc, v95_data);
            tensorforge::fmacdpp16<8>(v102_acc, v116_bc, v84_data);
            tensorforge::fmacdpp16<9>(v102_acc, v116_bc, v85_data);
            tensorforge::fmacdpp16<10>(v102_acc, v116_bc, v86_data);
            tensorforge::fmacdpp16<11>(v102_acc, v116_bc, v87_data);
            tensorforge::fmacdpp16<12>(v102_acc, v116_bc, v88_data);
            tensorforge::fmacdpp16<13>(v102_acc, v116_bc, v89_data);
            tensorforge::fmacdpp16<14>(v102_acc, v116_bc, v90_data);
            tensorforge::fmacdpp16<15>(v102_acc, v116_bc, v91_data);
            float v117_bc = tensorforge::broadcast<32, 16, 1>(v115_lin);
            tensorforge::fmacdpp16<0>(v102_acc, v117_bc, v92_data);
            tensorforge::fmacdpp16<1>(v102_acc, v117_bc, v93_data);
            tensorforge::fmacdpp16<2>(v102_acc, v117_bc, v94_data);
            tensorforge::fmacdpp16<3>(v102_acc, v117_bc, v95_data);
            tensorforge::fmacdpp16<4>(v103_acc, v117_bc, v84_data);
            tensorforge::fmacdpp16<5>(v103_acc, v117_bc, v85_data);
            tensorforge::fmacdpp16<6>(v103_acc, v117_bc, v86_data);
            tensorforge::fmacdpp16<7>(v103_acc, v117_bc, v87_data);
            tensorforge::fmacdpp16<8>(v103_acc, v117_bc, v88_data);
            tensorforge::fmacdpp16<9>(v103_acc, v117_bc, v89_data);
            tensorforge::fmacdpp16<10>(v103_acc, v117_bc, v90_data);
            tensorforge::fmacdpp16<11>(v103_acc, v117_bc, v91_data);
            tensorforge::fmacdpp16<12>(v103_acc, v117_bc, v92_data);
            tensorforge::fmacdpp16<13>(v103_acc, v117_bc, v93_data);
            tensorforge::fmacdpp16<14>(v103_acc, v117_bc, v94_data);
            tensorforge::fmacdpp16<15>(v103_acc, v117_bc, v95_data);
            float v118_lin = r3[3];
            float v119_bc = tensorforge::broadcast<32, 16, 0>(v118_lin);
            tensorforge::fmacdpp16<0>(v104_acc, v119_bc, v84_data);
            tensorforge::fmacdpp16<1>(v104_acc, v119_bc, v85_data);
            tensorforge::fmacdpp16<2>(v104_acc, v119_bc, v86_data);
            tensorforge::fmacdpp16<3>(v104_acc, v119_bc, v87_data);
            tensorforge::fmacdpp16<4>(v104_acc, v119_bc, v88_data);
            tensorforge::fmacdpp16<5>(v104_acc, v119_bc, v89_data);
            tensorforge::fmacdpp16<6>(v104_acc, v119_bc, v90_data);
            tensorforge::fmacdpp16<7>(v104_acc, v119_bc, v91_data);
            tensorforge::fmacdpp16<8>(v104_acc, v119_bc, v92_data);
            tensorforge::fmacdpp16<9>(v104_acc, v119_bc, v93_data);
            tensorforge::fmacdpp16<10>(v104_acc, v119_bc, v94_data);
            tensorforge::fmacdpp16<11>(v104_acc, v119_bc, v95_data);
            tensorforge::fmacdpp16<12>(v105_acc, v119_bc, v84_data);
            tensorforge::fmacdpp16<13>(v105_acc, v119_bc, v85_data);
            tensorforge::fmacdpp16<14>(v105_acc, v119_bc, v86_data);
            tensorforge::fmacdpp16<15>(v105_acc, v119_bc, v87_data);
            float v120_bc = tensorforge::broadcast<32, 16, 1>(v118_lin);
            tensorforge::fmacdpp16<0>(v105_acc, v120_bc, v88_data);
            tensorforge::fmacdpp16<1>(v105_acc, v120_bc, v89_data);
            tensorforge::fmacdpp16<2>(v105_acc, v120_bc, v90_data);
            tensorforge::fmacdpp16<3>(v105_acc, v120_bc, v91_data);
            tensorforge::fmacdpp16<4>(v105_acc, v120_bc, v92_data);
            tensorforge::fmacdpp16<5>(v105_acc, v120_bc, v93_data);
            tensorforge::fmacdpp16<6>(v105_acc, v120_bc, v94_data);
            tensorforge::fmacdpp16<7>(v105_acc, v120_bc, v95_data);
            tensorforge::fmacdpp16<8>(v106_acc, v120_bc, v84_data);
            tensorforge::fmacdpp16<9>(v106_acc, v120_bc, v85_data);
            tensorforge::fmacdpp16<10>(v106_acc, v120_bc, v86_data);
            tensorforge::fmacdpp16<11>(v106_acc, v120_bc, v87_data);
            tensorforge::fmacdpp16<12>(v106_acc, v120_bc, v88_data);
            tensorforge::fmacdpp16<13>(v106_acc, v120_bc, v89_data);
            tensorforge::fmacdpp16<14>(v106_acc, v120_bc, v90_data);
            tensorforge::fmacdpp16<15>(v106_acc, v120_bc, v91_data);
            float v121_lin = r3[4];
            float v122_bc = tensorforge::broadcast<32, 16, 0>(v121_lin);
            tensorforge::fmacdpp16<0>(v106_acc, v122_bc, v92_data);
            tensorforge::fmacdpp16<1>(v106_acc, v122_bc, v93_data);
            tensorforge::fmacdpp16<2>(v106_acc, v122_bc, v94_data);
            tensorforge::fmacdpp16<3>(v106_acc, v122_bc, v95_data);
            tensorforge::fmacdpp16<4>(v107_acc, v122_bc, v84_data);
            tensorforge::fmacdpp16<5>(v107_acc, v122_bc, v85_data);
            tensorforge::fmacdpp16<6>(v107_acc, v122_bc, v86_data);
            tensorforge::fmacdpp16<7>(v107_acc, v122_bc, v87_data);
            tensorforge::fmacdpp16<8>(v107_acc, v122_bc, v88_data);
            tensorforge::fmacdpp16<9>(v107_acc, v122_bc, v89_data);
            tensorforge::fmacdpp16<10>(v107_acc, v122_bc, v90_data);
            tensorforge::fmacdpp16<11>(v107_acc, v122_bc, v91_data);
            tensorforge::fmacdpp16<12>(v107_acc, v122_bc, v92_data);
            tensorforge::fmacdpp16<13>(v107_acc, v122_bc, v93_data);
            tensorforge::fmacdpp16<14>(v107_acc, v122_bc, v94_data);
            tensorforge::fmacdpp16<15>(v107_acc, v122_bc, v95_data);
            float v123_bc = tensorforge::broadcast<32, 16, 1>(v121_lin);
            tensorforge::fmacdpp16<0>(v108_acc, v123_bc, v84_data);
            tensorforge::fmacdpp16<1>(v108_acc, v123_bc, v85_data);
            tensorforge::fmacdpp16<2>(v108_acc, v123_bc, v86_data);
            tensorforge::fmacdpp16<3>(v108_acc, v123_bc, v87_data);
            tensorforge::fmacdpp16<4>(v108_acc, v123_bc, v88_data);
            tensorforge::fmacdpp16<5>(v108_acc, v123_bc, v89_data);
            tensorforge::fmacdpp16<6>(v108_acc, v123_bc, v90_data);
            tensorforge::fmacdpp16<7>(v108_acc, v123_bc, v91_data);
            tensorforge::fmacdpp16<8>(v108_acc, v123_bc, v92_data);
            tensorforge::fmacdpp16<9>(v108_acc, v123_bc, v93_data);
            tensorforge::fmacdpp16<10>(v108_acc, v123_bc, v94_data);
            tensorforge::fmacdpp16<11>(v108_acc, v123_bc, v95_data);
            ir4[0] = v96_acc;
            ir4[1] = v97_acc;
            ir4[2] = v98_acc;
            ir4[3] = v99_acc;
            ir4[4] = v100_acc;
            ir4[5] = v101_acc;
            ir4[6] = v102_acc;
            ir4[7] = v103_acc;
            ir4[8] = v104_acc;
            ir4[9] = v105_acc;
            ir4[10] = v106_acc;
            ir4[11] = v107_acc;
            ir4[12] = v108_acc;
            #pragma unroll
            for (int32_t v127_n0 = 0; v127_n0 < 1; ++v127_n0) {
              #pragma unroll
              for (int32_t v128_n1 = 0; v128_n1 < 13; ++v128_n1) {
                int32_t v129_a = v127_n0 + v128_n1;
                int32_t v130_a = v127_n0 + v128_n1;
                float v131_data = ir4[v130_a];
                int32_t v132_a = v127_n0 + v128_n1;
                float v134_data = r1[v130_a];
                int32_t v136_a = v127_n0 + v128_n1;
                r4[v130_a] = (v134_data + v131_data);
              }
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          auto& ir5 = r5;
          float v141_data = r4[4];
          float v142_data = ir5[0];
          ir5[0] = (v142_data + v141_data);
          // glb_m0 = store{r>g}(r5);
          int32_t v146_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v147_i0 = 0; v147_i0 < 1; ++v147_i0) {
            int32_t v156_lead = v146_lead + (v147_i0 * 32);
            #pragma unroll
            for (int32_t v148_i1 = 0; v148_i1 < 1; ++v148_i1) {
              int32_t v149_a = v147_i0 + v148_i1;
              float v151_data = r5[(v147_i0 + v148_i1)];
              int32_t v159_a = v156_lead + ((v148_i1 + 4) * 32);
              glb_m0[v159_a] = v151_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          int32_t v162_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v163_i0 = 0; v163_i0 < 1; ++v163_i0) {
            int32_t v168_lead = v163_i0 * 32;
            int32_t v169_lead = v162_lead + v168_lead;
            int32_t v176_lead = v162_lead + v168_lead;
            #pragma unroll
            for (int32_t v164_i1 = 0; v164_i1 < 13; ++v164_i1) {
              int32_t v170_a = v164_i1 * 32;
              int32_t v171_a = v169_lead + v170_a;
              float v179_data = glb_m0[(v176_lead + v170_a)];
              int32_t v180_a = v163_i0 + v164_i1;
              r6[v180_a] = v179_data;
            }
          }
          float r7[13]{};
          {
            // r7 = load{g>r}(glb_m4);
            float v0 = glb_m4[0 + threadIdx.x * 1];
            r7[0] = v0;
            float v32 = glb_m4[32 + threadIdx.x * 1];
            r7[1] = v32;
            float v64 = glb_m4[64 + threadIdx.x * 1];
            r7[2] = v64;
            float v96 = glb_m4[96 + threadIdx.x * 1];
            r7[3] = v96;
            float v128 = glb_m4[128 + threadIdx.x * 1];
            r7[4] = v128;
            float v160 = glb_m4[160 + threadIdx.x * 1];
            r7[5] = v160;
          }
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          auto& ir8 = r8;
          float v181_data = r6[0];
          float v182_data = r6[1];
          float v183_data = r6[2];
          float v184_data = r6[3];
          float v185_data = r6[4];
          float v186_data = r6[5];
          float v187_data = r6[6];
          float v188_data = r6[7];
          float v189_data = r6[8];
          float v190_data = r6[9];
          float v191_data = r6[10];
          float v192_data = r6[11];
          float v193_data = r6[12];
          float v194_acc{};
          float v195_acc{};
          float v196_acc{};
          float v197_acc{};
          float v198_acc{};
          float v199_acc{};
          float v200_acc{};
          float v201_acc{};
          float v202_acc{};
          float v203_acc{};
          float v204_acc{};
          float v205_acc{};
          float v206_acc{};
          float v207_lin = r7[0];
          float v208_bc = tensorforge::broadcast<32, 16, 0>(v207_lin);
          tensorforge::fmacdpp16<0>(v194_acc, v208_bc, v181_data);
          tensorforge::fmacdpp16<1>(v194_acc, v208_bc, v182_data);
          tensorforge::fmacdpp16<2>(v194_acc, v208_bc, v183_data);
          tensorforge::fmacdpp16<3>(v194_acc, v208_bc, v184_data);
          tensorforge::fmacdpp16<4>(v194_acc, v208_bc, v185_data);
          tensorforge::fmacdpp16<5>(v194_acc, v208_bc, v186_data);
          tensorforge::fmacdpp16<6>(v194_acc, v208_bc, v187_data);
          tensorforge::fmacdpp16<7>(v194_acc, v208_bc, v188_data);
          tensorforge::fmacdpp16<8>(v194_acc, v208_bc, v189_data);
          tensorforge::fmacdpp16<9>(v194_acc, v208_bc, v190_data);
          tensorforge::fmacdpp16<10>(v194_acc, v208_bc, v191_data);
          tensorforge::fmacdpp16<11>(v194_acc, v208_bc, v192_data);
          tensorforge::fmacdpp16<12>(v194_acc, v208_bc, v193_data);
          tensorforge::fmacdpp16<13>(v195_acc, v208_bc, v181_data);
          tensorforge::fmacdpp16<14>(v195_acc, v208_bc, v182_data);
          tensorforge::fmacdpp16<15>(v195_acc, v208_bc, v183_data);
          float v209_bc = tensorforge::broadcast<32, 16, 1>(v207_lin);
          tensorforge::fmacdpp16<0>(v195_acc, v209_bc, v184_data);
          tensorforge::fmacdpp16<1>(v195_acc, v209_bc, v185_data);
          tensorforge::fmacdpp16<2>(v195_acc, v209_bc, v186_data);
          tensorforge::fmacdpp16<3>(v195_acc, v209_bc, v187_data);
          tensorforge::fmacdpp16<4>(v195_acc, v209_bc, v188_data);
          tensorforge::fmacdpp16<5>(v195_acc, v209_bc, v189_data);
          tensorforge::fmacdpp16<6>(v195_acc, v209_bc, v190_data);
          tensorforge::fmacdpp16<7>(v195_acc, v209_bc, v191_data);
          tensorforge::fmacdpp16<8>(v195_acc, v209_bc, v192_data);
          tensorforge::fmacdpp16<9>(v195_acc, v209_bc, v193_data);
          tensorforge::fmacdpp16<10>(v196_acc, v209_bc, v181_data);
          tensorforge::fmacdpp16<11>(v196_acc, v209_bc, v182_data);
          tensorforge::fmacdpp16<12>(v196_acc, v209_bc, v183_data);
          tensorforge::fmacdpp16<13>(v196_acc, v209_bc, v184_data);
          tensorforge::fmacdpp16<14>(v196_acc, v209_bc, v185_data);
          tensorforge::fmacdpp16<15>(v196_acc, v209_bc, v186_data);
          float v210_lin = r7[1];
          float v211_bc = tensorforge::broadcast<32, 16, 0>(v210_lin);
          tensorforge::fmacdpp16<0>(v196_acc, v211_bc, v187_data);
          tensorforge::fmacdpp16<1>(v196_acc, v211_bc, v188_data);
          tensorforge::fmacdpp16<2>(v196_acc, v211_bc, v189_data);
          tensorforge::fmacdpp16<3>(v196_acc, v211_bc, v190_data);
          tensorforge::fmacdpp16<4>(v196_acc, v211_bc, v191_data);
          tensorforge::fmacdpp16<5>(v196_acc, v211_bc, v192_data);
          tensorforge::fmacdpp16<6>(v196_acc, v211_bc, v193_data);
          tensorforge::fmacdpp16<7>(v197_acc, v211_bc, v181_data);
          tensorforge::fmacdpp16<8>(v197_acc, v211_bc, v182_data);
          tensorforge::fmacdpp16<9>(v197_acc, v211_bc, v183_data);
          tensorforge::fmacdpp16<10>(v197_acc, v211_bc, v184_data);
          tensorforge::fmacdpp16<11>(v197_acc, v211_bc, v185_data);
          tensorforge::fmacdpp16<12>(v197_acc, v211_bc, v186_data);
          tensorforge::fmacdpp16<13>(v197_acc, v211_bc, v187_data);
          tensorforge::fmacdpp16<14>(v197_acc, v211_bc, v188_data);
          tensorforge::fmacdpp16<15>(v197_acc, v211_bc, v189_data);
          float v212_bc = tensorforge::broadcast<32, 16, 1>(v210_lin);
          tensorforge::fmacdpp16<0>(v197_acc, v212_bc, v190_data);
          tensorforge::fmacdpp16<1>(v197_acc, v212_bc, v191_data);
          tensorforge::fmacdpp16<2>(v197_acc, v212_bc, v192_data);
          tensorforge::fmacdpp16<3>(v197_acc, v212_bc, v193_data);
          tensorforge::fmacdpp16<4>(v198_acc, v212_bc, v181_data);
          tensorforge::fmacdpp16<5>(v198_acc, v212_bc, v182_data);
          tensorforge::fmacdpp16<6>(v198_acc, v212_bc, v183_data);
          tensorforge::fmacdpp16<7>(v198_acc, v212_bc, v184_data);
          tensorforge::fmacdpp16<8>(v198_acc, v212_bc, v185_data);
          tensorforge::fmacdpp16<9>(v198_acc, v212_bc, v186_data);
          tensorforge::fmacdpp16<10>(v198_acc, v212_bc, v187_data);
          tensorforge::fmacdpp16<11>(v198_acc, v212_bc, v188_data);
          tensorforge::fmacdpp16<12>(v198_acc, v212_bc, v189_data);
          tensorforge::fmacdpp16<13>(v198_acc, v212_bc, v190_data);
          tensorforge::fmacdpp16<14>(v198_acc, v212_bc, v191_data);
          tensorforge::fmacdpp16<15>(v198_acc, v212_bc, v192_data);
          float v213_lin = r7[2];
          float v214_bc = tensorforge::broadcast<32, 16, 0>(v213_lin);
          tensorforge::fmacdpp16<0>(v198_acc, v214_bc, v193_data);
          tensorforge::fmacdpp16<1>(v199_acc, v214_bc, v181_data);
          tensorforge::fmacdpp16<2>(v199_acc, v214_bc, v182_data);
          tensorforge::fmacdpp16<3>(v199_acc, v214_bc, v183_data);
          tensorforge::fmacdpp16<4>(v199_acc, v214_bc, v184_data);
          tensorforge::fmacdpp16<5>(v199_acc, v214_bc, v185_data);
          tensorforge::fmacdpp16<6>(v199_acc, v214_bc, v186_data);
          tensorforge::fmacdpp16<7>(v199_acc, v214_bc, v187_data);
          tensorforge::fmacdpp16<8>(v199_acc, v214_bc, v188_data);
          tensorforge::fmacdpp16<9>(v199_acc, v214_bc, v189_data);
          tensorforge::fmacdpp16<10>(v199_acc, v214_bc, v190_data);
          tensorforge::fmacdpp16<11>(v199_acc, v214_bc, v191_data);
          tensorforge::fmacdpp16<12>(v199_acc, v214_bc, v192_data);
          tensorforge::fmacdpp16<13>(v199_acc, v214_bc, v193_data);
          tensorforge::fmacdpp16<14>(v200_acc, v214_bc, v181_data);
          tensorforge::fmacdpp16<15>(v200_acc, v214_bc, v182_data);
          float v215_bc = tensorforge::broadcast<32, 16, 1>(v213_lin);
          tensorforge::fmacdpp16<0>(v200_acc, v215_bc, v183_data);
          tensorforge::fmacdpp16<1>(v200_acc, v215_bc, v184_data);
          tensorforge::fmacdpp16<2>(v200_acc, v215_bc, v185_data);
          tensorforge::fmacdpp16<3>(v200_acc, v215_bc, v186_data);
          tensorforge::fmacdpp16<4>(v200_acc, v215_bc, v187_data);
          tensorforge::fmacdpp16<5>(v200_acc, v215_bc, v188_data);
          tensorforge::fmacdpp16<6>(v200_acc, v215_bc, v189_data);
          tensorforge::fmacdpp16<7>(v200_acc, v215_bc, v190_data);
          tensorforge::fmacdpp16<8>(v200_acc, v215_bc, v191_data);
          tensorforge::fmacdpp16<9>(v200_acc, v215_bc, v192_data);
          tensorforge::fmacdpp16<10>(v200_acc, v215_bc, v193_data);
          tensorforge::fmacdpp16<11>(v201_acc, v215_bc, v181_data);
          tensorforge::fmacdpp16<12>(v201_acc, v215_bc, v182_data);
          tensorforge::fmacdpp16<13>(v201_acc, v215_bc, v183_data);
          tensorforge::fmacdpp16<14>(v201_acc, v215_bc, v184_data);
          tensorforge::fmacdpp16<15>(v201_acc, v215_bc, v185_data);
          float v216_lin = r7[3];
          float v217_bc = tensorforge::broadcast<32, 16, 0>(v216_lin);
          tensorforge::fmacdpp16<0>(v201_acc, v217_bc, v186_data);
          tensorforge::fmacdpp16<1>(v201_acc, v217_bc, v187_data);
          tensorforge::fmacdpp16<2>(v201_acc, v217_bc, v188_data);
          tensorforge::fmacdpp16<3>(v201_acc, v217_bc, v189_data);
          tensorforge::fmacdpp16<4>(v201_acc, v217_bc, v190_data);
          tensorforge::fmacdpp16<5>(v201_acc, v217_bc, v191_data);
          tensorforge::fmacdpp16<6>(v201_acc, v217_bc, v192_data);
          tensorforge::fmacdpp16<7>(v201_acc, v217_bc, v193_data);
          tensorforge::fmacdpp16<8>(v202_acc, v217_bc, v181_data);
          tensorforge::fmacdpp16<9>(v202_acc, v217_bc, v182_data);
          tensorforge::fmacdpp16<10>(v202_acc, v217_bc, v183_data);
          tensorforge::fmacdpp16<11>(v202_acc, v217_bc, v184_data);
          tensorforge::fmacdpp16<12>(v202_acc, v217_bc, v185_data);
          tensorforge::fmacdpp16<13>(v202_acc, v217_bc, v186_data);
          tensorforge::fmacdpp16<14>(v202_acc, v217_bc, v187_data);
          tensorforge::fmacdpp16<15>(v202_acc, v217_bc, v188_data);
          float v218_bc = tensorforge::broadcast<32, 16, 1>(v216_lin);
          tensorforge::fmacdpp16<0>(v202_acc, v218_bc, v189_data);
          tensorforge::fmacdpp16<1>(v202_acc, v218_bc, v190_data);
          tensorforge::fmacdpp16<2>(v202_acc, v218_bc, v191_data);
          tensorforge::fmacdpp16<3>(v202_acc, v218_bc, v192_data);
          tensorforge::fmacdpp16<4>(v202_acc, v218_bc, v193_data);
          tensorforge::fmacdpp16<5>(v203_acc, v218_bc, v181_data);
          tensorforge::fmacdpp16<6>(v203_acc, v218_bc, v182_data);
          tensorforge::fmacdpp16<7>(v203_acc, v218_bc, v183_data);
          tensorforge::fmacdpp16<8>(v203_acc, v218_bc, v184_data);
          tensorforge::fmacdpp16<9>(v203_acc, v218_bc, v185_data);
          tensorforge::fmacdpp16<10>(v203_acc, v218_bc, v186_data);
          tensorforge::fmacdpp16<11>(v203_acc, v218_bc, v187_data);
          tensorforge::fmacdpp16<12>(v203_acc, v218_bc, v188_data);
          tensorforge::fmacdpp16<13>(v203_acc, v218_bc, v189_data);
          tensorforge::fmacdpp16<14>(v203_acc, v218_bc, v190_data);
          tensorforge::fmacdpp16<15>(v203_acc, v218_bc, v191_data);
          float v219_lin = r7[4];
          float v220_bc = tensorforge::broadcast<32, 16, 0>(v219_lin);
          tensorforge::fmacdpp16<0>(v203_acc, v220_bc, v192_data);
          tensorforge::fmacdpp16<1>(v203_acc, v220_bc, v193_data);
          tensorforge::fmacdpp16<2>(v204_acc, v220_bc, v181_data);
          tensorforge::fmacdpp16<3>(v204_acc, v220_bc, v182_data);
          tensorforge::fmacdpp16<4>(v204_acc, v220_bc, v183_data);
          tensorforge::fmacdpp16<5>(v204_acc, v220_bc, v184_data);
          tensorforge::fmacdpp16<6>(v204_acc, v220_bc, v185_data);
          tensorforge::fmacdpp16<7>(v204_acc, v220_bc, v186_data);
          tensorforge::fmacdpp16<8>(v204_acc, v220_bc, v187_data);
          tensorforge::fmacdpp16<9>(v204_acc, v220_bc, v188_data);
          tensorforge::fmacdpp16<10>(v204_acc, v220_bc, v189_data);
          tensorforge::fmacdpp16<11>(v204_acc, v220_bc, v190_data);
          tensorforge::fmacdpp16<12>(v204_acc, v220_bc, v191_data);
          tensorforge::fmacdpp16<13>(v204_acc, v220_bc, v192_data);
          tensorforge::fmacdpp16<14>(v204_acc, v220_bc, v193_data);
          tensorforge::fmacdpp16<15>(v205_acc, v220_bc, v181_data);
          float v221_bc = tensorforge::broadcast<32, 16, 1>(v219_lin);
          tensorforge::fmacdpp16<0>(v205_acc, v221_bc, v182_data);
          tensorforge::fmacdpp16<1>(v205_acc, v221_bc, v183_data);
          tensorforge::fmacdpp16<2>(v205_acc, v221_bc, v184_data);
          tensorforge::fmacdpp16<3>(v205_acc, v221_bc, v185_data);
          tensorforge::fmacdpp16<4>(v205_acc, v221_bc, v186_data);
          tensorforge::fmacdpp16<5>(v205_acc, v221_bc, v187_data);
          tensorforge::fmacdpp16<6>(v205_acc, v221_bc, v188_data);
          tensorforge::fmacdpp16<7>(v205_acc, v221_bc, v189_data);
          tensorforge::fmacdpp16<8>(v205_acc, v221_bc, v190_data);
          tensorforge::fmacdpp16<9>(v205_acc, v221_bc, v191_data);
          tensorforge::fmacdpp16<10>(v205_acc, v221_bc, v192_data);
          tensorforge::fmacdpp16<11>(v205_acc, v221_bc, v193_data);
          tensorforge::fmacdpp16<12>(v206_acc, v221_bc, v181_data);
          tensorforge::fmacdpp16<13>(v206_acc, v221_bc, v182_data);
          tensorforge::fmacdpp16<14>(v206_acc, v221_bc, v183_data);
          tensorforge::fmacdpp16<15>(v206_acc, v221_bc, v184_data);
          float v222_lin = r7[5];
          float v223_bc = tensorforge::broadcast<32, 16, 0>(v222_lin);
          tensorforge::fmacdpp16<0>(v206_acc, v223_bc, v185_data);
          tensorforge::fmacdpp16<1>(v206_acc, v223_bc, v186_data);
          tensorforge::fmacdpp16<2>(v206_acc, v223_bc, v187_data);
          tensorforge::fmacdpp16<3>(v206_acc, v223_bc, v188_data);
          tensorforge::fmacdpp16<4>(v206_acc, v223_bc, v189_data);
          tensorforge::fmacdpp16<5>(v206_acc, v223_bc, v190_data);
          tensorforge::fmacdpp16<6>(v206_acc, v223_bc, v191_data);
          tensorforge::fmacdpp16<7>(v206_acc, v223_bc, v192_data);
          tensorforge::fmacdpp16<8>(v206_acc, v223_bc, v193_data);
          ir8[0] = v194_acc;
          ir8[1] = v195_acc;
          ir8[2] = v196_acc;
          ir8[3] = v197_acc;
          ir8[4] = v198_acc;
          ir8[5] = v199_acc;
          ir8[6] = v200_acc;
          ir8[7] = v201_acc;
          ir8[8] = v202_acc;
          ir8[9] = v203_acc;
          ir8[10] = v204_acc;
          ir8[11] = v205_acc;
          ir8[12] = v206_acc;
          // glb_m3 = store{r>g}(r8);
          int32_t v226_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v227_i0 = 0; v227_i0 < 1; ++v227_i0) {
            int32_t v236_lead = v226_lead + (v227_i0 * 32);
            #pragma unroll
            for (int32_t v228_i1 = 0; v228_i1 < 13; ++v228_i1) {
              int32_t v229_a = v227_i0 + v228_i1;
              float v231_data = r8[(v227_i0 + v228_i1)];
              int32_t v238_a = v236_lead + (v228_i1 * 32);
              glb_m3[v238_a] = v231_data;
            }
          }
          ;
        }
      }
    }
  }
}

