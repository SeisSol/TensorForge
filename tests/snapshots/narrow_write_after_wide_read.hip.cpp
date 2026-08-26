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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 13; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = glb_m0[(v17_lead + v11_a)];
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v31_lead = v26_i0 * 32;
            int32_t v32_lead = v3_lead + v31_lead;
            int32_t v39_lead = v3_lead + v31_lead;
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              int32_t v33_a = v27_i1 * 32;
              int32_t v34_a = v32_lead + v33_a;
              float v42_data = __builtin_nontemporal_load(&glb_m1[(v39_lead + v33_a)]);
              int32_t v43_a = v26_i0 + v27_i1;
              r2[v43_a] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          auto& ir1 = r1;
          float v48_data = r0[0];
          float v49_data = ir1[0];
          ir1[0] = (v49_data + v48_data);
          float v51_data = r0[1];
          float v52_data = ir1[1];
          ir1[1] = (v52_data + v51_data);
          float v54_data = r0[2];
          float v55_data = ir1[2];
          ir1[2] = (v55_data + v54_data);
          float v57_data = r0[3];
          float v58_data = ir1[3];
          ir1[3] = (v58_data + v57_data);
          float v60_data = r0[4];
          float v61_data = ir1[4];
          ir1[4] = (v61_data + v60_data);
          float v63_data = r0[5];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + v63_data);
          float v66_data = r0[6];
          float v67_data = ir1[6];
          ir1[6] = (v67_data + v66_data);
          float v69_data = r0[7];
          float v70_data = ir1[7];
          ir1[7] = (v70_data + v69_data);
          float v72_data = r0[8];
          float v73_data = ir1[8];
          ir1[8] = (v73_data + v72_data);
          float v75_data = r0[9];
          float v76_data = ir1[9];
          ir1[9] = (v76_data + v75_data);
          float v78_data = r0[10];
          float v79_data = ir1[10];
          ir1[10] = (v79_data + v78_data);
          float v81_data = r0[11];
          float v82_data = ir1[11];
          ir1[11] = (v82_data + v81_data);
          float v84_data = r0[12];
          float v85_data = ir1[12];
          ir1[12] = (v85_data + v84_data);
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
            float v89_data = r2[0];
            float v90_data = r2[1];
            float v91_data = r2[2];
            float v92_data = r2[3];
            float v93_data = r2[4];
            float v94_data = r2[5];
            float v95_data = r2[6];
            float v96_data = r2[7];
            float v97_data = r2[8];
            float v98_data = r2[9];
            float v99_data = r2[10];
            float v100_data = r2[11];
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
            float v114_lin = r3[0];
            float v115_bc = tensorforge::broadcast<32, 16, 0>(v114_lin);
            tensorforge::fmacdpp16<0>(v101_acc, v115_bc, v89_data);
            tensorforge::fmacdpp16<1>(v101_acc, v115_bc, v90_data);
            tensorforge::fmacdpp16<2>(v101_acc, v115_bc, v91_data);
            tensorforge::fmacdpp16<3>(v101_acc, v115_bc, v92_data);
            tensorforge::fmacdpp16<4>(v101_acc, v115_bc, v93_data);
            tensorforge::fmacdpp16<5>(v101_acc, v115_bc, v94_data);
            tensorforge::fmacdpp16<6>(v101_acc, v115_bc, v95_data);
            tensorforge::fmacdpp16<7>(v101_acc, v115_bc, v96_data);
            tensorforge::fmacdpp16<8>(v101_acc, v115_bc, v97_data);
            tensorforge::fmacdpp16<9>(v101_acc, v115_bc, v98_data);
            tensorforge::fmacdpp16<10>(v101_acc, v115_bc, v99_data);
            tensorforge::fmacdpp16<11>(v101_acc, v115_bc, v100_data);
            tensorforge::fmacdpp16<12>(v102_acc, v115_bc, v89_data);
            tensorforge::fmacdpp16<13>(v102_acc, v115_bc, v90_data);
            tensorforge::fmacdpp16<14>(v102_acc, v115_bc, v91_data);
            tensorforge::fmacdpp16<15>(v102_acc, v115_bc, v92_data);
            float v116_bc = tensorforge::broadcast<32, 16, 1>(v114_lin);
            tensorforge::fmacdpp16<0>(v102_acc, v116_bc, v93_data);
            tensorforge::fmacdpp16<1>(v102_acc, v116_bc, v94_data);
            tensorforge::fmacdpp16<2>(v102_acc, v116_bc, v95_data);
            tensorforge::fmacdpp16<3>(v102_acc, v116_bc, v96_data);
            tensorforge::fmacdpp16<4>(v102_acc, v116_bc, v97_data);
            tensorforge::fmacdpp16<5>(v102_acc, v116_bc, v98_data);
            tensorforge::fmacdpp16<6>(v102_acc, v116_bc, v99_data);
            tensorforge::fmacdpp16<7>(v102_acc, v116_bc, v100_data);
            tensorforge::fmacdpp16<8>(v103_acc, v116_bc, v89_data);
            tensorforge::fmacdpp16<9>(v103_acc, v116_bc, v90_data);
            tensorforge::fmacdpp16<10>(v103_acc, v116_bc, v91_data);
            tensorforge::fmacdpp16<11>(v103_acc, v116_bc, v92_data);
            tensorforge::fmacdpp16<12>(v103_acc, v116_bc, v93_data);
            tensorforge::fmacdpp16<13>(v103_acc, v116_bc, v94_data);
            tensorforge::fmacdpp16<14>(v103_acc, v116_bc, v95_data);
            tensorforge::fmacdpp16<15>(v103_acc, v116_bc, v96_data);
            float v117_lin = r3[1];
            float v118_bc = tensorforge::broadcast<32, 16, 0>(v117_lin);
            tensorforge::fmacdpp16<0>(v103_acc, v118_bc, v97_data);
            tensorforge::fmacdpp16<1>(v103_acc, v118_bc, v98_data);
            tensorforge::fmacdpp16<2>(v103_acc, v118_bc, v99_data);
            tensorforge::fmacdpp16<3>(v103_acc, v118_bc, v100_data);
            tensorforge::fmacdpp16<4>(v104_acc, v118_bc, v89_data);
            tensorforge::fmacdpp16<5>(v104_acc, v118_bc, v90_data);
            tensorforge::fmacdpp16<6>(v104_acc, v118_bc, v91_data);
            tensorforge::fmacdpp16<7>(v104_acc, v118_bc, v92_data);
            tensorforge::fmacdpp16<8>(v104_acc, v118_bc, v93_data);
            tensorforge::fmacdpp16<9>(v104_acc, v118_bc, v94_data);
            tensorforge::fmacdpp16<10>(v104_acc, v118_bc, v95_data);
            tensorforge::fmacdpp16<11>(v104_acc, v118_bc, v96_data);
            tensorforge::fmacdpp16<12>(v104_acc, v118_bc, v97_data);
            tensorforge::fmacdpp16<13>(v104_acc, v118_bc, v98_data);
            tensorforge::fmacdpp16<14>(v104_acc, v118_bc, v99_data);
            tensorforge::fmacdpp16<15>(v104_acc, v118_bc, v100_data);
            float v119_bc = tensorforge::broadcast<32, 16, 1>(v117_lin);
            tensorforge::fmacdpp16<0>(v105_acc, v119_bc, v89_data);
            tensorforge::fmacdpp16<1>(v105_acc, v119_bc, v90_data);
            tensorforge::fmacdpp16<2>(v105_acc, v119_bc, v91_data);
            tensorforge::fmacdpp16<3>(v105_acc, v119_bc, v92_data);
            tensorforge::fmacdpp16<4>(v105_acc, v119_bc, v93_data);
            tensorforge::fmacdpp16<5>(v105_acc, v119_bc, v94_data);
            tensorforge::fmacdpp16<6>(v105_acc, v119_bc, v95_data);
            tensorforge::fmacdpp16<7>(v105_acc, v119_bc, v96_data);
            tensorforge::fmacdpp16<8>(v105_acc, v119_bc, v97_data);
            tensorforge::fmacdpp16<9>(v105_acc, v119_bc, v98_data);
            tensorforge::fmacdpp16<10>(v105_acc, v119_bc, v99_data);
            tensorforge::fmacdpp16<11>(v105_acc, v119_bc, v100_data);
            tensorforge::fmacdpp16<12>(v106_acc, v119_bc, v89_data);
            tensorforge::fmacdpp16<13>(v106_acc, v119_bc, v90_data);
            tensorforge::fmacdpp16<14>(v106_acc, v119_bc, v91_data);
            tensorforge::fmacdpp16<15>(v106_acc, v119_bc, v92_data);
            float v120_lin = r3[2];
            float v121_bc = tensorforge::broadcast<32, 16, 0>(v120_lin);
            tensorforge::fmacdpp16<0>(v106_acc, v121_bc, v93_data);
            tensorforge::fmacdpp16<1>(v106_acc, v121_bc, v94_data);
            tensorforge::fmacdpp16<2>(v106_acc, v121_bc, v95_data);
            tensorforge::fmacdpp16<3>(v106_acc, v121_bc, v96_data);
            tensorforge::fmacdpp16<4>(v106_acc, v121_bc, v97_data);
            tensorforge::fmacdpp16<5>(v106_acc, v121_bc, v98_data);
            tensorforge::fmacdpp16<6>(v106_acc, v121_bc, v99_data);
            tensorforge::fmacdpp16<7>(v106_acc, v121_bc, v100_data);
            tensorforge::fmacdpp16<8>(v107_acc, v121_bc, v89_data);
            tensorforge::fmacdpp16<9>(v107_acc, v121_bc, v90_data);
            tensorforge::fmacdpp16<10>(v107_acc, v121_bc, v91_data);
            tensorforge::fmacdpp16<11>(v107_acc, v121_bc, v92_data);
            tensorforge::fmacdpp16<12>(v107_acc, v121_bc, v93_data);
            tensorforge::fmacdpp16<13>(v107_acc, v121_bc, v94_data);
            tensorforge::fmacdpp16<14>(v107_acc, v121_bc, v95_data);
            tensorforge::fmacdpp16<15>(v107_acc, v121_bc, v96_data);
            float v122_bc = tensorforge::broadcast<32, 16, 1>(v120_lin);
            tensorforge::fmacdpp16<0>(v107_acc, v122_bc, v97_data);
            tensorforge::fmacdpp16<1>(v107_acc, v122_bc, v98_data);
            tensorforge::fmacdpp16<2>(v107_acc, v122_bc, v99_data);
            tensorforge::fmacdpp16<3>(v107_acc, v122_bc, v100_data);
            tensorforge::fmacdpp16<4>(v108_acc, v122_bc, v89_data);
            tensorforge::fmacdpp16<5>(v108_acc, v122_bc, v90_data);
            tensorforge::fmacdpp16<6>(v108_acc, v122_bc, v91_data);
            tensorforge::fmacdpp16<7>(v108_acc, v122_bc, v92_data);
            tensorforge::fmacdpp16<8>(v108_acc, v122_bc, v93_data);
            tensorforge::fmacdpp16<9>(v108_acc, v122_bc, v94_data);
            tensorforge::fmacdpp16<10>(v108_acc, v122_bc, v95_data);
            tensorforge::fmacdpp16<11>(v108_acc, v122_bc, v96_data);
            tensorforge::fmacdpp16<12>(v108_acc, v122_bc, v97_data);
            tensorforge::fmacdpp16<13>(v108_acc, v122_bc, v98_data);
            tensorforge::fmacdpp16<14>(v108_acc, v122_bc, v99_data);
            tensorforge::fmacdpp16<15>(v108_acc, v122_bc, v100_data);
            float v123_lin = r3[3];
            float v124_bc = tensorforge::broadcast<32, 16, 0>(v123_lin);
            tensorforge::fmacdpp16<0>(v109_acc, v124_bc, v89_data);
            tensorforge::fmacdpp16<1>(v109_acc, v124_bc, v90_data);
            tensorforge::fmacdpp16<2>(v109_acc, v124_bc, v91_data);
            tensorforge::fmacdpp16<3>(v109_acc, v124_bc, v92_data);
            tensorforge::fmacdpp16<4>(v109_acc, v124_bc, v93_data);
            tensorforge::fmacdpp16<5>(v109_acc, v124_bc, v94_data);
            tensorforge::fmacdpp16<6>(v109_acc, v124_bc, v95_data);
            tensorforge::fmacdpp16<7>(v109_acc, v124_bc, v96_data);
            tensorforge::fmacdpp16<8>(v109_acc, v124_bc, v97_data);
            tensorforge::fmacdpp16<9>(v109_acc, v124_bc, v98_data);
            tensorforge::fmacdpp16<10>(v109_acc, v124_bc, v99_data);
            tensorforge::fmacdpp16<11>(v109_acc, v124_bc, v100_data);
            tensorforge::fmacdpp16<12>(v110_acc, v124_bc, v89_data);
            tensorforge::fmacdpp16<13>(v110_acc, v124_bc, v90_data);
            tensorforge::fmacdpp16<14>(v110_acc, v124_bc, v91_data);
            tensorforge::fmacdpp16<15>(v110_acc, v124_bc, v92_data);
            float v125_bc = tensorforge::broadcast<32, 16, 1>(v123_lin);
            tensorforge::fmacdpp16<0>(v110_acc, v125_bc, v93_data);
            tensorforge::fmacdpp16<1>(v110_acc, v125_bc, v94_data);
            tensorforge::fmacdpp16<2>(v110_acc, v125_bc, v95_data);
            tensorforge::fmacdpp16<3>(v110_acc, v125_bc, v96_data);
            tensorforge::fmacdpp16<4>(v110_acc, v125_bc, v97_data);
            tensorforge::fmacdpp16<5>(v110_acc, v125_bc, v98_data);
            tensorforge::fmacdpp16<6>(v110_acc, v125_bc, v99_data);
            tensorforge::fmacdpp16<7>(v110_acc, v125_bc, v100_data);
            tensorforge::fmacdpp16<8>(v111_acc, v125_bc, v89_data);
            tensorforge::fmacdpp16<9>(v111_acc, v125_bc, v90_data);
            tensorforge::fmacdpp16<10>(v111_acc, v125_bc, v91_data);
            tensorforge::fmacdpp16<11>(v111_acc, v125_bc, v92_data);
            tensorforge::fmacdpp16<12>(v111_acc, v125_bc, v93_data);
            tensorforge::fmacdpp16<13>(v111_acc, v125_bc, v94_data);
            tensorforge::fmacdpp16<14>(v111_acc, v125_bc, v95_data);
            tensorforge::fmacdpp16<15>(v111_acc, v125_bc, v96_data);
            float v126_lin = r3[4];
            float v127_bc = tensorforge::broadcast<32, 16, 0>(v126_lin);
            tensorforge::fmacdpp16<0>(v111_acc, v127_bc, v97_data);
            tensorforge::fmacdpp16<1>(v111_acc, v127_bc, v98_data);
            tensorforge::fmacdpp16<2>(v111_acc, v127_bc, v99_data);
            tensorforge::fmacdpp16<3>(v111_acc, v127_bc, v100_data);
            tensorforge::fmacdpp16<4>(v112_acc, v127_bc, v89_data);
            tensorforge::fmacdpp16<5>(v112_acc, v127_bc, v90_data);
            tensorforge::fmacdpp16<6>(v112_acc, v127_bc, v91_data);
            tensorforge::fmacdpp16<7>(v112_acc, v127_bc, v92_data);
            tensorforge::fmacdpp16<8>(v112_acc, v127_bc, v93_data);
            tensorforge::fmacdpp16<9>(v112_acc, v127_bc, v94_data);
            tensorforge::fmacdpp16<10>(v112_acc, v127_bc, v95_data);
            tensorforge::fmacdpp16<11>(v112_acc, v127_bc, v96_data);
            tensorforge::fmacdpp16<12>(v112_acc, v127_bc, v97_data);
            tensorforge::fmacdpp16<13>(v112_acc, v127_bc, v98_data);
            tensorforge::fmacdpp16<14>(v112_acc, v127_bc, v99_data);
            tensorforge::fmacdpp16<15>(v112_acc, v127_bc, v100_data);
            float v128_bc = tensorforge::broadcast<32, 16, 1>(v126_lin);
            tensorforge::fmacdpp16<0>(v113_acc, v128_bc, v89_data);
            tensorforge::fmacdpp16<1>(v113_acc, v128_bc, v90_data);
            tensorforge::fmacdpp16<2>(v113_acc, v128_bc, v91_data);
            tensorforge::fmacdpp16<3>(v113_acc, v128_bc, v92_data);
            tensorforge::fmacdpp16<4>(v113_acc, v128_bc, v93_data);
            tensorforge::fmacdpp16<5>(v113_acc, v128_bc, v94_data);
            tensorforge::fmacdpp16<6>(v113_acc, v128_bc, v95_data);
            tensorforge::fmacdpp16<7>(v113_acc, v128_bc, v96_data);
            tensorforge::fmacdpp16<8>(v113_acc, v128_bc, v97_data);
            tensorforge::fmacdpp16<9>(v113_acc, v128_bc, v98_data);
            tensorforge::fmacdpp16<10>(v113_acc, v128_bc, v99_data);
            tensorforge::fmacdpp16<11>(v113_acc, v128_bc, v100_data);
            ir4[0] = v101_acc;
            ir4[1] = v102_acc;
            ir4[2] = v103_acc;
            ir4[3] = v104_acc;
            ir4[4] = v105_acc;
            ir4[5] = v106_acc;
            ir4[6] = v107_acc;
            ir4[7] = v108_acc;
            ir4[8] = v109_acc;
            ir4[9] = v110_acc;
            ir4[10] = v111_acc;
            ir4[11] = v112_acc;
            ir4[12] = v113_acc;
            #pragma unroll
            for (int32_t v132_n0 = 0; v132_n0 < 1; ++v132_n0) {
              #pragma unroll
              for (int32_t v133_n1 = 0; v133_n1 < 13; ++v133_n1) {
                int32_t v134_a = v132_n0 + v133_n1;
                int32_t v135_a = v132_n0 + v133_n1;
                float v136_data = ir4[v135_a];
                int32_t v137_a = v132_n0 + v133_n1;
                float v139_data = r1[v135_a];
                int32_t v141_a = v132_n0 + v133_n1;
                r4[v135_a] = (v139_data + v136_data);
              }
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          auto& ir5 = r5;
          float v147_data = r4[4];
          float v148_data = ir5[0];
          ir5[0] = (v148_data + v147_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v153_i0 = 0; v153_i0 < 1; ++v153_i0) {
            int32_t v162_lead = v3_lead + (v153_i0 * 32);
            #pragma unroll
            for (int32_t v154_i1 = 0; v154_i1 < 1; ++v154_i1) {
              int32_t v155_a = v153_i0 + v154_i1;
              float v157_data = r5[(v153_i0 + v154_i1)];
              int32_t v165_a = v162_lead + ((v154_i1 + 4) * 32);
              glb_m0[v165_a] = v157_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v170_i0 = 0; v170_i0 < 1; ++v170_i0) {
            int32_t v175_lead = v170_i0 * 32;
            int32_t v176_lead = v3_lead + v175_lead;
            int32_t v183_lead = v3_lead + v175_lead;
            #pragma unroll
            for (int32_t v171_i1 = 0; v171_i1 < 13; ++v171_i1) {
              int32_t v177_a = v171_i1 * 32;
              int32_t v178_a = v176_lead + v177_a;
              float v186_data = glb_m0[(v183_lead + v177_a)];
              int32_t v187_a = v170_i0 + v171_i1;
              r6[v187_a] = v186_data;
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
          float v190_data = r6[0];
          float v191_data = r6[1];
          float v192_data = r6[2];
          float v193_data = r6[3];
          float v194_data = r6[4];
          float v195_data = r6[5];
          float v196_data = r6[6];
          float v197_data = r6[7];
          float v198_data = r6[8];
          float v199_data = r6[9];
          float v200_data = r6[10];
          float v201_data = r6[11];
          float v202_data = r6[12];
          float v203_acc{};
          float v204_acc{};
          float v205_acc{};
          float v206_acc{};
          float v207_acc{};
          float v208_acc{};
          float v209_acc{};
          float v210_acc{};
          float v211_acc{};
          float v212_acc{};
          float v213_acc{};
          float v214_acc{};
          float v215_acc{};
          float v216_lin = r7[0];
          float v217_bc = tensorforge::broadcast<32, 16, 0>(v216_lin);
          tensorforge::fmacdpp16<0>(v203_acc, v217_bc, v190_data);
          tensorforge::fmacdpp16<1>(v203_acc, v217_bc, v191_data);
          tensorforge::fmacdpp16<2>(v203_acc, v217_bc, v192_data);
          tensorforge::fmacdpp16<3>(v203_acc, v217_bc, v193_data);
          tensorforge::fmacdpp16<4>(v203_acc, v217_bc, v194_data);
          tensorforge::fmacdpp16<5>(v203_acc, v217_bc, v195_data);
          tensorforge::fmacdpp16<6>(v203_acc, v217_bc, v196_data);
          tensorforge::fmacdpp16<7>(v203_acc, v217_bc, v197_data);
          tensorforge::fmacdpp16<8>(v203_acc, v217_bc, v198_data);
          tensorforge::fmacdpp16<9>(v203_acc, v217_bc, v199_data);
          tensorforge::fmacdpp16<10>(v203_acc, v217_bc, v200_data);
          tensorforge::fmacdpp16<11>(v203_acc, v217_bc, v201_data);
          tensorforge::fmacdpp16<12>(v203_acc, v217_bc, v202_data);
          tensorforge::fmacdpp16<13>(v204_acc, v217_bc, v190_data);
          tensorforge::fmacdpp16<14>(v204_acc, v217_bc, v191_data);
          tensorforge::fmacdpp16<15>(v204_acc, v217_bc, v192_data);
          float v218_bc = tensorforge::broadcast<32, 16, 1>(v216_lin);
          tensorforge::fmacdpp16<0>(v204_acc, v218_bc, v193_data);
          tensorforge::fmacdpp16<1>(v204_acc, v218_bc, v194_data);
          tensorforge::fmacdpp16<2>(v204_acc, v218_bc, v195_data);
          tensorforge::fmacdpp16<3>(v204_acc, v218_bc, v196_data);
          tensorforge::fmacdpp16<4>(v204_acc, v218_bc, v197_data);
          tensorforge::fmacdpp16<5>(v204_acc, v218_bc, v198_data);
          tensorforge::fmacdpp16<6>(v204_acc, v218_bc, v199_data);
          tensorforge::fmacdpp16<7>(v204_acc, v218_bc, v200_data);
          tensorforge::fmacdpp16<8>(v204_acc, v218_bc, v201_data);
          tensorforge::fmacdpp16<9>(v204_acc, v218_bc, v202_data);
          tensorforge::fmacdpp16<10>(v205_acc, v218_bc, v190_data);
          tensorforge::fmacdpp16<11>(v205_acc, v218_bc, v191_data);
          tensorforge::fmacdpp16<12>(v205_acc, v218_bc, v192_data);
          tensorforge::fmacdpp16<13>(v205_acc, v218_bc, v193_data);
          tensorforge::fmacdpp16<14>(v205_acc, v218_bc, v194_data);
          tensorforge::fmacdpp16<15>(v205_acc, v218_bc, v195_data);
          float v219_lin = r7[1];
          float v220_bc = tensorforge::broadcast<32, 16, 0>(v219_lin);
          tensorforge::fmacdpp16<0>(v205_acc, v220_bc, v196_data);
          tensorforge::fmacdpp16<1>(v205_acc, v220_bc, v197_data);
          tensorforge::fmacdpp16<2>(v205_acc, v220_bc, v198_data);
          tensorforge::fmacdpp16<3>(v205_acc, v220_bc, v199_data);
          tensorforge::fmacdpp16<4>(v205_acc, v220_bc, v200_data);
          tensorforge::fmacdpp16<5>(v205_acc, v220_bc, v201_data);
          tensorforge::fmacdpp16<6>(v205_acc, v220_bc, v202_data);
          tensorforge::fmacdpp16<7>(v206_acc, v220_bc, v190_data);
          tensorforge::fmacdpp16<8>(v206_acc, v220_bc, v191_data);
          tensorforge::fmacdpp16<9>(v206_acc, v220_bc, v192_data);
          tensorforge::fmacdpp16<10>(v206_acc, v220_bc, v193_data);
          tensorforge::fmacdpp16<11>(v206_acc, v220_bc, v194_data);
          tensorforge::fmacdpp16<12>(v206_acc, v220_bc, v195_data);
          tensorforge::fmacdpp16<13>(v206_acc, v220_bc, v196_data);
          tensorforge::fmacdpp16<14>(v206_acc, v220_bc, v197_data);
          tensorforge::fmacdpp16<15>(v206_acc, v220_bc, v198_data);
          float v221_bc = tensorforge::broadcast<32, 16, 1>(v219_lin);
          tensorforge::fmacdpp16<0>(v206_acc, v221_bc, v199_data);
          tensorforge::fmacdpp16<1>(v206_acc, v221_bc, v200_data);
          tensorforge::fmacdpp16<2>(v206_acc, v221_bc, v201_data);
          tensorforge::fmacdpp16<3>(v206_acc, v221_bc, v202_data);
          tensorforge::fmacdpp16<4>(v207_acc, v221_bc, v190_data);
          tensorforge::fmacdpp16<5>(v207_acc, v221_bc, v191_data);
          tensorforge::fmacdpp16<6>(v207_acc, v221_bc, v192_data);
          tensorforge::fmacdpp16<7>(v207_acc, v221_bc, v193_data);
          tensorforge::fmacdpp16<8>(v207_acc, v221_bc, v194_data);
          tensorforge::fmacdpp16<9>(v207_acc, v221_bc, v195_data);
          tensorforge::fmacdpp16<10>(v207_acc, v221_bc, v196_data);
          tensorforge::fmacdpp16<11>(v207_acc, v221_bc, v197_data);
          tensorforge::fmacdpp16<12>(v207_acc, v221_bc, v198_data);
          tensorforge::fmacdpp16<13>(v207_acc, v221_bc, v199_data);
          tensorforge::fmacdpp16<14>(v207_acc, v221_bc, v200_data);
          tensorforge::fmacdpp16<15>(v207_acc, v221_bc, v201_data);
          float v222_lin = r7[2];
          float v223_bc = tensorforge::broadcast<32, 16, 0>(v222_lin);
          tensorforge::fmacdpp16<0>(v207_acc, v223_bc, v202_data);
          tensorforge::fmacdpp16<1>(v208_acc, v223_bc, v190_data);
          tensorforge::fmacdpp16<2>(v208_acc, v223_bc, v191_data);
          tensorforge::fmacdpp16<3>(v208_acc, v223_bc, v192_data);
          tensorforge::fmacdpp16<4>(v208_acc, v223_bc, v193_data);
          tensorforge::fmacdpp16<5>(v208_acc, v223_bc, v194_data);
          tensorforge::fmacdpp16<6>(v208_acc, v223_bc, v195_data);
          tensorforge::fmacdpp16<7>(v208_acc, v223_bc, v196_data);
          tensorforge::fmacdpp16<8>(v208_acc, v223_bc, v197_data);
          tensorforge::fmacdpp16<9>(v208_acc, v223_bc, v198_data);
          tensorforge::fmacdpp16<10>(v208_acc, v223_bc, v199_data);
          tensorforge::fmacdpp16<11>(v208_acc, v223_bc, v200_data);
          tensorforge::fmacdpp16<12>(v208_acc, v223_bc, v201_data);
          tensorforge::fmacdpp16<13>(v208_acc, v223_bc, v202_data);
          tensorforge::fmacdpp16<14>(v209_acc, v223_bc, v190_data);
          tensorforge::fmacdpp16<15>(v209_acc, v223_bc, v191_data);
          float v224_bc = tensorforge::broadcast<32, 16, 1>(v222_lin);
          tensorforge::fmacdpp16<0>(v209_acc, v224_bc, v192_data);
          tensorforge::fmacdpp16<1>(v209_acc, v224_bc, v193_data);
          tensorforge::fmacdpp16<2>(v209_acc, v224_bc, v194_data);
          tensorforge::fmacdpp16<3>(v209_acc, v224_bc, v195_data);
          tensorforge::fmacdpp16<4>(v209_acc, v224_bc, v196_data);
          tensorforge::fmacdpp16<5>(v209_acc, v224_bc, v197_data);
          tensorforge::fmacdpp16<6>(v209_acc, v224_bc, v198_data);
          tensorforge::fmacdpp16<7>(v209_acc, v224_bc, v199_data);
          tensorforge::fmacdpp16<8>(v209_acc, v224_bc, v200_data);
          tensorforge::fmacdpp16<9>(v209_acc, v224_bc, v201_data);
          tensorforge::fmacdpp16<10>(v209_acc, v224_bc, v202_data);
          tensorforge::fmacdpp16<11>(v210_acc, v224_bc, v190_data);
          tensorforge::fmacdpp16<12>(v210_acc, v224_bc, v191_data);
          tensorforge::fmacdpp16<13>(v210_acc, v224_bc, v192_data);
          tensorforge::fmacdpp16<14>(v210_acc, v224_bc, v193_data);
          tensorforge::fmacdpp16<15>(v210_acc, v224_bc, v194_data);
          float v225_lin = r7[3];
          float v226_bc = tensorforge::broadcast<32, 16, 0>(v225_lin);
          tensorforge::fmacdpp16<0>(v210_acc, v226_bc, v195_data);
          tensorforge::fmacdpp16<1>(v210_acc, v226_bc, v196_data);
          tensorforge::fmacdpp16<2>(v210_acc, v226_bc, v197_data);
          tensorforge::fmacdpp16<3>(v210_acc, v226_bc, v198_data);
          tensorforge::fmacdpp16<4>(v210_acc, v226_bc, v199_data);
          tensorforge::fmacdpp16<5>(v210_acc, v226_bc, v200_data);
          tensorforge::fmacdpp16<6>(v210_acc, v226_bc, v201_data);
          tensorforge::fmacdpp16<7>(v210_acc, v226_bc, v202_data);
          tensorforge::fmacdpp16<8>(v211_acc, v226_bc, v190_data);
          tensorforge::fmacdpp16<9>(v211_acc, v226_bc, v191_data);
          tensorforge::fmacdpp16<10>(v211_acc, v226_bc, v192_data);
          tensorforge::fmacdpp16<11>(v211_acc, v226_bc, v193_data);
          tensorforge::fmacdpp16<12>(v211_acc, v226_bc, v194_data);
          tensorforge::fmacdpp16<13>(v211_acc, v226_bc, v195_data);
          tensorforge::fmacdpp16<14>(v211_acc, v226_bc, v196_data);
          tensorforge::fmacdpp16<15>(v211_acc, v226_bc, v197_data);
          float v227_bc = tensorforge::broadcast<32, 16, 1>(v225_lin);
          tensorforge::fmacdpp16<0>(v211_acc, v227_bc, v198_data);
          tensorforge::fmacdpp16<1>(v211_acc, v227_bc, v199_data);
          tensorforge::fmacdpp16<2>(v211_acc, v227_bc, v200_data);
          tensorforge::fmacdpp16<3>(v211_acc, v227_bc, v201_data);
          tensorforge::fmacdpp16<4>(v211_acc, v227_bc, v202_data);
          tensorforge::fmacdpp16<5>(v212_acc, v227_bc, v190_data);
          tensorforge::fmacdpp16<6>(v212_acc, v227_bc, v191_data);
          tensorforge::fmacdpp16<7>(v212_acc, v227_bc, v192_data);
          tensorforge::fmacdpp16<8>(v212_acc, v227_bc, v193_data);
          tensorforge::fmacdpp16<9>(v212_acc, v227_bc, v194_data);
          tensorforge::fmacdpp16<10>(v212_acc, v227_bc, v195_data);
          tensorforge::fmacdpp16<11>(v212_acc, v227_bc, v196_data);
          tensorforge::fmacdpp16<12>(v212_acc, v227_bc, v197_data);
          tensorforge::fmacdpp16<13>(v212_acc, v227_bc, v198_data);
          tensorforge::fmacdpp16<14>(v212_acc, v227_bc, v199_data);
          tensorforge::fmacdpp16<15>(v212_acc, v227_bc, v200_data);
          float v228_lin = r7[4];
          float v229_bc = tensorforge::broadcast<32, 16, 0>(v228_lin);
          tensorforge::fmacdpp16<0>(v212_acc, v229_bc, v201_data);
          tensorforge::fmacdpp16<1>(v212_acc, v229_bc, v202_data);
          tensorforge::fmacdpp16<2>(v213_acc, v229_bc, v190_data);
          tensorforge::fmacdpp16<3>(v213_acc, v229_bc, v191_data);
          tensorforge::fmacdpp16<4>(v213_acc, v229_bc, v192_data);
          tensorforge::fmacdpp16<5>(v213_acc, v229_bc, v193_data);
          tensorforge::fmacdpp16<6>(v213_acc, v229_bc, v194_data);
          tensorforge::fmacdpp16<7>(v213_acc, v229_bc, v195_data);
          tensorforge::fmacdpp16<8>(v213_acc, v229_bc, v196_data);
          tensorforge::fmacdpp16<9>(v213_acc, v229_bc, v197_data);
          tensorforge::fmacdpp16<10>(v213_acc, v229_bc, v198_data);
          tensorforge::fmacdpp16<11>(v213_acc, v229_bc, v199_data);
          tensorforge::fmacdpp16<12>(v213_acc, v229_bc, v200_data);
          tensorforge::fmacdpp16<13>(v213_acc, v229_bc, v201_data);
          tensorforge::fmacdpp16<14>(v213_acc, v229_bc, v202_data);
          tensorforge::fmacdpp16<15>(v214_acc, v229_bc, v190_data);
          float v230_bc = tensorforge::broadcast<32, 16, 1>(v228_lin);
          tensorforge::fmacdpp16<0>(v214_acc, v230_bc, v191_data);
          tensorforge::fmacdpp16<1>(v214_acc, v230_bc, v192_data);
          tensorforge::fmacdpp16<2>(v214_acc, v230_bc, v193_data);
          tensorforge::fmacdpp16<3>(v214_acc, v230_bc, v194_data);
          tensorforge::fmacdpp16<4>(v214_acc, v230_bc, v195_data);
          tensorforge::fmacdpp16<5>(v214_acc, v230_bc, v196_data);
          tensorforge::fmacdpp16<6>(v214_acc, v230_bc, v197_data);
          tensorforge::fmacdpp16<7>(v214_acc, v230_bc, v198_data);
          tensorforge::fmacdpp16<8>(v214_acc, v230_bc, v199_data);
          tensorforge::fmacdpp16<9>(v214_acc, v230_bc, v200_data);
          tensorforge::fmacdpp16<10>(v214_acc, v230_bc, v201_data);
          tensorforge::fmacdpp16<11>(v214_acc, v230_bc, v202_data);
          tensorforge::fmacdpp16<12>(v215_acc, v230_bc, v190_data);
          tensorforge::fmacdpp16<13>(v215_acc, v230_bc, v191_data);
          tensorforge::fmacdpp16<14>(v215_acc, v230_bc, v192_data);
          tensorforge::fmacdpp16<15>(v215_acc, v230_bc, v193_data);
          float v231_lin = r7[5];
          float v232_bc = tensorforge::broadcast<32, 16, 0>(v231_lin);
          tensorforge::fmacdpp16<0>(v215_acc, v232_bc, v194_data);
          tensorforge::fmacdpp16<1>(v215_acc, v232_bc, v195_data);
          tensorforge::fmacdpp16<2>(v215_acc, v232_bc, v196_data);
          tensorforge::fmacdpp16<3>(v215_acc, v232_bc, v197_data);
          tensorforge::fmacdpp16<4>(v215_acc, v232_bc, v198_data);
          tensorforge::fmacdpp16<5>(v215_acc, v232_bc, v199_data);
          tensorforge::fmacdpp16<6>(v215_acc, v232_bc, v200_data);
          tensorforge::fmacdpp16<7>(v215_acc, v232_bc, v201_data);
          tensorforge::fmacdpp16<8>(v215_acc, v232_bc, v202_data);
          ir8[0] = v203_acc;
          ir8[1] = v204_acc;
          ir8[2] = v205_acc;
          ir8[3] = v206_acc;
          ir8[4] = v207_acc;
          ir8[5] = v208_acc;
          ir8[6] = v209_acc;
          ir8[7] = v210_acc;
          ir8[8] = v211_acc;
          ir8[9] = v212_acc;
          ir8[10] = v213_acc;
          ir8[11] = v214_acc;
          ir8[12] = v215_acc;
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v236_i0 = 0; v236_i0 < 1; ++v236_i0) {
            int32_t v245_lead = v3_lead + (v236_i0 * 32);
            #pragma unroll
            for (int32_t v237_i1 = 0; v237_i1 < 13; ++v237_i1) {
              int32_t v238_a = v236_i0 + v237_i1;
              float v240_data = r8[(v236_i0 + v237_i1)];
              int32_t v247_a = v245_lead + (v237_i1 * 32);
              glb_m3[v247_a] = v240_data;
            }
          }
          ;
        }
      }
    }
  }
}

