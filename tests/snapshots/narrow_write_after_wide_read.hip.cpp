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
          // r3 = load{g>r}(glb_m2);
          float v88_lin = glb_m2[0 + threadIdx.x * 1];
          r3[0] = v88_lin;
          float v89_lin = glb_m2[32 + threadIdx.x * 1];
          r3[1] = v89_lin;
          float v90_lin = glb_m2[64 + threadIdx.x * 1];
          r3[2] = v90_lin;
          float v91_lin = glb_m2[96 + threadIdx.x * 1];
          r3[3] = v91_lin;
          float v92_lin = glb_m2[128 + threadIdx.x * 1];
          r3[4] = v92_lin;
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          {
            // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 32), (0, 13)] [(0, 12)]
            float ir4[13]{};
            float v94_data = r2[0];
            float v95_data = r2[1];
            float v96_data = r2[2];
            float v97_data = r2[3];
            float v98_data = r2[4];
            float v99_data = r2[5];
            float v100_data = r2[6];
            float v101_data = r2[7];
            float v102_data = r2[8];
            float v103_data = r2[9];
            float v104_data = r2[10];
            float v105_data = r2[11];
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
            float v118_acc{};
            float v119_lin = r3[0];
            float v120_bc = tensorforge::broadcast<32, 16, 0>(v119_lin);
            tensorforge::fmacdpp16<0>(v106_acc, v120_bc, v94_data);
            tensorforge::fmacdpp16<1>(v106_acc, v120_bc, v95_data);
            tensorforge::fmacdpp16<2>(v106_acc, v120_bc, v96_data);
            tensorforge::fmacdpp16<3>(v106_acc, v120_bc, v97_data);
            tensorforge::fmacdpp16<4>(v106_acc, v120_bc, v98_data);
            tensorforge::fmacdpp16<5>(v106_acc, v120_bc, v99_data);
            tensorforge::fmacdpp16<6>(v106_acc, v120_bc, v100_data);
            tensorforge::fmacdpp16<7>(v106_acc, v120_bc, v101_data);
            tensorforge::fmacdpp16<8>(v106_acc, v120_bc, v102_data);
            tensorforge::fmacdpp16<9>(v106_acc, v120_bc, v103_data);
            tensorforge::fmacdpp16<10>(v106_acc, v120_bc, v104_data);
            tensorforge::fmacdpp16<11>(v106_acc, v120_bc, v105_data);
            tensorforge::fmacdpp16<12>(v107_acc, v120_bc, v94_data);
            tensorforge::fmacdpp16<13>(v107_acc, v120_bc, v95_data);
            tensorforge::fmacdpp16<14>(v107_acc, v120_bc, v96_data);
            tensorforge::fmacdpp16<15>(v107_acc, v120_bc, v97_data);
            float v121_bc = tensorforge::broadcast<32, 16, 1>(v119_lin);
            tensorforge::fmacdpp16<0>(v107_acc, v121_bc, v98_data);
            tensorforge::fmacdpp16<1>(v107_acc, v121_bc, v99_data);
            tensorforge::fmacdpp16<2>(v107_acc, v121_bc, v100_data);
            tensorforge::fmacdpp16<3>(v107_acc, v121_bc, v101_data);
            tensorforge::fmacdpp16<4>(v107_acc, v121_bc, v102_data);
            tensorforge::fmacdpp16<5>(v107_acc, v121_bc, v103_data);
            tensorforge::fmacdpp16<6>(v107_acc, v121_bc, v104_data);
            tensorforge::fmacdpp16<7>(v107_acc, v121_bc, v105_data);
            tensorforge::fmacdpp16<8>(v108_acc, v121_bc, v94_data);
            tensorforge::fmacdpp16<9>(v108_acc, v121_bc, v95_data);
            tensorforge::fmacdpp16<10>(v108_acc, v121_bc, v96_data);
            tensorforge::fmacdpp16<11>(v108_acc, v121_bc, v97_data);
            tensorforge::fmacdpp16<12>(v108_acc, v121_bc, v98_data);
            tensorforge::fmacdpp16<13>(v108_acc, v121_bc, v99_data);
            tensorforge::fmacdpp16<14>(v108_acc, v121_bc, v100_data);
            tensorforge::fmacdpp16<15>(v108_acc, v121_bc, v101_data);
            float v122_lin = r3[1];
            float v123_bc = tensorforge::broadcast<32, 16, 0>(v122_lin);
            tensorforge::fmacdpp16<0>(v108_acc, v123_bc, v102_data);
            tensorforge::fmacdpp16<1>(v108_acc, v123_bc, v103_data);
            tensorforge::fmacdpp16<2>(v108_acc, v123_bc, v104_data);
            tensorforge::fmacdpp16<3>(v108_acc, v123_bc, v105_data);
            tensorforge::fmacdpp16<4>(v109_acc, v123_bc, v94_data);
            tensorforge::fmacdpp16<5>(v109_acc, v123_bc, v95_data);
            tensorforge::fmacdpp16<6>(v109_acc, v123_bc, v96_data);
            tensorforge::fmacdpp16<7>(v109_acc, v123_bc, v97_data);
            tensorforge::fmacdpp16<8>(v109_acc, v123_bc, v98_data);
            tensorforge::fmacdpp16<9>(v109_acc, v123_bc, v99_data);
            tensorforge::fmacdpp16<10>(v109_acc, v123_bc, v100_data);
            tensorforge::fmacdpp16<11>(v109_acc, v123_bc, v101_data);
            tensorforge::fmacdpp16<12>(v109_acc, v123_bc, v102_data);
            tensorforge::fmacdpp16<13>(v109_acc, v123_bc, v103_data);
            tensorforge::fmacdpp16<14>(v109_acc, v123_bc, v104_data);
            tensorforge::fmacdpp16<15>(v109_acc, v123_bc, v105_data);
            float v124_bc = tensorforge::broadcast<32, 16, 1>(v122_lin);
            tensorforge::fmacdpp16<0>(v110_acc, v124_bc, v94_data);
            tensorforge::fmacdpp16<1>(v110_acc, v124_bc, v95_data);
            tensorforge::fmacdpp16<2>(v110_acc, v124_bc, v96_data);
            tensorforge::fmacdpp16<3>(v110_acc, v124_bc, v97_data);
            tensorforge::fmacdpp16<4>(v110_acc, v124_bc, v98_data);
            tensorforge::fmacdpp16<5>(v110_acc, v124_bc, v99_data);
            tensorforge::fmacdpp16<6>(v110_acc, v124_bc, v100_data);
            tensorforge::fmacdpp16<7>(v110_acc, v124_bc, v101_data);
            tensorforge::fmacdpp16<8>(v110_acc, v124_bc, v102_data);
            tensorforge::fmacdpp16<9>(v110_acc, v124_bc, v103_data);
            tensorforge::fmacdpp16<10>(v110_acc, v124_bc, v104_data);
            tensorforge::fmacdpp16<11>(v110_acc, v124_bc, v105_data);
            tensorforge::fmacdpp16<12>(v111_acc, v124_bc, v94_data);
            tensorforge::fmacdpp16<13>(v111_acc, v124_bc, v95_data);
            tensorforge::fmacdpp16<14>(v111_acc, v124_bc, v96_data);
            tensorforge::fmacdpp16<15>(v111_acc, v124_bc, v97_data);
            float v125_lin = r3[2];
            float v126_bc = tensorforge::broadcast<32, 16, 0>(v125_lin);
            tensorforge::fmacdpp16<0>(v111_acc, v126_bc, v98_data);
            tensorforge::fmacdpp16<1>(v111_acc, v126_bc, v99_data);
            tensorforge::fmacdpp16<2>(v111_acc, v126_bc, v100_data);
            tensorforge::fmacdpp16<3>(v111_acc, v126_bc, v101_data);
            tensorforge::fmacdpp16<4>(v111_acc, v126_bc, v102_data);
            tensorforge::fmacdpp16<5>(v111_acc, v126_bc, v103_data);
            tensorforge::fmacdpp16<6>(v111_acc, v126_bc, v104_data);
            tensorforge::fmacdpp16<7>(v111_acc, v126_bc, v105_data);
            tensorforge::fmacdpp16<8>(v112_acc, v126_bc, v94_data);
            tensorforge::fmacdpp16<9>(v112_acc, v126_bc, v95_data);
            tensorforge::fmacdpp16<10>(v112_acc, v126_bc, v96_data);
            tensorforge::fmacdpp16<11>(v112_acc, v126_bc, v97_data);
            tensorforge::fmacdpp16<12>(v112_acc, v126_bc, v98_data);
            tensorforge::fmacdpp16<13>(v112_acc, v126_bc, v99_data);
            tensorforge::fmacdpp16<14>(v112_acc, v126_bc, v100_data);
            tensorforge::fmacdpp16<15>(v112_acc, v126_bc, v101_data);
            float v127_bc = tensorforge::broadcast<32, 16, 1>(v125_lin);
            tensorforge::fmacdpp16<0>(v112_acc, v127_bc, v102_data);
            tensorforge::fmacdpp16<1>(v112_acc, v127_bc, v103_data);
            tensorforge::fmacdpp16<2>(v112_acc, v127_bc, v104_data);
            tensorforge::fmacdpp16<3>(v112_acc, v127_bc, v105_data);
            tensorforge::fmacdpp16<4>(v113_acc, v127_bc, v94_data);
            tensorforge::fmacdpp16<5>(v113_acc, v127_bc, v95_data);
            tensorforge::fmacdpp16<6>(v113_acc, v127_bc, v96_data);
            tensorforge::fmacdpp16<7>(v113_acc, v127_bc, v97_data);
            tensorforge::fmacdpp16<8>(v113_acc, v127_bc, v98_data);
            tensorforge::fmacdpp16<9>(v113_acc, v127_bc, v99_data);
            tensorforge::fmacdpp16<10>(v113_acc, v127_bc, v100_data);
            tensorforge::fmacdpp16<11>(v113_acc, v127_bc, v101_data);
            tensorforge::fmacdpp16<12>(v113_acc, v127_bc, v102_data);
            tensorforge::fmacdpp16<13>(v113_acc, v127_bc, v103_data);
            tensorforge::fmacdpp16<14>(v113_acc, v127_bc, v104_data);
            tensorforge::fmacdpp16<15>(v113_acc, v127_bc, v105_data);
            float v128_lin = r3[3];
            float v129_bc = tensorforge::broadcast<32, 16, 0>(v128_lin);
            tensorforge::fmacdpp16<0>(v114_acc, v129_bc, v94_data);
            tensorforge::fmacdpp16<1>(v114_acc, v129_bc, v95_data);
            tensorforge::fmacdpp16<2>(v114_acc, v129_bc, v96_data);
            tensorforge::fmacdpp16<3>(v114_acc, v129_bc, v97_data);
            tensorforge::fmacdpp16<4>(v114_acc, v129_bc, v98_data);
            tensorforge::fmacdpp16<5>(v114_acc, v129_bc, v99_data);
            tensorforge::fmacdpp16<6>(v114_acc, v129_bc, v100_data);
            tensorforge::fmacdpp16<7>(v114_acc, v129_bc, v101_data);
            tensorforge::fmacdpp16<8>(v114_acc, v129_bc, v102_data);
            tensorforge::fmacdpp16<9>(v114_acc, v129_bc, v103_data);
            tensorforge::fmacdpp16<10>(v114_acc, v129_bc, v104_data);
            tensorforge::fmacdpp16<11>(v114_acc, v129_bc, v105_data);
            tensorforge::fmacdpp16<12>(v115_acc, v129_bc, v94_data);
            tensorforge::fmacdpp16<13>(v115_acc, v129_bc, v95_data);
            tensorforge::fmacdpp16<14>(v115_acc, v129_bc, v96_data);
            tensorforge::fmacdpp16<15>(v115_acc, v129_bc, v97_data);
            float v130_bc = tensorforge::broadcast<32, 16, 1>(v128_lin);
            tensorforge::fmacdpp16<0>(v115_acc, v130_bc, v98_data);
            tensorforge::fmacdpp16<1>(v115_acc, v130_bc, v99_data);
            tensorforge::fmacdpp16<2>(v115_acc, v130_bc, v100_data);
            tensorforge::fmacdpp16<3>(v115_acc, v130_bc, v101_data);
            tensorforge::fmacdpp16<4>(v115_acc, v130_bc, v102_data);
            tensorforge::fmacdpp16<5>(v115_acc, v130_bc, v103_data);
            tensorforge::fmacdpp16<6>(v115_acc, v130_bc, v104_data);
            tensorforge::fmacdpp16<7>(v115_acc, v130_bc, v105_data);
            tensorforge::fmacdpp16<8>(v116_acc, v130_bc, v94_data);
            tensorforge::fmacdpp16<9>(v116_acc, v130_bc, v95_data);
            tensorforge::fmacdpp16<10>(v116_acc, v130_bc, v96_data);
            tensorforge::fmacdpp16<11>(v116_acc, v130_bc, v97_data);
            tensorforge::fmacdpp16<12>(v116_acc, v130_bc, v98_data);
            tensorforge::fmacdpp16<13>(v116_acc, v130_bc, v99_data);
            tensorforge::fmacdpp16<14>(v116_acc, v130_bc, v100_data);
            tensorforge::fmacdpp16<15>(v116_acc, v130_bc, v101_data);
            float v131_lin = r3[4];
            float v132_bc = tensorforge::broadcast<32, 16, 0>(v131_lin);
            tensorforge::fmacdpp16<0>(v116_acc, v132_bc, v102_data);
            tensorforge::fmacdpp16<1>(v116_acc, v132_bc, v103_data);
            tensorforge::fmacdpp16<2>(v116_acc, v132_bc, v104_data);
            tensorforge::fmacdpp16<3>(v116_acc, v132_bc, v105_data);
            tensorforge::fmacdpp16<4>(v117_acc, v132_bc, v94_data);
            tensorforge::fmacdpp16<5>(v117_acc, v132_bc, v95_data);
            tensorforge::fmacdpp16<6>(v117_acc, v132_bc, v96_data);
            tensorforge::fmacdpp16<7>(v117_acc, v132_bc, v97_data);
            tensorforge::fmacdpp16<8>(v117_acc, v132_bc, v98_data);
            tensorforge::fmacdpp16<9>(v117_acc, v132_bc, v99_data);
            tensorforge::fmacdpp16<10>(v117_acc, v132_bc, v100_data);
            tensorforge::fmacdpp16<11>(v117_acc, v132_bc, v101_data);
            tensorforge::fmacdpp16<12>(v117_acc, v132_bc, v102_data);
            tensorforge::fmacdpp16<13>(v117_acc, v132_bc, v103_data);
            tensorforge::fmacdpp16<14>(v117_acc, v132_bc, v104_data);
            tensorforge::fmacdpp16<15>(v117_acc, v132_bc, v105_data);
            float v133_bc = tensorforge::broadcast<32, 16, 1>(v131_lin);
            tensorforge::fmacdpp16<0>(v118_acc, v133_bc, v94_data);
            tensorforge::fmacdpp16<1>(v118_acc, v133_bc, v95_data);
            tensorforge::fmacdpp16<2>(v118_acc, v133_bc, v96_data);
            tensorforge::fmacdpp16<3>(v118_acc, v133_bc, v97_data);
            tensorforge::fmacdpp16<4>(v118_acc, v133_bc, v98_data);
            tensorforge::fmacdpp16<5>(v118_acc, v133_bc, v99_data);
            tensorforge::fmacdpp16<6>(v118_acc, v133_bc, v100_data);
            tensorforge::fmacdpp16<7>(v118_acc, v133_bc, v101_data);
            tensorforge::fmacdpp16<8>(v118_acc, v133_bc, v102_data);
            tensorforge::fmacdpp16<9>(v118_acc, v133_bc, v103_data);
            tensorforge::fmacdpp16<10>(v118_acc, v133_bc, v104_data);
            tensorforge::fmacdpp16<11>(v118_acc, v133_bc, v105_data);
            ir4[0] = v106_acc;
            ir4[1] = v107_acc;
            ir4[2] = v108_acc;
            ir4[3] = v109_acc;
            ir4[4] = v110_acc;
            ir4[5] = v111_acc;
            ir4[6] = v112_acc;
            ir4[7] = v113_acc;
            ir4[8] = v114_acc;
            ir4[9] = v115_acc;
            ir4[10] = v116_acc;
            ir4[11] = v117_acc;
            ir4[12] = v118_acc;
            #pragma unroll
            for (int32_t v137_n0 = 0; v137_n0 < 1; ++v137_n0) {
              #pragma unroll
              for (int32_t v138_n1 = 0; v138_n1 < 13; ++v138_n1) {
                int32_t v139_a = v137_n0 + v138_n1;
                int32_t v140_a = v137_n0 + v138_n1;
                float v141_data = ir4[v140_a];
                int32_t v142_a = v137_n0 + v138_n1;
                float v144_data = r1[v140_a];
                int32_t v146_a = v137_n0 + v138_n1;
                r4[v140_a] = (v144_data + v141_data);
              }
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          auto& ir5 = r5;
          float v152_data = r4[4];
          float v153_data = ir5[0];
          ir5[0] = (v153_data + v152_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v158_i0 = 0; v158_i0 < 1; ++v158_i0) {
            int32_t v167_lead = v3_lead + (v158_i0 * 32);
            #pragma unroll
            for (int32_t v159_i1 = 0; v159_i1 < 1; ++v159_i1) {
              int32_t v160_a = v158_i0 + v159_i1;
              float v162_data = r5[(v158_i0 + v159_i1)];
              int32_t v170_a = v167_lead + ((v159_i1 + 4) * 32);
              glb_m0[v170_a] = v162_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v175_i0 = 0; v175_i0 < 1; ++v175_i0) {
            int32_t v180_lead = v175_i0 * 32;
            int32_t v181_lead = v3_lead + v180_lead;
            int32_t v188_lead = v3_lead + v180_lead;
            #pragma unroll
            for (int32_t v176_i1 = 0; v176_i1 < 13; ++v176_i1) {
              int32_t v182_a = v176_i1 * 32;
              int32_t v183_a = v181_lead + v182_a;
              float v191_data = glb_m0[(v188_lead + v182_a)];
              int32_t v192_a = v175_i0 + v176_i1;
              r6[v192_a] = v191_data;
            }
          }
          float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          float v194_lin = glb_m4[0 + threadIdx.x * 1];
          r7[0] = v194_lin;
          float v195_lin = glb_m4[32 + threadIdx.x * 1];
          r7[1] = v195_lin;
          float v196_lin = glb_m4[64 + threadIdx.x * 1];
          r7[2] = v196_lin;
          float v197_lin = glb_m4[96 + threadIdx.x * 1];
          r7[3] = v197_lin;
          float v198_lin = glb_m4[128 + threadIdx.x * 1];
          r7[4] = v198_lin;
          float v199_lin = glb_m4[160 + threadIdx.x * 1];
          r7[5] = v199_lin;
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          auto& ir8 = r8;
          float v201_data = r6[0];
          float v202_data = r6[1];
          float v203_data = r6[2];
          float v204_data = r6[3];
          float v205_data = r6[4];
          float v206_data = r6[5];
          float v207_data = r6[6];
          float v208_data = r6[7];
          float v209_data = r6[8];
          float v210_data = r6[9];
          float v211_data = r6[10];
          float v212_data = r6[11];
          float v213_data = r6[12];
          float v214_acc{};
          float v215_acc{};
          float v216_acc{};
          float v217_acc{};
          float v218_acc{};
          float v219_acc{};
          float v220_acc{};
          float v221_acc{};
          float v222_acc{};
          float v223_acc{};
          float v224_acc{};
          float v225_acc{};
          float v226_acc{};
          float v227_lin = r7[0];
          float v228_bc = tensorforge::broadcast<32, 16, 0>(v227_lin);
          tensorforge::fmacdpp16<0>(v214_acc, v228_bc, v201_data);
          tensorforge::fmacdpp16<1>(v214_acc, v228_bc, v202_data);
          tensorforge::fmacdpp16<2>(v214_acc, v228_bc, v203_data);
          tensorforge::fmacdpp16<3>(v214_acc, v228_bc, v204_data);
          tensorforge::fmacdpp16<4>(v214_acc, v228_bc, v205_data);
          tensorforge::fmacdpp16<5>(v214_acc, v228_bc, v206_data);
          tensorforge::fmacdpp16<6>(v214_acc, v228_bc, v207_data);
          tensorforge::fmacdpp16<7>(v214_acc, v228_bc, v208_data);
          tensorforge::fmacdpp16<8>(v214_acc, v228_bc, v209_data);
          tensorforge::fmacdpp16<9>(v214_acc, v228_bc, v210_data);
          tensorforge::fmacdpp16<10>(v214_acc, v228_bc, v211_data);
          tensorforge::fmacdpp16<11>(v214_acc, v228_bc, v212_data);
          tensorforge::fmacdpp16<12>(v214_acc, v228_bc, v213_data);
          tensorforge::fmacdpp16<13>(v215_acc, v228_bc, v201_data);
          tensorforge::fmacdpp16<14>(v215_acc, v228_bc, v202_data);
          tensorforge::fmacdpp16<15>(v215_acc, v228_bc, v203_data);
          float v229_bc = tensorforge::broadcast<32, 16, 1>(v227_lin);
          tensorforge::fmacdpp16<0>(v215_acc, v229_bc, v204_data);
          tensorforge::fmacdpp16<1>(v215_acc, v229_bc, v205_data);
          tensorforge::fmacdpp16<2>(v215_acc, v229_bc, v206_data);
          tensorforge::fmacdpp16<3>(v215_acc, v229_bc, v207_data);
          tensorforge::fmacdpp16<4>(v215_acc, v229_bc, v208_data);
          tensorforge::fmacdpp16<5>(v215_acc, v229_bc, v209_data);
          tensorforge::fmacdpp16<6>(v215_acc, v229_bc, v210_data);
          tensorforge::fmacdpp16<7>(v215_acc, v229_bc, v211_data);
          tensorforge::fmacdpp16<8>(v215_acc, v229_bc, v212_data);
          tensorforge::fmacdpp16<9>(v215_acc, v229_bc, v213_data);
          tensorforge::fmacdpp16<10>(v216_acc, v229_bc, v201_data);
          tensorforge::fmacdpp16<11>(v216_acc, v229_bc, v202_data);
          tensorforge::fmacdpp16<12>(v216_acc, v229_bc, v203_data);
          tensorforge::fmacdpp16<13>(v216_acc, v229_bc, v204_data);
          tensorforge::fmacdpp16<14>(v216_acc, v229_bc, v205_data);
          tensorforge::fmacdpp16<15>(v216_acc, v229_bc, v206_data);
          float v230_lin = r7[1];
          float v231_bc = tensorforge::broadcast<32, 16, 0>(v230_lin);
          tensorforge::fmacdpp16<0>(v216_acc, v231_bc, v207_data);
          tensorforge::fmacdpp16<1>(v216_acc, v231_bc, v208_data);
          tensorforge::fmacdpp16<2>(v216_acc, v231_bc, v209_data);
          tensorforge::fmacdpp16<3>(v216_acc, v231_bc, v210_data);
          tensorforge::fmacdpp16<4>(v216_acc, v231_bc, v211_data);
          tensorforge::fmacdpp16<5>(v216_acc, v231_bc, v212_data);
          tensorforge::fmacdpp16<6>(v216_acc, v231_bc, v213_data);
          tensorforge::fmacdpp16<7>(v217_acc, v231_bc, v201_data);
          tensorforge::fmacdpp16<8>(v217_acc, v231_bc, v202_data);
          tensorforge::fmacdpp16<9>(v217_acc, v231_bc, v203_data);
          tensorforge::fmacdpp16<10>(v217_acc, v231_bc, v204_data);
          tensorforge::fmacdpp16<11>(v217_acc, v231_bc, v205_data);
          tensorforge::fmacdpp16<12>(v217_acc, v231_bc, v206_data);
          tensorforge::fmacdpp16<13>(v217_acc, v231_bc, v207_data);
          tensorforge::fmacdpp16<14>(v217_acc, v231_bc, v208_data);
          tensorforge::fmacdpp16<15>(v217_acc, v231_bc, v209_data);
          float v232_bc = tensorforge::broadcast<32, 16, 1>(v230_lin);
          tensorforge::fmacdpp16<0>(v217_acc, v232_bc, v210_data);
          tensorforge::fmacdpp16<1>(v217_acc, v232_bc, v211_data);
          tensorforge::fmacdpp16<2>(v217_acc, v232_bc, v212_data);
          tensorforge::fmacdpp16<3>(v217_acc, v232_bc, v213_data);
          tensorforge::fmacdpp16<4>(v218_acc, v232_bc, v201_data);
          tensorforge::fmacdpp16<5>(v218_acc, v232_bc, v202_data);
          tensorforge::fmacdpp16<6>(v218_acc, v232_bc, v203_data);
          tensorforge::fmacdpp16<7>(v218_acc, v232_bc, v204_data);
          tensorforge::fmacdpp16<8>(v218_acc, v232_bc, v205_data);
          tensorforge::fmacdpp16<9>(v218_acc, v232_bc, v206_data);
          tensorforge::fmacdpp16<10>(v218_acc, v232_bc, v207_data);
          tensorforge::fmacdpp16<11>(v218_acc, v232_bc, v208_data);
          tensorforge::fmacdpp16<12>(v218_acc, v232_bc, v209_data);
          tensorforge::fmacdpp16<13>(v218_acc, v232_bc, v210_data);
          tensorforge::fmacdpp16<14>(v218_acc, v232_bc, v211_data);
          tensorforge::fmacdpp16<15>(v218_acc, v232_bc, v212_data);
          float v233_lin = r7[2];
          float v234_bc = tensorforge::broadcast<32, 16, 0>(v233_lin);
          tensorforge::fmacdpp16<0>(v218_acc, v234_bc, v213_data);
          tensorforge::fmacdpp16<1>(v219_acc, v234_bc, v201_data);
          tensorforge::fmacdpp16<2>(v219_acc, v234_bc, v202_data);
          tensorforge::fmacdpp16<3>(v219_acc, v234_bc, v203_data);
          tensorforge::fmacdpp16<4>(v219_acc, v234_bc, v204_data);
          tensorforge::fmacdpp16<5>(v219_acc, v234_bc, v205_data);
          tensorforge::fmacdpp16<6>(v219_acc, v234_bc, v206_data);
          tensorforge::fmacdpp16<7>(v219_acc, v234_bc, v207_data);
          tensorforge::fmacdpp16<8>(v219_acc, v234_bc, v208_data);
          tensorforge::fmacdpp16<9>(v219_acc, v234_bc, v209_data);
          tensorforge::fmacdpp16<10>(v219_acc, v234_bc, v210_data);
          tensorforge::fmacdpp16<11>(v219_acc, v234_bc, v211_data);
          tensorforge::fmacdpp16<12>(v219_acc, v234_bc, v212_data);
          tensorforge::fmacdpp16<13>(v219_acc, v234_bc, v213_data);
          tensorforge::fmacdpp16<14>(v220_acc, v234_bc, v201_data);
          tensorforge::fmacdpp16<15>(v220_acc, v234_bc, v202_data);
          float v235_bc = tensorforge::broadcast<32, 16, 1>(v233_lin);
          tensorforge::fmacdpp16<0>(v220_acc, v235_bc, v203_data);
          tensorforge::fmacdpp16<1>(v220_acc, v235_bc, v204_data);
          tensorforge::fmacdpp16<2>(v220_acc, v235_bc, v205_data);
          tensorforge::fmacdpp16<3>(v220_acc, v235_bc, v206_data);
          tensorforge::fmacdpp16<4>(v220_acc, v235_bc, v207_data);
          tensorforge::fmacdpp16<5>(v220_acc, v235_bc, v208_data);
          tensorforge::fmacdpp16<6>(v220_acc, v235_bc, v209_data);
          tensorforge::fmacdpp16<7>(v220_acc, v235_bc, v210_data);
          tensorforge::fmacdpp16<8>(v220_acc, v235_bc, v211_data);
          tensorforge::fmacdpp16<9>(v220_acc, v235_bc, v212_data);
          tensorforge::fmacdpp16<10>(v220_acc, v235_bc, v213_data);
          tensorforge::fmacdpp16<11>(v221_acc, v235_bc, v201_data);
          tensorforge::fmacdpp16<12>(v221_acc, v235_bc, v202_data);
          tensorforge::fmacdpp16<13>(v221_acc, v235_bc, v203_data);
          tensorforge::fmacdpp16<14>(v221_acc, v235_bc, v204_data);
          tensorforge::fmacdpp16<15>(v221_acc, v235_bc, v205_data);
          float v236_lin = r7[3];
          float v237_bc = tensorforge::broadcast<32, 16, 0>(v236_lin);
          tensorforge::fmacdpp16<0>(v221_acc, v237_bc, v206_data);
          tensorforge::fmacdpp16<1>(v221_acc, v237_bc, v207_data);
          tensorforge::fmacdpp16<2>(v221_acc, v237_bc, v208_data);
          tensorforge::fmacdpp16<3>(v221_acc, v237_bc, v209_data);
          tensorforge::fmacdpp16<4>(v221_acc, v237_bc, v210_data);
          tensorforge::fmacdpp16<5>(v221_acc, v237_bc, v211_data);
          tensorforge::fmacdpp16<6>(v221_acc, v237_bc, v212_data);
          tensorforge::fmacdpp16<7>(v221_acc, v237_bc, v213_data);
          tensorforge::fmacdpp16<8>(v222_acc, v237_bc, v201_data);
          tensorforge::fmacdpp16<9>(v222_acc, v237_bc, v202_data);
          tensorforge::fmacdpp16<10>(v222_acc, v237_bc, v203_data);
          tensorforge::fmacdpp16<11>(v222_acc, v237_bc, v204_data);
          tensorforge::fmacdpp16<12>(v222_acc, v237_bc, v205_data);
          tensorforge::fmacdpp16<13>(v222_acc, v237_bc, v206_data);
          tensorforge::fmacdpp16<14>(v222_acc, v237_bc, v207_data);
          tensorforge::fmacdpp16<15>(v222_acc, v237_bc, v208_data);
          float v238_bc = tensorforge::broadcast<32, 16, 1>(v236_lin);
          tensorforge::fmacdpp16<0>(v222_acc, v238_bc, v209_data);
          tensorforge::fmacdpp16<1>(v222_acc, v238_bc, v210_data);
          tensorforge::fmacdpp16<2>(v222_acc, v238_bc, v211_data);
          tensorforge::fmacdpp16<3>(v222_acc, v238_bc, v212_data);
          tensorforge::fmacdpp16<4>(v222_acc, v238_bc, v213_data);
          tensorforge::fmacdpp16<5>(v223_acc, v238_bc, v201_data);
          tensorforge::fmacdpp16<6>(v223_acc, v238_bc, v202_data);
          tensorforge::fmacdpp16<7>(v223_acc, v238_bc, v203_data);
          tensorforge::fmacdpp16<8>(v223_acc, v238_bc, v204_data);
          tensorforge::fmacdpp16<9>(v223_acc, v238_bc, v205_data);
          tensorforge::fmacdpp16<10>(v223_acc, v238_bc, v206_data);
          tensorforge::fmacdpp16<11>(v223_acc, v238_bc, v207_data);
          tensorforge::fmacdpp16<12>(v223_acc, v238_bc, v208_data);
          tensorforge::fmacdpp16<13>(v223_acc, v238_bc, v209_data);
          tensorforge::fmacdpp16<14>(v223_acc, v238_bc, v210_data);
          tensorforge::fmacdpp16<15>(v223_acc, v238_bc, v211_data);
          float v239_lin = r7[4];
          float v240_bc = tensorforge::broadcast<32, 16, 0>(v239_lin);
          tensorforge::fmacdpp16<0>(v223_acc, v240_bc, v212_data);
          tensorforge::fmacdpp16<1>(v223_acc, v240_bc, v213_data);
          tensorforge::fmacdpp16<2>(v224_acc, v240_bc, v201_data);
          tensorforge::fmacdpp16<3>(v224_acc, v240_bc, v202_data);
          tensorforge::fmacdpp16<4>(v224_acc, v240_bc, v203_data);
          tensorforge::fmacdpp16<5>(v224_acc, v240_bc, v204_data);
          tensorforge::fmacdpp16<6>(v224_acc, v240_bc, v205_data);
          tensorforge::fmacdpp16<7>(v224_acc, v240_bc, v206_data);
          tensorforge::fmacdpp16<8>(v224_acc, v240_bc, v207_data);
          tensorforge::fmacdpp16<9>(v224_acc, v240_bc, v208_data);
          tensorforge::fmacdpp16<10>(v224_acc, v240_bc, v209_data);
          tensorforge::fmacdpp16<11>(v224_acc, v240_bc, v210_data);
          tensorforge::fmacdpp16<12>(v224_acc, v240_bc, v211_data);
          tensorforge::fmacdpp16<13>(v224_acc, v240_bc, v212_data);
          tensorforge::fmacdpp16<14>(v224_acc, v240_bc, v213_data);
          tensorforge::fmacdpp16<15>(v225_acc, v240_bc, v201_data);
          float v241_bc = tensorforge::broadcast<32, 16, 1>(v239_lin);
          tensorforge::fmacdpp16<0>(v225_acc, v241_bc, v202_data);
          tensorforge::fmacdpp16<1>(v225_acc, v241_bc, v203_data);
          tensorforge::fmacdpp16<2>(v225_acc, v241_bc, v204_data);
          tensorforge::fmacdpp16<3>(v225_acc, v241_bc, v205_data);
          tensorforge::fmacdpp16<4>(v225_acc, v241_bc, v206_data);
          tensorforge::fmacdpp16<5>(v225_acc, v241_bc, v207_data);
          tensorforge::fmacdpp16<6>(v225_acc, v241_bc, v208_data);
          tensorforge::fmacdpp16<7>(v225_acc, v241_bc, v209_data);
          tensorforge::fmacdpp16<8>(v225_acc, v241_bc, v210_data);
          tensorforge::fmacdpp16<9>(v225_acc, v241_bc, v211_data);
          tensorforge::fmacdpp16<10>(v225_acc, v241_bc, v212_data);
          tensorforge::fmacdpp16<11>(v225_acc, v241_bc, v213_data);
          tensorforge::fmacdpp16<12>(v226_acc, v241_bc, v201_data);
          tensorforge::fmacdpp16<13>(v226_acc, v241_bc, v202_data);
          tensorforge::fmacdpp16<14>(v226_acc, v241_bc, v203_data);
          tensorforge::fmacdpp16<15>(v226_acc, v241_bc, v204_data);
          float v242_lin = r7[5];
          float v243_bc = tensorforge::broadcast<32, 16, 0>(v242_lin);
          tensorforge::fmacdpp16<0>(v226_acc, v243_bc, v205_data);
          tensorforge::fmacdpp16<1>(v226_acc, v243_bc, v206_data);
          tensorforge::fmacdpp16<2>(v226_acc, v243_bc, v207_data);
          tensorforge::fmacdpp16<3>(v226_acc, v243_bc, v208_data);
          tensorforge::fmacdpp16<4>(v226_acc, v243_bc, v209_data);
          tensorforge::fmacdpp16<5>(v226_acc, v243_bc, v210_data);
          tensorforge::fmacdpp16<6>(v226_acc, v243_bc, v211_data);
          tensorforge::fmacdpp16<7>(v226_acc, v243_bc, v212_data);
          tensorforge::fmacdpp16<8>(v226_acc, v243_bc, v213_data);
          ir8[0] = v214_acc;
          ir8[1] = v215_acc;
          ir8[2] = v216_acc;
          ir8[3] = v217_acc;
          ir8[4] = v218_acc;
          ir8[5] = v219_acc;
          ir8[6] = v220_acc;
          ir8[7] = v221_acc;
          ir8[8] = v222_acc;
          ir8[9] = v223_acc;
          ir8[10] = v224_acc;
          ir8[11] = v225_acc;
          ir8[12] = v226_acc;
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v247_i0 = 0; v247_i0 < 1; ++v247_i0) {
            int32_t v256_lead = v3_lead + (v247_i0 * 32);
            #pragma unroll
            for (int32_t v248_i1 = 0; v248_i1 < 13; ++v248_i1) {
              int32_t v249_a = v247_i0 + v248_i1;
              float v251_data = r8[(v247_i0 + v248_i1)];
              int32_t v258_a = v256_lead + (v248_i1 * 32);
              glb_m3[v258_a] = v251_data;
            }
          }
          ;
        }
      }
    }
  }
}

