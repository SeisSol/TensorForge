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
          alignas(16) float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v8_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
            int32_t v14_lead = v9_i0 * 32;
            int32_t v15_lead = v8_lead + v14_lead;
            int32_t v22_lead = v8_lead + v14_lead;
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 13; ++v10_i1) {
              int32_t v16_a = v10_i1 * 32;
              int32_t v17_a = v15_lead + v16_a;
              float v25_data = glb_m0[(v22_lead + v16_a)];
              int32_t v26_a = v9_i0 + v10_i1;
              r0[v26_a] = v25_data;
            }
          }
          alignas(16) float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
            int32_t v36_lead = v31_i0 * 32;
            int32_t v37_lead = v8_lead + v36_lead;
            int32_t v44_lead = v8_lead + v36_lead;
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
              int32_t v38_a = v32_i1 * 32;
              int32_t v39_a = v37_lead + v38_a;
              float v47_data = __builtin_nontemporal_load(&glb_m1[(v44_lead + v38_a)]);
              int32_t v48_a = v31_i0 + v32_i1;
              r2[v48_a] = v47_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          alignas(16) float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v53_data = r0[0];
          float v54_data = r1[0];
          r1[0] = (v54_data + v53_data);
          float v56_data = r0[1];
          float v57_data = r1[1];
          r1[1] = (v57_data + v56_data);
          float v59_data = r0[2];
          float v60_data = r1[2];
          r1[2] = (v60_data + v59_data);
          float v62_data = r0[3];
          float v63_data = r1[3];
          r1[3] = (v63_data + v62_data);
          float v65_data = r0[4];
          float v66_data = r1[4];
          r1[4] = (v66_data + v65_data);
          float v68_data = r0[5];
          float v69_data = r1[5];
          r1[5] = (v69_data + v68_data);
          float v71_data = r0[6];
          float v72_data = r1[6];
          r1[6] = (v72_data + v71_data);
          float v74_data = r0[7];
          float v75_data = r1[7];
          r1[7] = (v75_data + v74_data);
          float v77_data = r0[8];
          float v78_data = r1[8];
          r1[8] = (v78_data + v77_data);
          float v80_data = r0[9];
          float v81_data = r1[9];
          r1[9] = (v81_data + v80_data);
          float v83_data = r0[10];
          float v84_data = r1[10];
          r1[10] = (v84_data + v83_data);
          float v86_data = r0[11];
          float v87_data = r1[11];
          r1[11] = (v87_data + v86_data);
          float v89_data = r0[12];
          float v90_data = r1[12];
          r1[12] = (v90_data + v89_data);
          alignas(16) float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          float v93_lin = glb_m2[0 + threadIdx.x * 1];
          r3[0] = v93_lin;
          float v94_lin = glb_m2[32 + threadIdx.x * 1];
          r3[1] = v94_lin;
          float v95_lin = glb_m2[64 + threadIdx.x * 1];
          r3[2] = v95_lin;
          float v96_lin = glb_m2[96 + threadIdx.x * 1];
          r3[3] = v96_lin;
          float v97_lin = glb_m2[128 + threadIdx.x * 1];
          r3[4] = v97_lin;
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          alignas(16) float r4[13]{};
          // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir4[13]{};
          float v100_data = r2[0];
          float v101_data = r2[1];
          float v102_data = r2[2];
          float v103_data = r2[3];
          float v104_data = r2[4];
          float v105_data = r2[5];
          float v106_data = r2[6];
          float v107_data = r2[7];
          float v108_data = r2[8];
          float v109_data = r2[9];
          float v110_data = r2[10];
          float v111_data = r2[11];
          float v112_acc{};
          float v113_acc{};
          float v114_acc{};
          float v115_acc{};
          float v116_acc{};
          float v117_acc{};
          float v118_acc{};
          float v119_acc{};
          float v120_acc{};
          float v121_acc{};
          float v122_acc{};
          float v123_acc{};
          float v124_acc{};
          float v125_lin = r3[0];
          float v126_bc = tensorforge::broadcast<32, 16, 0>(v125_lin);
          tensorforge::fmacdpp16<0>(v112_acc, v126_bc, v100_data);
          tensorforge::fmacdpp16<1>(v112_acc, v126_bc, v101_data);
          tensorforge::fmacdpp16<2>(v112_acc, v126_bc, v102_data);
          tensorforge::fmacdpp16<3>(v112_acc, v126_bc, v103_data);
          tensorforge::fmacdpp16<4>(v112_acc, v126_bc, v104_data);
          tensorforge::fmacdpp16<5>(v112_acc, v126_bc, v105_data);
          tensorforge::fmacdpp16<6>(v112_acc, v126_bc, v106_data);
          tensorforge::fmacdpp16<7>(v112_acc, v126_bc, v107_data);
          tensorforge::fmacdpp16<8>(v112_acc, v126_bc, v108_data);
          tensorforge::fmacdpp16<9>(v112_acc, v126_bc, v109_data);
          tensorforge::fmacdpp16<10>(v112_acc, v126_bc, v110_data);
          tensorforge::fmacdpp16<11>(v112_acc, v126_bc, v111_data);
          tensorforge::fmacdpp16<12>(v113_acc, v126_bc, v100_data);
          tensorforge::fmacdpp16<13>(v113_acc, v126_bc, v101_data);
          tensorforge::fmacdpp16<14>(v113_acc, v126_bc, v102_data);
          tensorforge::fmacdpp16<15>(v113_acc, v126_bc, v103_data);
          float v127_bc = tensorforge::broadcast<32, 16, 1>(v125_lin);
          tensorforge::fmacdpp16<0>(v113_acc, v127_bc, v104_data);
          tensorforge::fmacdpp16<1>(v113_acc, v127_bc, v105_data);
          tensorforge::fmacdpp16<2>(v113_acc, v127_bc, v106_data);
          tensorforge::fmacdpp16<3>(v113_acc, v127_bc, v107_data);
          tensorforge::fmacdpp16<4>(v113_acc, v127_bc, v108_data);
          tensorforge::fmacdpp16<5>(v113_acc, v127_bc, v109_data);
          tensorforge::fmacdpp16<6>(v113_acc, v127_bc, v110_data);
          tensorforge::fmacdpp16<7>(v113_acc, v127_bc, v111_data);
          tensorforge::fmacdpp16<8>(v114_acc, v127_bc, v100_data);
          tensorforge::fmacdpp16<9>(v114_acc, v127_bc, v101_data);
          tensorforge::fmacdpp16<10>(v114_acc, v127_bc, v102_data);
          tensorforge::fmacdpp16<11>(v114_acc, v127_bc, v103_data);
          tensorforge::fmacdpp16<12>(v114_acc, v127_bc, v104_data);
          tensorforge::fmacdpp16<13>(v114_acc, v127_bc, v105_data);
          tensorforge::fmacdpp16<14>(v114_acc, v127_bc, v106_data);
          tensorforge::fmacdpp16<15>(v114_acc, v127_bc, v107_data);
          float v128_lin = r3[1];
          float v129_bc = tensorforge::broadcast<32, 16, 0>(v128_lin);
          tensorforge::fmacdpp16<0>(v114_acc, v129_bc, v108_data);
          tensorforge::fmacdpp16<1>(v114_acc, v129_bc, v109_data);
          tensorforge::fmacdpp16<2>(v114_acc, v129_bc, v110_data);
          tensorforge::fmacdpp16<3>(v114_acc, v129_bc, v111_data);
          tensorforge::fmacdpp16<4>(v115_acc, v129_bc, v100_data);
          tensorforge::fmacdpp16<5>(v115_acc, v129_bc, v101_data);
          tensorforge::fmacdpp16<6>(v115_acc, v129_bc, v102_data);
          tensorforge::fmacdpp16<7>(v115_acc, v129_bc, v103_data);
          tensorforge::fmacdpp16<8>(v115_acc, v129_bc, v104_data);
          tensorforge::fmacdpp16<9>(v115_acc, v129_bc, v105_data);
          tensorforge::fmacdpp16<10>(v115_acc, v129_bc, v106_data);
          tensorforge::fmacdpp16<11>(v115_acc, v129_bc, v107_data);
          tensorforge::fmacdpp16<12>(v115_acc, v129_bc, v108_data);
          tensorforge::fmacdpp16<13>(v115_acc, v129_bc, v109_data);
          tensorforge::fmacdpp16<14>(v115_acc, v129_bc, v110_data);
          tensorforge::fmacdpp16<15>(v115_acc, v129_bc, v111_data);
          float v130_bc = tensorforge::broadcast<32, 16, 1>(v128_lin);
          tensorforge::fmacdpp16<0>(v116_acc, v130_bc, v100_data);
          tensorforge::fmacdpp16<1>(v116_acc, v130_bc, v101_data);
          tensorforge::fmacdpp16<2>(v116_acc, v130_bc, v102_data);
          tensorforge::fmacdpp16<3>(v116_acc, v130_bc, v103_data);
          tensorforge::fmacdpp16<4>(v116_acc, v130_bc, v104_data);
          tensorforge::fmacdpp16<5>(v116_acc, v130_bc, v105_data);
          tensorforge::fmacdpp16<6>(v116_acc, v130_bc, v106_data);
          tensorforge::fmacdpp16<7>(v116_acc, v130_bc, v107_data);
          tensorforge::fmacdpp16<8>(v116_acc, v130_bc, v108_data);
          tensorforge::fmacdpp16<9>(v116_acc, v130_bc, v109_data);
          tensorforge::fmacdpp16<10>(v116_acc, v130_bc, v110_data);
          tensorforge::fmacdpp16<11>(v116_acc, v130_bc, v111_data);
          tensorforge::fmacdpp16<12>(v117_acc, v130_bc, v100_data);
          tensorforge::fmacdpp16<13>(v117_acc, v130_bc, v101_data);
          tensorforge::fmacdpp16<14>(v117_acc, v130_bc, v102_data);
          tensorforge::fmacdpp16<15>(v117_acc, v130_bc, v103_data);
          float v131_lin = r3[2];
          float v132_bc = tensorforge::broadcast<32, 16, 0>(v131_lin);
          tensorforge::fmacdpp16<0>(v117_acc, v132_bc, v104_data);
          tensorforge::fmacdpp16<1>(v117_acc, v132_bc, v105_data);
          tensorforge::fmacdpp16<2>(v117_acc, v132_bc, v106_data);
          tensorforge::fmacdpp16<3>(v117_acc, v132_bc, v107_data);
          tensorforge::fmacdpp16<4>(v117_acc, v132_bc, v108_data);
          tensorforge::fmacdpp16<5>(v117_acc, v132_bc, v109_data);
          tensorforge::fmacdpp16<6>(v117_acc, v132_bc, v110_data);
          tensorforge::fmacdpp16<7>(v117_acc, v132_bc, v111_data);
          tensorforge::fmacdpp16<8>(v118_acc, v132_bc, v100_data);
          tensorforge::fmacdpp16<9>(v118_acc, v132_bc, v101_data);
          tensorforge::fmacdpp16<10>(v118_acc, v132_bc, v102_data);
          tensorforge::fmacdpp16<11>(v118_acc, v132_bc, v103_data);
          tensorforge::fmacdpp16<12>(v118_acc, v132_bc, v104_data);
          tensorforge::fmacdpp16<13>(v118_acc, v132_bc, v105_data);
          tensorforge::fmacdpp16<14>(v118_acc, v132_bc, v106_data);
          tensorforge::fmacdpp16<15>(v118_acc, v132_bc, v107_data);
          float v133_bc = tensorforge::broadcast<32, 16, 1>(v131_lin);
          tensorforge::fmacdpp16<0>(v118_acc, v133_bc, v108_data);
          tensorforge::fmacdpp16<1>(v118_acc, v133_bc, v109_data);
          tensorforge::fmacdpp16<2>(v118_acc, v133_bc, v110_data);
          tensorforge::fmacdpp16<3>(v118_acc, v133_bc, v111_data);
          tensorforge::fmacdpp16<4>(v119_acc, v133_bc, v100_data);
          tensorforge::fmacdpp16<5>(v119_acc, v133_bc, v101_data);
          tensorforge::fmacdpp16<6>(v119_acc, v133_bc, v102_data);
          tensorforge::fmacdpp16<7>(v119_acc, v133_bc, v103_data);
          tensorforge::fmacdpp16<8>(v119_acc, v133_bc, v104_data);
          tensorforge::fmacdpp16<9>(v119_acc, v133_bc, v105_data);
          tensorforge::fmacdpp16<10>(v119_acc, v133_bc, v106_data);
          tensorforge::fmacdpp16<11>(v119_acc, v133_bc, v107_data);
          tensorforge::fmacdpp16<12>(v119_acc, v133_bc, v108_data);
          tensorforge::fmacdpp16<13>(v119_acc, v133_bc, v109_data);
          tensorforge::fmacdpp16<14>(v119_acc, v133_bc, v110_data);
          tensorforge::fmacdpp16<15>(v119_acc, v133_bc, v111_data);
          float v134_lin = r3[3];
          float v135_bc = tensorforge::broadcast<32, 16, 0>(v134_lin);
          tensorforge::fmacdpp16<0>(v120_acc, v135_bc, v100_data);
          tensorforge::fmacdpp16<1>(v120_acc, v135_bc, v101_data);
          tensorforge::fmacdpp16<2>(v120_acc, v135_bc, v102_data);
          tensorforge::fmacdpp16<3>(v120_acc, v135_bc, v103_data);
          tensorforge::fmacdpp16<4>(v120_acc, v135_bc, v104_data);
          tensorforge::fmacdpp16<5>(v120_acc, v135_bc, v105_data);
          tensorforge::fmacdpp16<6>(v120_acc, v135_bc, v106_data);
          tensorforge::fmacdpp16<7>(v120_acc, v135_bc, v107_data);
          tensorforge::fmacdpp16<8>(v120_acc, v135_bc, v108_data);
          tensorforge::fmacdpp16<9>(v120_acc, v135_bc, v109_data);
          tensorforge::fmacdpp16<10>(v120_acc, v135_bc, v110_data);
          tensorforge::fmacdpp16<11>(v120_acc, v135_bc, v111_data);
          tensorforge::fmacdpp16<12>(v121_acc, v135_bc, v100_data);
          tensorforge::fmacdpp16<13>(v121_acc, v135_bc, v101_data);
          tensorforge::fmacdpp16<14>(v121_acc, v135_bc, v102_data);
          tensorforge::fmacdpp16<15>(v121_acc, v135_bc, v103_data);
          float v136_bc = tensorforge::broadcast<32, 16, 1>(v134_lin);
          tensorforge::fmacdpp16<0>(v121_acc, v136_bc, v104_data);
          tensorforge::fmacdpp16<1>(v121_acc, v136_bc, v105_data);
          tensorforge::fmacdpp16<2>(v121_acc, v136_bc, v106_data);
          tensorforge::fmacdpp16<3>(v121_acc, v136_bc, v107_data);
          tensorforge::fmacdpp16<4>(v121_acc, v136_bc, v108_data);
          tensorforge::fmacdpp16<5>(v121_acc, v136_bc, v109_data);
          tensorforge::fmacdpp16<6>(v121_acc, v136_bc, v110_data);
          tensorforge::fmacdpp16<7>(v121_acc, v136_bc, v111_data);
          tensorforge::fmacdpp16<8>(v122_acc, v136_bc, v100_data);
          tensorforge::fmacdpp16<9>(v122_acc, v136_bc, v101_data);
          tensorforge::fmacdpp16<10>(v122_acc, v136_bc, v102_data);
          tensorforge::fmacdpp16<11>(v122_acc, v136_bc, v103_data);
          tensorforge::fmacdpp16<12>(v122_acc, v136_bc, v104_data);
          tensorforge::fmacdpp16<13>(v122_acc, v136_bc, v105_data);
          tensorforge::fmacdpp16<14>(v122_acc, v136_bc, v106_data);
          tensorforge::fmacdpp16<15>(v122_acc, v136_bc, v107_data);
          float v137_lin = r3[4];
          float v138_bc = tensorforge::broadcast<32, 16, 0>(v137_lin);
          tensorforge::fmacdpp16<0>(v122_acc, v138_bc, v108_data);
          tensorforge::fmacdpp16<1>(v122_acc, v138_bc, v109_data);
          tensorforge::fmacdpp16<2>(v122_acc, v138_bc, v110_data);
          tensorforge::fmacdpp16<3>(v122_acc, v138_bc, v111_data);
          tensorforge::fmacdpp16<4>(v123_acc, v138_bc, v100_data);
          tensorforge::fmacdpp16<5>(v123_acc, v138_bc, v101_data);
          tensorforge::fmacdpp16<6>(v123_acc, v138_bc, v102_data);
          tensorforge::fmacdpp16<7>(v123_acc, v138_bc, v103_data);
          tensorforge::fmacdpp16<8>(v123_acc, v138_bc, v104_data);
          tensorforge::fmacdpp16<9>(v123_acc, v138_bc, v105_data);
          tensorforge::fmacdpp16<10>(v123_acc, v138_bc, v106_data);
          tensorforge::fmacdpp16<11>(v123_acc, v138_bc, v107_data);
          tensorforge::fmacdpp16<12>(v123_acc, v138_bc, v108_data);
          tensorforge::fmacdpp16<13>(v123_acc, v138_bc, v109_data);
          tensorforge::fmacdpp16<14>(v123_acc, v138_bc, v110_data);
          tensorforge::fmacdpp16<15>(v123_acc, v138_bc, v111_data);
          float v139_bc = tensorforge::broadcast<32, 16, 1>(v137_lin);
          tensorforge::fmacdpp16<0>(v124_acc, v139_bc, v100_data);
          tensorforge::fmacdpp16<1>(v124_acc, v139_bc, v101_data);
          tensorforge::fmacdpp16<2>(v124_acc, v139_bc, v102_data);
          tensorforge::fmacdpp16<3>(v124_acc, v139_bc, v103_data);
          tensorforge::fmacdpp16<4>(v124_acc, v139_bc, v104_data);
          tensorforge::fmacdpp16<5>(v124_acc, v139_bc, v105_data);
          tensorforge::fmacdpp16<6>(v124_acc, v139_bc, v106_data);
          tensorforge::fmacdpp16<7>(v124_acc, v139_bc, v107_data);
          tensorforge::fmacdpp16<8>(v124_acc, v139_bc, v108_data);
          tensorforge::fmacdpp16<9>(v124_acc, v139_bc, v109_data);
          tensorforge::fmacdpp16<10>(v124_acc, v139_bc, v110_data);
          tensorforge::fmacdpp16<11>(v124_acc, v139_bc, v111_data);
          ir4[0] = v112_acc;
          ir4[1] = v113_acc;
          ir4[2] = v114_acc;
          ir4[3] = v115_acc;
          ir4[4] = v116_acc;
          ir4[5] = v117_acc;
          ir4[6] = v118_acc;
          ir4[7] = v119_acc;
          ir4[8] = v120_acc;
          ir4[9] = v121_acc;
          ir4[10] = v122_acc;
          ir4[11] = v123_acc;
          ir4[12] = v124_acc;
          #pragma unroll
          for (int32_t v143_n0 = 0; v143_n0 < 1; ++v143_n0) {
            #pragma unroll
            for (int32_t v144_n1 = 0; v144_n1 < 13; ++v144_n1) {
              int32_t v145_a = v143_n0 + v144_n1;
              int32_t v146_a = v143_n0 + v144_n1;
              float v147_data = ir4[v146_a];
              int32_t v148_a = v143_n0 + v144_n1;
              float v150_data = r1[v146_a];
              r4[v146_a] = (v150_data + v147_data);
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          float v157_data = r4[4];
          float v158_data = r5[0];
          r5[0] = (v158_data + v157_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v163_i0 = 0; v163_i0 < 1; ++v163_i0) {
            int32_t v172_lead = v8_lead + (v163_i0 * 32);
            #pragma unroll
            for (int32_t v164_i1 = 0; v164_i1 < 1; ++v164_i1) {
              int32_t v165_a = v163_i0 + v164_i1;
              float v167_data = r5[(v163_i0 + v164_i1)];
              glb_m0[(v172_lead + ((v164_i1 + 4) * 32))] = v167_data;
            }
          }
          alignas(16) float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v180_i0 = 0; v180_i0 < 1; ++v180_i0) {
            int32_t v185_lead = v180_i0 * 32;
            int32_t v186_lead = v8_lead + v185_lead;
            int32_t v193_lead = v8_lead + v185_lead;
            #pragma unroll
            for (int32_t v181_i1 = 0; v181_i1 < 13; ++v181_i1) {
              int32_t v187_a = v181_i1 * 32;
              int32_t v188_a = v186_lead + v187_a;
              float v196_data = glb_m0[(v193_lead + v187_a)];
              int32_t v197_a = v180_i0 + v181_i1;
              r6[v197_a] = v196_data;
            }
          }
          alignas(16) float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          float v199_lin = glb_m4[0 + threadIdx.x * 1];
          r7[0] = v199_lin;
          float v200_lin = glb_m4[32 + threadIdx.x * 1];
          r7[1] = v200_lin;
          float v201_lin = glb_m4[64 + threadIdx.x * 1];
          r7[2] = v201_lin;
          float v202_lin = glb_m4[96 + threadIdx.x * 1];
          r7[3] = v202_lin;
          float v203_lin = glb_m4[128 + threadIdx.x * 1];
          r7[4] = v203_lin;
          float v204_lin = glb_m4[160 + threadIdx.x * 1];
          r7[5] = v204_lin;
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          alignas(16) float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v206_data = r6[0];
          float v207_data = r6[1];
          float v208_data = r6[2];
          float v209_data = r6[3];
          float v210_data = r6[4];
          float v211_data = r6[5];
          float v212_data = r6[6];
          float v213_data = r6[7];
          float v214_data = r6[8];
          float v215_data = r6[9];
          float v216_data = r6[10];
          float v217_data = r6[11];
          float v218_data = r6[12];
          float v219_acc{};
          float v220_acc{};
          float v221_acc{};
          float v222_acc{};
          float v223_acc{};
          float v224_acc{};
          float v225_acc{};
          float v226_acc{};
          float v227_acc{};
          float v228_acc{};
          float v229_acc{};
          float v230_acc{};
          float v231_acc{};
          float v232_lin = r7[0];
          float v233_bc = tensorforge::broadcast<32, 16, 0>(v232_lin);
          tensorforge::fmacdpp16<0>(v219_acc, v233_bc, v206_data);
          tensorforge::fmacdpp16<1>(v219_acc, v233_bc, v207_data);
          tensorforge::fmacdpp16<2>(v219_acc, v233_bc, v208_data);
          tensorforge::fmacdpp16<3>(v219_acc, v233_bc, v209_data);
          tensorforge::fmacdpp16<4>(v219_acc, v233_bc, v210_data);
          tensorforge::fmacdpp16<5>(v219_acc, v233_bc, v211_data);
          tensorforge::fmacdpp16<6>(v219_acc, v233_bc, v212_data);
          tensorforge::fmacdpp16<7>(v219_acc, v233_bc, v213_data);
          tensorforge::fmacdpp16<8>(v219_acc, v233_bc, v214_data);
          tensorforge::fmacdpp16<9>(v219_acc, v233_bc, v215_data);
          tensorforge::fmacdpp16<10>(v219_acc, v233_bc, v216_data);
          tensorforge::fmacdpp16<11>(v219_acc, v233_bc, v217_data);
          tensorforge::fmacdpp16<12>(v219_acc, v233_bc, v218_data);
          tensorforge::fmacdpp16<13>(v220_acc, v233_bc, v206_data);
          tensorforge::fmacdpp16<14>(v220_acc, v233_bc, v207_data);
          tensorforge::fmacdpp16<15>(v220_acc, v233_bc, v208_data);
          float v234_bc = tensorforge::broadcast<32, 16, 1>(v232_lin);
          tensorforge::fmacdpp16<0>(v220_acc, v234_bc, v209_data);
          tensorforge::fmacdpp16<1>(v220_acc, v234_bc, v210_data);
          tensorforge::fmacdpp16<2>(v220_acc, v234_bc, v211_data);
          tensorforge::fmacdpp16<3>(v220_acc, v234_bc, v212_data);
          tensorforge::fmacdpp16<4>(v220_acc, v234_bc, v213_data);
          tensorforge::fmacdpp16<5>(v220_acc, v234_bc, v214_data);
          tensorforge::fmacdpp16<6>(v220_acc, v234_bc, v215_data);
          tensorforge::fmacdpp16<7>(v220_acc, v234_bc, v216_data);
          tensorforge::fmacdpp16<8>(v220_acc, v234_bc, v217_data);
          tensorforge::fmacdpp16<9>(v220_acc, v234_bc, v218_data);
          tensorforge::fmacdpp16<10>(v221_acc, v234_bc, v206_data);
          tensorforge::fmacdpp16<11>(v221_acc, v234_bc, v207_data);
          tensorforge::fmacdpp16<12>(v221_acc, v234_bc, v208_data);
          tensorforge::fmacdpp16<13>(v221_acc, v234_bc, v209_data);
          tensorforge::fmacdpp16<14>(v221_acc, v234_bc, v210_data);
          tensorforge::fmacdpp16<15>(v221_acc, v234_bc, v211_data);
          float v235_lin = r7[1];
          float v236_bc = tensorforge::broadcast<32, 16, 0>(v235_lin);
          tensorforge::fmacdpp16<0>(v221_acc, v236_bc, v212_data);
          tensorforge::fmacdpp16<1>(v221_acc, v236_bc, v213_data);
          tensorforge::fmacdpp16<2>(v221_acc, v236_bc, v214_data);
          tensorforge::fmacdpp16<3>(v221_acc, v236_bc, v215_data);
          tensorforge::fmacdpp16<4>(v221_acc, v236_bc, v216_data);
          tensorforge::fmacdpp16<5>(v221_acc, v236_bc, v217_data);
          tensorforge::fmacdpp16<6>(v221_acc, v236_bc, v218_data);
          tensorforge::fmacdpp16<7>(v222_acc, v236_bc, v206_data);
          tensorforge::fmacdpp16<8>(v222_acc, v236_bc, v207_data);
          tensorforge::fmacdpp16<9>(v222_acc, v236_bc, v208_data);
          tensorforge::fmacdpp16<10>(v222_acc, v236_bc, v209_data);
          tensorforge::fmacdpp16<11>(v222_acc, v236_bc, v210_data);
          tensorforge::fmacdpp16<12>(v222_acc, v236_bc, v211_data);
          tensorforge::fmacdpp16<13>(v222_acc, v236_bc, v212_data);
          tensorforge::fmacdpp16<14>(v222_acc, v236_bc, v213_data);
          tensorforge::fmacdpp16<15>(v222_acc, v236_bc, v214_data);
          float v237_bc = tensorforge::broadcast<32, 16, 1>(v235_lin);
          tensorforge::fmacdpp16<0>(v222_acc, v237_bc, v215_data);
          tensorforge::fmacdpp16<1>(v222_acc, v237_bc, v216_data);
          tensorforge::fmacdpp16<2>(v222_acc, v237_bc, v217_data);
          tensorforge::fmacdpp16<3>(v222_acc, v237_bc, v218_data);
          tensorforge::fmacdpp16<4>(v223_acc, v237_bc, v206_data);
          tensorforge::fmacdpp16<5>(v223_acc, v237_bc, v207_data);
          tensorforge::fmacdpp16<6>(v223_acc, v237_bc, v208_data);
          tensorforge::fmacdpp16<7>(v223_acc, v237_bc, v209_data);
          tensorforge::fmacdpp16<8>(v223_acc, v237_bc, v210_data);
          tensorforge::fmacdpp16<9>(v223_acc, v237_bc, v211_data);
          tensorforge::fmacdpp16<10>(v223_acc, v237_bc, v212_data);
          tensorforge::fmacdpp16<11>(v223_acc, v237_bc, v213_data);
          tensorforge::fmacdpp16<12>(v223_acc, v237_bc, v214_data);
          tensorforge::fmacdpp16<13>(v223_acc, v237_bc, v215_data);
          tensorforge::fmacdpp16<14>(v223_acc, v237_bc, v216_data);
          tensorforge::fmacdpp16<15>(v223_acc, v237_bc, v217_data);
          float v238_lin = r7[2];
          float v239_bc = tensorforge::broadcast<32, 16, 0>(v238_lin);
          tensorforge::fmacdpp16<0>(v223_acc, v239_bc, v218_data);
          tensorforge::fmacdpp16<1>(v224_acc, v239_bc, v206_data);
          tensorforge::fmacdpp16<2>(v224_acc, v239_bc, v207_data);
          tensorforge::fmacdpp16<3>(v224_acc, v239_bc, v208_data);
          tensorforge::fmacdpp16<4>(v224_acc, v239_bc, v209_data);
          tensorforge::fmacdpp16<5>(v224_acc, v239_bc, v210_data);
          tensorforge::fmacdpp16<6>(v224_acc, v239_bc, v211_data);
          tensorforge::fmacdpp16<7>(v224_acc, v239_bc, v212_data);
          tensorforge::fmacdpp16<8>(v224_acc, v239_bc, v213_data);
          tensorforge::fmacdpp16<9>(v224_acc, v239_bc, v214_data);
          tensorforge::fmacdpp16<10>(v224_acc, v239_bc, v215_data);
          tensorforge::fmacdpp16<11>(v224_acc, v239_bc, v216_data);
          tensorforge::fmacdpp16<12>(v224_acc, v239_bc, v217_data);
          tensorforge::fmacdpp16<13>(v224_acc, v239_bc, v218_data);
          tensorforge::fmacdpp16<14>(v225_acc, v239_bc, v206_data);
          tensorforge::fmacdpp16<15>(v225_acc, v239_bc, v207_data);
          float v240_bc = tensorforge::broadcast<32, 16, 1>(v238_lin);
          tensorforge::fmacdpp16<0>(v225_acc, v240_bc, v208_data);
          tensorforge::fmacdpp16<1>(v225_acc, v240_bc, v209_data);
          tensorforge::fmacdpp16<2>(v225_acc, v240_bc, v210_data);
          tensorforge::fmacdpp16<3>(v225_acc, v240_bc, v211_data);
          tensorforge::fmacdpp16<4>(v225_acc, v240_bc, v212_data);
          tensorforge::fmacdpp16<5>(v225_acc, v240_bc, v213_data);
          tensorforge::fmacdpp16<6>(v225_acc, v240_bc, v214_data);
          tensorforge::fmacdpp16<7>(v225_acc, v240_bc, v215_data);
          tensorforge::fmacdpp16<8>(v225_acc, v240_bc, v216_data);
          tensorforge::fmacdpp16<9>(v225_acc, v240_bc, v217_data);
          tensorforge::fmacdpp16<10>(v225_acc, v240_bc, v218_data);
          tensorforge::fmacdpp16<11>(v226_acc, v240_bc, v206_data);
          tensorforge::fmacdpp16<12>(v226_acc, v240_bc, v207_data);
          tensorforge::fmacdpp16<13>(v226_acc, v240_bc, v208_data);
          tensorforge::fmacdpp16<14>(v226_acc, v240_bc, v209_data);
          tensorforge::fmacdpp16<15>(v226_acc, v240_bc, v210_data);
          float v241_lin = r7[3];
          float v242_bc = tensorforge::broadcast<32, 16, 0>(v241_lin);
          tensorforge::fmacdpp16<0>(v226_acc, v242_bc, v211_data);
          tensorforge::fmacdpp16<1>(v226_acc, v242_bc, v212_data);
          tensorforge::fmacdpp16<2>(v226_acc, v242_bc, v213_data);
          tensorforge::fmacdpp16<3>(v226_acc, v242_bc, v214_data);
          tensorforge::fmacdpp16<4>(v226_acc, v242_bc, v215_data);
          tensorforge::fmacdpp16<5>(v226_acc, v242_bc, v216_data);
          tensorforge::fmacdpp16<6>(v226_acc, v242_bc, v217_data);
          tensorforge::fmacdpp16<7>(v226_acc, v242_bc, v218_data);
          tensorforge::fmacdpp16<8>(v227_acc, v242_bc, v206_data);
          tensorforge::fmacdpp16<9>(v227_acc, v242_bc, v207_data);
          tensorforge::fmacdpp16<10>(v227_acc, v242_bc, v208_data);
          tensorforge::fmacdpp16<11>(v227_acc, v242_bc, v209_data);
          tensorforge::fmacdpp16<12>(v227_acc, v242_bc, v210_data);
          tensorforge::fmacdpp16<13>(v227_acc, v242_bc, v211_data);
          tensorforge::fmacdpp16<14>(v227_acc, v242_bc, v212_data);
          tensorforge::fmacdpp16<15>(v227_acc, v242_bc, v213_data);
          float v243_bc = tensorforge::broadcast<32, 16, 1>(v241_lin);
          tensorforge::fmacdpp16<0>(v227_acc, v243_bc, v214_data);
          tensorforge::fmacdpp16<1>(v227_acc, v243_bc, v215_data);
          tensorforge::fmacdpp16<2>(v227_acc, v243_bc, v216_data);
          tensorforge::fmacdpp16<3>(v227_acc, v243_bc, v217_data);
          tensorforge::fmacdpp16<4>(v227_acc, v243_bc, v218_data);
          tensorforge::fmacdpp16<5>(v228_acc, v243_bc, v206_data);
          tensorforge::fmacdpp16<6>(v228_acc, v243_bc, v207_data);
          tensorforge::fmacdpp16<7>(v228_acc, v243_bc, v208_data);
          tensorforge::fmacdpp16<8>(v228_acc, v243_bc, v209_data);
          tensorforge::fmacdpp16<9>(v228_acc, v243_bc, v210_data);
          tensorforge::fmacdpp16<10>(v228_acc, v243_bc, v211_data);
          tensorforge::fmacdpp16<11>(v228_acc, v243_bc, v212_data);
          tensorforge::fmacdpp16<12>(v228_acc, v243_bc, v213_data);
          tensorforge::fmacdpp16<13>(v228_acc, v243_bc, v214_data);
          tensorforge::fmacdpp16<14>(v228_acc, v243_bc, v215_data);
          tensorforge::fmacdpp16<15>(v228_acc, v243_bc, v216_data);
          float v244_lin = r7[4];
          float v245_bc = tensorforge::broadcast<32, 16, 0>(v244_lin);
          tensorforge::fmacdpp16<0>(v228_acc, v245_bc, v217_data);
          tensorforge::fmacdpp16<1>(v228_acc, v245_bc, v218_data);
          tensorforge::fmacdpp16<2>(v229_acc, v245_bc, v206_data);
          tensorforge::fmacdpp16<3>(v229_acc, v245_bc, v207_data);
          tensorforge::fmacdpp16<4>(v229_acc, v245_bc, v208_data);
          tensorforge::fmacdpp16<5>(v229_acc, v245_bc, v209_data);
          tensorforge::fmacdpp16<6>(v229_acc, v245_bc, v210_data);
          tensorforge::fmacdpp16<7>(v229_acc, v245_bc, v211_data);
          tensorforge::fmacdpp16<8>(v229_acc, v245_bc, v212_data);
          tensorforge::fmacdpp16<9>(v229_acc, v245_bc, v213_data);
          tensorforge::fmacdpp16<10>(v229_acc, v245_bc, v214_data);
          tensorforge::fmacdpp16<11>(v229_acc, v245_bc, v215_data);
          tensorforge::fmacdpp16<12>(v229_acc, v245_bc, v216_data);
          tensorforge::fmacdpp16<13>(v229_acc, v245_bc, v217_data);
          tensorforge::fmacdpp16<14>(v229_acc, v245_bc, v218_data);
          tensorforge::fmacdpp16<15>(v230_acc, v245_bc, v206_data);
          float v246_bc = tensorforge::broadcast<32, 16, 1>(v244_lin);
          tensorforge::fmacdpp16<0>(v230_acc, v246_bc, v207_data);
          tensorforge::fmacdpp16<1>(v230_acc, v246_bc, v208_data);
          tensorforge::fmacdpp16<2>(v230_acc, v246_bc, v209_data);
          tensorforge::fmacdpp16<3>(v230_acc, v246_bc, v210_data);
          tensorforge::fmacdpp16<4>(v230_acc, v246_bc, v211_data);
          tensorforge::fmacdpp16<5>(v230_acc, v246_bc, v212_data);
          tensorforge::fmacdpp16<6>(v230_acc, v246_bc, v213_data);
          tensorforge::fmacdpp16<7>(v230_acc, v246_bc, v214_data);
          tensorforge::fmacdpp16<8>(v230_acc, v246_bc, v215_data);
          tensorforge::fmacdpp16<9>(v230_acc, v246_bc, v216_data);
          tensorforge::fmacdpp16<10>(v230_acc, v246_bc, v217_data);
          tensorforge::fmacdpp16<11>(v230_acc, v246_bc, v218_data);
          tensorforge::fmacdpp16<12>(v231_acc, v246_bc, v206_data);
          tensorforge::fmacdpp16<13>(v231_acc, v246_bc, v207_data);
          tensorforge::fmacdpp16<14>(v231_acc, v246_bc, v208_data);
          tensorforge::fmacdpp16<15>(v231_acc, v246_bc, v209_data);
          float v247_lin = r7[5];
          float v248_bc = tensorforge::broadcast<32, 16, 0>(v247_lin);
          tensorforge::fmacdpp16<0>(v231_acc, v248_bc, v210_data);
          tensorforge::fmacdpp16<1>(v231_acc, v248_bc, v211_data);
          tensorforge::fmacdpp16<2>(v231_acc, v248_bc, v212_data);
          tensorforge::fmacdpp16<3>(v231_acc, v248_bc, v213_data);
          tensorforge::fmacdpp16<4>(v231_acc, v248_bc, v214_data);
          tensorforge::fmacdpp16<5>(v231_acc, v248_bc, v215_data);
          tensorforge::fmacdpp16<6>(v231_acc, v248_bc, v216_data);
          tensorforge::fmacdpp16<7>(v231_acc, v248_bc, v217_data);
          tensorforge::fmacdpp16<8>(v231_acc, v248_bc, v218_data);
          r8[0] = v219_acc;
          r8[1] = v220_acc;
          r8[2] = v221_acc;
          r8[3] = v222_acc;
          r8[4] = v223_acc;
          r8[5] = v224_acc;
          r8[6] = v225_acc;
          r8[7] = v226_acc;
          r8[8] = v227_acc;
          r8[9] = v228_acc;
          r8[10] = v229_acc;
          r8[11] = v230_acc;
          r8[12] = v231_acc;
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v252_i0 = 0; v252_i0 < 1; ++v252_i0) {
            int32_t v261_lead = v8_lead + (v252_i0 * 32);
            #pragma unroll
            for (int32_t v253_i1 = 0; v253_i1 < 13; ++v253_i1) {
              int32_t v254_a = v252_i0 + v253_i1;
              float v256_data = r8[(v252_i0 + v253_i1)];
              glb_m3[(v261_lead + (v253_i1 * 32))] = v256_data;
            }
          }
        }
      }
    }
  }
}

