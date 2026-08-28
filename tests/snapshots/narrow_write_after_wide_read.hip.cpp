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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 156 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v18_lead = v13_i0 * 32;
            int32_t v19_lead = v12_lead + v18_lead;
            int32_t v26_lead = v12_lead + v18_lead;
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 13; ++v14_i1) {
              int32_t v20_a = v14_i1 * 32;
              int32_t v21_a = v19_lead + v20_a;
              float v29_data = glb_m0[(v26_lead + v20_a)];
              r0[(v13_i0 + v14_i1)] = v29_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
            int32_t v40_lead = v35_i0 * 32;
            int32_t v41_lead = v12_lead + v40_lead;
            int32_t v48_lead = v12_lead + v40_lead;
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              int32_t v42_a = v36_i1 * 32;
              int32_t v43_a = v41_lead + v42_a;
              float v51_data = __builtin_nontemporal_load(&glb_m1[(v48_lead + v42_a)]);
              r2[(v35_i0 + v36_i1)] = v51_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v57_data = r0[0];
          float v58_data = r1[0];
          r1[0] = (v58_data + v57_data);
          float v60_data = r0[1];
          float v61_data = r1[1];
          r1[1] = (v61_data + v60_data);
          float v63_data = r0[2];
          float v64_data = r1[2];
          r1[2] = (v64_data + v63_data);
          float v66_data = r0[3];
          float v67_data = r1[3];
          r1[3] = (v67_data + v66_data);
          float v69_data = r0[4];
          float v70_data = r1[4];
          r1[4] = (v70_data + v69_data);
          float v72_data = r0[5];
          float v73_data = r1[5];
          r1[5] = (v73_data + v72_data);
          float v75_data = r0[6];
          float v76_data = r1[6];
          r1[6] = (v76_data + v75_data);
          float v78_data = r0[7];
          float v79_data = r1[7];
          r1[7] = (v79_data + v78_data);
          float v81_data = r0[8];
          float v82_data = r1[8];
          r1[8] = (v82_data + v81_data);
          float v84_data = r0[9];
          float v85_data = r1[9];
          r1[9] = (v85_data + v84_data);
          float v87_data = r0[10];
          float v88_data = r1[10];
          r1[10] = (v88_data + v87_data);
          float v90_data = r0[11];
          float v91_data = r1[11];
          r1[11] = (v91_data + v90_data);
          float v93_data = r0[12];
          float v94_data = r1[12];
          r1[12] = (v94_data + v93_data);
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          float v97_lin = glb_m2[0 + threadIdx.x * 1];
          r3[0] = v97_lin;
          float v98_lin = glb_m2[32 + threadIdx.x * 1];
          r3[1] = v98_lin;
          float v99_lin = glb_m2[64 + threadIdx.x * 1];
          r3[2] = v99_lin;
          float v100_lin = glb_m2[96 + threadIdx.x * 1];
          r3[3] = v100_lin;
          float v101_lin = glb_m2[128 + threadIdx.x * 1];
          r3[4] = v101_lin;
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir4[13]{};
          float v104_data = r2[0];
          float v105_data = r2[1];
          float v106_data = r2[2];
          float v107_data = r2[3];
          float v108_data = r2[4];
          float v109_data = r2[5];
          float v110_data = r2[6];
          float v111_data = r2[7];
          float v112_data = r2[8];
          float v113_data = r2[9];
          float v114_data = r2[10];
          float v115_data = r2[11];
          float v116_acc{};
          float v117_acc{};
          float v118_acc{};
          float v119_acc{};
          float v120_acc{};
          float v121_acc{};
          float v122_acc{};
          float v123_acc{};
          float v124_acc{};
          float v125_acc{};
          float v126_acc{};
          float v127_acc{};
          float v128_acc{};
          float v129_lin = r3[0];
          float v130_bc = tensorforge::broadcast<32, 16, 0>(v129_lin);
          tensorforge::fmacdpp16<0>(v116_acc, v130_bc, v104_data);
          tensorforge::fmacdpp16<1>(v116_acc, v130_bc, v105_data);
          tensorforge::fmacdpp16<2>(v116_acc, v130_bc, v106_data);
          tensorforge::fmacdpp16<3>(v116_acc, v130_bc, v107_data);
          tensorforge::fmacdpp16<4>(v116_acc, v130_bc, v108_data);
          tensorforge::fmacdpp16<5>(v116_acc, v130_bc, v109_data);
          tensorforge::fmacdpp16<6>(v116_acc, v130_bc, v110_data);
          tensorforge::fmacdpp16<7>(v116_acc, v130_bc, v111_data);
          tensorforge::fmacdpp16<8>(v116_acc, v130_bc, v112_data);
          tensorforge::fmacdpp16<9>(v116_acc, v130_bc, v113_data);
          tensorforge::fmacdpp16<10>(v116_acc, v130_bc, v114_data);
          tensorforge::fmacdpp16<11>(v116_acc, v130_bc, v115_data);
          tensorforge::fmacdpp16<12>(v117_acc, v130_bc, v104_data);
          tensorforge::fmacdpp16<13>(v117_acc, v130_bc, v105_data);
          tensorforge::fmacdpp16<14>(v117_acc, v130_bc, v106_data);
          tensorforge::fmacdpp16<15>(v117_acc, v130_bc, v107_data);
          float v131_bc = tensorforge::broadcast<32, 16, 1>(v129_lin);
          tensorforge::fmacdpp16<0>(v117_acc, v131_bc, v108_data);
          tensorforge::fmacdpp16<1>(v117_acc, v131_bc, v109_data);
          tensorforge::fmacdpp16<2>(v117_acc, v131_bc, v110_data);
          tensorforge::fmacdpp16<3>(v117_acc, v131_bc, v111_data);
          tensorforge::fmacdpp16<4>(v117_acc, v131_bc, v112_data);
          tensorforge::fmacdpp16<5>(v117_acc, v131_bc, v113_data);
          tensorforge::fmacdpp16<6>(v117_acc, v131_bc, v114_data);
          tensorforge::fmacdpp16<7>(v117_acc, v131_bc, v115_data);
          tensorforge::fmacdpp16<8>(v118_acc, v131_bc, v104_data);
          tensorforge::fmacdpp16<9>(v118_acc, v131_bc, v105_data);
          tensorforge::fmacdpp16<10>(v118_acc, v131_bc, v106_data);
          tensorforge::fmacdpp16<11>(v118_acc, v131_bc, v107_data);
          tensorforge::fmacdpp16<12>(v118_acc, v131_bc, v108_data);
          tensorforge::fmacdpp16<13>(v118_acc, v131_bc, v109_data);
          tensorforge::fmacdpp16<14>(v118_acc, v131_bc, v110_data);
          tensorforge::fmacdpp16<15>(v118_acc, v131_bc, v111_data);
          float v132_lin = r3[1];
          float v133_bc = tensorforge::broadcast<32, 16, 0>(v132_lin);
          tensorforge::fmacdpp16<0>(v118_acc, v133_bc, v112_data);
          tensorforge::fmacdpp16<1>(v118_acc, v133_bc, v113_data);
          tensorforge::fmacdpp16<2>(v118_acc, v133_bc, v114_data);
          tensorforge::fmacdpp16<3>(v118_acc, v133_bc, v115_data);
          tensorforge::fmacdpp16<4>(v119_acc, v133_bc, v104_data);
          tensorforge::fmacdpp16<5>(v119_acc, v133_bc, v105_data);
          tensorforge::fmacdpp16<6>(v119_acc, v133_bc, v106_data);
          tensorforge::fmacdpp16<7>(v119_acc, v133_bc, v107_data);
          tensorforge::fmacdpp16<8>(v119_acc, v133_bc, v108_data);
          tensorforge::fmacdpp16<9>(v119_acc, v133_bc, v109_data);
          tensorforge::fmacdpp16<10>(v119_acc, v133_bc, v110_data);
          tensorforge::fmacdpp16<11>(v119_acc, v133_bc, v111_data);
          tensorforge::fmacdpp16<12>(v119_acc, v133_bc, v112_data);
          tensorforge::fmacdpp16<13>(v119_acc, v133_bc, v113_data);
          tensorforge::fmacdpp16<14>(v119_acc, v133_bc, v114_data);
          tensorforge::fmacdpp16<15>(v119_acc, v133_bc, v115_data);
          float v134_bc = tensorforge::broadcast<32, 16, 1>(v132_lin);
          tensorforge::fmacdpp16<0>(v120_acc, v134_bc, v104_data);
          tensorforge::fmacdpp16<1>(v120_acc, v134_bc, v105_data);
          tensorforge::fmacdpp16<2>(v120_acc, v134_bc, v106_data);
          tensorforge::fmacdpp16<3>(v120_acc, v134_bc, v107_data);
          tensorforge::fmacdpp16<4>(v120_acc, v134_bc, v108_data);
          tensorforge::fmacdpp16<5>(v120_acc, v134_bc, v109_data);
          tensorforge::fmacdpp16<6>(v120_acc, v134_bc, v110_data);
          tensorforge::fmacdpp16<7>(v120_acc, v134_bc, v111_data);
          tensorforge::fmacdpp16<8>(v120_acc, v134_bc, v112_data);
          tensorforge::fmacdpp16<9>(v120_acc, v134_bc, v113_data);
          tensorforge::fmacdpp16<10>(v120_acc, v134_bc, v114_data);
          tensorforge::fmacdpp16<11>(v120_acc, v134_bc, v115_data);
          tensorforge::fmacdpp16<12>(v121_acc, v134_bc, v104_data);
          tensorforge::fmacdpp16<13>(v121_acc, v134_bc, v105_data);
          tensorforge::fmacdpp16<14>(v121_acc, v134_bc, v106_data);
          tensorforge::fmacdpp16<15>(v121_acc, v134_bc, v107_data);
          float v135_lin = r3[2];
          float v136_bc = tensorforge::broadcast<32, 16, 0>(v135_lin);
          tensorforge::fmacdpp16<0>(v121_acc, v136_bc, v108_data);
          tensorforge::fmacdpp16<1>(v121_acc, v136_bc, v109_data);
          tensorforge::fmacdpp16<2>(v121_acc, v136_bc, v110_data);
          tensorforge::fmacdpp16<3>(v121_acc, v136_bc, v111_data);
          tensorforge::fmacdpp16<4>(v121_acc, v136_bc, v112_data);
          tensorforge::fmacdpp16<5>(v121_acc, v136_bc, v113_data);
          tensorforge::fmacdpp16<6>(v121_acc, v136_bc, v114_data);
          tensorforge::fmacdpp16<7>(v121_acc, v136_bc, v115_data);
          tensorforge::fmacdpp16<8>(v122_acc, v136_bc, v104_data);
          tensorforge::fmacdpp16<9>(v122_acc, v136_bc, v105_data);
          tensorforge::fmacdpp16<10>(v122_acc, v136_bc, v106_data);
          tensorforge::fmacdpp16<11>(v122_acc, v136_bc, v107_data);
          tensorforge::fmacdpp16<12>(v122_acc, v136_bc, v108_data);
          tensorforge::fmacdpp16<13>(v122_acc, v136_bc, v109_data);
          tensorforge::fmacdpp16<14>(v122_acc, v136_bc, v110_data);
          tensorforge::fmacdpp16<15>(v122_acc, v136_bc, v111_data);
          float v137_bc = tensorforge::broadcast<32, 16, 1>(v135_lin);
          tensorforge::fmacdpp16<0>(v122_acc, v137_bc, v112_data);
          tensorforge::fmacdpp16<1>(v122_acc, v137_bc, v113_data);
          tensorforge::fmacdpp16<2>(v122_acc, v137_bc, v114_data);
          tensorforge::fmacdpp16<3>(v122_acc, v137_bc, v115_data);
          tensorforge::fmacdpp16<4>(v123_acc, v137_bc, v104_data);
          tensorforge::fmacdpp16<5>(v123_acc, v137_bc, v105_data);
          tensorforge::fmacdpp16<6>(v123_acc, v137_bc, v106_data);
          tensorforge::fmacdpp16<7>(v123_acc, v137_bc, v107_data);
          tensorforge::fmacdpp16<8>(v123_acc, v137_bc, v108_data);
          tensorforge::fmacdpp16<9>(v123_acc, v137_bc, v109_data);
          tensorforge::fmacdpp16<10>(v123_acc, v137_bc, v110_data);
          tensorforge::fmacdpp16<11>(v123_acc, v137_bc, v111_data);
          tensorforge::fmacdpp16<12>(v123_acc, v137_bc, v112_data);
          tensorforge::fmacdpp16<13>(v123_acc, v137_bc, v113_data);
          tensorforge::fmacdpp16<14>(v123_acc, v137_bc, v114_data);
          tensorforge::fmacdpp16<15>(v123_acc, v137_bc, v115_data);
          float v138_lin = r3[3];
          float v139_bc = tensorforge::broadcast<32, 16, 0>(v138_lin);
          tensorforge::fmacdpp16<0>(v124_acc, v139_bc, v104_data);
          tensorforge::fmacdpp16<1>(v124_acc, v139_bc, v105_data);
          tensorforge::fmacdpp16<2>(v124_acc, v139_bc, v106_data);
          tensorforge::fmacdpp16<3>(v124_acc, v139_bc, v107_data);
          tensorforge::fmacdpp16<4>(v124_acc, v139_bc, v108_data);
          tensorforge::fmacdpp16<5>(v124_acc, v139_bc, v109_data);
          tensorforge::fmacdpp16<6>(v124_acc, v139_bc, v110_data);
          tensorforge::fmacdpp16<7>(v124_acc, v139_bc, v111_data);
          tensorforge::fmacdpp16<8>(v124_acc, v139_bc, v112_data);
          tensorforge::fmacdpp16<9>(v124_acc, v139_bc, v113_data);
          tensorforge::fmacdpp16<10>(v124_acc, v139_bc, v114_data);
          tensorforge::fmacdpp16<11>(v124_acc, v139_bc, v115_data);
          tensorforge::fmacdpp16<12>(v125_acc, v139_bc, v104_data);
          tensorforge::fmacdpp16<13>(v125_acc, v139_bc, v105_data);
          tensorforge::fmacdpp16<14>(v125_acc, v139_bc, v106_data);
          tensorforge::fmacdpp16<15>(v125_acc, v139_bc, v107_data);
          float v140_bc = tensorforge::broadcast<32, 16, 1>(v138_lin);
          tensorforge::fmacdpp16<0>(v125_acc, v140_bc, v108_data);
          tensorforge::fmacdpp16<1>(v125_acc, v140_bc, v109_data);
          tensorforge::fmacdpp16<2>(v125_acc, v140_bc, v110_data);
          tensorforge::fmacdpp16<3>(v125_acc, v140_bc, v111_data);
          tensorforge::fmacdpp16<4>(v125_acc, v140_bc, v112_data);
          tensorforge::fmacdpp16<5>(v125_acc, v140_bc, v113_data);
          tensorforge::fmacdpp16<6>(v125_acc, v140_bc, v114_data);
          tensorforge::fmacdpp16<7>(v125_acc, v140_bc, v115_data);
          tensorforge::fmacdpp16<8>(v126_acc, v140_bc, v104_data);
          tensorforge::fmacdpp16<9>(v126_acc, v140_bc, v105_data);
          tensorforge::fmacdpp16<10>(v126_acc, v140_bc, v106_data);
          tensorforge::fmacdpp16<11>(v126_acc, v140_bc, v107_data);
          tensorforge::fmacdpp16<12>(v126_acc, v140_bc, v108_data);
          tensorforge::fmacdpp16<13>(v126_acc, v140_bc, v109_data);
          tensorforge::fmacdpp16<14>(v126_acc, v140_bc, v110_data);
          tensorforge::fmacdpp16<15>(v126_acc, v140_bc, v111_data);
          float v141_lin = r3[4];
          float v142_bc = tensorforge::broadcast<32, 16, 0>(v141_lin);
          tensorforge::fmacdpp16<0>(v126_acc, v142_bc, v112_data);
          tensorforge::fmacdpp16<1>(v126_acc, v142_bc, v113_data);
          tensorforge::fmacdpp16<2>(v126_acc, v142_bc, v114_data);
          tensorforge::fmacdpp16<3>(v126_acc, v142_bc, v115_data);
          tensorforge::fmacdpp16<4>(v127_acc, v142_bc, v104_data);
          tensorforge::fmacdpp16<5>(v127_acc, v142_bc, v105_data);
          tensorforge::fmacdpp16<6>(v127_acc, v142_bc, v106_data);
          tensorforge::fmacdpp16<7>(v127_acc, v142_bc, v107_data);
          tensorforge::fmacdpp16<8>(v127_acc, v142_bc, v108_data);
          tensorforge::fmacdpp16<9>(v127_acc, v142_bc, v109_data);
          tensorforge::fmacdpp16<10>(v127_acc, v142_bc, v110_data);
          tensorforge::fmacdpp16<11>(v127_acc, v142_bc, v111_data);
          tensorforge::fmacdpp16<12>(v127_acc, v142_bc, v112_data);
          tensorforge::fmacdpp16<13>(v127_acc, v142_bc, v113_data);
          tensorforge::fmacdpp16<14>(v127_acc, v142_bc, v114_data);
          tensorforge::fmacdpp16<15>(v127_acc, v142_bc, v115_data);
          float v143_bc = tensorforge::broadcast<32, 16, 1>(v141_lin);
          tensorforge::fmacdpp16<0>(v128_acc, v143_bc, v104_data);
          tensorforge::fmacdpp16<1>(v128_acc, v143_bc, v105_data);
          tensorforge::fmacdpp16<2>(v128_acc, v143_bc, v106_data);
          tensorforge::fmacdpp16<3>(v128_acc, v143_bc, v107_data);
          tensorforge::fmacdpp16<4>(v128_acc, v143_bc, v108_data);
          tensorforge::fmacdpp16<5>(v128_acc, v143_bc, v109_data);
          tensorforge::fmacdpp16<6>(v128_acc, v143_bc, v110_data);
          tensorforge::fmacdpp16<7>(v128_acc, v143_bc, v111_data);
          tensorforge::fmacdpp16<8>(v128_acc, v143_bc, v112_data);
          tensorforge::fmacdpp16<9>(v128_acc, v143_bc, v113_data);
          tensorforge::fmacdpp16<10>(v128_acc, v143_bc, v114_data);
          tensorforge::fmacdpp16<11>(v128_acc, v143_bc, v115_data);
          ir4[0] = v116_acc;
          ir4[1] = v117_acc;
          ir4[2] = v118_acc;
          ir4[3] = v119_acc;
          ir4[4] = v120_acc;
          ir4[5] = v121_acc;
          ir4[6] = v122_acc;
          ir4[7] = v123_acc;
          ir4[8] = v124_acc;
          ir4[9] = v125_acc;
          ir4[10] = v126_acc;
          ir4[11] = v127_acc;
          ir4[12] = v128_acc;
          #pragma unroll
          for (int32_t v147_n0 = 0; v147_n0 < 1; ++v147_n0) {
            #pragma unroll
            for (int32_t v148_n1 = 0; v148_n1 < 13; ++v148_n1) {
              int32_t v149_a = v147_n0 + v148_n1;
              int32_t v150_a = v147_n0 + v148_n1;
              float v151_data = ir4[v150_a];
              int32_t v152_a = v147_n0 + v148_n1;
              float v154_data = r1[v150_a];
              r4[v150_a] = (v154_data + v151_data);
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          float v161_data = r4[4];
          float v162_data = r5[0];
          r5[0] = (v162_data + v161_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v167_i0 = 0; v167_i0 < 1; ++v167_i0) {
            int32_t v176_lead = v12_lead + (v167_i0 * 32);
            #pragma unroll
            for (int32_t v168_i1 = 0; v168_i1 < 1; ++v168_i1) {
              int32_t v169_a = v167_i0 + v168_i1;
              float v171_data = r5[(v167_i0 + v168_i1)];
              glb_m0[(v176_lead + ((v168_i1 + 4) * 32))] = v171_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v184_i0 = 0; v184_i0 < 1; ++v184_i0) {
            int32_t v189_lead = v184_i0 * 32;
            int32_t v190_lead = v12_lead + v189_lead;
            int32_t v197_lead = v12_lead + v189_lead;
            #pragma unroll
            for (int32_t v185_i1 = 0; v185_i1 < 13; ++v185_i1) {
              int32_t v191_a = v185_i1 * 32;
              int32_t v192_a = v190_lead + v191_a;
              float v200_data = glb_m0[(v197_lead + v191_a)];
              r6[(v184_i0 + v185_i1)] = v200_data;
            }
          }
          float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          float v203_lin = glb_m4[0 + threadIdx.x * 1];
          r7[0] = v203_lin;
          float v204_lin = glb_m4[32 + threadIdx.x * 1];
          r7[1] = v204_lin;
          float v205_lin = glb_m4[64 + threadIdx.x * 1];
          r7[2] = v205_lin;
          float v206_lin = glb_m4[96 + threadIdx.x * 1];
          r7[3] = v206_lin;
          float v207_lin = glb_m4[128 + threadIdx.x * 1];
          r7[4] = v207_lin;
          float v208_lin = glb_m4[160 + threadIdx.x * 1];
          r7[5] = v208_lin;
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v210_data = r6[0];
          float v211_data = r6[1];
          float v212_data = r6[2];
          float v213_data = r6[3];
          float v214_data = r6[4];
          float v215_data = r6[5];
          float v216_data = r6[6];
          float v217_data = r6[7];
          float v218_data = r6[8];
          float v219_data = r6[9];
          float v220_data = r6[10];
          float v221_data = r6[11];
          float v222_data = r6[12];
          float v223_acc{};
          float v224_acc{};
          float v225_acc{};
          float v226_acc{};
          float v227_acc{};
          float v228_acc{};
          float v229_acc{};
          float v230_acc{};
          float v231_acc{};
          float v232_acc{};
          float v233_acc{};
          float v234_acc{};
          float v235_acc{};
          float v236_lin = r7[0];
          float v237_bc = tensorforge::broadcast<32, 16, 0>(v236_lin);
          tensorforge::fmacdpp16<0>(v223_acc, v237_bc, v210_data);
          tensorforge::fmacdpp16<1>(v223_acc, v237_bc, v211_data);
          tensorforge::fmacdpp16<2>(v223_acc, v237_bc, v212_data);
          tensorforge::fmacdpp16<3>(v223_acc, v237_bc, v213_data);
          tensorforge::fmacdpp16<4>(v223_acc, v237_bc, v214_data);
          tensorforge::fmacdpp16<5>(v223_acc, v237_bc, v215_data);
          tensorforge::fmacdpp16<6>(v223_acc, v237_bc, v216_data);
          tensorforge::fmacdpp16<7>(v223_acc, v237_bc, v217_data);
          tensorforge::fmacdpp16<8>(v223_acc, v237_bc, v218_data);
          tensorforge::fmacdpp16<9>(v223_acc, v237_bc, v219_data);
          tensorforge::fmacdpp16<10>(v223_acc, v237_bc, v220_data);
          tensorforge::fmacdpp16<11>(v223_acc, v237_bc, v221_data);
          tensorforge::fmacdpp16<12>(v223_acc, v237_bc, v222_data);
          tensorforge::fmacdpp16<13>(v224_acc, v237_bc, v210_data);
          tensorforge::fmacdpp16<14>(v224_acc, v237_bc, v211_data);
          tensorforge::fmacdpp16<15>(v224_acc, v237_bc, v212_data);
          float v238_bc = tensorforge::broadcast<32, 16, 1>(v236_lin);
          tensorforge::fmacdpp16<0>(v224_acc, v238_bc, v213_data);
          tensorforge::fmacdpp16<1>(v224_acc, v238_bc, v214_data);
          tensorforge::fmacdpp16<2>(v224_acc, v238_bc, v215_data);
          tensorforge::fmacdpp16<3>(v224_acc, v238_bc, v216_data);
          tensorforge::fmacdpp16<4>(v224_acc, v238_bc, v217_data);
          tensorforge::fmacdpp16<5>(v224_acc, v238_bc, v218_data);
          tensorforge::fmacdpp16<6>(v224_acc, v238_bc, v219_data);
          tensorforge::fmacdpp16<7>(v224_acc, v238_bc, v220_data);
          tensorforge::fmacdpp16<8>(v224_acc, v238_bc, v221_data);
          tensorforge::fmacdpp16<9>(v224_acc, v238_bc, v222_data);
          tensorforge::fmacdpp16<10>(v225_acc, v238_bc, v210_data);
          tensorforge::fmacdpp16<11>(v225_acc, v238_bc, v211_data);
          tensorforge::fmacdpp16<12>(v225_acc, v238_bc, v212_data);
          tensorforge::fmacdpp16<13>(v225_acc, v238_bc, v213_data);
          tensorforge::fmacdpp16<14>(v225_acc, v238_bc, v214_data);
          tensorforge::fmacdpp16<15>(v225_acc, v238_bc, v215_data);
          float v239_lin = r7[1];
          float v240_bc = tensorforge::broadcast<32, 16, 0>(v239_lin);
          tensorforge::fmacdpp16<0>(v225_acc, v240_bc, v216_data);
          tensorforge::fmacdpp16<1>(v225_acc, v240_bc, v217_data);
          tensorforge::fmacdpp16<2>(v225_acc, v240_bc, v218_data);
          tensorforge::fmacdpp16<3>(v225_acc, v240_bc, v219_data);
          tensorforge::fmacdpp16<4>(v225_acc, v240_bc, v220_data);
          tensorforge::fmacdpp16<5>(v225_acc, v240_bc, v221_data);
          tensorforge::fmacdpp16<6>(v225_acc, v240_bc, v222_data);
          tensorforge::fmacdpp16<7>(v226_acc, v240_bc, v210_data);
          tensorforge::fmacdpp16<8>(v226_acc, v240_bc, v211_data);
          tensorforge::fmacdpp16<9>(v226_acc, v240_bc, v212_data);
          tensorforge::fmacdpp16<10>(v226_acc, v240_bc, v213_data);
          tensorforge::fmacdpp16<11>(v226_acc, v240_bc, v214_data);
          tensorforge::fmacdpp16<12>(v226_acc, v240_bc, v215_data);
          tensorforge::fmacdpp16<13>(v226_acc, v240_bc, v216_data);
          tensorforge::fmacdpp16<14>(v226_acc, v240_bc, v217_data);
          tensorforge::fmacdpp16<15>(v226_acc, v240_bc, v218_data);
          float v241_bc = tensorforge::broadcast<32, 16, 1>(v239_lin);
          tensorforge::fmacdpp16<0>(v226_acc, v241_bc, v219_data);
          tensorforge::fmacdpp16<1>(v226_acc, v241_bc, v220_data);
          tensorforge::fmacdpp16<2>(v226_acc, v241_bc, v221_data);
          tensorforge::fmacdpp16<3>(v226_acc, v241_bc, v222_data);
          tensorforge::fmacdpp16<4>(v227_acc, v241_bc, v210_data);
          tensorforge::fmacdpp16<5>(v227_acc, v241_bc, v211_data);
          tensorforge::fmacdpp16<6>(v227_acc, v241_bc, v212_data);
          tensorforge::fmacdpp16<7>(v227_acc, v241_bc, v213_data);
          tensorforge::fmacdpp16<8>(v227_acc, v241_bc, v214_data);
          tensorforge::fmacdpp16<9>(v227_acc, v241_bc, v215_data);
          tensorforge::fmacdpp16<10>(v227_acc, v241_bc, v216_data);
          tensorforge::fmacdpp16<11>(v227_acc, v241_bc, v217_data);
          tensorforge::fmacdpp16<12>(v227_acc, v241_bc, v218_data);
          tensorforge::fmacdpp16<13>(v227_acc, v241_bc, v219_data);
          tensorforge::fmacdpp16<14>(v227_acc, v241_bc, v220_data);
          tensorforge::fmacdpp16<15>(v227_acc, v241_bc, v221_data);
          float v242_lin = r7[2];
          float v243_bc = tensorforge::broadcast<32, 16, 0>(v242_lin);
          tensorforge::fmacdpp16<0>(v227_acc, v243_bc, v222_data);
          tensorforge::fmacdpp16<1>(v228_acc, v243_bc, v210_data);
          tensorforge::fmacdpp16<2>(v228_acc, v243_bc, v211_data);
          tensorforge::fmacdpp16<3>(v228_acc, v243_bc, v212_data);
          tensorforge::fmacdpp16<4>(v228_acc, v243_bc, v213_data);
          tensorforge::fmacdpp16<5>(v228_acc, v243_bc, v214_data);
          tensorforge::fmacdpp16<6>(v228_acc, v243_bc, v215_data);
          tensorforge::fmacdpp16<7>(v228_acc, v243_bc, v216_data);
          tensorforge::fmacdpp16<8>(v228_acc, v243_bc, v217_data);
          tensorforge::fmacdpp16<9>(v228_acc, v243_bc, v218_data);
          tensorforge::fmacdpp16<10>(v228_acc, v243_bc, v219_data);
          tensorforge::fmacdpp16<11>(v228_acc, v243_bc, v220_data);
          tensorforge::fmacdpp16<12>(v228_acc, v243_bc, v221_data);
          tensorforge::fmacdpp16<13>(v228_acc, v243_bc, v222_data);
          tensorforge::fmacdpp16<14>(v229_acc, v243_bc, v210_data);
          tensorforge::fmacdpp16<15>(v229_acc, v243_bc, v211_data);
          float v244_bc = tensorforge::broadcast<32, 16, 1>(v242_lin);
          tensorforge::fmacdpp16<0>(v229_acc, v244_bc, v212_data);
          tensorforge::fmacdpp16<1>(v229_acc, v244_bc, v213_data);
          tensorforge::fmacdpp16<2>(v229_acc, v244_bc, v214_data);
          tensorforge::fmacdpp16<3>(v229_acc, v244_bc, v215_data);
          tensorforge::fmacdpp16<4>(v229_acc, v244_bc, v216_data);
          tensorforge::fmacdpp16<5>(v229_acc, v244_bc, v217_data);
          tensorforge::fmacdpp16<6>(v229_acc, v244_bc, v218_data);
          tensorforge::fmacdpp16<7>(v229_acc, v244_bc, v219_data);
          tensorforge::fmacdpp16<8>(v229_acc, v244_bc, v220_data);
          tensorforge::fmacdpp16<9>(v229_acc, v244_bc, v221_data);
          tensorforge::fmacdpp16<10>(v229_acc, v244_bc, v222_data);
          tensorforge::fmacdpp16<11>(v230_acc, v244_bc, v210_data);
          tensorforge::fmacdpp16<12>(v230_acc, v244_bc, v211_data);
          tensorforge::fmacdpp16<13>(v230_acc, v244_bc, v212_data);
          tensorforge::fmacdpp16<14>(v230_acc, v244_bc, v213_data);
          tensorforge::fmacdpp16<15>(v230_acc, v244_bc, v214_data);
          float v245_lin = r7[3];
          float v246_bc = tensorforge::broadcast<32, 16, 0>(v245_lin);
          tensorforge::fmacdpp16<0>(v230_acc, v246_bc, v215_data);
          tensorforge::fmacdpp16<1>(v230_acc, v246_bc, v216_data);
          tensorforge::fmacdpp16<2>(v230_acc, v246_bc, v217_data);
          tensorforge::fmacdpp16<3>(v230_acc, v246_bc, v218_data);
          tensorforge::fmacdpp16<4>(v230_acc, v246_bc, v219_data);
          tensorforge::fmacdpp16<5>(v230_acc, v246_bc, v220_data);
          tensorforge::fmacdpp16<6>(v230_acc, v246_bc, v221_data);
          tensorforge::fmacdpp16<7>(v230_acc, v246_bc, v222_data);
          tensorforge::fmacdpp16<8>(v231_acc, v246_bc, v210_data);
          tensorforge::fmacdpp16<9>(v231_acc, v246_bc, v211_data);
          tensorforge::fmacdpp16<10>(v231_acc, v246_bc, v212_data);
          tensorforge::fmacdpp16<11>(v231_acc, v246_bc, v213_data);
          tensorforge::fmacdpp16<12>(v231_acc, v246_bc, v214_data);
          tensorforge::fmacdpp16<13>(v231_acc, v246_bc, v215_data);
          tensorforge::fmacdpp16<14>(v231_acc, v246_bc, v216_data);
          tensorforge::fmacdpp16<15>(v231_acc, v246_bc, v217_data);
          float v247_bc = tensorforge::broadcast<32, 16, 1>(v245_lin);
          tensorforge::fmacdpp16<0>(v231_acc, v247_bc, v218_data);
          tensorforge::fmacdpp16<1>(v231_acc, v247_bc, v219_data);
          tensorforge::fmacdpp16<2>(v231_acc, v247_bc, v220_data);
          tensorforge::fmacdpp16<3>(v231_acc, v247_bc, v221_data);
          tensorforge::fmacdpp16<4>(v231_acc, v247_bc, v222_data);
          tensorforge::fmacdpp16<5>(v232_acc, v247_bc, v210_data);
          tensorforge::fmacdpp16<6>(v232_acc, v247_bc, v211_data);
          tensorforge::fmacdpp16<7>(v232_acc, v247_bc, v212_data);
          tensorforge::fmacdpp16<8>(v232_acc, v247_bc, v213_data);
          tensorforge::fmacdpp16<9>(v232_acc, v247_bc, v214_data);
          tensorforge::fmacdpp16<10>(v232_acc, v247_bc, v215_data);
          tensorforge::fmacdpp16<11>(v232_acc, v247_bc, v216_data);
          tensorforge::fmacdpp16<12>(v232_acc, v247_bc, v217_data);
          tensorforge::fmacdpp16<13>(v232_acc, v247_bc, v218_data);
          tensorforge::fmacdpp16<14>(v232_acc, v247_bc, v219_data);
          tensorforge::fmacdpp16<15>(v232_acc, v247_bc, v220_data);
          float v248_lin = r7[4];
          float v249_bc = tensorforge::broadcast<32, 16, 0>(v248_lin);
          tensorforge::fmacdpp16<0>(v232_acc, v249_bc, v221_data);
          tensorforge::fmacdpp16<1>(v232_acc, v249_bc, v222_data);
          tensorforge::fmacdpp16<2>(v233_acc, v249_bc, v210_data);
          tensorforge::fmacdpp16<3>(v233_acc, v249_bc, v211_data);
          tensorforge::fmacdpp16<4>(v233_acc, v249_bc, v212_data);
          tensorforge::fmacdpp16<5>(v233_acc, v249_bc, v213_data);
          tensorforge::fmacdpp16<6>(v233_acc, v249_bc, v214_data);
          tensorforge::fmacdpp16<7>(v233_acc, v249_bc, v215_data);
          tensorforge::fmacdpp16<8>(v233_acc, v249_bc, v216_data);
          tensorforge::fmacdpp16<9>(v233_acc, v249_bc, v217_data);
          tensorforge::fmacdpp16<10>(v233_acc, v249_bc, v218_data);
          tensorforge::fmacdpp16<11>(v233_acc, v249_bc, v219_data);
          tensorforge::fmacdpp16<12>(v233_acc, v249_bc, v220_data);
          tensorforge::fmacdpp16<13>(v233_acc, v249_bc, v221_data);
          tensorforge::fmacdpp16<14>(v233_acc, v249_bc, v222_data);
          tensorforge::fmacdpp16<15>(v234_acc, v249_bc, v210_data);
          float v250_bc = tensorforge::broadcast<32, 16, 1>(v248_lin);
          tensorforge::fmacdpp16<0>(v234_acc, v250_bc, v211_data);
          tensorforge::fmacdpp16<1>(v234_acc, v250_bc, v212_data);
          tensorforge::fmacdpp16<2>(v234_acc, v250_bc, v213_data);
          tensorforge::fmacdpp16<3>(v234_acc, v250_bc, v214_data);
          tensorforge::fmacdpp16<4>(v234_acc, v250_bc, v215_data);
          tensorforge::fmacdpp16<5>(v234_acc, v250_bc, v216_data);
          tensorforge::fmacdpp16<6>(v234_acc, v250_bc, v217_data);
          tensorforge::fmacdpp16<7>(v234_acc, v250_bc, v218_data);
          tensorforge::fmacdpp16<8>(v234_acc, v250_bc, v219_data);
          tensorforge::fmacdpp16<9>(v234_acc, v250_bc, v220_data);
          tensorforge::fmacdpp16<10>(v234_acc, v250_bc, v221_data);
          tensorforge::fmacdpp16<11>(v234_acc, v250_bc, v222_data);
          tensorforge::fmacdpp16<12>(v235_acc, v250_bc, v210_data);
          tensorforge::fmacdpp16<13>(v235_acc, v250_bc, v211_data);
          tensorforge::fmacdpp16<14>(v235_acc, v250_bc, v212_data);
          tensorforge::fmacdpp16<15>(v235_acc, v250_bc, v213_data);
          float v251_lin = r7[5];
          float v252_bc = tensorforge::broadcast<32, 16, 0>(v251_lin);
          tensorforge::fmacdpp16<0>(v235_acc, v252_bc, v214_data);
          tensorforge::fmacdpp16<1>(v235_acc, v252_bc, v215_data);
          tensorforge::fmacdpp16<2>(v235_acc, v252_bc, v216_data);
          tensorforge::fmacdpp16<3>(v235_acc, v252_bc, v217_data);
          tensorforge::fmacdpp16<4>(v235_acc, v252_bc, v218_data);
          tensorforge::fmacdpp16<5>(v235_acc, v252_bc, v219_data);
          tensorforge::fmacdpp16<6>(v235_acc, v252_bc, v220_data);
          tensorforge::fmacdpp16<7>(v235_acc, v252_bc, v221_data);
          tensorforge::fmacdpp16<8>(v235_acc, v252_bc, v222_data);
          r8[0] = v223_acc;
          r8[1] = v224_acc;
          r8[2] = v225_acc;
          r8[3] = v226_acc;
          r8[4] = v227_acc;
          r8[5] = v228_acc;
          r8[6] = v229_acc;
          r8[7] = v230_acc;
          r8[8] = v231_acc;
          r8[9] = v232_acc;
          r8[10] = v233_acc;
          r8[11] = v234_acc;
          r8[12] = v235_acc;
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v256_i0 = 0; v256_i0 < 1; ++v256_i0) {
            int32_t v265_lead = v12_lead + (v256_i0 * 32);
            #pragma unroll
            for (int32_t v257_i1 = 0; v257_i1 < 13; ++v257_i1) {
              int32_t v258_a = v256_i0 + v257_i1;
              float v260_data = r8[(v256_i0 + v257_i1)];
              glb_m3[(v265_lead + (v257_i1 * 32))] = v260_data;
            }
          }
        }
      }
    }
  }
}

