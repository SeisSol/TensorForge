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
          // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir4[13]{};
          float v95_data = r2[0];
          float v96_data = r2[1];
          float v97_data = r2[2];
          float v98_data = r2[3];
          float v99_data = r2[4];
          float v100_data = r2[5];
          float v101_data = r2[6];
          float v102_data = r2[7];
          float v103_data = r2[8];
          float v104_data = r2[9];
          float v105_data = r2[10];
          float v106_data = r2[11];
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
          float v119_acc{};
          float v120_lin = r3[0];
          float v121_bc = tensorforge::broadcast<32, 16, 0>(v120_lin);
          tensorforge::fmacdpp16<0>(v107_acc, v121_bc, v95_data);
          tensorforge::fmacdpp16<1>(v107_acc, v121_bc, v96_data);
          tensorforge::fmacdpp16<2>(v107_acc, v121_bc, v97_data);
          tensorforge::fmacdpp16<3>(v107_acc, v121_bc, v98_data);
          tensorforge::fmacdpp16<4>(v107_acc, v121_bc, v99_data);
          tensorforge::fmacdpp16<5>(v107_acc, v121_bc, v100_data);
          tensorforge::fmacdpp16<6>(v107_acc, v121_bc, v101_data);
          tensorforge::fmacdpp16<7>(v107_acc, v121_bc, v102_data);
          tensorforge::fmacdpp16<8>(v107_acc, v121_bc, v103_data);
          tensorforge::fmacdpp16<9>(v107_acc, v121_bc, v104_data);
          tensorforge::fmacdpp16<10>(v107_acc, v121_bc, v105_data);
          tensorforge::fmacdpp16<11>(v107_acc, v121_bc, v106_data);
          tensorforge::fmacdpp16<12>(v108_acc, v121_bc, v95_data);
          tensorforge::fmacdpp16<13>(v108_acc, v121_bc, v96_data);
          tensorforge::fmacdpp16<14>(v108_acc, v121_bc, v97_data);
          tensorforge::fmacdpp16<15>(v108_acc, v121_bc, v98_data);
          float v122_bc = tensorforge::broadcast<32, 16, 1>(v120_lin);
          tensorforge::fmacdpp16<0>(v108_acc, v122_bc, v99_data);
          tensorforge::fmacdpp16<1>(v108_acc, v122_bc, v100_data);
          tensorforge::fmacdpp16<2>(v108_acc, v122_bc, v101_data);
          tensorforge::fmacdpp16<3>(v108_acc, v122_bc, v102_data);
          tensorforge::fmacdpp16<4>(v108_acc, v122_bc, v103_data);
          tensorforge::fmacdpp16<5>(v108_acc, v122_bc, v104_data);
          tensorforge::fmacdpp16<6>(v108_acc, v122_bc, v105_data);
          tensorforge::fmacdpp16<7>(v108_acc, v122_bc, v106_data);
          tensorforge::fmacdpp16<8>(v109_acc, v122_bc, v95_data);
          tensorforge::fmacdpp16<9>(v109_acc, v122_bc, v96_data);
          tensorforge::fmacdpp16<10>(v109_acc, v122_bc, v97_data);
          tensorforge::fmacdpp16<11>(v109_acc, v122_bc, v98_data);
          tensorforge::fmacdpp16<12>(v109_acc, v122_bc, v99_data);
          tensorforge::fmacdpp16<13>(v109_acc, v122_bc, v100_data);
          tensorforge::fmacdpp16<14>(v109_acc, v122_bc, v101_data);
          tensorforge::fmacdpp16<15>(v109_acc, v122_bc, v102_data);
          float v123_lin = r3[1];
          float v124_bc = tensorforge::broadcast<32, 16, 0>(v123_lin);
          tensorforge::fmacdpp16<0>(v109_acc, v124_bc, v103_data);
          tensorforge::fmacdpp16<1>(v109_acc, v124_bc, v104_data);
          tensorforge::fmacdpp16<2>(v109_acc, v124_bc, v105_data);
          tensorforge::fmacdpp16<3>(v109_acc, v124_bc, v106_data);
          tensorforge::fmacdpp16<4>(v110_acc, v124_bc, v95_data);
          tensorforge::fmacdpp16<5>(v110_acc, v124_bc, v96_data);
          tensorforge::fmacdpp16<6>(v110_acc, v124_bc, v97_data);
          tensorforge::fmacdpp16<7>(v110_acc, v124_bc, v98_data);
          tensorforge::fmacdpp16<8>(v110_acc, v124_bc, v99_data);
          tensorforge::fmacdpp16<9>(v110_acc, v124_bc, v100_data);
          tensorforge::fmacdpp16<10>(v110_acc, v124_bc, v101_data);
          tensorforge::fmacdpp16<11>(v110_acc, v124_bc, v102_data);
          tensorforge::fmacdpp16<12>(v110_acc, v124_bc, v103_data);
          tensorforge::fmacdpp16<13>(v110_acc, v124_bc, v104_data);
          tensorforge::fmacdpp16<14>(v110_acc, v124_bc, v105_data);
          tensorforge::fmacdpp16<15>(v110_acc, v124_bc, v106_data);
          float v125_bc = tensorforge::broadcast<32, 16, 1>(v123_lin);
          tensorforge::fmacdpp16<0>(v111_acc, v125_bc, v95_data);
          tensorforge::fmacdpp16<1>(v111_acc, v125_bc, v96_data);
          tensorforge::fmacdpp16<2>(v111_acc, v125_bc, v97_data);
          tensorforge::fmacdpp16<3>(v111_acc, v125_bc, v98_data);
          tensorforge::fmacdpp16<4>(v111_acc, v125_bc, v99_data);
          tensorforge::fmacdpp16<5>(v111_acc, v125_bc, v100_data);
          tensorforge::fmacdpp16<6>(v111_acc, v125_bc, v101_data);
          tensorforge::fmacdpp16<7>(v111_acc, v125_bc, v102_data);
          tensorforge::fmacdpp16<8>(v111_acc, v125_bc, v103_data);
          tensorforge::fmacdpp16<9>(v111_acc, v125_bc, v104_data);
          tensorforge::fmacdpp16<10>(v111_acc, v125_bc, v105_data);
          tensorforge::fmacdpp16<11>(v111_acc, v125_bc, v106_data);
          tensorforge::fmacdpp16<12>(v112_acc, v125_bc, v95_data);
          tensorforge::fmacdpp16<13>(v112_acc, v125_bc, v96_data);
          tensorforge::fmacdpp16<14>(v112_acc, v125_bc, v97_data);
          tensorforge::fmacdpp16<15>(v112_acc, v125_bc, v98_data);
          float v126_lin = r3[2];
          float v127_bc = tensorforge::broadcast<32, 16, 0>(v126_lin);
          tensorforge::fmacdpp16<0>(v112_acc, v127_bc, v99_data);
          tensorforge::fmacdpp16<1>(v112_acc, v127_bc, v100_data);
          tensorforge::fmacdpp16<2>(v112_acc, v127_bc, v101_data);
          tensorforge::fmacdpp16<3>(v112_acc, v127_bc, v102_data);
          tensorforge::fmacdpp16<4>(v112_acc, v127_bc, v103_data);
          tensorforge::fmacdpp16<5>(v112_acc, v127_bc, v104_data);
          tensorforge::fmacdpp16<6>(v112_acc, v127_bc, v105_data);
          tensorforge::fmacdpp16<7>(v112_acc, v127_bc, v106_data);
          tensorforge::fmacdpp16<8>(v113_acc, v127_bc, v95_data);
          tensorforge::fmacdpp16<9>(v113_acc, v127_bc, v96_data);
          tensorforge::fmacdpp16<10>(v113_acc, v127_bc, v97_data);
          tensorforge::fmacdpp16<11>(v113_acc, v127_bc, v98_data);
          tensorforge::fmacdpp16<12>(v113_acc, v127_bc, v99_data);
          tensorforge::fmacdpp16<13>(v113_acc, v127_bc, v100_data);
          tensorforge::fmacdpp16<14>(v113_acc, v127_bc, v101_data);
          tensorforge::fmacdpp16<15>(v113_acc, v127_bc, v102_data);
          float v128_bc = tensorforge::broadcast<32, 16, 1>(v126_lin);
          tensorforge::fmacdpp16<0>(v113_acc, v128_bc, v103_data);
          tensorforge::fmacdpp16<1>(v113_acc, v128_bc, v104_data);
          tensorforge::fmacdpp16<2>(v113_acc, v128_bc, v105_data);
          tensorforge::fmacdpp16<3>(v113_acc, v128_bc, v106_data);
          tensorforge::fmacdpp16<4>(v114_acc, v128_bc, v95_data);
          tensorforge::fmacdpp16<5>(v114_acc, v128_bc, v96_data);
          tensorforge::fmacdpp16<6>(v114_acc, v128_bc, v97_data);
          tensorforge::fmacdpp16<7>(v114_acc, v128_bc, v98_data);
          tensorforge::fmacdpp16<8>(v114_acc, v128_bc, v99_data);
          tensorforge::fmacdpp16<9>(v114_acc, v128_bc, v100_data);
          tensorforge::fmacdpp16<10>(v114_acc, v128_bc, v101_data);
          tensorforge::fmacdpp16<11>(v114_acc, v128_bc, v102_data);
          tensorforge::fmacdpp16<12>(v114_acc, v128_bc, v103_data);
          tensorforge::fmacdpp16<13>(v114_acc, v128_bc, v104_data);
          tensorforge::fmacdpp16<14>(v114_acc, v128_bc, v105_data);
          tensorforge::fmacdpp16<15>(v114_acc, v128_bc, v106_data);
          float v129_lin = r3[3];
          float v130_bc = tensorforge::broadcast<32, 16, 0>(v129_lin);
          tensorforge::fmacdpp16<0>(v115_acc, v130_bc, v95_data);
          tensorforge::fmacdpp16<1>(v115_acc, v130_bc, v96_data);
          tensorforge::fmacdpp16<2>(v115_acc, v130_bc, v97_data);
          tensorforge::fmacdpp16<3>(v115_acc, v130_bc, v98_data);
          tensorforge::fmacdpp16<4>(v115_acc, v130_bc, v99_data);
          tensorforge::fmacdpp16<5>(v115_acc, v130_bc, v100_data);
          tensorforge::fmacdpp16<6>(v115_acc, v130_bc, v101_data);
          tensorforge::fmacdpp16<7>(v115_acc, v130_bc, v102_data);
          tensorforge::fmacdpp16<8>(v115_acc, v130_bc, v103_data);
          tensorforge::fmacdpp16<9>(v115_acc, v130_bc, v104_data);
          tensorforge::fmacdpp16<10>(v115_acc, v130_bc, v105_data);
          tensorforge::fmacdpp16<11>(v115_acc, v130_bc, v106_data);
          tensorforge::fmacdpp16<12>(v116_acc, v130_bc, v95_data);
          tensorforge::fmacdpp16<13>(v116_acc, v130_bc, v96_data);
          tensorforge::fmacdpp16<14>(v116_acc, v130_bc, v97_data);
          tensorforge::fmacdpp16<15>(v116_acc, v130_bc, v98_data);
          float v131_bc = tensorforge::broadcast<32, 16, 1>(v129_lin);
          tensorforge::fmacdpp16<0>(v116_acc, v131_bc, v99_data);
          tensorforge::fmacdpp16<1>(v116_acc, v131_bc, v100_data);
          tensorforge::fmacdpp16<2>(v116_acc, v131_bc, v101_data);
          tensorforge::fmacdpp16<3>(v116_acc, v131_bc, v102_data);
          tensorforge::fmacdpp16<4>(v116_acc, v131_bc, v103_data);
          tensorforge::fmacdpp16<5>(v116_acc, v131_bc, v104_data);
          tensorforge::fmacdpp16<6>(v116_acc, v131_bc, v105_data);
          tensorforge::fmacdpp16<7>(v116_acc, v131_bc, v106_data);
          tensorforge::fmacdpp16<8>(v117_acc, v131_bc, v95_data);
          tensorforge::fmacdpp16<9>(v117_acc, v131_bc, v96_data);
          tensorforge::fmacdpp16<10>(v117_acc, v131_bc, v97_data);
          tensorforge::fmacdpp16<11>(v117_acc, v131_bc, v98_data);
          tensorforge::fmacdpp16<12>(v117_acc, v131_bc, v99_data);
          tensorforge::fmacdpp16<13>(v117_acc, v131_bc, v100_data);
          tensorforge::fmacdpp16<14>(v117_acc, v131_bc, v101_data);
          tensorforge::fmacdpp16<15>(v117_acc, v131_bc, v102_data);
          float v132_lin = r3[4];
          float v133_bc = tensorforge::broadcast<32, 16, 0>(v132_lin);
          tensorforge::fmacdpp16<0>(v117_acc, v133_bc, v103_data);
          tensorforge::fmacdpp16<1>(v117_acc, v133_bc, v104_data);
          tensorforge::fmacdpp16<2>(v117_acc, v133_bc, v105_data);
          tensorforge::fmacdpp16<3>(v117_acc, v133_bc, v106_data);
          tensorforge::fmacdpp16<4>(v118_acc, v133_bc, v95_data);
          tensorforge::fmacdpp16<5>(v118_acc, v133_bc, v96_data);
          tensorforge::fmacdpp16<6>(v118_acc, v133_bc, v97_data);
          tensorforge::fmacdpp16<7>(v118_acc, v133_bc, v98_data);
          tensorforge::fmacdpp16<8>(v118_acc, v133_bc, v99_data);
          tensorforge::fmacdpp16<9>(v118_acc, v133_bc, v100_data);
          tensorforge::fmacdpp16<10>(v118_acc, v133_bc, v101_data);
          tensorforge::fmacdpp16<11>(v118_acc, v133_bc, v102_data);
          tensorforge::fmacdpp16<12>(v118_acc, v133_bc, v103_data);
          tensorforge::fmacdpp16<13>(v118_acc, v133_bc, v104_data);
          tensorforge::fmacdpp16<14>(v118_acc, v133_bc, v105_data);
          tensorforge::fmacdpp16<15>(v118_acc, v133_bc, v106_data);
          float v134_bc = tensorforge::broadcast<32, 16, 1>(v132_lin);
          tensorforge::fmacdpp16<0>(v119_acc, v134_bc, v95_data);
          tensorforge::fmacdpp16<1>(v119_acc, v134_bc, v96_data);
          tensorforge::fmacdpp16<2>(v119_acc, v134_bc, v97_data);
          tensorforge::fmacdpp16<3>(v119_acc, v134_bc, v98_data);
          tensorforge::fmacdpp16<4>(v119_acc, v134_bc, v99_data);
          tensorforge::fmacdpp16<5>(v119_acc, v134_bc, v100_data);
          tensorforge::fmacdpp16<6>(v119_acc, v134_bc, v101_data);
          tensorforge::fmacdpp16<7>(v119_acc, v134_bc, v102_data);
          tensorforge::fmacdpp16<8>(v119_acc, v134_bc, v103_data);
          tensorforge::fmacdpp16<9>(v119_acc, v134_bc, v104_data);
          tensorforge::fmacdpp16<10>(v119_acc, v134_bc, v105_data);
          tensorforge::fmacdpp16<11>(v119_acc, v134_bc, v106_data);
          ir4[0] = v107_acc;
          ir4[1] = v108_acc;
          ir4[2] = v109_acc;
          ir4[3] = v110_acc;
          ir4[4] = v111_acc;
          ir4[5] = v112_acc;
          ir4[6] = v113_acc;
          ir4[7] = v114_acc;
          ir4[8] = v115_acc;
          ir4[9] = v116_acc;
          ir4[10] = v117_acc;
          ir4[11] = v118_acc;
          ir4[12] = v119_acc;
          #pragma unroll
          for (int32_t v138_n0 = 0; v138_n0 < 1; ++v138_n0) {
            #pragma unroll
            for (int32_t v139_n1 = 0; v139_n1 < 13; ++v139_n1) {
              int32_t v140_a = v138_n0 + v139_n1;
              int32_t v141_a = v138_n0 + v139_n1;
              float v142_data = ir4[v141_a];
              int32_t v143_a = v138_n0 + v139_n1;
              float v145_data = r1[v141_a];
              int32_t v147_a = v138_n0 + v139_n1;
              r4[v141_a] = (v145_data + v142_data);
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          auto& ir5 = r5;
          float v153_data = r4[4];
          float v154_data = ir5[0];
          ir5[0] = (v154_data + v153_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v159_i0 = 0; v159_i0 < 1; ++v159_i0) {
            int32_t v168_lead = v3_lead + (v159_i0 * 32);
            #pragma unroll
            for (int32_t v160_i1 = 0; v160_i1 < 1; ++v160_i1) {
              int32_t v161_a = v159_i0 + v160_i1;
              float v163_data = r5[(v159_i0 + v160_i1)];
              int32_t v171_a = v168_lead + ((v160_i1 + 4) * 32);
              glb_m0[v171_a] = v163_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v176_i0 = 0; v176_i0 < 1; ++v176_i0) {
            int32_t v181_lead = v176_i0 * 32;
            int32_t v182_lead = v3_lead + v181_lead;
            int32_t v189_lead = v3_lead + v181_lead;
            #pragma unroll
            for (int32_t v177_i1 = 0; v177_i1 < 13; ++v177_i1) {
              int32_t v183_a = v177_i1 * 32;
              int32_t v184_a = v182_lead + v183_a;
              float v192_data = glb_m0[(v189_lead + v183_a)];
              int32_t v193_a = v176_i0 + v177_i1;
              r6[v193_a] = v192_data;
            }
          }
          float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          float v195_lin = glb_m4[0 + threadIdx.x * 1];
          r7[0] = v195_lin;
          float v196_lin = glb_m4[32 + threadIdx.x * 1];
          r7[1] = v196_lin;
          float v197_lin = glb_m4[64 + threadIdx.x * 1];
          r7[2] = v197_lin;
          float v198_lin = glb_m4[96 + threadIdx.x * 1];
          r7[3] = v198_lin;
          float v199_lin = glb_m4[128 + threadIdx.x * 1];
          r7[4] = v199_lin;
          float v200_lin = glb_m4[160 + threadIdx.x * 1];
          r7[5] = v200_lin;
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          auto& ir8 = r8;
          float v202_data = r6[0];
          float v203_data = r6[1];
          float v204_data = r6[2];
          float v205_data = r6[3];
          float v206_data = r6[4];
          float v207_data = r6[5];
          float v208_data = r6[6];
          float v209_data = r6[7];
          float v210_data = r6[8];
          float v211_data = r6[9];
          float v212_data = r6[10];
          float v213_data = r6[11];
          float v214_data = r6[12];
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
          float v227_acc{};
          float v228_lin = r7[0];
          float v229_bc = tensorforge::broadcast<32, 16, 0>(v228_lin);
          tensorforge::fmacdpp16<0>(v215_acc, v229_bc, v202_data);
          tensorforge::fmacdpp16<1>(v215_acc, v229_bc, v203_data);
          tensorforge::fmacdpp16<2>(v215_acc, v229_bc, v204_data);
          tensorforge::fmacdpp16<3>(v215_acc, v229_bc, v205_data);
          tensorforge::fmacdpp16<4>(v215_acc, v229_bc, v206_data);
          tensorforge::fmacdpp16<5>(v215_acc, v229_bc, v207_data);
          tensorforge::fmacdpp16<6>(v215_acc, v229_bc, v208_data);
          tensorforge::fmacdpp16<7>(v215_acc, v229_bc, v209_data);
          tensorforge::fmacdpp16<8>(v215_acc, v229_bc, v210_data);
          tensorforge::fmacdpp16<9>(v215_acc, v229_bc, v211_data);
          tensorforge::fmacdpp16<10>(v215_acc, v229_bc, v212_data);
          tensorforge::fmacdpp16<11>(v215_acc, v229_bc, v213_data);
          tensorforge::fmacdpp16<12>(v215_acc, v229_bc, v214_data);
          tensorforge::fmacdpp16<13>(v216_acc, v229_bc, v202_data);
          tensorforge::fmacdpp16<14>(v216_acc, v229_bc, v203_data);
          tensorforge::fmacdpp16<15>(v216_acc, v229_bc, v204_data);
          float v230_bc = tensorforge::broadcast<32, 16, 1>(v228_lin);
          tensorforge::fmacdpp16<0>(v216_acc, v230_bc, v205_data);
          tensorforge::fmacdpp16<1>(v216_acc, v230_bc, v206_data);
          tensorforge::fmacdpp16<2>(v216_acc, v230_bc, v207_data);
          tensorforge::fmacdpp16<3>(v216_acc, v230_bc, v208_data);
          tensorforge::fmacdpp16<4>(v216_acc, v230_bc, v209_data);
          tensorforge::fmacdpp16<5>(v216_acc, v230_bc, v210_data);
          tensorforge::fmacdpp16<6>(v216_acc, v230_bc, v211_data);
          tensorforge::fmacdpp16<7>(v216_acc, v230_bc, v212_data);
          tensorforge::fmacdpp16<8>(v216_acc, v230_bc, v213_data);
          tensorforge::fmacdpp16<9>(v216_acc, v230_bc, v214_data);
          tensorforge::fmacdpp16<10>(v217_acc, v230_bc, v202_data);
          tensorforge::fmacdpp16<11>(v217_acc, v230_bc, v203_data);
          tensorforge::fmacdpp16<12>(v217_acc, v230_bc, v204_data);
          tensorforge::fmacdpp16<13>(v217_acc, v230_bc, v205_data);
          tensorforge::fmacdpp16<14>(v217_acc, v230_bc, v206_data);
          tensorforge::fmacdpp16<15>(v217_acc, v230_bc, v207_data);
          float v231_lin = r7[1];
          float v232_bc = tensorforge::broadcast<32, 16, 0>(v231_lin);
          tensorforge::fmacdpp16<0>(v217_acc, v232_bc, v208_data);
          tensorforge::fmacdpp16<1>(v217_acc, v232_bc, v209_data);
          tensorforge::fmacdpp16<2>(v217_acc, v232_bc, v210_data);
          tensorforge::fmacdpp16<3>(v217_acc, v232_bc, v211_data);
          tensorforge::fmacdpp16<4>(v217_acc, v232_bc, v212_data);
          tensorforge::fmacdpp16<5>(v217_acc, v232_bc, v213_data);
          tensorforge::fmacdpp16<6>(v217_acc, v232_bc, v214_data);
          tensorforge::fmacdpp16<7>(v218_acc, v232_bc, v202_data);
          tensorforge::fmacdpp16<8>(v218_acc, v232_bc, v203_data);
          tensorforge::fmacdpp16<9>(v218_acc, v232_bc, v204_data);
          tensorforge::fmacdpp16<10>(v218_acc, v232_bc, v205_data);
          tensorforge::fmacdpp16<11>(v218_acc, v232_bc, v206_data);
          tensorforge::fmacdpp16<12>(v218_acc, v232_bc, v207_data);
          tensorforge::fmacdpp16<13>(v218_acc, v232_bc, v208_data);
          tensorforge::fmacdpp16<14>(v218_acc, v232_bc, v209_data);
          tensorforge::fmacdpp16<15>(v218_acc, v232_bc, v210_data);
          float v233_bc = tensorforge::broadcast<32, 16, 1>(v231_lin);
          tensorforge::fmacdpp16<0>(v218_acc, v233_bc, v211_data);
          tensorforge::fmacdpp16<1>(v218_acc, v233_bc, v212_data);
          tensorforge::fmacdpp16<2>(v218_acc, v233_bc, v213_data);
          tensorforge::fmacdpp16<3>(v218_acc, v233_bc, v214_data);
          tensorforge::fmacdpp16<4>(v219_acc, v233_bc, v202_data);
          tensorforge::fmacdpp16<5>(v219_acc, v233_bc, v203_data);
          tensorforge::fmacdpp16<6>(v219_acc, v233_bc, v204_data);
          tensorforge::fmacdpp16<7>(v219_acc, v233_bc, v205_data);
          tensorforge::fmacdpp16<8>(v219_acc, v233_bc, v206_data);
          tensorforge::fmacdpp16<9>(v219_acc, v233_bc, v207_data);
          tensorforge::fmacdpp16<10>(v219_acc, v233_bc, v208_data);
          tensorforge::fmacdpp16<11>(v219_acc, v233_bc, v209_data);
          tensorforge::fmacdpp16<12>(v219_acc, v233_bc, v210_data);
          tensorforge::fmacdpp16<13>(v219_acc, v233_bc, v211_data);
          tensorforge::fmacdpp16<14>(v219_acc, v233_bc, v212_data);
          tensorforge::fmacdpp16<15>(v219_acc, v233_bc, v213_data);
          float v234_lin = r7[2];
          float v235_bc = tensorforge::broadcast<32, 16, 0>(v234_lin);
          tensorforge::fmacdpp16<0>(v219_acc, v235_bc, v214_data);
          tensorforge::fmacdpp16<1>(v220_acc, v235_bc, v202_data);
          tensorforge::fmacdpp16<2>(v220_acc, v235_bc, v203_data);
          tensorforge::fmacdpp16<3>(v220_acc, v235_bc, v204_data);
          tensorforge::fmacdpp16<4>(v220_acc, v235_bc, v205_data);
          tensorforge::fmacdpp16<5>(v220_acc, v235_bc, v206_data);
          tensorforge::fmacdpp16<6>(v220_acc, v235_bc, v207_data);
          tensorforge::fmacdpp16<7>(v220_acc, v235_bc, v208_data);
          tensorforge::fmacdpp16<8>(v220_acc, v235_bc, v209_data);
          tensorforge::fmacdpp16<9>(v220_acc, v235_bc, v210_data);
          tensorforge::fmacdpp16<10>(v220_acc, v235_bc, v211_data);
          tensorforge::fmacdpp16<11>(v220_acc, v235_bc, v212_data);
          tensorforge::fmacdpp16<12>(v220_acc, v235_bc, v213_data);
          tensorforge::fmacdpp16<13>(v220_acc, v235_bc, v214_data);
          tensorforge::fmacdpp16<14>(v221_acc, v235_bc, v202_data);
          tensorforge::fmacdpp16<15>(v221_acc, v235_bc, v203_data);
          float v236_bc = tensorforge::broadcast<32, 16, 1>(v234_lin);
          tensorforge::fmacdpp16<0>(v221_acc, v236_bc, v204_data);
          tensorforge::fmacdpp16<1>(v221_acc, v236_bc, v205_data);
          tensorforge::fmacdpp16<2>(v221_acc, v236_bc, v206_data);
          tensorforge::fmacdpp16<3>(v221_acc, v236_bc, v207_data);
          tensorforge::fmacdpp16<4>(v221_acc, v236_bc, v208_data);
          tensorforge::fmacdpp16<5>(v221_acc, v236_bc, v209_data);
          tensorforge::fmacdpp16<6>(v221_acc, v236_bc, v210_data);
          tensorforge::fmacdpp16<7>(v221_acc, v236_bc, v211_data);
          tensorforge::fmacdpp16<8>(v221_acc, v236_bc, v212_data);
          tensorforge::fmacdpp16<9>(v221_acc, v236_bc, v213_data);
          tensorforge::fmacdpp16<10>(v221_acc, v236_bc, v214_data);
          tensorforge::fmacdpp16<11>(v222_acc, v236_bc, v202_data);
          tensorforge::fmacdpp16<12>(v222_acc, v236_bc, v203_data);
          tensorforge::fmacdpp16<13>(v222_acc, v236_bc, v204_data);
          tensorforge::fmacdpp16<14>(v222_acc, v236_bc, v205_data);
          tensorforge::fmacdpp16<15>(v222_acc, v236_bc, v206_data);
          float v237_lin = r7[3];
          float v238_bc = tensorforge::broadcast<32, 16, 0>(v237_lin);
          tensorforge::fmacdpp16<0>(v222_acc, v238_bc, v207_data);
          tensorforge::fmacdpp16<1>(v222_acc, v238_bc, v208_data);
          tensorforge::fmacdpp16<2>(v222_acc, v238_bc, v209_data);
          tensorforge::fmacdpp16<3>(v222_acc, v238_bc, v210_data);
          tensorforge::fmacdpp16<4>(v222_acc, v238_bc, v211_data);
          tensorforge::fmacdpp16<5>(v222_acc, v238_bc, v212_data);
          tensorforge::fmacdpp16<6>(v222_acc, v238_bc, v213_data);
          tensorforge::fmacdpp16<7>(v222_acc, v238_bc, v214_data);
          tensorforge::fmacdpp16<8>(v223_acc, v238_bc, v202_data);
          tensorforge::fmacdpp16<9>(v223_acc, v238_bc, v203_data);
          tensorforge::fmacdpp16<10>(v223_acc, v238_bc, v204_data);
          tensorforge::fmacdpp16<11>(v223_acc, v238_bc, v205_data);
          tensorforge::fmacdpp16<12>(v223_acc, v238_bc, v206_data);
          tensorforge::fmacdpp16<13>(v223_acc, v238_bc, v207_data);
          tensorforge::fmacdpp16<14>(v223_acc, v238_bc, v208_data);
          tensorforge::fmacdpp16<15>(v223_acc, v238_bc, v209_data);
          float v239_bc = tensorforge::broadcast<32, 16, 1>(v237_lin);
          tensorforge::fmacdpp16<0>(v223_acc, v239_bc, v210_data);
          tensorforge::fmacdpp16<1>(v223_acc, v239_bc, v211_data);
          tensorforge::fmacdpp16<2>(v223_acc, v239_bc, v212_data);
          tensorforge::fmacdpp16<3>(v223_acc, v239_bc, v213_data);
          tensorforge::fmacdpp16<4>(v223_acc, v239_bc, v214_data);
          tensorforge::fmacdpp16<5>(v224_acc, v239_bc, v202_data);
          tensorforge::fmacdpp16<6>(v224_acc, v239_bc, v203_data);
          tensorforge::fmacdpp16<7>(v224_acc, v239_bc, v204_data);
          tensorforge::fmacdpp16<8>(v224_acc, v239_bc, v205_data);
          tensorforge::fmacdpp16<9>(v224_acc, v239_bc, v206_data);
          tensorforge::fmacdpp16<10>(v224_acc, v239_bc, v207_data);
          tensorforge::fmacdpp16<11>(v224_acc, v239_bc, v208_data);
          tensorforge::fmacdpp16<12>(v224_acc, v239_bc, v209_data);
          tensorforge::fmacdpp16<13>(v224_acc, v239_bc, v210_data);
          tensorforge::fmacdpp16<14>(v224_acc, v239_bc, v211_data);
          tensorforge::fmacdpp16<15>(v224_acc, v239_bc, v212_data);
          float v240_lin = r7[4];
          float v241_bc = tensorforge::broadcast<32, 16, 0>(v240_lin);
          tensorforge::fmacdpp16<0>(v224_acc, v241_bc, v213_data);
          tensorforge::fmacdpp16<1>(v224_acc, v241_bc, v214_data);
          tensorforge::fmacdpp16<2>(v225_acc, v241_bc, v202_data);
          tensorforge::fmacdpp16<3>(v225_acc, v241_bc, v203_data);
          tensorforge::fmacdpp16<4>(v225_acc, v241_bc, v204_data);
          tensorforge::fmacdpp16<5>(v225_acc, v241_bc, v205_data);
          tensorforge::fmacdpp16<6>(v225_acc, v241_bc, v206_data);
          tensorforge::fmacdpp16<7>(v225_acc, v241_bc, v207_data);
          tensorforge::fmacdpp16<8>(v225_acc, v241_bc, v208_data);
          tensorforge::fmacdpp16<9>(v225_acc, v241_bc, v209_data);
          tensorforge::fmacdpp16<10>(v225_acc, v241_bc, v210_data);
          tensorforge::fmacdpp16<11>(v225_acc, v241_bc, v211_data);
          tensorforge::fmacdpp16<12>(v225_acc, v241_bc, v212_data);
          tensorforge::fmacdpp16<13>(v225_acc, v241_bc, v213_data);
          tensorforge::fmacdpp16<14>(v225_acc, v241_bc, v214_data);
          tensorforge::fmacdpp16<15>(v226_acc, v241_bc, v202_data);
          float v242_bc = tensorforge::broadcast<32, 16, 1>(v240_lin);
          tensorforge::fmacdpp16<0>(v226_acc, v242_bc, v203_data);
          tensorforge::fmacdpp16<1>(v226_acc, v242_bc, v204_data);
          tensorforge::fmacdpp16<2>(v226_acc, v242_bc, v205_data);
          tensorforge::fmacdpp16<3>(v226_acc, v242_bc, v206_data);
          tensorforge::fmacdpp16<4>(v226_acc, v242_bc, v207_data);
          tensorforge::fmacdpp16<5>(v226_acc, v242_bc, v208_data);
          tensorforge::fmacdpp16<6>(v226_acc, v242_bc, v209_data);
          tensorforge::fmacdpp16<7>(v226_acc, v242_bc, v210_data);
          tensorforge::fmacdpp16<8>(v226_acc, v242_bc, v211_data);
          tensorforge::fmacdpp16<9>(v226_acc, v242_bc, v212_data);
          tensorforge::fmacdpp16<10>(v226_acc, v242_bc, v213_data);
          tensorforge::fmacdpp16<11>(v226_acc, v242_bc, v214_data);
          tensorforge::fmacdpp16<12>(v227_acc, v242_bc, v202_data);
          tensorforge::fmacdpp16<13>(v227_acc, v242_bc, v203_data);
          tensorforge::fmacdpp16<14>(v227_acc, v242_bc, v204_data);
          tensorforge::fmacdpp16<15>(v227_acc, v242_bc, v205_data);
          float v243_lin = r7[5];
          float v244_bc = tensorforge::broadcast<32, 16, 0>(v243_lin);
          tensorforge::fmacdpp16<0>(v227_acc, v244_bc, v206_data);
          tensorforge::fmacdpp16<1>(v227_acc, v244_bc, v207_data);
          tensorforge::fmacdpp16<2>(v227_acc, v244_bc, v208_data);
          tensorforge::fmacdpp16<3>(v227_acc, v244_bc, v209_data);
          tensorforge::fmacdpp16<4>(v227_acc, v244_bc, v210_data);
          tensorforge::fmacdpp16<5>(v227_acc, v244_bc, v211_data);
          tensorforge::fmacdpp16<6>(v227_acc, v244_bc, v212_data);
          tensorforge::fmacdpp16<7>(v227_acc, v244_bc, v213_data);
          tensorforge::fmacdpp16<8>(v227_acc, v244_bc, v214_data);
          ir8[0] = v215_acc;
          ir8[1] = v216_acc;
          ir8[2] = v217_acc;
          ir8[3] = v218_acc;
          ir8[4] = v219_acc;
          ir8[5] = v220_acc;
          ir8[6] = v221_acc;
          ir8[7] = v222_acc;
          ir8[8] = v223_acc;
          ir8[9] = v224_acc;
          ir8[10] = v225_acc;
          ir8[11] = v226_acc;
          ir8[12] = v227_acc;
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v248_i0 = 0; v248_i0 < 1; ++v248_i0) {
            int32_t v257_lead = v3_lead + (v248_i0 * 32);
            #pragma unroll
            for (int32_t v249_i1 = 0; v249_i1 < 13; ++v249_i1) {
              int32_t v250_a = v248_i0 + v249_i1;
              float v252_data = r8[(v248_i0 + v249_i1)];
              int32_t v259_a = v257_lead + (v249_i1 * 32);
              glb_m3[v259_a] = v252_data;
            }
          }
          ;
        }
      }
    }
  }
}

