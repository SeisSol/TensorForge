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
            int32_t v19_lead = v12_lead + (v13_i0 * 32);
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 13; ++v14_i1) {
              float v22_data = glb_m0[(v19_lead + (v14_i1 * 32))];
              r0[(v13_i0 + v14_i1)] = v22_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
            int32_t v34_lead = v12_lead + (v28_i0 * 32);
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m1[(v34_lead + (v29_i1 * 32))]);
              r2[(v28_i0 + v29_i1)] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v43_data = r0[0];
          float v44_data = r1[0];
          r1[0] = (v44_data + v43_data);
          float v46_data = r0[1];
          float v47_data = r1[1];
          r1[1] = (v47_data + v46_data);
          float v49_data = r0[2];
          float v50_data = r1[2];
          r1[2] = (v50_data + v49_data);
          float v52_data = r0[3];
          float v53_data = r1[3];
          r1[3] = (v53_data + v52_data);
          float v55_data = r0[4];
          float v56_data = r1[4];
          r1[4] = (v56_data + v55_data);
          float v58_data = r0[5];
          float v59_data = r1[5];
          r1[5] = (v59_data + v58_data);
          float v61_data = r0[6];
          float v62_data = r1[6];
          r1[6] = (v62_data + v61_data);
          float v64_data = r0[7];
          float v65_data = r1[7];
          r1[7] = (v65_data + v64_data);
          float v67_data = r0[8];
          float v68_data = r1[8];
          r1[8] = (v68_data + v67_data);
          float v70_data = r0[9];
          float v71_data = r1[9];
          r1[9] = (v71_data + v70_data);
          float v73_data = r0[10];
          float v74_data = r1[10];
          r1[10] = (v74_data + v73_data);
          float v76_data = r0[11];
          float v77_data = r1[11];
          r1[11] = (v77_data + v76_data);
          float v79_data = r0[12];
          float v80_data = r1[12];
          r1[12] = (v80_data + v79_data);
          float r3[13]{};
          // r3 = load{g>r}(glb_m2);
          float v83_lin = glb_m2[0 + threadIdx.x * 1];
          r3[0] = v83_lin;
          float v84_lin = glb_m2[32 + threadIdx.x * 1];
          r3[1] = v84_lin;
          float v85_lin = glb_m2[64 + threadIdx.x * 1];
          r3[2] = v85_lin;
          float v86_lin = glb_m2[96 + threadIdx.x * 1];
          r3[3] = v86_lin;
          float v87_lin = glb_m2[128 + threadIdx.x * 1];
          r3[4] = v87_lin;
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir4[13]{};
          float v90_data = r2[0];
          float v91_data = r2[1];
          float v92_data = r2[2];
          float v93_data = r2[3];
          float v94_data = r2[4];
          float v95_data = r2[5];
          float v96_data = r2[6];
          float v97_data = r2[7];
          float v98_data = r2[8];
          float v99_data = r2[9];
          float v100_data = r2[10];
          float v101_data = r2[11];
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
          float v115_lin = r3[0];
          float v116_bc = tensorforge::broadcast<32, 16, 0>(v115_lin);
          tensorforge::fmacdpp16<0>(v102_acc, v116_bc, v90_data);
          tensorforge::fmacdpp16<1>(v102_acc, v116_bc, v91_data);
          tensorforge::fmacdpp16<2>(v102_acc, v116_bc, v92_data);
          tensorforge::fmacdpp16<3>(v102_acc, v116_bc, v93_data);
          tensorforge::fmacdpp16<4>(v102_acc, v116_bc, v94_data);
          tensorforge::fmacdpp16<5>(v102_acc, v116_bc, v95_data);
          tensorforge::fmacdpp16<6>(v102_acc, v116_bc, v96_data);
          tensorforge::fmacdpp16<7>(v102_acc, v116_bc, v97_data);
          tensorforge::fmacdpp16<8>(v102_acc, v116_bc, v98_data);
          tensorforge::fmacdpp16<9>(v102_acc, v116_bc, v99_data);
          tensorforge::fmacdpp16<10>(v102_acc, v116_bc, v100_data);
          tensorforge::fmacdpp16<11>(v102_acc, v116_bc, v101_data);
          tensorforge::fmacdpp16<12>(v103_acc, v116_bc, v90_data);
          tensorforge::fmacdpp16<13>(v103_acc, v116_bc, v91_data);
          tensorforge::fmacdpp16<14>(v103_acc, v116_bc, v92_data);
          tensorforge::fmacdpp16<15>(v103_acc, v116_bc, v93_data);
          float v117_bc = tensorforge::broadcast<32, 16, 1>(v115_lin);
          tensorforge::fmacdpp16<0>(v103_acc, v117_bc, v94_data);
          tensorforge::fmacdpp16<1>(v103_acc, v117_bc, v95_data);
          tensorforge::fmacdpp16<2>(v103_acc, v117_bc, v96_data);
          tensorforge::fmacdpp16<3>(v103_acc, v117_bc, v97_data);
          tensorforge::fmacdpp16<4>(v103_acc, v117_bc, v98_data);
          tensorforge::fmacdpp16<5>(v103_acc, v117_bc, v99_data);
          tensorforge::fmacdpp16<6>(v103_acc, v117_bc, v100_data);
          tensorforge::fmacdpp16<7>(v103_acc, v117_bc, v101_data);
          tensorforge::fmacdpp16<8>(v104_acc, v117_bc, v90_data);
          tensorforge::fmacdpp16<9>(v104_acc, v117_bc, v91_data);
          tensorforge::fmacdpp16<10>(v104_acc, v117_bc, v92_data);
          tensorforge::fmacdpp16<11>(v104_acc, v117_bc, v93_data);
          tensorforge::fmacdpp16<12>(v104_acc, v117_bc, v94_data);
          tensorforge::fmacdpp16<13>(v104_acc, v117_bc, v95_data);
          tensorforge::fmacdpp16<14>(v104_acc, v117_bc, v96_data);
          tensorforge::fmacdpp16<15>(v104_acc, v117_bc, v97_data);
          float v118_lin = r3[1];
          float v119_bc = tensorforge::broadcast<32, 16, 0>(v118_lin);
          tensorforge::fmacdpp16<0>(v104_acc, v119_bc, v98_data);
          tensorforge::fmacdpp16<1>(v104_acc, v119_bc, v99_data);
          tensorforge::fmacdpp16<2>(v104_acc, v119_bc, v100_data);
          tensorforge::fmacdpp16<3>(v104_acc, v119_bc, v101_data);
          tensorforge::fmacdpp16<4>(v105_acc, v119_bc, v90_data);
          tensorforge::fmacdpp16<5>(v105_acc, v119_bc, v91_data);
          tensorforge::fmacdpp16<6>(v105_acc, v119_bc, v92_data);
          tensorforge::fmacdpp16<7>(v105_acc, v119_bc, v93_data);
          tensorforge::fmacdpp16<8>(v105_acc, v119_bc, v94_data);
          tensorforge::fmacdpp16<9>(v105_acc, v119_bc, v95_data);
          tensorforge::fmacdpp16<10>(v105_acc, v119_bc, v96_data);
          tensorforge::fmacdpp16<11>(v105_acc, v119_bc, v97_data);
          tensorforge::fmacdpp16<12>(v105_acc, v119_bc, v98_data);
          tensorforge::fmacdpp16<13>(v105_acc, v119_bc, v99_data);
          tensorforge::fmacdpp16<14>(v105_acc, v119_bc, v100_data);
          tensorforge::fmacdpp16<15>(v105_acc, v119_bc, v101_data);
          float v120_bc = tensorforge::broadcast<32, 16, 1>(v118_lin);
          tensorforge::fmacdpp16<0>(v106_acc, v120_bc, v90_data);
          tensorforge::fmacdpp16<1>(v106_acc, v120_bc, v91_data);
          tensorforge::fmacdpp16<2>(v106_acc, v120_bc, v92_data);
          tensorforge::fmacdpp16<3>(v106_acc, v120_bc, v93_data);
          tensorforge::fmacdpp16<4>(v106_acc, v120_bc, v94_data);
          tensorforge::fmacdpp16<5>(v106_acc, v120_bc, v95_data);
          tensorforge::fmacdpp16<6>(v106_acc, v120_bc, v96_data);
          tensorforge::fmacdpp16<7>(v106_acc, v120_bc, v97_data);
          tensorforge::fmacdpp16<8>(v106_acc, v120_bc, v98_data);
          tensorforge::fmacdpp16<9>(v106_acc, v120_bc, v99_data);
          tensorforge::fmacdpp16<10>(v106_acc, v120_bc, v100_data);
          tensorforge::fmacdpp16<11>(v106_acc, v120_bc, v101_data);
          tensorforge::fmacdpp16<12>(v107_acc, v120_bc, v90_data);
          tensorforge::fmacdpp16<13>(v107_acc, v120_bc, v91_data);
          tensorforge::fmacdpp16<14>(v107_acc, v120_bc, v92_data);
          tensorforge::fmacdpp16<15>(v107_acc, v120_bc, v93_data);
          float v121_lin = r3[2];
          float v122_bc = tensorforge::broadcast<32, 16, 0>(v121_lin);
          tensorforge::fmacdpp16<0>(v107_acc, v122_bc, v94_data);
          tensorforge::fmacdpp16<1>(v107_acc, v122_bc, v95_data);
          tensorforge::fmacdpp16<2>(v107_acc, v122_bc, v96_data);
          tensorforge::fmacdpp16<3>(v107_acc, v122_bc, v97_data);
          tensorforge::fmacdpp16<4>(v107_acc, v122_bc, v98_data);
          tensorforge::fmacdpp16<5>(v107_acc, v122_bc, v99_data);
          tensorforge::fmacdpp16<6>(v107_acc, v122_bc, v100_data);
          tensorforge::fmacdpp16<7>(v107_acc, v122_bc, v101_data);
          tensorforge::fmacdpp16<8>(v108_acc, v122_bc, v90_data);
          tensorforge::fmacdpp16<9>(v108_acc, v122_bc, v91_data);
          tensorforge::fmacdpp16<10>(v108_acc, v122_bc, v92_data);
          tensorforge::fmacdpp16<11>(v108_acc, v122_bc, v93_data);
          tensorforge::fmacdpp16<12>(v108_acc, v122_bc, v94_data);
          tensorforge::fmacdpp16<13>(v108_acc, v122_bc, v95_data);
          tensorforge::fmacdpp16<14>(v108_acc, v122_bc, v96_data);
          tensorforge::fmacdpp16<15>(v108_acc, v122_bc, v97_data);
          float v123_bc = tensorforge::broadcast<32, 16, 1>(v121_lin);
          tensorforge::fmacdpp16<0>(v108_acc, v123_bc, v98_data);
          tensorforge::fmacdpp16<1>(v108_acc, v123_bc, v99_data);
          tensorforge::fmacdpp16<2>(v108_acc, v123_bc, v100_data);
          tensorforge::fmacdpp16<3>(v108_acc, v123_bc, v101_data);
          tensorforge::fmacdpp16<4>(v109_acc, v123_bc, v90_data);
          tensorforge::fmacdpp16<5>(v109_acc, v123_bc, v91_data);
          tensorforge::fmacdpp16<6>(v109_acc, v123_bc, v92_data);
          tensorforge::fmacdpp16<7>(v109_acc, v123_bc, v93_data);
          tensorforge::fmacdpp16<8>(v109_acc, v123_bc, v94_data);
          tensorforge::fmacdpp16<9>(v109_acc, v123_bc, v95_data);
          tensorforge::fmacdpp16<10>(v109_acc, v123_bc, v96_data);
          tensorforge::fmacdpp16<11>(v109_acc, v123_bc, v97_data);
          tensorforge::fmacdpp16<12>(v109_acc, v123_bc, v98_data);
          tensorforge::fmacdpp16<13>(v109_acc, v123_bc, v99_data);
          tensorforge::fmacdpp16<14>(v109_acc, v123_bc, v100_data);
          tensorforge::fmacdpp16<15>(v109_acc, v123_bc, v101_data);
          float v124_lin = r3[3];
          float v125_bc = tensorforge::broadcast<32, 16, 0>(v124_lin);
          tensorforge::fmacdpp16<0>(v110_acc, v125_bc, v90_data);
          tensorforge::fmacdpp16<1>(v110_acc, v125_bc, v91_data);
          tensorforge::fmacdpp16<2>(v110_acc, v125_bc, v92_data);
          tensorforge::fmacdpp16<3>(v110_acc, v125_bc, v93_data);
          tensorforge::fmacdpp16<4>(v110_acc, v125_bc, v94_data);
          tensorforge::fmacdpp16<5>(v110_acc, v125_bc, v95_data);
          tensorforge::fmacdpp16<6>(v110_acc, v125_bc, v96_data);
          tensorforge::fmacdpp16<7>(v110_acc, v125_bc, v97_data);
          tensorforge::fmacdpp16<8>(v110_acc, v125_bc, v98_data);
          tensorforge::fmacdpp16<9>(v110_acc, v125_bc, v99_data);
          tensorforge::fmacdpp16<10>(v110_acc, v125_bc, v100_data);
          tensorforge::fmacdpp16<11>(v110_acc, v125_bc, v101_data);
          tensorforge::fmacdpp16<12>(v111_acc, v125_bc, v90_data);
          tensorforge::fmacdpp16<13>(v111_acc, v125_bc, v91_data);
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
          tensorforge::fmacdpp16<8>(v112_acc, v126_bc, v90_data);
          tensorforge::fmacdpp16<9>(v112_acc, v126_bc, v91_data);
          tensorforge::fmacdpp16<10>(v112_acc, v126_bc, v92_data);
          tensorforge::fmacdpp16<11>(v112_acc, v126_bc, v93_data);
          tensorforge::fmacdpp16<12>(v112_acc, v126_bc, v94_data);
          tensorforge::fmacdpp16<13>(v112_acc, v126_bc, v95_data);
          tensorforge::fmacdpp16<14>(v112_acc, v126_bc, v96_data);
          tensorforge::fmacdpp16<15>(v112_acc, v126_bc, v97_data);
          float v127_lin = r3[4];
          float v128_bc = tensorforge::broadcast<32, 16, 0>(v127_lin);
          tensorforge::fmacdpp16<0>(v112_acc, v128_bc, v98_data);
          tensorforge::fmacdpp16<1>(v112_acc, v128_bc, v99_data);
          tensorforge::fmacdpp16<2>(v112_acc, v128_bc, v100_data);
          tensorforge::fmacdpp16<3>(v112_acc, v128_bc, v101_data);
          tensorforge::fmacdpp16<4>(v113_acc, v128_bc, v90_data);
          tensorforge::fmacdpp16<5>(v113_acc, v128_bc, v91_data);
          tensorforge::fmacdpp16<6>(v113_acc, v128_bc, v92_data);
          tensorforge::fmacdpp16<7>(v113_acc, v128_bc, v93_data);
          tensorforge::fmacdpp16<8>(v113_acc, v128_bc, v94_data);
          tensorforge::fmacdpp16<9>(v113_acc, v128_bc, v95_data);
          tensorforge::fmacdpp16<10>(v113_acc, v128_bc, v96_data);
          tensorforge::fmacdpp16<11>(v113_acc, v128_bc, v97_data);
          tensorforge::fmacdpp16<12>(v113_acc, v128_bc, v98_data);
          tensorforge::fmacdpp16<13>(v113_acc, v128_bc, v99_data);
          tensorforge::fmacdpp16<14>(v113_acc, v128_bc, v100_data);
          tensorforge::fmacdpp16<15>(v113_acc, v128_bc, v101_data);
          float v129_bc = tensorforge::broadcast<32, 16, 1>(v127_lin);
          tensorforge::fmacdpp16<0>(v114_acc, v129_bc, v90_data);
          tensorforge::fmacdpp16<1>(v114_acc, v129_bc, v91_data);
          tensorforge::fmacdpp16<2>(v114_acc, v129_bc, v92_data);
          tensorforge::fmacdpp16<3>(v114_acc, v129_bc, v93_data);
          tensorforge::fmacdpp16<4>(v114_acc, v129_bc, v94_data);
          tensorforge::fmacdpp16<5>(v114_acc, v129_bc, v95_data);
          tensorforge::fmacdpp16<6>(v114_acc, v129_bc, v96_data);
          tensorforge::fmacdpp16<7>(v114_acc, v129_bc, v97_data);
          tensorforge::fmacdpp16<8>(v114_acc, v129_bc, v98_data);
          tensorforge::fmacdpp16<9>(v114_acc, v129_bc, v99_data);
          tensorforge::fmacdpp16<10>(v114_acc, v129_bc, v100_data);
          tensorforge::fmacdpp16<11>(v114_acc, v129_bc, v101_data);
          ir4[0] = v102_acc;
          ir4[1] = v103_acc;
          ir4[2] = v104_acc;
          ir4[3] = v105_acc;
          ir4[4] = v106_acc;
          ir4[5] = v107_acc;
          ir4[6] = v108_acc;
          ir4[7] = v109_acc;
          ir4[8] = v110_acc;
          ir4[9] = v111_acc;
          ir4[10] = v112_acc;
          ir4[11] = v113_acc;
          ir4[12] = v114_acc;
          #pragma unroll
          for (int32_t v133_n0 = 0; v133_n0 < 1; ++v133_n0) {
            #pragma unroll
            for (int32_t v134_n1 = 0; v134_n1 < 13; ++v134_n1) {
              int32_t v135_a = v133_n0 + v134_n1;
              float v136_data = ir4[v135_a];
              float v138_data = r1[v135_a];
              r4[v135_a] = (v138_data + v136_data);
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          float v145_data = r4[4];
          float v146_data = r5[0];
          r5[0] = (v146_data + v145_data);
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v151_i0 = 0; v151_i0 < 1; ++v151_i0) {
            int32_t v159_lead = v12_lead + (v151_i0 * 32);
            #pragma unroll
            for (int32_t v152_i1 = 0; v152_i1 < 1; ++v152_i1) {
              float v154_data = r5[(v151_i0 + v152_i1)];
              glb_m0[(v159_lead + ((v152_i1 + 4) * 32))] = v154_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v167_i0 = 0; v167_i0 < 1; ++v167_i0) {
            int32_t v173_lead = v12_lead + (v167_i0 * 32);
            #pragma unroll
            for (int32_t v168_i1 = 0; v168_i1 < 13; ++v168_i1) {
              float v176_data = glb_m0[(v173_lead + (v168_i1 * 32))];
              r6[(v167_i0 + v168_i1)] = v176_data;
            }
          }
          float r7[13]{};
          // r7 = load{g>r}(glb_m4);
          float v179_lin = glb_m4[0 + threadIdx.x * 1];
          r7[0] = v179_lin;
          float v180_lin = glb_m4[32 + threadIdx.x * 1];
          r7[1] = v180_lin;
          float v181_lin = glb_m4[64 + threadIdx.x * 1];
          r7[2] = v181_lin;
          float v182_lin = glb_m4[96 + threadIdx.x * 1];
          r7[3] = v182_lin;
          float v183_lin = glb_m4[128 + threadIdx.x * 1];
          r7[4] = v183_lin;
          float v184_lin = glb_m4[160 + threadIdx.x * 1];
          r7[5] = v184_lin;
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v186_data = r6[0];
          float v187_data = r6[1];
          float v188_data = r6[2];
          float v189_data = r6[3];
          float v190_data = r6[4];
          float v191_data = r6[5];
          float v192_data = r6[6];
          float v193_data = r6[7];
          float v194_data = r6[8];
          float v195_data = r6[9];
          float v196_data = r6[10];
          float v197_data = r6[11];
          float v198_data = r6[12];
          float v199_acc{};
          float v200_acc{};
          float v201_acc{};
          float v202_acc{};
          float v203_acc{};
          float v204_acc{};
          float v205_acc{};
          float v206_acc{};
          float v207_acc{};
          float v208_acc{};
          float v209_acc{};
          float v210_acc{};
          float v211_acc{};
          float v212_lin = r7[0];
          float v213_bc = tensorforge::broadcast<32, 16, 0>(v212_lin);
          tensorforge::fmacdpp16<0>(v199_acc, v213_bc, v186_data);
          tensorforge::fmacdpp16<1>(v199_acc, v213_bc, v187_data);
          tensorforge::fmacdpp16<2>(v199_acc, v213_bc, v188_data);
          tensorforge::fmacdpp16<3>(v199_acc, v213_bc, v189_data);
          tensorforge::fmacdpp16<4>(v199_acc, v213_bc, v190_data);
          tensorforge::fmacdpp16<5>(v199_acc, v213_bc, v191_data);
          tensorforge::fmacdpp16<6>(v199_acc, v213_bc, v192_data);
          tensorforge::fmacdpp16<7>(v199_acc, v213_bc, v193_data);
          tensorforge::fmacdpp16<8>(v199_acc, v213_bc, v194_data);
          tensorforge::fmacdpp16<9>(v199_acc, v213_bc, v195_data);
          tensorforge::fmacdpp16<10>(v199_acc, v213_bc, v196_data);
          tensorforge::fmacdpp16<11>(v199_acc, v213_bc, v197_data);
          tensorforge::fmacdpp16<12>(v199_acc, v213_bc, v198_data);
          tensorforge::fmacdpp16<13>(v200_acc, v213_bc, v186_data);
          tensorforge::fmacdpp16<14>(v200_acc, v213_bc, v187_data);
          tensorforge::fmacdpp16<15>(v200_acc, v213_bc, v188_data);
          float v214_bc = tensorforge::broadcast<32, 16, 1>(v212_lin);
          tensorforge::fmacdpp16<0>(v200_acc, v214_bc, v189_data);
          tensorforge::fmacdpp16<1>(v200_acc, v214_bc, v190_data);
          tensorforge::fmacdpp16<2>(v200_acc, v214_bc, v191_data);
          tensorforge::fmacdpp16<3>(v200_acc, v214_bc, v192_data);
          tensorforge::fmacdpp16<4>(v200_acc, v214_bc, v193_data);
          tensorforge::fmacdpp16<5>(v200_acc, v214_bc, v194_data);
          tensorforge::fmacdpp16<6>(v200_acc, v214_bc, v195_data);
          tensorforge::fmacdpp16<7>(v200_acc, v214_bc, v196_data);
          tensorforge::fmacdpp16<8>(v200_acc, v214_bc, v197_data);
          tensorforge::fmacdpp16<9>(v200_acc, v214_bc, v198_data);
          tensorforge::fmacdpp16<10>(v201_acc, v214_bc, v186_data);
          tensorforge::fmacdpp16<11>(v201_acc, v214_bc, v187_data);
          tensorforge::fmacdpp16<12>(v201_acc, v214_bc, v188_data);
          tensorforge::fmacdpp16<13>(v201_acc, v214_bc, v189_data);
          tensorforge::fmacdpp16<14>(v201_acc, v214_bc, v190_data);
          tensorforge::fmacdpp16<15>(v201_acc, v214_bc, v191_data);
          float v215_lin = r7[1];
          float v216_bc = tensorforge::broadcast<32, 16, 0>(v215_lin);
          tensorforge::fmacdpp16<0>(v201_acc, v216_bc, v192_data);
          tensorforge::fmacdpp16<1>(v201_acc, v216_bc, v193_data);
          tensorforge::fmacdpp16<2>(v201_acc, v216_bc, v194_data);
          tensorforge::fmacdpp16<3>(v201_acc, v216_bc, v195_data);
          tensorforge::fmacdpp16<4>(v201_acc, v216_bc, v196_data);
          tensorforge::fmacdpp16<5>(v201_acc, v216_bc, v197_data);
          tensorforge::fmacdpp16<6>(v201_acc, v216_bc, v198_data);
          tensorforge::fmacdpp16<7>(v202_acc, v216_bc, v186_data);
          tensorforge::fmacdpp16<8>(v202_acc, v216_bc, v187_data);
          tensorforge::fmacdpp16<9>(v202_acc, v216_bc, v188_data);
          tensorforge::fmacdpp16<10>(v202_acc, v216_bc, v189_data);
          tensorforge::fmacdpp16<11>(v202_acc, v216_bc, v190_data);
          tensorforge::fmacdpp16<12>(v202_acc, v216_bc, v191_data);
          tensorforge::fmacdpp16<13>(v202_acc, v216_bc, v192_data);
          tensorforge::fmacdpp16<14>(v202_acc, v216_bc, v193_data);
          tensorforge::fmacdpp16<15>(v202_acc, v216_bc, v194_data);
          float v217_bc = tensorforge::broadcast<32, 16, 1>(v215_lin);
          tensorforge::fmacdpp16<0>(v202_acc, v217_bc, v195_data);
          tensorforge::fmacdpp16<1>(v202_acc, v217_bc, v196_data);
          tensorforge::fmacdpp16<2>(v202_acc, v217_bc, v197_data);
          tensorforge::fmacdpp16<3>(v202_acc, v217_bc, v198_data);
          tensorforge::fmacdpp16<4>(v203_acc, v217_bc, v186_data);
          tensorforge::fmacdpp16<5>(v203_acc, v217_bc, v187_data);
          tensorforge::fmacdpp16<6>(v203_acc, v217_bc, v188_data);
          tensorforge::fmacdpp16<7>(v203_acc, v217_bc, v189_data);
          tensorforge::fmacdpp16<8>(v203_acc, v217_bc, v190_data);
          tensorforge::fmacdpp16<9>(v203_acc, v217_bc, v191_data);
          tensorforge::fmacdpp16<10>(v203_acc, v217_bc, v192_data);
          tensorforge::fmacdpp16<11>(v203_acc, v217_bc, v193_data);
          tensorforge::fmacdpp16<12>(v203_acc, v217_bc, v194_data);
          tensorforge::fmacdpp16<13>(v203_acc, v217_bc, v195_data);
          tensorforge::fmacdpp16<14>(v203_acc, v217_bc, v196_data);
          tensorforge::fmacdpp16<15>(v203_acc, v217_bc, v197_data);
          float v218_lin = r7[2];
          float v219_bc = tensorforge::broadcast<32, 16, 0>(v218_lin);
          tensorforge::fmacdpp16<0>(v203_acc, v219_bc, v198_data);
          tensorforge::fmacdpp16<1>(v204_acc, v219_bc, v186_data);
          tensorforge::fmacdpp16<2>(v204_acc, v219_bc, v187_data);
          tensorforge::fmacdpp16<3>(v204_acc, v219_bc, v188_data);
          tensorforge::fmacdpp16<4>(v204_acc, v219_bc, v189_data);
          tensorforge::fmacdpp16<5>(v204_acc, v219_bc, v190_data);
          tensorforge::fmacdpp16<6>(v204_acc, v219_bc, v191_data);
          tensorforge::fmacdpp16<7>(v204_acc, v219_bc, v192_data);
          tensorforge::fmacdpp16<8>(v204_acc, v219_bc, v193_data);
          tensorforge::fmacdpp16<9>(v204_acc, v219_bc, v194_data);
          tensorforge::fmacdpp16<10>(v204_acc, v219_bc, v195_data);
          tensorforge::fmacdpp16<11>(v204_acc, v219_bc, v196_data);
          tensorforge::fmacdpp16<12>(v204_acc, v219_bc, v197_data);
          tensorforge::fmacdpp16<13>(v204_acc, v219_bc, v198_data);
          tensorforge::fmacdpp16<14>(v205_acc, v219_bc, v186_data);
          tensorforge::fmacdpp16<15>(v205_acc, v219_bc, v187_data);
          float v220_bc = tensorforge::broadcast<32, 16, 1>(v218_lin);
          tensorforge::fmacdpp16<0>(v205_acc, v220_bc, v188_data);
          tensorforge::fmacdpp16<1>(v205_acc, v220_bc, v189_data);
          tensorforge::fmacdpp16<2>(v205_acc, v220_bc, v190_data);
          tensorforge::fmacdpp16<3>(v205_acc, v220_bc, v191_data);
          tensorforge::fmacdpp16<4>(v205_acc, v220_bc, v192_data);
          tensorforge::fmacdpp16<5>(v205_acc, v220_bc, v193_data);
          tensorforge::fmacdpp16<6>(v205_acc, v220_bc, v194_data);
          tensorforge::fmacdpp16<7>(v205_acc, v220_bc, v195_data);
          tensorforge::fmacdpp16<8>(v205_acc, v220_bc, v196_data);
          tensorforge::fmacdpp16<9>(v205_acc, v220_bc, v197_data);
          tensorforge::fmacdpp16<10>(v205_acc, v220_bc, v198_data);
          tensorforge::fmacdpp16<11>(v206_acc, v220_bc, v186_data);
          tensorforge::fmacdpp16<12>(v206_acc, v220_bc, v187_data);
          tensorforge::fmacdpp16<13>(v206_acc, v220_bc, v188_data);
          tensorforge::fmacdpp16<14>(v206_acc, v220_bc, v189_data);
          tensorforge::fmacdpp16<15>(v206_acc, v220_bc, v190_data);
          float v221_lin = r7[3];
          float v222_bc = tensorforge::broadcast<32, 16, 0>(v221_lin);
          tensorforge::fmacdpp16<0>(v206_acc, v222_bc, v191_data);
          tensorforge::fmacdpp16<1>(v206_acc, v222_bc, v192_data);
          tensorforge::fmacdpp16<2>(v206_acc, v222_bc, v193_data);
          tensorforge::fmacdpp16<3>(v206_acc, v222_bc, v194_data);
          tensorforge::fmacdpp16<4>(v206_acc, v222_bc, v195_data);
          tensorforge::fmacdpp16<5>(v206_acc, v222_bc, v196_data);
          tensorforge::fmacdpp16<6>(v206_acc, v222_bc, v197_data);
          tensorforge::fmacdpp16<7>(v206_acc, v222_bc, v198_data);
          tensorforge::fmacdpp16<8>(v207_acc, v222_bc, v186_data);
          tensorforge::fmacdpp16<9>(v207_acc, v222_bc, v187_data);
          tensorforge::fmacdpp16<10>(v207_acc, v222_bc, v188_data);
          tensorforge::fmacdpp16<11>(v207_acc, v222_bc, v189_data);
          tensorforge::fmacdpp16<12>(v207_acc, v222_bc, v190_data);
          tensorforge::fmacdpp16<13>(v207_acc, v222_bc, v191_data);
          tensorforge::fmacdpp16<14>(v207_acc, v222_bc, v192_data);
          tensorforge::fmacdpp16<15>(v207_acc, v222_bc, v193_data);
          float v223_bc = tensorforge::broadcast<32, 16, 1>(v221_lin);
          tensorforge::fmacdpp16<0>(v207_acc, v223_bc, v194_data);
          tensorforge::fmacdpp16<1>(v207_acc, v223_bc, v195_data);
          tensorforge::fmacdpp16<2>(v207_acc, v223_bc, v196_data);
          tensorforge::fmacdpp16<3>(v207_acc, v223_bc, v197_data);
          tensorforge::fmacdpp16<4>(v207_acc, v223_bc, v198_data);
          tensorforge::fmacdpp16<5>(v208_acc, v223_bc, v186_data);
          tensorforge::fmacdpp16<6>(v208_acc, v223_bc, v187_data);
          tensorforge::fmacdpp16<7>(v208_acc, v223_bc, v188_data);
          tensorforge::fmacdpp16<8>(v208_acc, v223_bc, v189_data);
          tensorforge::fmacdpp16<9>(v208_acc, v223_bc, v190_data);
          tensorforge::fmacdpp16<10>(v208_acc, v223_bc, v191_data);
          tensorforge::fmacdpp16<11>(v208_acc, v223_bc, v192_data);
          tensorforge::fmacdpp16<12>(v208_acc, v223_bc, v193_data);
          tensorforge::fmacdpp16<13>(v208_acc, v223_bc, v194_data);
          tensorforge::fmacdpp16<14>(v208_acc, v223_bc, v195_data);
          tensorforge::fmacdpp16<15>(v208_acc, v223_bc, v196_data);
          float v224_lin = r7[4];
          float v225_bc = tensorforge::broadcast<32, 16, 0>(v224_lin);
          tensorforge::fmacdpp16<0>(v208_acc, v225_bc, v197_data);
          tensorforge::fmacdpp16<1>(v208_acc, v225_bc, v198_data);
          tensorforge::fmacdpp16<2>(v209_acc, v225_bc, v186_data);
          tensorforge::fmacdpp16<3>(v209_acc, v225_bc, v187_data);
          tensorforge::fmacdpp16<4>(v209_acc, v225_bc, v188_data);
          tensorforge::fmacdpp16<5>(v209_acc, v225_bc, v189_data);
          tensorforge::fmacdpp16<6>(v209_acc, v225_bc, v190_data);
          tensorforge::fmacdpp16<7>(v209_acc, v225_bc, v191_data);
          tensorforge::fmacdpp16<8>(v209_acc, v225_bc, v192_data);
          tensorforge::fmacdpp16<9>(v209_acc, v225_bc, v193_data);
          tensorforge::fmacdpp16<10>(v209_acc, v225_bc, v194_data);
          tensorforge::fmacdpp16<11>(v209_acc, v225_bc, v195_data);
          tensorforge::fmacdpp16<12>(v209_acc, v225_bc, v196_data);
          tensorforge::fmacdpp16<13>(v209_acc, v225_bc, v197_data);
          tensorforge::fmacdpp16<14>(v209_acc, v225_bc, v198_data);
          tensorforge::fmacdpp16<15>(v210_acc, v225_bc, v186_data);
          float v226_bc = tensorforge::broadcast<32, 16, 1>(v224_lin);
          tensorforge::fmacdpp16<0>(v210_acc, v226_bc, v187_data);
          tensorforge::fmacdpp16<1>(v210_acc, v226_bc, v188_data);
          tensorforge::fmacdpp16<2>(v210_acc, v226_bc, v189_data);
          tensorforge::fmacdpp16<3>(v210_acc, v226_bc, v190_data);
          tensorforge::fmacdpp16<4>(v210_acc, v226_bc, v191_data);
          tensorforge::fmacdpp16<5>(v210_acc, v226_bc, v192_data);
          tensorforge::fmacdpp16<6>(v210_acc, v226_bc, v193_data);
          tensorforge::fmacdpp16<7>(v210_acc, v226_bc, v194_data);
          tensorforge::fmacdpp16<8>(v210_acc, v226_bc, v195_data);
          tensorforge::fmacdpp16<9>(v210_acc, v226_bc, v196_data);
          tensorforge::fmacdpp16<10>(v210_acc, v226_bc, v197_data);
          tensorforge::fmacdpp16<11>(v210_acc, v226_bc, v198_data);
          tensorforge::fmacdpp16<12>(v211_acc, v226_bc, v186_data);
          tensorforge::fmacdpp16<13>(v211_acc, v226_bc, v187_data);
          tensorforge::fmacdpp16<14>(v211_acc, v226_bc, v188_data);
          tensorforge::fmacdpp16<15>(v211_acc, v226_bc, v189_data);
          float v227_lin = r7[5];
          float v228_bc = tensorforge::broadcast<32, 16, 0>(v227_lin);
          tensorforge::fmacdpp16<0>(v211_acc, v228_bc, v190_data);
          tensorforge::fmacdpp16<1>(v211_acc, v228_bc, v191_data);
          tensorforge::fmacdpp16<2>(v211_acc, v228_bc, v192_data);
          tensorforge::fmacdpp16<3>(v211_acc, v228_bc, v193_data);
          tensorforge::fmacdpp16<4>(v211_acc, v228_bc, v194_data);
          tensorforge::fmacdpp16<5>(v211_acc, v228_bc, v195_data);
          tensorforge::fmacdpp16<6>(v211_acc, v228_bc, v196_data);
          tensorforge::fmacdpp16<7>(v211_acc, v228_bc, v197_data);
          tensorforge::fmacdpp16<8>(v211_acc, v228_bc, v198_data);
          r8[0] = v199_acc;
          r8[1] = v200_acc;
          r8[2] = v201_acc;
          r8[3] = v202_acc;
          r8[4] = v203_acc;
          r8[5] = v204_acc;
          r8[6] = v205_acc;
          r8[7] = v206_acc;
          r8[8] = v207_acc;
          r8[9] = v208_acc;
          r8[10] = v209_acc;
          r8[11] = v210_acc;
          r8[12] = v211_acc;
          // glb_m3 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v232_i0 = 0; v232_i0 < 1; ++v232_i0) {
            int32_t v240_lead = v12_lead + (v232_i0 * 32);
            #pragma unroll
            for (int32_t v233_i1 = 0; v233_i1 < 13; ++v233_i1) {
              float v235_data = r8[(v232_i0 + v233_i1)];
              glb_m3[(v240_lead + (v233_i1 * 32))] = v235_data;
            }
          }
        }
      }
    }
  }
}

