// === base name ===
kernel_82283a2aa0

// === header ===
void launcher_kernel_82283a2aa0(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_82283a2aa0(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_82283a2aa0, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_82283a2aa0), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_82283a2aa0, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_82283a2aa0(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×16(32×16) {0..32}×{0..16} strided
    // m1 32×12(32×12) {0..32}×{0..12} strided
    // m2 12×16(12×16) {0..12}×{0..16} strided
    // m3 32×12(32×12) {0..32}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 32×12(32×12) {0..32}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..16})[0, 1] = m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[-1, 1]
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m3 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m5 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 512 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 192 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 384 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 384 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v16_lead = v11_i0 * 32;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 12; ++v12_i1) {
              int32_t v18_a = v12_i1 * 32;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + v18_a)]);
              int32_t v28_a = v11_i0 + v12_i1;
              r0[v28_a] = v27_data;
            }
          }
          float r1[16]{};
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
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v40_i0 = 0; v40_i0 < 1; ++v40_i0) {
            int32_t v45_lead = v40_i0 * 32;
            int32_t v46_lead = v10_lead + v45_lead;
            int32_t v53_lead = v10_lead + v45_lead;
            #pragma unroll
            for (int32_t v41_i1 = 0; v41_i1 < 12; ++v41_i1) {
              int32_t v47_a = v41_i1 * 32;
              int32_t v48_a = v46_lead + v47_a;
              float v56_data = __builtin_nontemporal_load(&glb_m3[(v53_lead + v47_a)]);
              int32_t v57_a = v40_i0 + v41_i1;
              r3[v57_a] = v56_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          float v59_data = r0[0];
          float v60_data = r0[1];
          float v61_data = r0[2];
          float v62_data = r0[3];
          float v63_data = r0[4];
          float v64_data = r0[5];
          float v65_data = r0[6];
          float v66_data = r0[7];
          float v67_data = r0[8];
          float v68_data = r0[9];
          float v69_data = r0[10];
          float v70_data = r0[11];
          float v71_acc{};
          float v72_acc{};
          float v73_acc{};
          float v74_acc{};
          float v75_acc{};
          float v76_acc{};
          float v77_acc{};
          float v78_acc{};
          float v79_acc{};
          float v80_acc{};
          float v81_acc{};
          float v82_acc{};
          float v83_acc{};
          float v84_acc{};
          float v85_acc{};
          float v86_acc{};
          float v87_lin = r1[0];
          float v88_bc = tensorforge::broadcast<32, 16, 0>(v87_lin);
          tensorforge::fmacdpp16<0>(v71_acc, v88_bc, v59_data);
          tensorforge::fmacdpp16<1>(v71_acc, v88_bc, v60_data);
          tensorforge::fmacdpp16<2>(v71_acc, v88_bc, v61_data);
          tensorforge::fmacdpp16<3>(v71_acc, v88_bc, v62_data);
          tensorforge::fmacdpp16<4>(v71_acc, v88_bc, v63_data);
          tensorforge::fmacdpp16<5>(v71_acc, v88_bc, v64_data);
          tensorforge::fmacdpp16<6>(v71_acc, v88_bc, v65_data);
          tensorforge::fmacdpp16<7>(v71_acc, v88_bc, v66_data);
          tensorforge::fmacdpp16<8>(v71_acc, v88_bc, v67_data);
          tensorforge::fmacdpp16<9>(v71_acc, v88_bc, v68_data);
          tensorforge::fmacdpp16<10>(v71_acc, v88_bc, v69_data);
          tensorforge::fmacdpp16<11>(v71_acc, v88_bc, v70_data);
          tensorforge::fmacdpp16<12>(v72_acc, v88_bc, v59_data);
          tensorforge::fmacdpp16<13>(v72_acc, v88_bc, v60_data);
          tensorforge::fmacdpp16<14>(v72_acc, v88_bc, v61_data);
          tensorforge::fmacdpp16<15>(v72_acc, v88_bc, v62_data);
          float v89_bc = tensorforge::broadcast<32, 16, 1>(v87_lin);
          tensorforge::fmacdpp16<0>(v72_acc, v89_bc, v63_data);
          tensorforge::fmacdpp16<1>(v72_acc, v89_bc, v64_data);
          tensorforge::fmacdpp16<2>(v72_acc, v89_bc, v65_data);
          tensorforge::fmacdpp16<3>(v72_acc, v89_bc, v66_data);
          tensorforge::fmacdpp16<4>(v72_acc, v89_bc, v67_data);
          tensorforge::fmacdpp16<5>(v72_acc, v89_bc, v68_data);
          tensorforge::fmacdpp16<6>(v72_acc, v89_bc, v69_data);
          tensorforge::fmacdpp16<7>(v72_acc, v89_bc, v70_data);
          tensorforge::fmacdpp16<8>(v73_acc, v89_bc, v59_data);
          tensorforge::fmacdpp16<9>(v73_acc, v89_bc, v60_data);
          tensorforge::fmacdpp16<10>(v73_acc, v89_bc, v61_data);
          tensorforge::fmacdpp16<11>(v73_acc, v89_bc, v62_data);
          tensorforge::fmacdpp16<12>(v73_acc, v89_bc, v63_data);
          tensorforge::fmacdpp16<13>(v73_acc, v89_bc, v64_data);
          tensorforge::fmacdpp16<14>(v73_acc, v89_bc, v65_data);
          tensorforge::fmacdpp16<15>(v73_acc, v89_bc, v66_data);
          float v90_lin = r1[1];
          float v91_bc = tensorforge::broadcast<32, 16, 0>(v90_lin);
          tensorforge::fmacdpp16<0>(v73_acc, v91_bc, v67_data);
          tensorforge::fmacdpp16<1>(v73_acc, v91_bc, v68_data);
          tensorforge::fmacdpp16<2>(v73_acc, v91_bc, v69_data);
          tensorforge::fmacdpp16<3>(v73_acc, v91_bc, v70_data);
          tensorforge::fmacdpp16<4>(v74_acc, v91_bc, v59_data);
          tensorforge::fmacdpp16<5>(v74_acc, v91_bc, v60_data);
          tensorforge::fmacdpp16<6>(v74_acc, v91_bc, v61_data);
          tensorforge::fmacdpp16<7>(v74_acc, v91_bc, v62_data);
          tensorforge::fmacdpp16<8>(v74_acc, v91_bc, v63_data);
          tensorforge::fmacdpp16<9>(v74_acc, v91_bc, v64_data);
          tensorforge::fmacdpp16<10>(v74_acc, v91_bc, v65_data);
          tensorforge::fmacdpp16<11>(v74_acc, v91_bc, v66_data);
          tensorforge::fmacdpp16<12>(v74_acc, v91_bc, v67_data);
          tensorforge::fmacdpp16<13>(v74_acc, v91_bc, v68_data);
          tensorforge::fmacdpp16<14>(v74_acc, v91_bc, v69_data);
          tensorforge::fmacdpp16<15>(v74_acc, v91_bc, v70_data);
          float v92_bc = tensorforge::broadcast<32, 16, 1>(v90_lin);
          tensorforge::fmacdpp16<0>(v75_acc, v92_bc, v59_data);
          tensorforge::fmacdpp16<1>(v75_acc, v92_bc, v60_data);
          tensorforge::fmacdpp16<2>(v75_acc, v92_bc, v61_data);
          tensorforge::fmacdpp16<3>(v75_acc, v92_bc, v62_data);
          tensorforge::fmacdpp16<4>(v75_acc, v92_bc, v63_data);
          tensorforge::fmacdpp16<5>(v75_acc, v92_bc, v64_data);
          tensorforge::fmacdpp16<6>(v75_acc, v92_bc, v65_data);
          tensorforge::fmacdpp16<7>(v75_acc, v92_bc, v66_data);
          tensorforge::fmacdpp16<8>(v75_acc, v92_bc, v67_data);
          tensorforge::fmacdpp16<9>(v75_acc, v92_bc, v68_data);
          tensorforge::fmacdpp16<10>(v75_acc, v92_bc, v69_data);
          tensorforge::fmacdpp16<11>(v75_acc, v92_bc, v70_data);
          tensorforge::fmacdpp16<12>(v76_acc, v92_bc, v59_data);
          tensorforge::fmacdpp16<13>(v76_acc, v92_bc, v60_data);
          tensorforge::fmacdpp16<14>(v76_acc, v92_bc, v61_data);
          tensorforge::fmacdpp16<15>(v76_acc, v92_bc, v62_data);
          float v93_lin = r1[2];
          float v94_bc = tensorforge::broadcast<32, 16, 0>(v93_lin);
          tensorforge::fmacdpp16<0>(v76_acc, v94_bc, v63_data);
          tensorforge::fmacdpp16<1>(v76_acc, v94_bc, v64_data);
          tensorforge::fmacdpp16<2>(v76_acc, v94_bc, v65_data);
          tensorforge::fmacdpp16<3>(v76_acc, v94_bc, v66_data);
          tensorforge::fmacdpp16<4>(v76_acc, v94_bc, v67_data);
          tensorforge::fmacdpp16<5>(v76_acc, v94_bc, v68_data);
          tensorforge::fmacdpp16<6>(v76_acc, v94_bc, v69_data);
          tensorforge::fmacdpp16<7>(v76_acc, v94_bc, v70_data);
          tensorforge::fmacdpp16<8>(v77_acc, v94_bc, v59_data);
          tensorforge::fmacdpp16<9>(v77_acc, v94_bc, v60_data);
          tensorforge::fmacdpp16<10>(v77_acc, v94_bc, v61_data);
          tensorforge::fmacdpp16<11>(v77_acc, v94_bc, v62_data);
          tensorforge::fmacdpp16<12>(v77_acc, v94_bc, v63_data);
          tensorforge::fmacdpp16<13>(v77_acc, v94_bc, v64_data);
          tensorforge::fmacdpp16<14>(v77_acc, v94_bc, v65_data);
          tensorforge::fmacdpp16<15>(v77_acc, v94_bc, v66_data);
          float v95_bc = tensorforge::broadcast<32, 16, 1>(v93_lin);
          tensorforge::fmacdpp16<0>(v77_acc, v95_bc, v67_data);
          tensorforge::fmacdpp16<1>(v77_acc, v95_bc, v68_data);
          tensorforge::fmacdpp16<2>(v77_acc, v95_bc, v69_data);
          tensorforge::fmacdpp16<3>(v77_acc, v95_bc, v70_data);
          tensorforge::fmacdpp16<4>(v78_acc, v95_bc, v59_data);
          tensorforge::fmacdpp16<5>(v78_acc, v95_bc, v60_data);
          tensorforge::fmacdpp16<6>(v78_acc, v95_bc, v61_data);
          tensorforge::fmacdpp16<7>(v78_acc, v95_bc, v62_data);
          tensorforge::fmacdpp16<8>(v78_acc, v95_bc, v63_data);
          tensorforge::fmacdpp16<9>(v78_acc, v95_bc, v64_data);
          tensorforge::fmacdpp16<10>(v78_acc, v95_bc, v65_data);
          tensorforge::fmacdpp16<11>(v78_acc, v95_bc, v66_data);
          tensorforge::fmacdpp16<12>(v78_acc, v95_bc, v67_data);
          tensorforge::fmacdpp16<13>(v78_acc, v95_bc, v68_data);
          tensorforge::fmacdpp16<14>(v78_acc, v95_bc, v69_data);
          tensorforge::fmacdpp16<15>(v78_acc, v95_bc, v70_data);
          float v96_lin = r1[3];
          float v97_bc = tensorforge::broadcast<32, 16, 0>(v96_lin);
          tensorforge::fmacdpp16<0>(v79_acc, v97_bc, v59_data);
          tensorforge::fmacdpp16<1>(v79_acc, v97_bc, v60_data);
          tensorforge::fmacdpp16<2>(v79_acc, v97_bc, v61_data);
          tensorforge::fmacdpp16<3>(v79_acc, v97_bc, v62_data);
          tensorforge::fmacdpp16<4>(v79_acc, v97_bc, v63_data);
          tensorforge::fmacdpp16<5>(v79_acc, v97_bc, v64_data);
          tensorforge::fmacdpp16<6>(v79_acc, v97_bc, v65_data);
          tensorforge::fmacdpp16<7>(v79_acc, v97_bc, v66_data);
          tensorforge::fmacdpp16<8>(v79_acc, v97_bc, v67_data);
          tensorforge::fmacdpp16<9>(v79_acc, v97_bc, v68_data);
          tensorforge::fmacdpp16<10>(v79_acc, v97_bc, v69_data);
          tensorforge::fmacdpp16<11>(v79_acc, v97_bc, v70_data);
          tensorforge::fmacdpp16<12>(v80_acc, v97_bc, v59_data);
          tensorforge::fmacdpp16<13>(v80_acc, v97_bc, v60_data);
          tensorforge::fmacdpp16<14>(v80_acc, v97_bc, v61_data);
          tensorforge::fmacdpp16<15>(v80_acc, v97_bc, v62_data);
          float v98_bc = tensorforge::broadcast<32, 16, 1>(v96_lin);
          tensorforge::fmacdpp16<0>(v80_acc, v98_bc, v63_data);
          tensorforge::fmacdpp16<1>(v80_acc, v98_bc, v64_data);
          tensorforge::fmacdpp16<2>(v80_acc, v98_bc, v65_data);
          tensorforge::fmacdpp16<3>(v80_acc, v98_bc, v66_data);
          tensorforge::fmacdpp16<4>(v80_acc, v98_bc, v67_data);
          tensorforge::fmacdpp16<5>(v80_acc, v98_bc, v68_data);
          tensorforge::fmacdpp16<6>(v80_acc, v98_bc, v69_data);
          tensorforge::fmacdpp16<7>(v80_acc, v98_bc, v70_data);
          tensorforge::fmacdpp16<8>(v81_acc, v98_bc, v59_data);
          tensorforge::fmacdpp16<9>(v81_acc, v98_bc, v60_data);
          tensorforge::fmacdpp16<10>(v81_acc, v98_bc, v61_data);
          tensorforge::fmacdpp16<11>(v81_acc, v98_bc, v62_data);
          tensorforge::fmacdpp16<12>(v81_acc, v98_bc, v63_data);
          tensorforge::fmacdpp16<13>(v81_acc, v98_bc, v64_data);
          tensorforge::fmacdpp16<14>(v81_acc, v98_bc, v65_data);
          tensorforge::fmacdpp16<15>(v81_acc, v98_bc, v66_data);
          float v99_lin = r1[4];
          float v100_bc = tensorforge::broadcast<32, 16, 0>(v99_lin);
          tensorforge::fmacdpp16<0>(v81_acc, v100_bc, v67_data);
          tensorforge::fmacdpp16<1>(v81_acc, v100_bc, v68_data);
          tensorforge::fmacdpp16<2>(v81_acc, v100_bc, v69_data);
          tensorforge::fmacdpp16<3>(v81_acc, v100_bc, v70_data);
          tensorforge::fmacdpp16<4>(v82_acc, v100_bc, v59_data);
          tensorforge::fmacdpp16<5>(v82_acc, v100_bc, v60_data);
          tensorforge::fmacdpp16<6>(v82_acc, v100_bc, v61_data);
          tensorforge::fmacdpp16<7>(v82_acc, v100_bc, v62_data);
          tensorforge::fmacdpp16<8>(v82_acc, v100_bc, v63_data);
          tensorforge::fmacdpp16<9>(v82_acc, v100_bc, v64_data);
          tensorforge::fmacdpp16<10>(v82_acc, v100_bc, v65_data);
          tensorforge::fmacdpp16<11>(v82_acc, v100_bc, v66_data);
          tensorforge::fmacdpp16<12>(v82_acc, v100_bc, v67_data);
          tensorforge::fmacdpp16<13>(v82_acc, v100_bc, v68_data);
          tensorforge::fmacdpp16<14>(v82_acc, v100_bc, v69_data);
          tensorforge::fmacdpp16<15>(v82_acc, v100_bc, v70_data);
          float v101_bc = tensorforge::broadcast<32, 16, 1>(v99_lin);
          tensorforge::fmacdpp16<0>(v83_acc, v101_bc, v59_data);
          tensorforge::fmacdpp16<1>(v83_acc, v101_bc, v60_data);
          tensorforge::fmacdpp16<2>(v83_acc, v101_bc, v61_data);
          tensorforge::fmacdpp16<3>(v83_acc, v101_bc, v62_data);
          tensorforge::fmacdpp16<4>(v83_acc, v101_bc, v63_data);
          tensorforge::fmacdpp16<5>(v83_acc, v101_bc, v64_data);
          tensorforge::fmacdpp16<6>(v83_acc, v101_bc, v65_data);
          tensorforge::fmacdpp16<7>(v83_acc, v101_bc, v66_data);
          tensorforge::fmacdpp16<8>(v83_acc, v101_bc, v67_data);
          tensorforge::fmacdpp16<9>(v83_acc, v101_bc, v68_data);
          tensorforge::fmacdpp16<10>(v83_acc, v101_bc, v69_data);
          tensorforge::fmacdpp16<11>(v83_acc, v101_bc, v70_data);
          tensorforge::fmacdpp16<12>(v84_acc, v101_bc, v59_data);
          tensorforge::fmacdpp16<13>(v84_acc, v101_bc, v60_data);
          tensorforge::fmacdpp16<14>(v84_acc, v101_bc, v61_data);
          tensorforge::fmacdpp16<15>(v84_acc, v101_bc, v62_data);
          float v102_lin = r1[5];
          float v103_bc = tensorforge::broadcast<32, 16, 0>(v102_lin);
          tensorforge::fmacdpp16<0>(v84_acc, v103_bc, v63_data);
          tensorforge::fmacdpp16<1>(v84_acc, v103_bc, v64_data);
          tensorforge::fmacdpp16<2>(v84_acc, v103_bc, v65_data);
          tensorforge::fmacdpp16<3>(v84_acc, v103_bc, v66_data);
          tensorforge::fmacdpp16<4>(v84_acc, v103_bc, v67_data);
          tensorforge::fmacdpp16<5>(v84_acc, v103_bc, v68_data);
          tensorforge::fmacdpp16<6>(v84_acc, v103_bc, v69_data);
          tensorforge::fmacdpp16<7>(v84_acc, v103_bc, v70_data);
          tensorforge::fmacdpp16<8>(v85_acc, v103_bc, v59_data);
          tensorforge::fmacdpp16<9>(v85_acc, v103_bc, v60_data);
          tensorforge::fmacdpp16<10>(v85_acc, v103_bc, v61_data);
          tensorforge::fmacdpp16<11>(v85_acc, v103_bc, v62_data);
          tensorforge::fmacdpp16<12>(v85_acc, v103_bc, v63_data);
          tensorforge::fmacdpp16<13>(v85_acc, v103_bc, v64_data);
          tensorforge::fmacdpp16<14>(v85_acc, v103_bc, v65_data);
          tensorforge::fmacdpp16<15>(v85_acc, v103_bc, v66_data);
          float v104_bc = tensorforge::broadcast<32, 16, 1>(v102_lin);
          tensorforge::fmacdpp16<0>(v85_acc, v104_bc, v67_data);
          tensorforge::fmacdpp16<1>(v85_acc, v104_bc, v68_data);
          tensorforge::fmacdpp16<2>(v85_acc, v104_bc, v69_data);
          tensorforge::fmacdpp16<3>(v85_acc, v104_bc, v70_data);
          tensorforge::fmacdpp16<4>(v86_acc, v104_bc, v59_data);
          tensorforge::fmacdpp16<5>(v86_acc, v104_bc, v60_data);
          tensorforge::fmacdpp16<6>(v86_acc, v104_bc, v61_data);
          tensorforge::fmacdpp16<7>(v86_acc, v104_bc, v62_data);
          tensorforge::fmacdpp16<8>(v86_acc, v104_bc, v63_data);
          tensorforge::fmacdpp16<9>(v86_acc, v104_bc, v64_data);
          tensorforge::fmacdpp16<10>(v86_acc, v104_bc, v65_data);
          tensorforge::fmacdpp16<11>(v86_acc, v104_bc, v66_data);
          tensorforge::fmacdpp16<12>(v86_acc, v104_bc, v67_data);
          tensorforge::fmacdpp16<13>(v86_acc, v104_bc, v68_data);
          tensorforge::fmacdpp16<14>(v86_acc, v104_bc, v69_data);
          tensorforge::fmacdpp16<15>(v86_acc, v104_bc, v70_data);
          r2[0] = v71_acc;
          r2[1] = v72_acc;
          r2[2] = v73_acc;
          r2[3] = v74_acc;
          r2[4] = v75_acc;
          r2[5] = v76_acc;
          r2[6] = v77_acc;
          r2[7] = v78_acc;
          r2[8] = v79_acc;
          r2[9] = v80_acc;
          r2[10] = v81_acc;
          r2[11] = v82_acc;
          r2[12] = v83_acc;
          r2[13] = v84_acc;
          r2[14] = v85_acc;
          r2[15] = v86_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v108_i0 = 0; v108_i0 < 1; ++v108_i0) {
            int32_t v117_lead = v10_lead + (v108_i0 * 32);
            #pragma unroll
            for (int32_t v109_i1 = 0; v109_i1 < 16; ++v109_i1) {
              int32_t v110_a = v108_i0 + v109_i1;
              float v112_data = r2[(v108_i0 + v109_i1)];
              glb_m0[(v117_lead + (v109_i1 * 32))] = v112_data;
            }
          }
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v121_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v121_lin;
          float v122_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v122_lin;
          float v123_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v123_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v128_i0 = 0; v128_i0 < 1; ++v128_i0) {
            int32_t v133_lead = v128_i0 * 32;
            int32_t v134_lead = v10_lead + v133_lead;
            int32_t v141_lead = v10_lead + v133_lead;
            #pragma unroll
            for (int32_t v129_i1 = 0; v129_i1 < 12; ++v129_i1) {
              int32_t v135_a = v129_i1 * 32;
              int32_t v136_a = v134_lead + v135_a;
              float v144_data = __builtin_nontemporal_load(&glb_m5[(v141_lead + v135_a)]);
              int32_t v145_a = v128_i0 + v129_i1;
              r6[v145_a] = v144_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v147_data = r3[0];
          float v148_data = r3[1];
          float v149_data = r3[2];
          float v150_data = r3[3];
          float v151_data = r3[4];
          float v152_data = r3[5];
          float v153_data = r3[6];
          float v154_data = r3[7];
          float v155_data = r3[8];
          float v156_data = r3[9];
          float v157_data = r3[10];
          float v158_data = r3[11];
          float v159_acc{};
          float v160_acc{};
          float v161_acc{};
          float v162_acc{};
          float v163_acc{};
          float v164_acc{};
          float v165_acc{};
          float v166_acc{};
          float v167_lin = r4[0];
          float v168_bc = tensorforge::broadcast<32, 16, 0>(v167_lin);
          tensorforge::fmacdpp16<0>(v159_acc, v168_bc, v147_data);
          tensorforge::fmacdpp16<1>(v159_acc, v168_bc, v148_data);
          tensorforge::fmacdpp16<2>(v159_acc, v168_bc, v149_data);
          tensorforge::fmacdpp16<3>(v159_acc, v168_bc, v150_data);
          tensorforge::fmacdpp16<4>(v159_acc, v168_bc, v151_data);
          tensorforge::fmacdpp16<5>(v159_acc, v168_bc, v152_data);
          tensorforge::fmacdpp16<6>(v159_acc, v168_bc, v153_data);
          tensorforge::fmacdpp16<7>(v159_acc, v168_bc, v154_data);
          tensorforge::fmacdpp16<8>(v159_acc, v168_bc, v155_data);
          tensorforge::fmacdpp16<9>(v159_acc, v168_bc, v156_data);
          tensorforge::fmacdpp16<10>(v159_acc, v168_bc, v157_data);
          tensorforge::fmacdpp16<11>(v159_acc, v168_bc, v158_data);
          tensorforge::fmacdpp16<12>(v160_acc, v168_bc, v147_data);
          tensorforge::fmacdpp16<13>(v160_acc, v168_bc, v148_data);
          tensorforge::fmacdpp16<14>(v160_acc, v168_bc, v149_data);
          tensorforge::fmacdpp16<15>(v160_acc, v168_bc, v150_data);
          float v169_bc = tensorforge::broadcast<32, 16, 1>(v167_lin);
          tensorforge::fmacdpp16<0>(v160_acc, v169_bc, v151_data);
          tensorforge::fmacdpp16<1>(v160_acc, v169_bc, v152_data);
          tensorforge::fmacdpp16<2>(v160_acc, v169_bc, v153_data);
          tensorforge::fmacdpp16<3>(v160_acc, v169_bc, v154_data);
          tensorforge::fmacdpp16<4>(v160_acc, v169_bc, v155_data);
          tensorforge::fmacdpp16<5>(v160_acc, v169_bc, v156_data);
          tensorforge::fmacdpp16<6>(v160_acc, v169_bc, v157_data);
          tensorforge::fmacdpp16<7>(v160_acc, v169_bc, v158_data);
          tensorforge::fmacdpp16<8>(v161_acc, v169_bc, v147_data);
          tensorforge::fmacdpp16<9>(v161_acc, v169_bc, v148_data);
          tensorforge::fmacdpp16<10>(v161_acc, v169_bc, v149_data);
          tensorforge::fmacdpp16<11>(v161_acc, v169_bc, v150_data);
          tensorforge::fmacdpp16<12>(v161_acc, v169_bc, v151_data);
          tensorforge::fmacdpp16<13>(v161_acc, v169_bc, v152_data);
          tensorforge::fmacdpp16<14>(v161_acc, v169_bc, v153_data);
          tensorforge::fmacdpp16<15>(v161_acc, v169_bc, v154_data);
          float v170_lin = r4[1];
          float v171_bc = tensorforge::broadcast<32, 16, 0>(v170_lin);
          tensorforge::fmacdpp16<0>(v161_acc, v171_bc, v155_data);
          tensorforge::fmacdpp16<1>(v161_acc, v171_bc, v156_data);
          tensorforge::fmacdpp16<2>(v161_acc, v171_bc, v157_data);
          tensorforge::fmacdpp16<3>(v161_acc, v171_bc, v158_data);
          tensorforge::fmacdpp16<4>(v162_acc, v171_bc, v147_data);
          tensorforge::fmacdpp16<5>(v162_acc, v171_bc, v148_data);
          tensorforge::fmacdpp16<6>(v162_acc, v171_bc, v149_data);
          tensorforge::fmacdpp16<7>(v162_acc, v171_bc, v150_data);
          tensorforge::fmacdpp16<8>(v162_acc, v171_bc, v151_data);
          tensorforge::fmacdpp16<9>(v162_acc, v171_bc, v152_data);
          tensorforge::fmacdpp16<10>(v162_acc, v171_bc, v153_data);
          tensorforge::fmacdpp16<11>(v162_acc, v171_bc, v154_data);
          tensorforge::fmacdpp16<12>(v162_acc, v171_bc, v155_data);
          tensorforge::fmacdpp16<13>(v162_acc, v171_bc, v156_data);
          tensorforge::fmacdpp16<14>(v162_acc, v171_bc, v157_data);
          tensorforge::fmacdpp16<15>(v162_acc, v171_bc, v158_data);
          float v172_bc = tensorforge::broadcast<32, 16, 1>(v170_lin);
          tensorforge::fmacdpp16<0>(v163_acc, v172_bc, v147_data);
          tensorforge::fmacdpp16<1>(v163_acc, v172_bc, v148_data);
          tensorforge::fmacdpp16<2>(v163_acc, v172_bc, v149_data);
          tensorforge::fmacdpp16<3>(v163_acc, v172_bc, v150_data);
          tensorforge::fmacdpp16<4>(v163_acc, v172_bc, v151_data);
          tensorforge::fmacdpp16<5>(v163_acc, v172_bc, v152_data);
          tensorforge::fmacdpp16<6>(v163_acc, v172_bc, v153_data);
          tensorforge::fmacdpp16<7>(v163_acc, v172_bc, v154_data);
          tensorforge::fmacdpp16<8>(v163_acc, v172_bc, v155_data);
          tensorforge::fmacdpp16<9>(v163_acc, v172_bc, v156_data);
          tensorforge::fmacdpp16<10>(v163_acc, v172_bc, v157_data);
          tensorforge::fmacdpp16<11>(v163_acc, v172_bc, v158_data);
          tensorforge::fmacdpp16<12>(v164_acc, v172_bc, v147_data);
          tensorforge::fmacdpp16<13>(v164_acc, v172_bc, v148_data);
          tensorforge::fmacdpp16<14>(v164_acc, v172_bc, v149_data);
          tensorforge::fmacdpp16<15>(v164_acc, v172_bc, v150_data);
          float v173_lin = r4[2];
          float v174_bc = tensorforge::broadcast<32, 16, 0>(v173_lin);
          tensorforge::fmacdpp16<0>(v164_acc, v174_bc, v151_data);
          tensorforge::fmacdpp16<1>(v164_acc, v174_bc, v152_data);
          tensorforge::fmacdpp16<2>(v164_acc, v174_bc, v153_data);
          tensorforge::fmacdpp16<3>(v164_acc, v174_bc, v154_data);
          tensorforge::fmacdpp16<4>(v164_acc, v174_bc, v155_data);
          tensorforge::fmacdpp16<5>(v164_acc, v174_bc, v156_data);
          tensorforge::fmacdpp16<6>(v164_acc, v174_bc, v157_data);
          tensorforge::fmacdpp16<7>(v164_acc, v174_bc, v158_data);
          tensorforge::fmacdpp16<8>(v165_acc, v174_bc, v147_data);
          tensorforge::fmacdpp16<9>(v165_acc, v174_bc, v148_data);
          tensorforge::fmacdpp16<10>(v165_acc, v174_bc, v149_data);
          tensorforge::fmacdpp16<11>(v165_acc, v174_bc, v150_data);
          tensorforge::fmacdpp16<12>(v165_acc, v174_bc, v151_data);
          tensorforge::fmacdpp16<13>(v165_acc, v174_bc, v152_data);
          tensorforge::fmacdpp16<14>(v165_acc, v174_bc, v153_data);
          tensorforge::fmacdpp16<15>(v165_acc, v174_bc, v154_data);
          float v175_bc = tensorforge::broadcast<32, 16, 1>(v173_lin);
          tensorforge::fmacdpp16<0>(v165_acc, v175_bc, v155_data);
          tensorforge::fmacdpp16<1>(v165_acc, v175_bc, v156_data);
          tensorforge::fmacdpp16<2>(v165_acc, v175_bc, v157_data);
          tensorforge::fmacdpp16<3>(v165_acc, v175_bc, v158_data);
          tensorforge::fmacdpp16<4>(v166_acc, v175_bc, v147_data);
          tensorforge::fmacdpp16<5>(v166_acc, v175_bc, v148_data);
          tensorforge::fmacdpp16<6>(v166_acc, v175_bc, v149_data);
          tensorforge::fmacdpp16<7>(v166_acc, v175_bc, v150_data);
          tensorforge::fmacdpp16<8>(v166_acc, v175_bc, v151_data);
          tensorforge::fmacdpp16<9>(v166_acc, v175_bc, v152_data);
          tensorforge::fmacdpp16<10>(v166_acc, v175_bc, v153_data);
          tensorforge::fmacdpp16<11>(v166_acc, v175_bc, v154_data);
          tensorforge::fmacdpp16<12>(v166_acc, v175_bc, v155_data);
          tensorforge::fmacdpp16<13>(v166_acc, v175_bc, v156_data);
          tensorforge::fmacdpp16<14>(v166_acc, v175_bc, v157_data);
          tensorforge::fmacdpp16<15>(v166_acc, v175_bc, v158_data);
          r5[0] = v159_acc;
          r5[1] = v160_acc;
          r5[2] = v161_acc;
          r5[3] = v162_acc;
          r5[4] = v163_acc;
          r5[5] = v164_acc;
          r5[6] = v165_acc;
          r5[7] = v166_acc;
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v179_i0 = 0; v179_i0 < 1; ++v179_i0) {
            int32_t v188_lead = v10_lead + (v179_i0 * 32);
            #pragma unroll
            for (int32_t v180_i1 = 0; v180_i1 < 8; ++v180_i1) {
              int32_t v181_a = v179_i0 + v180_i1;
              float v183_data = r5[(v179_i0 + v180_i1)];
              int32_t v190_a = v188_lead + (v180_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v190_a], v183_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v192_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v192_lin;
          float v193_lin = glb_m6[32 + threadIdx.x * 1];
          r7[1] = v193_lin;
          float v194_lin = glb_m6[64 + threadIdx.x * 1];
          r7[2] = v194_lin;
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v196_data = r6[0];
          float v197_data = r6[1];
          float v198_data = r6[2];
          float v199_data = r6[3];
          float v200_data = r6[4];
          float v201_data = r6[5];
          float v202_data = r6[6];
          float v203_data = r6[7];
          float v204_data = r6[8];
          float v205_data = r6[9];
          float v206_data = r6[10];
          float v207_data = r6[11];
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
          tensorforge::fmacdpp16<0>(v208_acc, v217_bc, v196_data);
          tensorforge::fmacdpp16<1>(v208_acc, v217_bc, v197_data);
          tensorforge::fmacdpp16<2>(v208_acc, v217_bc, v198_data);
          tensorforge::fmacdpp16<3>(v208_acc, v217_bc, v199_data);
          tensorforge::fmacdpp16<4>(v208_acc, v217_bc, v200_data);
          tensorforge::fmacdpp16<5>(v208_acc, v217_bc, v201_data);
          tensorforge::fmacdpp16<6>(v208_acc, v217_bc, v202_data);
          tensorforge::fmacdpp16<7>(v208_acc, v217_bc, v203_data);
          tensorforge::fmacdpp16<8>(v208_acc, v217_bc, v204_data);
          tensorforge::fmacdpp16<9>(v208_acc, v217_bc, v205_data);
          tensorforge::fmacdpp16<10>(v208_acc, v217_bc, v206_data);
          tensorforge::fmacdpp16<11>(v208_acc, v217_bc, v207_data);
          tensorforge::fmacdpp16<12>(v209_acc, v217_bc, v196_data);
          tensorforge::fmacdpp16<13>(v209_acc, v217_bc, v197_data);
          tensorforge::fmacdpp16<14>(v209_acc, v217_bc, v198_data);
          tensorforge::fmacdpp16<15>(v209_acc, v217_bc, v199_data);
          float v218_bc = tensorforge::broadcast<32, 16, 1>(v216_lin);
          tensorforge::fmacdpp16<0>(v209_acc, v218_bc, v200_data);
          tensorforge::fmacdpp16<1>(v209_acc, v218_bc, v201_data);
          tensorforge::fmacdpp16<2>(v209_acc, v218_bc, v202_data);
          tensorforge::fmacdpp16<3>(v209_acc, v218_bc, v203_data);
          tensorforge::fmacdpp16<4>(v209_acc, v218_bc, v204_data);
          tensorforge::fmacdpp16<5>(v209_acc, v218_bc, v205_data);
          tensorforge::fmacdpp16<6>(v209_acc, v218_bc, v206_data);
          tensorforge::fmacdpp16<7>(v209_acc, v218_bc, v207_data);
          tensorforge::fmacdpp16<8>(v210_acc, v218_bc, v196_data);
          tensorforge::fmacdpp16<9>(v210_acc, v218_bc, v197_data);
          tensorforge::fmacdpp16<10>(v210_acc, v218_bc, v198_data);
          tensorforge::fmacdpp16<11>(v210_acc, v218_bc, v199_data);
          tensorforge::fmacdpp16<12>(v210_acc, v218_bc, v200_data);
          tensorforge::fmacdpp16<13>(v210_acc, v218_bc, v201_data);
          tensorforge::fmacdpp16<14>(v210_acc, v218_bc, v202_data);
          tensorforge::fmacdpp16<15>(v210_acc, v218_bc, v203_data);
          float v219_lin = r7[1];
          float v220_bc = tensorforge::broadcast<32, 16, 0>(v219_lin);
          tensorforge::fmacdpp16<0>(v210_acc, v220_bc, v204_data);
          tensorforge::fmacdpp16<1>(v210_acc, v220_bc, v205_data);
          tensorforge::fmacdpp16<2>(v210_acc, v220_bc, v206_data);
          tensorforge::fmacdpp16<3>(v210_acc, v220_bc, v207_data);
          tensorforge::fmacdpp16<4>(v211_acc, v220_bc, v196_data);
          tensorforge::fmacdpp16<5>(v211_acc, v220_bc, v197_data);
          tensorforge::fmacdpp16<6>(v211_acc, v220_bc, v198_data);
          tensorforge::fmacdpp16<7>(v211_acc, v220_bc, v199_data);
          tensorforge::fmacdpp16<8>(v211_acc, v220_bc, v200_data);
          tensorforge::fmacdpp16<9>(v211_acc, v220_bc, v201_data);
          tensorforge::fmacdpp16<10>(v211_acc, v220_bc, v202_data);
          tensorforge::fmacdpp16<11>(v211_acc, v220_bc, v203_data);
          tensorforge::fmacdpp16<12>(v211_acc, v220_bc, v204_data);
          tensorforge::fmacdpp16<13>(v211_acc, v220_bc, v205_data);
          tensorforge::fmacdpp16<14>(v211_acc, v220_bc, v206_data);
          tensorforge::fmacdpp16<15>(v211_acc, v220_bc, v207_data);
          float v221_bc = tensorforge::broadcast<32, 16, 1>(v219_lin);
          tensorforge::fmacdpp16<0>(v212_acc, v221_bc, v196_data);
          tensorforge::fmacdpp16<1>(v212_acc, v221_bc, v197_data);
          tensorforge::fmacdpp16<2>(v212_acc, v221_bc, v198_data);
          tensorforge::fmacdpp16<3>(v212_acc, v221_bc, v199_data);
          tensorforge::fmacdpp16<4>(v212_acc, v221_bc, v200_data);
          tensorforge::fmacdpp16<5>(v212_acc, v221_bc, v201_data);
          tensorforge::fmacdpp16<6>(v212_acc, v221_bc, v202_data);
          tensorforge::fmacdpp16<7>(v212_acc, v221_bc, v203_data);
          tensorforge::fmacdpp16<8>(v212_acc, v221_bc, v204_data);
          tensorforge::fmacdpp16<9>(v212_acc, v221_bc, v205_data);
          tensorforge::fmacdpp16<10>(v212_acc, v221_bc, v206_data);
          tensorforge::fmacdpp16<11>(v212_acc, v221_bc, v207_data);
          tensorforge::fmacdpp16<12>(v213_acc, v221_bc, v196_data);
          tensorforge::fmacdpp16<13>(v213_acc, v221_bc, v197_data);
          tensorforge::fmacdpp16<14>(v213_acc, v221_bc, v198_data);
          tensorforge::fmacdpp16<15>(v213_acc, v221_bc, v199_data);
          float v222_lin = r7[2];
          float v223_bc = tensorforge::broadcast<32, 16, 0>(v222_lin);
          tensorforge::fmacdpp16<0>(v213_acc, v223_bc, v200_data);
          tensorforge::fmacdpp16<1>(v213_acc, v223_bc, v201_data);
          tensorforge::fmacdpp16<2>(v213_acc, v223_bc, v202_data);
          tensorforge::fmacdpp16<3>(v213_acc, v223_bc, v203_data);
          tensorforge::fmacdpp16<4>(v213_acc, v223_bc, v204_data);
          tensorforge::fmacdpp16<5>(v213_acc, v223_bc, v205_data);
          tensorforge::fmacdpp16<6>(v213_acc, v223_bc, v206_data);
          tensorforge::fmacdpp16<7>(v213_acc, v223_bc, v207_data);
          tensorforge::fmacdpp16<8>(v214_acc, v223_bc, v196_data);
          tensorforge::fmacdpp16<9>(v214_acc, v223_bc, v197_data);
          tensorforge::fmacdpp16<10>(v214_acc, v223_bc, v198_data);
          tensorforge::fmacdpp16<11>(v214_acc, v223_bc, v199_data);
          tensorforge::fmacdpp16<12>(v214_acc, v223_bc, v200_data);
          tensorforge::fmacdpp16<13>(v214_acc, v223_bc, v201_data);
          tensorforge::fmacdpp16<14>(v214_acc, v223_bc, v202_data);
          tensorforge::fmacdpp16<15>(v214_acc, v223_bc, v203_data);
          float v224_bc = tensorforge::broadcast<32, 16, 1>(v222_lin);
          tensorforge::fmacdpp16<0>(v214_acc, v224_bc, v204_data);
          tensorforge::fmacdpp16<1>(v214_acc, v224_bc, v205_data);
          tensorforge::fmacdpp16<2>(v214_acc, v224_bc, v206_data);
          tensorforge::fmacdpp16<3>(v214_acc, v224_bc, v207_data);
          tensorforge::fmacdpp16<4>(v215_acc, v224_bc, v196_data);
          tensorforge::fmacdpp16<5>(v215_acc, v224_bc, v197_data);
          tensorforge::fmacdpp16<6>(v215_acc, v224_bc, v198_data);
          tensorforge::fmacdpp16<7>(v215_acc, v224_bc, v199_data);
          tensorforge::fmacdpp16<8>(v215_acc, v224_bc, v200_data);
          tensorforge::fmacdpp16<9>(v215_acc, v224_bc, v201_data);
          tensorforge::fmacdpp16<10>(v215_acc, v224_bc, v202_data);
          tensorforge::fmacdpp16<11>(v215_acc, v224_bc, v203_data);
          tensorforge::fmacdpp16<12>(v215_acc, v224_bc, v204_data);
          tensorforge::fmacdpp16<13>(v215_acc, v224_bc, v205_data);
          tensorforge::fmacdpp16<14>(v215_acc, v224_bc, v206_data);
          tensorforge::fmacdpp16<15>(v215_acc, v224_bc, v207_data);
          r8[0] = v208_acc;
          r8[1] = v209_acc;
          r8[2] = v210_acc;
          r8[3] = v211_acc;
          r8[4] = v212_acc;
          r8[5] = v213_acc;
          r8[6] = v214_acc;
          r8[7] = v215_acc;
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v228_i0 = 0; v228_i0 < 1; ++v228_i0) {
            int32_t v237_lead = v10_lead + (v228_i0 * 32);
            #pragma unroll
            for (int32_t v229_i1 = 0; v229_i1 < 8; ++v229_i1) {
              int32_t v230_a = v228_i0 + v229_i1;
              float v232_data = r8[(v228_i0 + v229_i1)];
              int32_t v240_a = v237_lead + ((v229_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v240_a], v232_data);
            }
          }
        }
      }
    }
  }
}

