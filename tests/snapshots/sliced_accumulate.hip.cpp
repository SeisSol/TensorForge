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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
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
          int32_t v14_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
            int32_t v20_lead = v15_i0 * 32;
            int32_t v21_lead = v14_lead + v20_lead;
            int32_t v28_lead = v14_lead + v20_lead;
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 12; ++v16_i1) {
              int32_t v22_a = v16_i1 * 32;
              int32_t v23_a = v21_lead + v22_a;
              float v31_data = __builtin_nontemporal_load(&glb_m1[(v28_lead + v22_a)]);
              r0[(v15_i0 + v16_i1)] = v31_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v34_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v34_lin;
          float v35_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v35_lin;
          float v36_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v36_lin;
          float v37_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v37_lin;
          float v38_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v38_lin;
          float v39_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v39_lin;
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v44_i0 = 0; v44_i0 < 1; ++v44_i0) {
            int32_t v49_lead = v44_i0 * 32;
            int32_t v50_lead = v14_lead + v49_lead;
            int32_t v57_lead = v14_lead + v49_lead;
            #pragma unroll
            for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
              int32_t v51_a = v45_i1 * 32;
              int32_t v52_a = v50_lead + v51_a;
              float v60_data = __builtin_nontemporal_load(&glb_m3[(v57_lead + v51_a)]);
              r3[(v44_i0 + v45_i1)] = v60_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          float v63_data = r0[0];
          float v64_data = r0[1];
          float v65_data = r0[2];
          float v66_data = r0[3];
          float v67_data = r0[4];
          float v68_data = r0[5];
          float v69_data = r0[6];
          float v70_data = r0[7];
          float v71_data = r0[8];
          float v72_data = r0[9];
          float v73_data = r0[10];
          float v74_data = r0[11];
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
          float v87_acc{};
          float v88_acc{};
          float v89_acc{};
          float v90_acc{};
          float v91_lin = r1[0];
          float v92_bc = tensorforge::broadcast<32, 16, 0>(v91_lin);
          tensorforge::fmacdpp16<0>(v75_acc, v92_bc, v63_data);
          tensorforge::fmacdpp16<1>(v75_acc, v92_bc, v64_data);
          tensorforge::fmacdpp16<2>(v75_acc, v92_bc, v65_data);
          tensorforge::fmacdpp16<3>(v75_acc, v92_bc, v66_data);
          tensorforge::fmacdpp16<4>(v75_acc, v92_bc, v67_data);
          tensorforge::fmacdpp16<5>(v75_acc, v92_bc, v68_data);
          tensorforge::fmacdpp16<6>(v75_acc, v92_bc, v69_data);
          tensorforge::fmacdpp16<7>(v75_acc, v92_bc, v70_data);
          tensorforge::fmacdpp16<8>(v75_acc, v92_bc, v71_data);
          tensorforge::fmacdpp16<9>(v75_acc, v92_bc, v72_data);
          tensorforge::fmacdpp16<10>(v75_acc, v92_bc, v73_data);
          tensorforge::fmacdpp16<11>(v75_acc, v92_bc, v74_data);
          tensorforge::fmacdpp16<12>(v76_acc, v92_bc, v63_data);
          tensorforge::fmacdpp16<13>(v76_acc, v92_bc, v64_data);
          tensorforge::fmacdpp16<14>(v76_acc, v92_bc, v65_data);
          tensorforge::fmacdpp16<15>(v76_acc, v92_bc, v66_data);
          float v93_bc = tensorforge::broadcast<32, 16, 1>(v91_lin);
          tensorforge::fmacdpp16<0>(v76_acc, v93_bc, v67_data);
          tensorforge::fmacdpp16<1>(v76_acc, v93_bc, v68_data);
          tensorforge::fmacdpp16<2>(v76_acc, v93_bc, v69_data);
          tensorforge::fmacdpp16<3>(v76_acc, v93_bc, v70_data);
          tensorforge::fmacdpp16<4>(v76_acc, v93_bc, v71_data);
          tensorforge::fmacdpp16<5>(v76_acc, v93_bc, v72_data);
          tensorforge::fmacdpp16<6>(v76_acc, v93_bc, v73_data);
          tensorforge::fmacdpp16<7>(v76_acc, v93_bc, v74_data);
          tensorforge::fmacdpp16<8>(v77_acc, v93_bc, v63_data);
          tensorforge::fmacdpp16<9>(v77_acc, v93_bc, v64_data);
          tensorforge::fmacdpp16<10>(v77_acc, v93_bc, v65_data);
          tensorforge::fmacdpp16<11>(v77_acc, v93_bc, v66_data);
          tensorforge::fmacdpp16<12>(v77_acc, v93_bc, v67_data);
          tensorforge::fmacdpp16<13>(v77_acc, v93_bc, v68_data);
          tensorforge::fmacdpp16<14>(v77_acc, v93_bc, v69_data);
          tensorforge::fmacdpp16<15>(v77_acc, v93_bc, v70_data);
          float v94_lin = r1[1];
          float v95_bc = tensorforge::broadcast<32, 16, 0>(v94_lin);
          tensorforge::fmacdpp16<0>(v77_acc, v95_bc, v71_data);
          tensorforge::fmacdpp16<1>(v77_acc, v95_bc, v72_data);
          tensorforge::fmacdpp16<2>(v77_acc, v95_bc, v73_data);
          tensorforge::fmacdpp16<3>(v77_acc, v95_bc, v74_data);
          tensorforge::fmacdpp16<4>(v78_acc, v95_bc, v63_data);
          tensorforge::fmacdpp16<5>(v78_acc, v95_bc, v64_data);
          tensorforge::fmacdpp16<6>(v78_acc, v95_bc, v65_data);
          tensorforge::fmacdpp16<7>(v78_acc, v95_bc, v66_data);
          tensorforge::fmacdpp16<8>(v78_acc, v95_bc, v67_data);
          tensorforge::fmacdpp16<9>(v78_acc, v95_bc, v68_data);
          tensorforge::fmacdpp16<10>(v78_acc, v95_bc, v69_data);
          tensorforge::fmacdpp16<11>(v78_acc, v95_bc, v70_data);
          tensorforge::fmacdpp16<12>(v78_acc, v95_bc, v71_data);
          tensorforge::fmacdpp16<13>(v78_acc, v95_bc, v72_data);
          tensorforge::fmacdpp16<14>(v78_acc, v95_bc, v73_data);
          tensorforge::fmacdpp16<15>(v78_acc, v95_bc, v74_data);
          float v96_bc = tensorforge::broadcast<32, 16, 1>(v94_lin);
          tensorforge::fmacdpp16<0>(v79_acc, v96_bc, v63_data);
          tensorforge::fmacdpp16<1>(v79_acc, v96_bc, v64_data);
          tensorforge::fmacdpp16<2>(v79_acc, v96_bc, v65_data);
          tensorforge::fmacdpp16<3>(v79_acc, v96_bc, v66_data);
          tensorforge::fmacdpp16<4>(v79_acc, v96_bc, v67_data);
          tensorforge::fmacdpp16<5>(v79_acc, v96_bc, v68_data);
          tensorforge::fmacdpp16<6>(v79_acc, v96_bc, v69_data);
          tensorforge::fmacdpp16<7>(v79_acc, v96_bc, v70_data);
          tensorforge::fmacdpp16<8>(v79_acc, v96_bc, v71_data);
          tensorforge::fmacdpp16<9>(v79_acc, v96_bc, v72_data);
          tensorforge::fmacdpp16<10>(v79_acc, v96_bc, v73_data);
          tensorforge::fmacdpp16<11>(v79_acc, v96_bc, v74_data);
          tensorforge::fmacdpp16<12>(v80_acc, v96_bc, v63_data);
          tensorforge::fmacdpp16<13>(v80_acc, v96_bc, v64_data);
          tensorforge::fmacdpp16<14>(v80_acc, v96_bc, v65_data);
          tensorforge::fmacdpp16<15>(v80_acc, v96_bc, v66_data);
          float v97_lin = r1[2];
          float v98_bc = tensorforge::broadcast<32, 16, 0>(v97_lin);
          tensorforge::fmacdpp16<0>(v80_acc, v98_bc, v67_data);
          tensorforge::fmacdpp16<1>(v80_acc, v98_bc, v68_data);
          tensorforge::fmacdpp16<2>(v80_acc, v98_bc, v69_data);
          tensorforge::fmacdpp16<3>(v80_acc, v98_bc, v70_data);
          tensorforge::fmacdpp16<4>(v80_acc, v98_bc, v71_data);
          tensorforge::fmacdpp16<5>(v80_acc, v98_bc, v72_data);
          tensorforge::fmacdpp16<6>(v80_acc, v98_bc, v73_data);
          tensorforge::fmacdpp16<7>(v80_acc, v98_bc, v74_data);
          tensorforge::fmacdpp16<8>(v81_acc, v98_bc, v63_data);
          tensorforge::fmacdpp16<9>(v81_acc, v98_bc, v64_data);
          tensorforge::fmacdpp16<10>(v81_acc, v98_bc, v65_data);
          tensorforge::fmacdpp16<11>(v81_acc, v98_bc, v66_data);
          tensorforge::fmacdpp16<12>(v81_acc, v98_bc, v67_data);
          tensorforge::fmacdpp16<13>(v81_acc, v98_bc, v68_data);
          tensorforge::fmacdpp16<14>(v81_acc, v98_bc, v69_data);
          tensorforge::fmacdpp16<15>(v81_acc, v98_bc, v70_data);
          float v99_bc = tensorforge::broadcast<32, 16, 1>(v97_lin);
          tensorforge::fmacdpp16<0>(v81_acc, v99_bc, v71_data);
          tensorforge::fmacdpp16<1>(v81_acc, v99_bc, v72_data);
          tensorforge::fmacdpp16<2>(v81_acc, v99_bc, v73_data);
          tensorforge::fmacdpp16<3>(v81_acc, v99_bc, v74_data);
          tensorforge::fmacdpp16<4>(v82_acc, v99_bc, v63_data);
          tensorforge::fmacdpp16<5>(v82_acc, v99_bc, v64_data);
          tensorforge::fmacdpp16<6>(v82_acc, v99_bc, v65_data);
          tensorforge::fmacdpp16<7>(v82_acc, v99_bc, v66_data);
          tensorforge::fmacdpp16<8>(v82_acc, v99_bc, v67_data);
          tensorforge::fmacdpp16<9>(v82_acc, v99_bc, v68_data);
          tensorforge::fmacdpp16<10>(v82_acc, v99_bc, v69_data);
          tensorforge::fmacdpp16<11>(v82_acc, v99_bc, v70_data);
          tensorforge::fmacdpp16<12>(v82_acc, v99_bc, v71_data);
          tensorforge::fmacdpp16<13>(v82_acc, v99_bc, v72_data);
          tensorforge::fmacdpp16<14>(v82_acc, v99_bc, v73_data);
          tensorforge::fmacdpp16<15>(v82_acc, v99_bc, v74_data);
          float v100_lin = r1[3];
          float v101_bc = tensorforge::broadcast<32, 16, 0>(v100_lin);
          tensorforge::fmacdpp16<0>(v83_acc, v101_bc, v63_data);
          tensorforge::fmacdpp16<1>(v83_acc, v101_bc, v64_data);
          tensorforge::fmacdpp16<2>(v83_acc, v101_bc, v65_data);
          tensorforge::fmacdpp16<3>(v83_acc, v101_bc, v66_data);
          tensorforge::fmacdpp16<4>(v83_acc, v101_bc, v67_data);
          tensorforge::fmacdpp16<5>(v83_acc, v101_bc, v68_data);
          tensorforge::fmacdpp16<6>(v83_acc, v101_bc, v69_data);
          tensorforge::fmacdpp16<7>(v83_acc, v101_bc, v70_data);
          tensorforge::fmacdpp16<8>(v83_acc, v101_bc, v71_data);
          tensorforge::fmacdpp16<9>(v83_acc, v101_bc, v72_data);
          tensorforge::fmacdpp16<10>(v83_acc, v101_bc, v73_data);
          tensorforge::fmacdpp16<11>(v83_acc, v101_bc, v74_data);
          tensorforge::fmacdpp16<12>(v84_acc, v101_bc, v63_data);
          tensorforge::fmacdpp16<13>(v84_acc, v101_bc, v64_data);
          tensorforge::fmacdpp16<14>(v84_acc, v101_bc, v65_data);
          tensorforge::fmacdpp16<15>(v84_acc, v101_bc, v66_data);
          float v102_bc = tensorforge::broadcast<32, 16, 1>(v100_lin);
          tensorforge::fmacdpp16<0>(v84_acc, v102_bc, v67_data);
          tensorforge::fmacdpp16<1>(v84_acc, v102_bc, v68_data);
          tensorforge::fmacdpp16<2>(v84_acc, v102_bc, v69_data);
          tensorforge::fmacdpp16<3>(v84_acc, v102_bc, v70_data);
          tensorforge::fmacdpp16<4>(v84_acc, v102_bc, v71_data);
          tensorforge::fmacdpp16<5>(v84_acc, v102_bc, v72_data);
          tensorforge::fmacdpp16<6>(v84_acc, v102_bc, v73_data);
          tensorforge::fmacdpp16<7>(v84_acc, v102_bc, v74_data);
          tensorforge::fmacdpp16<8>(v85_acc, v102_bc, v63_data);
          tensorforge::fmacdpp16<9>(v85_acc, v102_bc, v64_data);
          tensorforge::fmacdpp16<10>(v85_acc, v102_bc, v65_data);
          tensorforge::fmacdpp16<11>(v85_acc, v102_bc, v66_data);
          tensorforge::fmacdpp16<12>(v85_acc, v102_bc, v67_data);
          tensorforge::fmacdpp16<13>(v85_acc, v102_bc, v68_data);
          tensorforge::fmacdpp16<14>(v85_acc, v102_bc, v69_data);
          tensorforge::fmacdpp16<15>(v85_acc, v102_bc, v70_data);
          float v103_lin = r1[4];
          float v104_bc = tensorforge::broadcast<32, 16, 0>(v103_lin);
          tensorforge::fmacdpp16<0>(v85_acc, v104_bc, v71_data);
          tensorforge::fmacdpp16<1>(v85_acc, v104_bc, v72_data);
          tensorforge::fmacdpp16<2>(v85_acc, v104_bc, v73_data);
          tensorforge::fmacdpp16<3>(v85_acc, v104_bc, v74_data);
          tensorforge::fmacdpp16<4>(v86_acc, v104_bc, v63_data);
          tensorforge::fmacdpp16<5>(v86_acc, v104_bc, v64_data);
          tensorforge::fmacdpp16<6>(v86_acc, v104_bc, v65_data);
          tensorforge::fmacdpp16<7>(v86_acc, v104_bc, v66_data);
          tensorforge::fmacdpp16<8>(v86_acc, v104_bc, v67_data);
          tensorforge::fmacdpp16<9>(v86_acc, v104_bc, v68_data);
          tensorforge::fmacdpp16<10>(v86_acc, v104_bc, v69_data);
          tensorforge::fmacdpp16<11>(v86_acc, v104_bc, v70_data);
          tensorforge::fmacdpp16<12>(v86_acc, v104_bc, v71_data);
          tensorforge::fmacdpp16<13>(v86_acc, v104_bc, v72_data);
          tensorforge::fmacdpp16<14>(v86_acc, v104_bc, v73_data);
          tensorforge::fmacdpp16<15>(v86_acc, v104_bc, v74_data);
          float v105_bc = tensorforge::broadcast<32, 16, 1>(v103_lin);
          tensorforge::fmacdpp16<0>(v87_acc, v105_bc, v63_data);
          tensorforge::fmacdpp16<1>(v87_acc, v105_bc, v64_data);
          tensorforge::fmacdpp16<2>(v87_acc, v105_bc, v65_data);
          tensorforge::fmacdpp16<3>(v87_acc, v105_bc, v66_data);
          tensorforge::fmacdpp16<4>(v87_acc, v105_bc, v67_data);
          tensorforge::fmacdpp16<5>(v87_acc, v105_bc, v68_data);
          tensorforge::fmacdpp16<6>(v87_acc, v105_bc, v69_data);
          tensorforge::fmacdpp16<7>(v87_acc, v105_bc, v70_data);
          tensorforge::fmacdpp16<8>(v87_acc, v105_bc, v71_data);
          tensorforge::fmacdpp16<9>(v87_acc, v105_bc, v72_data);
          tensorforge::fmacdpp16<10>(v87_acc, v105_bc, v73_data);
          tensorforge::fmacdpp16<11>(v87_acc, v105_bc, v74_data);
          tensorforge::fmacdpp16<12>(v88_acc, v105_bc, v63_data);
          tensorforge::fmacdpp16<13>(v88_acc, v105_bc, v64_data);
          tensorforge::fmacdpp16<14>(v88_acc, v105_bc, v65_data);
          tensorforge::fmacdpp16<15>(v88_acc, v105_bc, v66_data);
          float v106_lin = r1[5];
          float v107_bc = tensorforge::broadcast<32, 16, 0>(v106_lin);
          tensorforge::fmacdpp16<0>(v88_acc, v107_bc, v67_data);
          tensorforge::fmacdpp16<1>(v88_acc, v107_bc, v68_data);
          tensorforge::fmacdpp16<2>(v88_acc, v107_bc, v69_data);
          tensorforge::fmacdpp16<3>(v88_acc, v107_bc, v70_data);
          tensorforge::fmacdpp16<4>(v88_acc, v107_bc, v71_data);
          tensorforge::fmacdpp16<5>(v88_acc, v107_bc, v72_data);
          tensorforge::fmacdpp16<6>(v88_acc, v107_bc, v73_data);
          tensorforge::fmacdpp16<7>(v88_acc, v107_bc, v74_data);
          tensorforge::fmacdpp16<8>(v89_acc, v107_bc, v63_data);
          tensorforge::fmacdpp16<9>(v89_acc, v107_bc, v64_data);
          tensorforge::fmacdpp16<10>(v89_acc, v107_bc, v65_data);
          tensorforge::fmacdpp16<11>(v89_acc, v107_bc, v66_data);
          tensorforge::fmacdpp16<12>(v89_acc, v107_bc, v67_data);
          tensorforge::fmacdpp16<13>(v89_acc, v107_bc, v68_data);
          tensorforge::fmacdpp16<14>(v89_acc, v107_bc, v69_data);
          tensorforge::fmacdpp16<15>(v89_acc, v107_bc, v70_data);
          float v108_bc = tensorforge::broadcast<32, 16, 1>(v106_lin);
          tensorforge::fmacdpp16<0>(v89_acc, v108_bc, v71_data);
          tensorforge::fmacdpp16<1>(v89_acc, v108_bc, v72_data);
          tensorforge::fmacdpp16<2>(v89_acc, v108_bc, v73_data);
          tensorforge::fmacdpp16<3>(v89_acc, v108_bc, v74_data);
          tensorforge::fmacdpp16<4>(v90_acc, v108_bc, v63_data);
          tensorforge::fmacdpp16<5>(v90_acc, v108_bc, v64_data);
          tensorforge::fmacdpp16<6>(v90_acc, v108_bc, v65_data);
          tensorforge::fmacdpp16<7>(v90_acc, v108_bc, v66_data);
          tensorforge::fmacdpp16<8>(v90_acc, v108_bc, v67_data);
          tensorforge::fmacdpp16<9>(v90_acc, v108_bc, v68_data);
          tensorforge::fmacdpp16<10>(v90_acc, v108_bc, v69_data);
          tensorforge::fmacdpp16<11>(v90_acc, v108_bc, v70_data);
          tensorforge::fmacdpp16<12>(v90_acc, v108_bc, v71_data);
          tensorforge::fmacdpp16<13>(v90_acc, v108_bc, v72_data);
          tensorforge::fmacdpp16<14>(v90_acc, v108_bc, v73_data);
          tensorforge::fmacdpp16<15>(v90_acc, v108_bc, v74_data);
          r2[0] = v75_acc;
          r2[1] = v76_acc;
          r2[2] = v77_acc;
          r2[3] = v78_acc;
          r2[4] = v79_acc;
          r2[5] = v80_acc;
          r2[6] = v81_acc;
          r2[7] = v82_acc;
          r2[8] = v83_acc;
          r2[9] = v84_acc;
          r2[10] = v85_acc;
          r2[11] = v86_acc;
          r2[12] = v87_acc;
          r2[13] = v88_acc;
          r2[14] = v89_acc;
          r2[15] = v90_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v112_i0 = 0; v112_i0 < 1; ++v112_i0) {
            int32_t v121_lead = v14_lead + (v112_i0 * 32);
            #pragma unroll
            for (int32_t v113_i1 = 0; v113_i1 < 16; ++v113_i1) {
              int32_t v114_a = v112_i0 + v113_i1;
              float v116_data = r2[(v112_i0 + v113_i1)];
              glb_m0[(v121_lead + (v113_i1 * 32))] = v116_data;
            }
          }
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v125_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v125_lin;
          float v126_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v126_lin;
          float v127_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v127_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v132_i0 = 0; v132_i0 < 1; ++v132_i0) {
            int32_t v137_lead = v132_i0 * 32;
            int32_t v138_lead = v14_lead + v137_lead;
            int32_t v145_lead = v14_lead + v137_lead;
            #pragma unroll
            for (int32_t v133_i1 = 0; v133_i1 < 12; ++v133_i1) {
              int32_t v139_a = v133_i1 * 32;
              int32_t v140_a = v138_lead + v139_a;
              float v148_data = __builtin_nontemporal_load(&glb_m5[(v145_lead + v139_a)]);
              r6[(v132_i0 + v133_i1)] = v148_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v151_data = r3[0];
          float v152_data = r3[1];
          float v153_data = r3[2];
          float v154_data = r3[3];
          float v155_data = r3[4];
          float v156_data = r3[5];
          float v157_data = r3[6];
          float v158_data = r3[7];
          float v159_data = r3[8];
          float v160_data = r3[9];
          float v161_data = r3[10];
          float v162_data = r3[11];
          float v163_acc{};
          float v164_acc{};
          float v165_acc{};
          float v166_acc{};
          float v167_acc{};
          float v168_acc{};
          float v169_acc{};
          float v170_acc{};
          float v171_lin = r4[0];
          float v172_bc = tensorforge::broadcast<32, 16, 0>(v171_lin);
          tensorforge::fmacdpp16<0>(v163_acc, v172_bc, v151_data);
          tensorforge::fmacdpp16<1>(v163_acc, v172_bc, v152_data);
          tensorforge::fmacdpp16<2>(v163_acc, v172_bc, v153_data);
          tensorforge::fmacdpp16<3>(v163_acc, v172_bc, v154_data);
          tensorforge::fmacdpp16<4>(v163_acc, v172_bc, v155_data);
          tensorforge::fmacdpp16<5>(v163_acc, v172_bc, v156_data);
          tensorforge::fmacdpp16<6>(v163_acc, v172_bc, v157_data);
          tensorforge::fmacdpp16<7>(v163_acc, v172_bc, v158_data);
          tensorforge::fmacdpp16<8>(v163_acc, v172_bc, v159_data);
          tensorforge::fmacdpp16<9>(v163_acc, v172_bc, v160_data);
          tensorforge::fmacdpp16<10>(v163_acc, v172_bc, v161_data);
          tensorforge::fmacdpp16<11>(v163_acc, v172_bc, v162_data);
          tensorforge::fmacdpp16<12>(v164_acc, v172_bc, v151_data);
          tensorforge::fmacdpp16<13>(v164_acc, v172_bc, v152_data);
          tensorforge::fmacdpp16<14>(v164_acc, v172_bc, v153_data);
          tensorforge::fmacdpp16<15>(v164_acc, v172_bc, v154_data);
          float v173_bc = tensorforge::broadcast<32, 16, 1>(v171_lin);
          tensorforge::fmacdpp16<0>(v164_acc, v173_bc, v155_data);
          tensorforge::fmacdpp16<1>(v164_acc, v173_bc, v156_data);
          tensorforge::fmacdpp16<2>(v164_acc, v173_bc, v157_data);
          tensorforge::fmacdpp16<3>(v164_acc, v173_bc, v158_data);
          tensorforge::fmacdpp16<4>(v164_acc, v173_bc, v159_data);
          tensorforge::fmacdpp16<5>(v164_acc, v173_bc, v160_data);
          tensorforge::fmacdpp16<6>(v164_acc, v173_bc, v161_data);
          tensorforge::fmacdpp16<7>(v164_acc, v173_bc, v162_data);
          tensorforge::fmacdpp16<8>(v165_acc, v173_bc, v151_data);
          tensorforge::fmacdpp16<9>(v165_acc, v173_bc, v152_data);
          tensorforge::fmacdpp16<10>(v165_acc, v173_bc, v153_data);
          tensorforge::fmacdpp16<11>(v165_acc, v173_bc, v154_data);
          tensorforge::fmacdpp16<12>(v165_acc, v173_bc, v155_data);
          tensorforge::fmacdpp16<13>(v165_acc, v173_bc, v156_data);
          tensorforge::fmacdpp16<14>(v165_acc, v173_bc, v157_data);
          tensorforge::fmacdpp16<15>(v165_acc, v173_bc, v158_data);
          float v174_lin = r4[1];
          float v175_bc = tensorforge::broadcast<32, 16, 0>(v174_lin);
          tensorforge::fmacdpp16<0>(v165_acc, v175_bc, v159_data);
          tensorforge::fmacdpp16<1>(v165_acc, v175_bc, v160_data);
          tensorforge::fmacdpp16<2>(v165_acc, v175_bc, v161_data);
          tensorforge::fmacdpp16<3>(v165_acc, v175_bc, v162_data);
          tensorforge::fmacdpp16<4>(v166_acc, v175_bc, v151_data);
          tensorforge::fmacdpp16<5>(v166_acc, v175_bc, v152_data);
          tensorforge::fmacdpp16<6>(v166_acc, v175_bc, v153_data);
          tensorforge::fmacdpp16<7>(v166_acc, v175_bc, v154_data);
          tensorforge::fmacdpp16<8>(v166_acc, v175_bc, v155_data);
          tensorforge::fmacdpp16<9>(v166_acc, v175_bc, v156_data);
          tensorforge::fmacdpp16<10>(v166_acc, v175_bc, v157_data);
          tensorforge::fmacdpp16<11>(v166_acc, v175_bc, v158_data);
          tensorforge::fmacdpp16<12>(v166_acc, v175_bc, v159_data);
          tensorforge::fmacdpp16<13>(v166_acc, v175_bc, v160_data);
          tensorforge::fmacdpp16<14>(v166_acc, v175_bc, v161_data);
          tensorforge::fmacdpp16<15>(v166_acc, v175_bc, v162_data);
          float v176_bc = tensorforge::broadcast<32, 16, 1>(v174_lin);
          tensorforge::fmacdpp16<0>(v167_acc, v176_bc, v151_data);
          tensorforge::fmacdpp16<1>(v167_acc, v176_bc, v152_data);
          tensorforge::fmacdpp16<2>(v167_acc, v176_bc, v153_data);
          tensorforge::fmacdpp16<3>(v167_acc, v176_bc, v154_data);
          tensorforge::fmacdpp16<4>(v167_acc, v176_bc, v155_data);
          tensorforge::fmacdpp16<5>(v167_acc, v176_bc, v156_data);
          tensorforge::fmacdpp16<6>(v167_acc, v176_bc, v157_data);
          tensorforge::fmacdpp16<7>(v167_acc, v176_bc, v158_data);
          tensorforge::fmacdpp16<8>(v167_acc, v176_bc, v159_data);
          tensorforge::fmacdpp16<9>(v167_acc, v176_bc, v160_data);
          tensorforge::fmacdpp16<10>(v167_acc, v176_bc, v161_data);
          tensorforge::fmacdpp16<11>(v167_acc, v176_bc, v162_data);
          tensorforge::fmacdpp16<12>(v168_acc, v176_bc, v151_data);
          tensorforge::fmacdpp16<13>(v168_acc, v176_bc, v152_data);
          tensorforge::fmacdpp16<14>(v168_acc, v176_bc, v153_data);
          tensorforge::fmacdpp16<15>(v168_acc, v176_bc, v154_data);
          float v177_lin = r4[2];
          float v178_bc = tensorforge::broadcast<32, 16, 0>(v177_lin);
          tensorforge::fmacdpp16<0>(v168_acc, v178_bc, v155_data);
          tensorforge::fmacdpp16<1>(v168_acc, v178_bc, v156_data);
          tensorforge::fmacdpp16<2>(v168_acc, v178_bc, v157_data);
          tensorforge::fmacdpp16<3>(v168_acc, v178_bc, v158_data);
          tensorforge::fmacdpp16<4>(v168_acc, v178_bc, v159_data);
          tensorforge::fmacdpp16<5>(v168_acc, v178_bc, v160_data);
          tensorforge::fmacdpp16<6>(v168_acc, v178_bc, v161_data);
          tensorforge::fmacdpp16<7>(v168_acc, v178_bc, v162_data);
          tensorforge::fmacdpp16<8>(v169_acc, v178_bc, v151_data);
          tensorforge::fmacdpp16<9>(v169_acc, v178_bc, v152_data);
          tensorforge::fmacdpp16<10>(v169_acc, v178_bc, v153_data);
          tensorforge::fmacdpp16<11>(v169_acc, v178_bc, v154_data);
          tensorforge::fmacdpp16<12>(v169_acc, v178_bc, v155_data);
          tensorforge::fmacdpp16<13>(v169_acc, v178_bc, v156_data);
          tensorforge::fmacdpp16<14>(v169_acc, v178_bc, v157_data);
          tensorforge::fmacdpp16<15>(v169_acc, v178_bc, v158_data);
          float v179_bc = tensorforge::broadcast<32, 16, 1>(v177_lin);
          tensorforge::fmacdpp16<0>(v169_acc, v179_bc, v159_data);
          tensorforge::fmacdpp16<1>(v169_acc, v179_bc, v160_data);
          tensorforge::fmacdpp16<2>(v169_acc, v179_bc, v161_data);
          tensorforge::fmacdpp16<3>(v169_acc, v179_bc, v162_data);
          tensorforge::fmacdpp16<4>(v170_acc, v179_bc, v151_data);
          tensorforge::fmacdpp16<5>(v170_acc, v179_bc, v152_data);
          tensorforge::fmacdpp16<6>(v170_acc, v179_bc, v153_data);
          tensorforge::fmacdpp16<7>(v170_acc, v179_bc, v154_data);
          tensorforge::fmacdpp16<8>(v170_acc, v179_bc, v155_data);
          tensorforge::fmacdpp16<9>(v170_acc, v179_bc, v156_data);
          tensorforge::fmacdpp16<10>(v170_acc, v179_bc, v157_data);
          tensorforge::fmacdpp16<11>(v170_acc, v179_bc, v158_data);
          tensorforge::fmacdpp16<12>(v170_acc, v179_bc, v159_data);
          tensorforge::fmacdpp16<13>(v170_acc, v179_bc, v160_data);
          tensorforge::fmacdpp16<14>(v170_acc, v179_bc, v161_data);
          tensorforge::fmacdpp16<15>(v170_acc, v179_bc, v162_data);
          r5[0] = v163_acc;
          r5[1] = v164_acc;
          r5[2] = v165_acc;
          r5[3] = v166_acc;
          r5[4] = v167_acc;
          r5[5] = v168_acc;
          r5[6] = v169_acc;
          r5[7] = v170_acc;
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v183_i0 = 0; v183_i0 < 1; ++v183_i0) {
            int32_t v192_lead = v14_lead + (v183_i0 * 32);
            #pragma unroll
            for (int32_t v184_i1 = 0; v184_i1 < 8; ++v184_i1) {
              int32_t v185_a = v183_i0 + v184_i1;
              float v187_data = r5[(v183_i0 + v184_i1)];
              int32_t v194_a = v192_lead + (v184_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v194_a], v187_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v196_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v196_lin;
          float v197_lin = glb_m6[32 + threadIdx.x * 1];
          r7[1] = v197_lin;
          float v198_lin = glb_m6[64 + threadIdx.x * 1];
          r7[2] = v198_lin;
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v200_data = r6[0];
          float v201_data = r6[1];
          float v202_data = r6[2];
          float v203_data = r6[3];
          float v204_data = r6[4];
          float v205_data = r6[5];
          float v206_data = r6[6];
          float v207_data = r6[7];
          float v208_data = r6[8];
          float v209_data = r6[9];
          float v210_data = r6[10];
          float v211_data = r6[11];
          float v212_acc{};
          float v213_acc{};
          float v214_acc{};
          float v215_acc{};
          float v216_acc{};
          float v217_acc{};
          float v218_acc{};
          float v219_acc{};
          float v220_lin = r7[0];
          float v221_bc = tensorforge::broadcast<32, 16, 0>(v220_lin);
          tensorforge::fmacdpp16<0>(v212_acc, v221_bc, v200_data);
          tensorforge::fmacdpp16<1>(v212_acc, v221_bc, v201_data);
          tensorforge::fmacdpp16<2>(v212_acc, v221_bc, v202_data);
          tensorforge::fmacdpp16<3>(v212_acc, v221_bc, v203_data);
          tensorforge::fmacdpp16<4>(v212_acc, v221_bc, v204_data);
          tensorforge::fmacdpp16<5>(v212_acc, v221_bc, v205_data);
          tensorforge::fmacdpp16<6>(v212_acc, v221_bc, v206_data);
          tensorforge::fmacdpp16<7>(v212_acc, v221_bc, v207_data);
          tensorforge::fmacdpp16<8>(v212_acc, v221_bc, v208_data);
          tensorforge::fmacdpp16<9>(v212_acc, v221_bc, v209_data);
          tensorforge::fmacdpp16<10>(v212_acc, v221_bc, v210_data);
          tensorforge::fmacdpp16<11>(v212_acc, v221_bc, v211_data);
          tensorforge::fmacdpp16<12>(v213_acc, v221_bc, v200_data);
          tensorforge::fmacdpp16<13>(v213_acc, v221_bc, v201_data);
          tensorforge::fmacdpp16<14>(v213_acc, v221_bc, v202_data);
          tensorforge::fmacdpp16<15>(v213_acc, v221_bc, v203_data);
          float v222_bc = tensorforge::broadcast<32, 16, 1>(v220_lin);
          tensorforge::fmacdpp16<0>(v213_acc, v222_bc, v204_data);
          tensorforge::fmacdpp16<1>(v213_acc, v222_bc, v205_data);
          tensorforge::fmacdpp16<2>(v213_acc, v222_bc, v206_data);
          tensorforge::fmacdpp16<3>(v213_acc, v222_bc, v207_data);
          tensorforge::fmacdpp16<4>(v213_acc, v222_bc, v208_data);
          tensorforge::fmacdpp16<5>(v213_acc, v222_bc, v209_data);
          tensorforge::fmacdpp16<6>(v213_acc, v222_bc, v210_data);
          tensorforge::fmacdpp16<7>(v213_acc, v222_bc, v211_data);
          tensorforge::fmacdpp16<8>(v214_acc, v222_bc, v200_data);
          tensorforge::fmacdpp16<9>(v214_acc, v222_bc, v201_data);
          tensorforge::fmacdpp16<10>(v214_acc, v222_bc, v202_data);
          tensorforge::fmacdpp16<11>(v214_acc, v222_bc, v203_data);
          tensorforge::fmacdpp16<12>(v214_acc, v222_bc, v204_data);
          tensorforge::fmacdpp16<13>(v214_acc, v222_bc, v205_data);
          tensorforge::fmacdpp16<14>(v214_acc, v222_bc, v206_data);
          tensorforge::fmacdpp16<15>(v214_acc, v222_bc, v207_data);
          float v223_lin = r7[1];
          float v224_bc = tensorforge::broadcast<32, 16, 0>(v223_lin);
          tensorforge::fmacdpp16<0>(v214_acc, v224_bc, v208_data);
          tensorforge::fmacdpp16<1>(v214_acc, v224_bc, v209_data);
          tensorforge::fmacdpp16<2>(v214_acc, v224_bc, v210_data);
          tensorforge::fmacdpp16<3>(v214_acc, v224_bc, v211_data);
          tensorforge::fmacdpp16<4>(v215_acc, v224_bc, v200_data);
          tensorforge::fmacdpp16<5>(v215_acc, v224_bc, v201_data);
          tensorforge::fmacdpp16<6>(v215_acc, v224_bc, v202_data);
          tensorforge::fmacdpp16<7>(v215_acc, v224_bc, v203_data);
          tensorforge::fmacdpp16<8>(v215_acc, v224_bc, v204_data);
          tensorforge::fmacdpp16<9>(v215_acc, v224_bc, v205_data);
          tensorforge::fmacdpp16<10>(v215_acc, v224_bc, v206_data);
          tensorforge::fmacdpp16<11>(v215_acc, v224_bc, v207_data);
          tensorforge::fmacdpp16<12>(v215_acc, v224_bc, v208_data);
          tensorforge::fmacdpp16<13>(v215_acc, v224_bc, v209_data);
          tensorforge::fmacdpp16<14>(v215_acc, v224_bc, v210_data);
          tensorforge::fmacdpp16<15>(v215_acc, v224_bc, v211_data);
          float v225_bc = tensorforge::broadcast<32, 16, 1>(v223_lin);
          tensorforge::fmacdpp16<0>(v216_acc, v225_bc, v200_data);
          tensorforge::fmacdpp16<1>(v216_acc, v225_bc, v201_data);
          tensorforge::fmacdpp16<2>(v216_acc, v225_bc, v202_data);
          tensorforge::fmacdpp16<3>(v216_acc, v225_bc, v203_data);
          tensorforge::fmacdpp16<4>(v216_acc, v225_bc, v204_data);
          tensorforge::fmacdpp16<5>(v216_acc, v225_bc, v205_data);
          tensorforge::fmacdpp16<6>(v216_acc, v225_bc, v206_data);
          tensorforge::fmacdpp16<7>(v216_acc, v225_bc, v207_data);
          tensorforge::fmacdpp16<8>(v216_acc, v225_bc, v208_data);
          tensorforge::fmacdpp16<9>(v216_acc, v225_bc, v209_data);
          tensorforge::fmacdpp16<10>(v216_acc, v225_bc, v210_data);
          tensorforge::fmacdpp16<11>(v216_acc, v225_bc, v211_data);
          tensorforge::fmacdpp16<12>(v217_acc, v225_bc, v200_data);
          tensorforge::fmacdpp16<13>(v217_acc, v225_bc, v201_data);
          tensorforge::fmacdpp16<14>(v217_acc, v225_bc, v202_data);
          tensorforge::fmacdpp16<15>(v217_acc, v225_bc, v203_data);
          float v226_lin = r7[2];
          float v227_bc = tensorforge::broadcast<32, 16, 0>(v226_lin);
          tensorforge::fmacdpp16<0>(v217_acc, v227_bc, v204_data);
          tensorforge::fmacdpp16<1>(v217_acc, v227_bc, v205_data);
          tensorforge::fmacdpp16<2>(v217_acc, v227_bc, v206_data);
          tensorforge::fmacdpp16<3>(v217_acc, v227_bc, v207_data);
          tensorforge::fmacdpp16<4>(v217_acc, v227_bc, v208_data);
          tensorforge::fmacdpp16<5>(v217_acc, v227_bc, v209_data);
          tensorforge::fmacdpp16<6>(v217_acc, v227_bc, v210_data);
          tensorforge::fmacdpp16<7>(v217_acc, v227_bc, v211_data);
          tensorforge::fmacdpp16<8>(v218_acc, v227_bc, v200_data);
          tensorforge::fmacdpp16<9>(v218_acc, v227_bc, v201_data);
          tensorforge::fmacdpp16<10>(v218_acc, v227_bc, v202_data);
          tensorforge::fmacdpp16<11>(v218_acc, v227_bc, v203_data);
          tensorforge::fmacdpp16<12>(v218_acc, v227_bc, v204_data);
          tensorforge::fmacdpp16<13>(v218_acc, v227_bc, v205_data);
          tensorforge::fmacdpp16<14>(v218_acc, v227_bc, v206_data);
          tensorforge::fmacdpp16<15>(v218_acc, v227_bc, v207_data);
          float v228_bc = tensorforge::broadcast<32, 16, 1>(v226_lin);
          tensorforge::fmacdpp16<0>(v218_acc, v228_bc, v208_data);
          tensorforge::fmacdpp16<1>(v218_acc, v228_bc, v209_data);
          tensorforge::fmacdpp16<2>(v218_acc, v228_bc, v210_data);
          tensorforge::fmacdpp16<3>(v218_acc, v228_bc, v211_data);
          tensorforge::fmacdpp16<4>(v219_acc, v228_bc, v200_data);
          tensorforge::fmacdpp16<5>(v219_acc, v228_bc, v201_data);
          tensorforge::fmacdpp16<6>(v219_acc, v228_bc, v202_data);
          tensorforge::fmacdpp16<7>(v219_acc, v228_bc, v203_data);
          tensorforge::fmacdpp16<8>(v219_acc, v228_bc, v204_data);
          tensorforge::fmacdpp16<9>(v219_acc, v228_bc, v205_data);
          tensorforge::fmacdpp16<10>(v219_acc, v228_bc, v206_data);
          tensorforge::fmacdpp16<11>(v219_acc, v228_bc, v207_data);
          tensorforge::fmacdpp16<12>(v219_acc, v228_bc, v208_data);
          tensorforge::fmacdpp16<13>(v219_acc, v228_bc, v209_data);
          tensorforge::fmacdpp16<14>(v219_acc, v228_bc, v210_data);
          tensorforge::fmacdpp16<15>(v219_acc, v228_bc, v211_data);
          r8[0] = v212_acc;
          r8[1] = v213_acc;
          r8[2] = v214_acc;
          r8[3] = v215_acc;
          r8[4] = v216_acc;
          r8[5] = v217_acc;
          r8[6] = v218_acc;
          r8[7] = v219_acc;
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v232_i0 = 0; v232_i0 < 1; ++v232_i0) {
            int32_t v241_lead = v14_lead + (v232_i0 * 32);
            #pragma unroll
            for (int32_t v233_i1 = 0; v233_i1 < 8; ++v233_i1) {
              int32_t v234_a = v232_i0 + v233_i1;
              float v236_data = r8[(v232_i0 + v233_i1)];
              int32_t v244_a = v241_lead + ((v233_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v244_a], v236_data);
            }
          }
        }
      }
    }
  }
}

