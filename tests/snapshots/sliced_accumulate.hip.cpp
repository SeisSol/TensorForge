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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v23_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v23_lin;
          float v24_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v24_lin;
          float v25_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v25_lin;
          float v26_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v26_lin;
          float v27_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v27_lin;
          float v28_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v28_lin;
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
            int32_t v38_lead = v33_i0 * 32;
            int32_t v39_lead = v3_lead + v38_lead;
            int32_t v46_lead = v3_lead + v38_lead;
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
              int32_t v40_a = v34_i1 * 32;
              int32_t v41_a = v39_lead + v40_a;
              float v49_data = __builtin_nontemporal_load(&glb_m3[(v46_lead + v40_a)]);
              int32_t v50_a = v33_i0 + v34_i1;
              r3[v50_a] = v49_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          auto& ir2 = r2;
          float v52_data = r0[0];
          float v53_data = r0[1];
          float v54_data = r0[2];
          float v55_data = r0[3];
          float v56_data = r0[4];
          float v57_data = r0[5];
          float v58_data = r0[6];
          float v59_data = r0[7];
          float v60_data = r0[8];
          float v61_data = r0[9];
          float v62_data = r0[10];
          float v63_data = r0[11];
          float v64_acc{};
          float v65_acc{};
          float v66_acc{};
          float v67_acc{};
          float v68_acc{};
          float v69_acc{};
          float v70_acc{};
          float v71_acc{};
          float v72_acc{};
          float v73_acc{};
          float v74_acc{};
          float v75_acc{};
          float v76_acc{};
          float v77_acc{};
          float v78_acc{};
          float v79_acc{};
          float v80_lin = r1[0];
          float v81_bc = tensorforge::broadcast<32, 16, 0>(v80_lin);
          tensorforge::fmacdpp16<0>(v64_acc, v81_bc, v52_data);
          tensorforge::fmacdpp16<1>(v64_acc, v81_bc, v53_data);
          tensorforge::fmacdpp16<2>(v64_acc, v81_bc, v54_data);
          tensorforge::fmacdpp16<3>(v64_acc, v81_bc, v55_data);
          tensorforge::fmacdpp16<4>(v64_acc, v81_bc, v56_data);
          tensorforge::fmacdpp16<5>(v64_acc, v81_bc, v57_data);
          tensorforge::fmacdpp16<6>(v64_acc, v81_bc, v58_data);
          tensorforge::fmacdpp16<7>(v64_acc, v81_bc, v59_data);
          tensorforge::fmacdpp16<8>(v64_acc, v81_bc, v60_data);
          tensorforge::fmacdpp16<9>(v64_acc, v81_bc, v61_data);
          tensorforge::fmacdpp16<10>(v64_acc, v81_bc, v62_data);
          tensorforge::fmacdpp16<11>(v64_acc, v81_bc, v63_data);
          tensorforge::fmacdpp16<12>(v65_acc, v81_bc, v52_data);
          tensorforge::fmacdpp16<13>(v65_acc, v81_bc, v53_data);
          tensorforge::fmacdpp16<14>(v65_acc, v81_bc, v54_data);
          tensorforge::fmacdpp16<15>(v65_acc, v81_bc, v55_data);
          float v82_bc = tensorforge::broadcast<32, 16, 1>(v80_lin);
          tensorforge::fmacdpp16<0>(v65_acc, v82_bc, v56_data);
          tensorforge::fmacdpp16<1>(v65_acc, v82_bc, v57_data);
          tensorforge::fmacdpp16<2>(v65_acc, v82_bc, v58_data);
          tensorforge::fmacdpp16<3>(v65_acc, v82_bc, v59_data);
          tensorforge::fmacdpp16<4>(v65_acc, v82_bc, v60_data);
          tensorforge::fmacdpp16<5>(v65_acc, v82_bc, v61_data);
          tensorforge::fmacdpp16<6>(v65_acc, v82_bc, v62_data);
          tensorforge::fmacdpp16<7>(v65_acc, v82_bc, v63_data);
          tensorforge::fmacdpp16<8>(v66_acc, v82_bc, v52_data);
          tensorforge::fmacdpp16<9>(v66_acc, v82_bc, v53_data);
          tensorforge::fmacdpp16<10>(v66_acc, v82_bc, v54_data);
          tensorforge::fmacdpp16<11>(v66_acc, v82_bc, v55_data);
          tensorforge::fmacdpp16<12>(v66_acc, v82_bc, v56_data);
          tensorforge::fmacdpp16<13>(v66_acc, v82_bc, v57_data);
          tensorforge::fmacdpp16<14>(v66_acc, v82_bc, v58_data);
          tensorforge::fmacdpp16<15>(v66_acc, v82_bc, v59_data);
          float v83_lin = r1[1];
          float v84_bc = tensorforge::broadcast<32, 16, 0>(v83_lin);
          tensorforge::fmacdpp16<0>(v66_acc, v84_bc, v60_data);
          tensorforge::fmacdpp16<1>(v66_acc, v84_bc, v61_data);
          tensorforge::fmacdpp16<2>(v66_acc, v84_bc, v62_data);
          tensorforge::fmacdpp16<3>(v66_acc, v84_bc, v63_data);
          tensorforge::fmacdpp16<4>(v67_acc, v84_bc, v52_data);
          tensorforge::fmacdpp16<5>(v67_acc, v84_bc, v53_data);
          tensorforge::fmacdpp16<6>(v67_acc, v84_bc, v54_data);
          tensorforge::fmacdpp16<7>(v67_acc, v84_bc, v55_data);
          tensorforge::fmacdpp16<8>(v67_acc, v84_bc, v56_data);
          tensorforge::fmacdpp16<9>(v67_acc, v84_bc, v57_data);
          tensorforge::fmacdpp16<10>(v67_acc, v84_bc, v58_data);
          tensorforge::fmacdpp16<11>(v67_acc, v84_bc, v59_data);
          tensorforge::fmacdpp16<12>(v67_acc, v84_bc, v60_data);
          tensorforge::fmacdpp16<13>(v67_acc, v84_bc, v61_data);
          tensorforge::fmacdpp16<14>(v67_acc, v84_bc, v62_data);
          tensorforge::fmacdpp16<15>(v67_acc, v84_bc, v63_data);
          float v85_bc = tensorforge::broadcast<32, 16, 1>(v83_lin);
          tensorforge::fmacdpp16<0>(v68_acc, v85_bc, v52_data);
          tensorforge::fmacdpp16<1>(v68_acc, v85_bc, v53_data);
          tensorforge::fmacdpp16<2>(v68_acc, v85_bc, v54_data);
          tensorforge::fmacdpp16<3>(v68_acc, v85_bc, v55_data);
          tensorforge::fmacdpp16<4>(v68_acc, v85_bc, v56_data);
          tensorforge::fmacdpp16<5>(v68_acc, v85_bc, v57_data);
          tensorforge::fmacdpp16<6>(v68_acc, v85_bc, v58_data);
          tensorforge::fmacdpp16<7>(v68_acc, v85_bc, v59_data);
          tensorforge::fmacdpp16<8>(v68_acc, v85_bc, v60_data);
          tensorforge::fmacdpp16<9>(v68_acc, v85_bc, v61_data);
          tensorforge::fmacdpp16<10>(v68_acc, v85_bc, v62_data);
          tensorforge::fmacdpp16<11>(v68_acc, v85_bc, v63_data);
          tensorforge::fmacdpp16<12>(v69_acc, v85_bc, v52_data);
          tensorforge::fmacdpp16<13>(v69_acc, v85_bc, v53_data);
          tensorforge::fmacdpp16<14>(v69_acc, v85_bc, v54_data);
          tensorforge::fmacdpp16<15>(v69_acc, v85_bc, v55_data);
          float v86_lin = r1[2];
          float v87_bc = tensorforge::broadcast<32, 16, 0>(v86_lin);
          tensorforge::fmacdpp16<0>(v69_acc, v87_bc, v56_data);
          tensorforge::fmacdpp16<1>(v69_acc, v87_bc, v57_data);
          tensorforge::fmacdpp16<2>(v69_acc, v87_bc, v58_data);
          tensorforge::fmacdpp16<3>(v69_acc, v87_bc, v59_data);
          tensorforge::fmacdpp16<4>(v69_acc, v87_bc, v60_data);
          tensorforge::fmacdpp16<5>(v69_acc, v87_bc, v61_data);
          tensorforge::fmacdpp16<6>(v69_acc, v87_bc, v62_data);
          tensorforge::fmacdpp16<7>(v69_acc, v87_bc, v63_data);
          tensorforge::fmacdpp16<8>(v70_acc, v87_bc, v52_data);
          tensorforge::fmacdpp16<9>(v70_acc, v87_bc, v53_data);
          tensorforge::fmacdpp16<10>(v70_acc, v87_bc, v54_data);
          tensorforge::fmacdpp16<11>(v70_acc, v87_bc, v55_data);
          tensorforge::fmacdpp16<12>(v70_acc, v87_bc, v56_data);
          tensorforge::fmacdpp16<13>(v70_acc, v87_bc, v57_data);
          tensorforge::fmacdpp16<14>(v70_acc, v87_bc, v58_data);
          tensorforge::fmacdpp16<15>(v70_acc, v87_bc, v59_data);
          float v88_bc = tensorforge::broadcast<32, 16, 1>(v86_lin);
          tensorforge::fmacdpp16<0>(v70_acc, v88_bc, v60_data);
          tensorforge::fmacdpp16<1>(v70_acc, v88_bc, v61_data);
          tensorforge::fmacdpp16<2>(v70_acc, v88_bc, v62_data);
          tensorforge::fmacdpp16<3>(v70_acc, v88_bc, v63_data);
          tensorforge::fmacdpp16<4>(v71_acc, v88_bc, v52_data);
          tensorforge::fmacdpp16<5>(v71_acc, v88_bc, v53_data);
          tensorforge::fmacdpp16<6>(v71_acc, v88_bc, v54_data);
          tensorforge::fmacdpp16<7>(v71_acc, v88_bc, v55_data);
          tensorforge::fmacdpp16<8>(v71_acc, v88_bc, v56_data);
          tensorforge::fmacdpp16<9>(v71_acc, v88_bc, v57_data);
          tensorforge::fmacdpp16<10>(v71_acc, v88_bc, v58_data);
          tensorforge::fmacdpp16<11>(v71_acc, v88_bc, v59_data);
          tensorforge::fmacdpp16<12>(v71_acc, v88_bc, v60_data);
          tensorforge::fmacdpp16<13>(v71_acc, v88_bc, v61_data);
          tensorforge::fmacdpp16<14>(v71_acc, v88_bc, v62_data);
          tensorforge::fmacdpp16<15>(v71_acc, v88_bc, v63_data);
          float v89_lin = r1[3];
          float v90_bc = tensorforge::broadcast<32, 16, 0>(v89_lin);
          tensorforge::fmacdpp16<0>(v72_acc, v90_bc, v52_data);
          tensorforge::fmacdpp16<1>(v72_acc, v90_bc, v53_data);
          tensorforge::fmacdpp16<2>(v72_acc, v90_bc, v54_data);
          tensorforge::fmacdpp16<3>(v72_acc, v90_bc, v55_data);
          tensorforge::fmacdpp16<4>(v72_acc, v90_bc, v56_data);
          tensorforge::fmacdpp16<5>(v72_acc, v90_bc, v57_data);
          tensorforge::fmacdpp16<6>(v72_acc, v90_bc, v58_data);
          tensorforge::fmacdpp16<7>(v72_acc, v90_bc, v59_data);
          tensorforge::fmacdpp16<8>(v72_acc, v90_bc, v60_data);
          tensorforge::fmacdpp16<9>(v72_acc, v90_bc, v61_data);
          tensorforge::fmacdpp16<10>(v72_acc, v90_bc, v62_data);
          tensorforge::fmacdpp16<11>(v72_acc, v90_bc, v63_data);
          tensorforge::fmacdpp16<12>(v73_acc, v90_bc, v52_data);
          tensorforge::fmacdpp16<13>(v73_acc, v90_bc, v53_data);
          tensorforge::fmacdpp16<14>(v73_acc, v90_bc, v54_data);
          tensorforge::fmacdpp16<15>(v73_acc, v90_bc, v55_data);
          float v91_bc = tensorforge::broadcast<32, 16, 1>(v89_lin);
          tensorforge::fmacdpp16<0>(v73_acc, v91_bc, v56_data);
          tensorforge::fmacdpp16<1>(v73_acc, v91_bc, v57_data);
          tensorforge::fmacdpp16<2>(v73_acc, v91_bc, v58_data);
          tensorforge::fmacdpp16<3>(v73_acc, v91_bc, v59_data);
          tensorforge::fmacdpp16<4>(v73_acc, v91_bc, v60_data);
          tensorforge::fmacdpp16<5>(v73_acc, v91_bc, v61_data);
          tensorforge::fmacdpp16<6>(v73_acc, v91_bc, v62_data);
          tensorforge::fmacdpp16<7>(v73_acc, v91_bc, v63_data);
          tensorforge::fmacdpp16<8>(v74_acc, v91_bc, v52_data);
          tensorforge::fmacdpp16<9>(v74_acc, v91_bc, v53_data);
          tensorforge::fmacdpp16<10>(v74_acc, v91_bc, v54_data);
          tensorforge::fmacdpp16<11>(v74_acc, v91_bc, v55_data);
          tensorforge::fmacdpp16<12>(v74_acc, v91_bc, v56_data);
          tensorforge::fmacdpp16<13>(v74_acc, v91_bc, v57_data);
          tensorforge::fmacdpp16<14>(v74_acc, v91_bc, v58_data);
          tensorforge::fmacdpp16<15>(v74_acc, v91_bc, v59_data);
          float v92_lin = r1[4];
          float v93_bc = tensorforge::broadcast<32, 16, 0>(v92_lin);
          tensorforge::fmacdpp16<0>(v74_acc, v93_bc, v60_data);
          tensorforge::fmacdpp16<1>(v74_acc, v93_bc, v61_data);
          tensorforge::fmacdpp16<2>(v74_acc, v93_bc, v62_data);
          tensorforge::fmacdpp16<3>(v74_acc, v93_bc, v63_data);
          tensorforge::fmacdpp16<4>(v75_acc, v93_bc, v52_data);
          tensorforge::fmacdpp16<5>(v75_acc, v93_bc, v53_data);
          tensorforge::fmacdpp16<6>(v75_acc, v93_bc, v54_data);
          tensorforge::fmacdpp16<7>(v75_acc, v93_bc, v55_data);
          tensorforge::fmacdpp16<8>(v75_acc, v93_bc, v56_data);
          tensorforge::fmacdpp16<9>(v75_acc, v93_bc, v57_data);
          tensorforge::fmacdpp16<10>(v75_acc, v93_bc, v58_data);
          tensorforge::fmacdpp16<11>(v75_acc, v93_bc, v59_data);
          tensorforge::fmacdpp16<12>(v75_acc, v93_bc, v60_data);
          tensorforge::fmacdpp16<13>(v75_acc, v93_bc, v61_data);
          tensorforge::fmacdpp16<14>(v75_acc, v93_bc, v62_data);
          tensorforge::fmacdpp16<15>(v75_acc, v93_bc, v63_data);
          float v94_bc = tensorforge::broadcast<32, 16, 1>(v92_lin);
          tensorforge::fmacdpp16<0>(v76_acc, v94_bc, v52_data);
          tensorforge::fmacdpp16<1>(v76_acc, v94_bc, v53_data);
          tensorforge::fmacdpp16<2>(v76_acc, v94_bc, v54_data);
          tensorforge::fmacdpp16<3>(v76_acc, v94_bc, v55_data);
          tensorforge::fmacdpp16<4>(v76_acc, v94_bc, v56_data);
          tensorforge::fmacdpp16<5>(v76_acc, v94_bc, v57_data);
          tensorforge::fmacdpp16<6>(v76_acc, v94_bc, v58_data);
          tensorforge::fmacdpp16<7>(v76_acc, v94_bc, v59_data);
          tensorforge::fmacdpp16<8>(v76_acc, v94_bc, v60_data);
          tensorforge::fmacdpp16<9>(v76_acc, v94_bc, v61_data);
          tensorforge::fmacdpp16<10>(v76_acc, v94_bc, v62_data);
          tensorforge::fmacdpp16<11>(v76_acc, v94_bc, v63_data);
          tensorforge::fmacdpp16<12>(v77_acc, v94_bc, v52_data);
          tensorforge::fmacdpp16<13>(v77_acc, v94_bc, v53_data);
          tensorforge::fmacdpp16<14>(v77_acc, v94_bc, v54_data);
          tensorforge::fmacdpp16<15>(v77_acc, v94_bc, v55_data);
          float v95_lin = r1[5];
          float v96_bc = tensorforge::broadcast<32, 16, 0>(v95_lin);
          tensorforge::fmacdpp16<0>(v77_acc, v96_bc, v56_data);
          tensorforge::fmacdpp16<1>(v77_acc, v96_bc, v57_data);
          tensorforge::fmacdpp16<2>(v77_acc, v96_bc, v58_data);
          tensorforge::fmacdpp16<3>(v77_acc, v96_bc, v59_data);
          tensorforge::fmacdpp16<4>(v77_acc, v96_bc, v60_data);
          tensorforge::fmacdpp16<5>(v77_acc, v96_bc, v61_data);
          tensorforge::fmacdpp16<6>(v77_acc, v96_bc, v62_data);
          tensorforge::fmacdpp16<7>(v77_acc, v96_bc, v63_data);
          tensorforge::fmacdpp16<8>(v78_acc, v96_bc, v52_data);
          tensorforge::fmacdpp16<9>(v78_acc, v96_bc, v53_data);
          tensorforge::fmacdpp16<10>(v78_acc, v96_bc, v54_data);
          tensorforge::fmacdpp16<11>(v78_acc, v96_bc, v55_data);
          tensorforge::fmacdpp16<12>(v78_acc, v96_bc, v56_data);
          tensorforge::fmacdpp16<13>(v78_acc, v96_bc, v57_data);
          tensorforge::fmacdpp16<14>(v78_acc, v96_bc, v58_data);
          tensorforge::fmacdpp16<15>(v78_acc, v96_bc, v59_data);
          float v97_bc = tensorforge::broadcast<32, 16, 1>(v95_lin);
          tensorforge::fmacdpp16<0>(v78_acc, v97_bc, v60_data);
          tensorforge::fmacdpp16<1>(v78_acc, v97_bc, v61_data);
          tensorforge::fmacdpp16<2>(v78_acc, v97_bc, v62_data);
          tensorforge::fmacdpp16<3>(v78_acc, v97_bc, v63_data);
          tensorforge::fmacdpp16<4>(v79_acc, v97_bc, v52_data);
          tensorforge::fmacdpp16<5>(v79_acc, v97_bc, v53_data);
          tensorforge::fmacdpp16<6>(v79_acc, v97_bc, v54_data);
          tensorforge::fmacdpp16<7>(v79_acc, v97_bc, v55_data);
          tensorforge::fmacdpp16<8>(v79_acc, v97_bc, v56_data);
          tensorforge::fmacdpp16<9>(v79_acc, v97_bc, v57_data);
          tensorforge::fmacdpp16<10>(v79_acc, v97_bc, v58_data);
          tensorforge::fmacdpp16<11>(v79_acc, v97_bc, v59_data);
          tensorforge::fmacdpp16<12>(v79_acc, v97_bc, v60_data);
          tensorforge::fmacdpp16<13>(v79_acc, v97_bc, v61_data);
          tensorforge::fmacdpp16<14>(v79_acc, v97_bc, v62_data);
          tensorforge::fmacdpp16<15>(v79_acc, v97_bc, v63_data);
          ir2[0] = v64_acc;
          ir2[1] = v65_acc;
          ir2[2] = v66_acc;
          ir2[3] = v67_acc;
          ir2[4] = v68_acc;
          ir2[5] = v69_acc;
          ir2[6] = v70_acc;
          ir2[7] = v71_acc;
          ir2[8] = v72_acc;
          ir2[9] = v73_acc;
          ir2[10] = v74_acc;
          ir2[11] = v75_acc;
          ir2[12] = v76_acc;
          ir2[13] = v77_acc;
          ir2[14] = v78_acc;
          ir2[15] = v79_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v101_i0 = 0; v101_i0 < 1; ++v101_i0) {
            int32_t v110_lead = v3_lead + (v101_i0 * 32);
            #pragma unroll
            for (int32_t v102_i1 = 0; v102_i1 < 16; ++v102_i1) {
              int32_t v103_a = v101_i0 + v102_i1;
              float v105_data = r2[(v101_i0 + v102_i1)];
              int32_t v112_a = v110_lead + (v102_i1 * 32);
              glb_m0[v112_a] = v105_data;
            }
          }
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v114_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v114_lin;
          float v115_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v115_lin;
          float v116_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v116_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v121_i0 = 0; v121_i0 < 1; ++v121_i0) {
            int32_t v126_lead = v121_i0 * 32;
            int32_t v127_lead = v3_lead + v126_lead;
            int32_t v134_lead = v3_lead + v126_lead;
            #pragma unroll
            for (int32_t v122_i1 = 0; v122_i1 < 12; ++v122_i1) {
              int32_t v128_a = v122_i1 * 32;
              int32_t v129_a = v127_lead + v128_a;
              float v137_data = __builtin_nontemporal_load(&glb_m5[(v134_lead + v128_a)]);
              int32_t v138_a = v121_i0 + v122_i1;
              r6[v138_a] = v137_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[16]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          auto& ir5 = r5;
          float v140_data = r3[0];
          float v141_data = r3[1];
          float v142_data = r3[2];
          float v143_data = r3[3];
          float v144_data = r3[4];
          float v145_data = r3[5];
          float v146_data = r3[6];
          float v147_data = r3[7];
          float v148_data = r3[8];
          float v149_data = r3[9];
          float v150_data = r3[10];
          float v151_data = r3[11];
          float v152_acc{};
          float v153_acc{};
          float v154_acc{};
          float v155_acc{};
          float v156_acc{};
          float v157_acc{};
          float v158_acc{};
          float v159_acc{};
          float v160_lin = r4[0];
          float v161_bc = tensorforge::broadcast<32, 16, 0>(v160_lin);
          tensorforge::fmacdpp16<0>(v152_acc, v161_bc, v140_data);
          tensorforge::fmacdpp16<1>(v152_acc, v161_bc, v141_data);
          tensorforge::fmacdpp16<2>(v152_acc, v161_bc, v142_data);
          tensorforge::fmacdpp16<3>(v152_acc, v161_bc, v143_data);
          tensorforge::fmacdpp16<4>(v152_acc, v161_bc, v144_data);
          tensorforge::fmacdpp16<5>(v152_acc, v161_bc, v145_data);
          tensorforge::fmacdpp16<6>(v152_acc, v161_bc, v146_data);
          tensorforge::fmacdpp16<7>(v152_acc, v161_bc, v147_data);
          tensorforge::fmacdpp16<8>(v152_acc, v161_bc, v148_data);
          tensorforge::fmacdpp16<9>(v152_acc, v161_bc, v149_data);
          tensorforge::fmacdpp16<10>(v152_acc, v161_bc, v150_data);
          tensorforge::fmacdpp16<11>(v152_acc, v161_bc, v151_data);
          tensorforge::fmacdpp16<12>(v153_acc, v161_bc, v140_data);
          tensorforge::fmacdpp16<13>(v153_acc, v161_bc, v141_data);
          tensorforge::fmacdpp16<14>(v153_acc, v161_bc, v142_data);
          tensorforge::fmacdpp16<15>(v153_acc, v161_bc, v143_data);
          float v162_bc = tensorforge::broadcast<32, 16, 1>(v160_lin);
          tensorforge::fmacdpp16<0>(v153_acc, v162_bc, v144_data);
          tensorforge::fmacdpp16<1>(v153_acc, v162_bc, v145_data);
          tensorforge::fmacdpp16<2>(v153_acc, v162_bc, v146_data);
          tensorforge::fmacdpp16<3>(v153_acc, v162_bc, v147_data);
          tensorforge::fmacdpp16<4>(v153_acc, v162_bc, v148_data);
          tensorforge::fmacdpp16<5>(v153_acc, v162_bc, v149_data);
          tensorforge::fmacdpp16<6>(v153_acc, v162_bc, v150_data);
          tensorforge::fmacdpp16<7>(v153_acc, v162_bc, v151_data);
          tensorforge::fmacdpp16<8>(v154_acc, v162_bc, v140_data);
          tensorforge::fmacdpp16<9>(v154_acc, v162_bc, v141_data);
          tensorforge::fmacdpp16<10>(v154_acc, v162_bc, v142_data);
          tensorforge::fmacdpp16<11>(v154_acc, v162_bc, v143_data);
          tensorforge::fmacdpp16<12>(v154_acc, v162_bc, v144_data);
          tensorforge::fmacdpp16<13>(v154_acc, v162_bc, v145_data);
          tensorforge::fmacdpp16<14>(v154_acc, v162_bc, v146_data);
          tensorforge::fmacdpp16<15>(v154_acc, v162_bc, v147_data);
          float v163_lin = r4[1];
          float v164_bc = tensorforge::broadcast<32, 16, 0>(v163_lin);
          tensorforge::fmacdpp16<0>(v154_acc, v164_bc, v148_data);
          tensorforge::fmacdpp16<1>(v154_acc, v164_bc, v149_data);
          tensorforge::fmacdpp16<2>(v154_acc, v164_bc, v150_data);
          tensorforge::fmacdpp16<3>(v154_acc, v164_bc, v151_data);
          tensorforge::fmacdpp16<4>(v155_acc, v164_bc, v140_data);
          tensorforge::fmacdpp16<5>(v155_acc, v164_bc, v141_data);
          tensorforge::fmacdpp16<6>(v155_acc, v164_bc, v142_data);
          tensorforge::fmacdpp16<7>(v155_acc, v164_bc, v143_data);
          tensorforge::fmacdpp16<8>(v155_acc, v164_bc, v144_data);
          tensorforge::fmacdpp16<9>(v155_acc, v164_bc, v145_data);
          tensorforge::fmacdpp16<10>(v155_acc, v164_bc, v146_data);
          tensorforge::fmacdpp16<11>(v155_acc, v164_bc, v147_data);
          tensorforge::fmacdpp16<12>(v155_acc, v164_bc, v148_data);
          tensorforge::fmacdpp16<13>(v155_acc, v164_bc, v149_data);
          tensorforge::fmacdpp16<14>(v155_acc, v164_bc, v150_data);
          tensorforge::fmacdpp16<15>(v155_acc, v164_bc, v151_data);
          float v165_bc = tensorforge::broadcast<32, 16, 1>(v163_lin);
          tensorforge::fmacdpp16<0>(v156_acc, v165_bc, v140_data);
          tensorforge::fmacdpp16<1>(v156_acc, v165_bc, v141_data);
          tensorforge::fmacdpp16<2>(v156_acc, v165_bc, v142_data);
          tensorforge::fmacdpp16<3>(v156_acc, v165_bc, v143_data);
          tensorforge::fmacdpp16<4>(v156_acc, v165_bc, v144_data);
          tensorforge::fmacdpp16<5>(v156_acc, v165_bc, v145_data);
          tensorforge::fmacdpp16<6>(v156_acc, v165_bc, v146_data);
          tensorforge::fmacdpp16<7>(v156_acc, v165_bc, v147_data);
          tensorforge::fmacdpp16<8>(v156_acc, v165_bc, v148_data);
          tensorforge::fmacdpp16<9>(v156_acc, v165_bc, v149_data);
          tensorforge::fmacdpp16<10>(v156_acc, v165_bc, v150_data);
          tensorforge::fmacdpp16<11>(v156_acc, v165_bc, v151_data);
          tensorforge::fmacdpp16<12>(v157_acc, v165_bc, v140_data);
          tensorforge::fmacdpp16<13>(v157_acc, v165_bc, v141_data);
          tensorforge::fmacdpp16<14>(v157_acc, v165_bc, v142_data);
          tensorforge::fmacdpp16<15>(v157_acc, v165_bc, v143_data);
          float v166_lin = r4[2];
          float v167_bc = tensorforge::broadcast<32, 16, 0>(v166_lin);
          tensorforge::fmacdpp16<0>(v157_acc, v167_bc, v144_data);
          tensorforge::fmacdpp16<1>(v157_acc, v167_bc, v145_data);
          tensorforge::fmacdpp16<2>(v157_acc, v167_bc, v146_data);
          tensorforge::fmacdpp16<3>(v157_acc, v167_bc, v147_data);
          tensorforge::fmacdpp16<4>(v157_acc, v167_bc, v148_data);
          tensorforge::fmacdpp16<5>(v157_acc, v167_bc, v149_data);
          tensorforge::fmacdpp16<6>(v157_acc, v167_bc, v150_data);
          tensorforge::fmacdpp16<7>(v157_acc, v167_bc, v151_data);
          tensorforge::fmacdpp16<8>(v158_acc, v167_bc, v140_data);
          tensorforge::fmacdpp16<9>(v158_acc, v167_bc, v141_data);
          tensorforge::fmacdpp16<10>(v158_acc, v167_bc, v142_data);
          tensorforge::fmacdpp16<11>(v158_acc, v167_bc, v143_data);
          tensorforge::fmacdpp16<12>(v158_acc, v167_bc, v144_data);
          tensorforge::fmacdpp16<13>(v158_acc, v167_bc, v145_data);
          tensorforge::fmacdpp16<14>(v158_acc, v167_bc, v146_data);
          tensorforge::fmacdpp16<15>(v158_acc, v167_bc, v147_data);
          float v168_bc = tensorforge::broadcast<32, 16, 1>(v166_lin);
          tensorforge::fmacdpp16<0>(v158_acc, v168_bc, v148_data);
          tensorforge::fmacdpp16<1>(v158_acc, v168_bc, v149_data);
          tensorforge::fmacdpp16<2>(v158_acc, v168_bc, v150_data);
          tensorforge::fmacdpp16<3>(v158_acc, v168_bc, v151_data);
          tensorforge::fmacdpp16<4>(v159_acc, v168_bc, v140_data);
          tensorforge::fmacdpp16<5>(v159_acc, v168_bc, v141_data);
          tensorforge::fmacdpp16<6>(v159_acc, v168_bc, v142_data);
          tensorforge::fmacdpp16<7>(v159_acc, v168_bc, v143_data);
          tensorforge::fmacdpp16<8>(v159_acc, v168_bc, v144_data);
          tensorforge::fmacdpp16<9>(v159_acc, v168_bc, v145_data);
          tensorforge::fmacdpp16<10>(v159_acc, v168_bc, v146_data);
          tensorforge::fmacdpp16<11>(v159_acc, v168_bc, v147_data);
          tensorforge::fmacdpp16<12>(v159_acc, v168_bc, v148_data);
          tensorforge::fmacdpp16<13>(v159_acc, v168_bc, v149_data);
          tensorforge::fmacdpp16<14>(v159_acc, v168_bc, v150_data);
          tensorforge::fmacdpp16<15>(v159_acc, v168_bc, v151_data);
          ir5[0] = v152_acc;
          ir5[1] = v153_acc;
          ir5[2] = v154_acc;
          ir5[3] = v155_acc;
          ir5[4] = v156_acc;
          ir5[5] = v157_acc;
          ir5[6] = v158_acc;
          ir5[7] = v159_acc;
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v172_i0 = 0; v172_i0 < 1; ++v172_i0) {
            int32_t v181_lead = v3_lead + (v172_i0 * 32);
            #pragma unroll
            for (int32_t v173_i1 = 0; v173_i1 < 8; ++v173_i1) {
              int32_t v174_a = v172_i0 + v173_i1;
              float v176_data = r5[(v172_i0 + v173_i1)];
              int32_t v183_a = v181_lead + (v173_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v183_a], v176_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v185_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v185_lin;
          float v186_lin = glb_m6[32 + threadIdx.x * 1];
          r7[1] = v186_lin;
          float v187_lin = glb_m6[64 + threadIdx.x * 1];
          r7[2] = v187_lin;
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
          float r8[16]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          auto& ir8 = r8;
          float v189_data = r6[0];
          float v190_data = r6[1];
          float v191_data = r6[2];
          float v192_data = r6[3];
          float v193_data = r6[4];
          float v194_data = r6[5];
          float v195_data = r6[6];
          float v196_data = r6[7];
          float v197_data = r6[8];
          float v198_data = r6[9];
          float v199_data = r6[10];
          float v200_data = r6[11];
          float v201_acc{};
          float v202_acc{};
          float v203_acc{};
          float v204_acc{};
          float v205_acc{};
          float v206_acc{};
          float v207_acc{};
          float v208_acc{};
          float v209_lin = r7[0];
          float v210_bc = tensorforge::broadcast<32, 16, 0>(v209_lin);
          tensorforge::fmacdpp16<0>(v201_acc, v210_bc, v189_data);
          tensorforge::fmacdpp16<1>(v201_acc, v210_bc, v190_data);
          tensorforge::fmacdpp16<2>(v201_acc, v210_bc, v191_data);
          tensorforge::fmacdpp16<3>(v201_acc, v210_bc, v192_data);
          tensorforge::fmacdpp16<4>(v201_acc, v210_bc, v193_data);
          tensorforge::fmacdpp16<5>(v201_acc, v210_bc, v194_data);
          tensorforge::fmacdpp16<6>(v201_acc, v210_bc, v195_data);
          tensorforge::fmacdpp16<7>(v201_acc, v210_bc, v196_data);
          tensorforge::fmacdpp16<8>(v201_acc, v210_bc, v197_data);
          tensorforge::fmacdpp16<9>(v201_acc, v210_bc, v198_data);
          tensorforge::fmacdpp16<10>(v201_acc, v210_bc, v199_data);
          tensorforge::fmacdpp16<11>(v201_acc, v210_bc, v200_data);
          tensorforge::fmacdpp16<12>(v202_acc, v210_bc, v189_data);
          tensorforge::fmacdpp16<13>(v202_acc, v210_bc, v190_data);
          tensorforge::fmacdpp16<14>(v202_acc, v210_bc, v191_data);
          tensorforge::fmacdpp16<15>(v202_acc, v210_bc, v192_data);
          float v211_bc = tensorforge::broadcast<32, 16, 1>(v209_lin);
          tensorforge::fmacdpp16<0>(v202_acc, v211_bc, v193_data);
          tensorforge::fmacdpp16<1>(v202_acc, v211_bc, v194_data);
          tensorforge::fmacdpp16<2>(v202_acc, v211_bc, v195_data);
          tensorforge::fmacdpp16<3>(v202_acc, v211_bc, v196_data);
          tensorforge::fmacdpp16<4>(v202_acc, v211_bc, v197_data);
          tensorforge::fmacdpp16<5>(v202_acc, v211_bc, v198_data);
          tensorforge::fmacdpp16<6>(v202_acc, v211_bc, v199_data);
          tensorforge::fmacdpp16<7>(v202_acc, v211_bc, v200_data);
          tensorforge::fmacdpp16<8>(v203_acc, v211_bc, v189_data);
          tensorforge::fmacdpp16<9>(v203_acc, v211_bc, v190_data);
          tensorforge::fmacdpp16<10>(v203_acc, v211_bc, v191_data);
          tensorforge::fmacdpp16<11>(v203_acc, v211_bc, v192_data);
          tensorforge::fmacdpp16<12>(v203_acc, v211_bc, v193_data);
          tensorforge::fmacdpp16<13>(v203_acc, v211_bc, v194_data);
          tensorforge::fmacdpp16<14>(v203_acc, v211_bc, v195_data);
          tensorforge::fmacdpp16<15>(v203_acc, v211_bc, v196_data);
          float v212_lin = r7[1];
          float v213_bc = tensorforge::broadcast<32, 16, 0>(v212_lin);
          tensorforge::fmacdpp16<0>(v203_acc, v213_bc, v197_data);
          tensorforge::fmacdpp16<1>(v203_acc, v213_bc, v198_data);
          tensorforge::fmacdpp16<2>(v203_acc, v213_bc, v199_data);
          tensorforge::fmacdpp16<3>(v203_acc, v213_bc, v200_data);
          tensorforge::fmacdpp16<4>(v204_acc, v213_bc, v189_data);
          tensorforge::fmacdpp16<5>(v204_acc, v213_bc, v190_data);
          tensorforge::fmacdpp16<6>(v204_acc, v213_bc, v191_data);
          tensorforge::fmacdpp16<7>(v204_acc, v213_bc, v192_data);
          tensorforge::fmacdpp16<8>(v204_acc, v213_bc, v193_data);
          tensorforge::fmacdpp16<9>(v204_acc, v213_bc, v194_data);
          tensorforge::fmacdpp16<10>(v204_acc, v213_bc, v195_data);
          tensorforge::fmacdpp16<11>(v204_acc, v213_bc, v196_data);
          tensorforge::fmacdpp16<12>(v204_acc, v213_bc, v197_data);
          tensorforge::fmacdpp16<13>(v204_acc, v213_bc, v198_data);
          tensorforge::fmacdpp16<14>(v204_acc, v213_bc, v199_data);
          tensorforge::fmacdpp16<15>(v204_acc, v213_bc, v200_data);
          float v214_bc = tensorforge::broadcast<32, 16, 1>(v212_lin);
          tensorforge::fmacdpp16<0>(v205_acc, v214_bc, v189_data);
          tensorforge::fmacdpp16<1>(v205_acc, v214_bc, v190_data);
          tensorforge::fmacdpp16<2>(v205_acc, v214_bc, v191_data);
          tensorforge::fmacdpp16<3>(v205_acc, v214_bc, v192_data);
          tensorforge::fmacdpp16<4>(v205_acc, v214_bc, v193_data);
          tensorforge::fmacdpp16<5>(v205_acc, v214_bc, v194_data);
          tensorforge::fmacdpp16<6>(v205_acc, v214_bc, v195_data);
          tensorforge::fmacdpp16<7>(v205_acc, v214_bc, v196_data);
          tensorforge::fmacdpp16<8>(v205_acc, v214_bc, v197_data);
          tensorforge::fmacdpp16<9>(v205_acc, v214_bc, v198_data);
          tensorforge::fmacdpp16<10>(v205_acc, v214_bc, v199_data);
          tensorforge::fmacdpp16<11>(v205_acc, v214_bc, v200_data);
          tensorforge::fmacdpp16<12>(v206_acc, v214_bc, v189_data);
          tensorforge::fmacdpp16<13>(v206_acc, v214_bc, v190_data);
          tensorforge::fmacdpp16<14>(v206_acc, v214_bc, v191_data);
          tensorforge::fmacdpp16<15>(v206_acc, v214_bc, v192_data);
          float v215_lin = r7[2];
          float v216_bc = tensorforge::broadcast<32, 16, 0>(v215_lin);
          tensorforge::fmacdpp16<0>(v206_acc, v216_bc, v193_data);
          tensorforge::fmacdpp16<1>(v206_acc, v216_bc, v194_data);
          tensorforge::fmacdpp16<2>(v206_acc, v216_bc, v195_data);
          tensorforge::fmacdpp16<3>(v206_acc, v216_bc, v196_data);
          tensorforge::fmacdpp16<4>(v206_acc, v216_bc, v197_data);
          tensorforge::fmacdpp16<5>(v206_acc, v216_bc, v198_data);
          tensorforge::fmacdpp16<6>(v206_acc, v216_bc, v199_data);
          tensorforge::fmacdpp16<7>(v206_acc, v216_bc, v200_data);
          tensorforge::fmacdpp16<8>(v207_acc, v216_bc, v189_data);
          tensorforge::fmacdpp16<9>(v207_acc, v216_bc, v190_data);
          tensorforge::fmacdpp16<10>(v207_acc, v216_bc, v191_data);
          tensorforge::fmacdpp16<11>(v207_acc, v216_bc, v192_data);
          tensorforge::fmacdpp16<12>(v207_acc, v216_bc, v193_data);
          tensorforge::fmacdpp16<13>(v207_acc, v216_bc, v194_data);
          tensorforge::fmacdpp16<14>(v207_acc, v216_bc, v195_data);
          tensorforge::fmacdpp16<15>(v207_acc, v216_bc, v196_data);
          float v217_bc = tensorforge::broadcast<32, 16, 1>(v215_lin);
          tensorforge::fmacdpp16<0>(v207_acc, v217_bc, v197_data);
          tensorforge::fmacdpp16<1>(v207_acc, v217_bc, v198_data);
          tensorforge::fmacdpp16<2>(v207_acc, v217_bc, v199_data);
          tensorforge::fmacdpp16<3>(v207_acc, v217_bc, v200_data);
          tensorforge::fmacdpp16<4>(v208_acc, v217_bc, v189_data);
          tensorforge::fmacdpp16<5>(v208_acc, v217_bc, v190_data);
          tensorforge::fmacdpp16<6>(v208_acc, v217_bc, v191_data);
          tensorforge::fmacdpp16<7>(v208_acc, v217_bc, v192_data);
          tensorforge::fmacdpp16<8>(v208_acc, v217_bc, v193_data);
          tensorforge::fmacdpp16<9>(v208_acc, v217_bc, v194_data);
          tensorforge::fmacdpp16<10>(v208_acc, v217_bc, v195_data);
          tensorforge::fmacdpp16<11>(v208_acc, v217_bc, v196_data);
          tensorforge::fmacdpp16<12>(v208_acc, v217_bc, v197_data);
          tensorforge::fmacdpp16<13>(v208_acc, v217_bc, v198_data);
          tensorforge::fmacdpp16<14>(v208_acc, v217_bc, v199_data);
          tensorforge::fmacdpp16<15>(v208_acc, v217_bc, v200_data);
          ir8[0] = v201_acc;
          ir8[1] = v202_acc;
          ir8[2] = v203_acc;
          ir8[3] = v204_acc;
          ir8[4] = v205_acc;
          ir8[5] = v206_acc;
          ir8[6] = v207_acc;
          ir8[7] = v208_acc;
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v221_i0 = 0; v221_i0 < 1; ++v221_i0) {
            int32_t v230_lead = v3_lead + (v221_i0 * 32);
            #pragma unroll
            for (int32_t v222_i1 = 0; v222_i1 < 8; ++v222_i1) {
              int32_t v223_a = v221_i0 + v222_i1;
              float v225_data = r8[(v221_i0 + v222_i1)];
              int32_t v233_a = v230_lead + ((v222_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v233_a], v225_data);
            }
          }
          ;
        }
      }
    }
  }
}

