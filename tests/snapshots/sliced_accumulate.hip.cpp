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
            int32_t v21_lead = v14_lead + (v15_i0 * 32);
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 12; ++v16_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + (v16_i1 * 32))]);
              r0[(v15_i0 + v16_i1)] = v24_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          float v27_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v27_lin;
          float v28_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v28_lin;
          float v29_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v29_lin;
          float v30_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v30_lin;
          float v31_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v31_lin;
          float v32_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v32_lin;
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v37_i0 = 0; v37_i0 < 1; ++v37_i0) {
            int32_t v43_lead = v14_lead + (v37_i0 * 32);
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
              float v46_data = __builtin_nontemporal_load(&glb_m3[(v43_lead + (v38_i1 * 32))]);
              r3[(v37_i0 + v38_i1)] = v46_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          float v49_data = r0[0];
          float v50_data = r0[1];
          float v51_data = r0[2];
          float v52_data = r0[3];
          float v53_data = r0[4];
          float v54_data = r0[5];
          float v55_data = r0[6];
          float v56_data = r0[7];
          float v57_data = r0[8];
          float v58_data = r0[9];
          float v59_data = r0[10];
          float v60_data = r0[11];
          float v61_acc{};
          float v62_acc{};
          float v63_acc{};
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
          float v77_lin = r1[0];
          float v78_bc = tensorforge::broadcast<32, 16, 0>(v77_lin);
          tensorforge::fmacdpp16<0>(v61_acc, v78_bc, v49_data);
          tensorforge::fmacdpp16<1>(v61_acc, v78_bc, v50_data);
          tensorforge::fmacdpp16<2>(v61_acc, v78_bc, v51_data);
          tensorforge::fmacdpp16<3>(v61_acc, v78_bc, v52_data);
          tensorforge::fmacdpp16<4>(v61_acc, v78_bc, v53_data);
          tensorforge::fmacdpp16<5>(v61_acc, v78_bc, v54_data);
          tensorforge::fmacdpp16<6>(v61_acc, v78_bc, v55_data);
          tensorforge::fmacdpp16<7>(v61_acc, v78_bc, v56_data);
          tensorforge::fmacdpp16<8>(v61_acc, v78_bc, v57_data);
          tensorforge::fmacdpp16<9>(v61_acc, v78_bc, v58_data);
          tensorforge::fmacdpp16<10>(v61_acc, v78_bc, v59_data);
          tensorforge::fmacdpp16<11>(v61_acc, v78_bc, v60_data);
          tensorforge::fmacdpp16<12>(v62_acc, v78_bc, v49_data);
          tensorforge::fmacdpp16<13>(v62_acc, v78_bc, v50_data);
          tensorforge::fmacdpp16<14>(v62_acc, v78_bc, v51_data);
          tensorforge::fmacdpp16<15>(v62_acc, v78_bc, v52_data);
          float v79_bc = tensorforge::broadcast<32, 16, 1>(v77_lin);
          tensorforge::fmacdpp16<0>(v62_acc, v79_bc, v53_data);
          tensorforge::fmacdpp16<1>(v62_acc, v79_bc, v54_data);
          tensorforge::fmacdpp16<2>(v62_acc, v79_bc, v55_data);
          tensorforge::fmacdpp16<3>(v62_acc, v79_bc, v56_data);
          tensorforge::fmacdpp16<4>(v62_acc, v79_bc, v57_data);
          tensorforge::fmacdpp16<5>(v62_acc, v79_bc, v58_data);
          tensorforge::fmacdpp16<6>(v62_acc, v79_bc, v59_data);
          tensorforge::fmacdpp16<7>(v62_acc, v79_bc, v60_data);
          tensorforge::fmacdpp16<8>(v63_acc, v79_bc, v49_data);
          tensorforge::fmacdpp16<9>(v63_acc, v79_bc, v50_data);
          tensorforge::fmacdpp16<10>(v63_acc, v79_bc, v51_data);
          tensorforge::fmacdpp16<11>(v63_acc, v79_bc, v52_data);
          tensorforge::fmacdpp16<12>(v63_acc, v79_bc, v53_data);
          tensorforge::fmacdpp16<13>(v63_acc, v79_bc, v54_data);
          tensorforge::fmacdpp16<14>(v63_acc, v79_bc, v55_data);
          tensorforge::fmacdpp16<15>(v63_acc, v79_bc, v56_data);
          float v80_lin = r1[1];
          float v81_bc = tensorforge::broadcast<32, 16, 0>(v80_lin);
          tensorforge::fmacdpp16<0>(v63_acc, v81_bc, v57_data);
          tensorforge::fmacdpp16<1>(v63_acc, v81_bc, v58_data);
          tensorforge::fmacdpp16<2>(v63_acc, v81_bc, v59_data);
          tensorforge::fmacdpp16<3>(v63_acc, v81_bc, v60_data);
          tensorforge::fmacdpp16<4>(v64_acc, v81_bc, v49_data);
          tensorforge::fmacdpp16<5>(v64_acc, v81_bc, v50_data);
          tensorforge::fmacdpp16<6>(v64_acc, v81_bc, v51_data);
          tensorforge::fmacdpp16<7>(v64_acc, v81_bc, v52_data);
          tensorforge::fmacdpp16<8>(v64_acc, v81_bc, v53_data);
          tensorforge::fmacdpp16<9>(v64_acc, v81_bc, v54_data);
          tensorforge::fmacdpp16<10>(v64_acc, v81_bc, v55_data);
          tensorforge::fmacdpp16<11>(v64_acc, v81_bc, v56_data);
          tensorforge::fmacdpp16<12>(v64_acc, v81_bc, v57_data);
          tensorforge::fmacdpp16<13>(v64_acc, v81_bc, v58_data);
          tensorforge::fmacdpp16<14>(v64_acc, v81_bc, v59_data);
          tensorforge::fmacdpp16<15>(v64_acc, v81_bc, v60_data);
          float v82_bc = tensorforge::broadcast<32, 16, 1>(v80_lin);
          tensorforge::fmacdpp16<0>(v65_acc, v82_bc, v49_data);
          tensorforge::fmacdpp16<1>(v65_acc, v82_bc, v50_data);
          tensorforge::fmacdpp16<2>(v65_acc, v82_bc, v51_data);
          tensorforge::fmacdpp16<3>(v65_acc, v82_bc, v52_data);
          tensorforge::fmacdpp16<4>(v65_acc, v82_bc, v53_data);
          tensorforge::fmacdpp16<5>(v65_acc, v82_bc, v54_data);
          tensorforge::fmacdpp16<6>(v65_acc, v82_bc, v55_data);
          tensorforge::fmacdpp16<7>(v65_acc, v82_bc, v56_data);
          tensorforge::fmacdpp16<8>(v65_acc, v82_bc, v57_data);
          tensorforge::fmacdpp16<9>(v65_acc, v82_bc, v58_data);
          tensorforge::fmacdpp16<10>(v65_acc, v82_bc, v59_data);
          tensorforge::fmacdpp16<11>(v65_acc, v82_bc, v60_data);
          tensorforge::fmacdpp16<12>(v66_acc, v82_bc, v49_data);
          tensorforge::fmacdpp16<13>(v66_acc, v82_bc, v50_data);
          tensorforge::fmacdpp16<14>(v66_acc, v82_bc, v51_data);
          tensorforge::fmacdpp16<15>(v66_acc, v82_bc, v52_data);
          float v83_lin = r1[2];
          float v84_bc = tensorforge::broadcast<32, 16, 0>(v83_lin);
          tensorforge::fmacdpp16<0>(v66_acc, v84_bc, v53_data);
          tensorforge::fmacdpp16<1>(v66_acc, v84_bc, v54_data);
          tensorforge::fmacdpp16<2>(v66_acc, v84_bc, v55_data);
          tensorforge::fmacdpp16<3>(v66_acc, v84_bc, v56_data);
          tensorforge::fmacdpp16<4>(v66_acc, v84_bc, v57_data);
          tensorforge::fmacdpp16<5>(v66_acc, v84_bc, v58_data);
          tensorforge::fmacdpp16<6>(v66_acc, v84_bc, v59_data);
          tensorforge::fmacdpp16<7>(v66_acc, v84_bc, v60_data);
          tensorforge::fmacdpp16<8>(v67_acc, v84_bc, v49_data);
          tensorforge::fmacdpp16<9>(v67_acc, v84_bc, v50_data);
          tensorforge::fmacdpp16<10>(v67_acc, v84_bc, v51_data);
          tensorforge::fmacdpp16<11>(v67_acc, v84_bc, v52_data);
          tensorforge::fmacdpp16<12>(v67_acc, v84_bc, v53_data);
          tensorforge::fmacdpp16<13>(v67_acc, v84_bc, v54_data);
          tensorforge::fmacdpp16<14>(v67_acc, v84_bc, v55_data);
          tensorforge::fmacdpp16<15>(v67_acc, v84_bc, v56_data);
          float v85_bc = tensorforge::broadcast<32, 16, 1>(v83_lin);
          tensorforge::fmacdpp16<0>(v67_acc, v85_bc, v57_data);
          tensorforge::fmacdpp16<1>(v67_acc, v85_bc, v58_data);
          tensorforge::fmacdpp16<2>(v67_acc, v85_bc, v59_data);
          tensorforge::fmacdpp16<3>(v67_acc, v85_bc, v60_data);
          tensorforge::fmacdpp16<4>(v68_acc, v85_bc, v49_data);
          tensorforge::fmacdpp16<5>(v68_acc, v85_bc, v50_data);
          tensorforge::fmacdpp16<6>(v68_acc, v85_bc, v51_data);
          tensorforge::fmacdpp16<7>(v68_acc, v85_bc, v52_data);
          tensorforge::fmacdpp16<8>(v68_acc, v85_bc, v53_data);
          tensorforge::fmacdpp16<9>(v68_acc, v85_bc, v54_data);
          tensorforge::fmacdpp16<10>(v68_acc, v85_bc, v55_data);
          tensorforge::fmacdpp16<11>(v68_acc, v85_bc, v56_data);
          tensorforge::fmacdpp16<12>(v68_acc, v85_bc, v57_data);
          tensorforge::fmacdpp16<13>(v68_acc, v85_bc, v58_data);
          tensorforge::fmacdpp16<14>(v68_acc, v85_bc, v59_data);
          tensorforge::fmacdpp16<15>(v68_acc, v85_bc, v60_data);
          float v86_lin = r1[3];
          float v87_bc = tensorforge::broadcast<32, 16, 0>(v86_lin);
          tensorforge::fmacdpp16<0>(v69_acc, v87_bc, v49_data);
          tensorforge::fmacdpp16<1>(v69_acc, v87_bc, v50_data);
          tensorforge::fmacdpp16<2>(v69_acc, v87_bc, v51_data);
          tensorforge::fmacdpp16<3>(v69_acc, v87_bc, v52_data);
          tensorforge::fmacdpp16<4>(v69_acc, v87_bc, v53_data);
          tensorforge::fmacdpp16<5>(v69_acc, v87_bc, v54_data);
          tensorforge::fmacdpp16<6>(v69_acc, v87_bc, v55_data);
          tensorforge::fmacdpp16<7>(v69_acc, v87_bc, v56_data);
          tensorforge::fmacdpp16<8>(v69_acc, v87_bc, v57_data);
          tensorforge::fmacdpp16<9>(v69_acc, v87_bc, v58_data);
          tensorforge::fmacdpp16<10>(v69_acc, v87_bc, v59_data);
          tensorforge::fmacdpp16<11>(v69_acc, v87_bc, v60_data);
          tensorforge::fmacdpp16<12>(v70_acc, v87_bc, v49_data);
          tensorforge::fmacdpp16<13>(v70_acc, v87_bc, v50_data);
          tensorforge::fmacdpp16<14>(v70_acc, v87_bc, v51_data);
          tensorforge::fmacdpp16<15>(v70_acc, v87_bc, v52_data);
          float v88_bc = tensorforge::broadcast<32, 16, 1>(v86_lin);
          tensorforge::fmacdpp16<0>(v70_acc, v88_bc, v53_data);
          tensorforge::fmacdpp16<1>(v70_acc, v88_bc, v54_data);
          tensorforge::fmacdpp16<2>(v70_acc, v88_bc, v55_data);
          tensorforge::fmacdpp16<3>(v70_acc, v88_bc, v56_data);
          tensorforge::fmacdpp16<4>(v70_acc, v88_bc, v57_data);
          tensorforge::fmacdpp16<5>(v70_acc, v88_bc, v58_data);
          tensorforge::fmacdpp16<6>(v70_acc, v88_bc, v59_data);
          tensorforge::fmacdpp16<7>(v70_acc, v88_bc, v60_data);
          tensorforge::fmacdpp16<8>(v71_acc, v88_bc, v49_data);
          tensorforge::fmacdpp16<9>(v71_acc, v88_bc, v50_data);
          tensorforge::fmacdpp16<10>(v71_acc, v88_bc, v51_data);
          tensorforge::fmacdpp16<11>(v71_acc, v88_bc, v52_data);
          tensorforge::fmacdpp16<12>(v71_acc, v88_bc, v53_data);
          tensorforge::fmacdpp16<13>(v71_acc, v88_bc, v54_data);
          tensorforge::fmacdpp16<14>(v71_acc, v88_bc, v55_data);
          tensorforge::fmacdpp16<15>(v71_acc, v88_bc, v56_data);
          float v89_lin = r1[4];
          float v90_bc = tensorforge::broadcast<32, 16, 0>(v89_lin);
          tensorforge::fmacdpp16<0>(v71_acc, v90_bc, v57_data);
          tensorforge::fmacdpp16<1>(v71_acc, v90_bc, v58_data);
          tensorforge::fmacdpp16<2>(v71_acc, v90_bc, v59_data);
          tensorforge::fmacdpp16<3>(v71_acc, v90_bc, v60_data);
          tensorforge::fmacdpp16<4>(v72_acc, v90_bc, v49_data);
          tensorforge::fmacdpp16<5>(v72_acc, v90_bc, v50_data);
          tensorforge::fmacdpp16<6>(v72_acc, v90_bc, v51_data);
          tensorforge::fmacdpp16<7>(v72_acc, v90_bc, v52_data);
          tensorforge::fmacdpp16<8>(v72_acc, v90_bc, v53_data);
          tensorforge::fmacdpp16<9>(v72_acc, v90_bc, v54_data);
          tensorforge::fmacdpp16<10>(v72_acc, v90_bc, v55_data);
          tensorforge::fmacdpp16<11>(v72_acc, v90_bc, v56_data);
          tensorforge::fmacdpp16<12>(v72_acc, v90_bc, v57_data);
          tensorforge::fmacdpp16<13>(v72_acc, v90_bc, v58_data);
          tensorforge::fmacdpp16<14>(v72_acc, v90_bc, v59_data);
          tensorforge::fmacdpp16<15>(v72_acc, v90_bc, v60_data);
          float v91_bc = tensorforge::broadcast<32, 16, 1>(v89_lin);
          tensorforge::fmacdpp16<0>(v73_acc, v91_bc, v49_data);
          tensorforge::fmacdpp16<1>(v73_acc, v91_bc, v50_data);
          tensorforge::fmacdpp16<2>(v73_acc, v91_bc, v51_data);
          tensorforge::fmacdpp16<3>(v73_acc, v91_bc, v52_data);
          tensorforge::fmacdpp16<4>(v73_acc, v91_bc, v53_data);
          tensorforge::fmacdpp16<5>(v73_acc, v91_bc, v54_data);
          tensorforge::fmacdpp16<6>(v73_acc, v91_bc, v55_data);
          tensorforge::fmacdpp16<7>(v73_acc, v91_bc, v56_data);
          tensorforge::fmacdpp16<8>(v73_acc, v91_bc, v57_data);
          tensorforge::fmacdpp16<9>(v73_acc, v91_bc, v58_data);
          tensorforge::fmacdpp16<10>(v73_acc, v91_bc, v59_data);
          tensorforge::fmacdpp16<11>(v73_acc, v91_bc, v60_data);
          tensorforge::fmacdpp16<12>(v74_acc, v91_bc, v49_data);
          tensorforge::fmacdpp16<13>(v74_acc, v91_bc, v50_data);
          tensorforge::fmacdpp16<14>(v74_acc, v91_bc, v51_data);
          tensorforge::fmacdpp16<15>(v74_acc, v91_bc, v52_data);
          float v92_lin = r1[5];
          float v93_bc = tensorforge::broadcast<32, 16, 0>(v92_lin);
          tensorforge::fmacdpp16<0>(v74_acc, v93_bc, v53_data);
          tensorforge::fmacdpp16<1>(v74_acc, v93_bc, v54_data);
          tensorforge::fmacdpp16<2>(v74_acc, v93_bc, v55_data);
          tensorforge::fmacdpp16<3>(v74_acc, v93_bc, v56_data);
          tensorforge::fmacdpp16<4>(v74_acc, v93_bc, v57_data);
          tensorforge::fmacdpp16<5>(v74_acc, v93_bc, v58_data);
          tensorforge::fmacdpp16<6>(v74_acc, v93_bc, v59_data);
          tensorforge::fmacdpp16<7>(v74_acc, v93_bc, v60_data);
          tensorforge::fmacdpp16<8>(v75_acc, v93_bc, v49_data);
          tensorforge::fmacdpp16<9>(v75_acc, v93_bc, v50_data);
          tensorforge::fmacdpp16<10>(v75_acc, v93_bc, v51_data);
          tensorforge::fmacdpp16<11>(v75_acc, v93_bc, v52_data);
          tensorforge::fmacdpp16<12>(v75_acc, v93_bc, v53_data);
          tensorforge::fmacdpp16<13>(v75_acc, v93_bc, v54_data);
          tensorforge::fmacdpp16<14>(v75_acc, v93_bc, v55_data);
          tensorforge::fmacdpp16<15>(v75_acc, v93_bc, v56_data);
          float v94_bc = tensorforge::broadcast<32, 16, 1>(v92_lin);
          tensorforge::fmacdpp16<0>(v75_acc, v94_bc, v57_data);
          tensorforge::fmacdpp16<1>(v75_acc, v94_bc, v58_data);
          tensorforge::fmacdpp16<2>(v75_acc, v94_bc, v59_data);
          tensorforge::fmacdpp16<3>(v75_acc, v94_bc, v60_data);
          tensorforge::fmacdpp16<4>(v76_acc, v94_bc, v49_data);
          tensorforge::fmacdpp16<5>(v76_acc, v94_bc, v50_data);
          tensorforge::fmacdpp16<6>(v76_acc, v94_bc, v51_data);
          tensorforge::fmacdpp16<7>(v76_acc, v94_bc, v52_data);
          tensorforge::fmacdpp16<8>(v76_acc, v94_bc, v53_data);
          tensorforge::fmacdpp16<9>(v76_acc, v94_bc, v54_data);
          tensorforge::fmacdpp16<10>(v76_acc, v94_bc, v55_data);
          tensorforge::fmacdpp16<11>(v76_acc, v94_bc, v56_data);
          tensorforge::fmacdpp16<12>(v76_acc, v94_bc, v57_data);
          tensorforge::fmacdpp16<13>(v76_acc, v94_bc, v58_data);
          tensorforge::fmacdpp16<14>(v76_acc, v94_bc, v59_data);
          tensorforge::fmacdpp16<15>(v76_acc, v94_bc, v60_data);
          r2[0] = v61_acc;
          r2[1] = v62_acc;
          r2[2] = v63_acc;
          r2[3] = v64_acc;
          r2[4] = v65_acc;
          r2[5] = v66_acc;
          r2[6] = v67_acc;
          r2[7] = v68_acc;
          r2[8] = v69_acc;
          r2[9] = v70_acc;
          r2[10] = v71_acc;
          r2[11] = v72_acc;
          r2[12] = v73_acc;
          r2[13] = v74_acc;
          r2[14] = v75_acc;
          r2[15] = v76_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v98_i0 = 0; v98_i0 < 1; ++v98_i0) {
            int32_t v106_lead = v14_lead + (v98_i0 * 32);
            #pragma unroll
            for (int32_t v99_i1 = 0; v99_i1 < 16; ++v99_i1) {
              float v101_data = r2[(v98_i0 + v99_i1)];
              glb_m0[(v106_lead + (v99_i1 * 32))] = v101_data;
            }
          }
          float r4[8]{};
          // r4 = load{g>r}(glb_m4);
          float v110_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v110_lin;
          float v111_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v111_lin;
          float v112_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v112_lin;
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v117_i0 = 0; v117_i0 < 1; ++v117_i0) {
            int32_t v123_lead = v14_lead + (v117_i0 * 32);
            #pragma unroll
            for (int32_t v118_i1 = 0; v118_i1 < 12; ++v118_i1) {
              float v126_data = __builtin_nontemporal_load(&glb_m5[(v123_lead + (v118_i1 * 32))]);
              r6[(v117_i0 + v118_i1)] = v126_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[8]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v129_data = r3[0];
          float v130_data = r3[1];
          float v131_data = r3[2];
          float v132_data = r3[3];
          float v133_data = r3[4];
          float v134_data = r3[5];
          float v135_data = r3[6];
          float v136_data = r3[7];
          float v137_data = r3[8];
          float v138_data = r3[9];
          float v139_data = r3[10];
          float v140_data = r3[11];
          float v141_acc{};
          float v142_acc{};
          float v143_acc{};
          float v144_acc{};
          float v145_acc{};
          float v146_acc{};
          float v147_acc{};
          float v148_acc{};
          float v149_lin = r4[0];
          float v150_bc = tensorforge::broadcast<32, 16, 0>(v149_lin);
          tensorforge::fmacdpp16<0>(v141_acc, v150_bc, v129_data);
          tensorforge::fmacdpp16<1>(v141_acc, v150_bc, v130_data);
          tensorforge::fmacdpp16<2>(v141_acc, v150_bc, v131_data);
          tensorforge::fmacdpp16<3>(v141_acc, v150_bc, v132_data);
          tensorforge::fmacdpp16<4>(v141_acc, v150_bc, v133_data);
          tensorforge::fmacdpp16<5>(v141_acc, v150_bc, v134_data);
          tensorforge::fmacdpp16<6>(v141_acc, v150_bc, v135_data);
          tensorforge::fmacdpp16<7>(v141_acc, v150_bc, v136_data);
          tensorforge::fmacdpp16<8>(v141_acc, v150_bc, v137_data);
          tensorforge::fmacdpp16<9>(v141_acc, v150_bc, v138_data);
          tensorforge::fmacdpp16<10>(v141_acc, v150_bc, v139_data);
          tensorforge::fmacdpp16<11>(v141_acc, v150_bc, v140_data);
          tensorforge::fmacdpp16<12>(v142_acc, v150_bc, v129_data);
          tensorforge::fmacdpp16<13>(v142_acc, v150_bc, v130_data);
          tensorforge::fmacdpp16<14>(v142_acc, v150_bc, v131_data);
          tensorforge::fmacdpp16<15>(v142_acc, v150_bc, v132_data);
          float v151_bc = tensorforge::broadcast<32, 16, 1>(v149_lin);
          tensorforge::fmacdpp16<0>(v142_acc, v151_bc, v133_data);
          tensorforge::fmacdpp16<1>(v142_acc, v151_bc, v134_data);
          tensorforge::fmacdpp16<2>(v142_acc, v151_bc, v135_data);
          tensorforge::fmacdpp16<3>(v142_acc, v151_bc, v136_data);
          tensorforge::fmacdpp16<4>(v142_acc, v151_bc, v137_data);
          tensorforge::fmacdpp16<5>(v142_acc, v151_bc, v138_data);
          tensorforge::fmacdpp16<6>(v142_acc, v151_bc, v139_data);
          tensorforge::fmacdpp16<7>(v142_acc, v151_bc, v140_data);
          tensorforge::fmacdpp16<8>(v143_acc, v151_bc, v129_data);
          tensorforge::fmacdpp16<9>(v143_acc, v151_bc, v130_data);
          tensorforge::fmacdpp16<10>(v143_acc, v151_bc, v131_data);
          tensorforge::fmacdpp16<11>(v143_acc, v151_bc, v132_data);
          tensorforge::fmacdpp16<12>(v143_acc, v151_bc, v133_data);
          tensorforge::fmacdpp16<13>(v143_acc, v151_bc, v134_data);
          tensorforge::fmacdpp16<14>(v143_acc, v151_bc, v135_data);
          tensorforge::fmacdpp16<15>(v143_acc, v151_bc, v136_data);
          float v152_lin = r4[1];
          float v153_bc = tensorforge::broadcast<32, 16, 0>(v152_lin);
          tensorforge::fmacdpp16<0>(v143_acc, v153_bc, v137_data);
          tensorforge::fmacdpp16<1>(v143_acc, v153_bc, v138_data);
          tensorforge::fmacdpp16<2>(v143_acc, v153_bc, v139_data);
          tensorforge::fmacdpp16<3>(v143_acc, v153_bc, v140_data);
          tensorforge::fmacdpp16<4>(v144_acc, v153_bc, v129_data);
          tensorforge::fmacdpp16<5>(v144_acc, v153_bc, v130_data);
          tensorforge::fmacdpp16<6>(v144_acc, v153_bc, v131_data);
          tensorforge::fmacdpp16<7>(v144_acc, v153_bc, v132_data);
          tensorforge::fmacdpp16<8>(v144_acc, v153_bc, v133_data);
          tensorforge::fmacdpp16<9>(v144_acc, v153_bc, v134_data);
          tensorforge::fmacdpp16<10>(v144_acc, v153_bc, v135_data);
          tensorforge::fmacdpp16<11>(v144_acc, v153_bc, v136_data);
          tensorforge::fmacdpp16<12>(v144_acc, v153_bc, v137_data);
          tensorforge::fmacdpp16<13>(v144_acc, v153_bc, v138_data);
          tensorforge::fmacdpp16<14>(v144_acc, v153_bc, v139_data);
          tensorforge::fmacdpp16<15>(v144_acc, v153_bc, v140_data);
          float v154_bc = tensorforge::broadcast<32, 16, 1>(v152_lin);
          tensorforge::fmacdpp16<0>(v145_acc, v154_bc, v129_data);
          tensorforge::fmacdpp16<1>(v145_acc, v154_bc, v130_data);
          tensorforge::fmacdpp16<2>(v145_acc, v154_bc, v131_data);
          tensorforge::fmacdpp16<3>(v145_acc, v154_bc, v132_data);
          tensorforge::fmacdpp16<4>(v145_acc, v154_bc, v133_data);
          tensorforge::fmacdpp16<5>(v145_acc, v154_bc, v134_data);
          tensorforge::fmacdpp16<6>(v145_acc, v154_bc, v135_data);
          tensorforge::fmacdpp16<7>(v145_acc, v154_bc, v136_data);
          tensorforge::fmacdpp16<8>(v145_acc, v154_bc, v137_data);
          tensorforge::fmacdpp16<9>(v145_acc, v154_bc, v138_data);
          tensorforge::fmacdpp16<10>(v145_acc, v154_bc, v139_data);
          tensorforge::fmacdpp16<11>(v145_acc, v154_bc, v140_data);
          tensorforge::fmacdpp16<12>(v146_acc, v154_bc, v129_data);
          tensorforge::fmacdpp16<13>(v146_acc, v154_bc, v130_data);
          tensorforge::fmacdpp16<14>(v146_acc, v154_bc, v131_data);
          tensorforge::fmacdpp16<15>(v146_acc, v154_bc, v132_data);
          float v155_lin = r4[2];
          float v156_bc = tensorforge::broadcast<32, 16, 0>(v155_lin);
          tensorforge::fmacdpp16<0>(v146_acc, v156_bc, v133_data);
          tensorforge::fmacdpp16<1>(v146_acc, v156_bc, v134_data);
          tensorforge::fmacdpp16<2>(v146_acc, v156_bc, v135_data);
          tensorforge::fmacdpp16<3>(v146_acc, v156_bc, v136_data);
          tensorforge::fmacdpp16<4>(v146_acc, v156_bc, v137_data);
          tensorforge::fmacdpp16<5>(v146_acc, v156_bc, v138_data);
          tensorforge::fmacdpp16<6>(v146_acc, v156_bc, v139_data);
          tensorforge::fmacdpp16<7>(v146_acc, v156_bc, v140_data);
          tensorforge::fmacdpp16<8>(v147_acc, v156_bc, v129_data);
          tensorforge::fmacdpp16<9>(v147_acc, v156_bc, v130_data);
          tensorforge::fmacdpp16<10>(v147_acc, v156_bc, v131_data);
          tensorforge::fmacdpp16<11>(v147_acc, v156_bc, v132_data);
          tensorforge::fmacdpp16<12>(v147_acc, v156_bc, v133_data);
          tensorforge::fmacdpp16<13>(v147_acc, v156_bc, v134_data);
          tensorforge::fmacdpp16<14>(v147_acc, v156_bc, v135_data);
          tensorforge::fmacdpp16<15>(v147_acc, v156_bc, v136_data);
          float v157_bc = tensorforge::broadcast<32, 16, 1>(v155_lin);
          tensorforge::fmacdpp16<0>(v147_acc, v157_bc, v137_data);
          tensorforge::fmacdpp16<1>(v147_acc, v157_bc, v138_data);
          tensorforge::fmacdpp16<2>(v147_acc, v157_bc, v139_data);
          tensorforge::fmacdpp16<3>(v147_acc, v157_bc, v140_data);
          tensorforge::fmacdpp16<4>(v148_acc, v157_bc, v129_data);
          tensorforge::fmacdpp16<5>(v148_acc, v157_bc, v130_data);
          tensorforge::fmacdpp16<6>(v148_acc, v157_bc, v131_data);
          tensorforge::fmacdpp16<7>(v148_acc, v157_bc, v132_data);
          tensorforge::fmacdpp16<8>(v148_acc, v157_bc, v133_data);
          tensorforge::fmacdpp16<9>(v148_acc, v157_bc, v134_data);
          tensorforge::fmacdpp16<10>(v148_acc, v157_bc, v135_data);
          tensorforge::fmacdpp16<11>(v148_acc, v157_bc, v136_data);
          tensorforge::fmacdpp16<12>(v148_acc, v157_bc, v137_data);
          tensorforge::fmacdpp16<13>(v148_acc, v157_bc, v138_data);
          tensorforge::fmacdpp16<14>(v148_acc, v157_bc, v139_data);
          tensorforge::fmacdpp16<15>(v148_acc, v157_bc, v140_data);
          r5[0] = v141_acc;
          r5[1] = v142_acc;
          r5[2] = v143_acc;
          r5[3] = v144_acc;
          r5[4] = v145_acc;
          r5[5] = v146_acc;
          r5[6] = v147_acc;
          r5[7] = v148_acc;
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v161_i0 = 0; v161_i0 < 1; ++v161_i0) {
            int32_t v169_lead = v14_lead + (v161_i0 * 32);
            #pragma unroll
            for (int32_t v162_i1 = 0; v162_i1 < 8; ++v162_i1) {
              float v164_data = r5[(v161_i0 + v162_i1)];
              int32_t v171_a = v169_lead + (v162_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v171_a], v164_data);
            }
          }
          float r7[8]{};
          // r7 = load{g>r}(glb_m6);
          float v173_lin = glb_m6[0 + threadIdx.x * 1];
          r7[0] = v173_lin;
          float v174_lin = glb_m6[32 + threadIdx.x * 1];
          r7[1] = v174_lin;
          float v175_lin = glb_m6[64 + threadIdx.x * 1];
          r7[2] = v175_lin;
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
          float r8[8]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          float v177_data = r6[0];
          float v178_data = r6[1];
          float v179_data = r6[2];
          float v180_data = r6[3];
          float v181_data = r6[4];
          float v182_data = r6[5];
          float v183_data = r6[6];
          float v184_data = r6[7];
          float v185_data = r6[8];
          float v186_data = r6[9];
          float v187_data = r6[10];
          float v188_data = r6[11];
          float v189_acc{};
          float v190_acc{};
          float v191_acc{};
          float v192_acc{};
          float v193_acc{};
          float v194_acc{};
          float v195_acc{};
          float v196_acc{};
          float v197_lin = r7[0];
          float v198_bc = tensorforge::broadcast<32, 16, 0>(v197_lin);
          tensorforge::fmacdpp16<0>(v189_acc, v198_bc, v177_data);
          tensorforge::fmacdpp16<1>(v189_acc, v198_bc, v178_data);
          tensorforge::fmacdpp16<2>(v189_acc, v198_bc, v179_data);
          tensorforge::fmacdpp16<3>(v189_acc, v198_bc, v180_data);
          tensorforge::fmacdpp16<4>(v189_acc, v198_bc, v181_data);
          tensorforge::fmacdpp16<5>(v189_acc, v198_bc, v182_data);
          tensorforge::fmacdpp16<6>(v189_acc, v198_bc, v183_data);
          tensorforge::fmacdpp16<7>(v189_acc, v198_bc, v184_data);
          tensorforge::fmacdpp16<8>(v189_acc, v198_bc, v185_data);
          tensorforge::fmacdpp16<9>(v189_acc, v198_bc, v186_data);
          tensorforge::fmacdpp16<10>(v189_acc, v198_bc, v187_data);
          tensorforge::fmacdpp16<11>(v189_acc, v198_bc, v188_data);
          tensorforge::fmacdpp16<12>(v190_acc, v198_bc, v177_data);
          tensorforge::fmacdpp16<13>(v190_acc, v198_bc, v178_data);
          tensorforge::fmacdpp16<14>(v190_acc, v198_bc, v179_data);
          tensorforge::fmacdpp16<15>(v190_acc, v198_bc, v180_data);
          float v199_bc = tensorforge::broadcast<32, 16, 1>(v197_lin);
          tensorforge::fmacdpp16<0>(v190_acc, v199_bc, v181_data);
          tensorforge::fmacdpp16<1>(v190_acc, v199_bc, v182_data);
          tensorforge::fmacdpp16<2>(v190_acc, v199_bc, v183_data);
          tensorforge::fmacdpp16<3>(v190_acc, v199_bc, v184_data);
          tensorforge::fmacdpp16<4>(v190_acc, v199_bc, v185_data);
          tensorforge::fmacdpp16<5>(v190_acc, v199_bc, v186_data);
          tensorforge::fmacdpp16<6>(v190_acc, v199_bc, v187_data);
          tensorforge::fmacdpp16<7>(v190_acc, v199_bc, v188_data);
          tensorforge::fmacdpp16<8>(v191_acc, v199_bc, v177_data);
          tensorforge::fmacdpp16<9>(v191_acc, v199_bc, v178_data);
          tensorforge::fmacdpp16<10>(v191_acc, v199_bc, v179_data);
          tensorforge::fmacdpp16<11>(v191_acc, v199_bc, v180_data);
          tensorforge::fmacdpp16<12>(v191_acc, v199_bc, v181_data);
          tensorforge::fmacdpp16<13>(v191_acc, v199_bc, v182_data);
          tensorforge::fmacdpp16<14>(v191_acc, v199_bc, v183_data);
          tensorforge::fmacdpp16<15>(v191_acc, v199_bc, v184_data);
          float v200_lin = r7[1];
          float v201_bc = tensorforge::broadcast<32, 16, 0>(v200_lin);
          tensorforge::fmacdpp16<0>(v191_acc, v201_bc, v185_data);
          tensorforge::fmacdpp16<1>(v191_acc, v201_bc, v186_data);
          tensorforge::fmacdpp16<2>(v191_acc, v201_bc, v187_data);
          tensorforge::fmacdpp16<3>(v191_acc, v201_bc, v188_data);
          tensorforge::fmacdpp16<4>(v192_acc, v201_bc, v177_data);
          tensorforge::fmacdpp16<5>(v192_acc, v201_bc, v178_data);
          tensorforge::fmacdpp16<6>(v192_acc, v201_bc, v179_data);
          tensorforge::fmacdpp16<7>(v192_acc, v201_bc, v180_data);
          tensorforge::fmacdpp16<8>(v192_acc, v201_bc, v181_data);
          tensorforge::fmacdpp16<9>(v192_acc, v201_bc, v182_data);
          tensorforge::fmacdpp16<10>(v192_acc, v201_bc, v183_data);
          tensorforge::fmacdpp16<11>(v192_acc, v201_bc, v184_data);
          tensorforge::fmacdpp16<12>(v192_acc, v201_bc, v185_data);
          tensorforge::fmacdpp16<13>(v192_acc, v201_bc, v186_data);
          tensorforge::fmacdpp16<14>(v192_acc, v201_bc, v187_data);
          tensorforge::fmacdpp16<15>(v192_acc, v201_bc, v188_data);
          float v202_bc = tensorforge::broadcast<32, 16, 1>(v200_lin);
          tensorforge::fmacdpp16<0>(v193_acc, v202_bc, v177_data);
          tensorforge::fmacdpp16<1>(v193_acc, v202_bc, v178_data);
          tensorforge::fmacdpp16<2>(v193_acc, v202_bc, v179_data);
          tensorforge::fmacdpp16<3>(v193_acc, v202_bc, v180_data);
          tensorforge::fmacdpp16<4>(v193_acc, v202_bc, v181_data);
          tensorforge::fmacdpp16<5>(v193_acc, v202_bc, v182_data);
          tensorforge::fmacdpp16<6>(v193_acc, v202_bc, v183_data);
          tensorforge::fmacdpp16<7>(v193_acc, v202_bc, v184_data);
          tensorforge::fmacdpp16<8>(v193_acc, v202_bc, v185_data);
          tensorforge::fmacdpp16<9>(v193_acc, v202_bc, v186_data);
          tensorforge::fmacdpp16<10>(v193_acc, v202_bc, v187_data);
          tensorforge::fmacdpp16<11>(v193_acc, v202_bc, v188_data);
          tensorforge::fmacdpp16<12>(v194_acc, v202_bc, v177_data);
          tensorforge::fmacdpp16<13>(v194_acc, v202_bc, v178_data);
          tensorforge::fmacdpp16<14>(v194_acc, v202_bc, v179_data);
          tensorforge::fmacdpp16<15>(v194_acc, v202_bc, v180_data);
          float v203_lin = r7[2];
          float v204_bc = tensorforge::broadcast<32, 16, 0>(v203_lin);
          tensorforge::fmacdpp16<0>(v194_acc, v204_bc, v181_data);
          tensorforge::fmacdpp16<1>(v194_acc, v204_bc, v182_data);
          tensorforge::fmacdpp16<2>(v194_acc, v204_bc, v183_data);
          tensorforge::fmacdpp16<3>(v194_acc, v204_bc, v184_data);
          tensorforge::fmacdpp16<4>(v194_acc, v204_bc, v185_data);
          tensorforge::fmacdpp16<5>(v194_acc, v204_bc, v186_data);
          tensorforge::fmacdpp16<6>(v194_acc, v204_bc, v187_data);
          tensorforge::fmacdpp16<7>(v194_acc, v204_bc, v188_data);
          tensorforge::fmacdpp16<8>(v195_acc, v204_bc, v177_data);
          tensorforge::fmacdpp16<9>(v195_acc, v204_bc, v178_data);
          tensorforge::fmacdpp16<10>(v195_acc, v204_bc, v179_data);
          tensorforge::fmacdpp16<11>(v195_acc, v204_bc, v180_data);
          tensorforge::fmacdpp16<12>(v195_acc, v204_bc, v181_data);
          tensorforge::fmacdpp16<13>(v195_acc, v204_bc, v182_data);
          tensorforge::fmacdpp16<14>(v195_acc, v204_bc, v183_data);
          tensorforge::fmacdpp16<15>(v195_acc, v204_bc, v184_data);
          float v205_bc = tensorforge::broadcast<32, 16, 1>(v203_lin);
          tensorforge::fmacdpp16<0>(v195_acc, v205_bc, v185_data);
          tensorforge::fmacdpp16<1>(v195_acc, v205_bc, v186_data);
          tensorforge::fmacdpp16<2>(v195_acc, v205_bc, v187_data);
          tensorforge::fmacdpp16<3>(v195_acc, v205_bc, v188_data);
          tensorforge::fmacdpp16<4>(v196_acc, v205_bc, v177_data);
          tensorforge::fmacdpp16<5>(v196_acc, v205_bc, v178_data);
          tensorforge::fmacdpp16<6>(v196_acc, v205_bc, v179_data);
          tensorforge::fmacdpp16<7>(v196_acc, v205_bc, v180_data);
          tensorforge::fmacdpp16<8>(v196_acc, v205_bc, v181_data);
          tensorforge::fmacdpp16<9>(v196_acc, v205_bc, v182_data);
          tensorforge::fmacdpp16<10>(v196_acc, v205_bc, v183_data);
          tensorforge::fmacdpp16<11>(v196_acc, v205_bc, v184_data);
          tensorforge::fmacdpp16<12>(v196_acc, v205_bc, v185_data);
          tensorforge::fmacdpp16<13>(v196_acc, v205_bc, v186_data);
          tensorforge::fmacdpp16<14>(v196_acc, v205_bc, v187_data);
          tensorforge::fmacdpp16<15>(v196_acc, v205_bc, v188_data);
          r8[0] = v189_acc;
          r8[1] = v190_acc;
          r8[2] = v191_acc;
          r8[3] = v192_acc;
          r8[4] = v193_acc;
          r8[5] = v194_acc;
          r8[6] = v195_acc;
          r8[7] = v196_acc;
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v209_i0 = 0; v209_i0 < 1; ++v209_i0) {
            int32_t v217_lead = v14_lead + (v209_i0 * 32);
            #pragma unroll
            for (int32_t v210_i1 = 0; v210_i1 < 8; ++v210_i1) {
              float v212_data = r8[(v209_i0 + v210_i1)];
              int32_t v220_a = v217_lead + ((v210_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v220_a], v212_data);
            }
          }
        }
      }
    }
  }
}

