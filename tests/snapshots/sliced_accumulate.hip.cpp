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
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 12; ++v4_i1) {
              int32_t v10_a = v4_i1 * 32;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + v10_a)]);
              int32_t v20_a = v3_i0 + v4_i1;
              r0[v20_a] = v19_data;
            }
          }
          float r1[16]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[1] = v32;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r1[2] = v64;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r1[3] = v96;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r1[4] = v128;
            float v160 = glb_m2[160 + threadIdx.x * 1];
            r1[5] = v160;
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[12]{};
          // r3 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v29_lead = v24_i0 * 32;
            int32_t v30_lead = v2_lead + v29_lead;
            int32_t v37_lead = v2_lead + v29_lead;
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              int32_t v31_a = v25_i1 * 32;
              int32_t v32_a = v30_lead + v31_a;
              float v40_data = __builtin_nontemporal_load(&glb_m3[(v37_lead + v31_a)]);
              int32_t v41_a = v24_i0 + v25_i1;
              r3[v41_a] = v40_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          auto& ir2 = r2;
          float v42_data = r0[0];
          float v43_data = r0[1];
          float v44_data = r0[2];
          float v45_data = r0[3];
          float v46_data = r0[4];
          float v47_data = r0[5];
          float v48_data = r0[6];
          float v49_data = r0[7];
          float v50_data = r0[8];
          float v51_data = r0[9];
          float v52_data = r0[10];
          float v53_data = r0[11];
          float v54_acc{};
          float v55_acc{};
          float v56_acc{};
          float v57_acc{};
          float v58_acc{};
          float v59_acc{};
          float v60_acc{};
          float v61_acc{};
          float v62_acc{};
          float v63_acc{};
          float v64_acc{};
          float v65_acc{};
          float v66_acc{};
          float v67_acc{};
          float v68_acc{};
          float v69_acc{};
          float v70_lin = r1[0];
          float v71_bc = tensorforge::broadcast<32, 16, 0>(v70_lin);
          tensorforge::fmacdpp16<0>(v54_acc, v71_bc, v42_data);
          tensorforge::fmacdpp16<1>(v54_acc, v71_bc, v43_data);
          tensorforge::fmacdpp16<2>(v54_acc, v71_bc, v44_data);
          tensorforge::fmacdpp16<3>(v54_acc, v71_bc, v45_data);
          tensorforge::fmacdpp16<4>(v54_acc, v71_bc, v46_data);
          tensorforge::fmacdpp16<5>(v54_acc, v71_bc, v47_data);
          tensorforge::fmacdpp16<6>(v54_acc, v71_bc, v48_data);
          tensorforge::fmacdpp16<7>(v54_acc, v71_bc, v49_data);
          tensorforge::fmacdpp16<8>(v54_acc, v71_bc, v50_data);
          tensorforge::fmacdpp16<9>(v54_acc, v71_bc, v51_data);
          tensorforge::fmacdpp16<10>(v54_acc, v71_bc, v52_data);
          tensorforge::fmacdpp16<11>(v54_acc, v71_bc, v53_data);
          tensorforge::fmacdpp16<12>(v55_acc, v71_bc, v42_data);
          tensorforge::fmacdpp16<13>(v55_acc, v71_bc, v43_data);
          tensorforge::fmacdpp16<14>(v55_acc, v71_bc, v44_data);
          tensorforge::fmacdpp16<15>(v55_acc, v71_bc, v45_data);
          float v72_bc = tensorforge::broadcast<32, 16, 1>(v70_lin);
          tensorforge::fmacdpp16<0>(v55_acc, v72_bc, v46_data);
          tensorforge::fmacdpp16<1>(v55_acc, v72_bc, v47_data);
          tensorforge::fmacdpp16<2>(v55_acc, v72_bc, v48_data);
          tensorforge::fmacdpp16<3>(v55_acc, v72_bc, v49_data);
          tensorforge::fmacdpp16<4>(v55_acc, v72_bc, v50_data);
          tensorforge::fmacdpp16<5>(v55_acc, v72_bc, v51_data);
          tensorforge::fmacdpp16<6>(v55_acc, v72_bc, v52_data);
          tensorforge::fmacdpp16<7>(v55_acc, v72_bc, v53_data);
          tensorforge::fmacdpp16<8>(v56_acc, v72_bc, v42_data);
          tensorforge::fmacdpp16<9>(v56_acc, v72_bc, v43_data);
          tensorforge::fmacdpp16<10>(v56_acc, v72_bc, v44_data);
          tensorforge::fmacdpp16<11>(v56_acc, v72_bc, v45_data);
          tensorforge::fmacdpp16<12>(v56_acc, v72_bc, v46_data);
          tensorforge::fmacdpp16<13>(v56_acc, v72_bc, v47_data);
          tensorforge::fmacdpp16<14>(v56_acc, v72_bc, v48_data);
          tensorforge::fmacdpp16<15>(v56_acc, v72_bc, v49_data);
          float v73_lin = r1[1];
          float v74_bc = tensorforge::broadcast<32, 16, 0>(v73_lin);
          tensorforge::fmacdpp16<0>(v56_acc, v74_bc, v50_data);
          tensorforge::fmacdpp16<1>(v56_acc, v74_bc, v51_data);
          tensorforge::fmacdpp16<2>(v56_acc, v74_bc, v52_data);
          tensorforge::fmacdpp16<3>(v56_acc, v74_bc, v53_data);
          tensorforge::fmacdpp16<4>(v57_acc, v74_bc, v42_data);
          tensorforge::fmacdpp16<5>(v57_acc, v74_bc, v43_data);
          tensorforge::fmacdpp16<6>(v57_acc, v74_bc, v44_data);
          tensorforge::fmacdpp16<7>(v57_acc, v74_bc, v45_data);
          tensorforge::fmacdpp16<8>(v57_acc, v74_bc, v46_data);
          tensorforge::fmacdpp16<9>(v57_acc, v74_bc, v47_data);
          tensorforge::fmacdpp16<10>(v57_acc, v74_bc, v48_data);
          tensorforge::fmacdpp16<11>(v57_acc, v74_bc, v49_data);
          tensorforge::fmacdpp16<12>(v57_acc, v74_bc, v50_data);
          tensorforge::fmacdpp16<13>(v57_acc, v74_bc, v51_data);
          tensorforge::fmacdpp16<14>(v57_acc, v74_bc, v52_data);
          tensorforge::fmacdpp16<15>(v57_acc, v74_bc, v53_data);
          float v75_bc = tensorforge::broadcast<32, 16, 1>(v73_lin);
          tensorforge::fmacdpp16<0>(v58_acc, v75_bc, v42_data);
          tensorforge::fmacdpp16<1>(v58_acc, v75_bc, v43_data);
          tensorforge::fmacdpp16<2>(v58_acc, v75_bc, v44_data);
          tensorforge::fmacdpp16<3>(v58_acc, v75_bc, v45_data);
          tensorforge::fmacdpp16<4>(v58_acc, v75_bc, v46_data);
          tensorforge::fmacdpp16<5>(v58_acc, v75_bc, v47_data);
          tensorforge::fmacdpp16<6>(v58_acc, v75_bc, v48_data);
          tensorforge::fmacdpp16<7>(v58_acc, v75_bc, v49_data);
          tensorforge::fmacdpp16<8>(v58_acc, v75_bc, v50_data);
          tensorforge::fmacdpp16<9>(v58_acc, v75_bc, v51_data);
          tensorforge::fmacdpp16<10>(v58_acc, v75_bc, v52_data);
          tensorforge::fmacdpp16<11>(v58_acc, v75_bc, v53_data);
          tensorforge::fmacdpp16<12>(v59_acc, v75_bc, v42_data);
          tensorforge::fmacdpp16<13>(v59_acc, v75_bc, v43_data);
          tensorforge::fmacdpp16<14>(v59_acc, v75_bc, v44_data);
          tensorforge::fmacdpp16<15>(v59_acc, v75_bc, v45_data);
          float v76_lin = r1[2];
          float v77_bc = tensorforge::broadcast<32, 16, 0>(v76_lin);
          tensorforge::fmacdpp16<0>(v59_acc, v77_bc, v46_data);
          tensorforge::fmacdpp16<1>(v59_acc, v77_bc, v47_data);
          tensorforge::fmacdpp16<2>(v59_acc, v77_bc, v48_data);
          tensorforge::fmacdpp16<3>(v59_acc, v77_bc, v49_data);
          tensorforge::fmacdpp16<4>(v59_acc, v77_bc, v50_data);
          tensorforge::fmacdpp16<5>(v59_acc, v77_bc, v51_data);
          tensorforge::fmacdpp16<6>(v59_acc, v77_bc, v52_data);
          tensorforge::fmacdpp16<7>(v59_acc, v77_bc, v53_data);
          tensorforge::fmacdpp16<8>(v60_acc, v77_bc, v42_data);
          tensorforge::fmacdpp16<9>(v60_acc, v77_bc, v43_data);
          tensorforge::fmacdpp16<10>(v60_acc, v77_bc, v44_data);
          tensorforge::fmacdpp16<11>(v60_acc, v77_bc, v45_data);
          tensorforge::fmacdpp16<12>(v60_acc, v77_bc, v46_data);
          tensorforge::fmacdpp16<13>(v60_acc, v77_bc, v47_data);
          tensorforge::fmacdpp16<14>(v60_acc, v77_bc, v48_data);
          tensorforge::fmacdpp16<15>(v60_acc, v77_bc, v49_data);
          float v78_bc = tensorforge::broadcast<32, 16, 1>(v76_lin);
          tensorforge::fmacdpp16<0>(v60_acc, v78_bc, v50_data);
          tensorforge::fmacdpp16<1>(v60_acc, v78_bc, v51_data);
          tensorforge::fmacdpp16<2>(v60_acc, v78_bc, v52_data);
          tensorforge::fmacdpp16<3>(v60_acc, v78_bc, v53_data);
          tensorforge::fmacdpp16<4>(v61_acc, v78_bc, v42_data);
          tensorforge::fmacdpp16<5>(v61_acc, v78_bc, v43_data);
          tensorforge::fmacdpp16<6>(v61_acc, v78_bc, v44_data);
          tensorforge::fmacdpp16<7>(v61_acc, v78_bc, v45_data);
          tensorforge::fmacdpp16<8>(v61_acc, v78_bc, v46_data);
          tensorforge::fmacdpp16<9>(v61_acc, v78_bc, v47_data);
          tensorforge::fmacdpp16<10>(v61_acc, v78_bc, v48_data);
          tensorforge::fmacdpp16<11>(v61_acc, v78_bc, v49_data);
          tensorforge::fmacdpp16<12>(v61_acc, v78_bc, v50_data);
          tensorforge::fmacdpp16<13>(v61_acc, v78_bc, v51_data);
          tensorforge::fmacdpp16<14>(v61_acc, v78_bc, v52_data);
          tensorforge::fmacdpp16<15>(v61_acc, v78_bc, v53_data);
          float v79_lin = r1[3];
          float v80_bc = tensorforge::broadcast<32, 16, 0>(v79_lin);
          tensorforge::fmacdpp16<0>(v62_acc, v80_bc, v42_data);
          tensorforge::fmacdpp16<1>(v62_acc, v80_bc, v43_data);
          tensorforge::fmacdpp16<2>(v62_acc, v80_bc, v44_data);
          tensorforge::fmacdpp16<3>(v62_acc, v80_bc, v45_data);
          tensorforge::fmacdpp16<4>(v62_acc, v80_bc, v46_data);
          tensorforge::fmacdpp16<5>(v62_acc, v80_bc, v47_data);
          tensorforge::fmacdpp16<6>(v62_acc, v80_bc, v48_data);
          tensorforge::fmacdpp16<7>(v62_acc, v80_bc, v49_data);
          tensorforge::fmacdpp16<8>(v62_acc, v80_bc, v50_data);
          tensorforge::fmacdpp16<9>(v62_acc, v80_bc, v51_data);
          tensorforge::fmacdpp16<10>(v62_acc, v80_bc, v52_data);
          tensorforge::fmacdpp16<11>(v62_acc, v80_bc, v53_data);
          tensorforge::fmacdpp16<12>(v63_acc, v80_bc, v42_data);
          tensorforge::fmacdpp16<13>(v63_acc, v80_bc, v43_data);
          tensorforge::fmacdpp16<14>(v63_acc, v80_bc, v44_data);
          tensorforge::fmacdpp16<15>(v63_acc, v80_bc, v45_data);
          float v81_bc = tensorforge::broadcast<32, 16, 1>(v79_lin);
          tensorforge::fmacdpp16<0>(v63_acc, v81_bc, v46_data);
          tensorforge::fmacdpp16<1>(v63_acc, v81_bc, v47_data);
          tensorforge::fmacdpp16<2>(v63_acc, v81_bc, v48_data);
          tensorforge::fmacdpp16<3>(v63_acc, v81_bc, v49_data);
          tensorforge::fmacdpp16<4>(v63_acc, v81_bc, v50_data);
          tensorforge::fmacdpp16<5>(v63_acc, v81_bc, v51_data);
          tensorforge::fmacdpp16<6>(v63_acc, v81_bc, v52_data);
          tensorforge::fmacdpp16<7>(v63_acc, v81_bc, v53_data);
          tensorforge::fmacdpp16<8>(v64_acc, v81_bc, v42_data);
          tensorforge::fmacdpp16<9>(v64_acc, v81_bc, v43_data);
          tensorforge::fmacdpp16<10>(v64_acc, v81_bc, v44_data);
          tensorforge::fmacdpp16<11>(v64_acc, v81_bc, v45_data);
          tensorforge::fmacdpp16<12>(v64_acc, v81_bc, v46_data);
          tensorforge::fmacdpp16<13>(v64_acc, v81_bc, v47_data);
          tensorforge::fmacdpp16<14>(v64_acc, v81_bc, v48_data);
          tensorforge::fmacdpp16<15>(v64_acc, v81_bc, v49_data);
          float v82_lin = r1[4];
          float v83_bc = tensorforge::broadcast<32, 16, 0>(v82_lin);
          tensorforge::fmacdpp16<0>(v64_acc, v83_bc, v50_data);
          tensorforge::fmacdpp16<1>(v64_acc, v83_bc, v51_data);
          tensorforge::fmacdpp16<2>(v64_acc, v83_bc, v52_data);
          tensorforge::fmacdpp16<3>(v64_acc, v83_bc, v53_data);
          tensorforge::fmacdpp16<4>(v65_acc, v83_bc, v42_data);
          tensorforge::fmacdpp16<5>(v65_acc, v83_bc, v43_data);
          tensorforge::fmacdpp16<6>(v65_acc, v83_bc, v44_data);
          tensorforge::fmacdpp16<7>(v65_acc, v83_bc, v45_data);
          tensorforge::fmacdpp16<8>(v65_acc, v83_bc, v46_data);
          tensorforge::fmacdpp16<9>(v65_acc, v83_bc, v47_data);
          tensorforge::fmacdpp16<10>(v65_acc, v83_bc, v48_data);
          tensorforge::fmacdpp16<11>(v65_acc, v83_bc, v49_data);
          tensorforge::fmacdpp16<12>(v65_acc, v83_bc, v50_data);
          tensorforge::fmacdpp16<13>(v65_acc, v83_bc, v51_data);
          tensorforge::fmacdpp16<14>(v65_acc, v83_bc, v52_data);
          tensorforge::fmacdpp16<15>(v65_acc, v83_bc, v53_data);
          float v84_bc = tensorforge::broadcast<32, 16, 1>(v82_lin);
          tensorforge::fmacdpp16<0>(v66_acc, v84_bc, v42_data);
          tensorforge::fmacdpp16<1>(v66_acc, v84_bc, v43_data);
          tensorforge::fmacdpp16<2>(v66_acc, v84_bc, v44_data);
          tensorforge::fmacdpp16<3>(v66_acc, v84_bc, v45_data);
          tensorforge::fmacdpp16<4>(v66_acc, v84_bc, v46_data);
          tensorforge::fmacdpp16<5>(v66_acc, v84_bc, v47_data);
          tensorforge::fmacdpp16<6>(v66_acc, v84_bc, v48_data);
          tensorforge::fmacdpp16<7>(v66_acc, v84_bc, v49_data);
          tensorforge::fmacdpp16<8>(v66_acc, v84_bc, v50_data);
          tensorforge::fmacdpp16<9>(v66_acc, v84_bc, v51_data);
          tensorforge::fmacdpp16<10>(v66_acc, v84_bc, v52_data);
          tensorforge::fmacdpp16<11>(v66_acc, v84_bc, v53_data);
          tensorforge::fmacdpp16<12>(v67_acc, v84_bc, v42_data);
          tensorforge::fmacdpp16<13>(v67_acc, v84_bc, v43_data);
          tensorforge::fmacdpp16<14>(v67_acc, v84_bc, v44_data);
          tensorforge::fmacdpp16<15>(v67_acc, v84_bc, v45_data);
          float v85_lin = r1[5];
          float v86_bc = tensorforge::broadcast<32, 16, 0>(v85_lin);
          tensorforge::fmacdpp16<0>(v67_acc, v86_bc, v46_data);
          tensorforge::fmacdpp16<1>(v67_acc, v86_bc, v47_data);
          tensorforge::fmacdpp16<2>(v67_acc, v86_bc, v48_data);
          tensorforge::fmacdpp16<3>(v67_acc, v86_bc, v49_data);
          tensorforge::fmacdpp16<4>(v67_acc, v86_bc, v50_data);
          tensorforge::fmacdpp16<5>(v67_acc, v86_bc, v51_data);
          tensorforge::fmacdpp16<6>(v67_acc, v86_bc, v52_data);
          tensorforge::fmacdpp16<7>(v67_acc, v86_bc, v53_data);
          tensorforge::fmacdpp16<8>(v68_acc, v86_bc, v42_data);
          tensorforge::fmacdpp16<9>(v68_acc, v86_bc, v43_data);
          tensorforge::fmacdpp16<10>(v68_acc, v86_bc, v44_data);
          tensorforge::fmacdpp16<11>(v68_acc, v86_bc, v45_data);
          tensorforge::fmacdpp16<12>(v68_acc, v86_bc, v46_data);
          tensorforge::fmacdpp16<13>(v68_acc, v86_bc, v47_data);
          tensorforge::fmacdpp16<14>(v68_acc, v86_bc, v48_data);
          tensorforge::fmacdpp16<15>(v68_acc, v86_bc, v49_data);
          float v87_bc = tensorforge::broadcast<32, 16, 1>(v85_lin);
          tensorforge::fmacdpp16<0>(v68_acc, v87_bc, v50_data);
          tensorforge::fmacdpp16<1>(v68_acc, v87_bc, v51_data);
          tensorforge::fmacdpp16<2>(v68_acc, v87_bc, v52_data);
          tensorforge::fmacdpp16<3>(v68_acc, v87_bc, v53_data);
          tensorforge::fmacdpp16<4>(v69_acc, v87_bc, v42_data);
          tensorforge::fmacdpp16<5>(v69_acc, v87_bc, v43_data);
          tensorforge::fmacdpp16<6>(v69_acc, v87_bc, v44_data);
          tensorforge::fmacdpp16<7>(v69_acc, v87_bc, v45_data);
          tensorforge::fmacdpp16<8>(v69_acc, v87_bc, v46_data);
          tensorforge::fmacdpp16<9>(v69_acc, v87_bc, v47_data);
          tensorforge::fmacdpp16<10>(v69_acc, v87_bc, v48_data);
          tensorforge::fmacdpp16<11>(v69_acc, v87_bc, v49_data);
          tensorforge::fmacdpp16<12>(v69_acc, v87_bc, v50_data);
          tensorforge::fmacdpp16<13>(v69_acc, v87_bc, v51_data);
          tensorforge::fmacdpp16<14>(v69_acc, v87_bc, v52_data);
          tensorforge::fmacdpp16<15>(v69_acc, v87_bc, v53_data);
          ir2[0] = v54_acc;
          ir2[1] = v55_acc;
          ir2[2] = v56_acc;
          ir2[3] = v57_acc;
          ir2[4] = v58_acc;
          ir2[5] = v59_acc;
          ir2[6] = v60_acc;
          ir2[7] = v61_acc;
          ir2[8] = v62_acc;
          ir2[9] = v63_acc;
          ir2[10] = v64_acc;
          ir2[11] = v65_acc;
          ir2[12] = v66_acc;
          ir2[13] = v67_acc;
          ir2[14] = v68_acc;
          ir2[15] = v69_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v91_i0 = 0; v91_i0 < 1; ++v91_i0) {
            int32_t v100_lead = v2_lead + (v91_i0 * 32);
            #pragma unroll
            for (int32_t v92_i1 = 0; v92_i1 < 16; ++v92_i1) {
              int32_t v93_a = v91_i0 + v92_i1;
              float v95_data = r2[(v91_i0 + v92_i1)];
              int32_t v102_a = v100_lead + (v92_i1 * 32);
              glb_m0[v102_a] = v95_data;
            }
          }
          float r4[8]{};
          {
            // r4 = load{g>r}(glb_m4);
            float v0 = glb_m4[0 + threadIdx.x * 1];
            r4[0] = v0;
            float v32 = glb_m4[32 + threadIdx.x * 1];
            r4[1] = v32;
            float v64 = glb_m4[64 + threadIdx.x * 1];
            r4[2] = v64;
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r6[12]{};
          // r6 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v106_i0 = 0; v106_i0 < 1; ++v106_i0) {
            int32_t v111_lead = v106_i0 * 32;
            int32_t v112_lead = v2_lead + v111_lead;
            int32_t v119_lead = v2_lead + v111_lead;
            #pragma unroll
            for (int32_t v107_i1 = 0; v107_i1 < 12; ++v107_i1) {
              int32_t v113_a = v107_i1 * 32;
              int32_t v114_a = v112_lead + v113_a;
              float v122_data = __builtin_nontemporal_load(&glb_m5[(v119_lead + v113_a)]);
              int32_t v123_a = v106_i0 + v107_i1;
              r6[v123_a] = v122_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[16]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          auto& ir5 = r5;
          float v124_data = r3[0];
          float v125_data = r3[1];
          float v126_data = r3[2];
          float v127_data = r3[3];
          float v128_data = r3[4];
          float v129_data = r3[5];
          float v130_data = r3[6];
          float v131_data = r3[7];
          float v132_data = r3[8];
          float v133_data = r3[9];
          float v134_data = r3[10];
          float v135_data = r3[11];
          float v136_acc{};
          float v137_acc{};
          float v138_acc{};
          float v139_acc{};
          float v140_acc{};
          float v141_acc{};
          float v142_acc{};
          float v143_acc{};
          float v144_lin = r4[0];
          float v145_bc = tensorforge::broadcast<32, 16, 0>(v144_lin);
          tensorforge::fmacdpp16<0>(v136_acc, v145_bc, v124_data);
          tensorforge::fmacdpp16<1>(v136_acc, v145_bc, v125_data);
          tensorforge::fmacdpp16<2>(v136_acc, v145_bc, v126_data);
          tensorforge::fmacdpp16<3>(v136_acc, v145_bc, v127_data);
          tensorforge::fmacdpp16<4>(v136_acc, v145_bc, v128_data);
          tensorforge::fmacdpp16<5>(v136_acc, v145_bc, v129_data);
          tensorforge::fmacdpp16<6>(v136_acc, v145_bc, v130_data);
          tensorforge::fmacdpp16<7>(v136_acc, v145_bc, v131_data);
          tensorforge::fmacdpp16<8>(v136_acc, v145_bc, v132_data);
          tensorforge::fmacdpp16<9>(v136_acc, v145_bc, v133_data);
          tensorforge::fmacdpp16<10>(v136_acc, v145_bc, v134_data);
          tensorforge::fmacdpp16<11>(v136_acc, v145_bc, v135_data);
          tensorforge::fmacdpp16<12>(v137_acc, v145_bc, v124_data);
          tensorforge::fmacdpp16<13>(v137_acc, v145_bc, v125_data);
          tensorforge::fmacdpp16<14>(v137_acc, v145_bc, v126_data);
          tensorforge::fmacdpp16<15>(v137_acc, v145_bc, v127_data);
          float v146_bc = tensorforge::broadcast<32, 16, 1>(v144_lin);
          tensorforge::fmacdpp16<0>(v137_acc, v146_bc, v128_data);
          tensorforge::fmacdpp16<1>(v137_acc, v146_bc, v129_data);
          tensorforge::fmacdpp16<2>(v137_acc, v146_bc, v130_data);
          tensorforge::fmacdpp16<3>(v137_acc, v146_bc, v131_data);
          tensorforge::fmacdpp16<4>(v137_acc, v146_bc, v132_data);
          tensorforge::fmacdpp16<5>(v137_acc, v146_bc, v133_data);
          tensorforge::fmacdpp16<6>(v137_acc, v146_bc, v134_data);
          tensorforge::fmacdpp16<7>(v137_acc, v146_bc, v135_data);
          tensorforge::fmacdpp16<8>(v138_acc, v146_bc, v124_data);
          tensorforge::fmacdpp16<9>(v138_acc, v146_bc, v125_data);
          tensorforge::fmacdpp16<10>(v138_acc, v146_bc, v126_data);
          tensorforge::fmacdpp16<11>(v138_acc, v146_bc, v127_data);
          tensorforge::fmacdpp16<12>(v138_acc, v146_bc, v128_data);
          tensorforge::fmacdpp16<13>(v138_acc, v146_bc, v129_data);
          tensorforge::fmacdpp16<14>(v138_acc, v146_bc, v130_data);
          tensorforge::fmacdpp16<15>(v138_acc, v146_bc, v131_data);
          float v147_lin = r4[1];
          float v148_bc = tensorforge::broadcast<32, 16, 0>(v147_lin);
          tensorforge::fmacdpp16<0>(v138_acc, v148_bc, v132_data);
          tensorforge::fmacdpp16<1>(v138_acc, v148_bc, v133_data);
          tensorforge::fmacdpp16<2>(v138_acc, v148_bc, v134_data);
          tensorforge::fmacdpp16<3>(v138_acc, v148_bc, v135_data);
          tensorforge::fmacdpp16<4>(v139_acc, v148_bc, v124_data);
          tensorforge::fmacdpp16<5>(v139_acc, v148_bc, v125_data);
          tensorforge::fmacdpp16<6>(v139_acc, v148_bc, v126_data);
          tensorforge::fmacdpp16<7>(v139_acc, v148_bc, v127_data);
          tensorforge::fmacdpp16<8>(v139_acc, v148_bc, v128_data);
          tensorforge::fmacdpp16<9>(v139_acc, v148_bc, v129_data);
          tensorforge::fmacdpp16<10>(v139_acc, v148_bc, v130_data);
          tensorforge::fmacdpp16<11>(v139_acc, v148_bc, v131_data);
          tensorforge::fmacdpp16<12>(v139_acc, v148_bc, v132_data);
          tensorforge::fmacdpp16<13>(v139_acc, v148_bc, v133_data);
          tensorforge::fmacdpp16<14>(v139_acc, v148_bc, v134_data);
          tensorforge::fmacdpp16<15>(v139_acc, v148_bc, v135_data);
          float v149_bc = tensorforge::broadcast<32, 16, 1>(v147_lin);
          tensorforge::fmacdpp16<0>(v140_acc, v149_bc, v124_data);
          tensorforge::fmacdpp16<1>(v140_acc, v149_bc, v125_data);
          tensorforge::fmacdpp16<2>(v140_acc, v149_bc, v126_data);
          tensorforge::fmacdpp16<3>(v140_acc, v149_bc, v127_data);
          tensorforge::fmacdpp16<4>(v140_acc, v149_bc, v128_data);
          tensorforge::fmacdpp16<5>(v140_acc, v149_bc, v129_data);
          tensorforge::fmacdpp16<6>(v140_acc, v149_bc, v130_data);
          tensorforge::fmacdpp16<7>(v140_acc, v149_bc, v131_data);
          tensorforge::fmacdpp16<8>(v140_acc, v149_bc, v132_data);
          tensorforge::fmacdpp16<9>(v140_acc, v149_bc, v133_data);
          tensorforge::fmacdpp16<10>(v140_acc, v149_bc, v134_data);
          tensorforge::fmacdpp16<11>(v140_acc, v149_bc, v135_data);
          tensorforge::fmacdpp16<12>(v141_acc, v149_bc, v124_data);
          tensorforge::fmacdpp16<13>(v141_acc, v149_bc, v125_data);
          tensorforge::fmacdpp16<14>(v141_acc, v149_bc, v126_data);
          tensorforge::fmacdpp16<15>(v141_acc, v149_bc, v127_data);
          float v150_lin = r4[2];
          float v151_bc = tensorforge::broadcast<32, 16, 0>(v150_lin);
          tensorforge::fmacdpp16<0>(v141_acc, v151_bc, v128_data);
          tensorforge::fmacdpp16<1>(v141_acc, v151_bc, v129_data);
          tensorforge::fmacdpp16<2>(v141_acc, v151_bc, v130_data);
          tensorforge::fmacdpp16<3>(v141_acc, v151_bc, v131_data);
          tensorforge::fmacdpp16<4>(v141_acc, v151_bc, v132_data);
          tensorforge::fmacdpp16<5>(v141_acc, v151_bc, v133_data);
          tensorforge::fmacdpp16<6>(v141_acc, v151_bc, v134_data);
          tensorforge::fmacdpp16<7>(v141_acc, v151_bc, v135_data);
          tensorforge::fmacdpp16<8>(v142_acc, v151_bc, v124_data);
          tensorforge::fmacdpp16<9>(v142_acc, v151_bc, v125_data);
          tensorforge::fmacdpp16<10>(v142_acc, v151_bc, v126_data);
          tensorforge::fmacdpp16<11>(v142_acc, v151_bc, v127_data);
          tensorforge::fmacdpp16<12>(v142_acc, v151_bc, v128_data);
          tensorforge::fmacdpp16<13>(v142_acc, v151_bc, v129_data);
          tensorforge::fmacdpp16<14>(v142_acc, v151_bc, v130_data);
          tensorforge::fmacdpp16<15>(v142_acc, v151_bc, v131_data);
          float v152_bc = tensorforge::broadcast<32, 16, 1>(v150_lin);
          tensorforge::fmacdpp16<0>(v142_acc, v152_bc, v132_data);
          tensorforge::fmacdpp16<1>(v142_acc, v152_bc, v133_data);
          tensorforge::fmacdpp16<2>(v142_acc, v152_bc, v134_data);
          tensorforge::fmacdpp16<3>(v142_acc, v152_bc, v135_data);
          tensorforge::fmacdpp16<4>(v143_acc, v152_bc, v124_data);
          tensorforge::fmacdpp16<5>(v143_acc, v152_bc, v125_data);
          tensorforge::fmacdpp16<6>(v143_acc, v152_bc, v126_data);
          tensorforge::fmacdpp16<7>(v143_acc, v152_bc, v127_data);
          tensorforge::fmacdpp16<8>(v143_acc, v152_bc, v128_data);
          tensorforge::fmacdpp16<9>(v143_acc, v152_bc, v129_data);
          tensorforge::fmacdpp16<10>(v143_acc, v152_bc, v130_data);
          tensorforge::fmacdpp16<11>(v143_acc, v152_bc, v131_data);
          tensorforge::fmacdpp16<12>(v143_acc, v152_bc, v132_data);
          tensorforge::fmacdpp16<13>(v143_acc, v152_bc, v133_data);
          tensorforge::fmacdpp16<14>(v143_acc, v152_bc, v134_data);
          tensorforge::fmacdpp16<15>(v143_acc, v152_bc, v135_data);
          ir5[0] = v136_acc;
          ir5[1] = v137_acc;
          ir5[2] = v138_acc;
          ir5[3] = v139_acc;
          ir5[4] = v140_acc;
          ir5[5] = v141_acc;
          ir5[6] = v142_acc;
          ir5[7] = v143_acc;
          // glb_m0 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v156_i0 = 0; v156_i0 < 1; ++v156_i0) {
            int32_t v165_lead = v2_lead + (v156_i0 * 32);
            #pragma unroll
            for (int32_t v157_i1 = 0; v157_i1 < 8; ++v157_i1) {
              int32_t v158_a = v156_i0 + v157_i1;
              float v160_data = r5[(v156_i0 + v157_i1)];
              int32_t v167_a = v165_lead + (v157_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v167_a], v160_data);
            }
          }
          float r7[8]{};
          {
            // r7 = load{g>r}(glb_m6);
            float v0 = glb_m6[0 + threadIdx.x * 1];
            r7[0] = v0;
            float v32 = glb_m6[32 + threadIdx.x * 1];
            r7[1] = v32;
            float v64 = glb_m6[64 + threadIdx.x * 1];
            r7[2] = v64;
          }
          // wait(r6 = load{g>r}(glb_m5););
          // wait(r7 = load{g>r}(glb_m6););
          float r8[16]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          auto& ir8 = r8;
          float v168_data = r6[0];
          float v169_data = r6[1];
          float v170_data = r6[2];
          float v171_data = r6[3];
          float v172_data = r6[4];
          float v173_data = r6[5];
          float v174_data = r6[6];
          float v175_data = r6[7];
          float v176_data = r6[8];
          float v177_data = r6[9];
          float v178_data = r6[10];
          float v179_data = r6[11];
          float v180_acc{};
          float v181_acc{};
          float v182_acc{};
          float v183_acc{};
          float v184_acc{};
          float v185_acc{};
          float v186_acc{};
          float v187_acc{};
          float v188_lin = r7[0];
          float v189_bc = tensorforge::broadcast<32, 16, 0>(v188_lin);
          tensorforge::fmacdpp16<0>(v180_acc, v189_bc, v168_data);
          tensorforge::fmacdpp16<1>(v180_acc, v189_bc, v169_data);
          tensorforge::fmacdpp16<2>(v180_acc, v189_bc, v170_data);
          tensorforge::fmacdpp16<3>(v180_acc, v189_bc, v171_data);
          tensorforge::fmacdpp16<4>(v180_acc, v189_bc, v172_data);
          tensorforge::fmacdpp16<5>(v180_acc, v189_bc, v173_data);
          tensorforge::fmacdpp16<6>(v180_acc, v189_bc, v174_data);
          tensorforge::fmacdpp16<7>(v180_acc, v189_bc, v175_data);
          tensorforge::fmacdpp16<8>(v180_acc, v189_bc, v176_data);
          tensorforge::fmacdpp16<9>(v180_acc, v189_bc, v177_data);
          tensorforge::fmacdpp16<10>(v180_acc, v189_bc, v178_data);
          tensorforge::fmacdpp16<11>(v180_acc, v189_bc, v179_data);
          tensorforge::fmacdpp16<12>(v181_acc, v189_bc, v168_data);
          tensorforge::fmacdpp16<13>(v181_acc, v189_bc, v169_data);
          tensorforge::fmacdpp16<14>(v181_acc, v189_bc, v170_data);
          tensorforge::fmacdpp16<15>(v181_acc, v189_bc, v171_data);
          float v190_bc = tensorforge::broadcast<32, 16, 1>(v188_lin);
          tensorforge::fmacdpp16<0>(v181_acc, v190_bc, v172_data);
          tensorforge::fmacdpp16<1>(v181_acc, v190_bc, v173_data);
          tensorforge::fmacdpp16<2>(v181_acc, v190_bc, v174_data);
          tensorforge::fmacdpp16<3>(v181_acc, v190_bc, v175_data);
          tensorforge::fmacdpp16<4>(v181_acc, v190_bc, v176_data);
          tensorforge::fmacdpp16<5>(v181_acc, v190_bc, v177_data);
          tensorforge::fmacdpp16<6>(v181_acc, v190_bc, v178_data);
          tensorforge::fmacdpp16<7>(v181_acc, v190_bc, v179_data);
          tensorforge::fmacdpp16<8>(v182_acc, v190_bc, v168_data);
          tensorforge::fmacdpp16<9>(v182_acc, v190_bc, v169_data);
          tensorforge::fmacdpp16<10>(v182_acc, v190_bc, v170_data);
          tensorforge::fmacdpp16<11>(v182_acc, v190_bc, v171_data);
          tensorforge::fmacdpp16<12>(v182_acc, v190_bc, v172_data);
          tensorforge::fmacdpp16<13>(v182_acc, v190_bc, v173_data);
          tensorforge::fmacdpp16<14>(v182_acc, v190_bc, v174_data);
          tensorforge::fmacdpp16<15>(v182_acc, v190_bc, v175_data);
          float v191_lin = r7[1];
          float v192_bc = tensorforge::broadcast<32, 16, 0>(v191_lin);
          tensorforge::fmacdpp16<0>(v182_acc, v192_bc, v176_data);
          tensorforge::fmacdpp16<1>(v182_acc, v192_bc, v177_data);
          tensorforge::fmacdpp16<2>(v182_acc, v192_bc, v178_data);
          tensorforge::fmacdpp16<3>(v182_acc, v192_bc, v179_data);
          tensorforge::fmacdpp16<4>(v183_acc, v192_bc, v168_data);
          tensorforge::fmacdpp16<5>(v183_acc, v192_bc, v169_data);
          tensorforge::fmacdpp16<6>(v183_acc, v192_bc, v170_data);
          tensorforge::fmacdpp16<7>(v183_acc, v192_bc, v171_data);
          tensorforge::fmacdpp16<8>(v183_acc, v192_bc, v172_data);
          tensorforge::fmacdpp16<9>(v183_acc, v192_bc, v173_data);
          tensorforge::fmacdpp16<10>(v183_acc, v192_bc, v174_data);
          tensorforge::fmacdpp16<11>(v183_acc, v192_bc, v175_data);
          tensorforge::fmacdpp16<12>(v183_acc, v192_bc, v176_data);
          tensorforge::fmacdpp16<13>(v183_acc, v192_bc, v177_data);
          tensorforge::fmacdpp16<14>(v183_acc, v192_bc, v178_data);
          tensorforge::fmacdpp16<15>(v183_acc, v192_bc, v179_data);
          float v193_bc = tensorforge::broadcast<32, 16, 1>(v191_lin);
          tensorforge::fmacdpp16<0>(v184_acc, v193_bc, v168_data);
          tensorforge::fmacdpp16<1>(v184_acc, v193_bc, v169_data);
          tensorforge::fmacdpp16<2>(v184_acc, v193_bc, v170_data);
          tensorforge::fmacdpp16<3>(v184_acc, v193_bc, v171_data);
          tensorforge::fmacdpp16<4>(v184_acc, v193_bc, v172_data);
          tensorforge::fmacdpp16<5>(v184_acc, v193_bc, v173_data);
          tensorforge::fmacdpp16<6>(v184_acc, v193_bc, v174_data);
          tensorforge::fmacdpp16<7>(v184_acc, v193_bc, v175_data);
          tensorforge::fmacdpp16<8>(v184_acc, v193_bc, v176_data);
          tensorforge::fmacdpp16<9>(v184_acc, v193_bc, v177_data);
          tensorforge::fmacdpp16<10>(v184_acc, v193_bc, v178_data);
          tensorforge::fmacdpp16<11>(v184_acc, v193_bc, v179_data);
          tensorforge::fmacdpp16<12>(v185_acc, v193_bc, v168_data);
          tensorforge::fmacdpp16<13>(v185_acc, v193_bc, v169_data);
          tensorforge::fmacdpp16<14>(v185_acc, v193_bc, v170_data);
          tensorforge::fmacdpp16<15>(v185_acc, v193_bc, v171_data);
          float v194_lin = r7[2];
          float v195_bc = tensorforge::broadcast<32, 16, 0>(v194_lin);
          tensorforge::fmacdpp16<0>(v185_acc, v195_bc, v172_data);
          tensorforge::fmacdpp16<1>(v185_acc, v195_bc, v173_data);
          tensorforge::fmacdpp16<2>(v185_acc, v195_bc, v174_data);
          tensorforge::fmacdpp16<3>(v185_acc, v195_bc, v175_data);
          tensorforge::fmacdpp16<4>(v185_acc, v195_bc, v176_data);
          tensorforge::fmacdpp16<5>(v185_acc, v195_bc, v177_data);
          tensorforge::fmacdpp16<6>(v185_acc, v195_bc, v178_data);
          tensorforge::fmacdpp16<7>(v185_acc, v195_bc, v179_data);
          tensorforge::fmacdpp16<8>(v186_acc, v195_bc, v168_data);
          tensorforge::fmacdpp16<9>(v186_acc, v195_bc, v169_data);
          tensorforge::fmacdpp16<10>(v186_acc, v195_bc, v170_data);
          tensorforge::fmacdpp16<11>(v186_acc, v195_bc, v171_data);
          tensorforge::fmacdpp16<12>(v186_acc, v195_bc, v172_data);
          tensorforge::fmacdpp16<13>(v186_acc, v195_bc, v173_data);
          tensorforge::fmacdpp16<14>(v186_acc, v195_bc, v174_data);
          tensorforge::fmacdpp16<15>(v186_acc, v195_bc, v175_data);
          float v196_bc = tensorforge::broadcast<32, 16, 1>(v194_lin);
          tensorforge::fmacdpp16<0>(v186_acc, v196_bc, v176_data);
          tensorforge::fmacdpp16<1>(v186_acc, v196_bc, v177_data);
          tensorforge::fmacdpp16<2>(v186_acc, v196_bc, v178_data);
          tensorforge::fmacdpp16<3>(v186_acc, v196_bc, v179_data);
          tensorforge::fmacdpp16<4>(v187_acc, v196_bc, v168_data);
          tensorforge::fmacdpp16<5>(v187_acc, v196_bc, v169_data);
          tensorforge::fmacdpp16<6>(v187_acc, v196_bc, v170_data);
          tensorforge::fmacdpp16<7>(v187_acc, v196_bc, v171_data);
          tensorforge::fmacdpp16<8>(v187_acc, v196_bc, v172_data);
          tensorforge::fmacdpp16<9>(v187_acc, v196_bc, v173_data);
          tensorforge::fmacdpp16<10>(v187_acc, v196_bc, v174_data);
          tensorforge::fmacdpp16<11>(v187_acc, v196_bc, v175_data);
          tensorforge::fmacdpp16<12>(v187_acc, v196_bc, v176_data);
          tensorforge::fmacdpp16<13>(v187_acc, v196_bc, v177_data);
          tensorforge::fmacdpp16<14>(v187_acc, v196_bc, v178_data);
          tensorforge::fmacdpp16<15>(v187_acc, v196_bc, v179_data);
          ir8[0] = v180_acc;
          ir8[1] = v181_acc;
          ir8[2] = v182_acc;
          ir8[3] = v183_acc;
          ir8[4] = v184_acc;
          ir8[5] = v185_acc;
          ir8[6] = v186_acc;
          ir8[7] = v187_acc;
          // glb_m0 = store{r>g}(r8);
          #pragma unroll
          for (int32_t v200_i0 = 0; v200_i0 < 1; ++v200_i0) {
            int32_t v209_lead = v2_lead + (v200_i0 * 32);
            #pragma unroll
            for (int32_t v201_i1 = 0; v201_i1 < 8; ++v201_i1) {
              int32_t v202_a = v200_i0 + v201_i1;
              float v204_data = r8[(v200_i0 + v201_i1)];
              int32_t v212_a = v209_lead + ((v201_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v212_a], v204_data);
            }
          }
          ;
        }
      }
    }
  }
}

