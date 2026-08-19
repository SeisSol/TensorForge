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
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 12; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
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
          int32_t v16_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
            int32_t v23_lead = v16_lead + (v17_i0 * 32);
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              int32_t v25_a = v23_lead + (v18_i1 * 32);
              float v26_data;
              {
                v26_data = __builtin_nontemporal_load(&glb_m3[v25_a]);
              }
              int32_t v27_a = v17_i0 + v18_i1;
              r3[v27_a] = v26_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          auto& ir2 = r2;
          float v28_data = r0[0];
          float v29_data = r0[1];
          float v30_data = r0[2];
          float v31_data = r0[3];
          float v32_data = r0[4];
          float v33_data = r0[5];
          float v34_data = r0[6];
          float v35_data = r0[7];
          float v36_data = r0[8];
          float v37_data = r0[9];
          float v38_data = r0[10];
          float v39_data = r0[11];
          float v40_acc{};
          float v41_acc{};
          float v42_acc{};
          float v43_acc{};
          float v44_acc{};
          float v45_acc{};
          float v46_acc{};
          float v47_acc{};
          float v48_acc{};
          float v49_acc{};
          float v50_acc{};
          float v51_acc{};
          float v52_acc{};
          float v53_acc{};
          float v54_acc{};
          float v55_acc{};
          float v56_lin = r1[0];
          float v57_bc = tensorforge::broadcast<32, 16, 0>(v56_lin);
          tensorforge::fmacdpp16<0>(v40_acc, v57_bc, v28_data);
          tensorforge::fmacdpp16<1>(v40_acc, v57_bc, v29_data);
          tensorforge::fmacdpp16<2>(v40_acc, v57_bc, v30_data);
          tensorforge::fmacdpp16<3>(v40_acc, v57_bc, v31_data);
          tensorforge::fmacdpp16<4>(v40_acc, v57_bc, v32_data);
          tensorforge::fmacdpp16<5>(v40_acc, v57_bc, v33_data);
          tensorforge::fmacdpp16<6>(v40_acc, v57_bc, v34_data);
          tensorforge::fmacdpp16<7>(v40_acc, v57_bc, v35_data);
          tensorforge::fmacdpp16<8>(v40_acc, v57_bc, v36_data);
          tensorforge::fmacdpp16<9>(v40_acc, v57_bc, v37_data);
          tensorforge::fmacdpp16<10>(v40_acc, v57_bc, v38_data);
          tensorforge::fmacdpp16<11>(v40_acc, v57_bc, v39_data);
          tensorforge::fmacdpp16<12>(v41_acc, v57_bc, v28_data);
          tensorforge::fmacdpp16<13>(v41_acc, v57_bc, v29_data);
          tensorforge::fmacdpp16<14>(v41_acc, v57_bc, v30_data);
          tensorforge::fmacdpp16<15>(v41_acc, v57_bc, v31_data);
          float v58_bc = tensorforge::broadcast<32, 16, 1>(v56_lin);
          tensorforge::fmacdpp16<0>(v41_acc, v58_bc, v32_data);
          tensorforge::fmacdpp16<1>(v41_acc, v58_bc, v33_data);
          tensorforge::fmacdpp16<2>(v41_acc, v58_bc, v34_data);
          tensorforge::fmacdpp16<3>(v41_acc, v58_bc, v35_data);
          tensorforge::fmacdpp16<4>(v41_acc, v58_bc, v36_data);
          tensorforge::fmacdpp16<5>(v41_acc, v58_bc, v37_data);
          tensorforge::fmacdpp16<6>(v41_acc, v58_bc, v38_data);
          tensorforge::fmacdpp16<7>(v41_acc, v58_bc, v39_data);
          tensorforge::fmacdpp16<8>(v42_acc, v58_bc, v28_data);
          tensorforge::fmacdpp16<9>(v42_acc, v58_bc, v29_data);
          tensorforge::fmacdpp16<10>(v42_acc, v58_bc, v30_data);
          tensorforge::fmacdpp16<11>(v42_acc, v58_bc, v31_data);
          tensorforge::fmacdpp16<12>(v42_acc, v58_bc, v32_data);
          tensorforge::fmacdpp16<13>(v42_acc, v58_bc, v33_data);
          tensorforge::fmacdpp16<14>(v42_acc, v58_bc, v34_data);
          tensorforge::fmacdpp16<15>(v42_acc, v58_bc, v35_data);
          float v59_lin = r1[1];
          float v60_bc = tensorforge::broadcast<32, 16, 0>(v59_lin);
          tensorforge::fmacdpp16<0>(v42_acc, v60_bc, v36_data);
          tensorforge::fmacdpp16<1>(v42_acc, v60_bc, v37_data);
          tensorforge::fmacdpp16<2>(v42_acc, v60_bc, v38_data);
          tensorforge::fmacdpp16<3>(v42_acc, v60_bc, v39_data);
          tensorforge::fmacdpp16<4>(v43_acc, v60_bc, v28_data);
          tensorforge::fmacdpp16<5>(v43_acc, v60_bc, v29_data);
          tensorforge::fmacdpp16<6>(v43_acc, v60_bc, v30_data);
          tensorforge::fmacdpp16<7>(v43_acc, v60_bc, v31_data);
          tensorforge::fmacdpp16<8>(v43_acc, v60_bc, v32_data);
          tensorforge::fmacdpp16<9>(v43_acc, v60_bc, v33_data);
          tensorforge::fmacdpp16<10>(v43_acc, v60_bc, v34_data);
          tensorforge::fmacdpp16<11>(v43_acc, v60_bc, v35_data);
          tensorforge::fmacdpp16<12>(v43_acc, v60_bc, v36_data);
          tensorforge::fmacdpp16<13>(v43_acc, v60_bc, v37_data);
          tensorforge::fmacdpp16<14>(v43_acc, v60_bc, v38_data);
          tensorforge::fmacdpp16<15>(v43_acc, v60_bc, v39_data);
          float v61_bc = tensorforge::broadcast<32, 16, 1>(v59_lin);
          tensorforge::fmacdpp16<0>(v44_acc, v61_bc, v28_data);
          tensorforge::fmacdpp16<1>(v44_acc, v61_bc, v29_data);
          tensorforge::fmacdpp16<2>(v44_acc, v61_bc, v30_data);
          tensorforge::fmacdpp16<3>(v44_acc, v61_bc, v31_data);
          tensorforge::fmacdpp16<4>(v44_acc, v61_bc, v32_data);
          tensorforge::fmacdpp16<5>(v44_acc, v61_bc, v33_data);
          tensorforge::fmacdpp16<6>(v44_acc, v61_bc, v34_data);
          tensorforge::fmacdpp16<7>(v44_acc, v61_bc, v35_data);
          tensorforge::fmacdpp16<8>(v44_acc, v61_bc, v36_data);
          tensorforge::fmacdpp16<9>(v44_acc, v61_bc, v37_data);
          tensorforge::fmacdpp16<10>(v44_acc, v61_bc, v38_data);
          tensorforge::fmacdpp16<11>(v44_acc, v61_bc, v39_data);
          tensorforge::fmacdpp16<12>(v45_acc, v61_bc, v28_data);
          tensorforge::fmacdpp16<13>(v45_acc, v61_bc, v29_data);
          tensorforge::fmacdpp16<14>(v45_acc, v61_bc, v30_data);
          tensorforge::fmacdpp16<15>(v45_acc, v61_bc, v31_data);
          float v62_lin = r1[2];
          float v63_bc = tensorforge::broadcast<32, 16, 0>(v62_lin);
          tensorforge::fmacdpp16<0>(v45_acc, v63_bc, v32_data);
          tensorforge::fmacdpp16<1>(v45_acc, v63_bc, v33_data);
          tensorforge::fmacdpp16<2>(v45_acc, v63_bc, v34_data);
          tensorforge::fmacdpp16<3>(v45_acc, v63_bc, v35_data);
          tensorforge::fmacdpp16<4>(v45_acc, v63_bc, v36_data);
          tensorforge::fmacdpp16<5>(v45_acc, v63_bc, v37_data);
          tensorforge::fmacdpp16<6>(v45_acc, v63_bc, v38_data);
          tensorforge::fmacdpp16<7>(v45_acc, v63_bc, v39_data);
          tensorforge::fmacdpp16<8>(v46_acc, v63_bc, v28_data);
          tensorforge::fmacdpp16<9>(v46_acc, v63_bc, v29_data);
          tensorforge::fmacdpp16<10>(v46_acc, v63_bc, v30_data);
          tensorforge::fmacdpp16<11>(v46_acc, v63_bc, v31_data);
          tensorforge::fmacdpp16<12>(v46_acc, v63_bc, v32_data);
          tensorforge::fmacdpp16<13>(v46_acc, v63_bc, v33_data);
          tensorforge::fmacdpp16<14>(v46_acc, v63_bc, v34_data);
          tensorforge::fmacdpp16<15>(v46_acc, v63_bc, v35_data);
          float v64_bc = tensorforge::broadcast<32, 16, 1>(v62_lin);
          tensorforge::fmacdpp16<0>(v46_acc, v64_bc, v36_data);
          tensorforge::fmacdpp16<1>(v46_acc, v64_bc, v37_data);
          tensorforge::fmacdpp16<2>(v46_acc, v64_bc, v38_data);
          tensorforge::fmacdpp16<3>(v46_acc, v64_bc, v39_data);
          tensorforge::fmacdpp16<4>(v47_acc, v64_bc, v28_data);
          tensorforge::fmacdpp16<5>(v47_acc, v64_bc, v29_data);
          tensorforge::fmacdpp16<6>(v47_acc, v64_bc, v30_data);
          tensorforge::fmacdpp16<7>(v47_acc, v64_bc, v31_data);
          tensorforge::fmacdpp16<8>(v47_acc, v64_bc, v32_data);
          tensorforge::fmacdpp16<9>(v47_acc, v64_bc, v33_data);
          tensorforge::fmacdpp16<10>(v47_acc, v64_bc, v34_data);
          tensorforge::fmacdpp16<11>(v47_acc, v64_bc, v35_data);
          tensorforge::fmacdpp16<12>(v47_acc, v64_bc, v36_data);
          tensorforge::fmacdpp16<13>(v47_acc, v64_bc, v37_data);
          tensorforge::fmacdpp16<14>(v47_acc, v64_bc, v38_data);
          tensorforge::fmacdpp16<15>(v47_acc, v64_bc, v39_data);
          float v65_lin = r1[3];
          float v66_bc = tensorforge::broadcast<32, 16, 0>(v65_lin);
          tensorforge::fmacdpp16<0>(v48_acc, v66_bc, v28_data);
          tensorforge::fmacdpp16<1>(v48_acc, v66_bc, v29_data);
          tensorforge::fmacdpp16<2>(v48_acc, v66_bc, v30_data);
          tensorforge::fmacdpp16<3>(v48_acc, v66_bc, v31_data);
          tensorforge::fmacdpp16<4>(v48_acc, v66_bc, v32_data);
          tensorforge::fmacdpp16<5>(v48_acc, v66_bc, v33_data);
          tensorforge::fmacdpp16<6>(v48_acc, v66_bc, v34_data);
          tensorforge::fmacdpp16<7>(v48_acc, v66_bc, v35_data);
          tensorforge::fmacdpp16<8>(v48_acc, v66_bc, v36_data);
          tensorforge::fmacdpp16<9>(v48_acc, v66_bc, v37_data);
          tensorforge::fmacdpp16<10>(v48_acc, v66_bc, v38_data);
          tensorforge::fmacdpp16<11>(v48_acc, v66_bc, v39_data);
          tensorforge::fmacdpp16<12>(v49_acc, v66_bc, v28_data);
          tensorforge::fmacdpp16<13>(v49_acc, v66_bc, v29_data);
          tensorforge::fmacdpp16<14>(v49_acc, v66_bc, v30_data);
          tensorforge::fmacdpp16<15>(v49_acc, v66_bc, v31_data);
          float v67_bc = tensorforge::broadcast<32, 16, 1>(v65_lin);
          tensorforge::fmacdpp16<0>(v49_acc, v67_bc, v32_data);
          tensorforge::fmacdpp16<1>(v49_acc, v67_bc, v33_data);
          tensorforge::fmacdpp16<2>(v49_acc, v67_bc, v34_data);
          tensorforge::fmacdpp16<3>(v49_acc, v67_bc, v35_data);
          tensorforge::fmacdpp16<4>(v49_acc, v67_bc, v36_data);
          tensorforge::fmacdpp16<5>(v49_acc, v67_bc, v37_data);
          tensorforge::fmacdpp16<6>(v49_acc, v67_bc, v38_data);
          tensorforge::fmacdpp16<7>(v49_acc, v67_bc, v39_data);
          tensorforge::fmacdpp16<8>(v50_acc, v67_bc, v28_data);
          tensorforge::fmacdpp16<9>(v50_acc, v67_bc, v29_data);
          tensorforge::fmacdpp16<10>(v50_acc, v67_bc, v30_data);
          tensorforge::fmacdpp16<11>(v50_acc, v67_bc, v31_data);
          tensorforge::fmacdpp16<12>(v50_acc, v67_bc, v32_data);
          tensorforge::fmacdpp16<13>(v50_acc, v67_bc, v33_data);
          tensorforge::fmacdpp16<14>(v50_acc, v67_bc, v34_data);
          tensorforge::fmacdpp16<15>(v50_acc, v67_bc, v35_data);
          float v68_lin = r1[4];
          float v69_bc = tensorforge::broadcast<32, 16, 0>(v68_lin);
          tensorforge::fmacdpp16<0>(v50_acc, v69_bc, v36_data);
          tensorforge::fmacdpp16<1>(v50_acc, v69_bc, v37_data);
          tensorforge::fmacdpp16<2>(v50_acc, v69_bc, v38_data);
          tensorforge::fmacdpp16<3>(v50_acc, v69_bc, v39_data);
          tensorforge::fmacdpp16<4>(v51_acc, v69_bc, v28_data);
          tensorforge::fmacdpp16<5>(v51_acc, v69_bc, v29_data);
          tensorforge::fmacdpp16<6>(v51_acc, v69_bc, v30_data);
          tensorforge::fmacdpp16<7>(v51_acc, v69_bc, v31_data);
          tensorforge::fmacdpp16<8>(v51_acc, v69_bc, v32_data);
          tensorforge::fmacdpp16<9>(v51_acc, v69_bc, v33_data);
          tensorforge::fmacdpp16<10>(v51_acc, v69_bc, v34_data);
          tensorforge::fmacdpp16<11>(v51_acc, v69_bc, v35_data);
          tensorforge::fmacdpp16<12>(v51_acc, v69_bc, v36_data);
          tensorforge::fmacdpp16<13>(v51_acc, v69_bc, v37_data);
          tensorforge::fmacdpp16<14>(v51_acc, v69_bc, v38_data);
          tensorforge::fmacdpp16<15>(v51_acc, v69_bc, v39_data);
          float v70_bc = tensorforge::broadcast<32, 16, 1>(v68_lin);
          tensorforge::fmacdpp16<0>(v52_acc, v70_bc, v28_data);
          tensorforge::fmacdpp16<1>(v52_acc, v70_bc, v29_data);
          tensorforge::fmacdpp16<2>(v52_acc, v70_bc, v30_data);
          tensorforge::fmacdpp16<3>(v52_acc, v70_bc, v31_data);
          tensorforge::fmacdpp16<4>(v52_acc, v70_bc, v32_data);
          tensorforge::fmacdpp16<5>(v52_acc, v70_bc, v33_data);
          tensorforge::fmacdpp16<6>(v52_acc, v70_bc, v34_data);
          tensorforge::fmacdpp16<7>(v52_acc, v70_bc, v35_data);
          tensorforge::fmacdpp16<8>(v52_acc, v70_bc, v36_data);
          tensorforge::fmacdpp16<9>(v52_acc, v70_bc, v37_data);
          tensorforge::fmacdpp16<10>(v52_acc, v70_bc, v38_data);
          tensorforge::fmacdpp16<11>(v52_acc, v70_bc, v39_data);
          tensorforge::fmacdpp16<12>(v53_acc, v70_bc, v28_data);
          tensorforge::fmacdpp16<13>(v53_acc, v70_bc, v29_data);
          tensorforge::fmacdpp16<14>(v53_acc, v70_bc, v30_data);
          tensorforge::fmacdpp16<15>(v53_acc, v70_bc, v31_data);
          float v71_lin = r1[5];
          float v72_bc = tensorforge::broadcast<32, 16, 0>(v71_lin);
          tensorforge::fmacdpp16<0>(v53_acc, v72_bc, v32_data);
          tensorforge::fmacdpp16<1>(v53_acc, v72_bc, v33_data);
          tensorforge::fmacdpp16<2>(v53_acc, v72_bc, v34_data);
          tensorforge::fmacdpp16<3>(v53_acc, v72_bc, v35_data);
          tensorforge::fmacdpp16<4>(v53_acc, v72_bc, v36_data);
          tensorforge::fmacdpp16<5>(v53_acc, v72_bc, v37_data);
          tensorforge::fmacdpp16<6>(v53_acc, v72_bc, v38_data);
          tensorforge::fmacdpp16<7>(v53_acc, v72_bc, v39_data);
          tensorforge::fmacdpp16<8>(v54_acc, v72_bc, v28_data);
          tensorforge::fmacdpp16<9>(v54_acc, v72_bc, v29_data);
          tensorforge::fmacdpp16<10>(v54_acc, v72_bc, v30_data);
          tensorforge::fmacdpp16<11>(v54_acc, v72_bc, v31_data);
          tensorforge::fmacdpp16<12>(v54_acc, v72_bc, v32_data);
          tensorforge::fmacdpp16<13>(v54_acc, v72_bc, v33_data);
          tensorforge::fmacdpp16<14>(v54_acc, v72_bc, v34_data);
          tensorforge::fmacdpp16<15>(v54_acc, v72_bc, v35_data);
          float v73_bc = tensorforge::broadcast<32, 16, 1>(v71_lin);
          tensorforge::fmacdpp16<0>(v54_acc, v73_bc, v36_data);
          tensorforge::fmacdpp16<1>(v54_acc, v73_bc, v37_data);
          tensorforge::fmacdpp16<2>(v54_acc, v73_bc, v38_data);
          tensorforge::fmacdpp16<3>(v54_acc, v73_bc, v39_data);
          tensorforge::fmacdpp16<4>(v55_acc, v73_bc, v28_data);
          tensorforge::fmacdpp16<5>(v55_acc, v73_bc, v29_data);
          tensorforge::fmacdpp16<6>(v55_acc, v73_bc, v30_data);
          tensorforge::fmacdpp16<7>(v55_acc, v73_bc, v31_data);
          tensorforge::fmacdpp16<8>(v55_acc, v73_bc, v32_data);
          tensorforge::fmacdpp16<9>(v55_acc, v73_bc, v33_data);
          tensorforge::fmacdpp16<10>(v55_acc, v73_bc, v34_data);
          tensorforge::fmacdpp16<11>(v55_acc, v73_bc, v35_data);
          tensorforge::fmacdpp16<12>(v55_acc, v73_bc, v36_data);
          tensorforge::fmacdpp16<13>(v55_acc, v73_bc, v37_data);
          tensorforge::fmacdpp16<14>(v55_acc, v73_bc, v38_data);
          tensorforge::fmacdpp16<15>(v55_acc, v73_bc, v39_data);
          ir2[0] = v40_acc;
          ir2[1] = v41_acc;
          ir2[2] = v42_acc;
          ir2[3] = v43_acc;
          ir2[4] = v44_acc;
          ir2[5] = v45_acc;
          ir2[6] = v46_acc;
          ir2[7] = v47_acc;
          ir2[8] = v48_acc;
          ir2[9] = v49_acc;
          ir2[10] = v50_acc;
          ir2[11] = v51_acc;
          ir2[12] = v52_acc;
          ir2[13] = v53_acc;
          ir2[14] = v54_acc;
          ir2[15] = v55_acc;
          // glb_m0 = store{r>g}(r2);
          int32_t v76_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v77_i0 = 0; v77_i0 < 1; ++v77_i0) {
            int32_t v86_lead = v76_lead + (v77_i0 * 32);
            #pragma unroll
            for (int32_t v78_i1 = 0; v78_i1 < 16; ++v78_i1) {
              int32_t v79_a = v77_i0 + v78_i1;
              float v81_data = r2[(v77_i0 + v78_i1)];
              int32_t v88_a = v86_lead + (v78_i1 * 32);
              glb_m0[v88_a] = v81_data;
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
          int32_t v91_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v92_i0 = 0; v92_i0 < 1; ++v92_i0) {
            int32_t v98_lead = v91_lead + (v92_i0 * 32);
            #pragma unroll
            for (int32_t v93_i1 = 0; v93_i1 < 12; ++v93_i1) {
              int32_t v100_a = v98_lead + (v93_i1 * 32);
              float v101_data;
              {
                v101_data = __builtin_nontemporal_load(&glb_m5[v100_a]);
              }
              int32_t v102_a = v92_i0 + v93_i1;
              r6[v102_a] = v101_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[16]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 8)] [(0, 12)]
          auto& ir5 = r5;
          float v103_data = r3[0];
          float v104_data = r3[1];
          float v105_data = r3[2];
          float v106_data = r3[3];
          float v107_data = r3[4];
          float v108_data = r3[5];
          float v109_data = r3[6];
          float v110_data = r3[7];
          float v111_data = r3[8];
          float v112_data = r3[9];
          float v113_data = r3[10];
          float v114_data = r3[11];
          float v115_acc{};
          float v116_acc{};
          float v117_acc{};
          float v118_acc{};
          float v119_acc{};
          float v120_acc{};
          float v121_acc{};
          float v122_acc{};
          float v123_lin = r4[0];
          float v124_bc = tensorforge::broadcast<32, 16, 0>(v123_lin);
          tensorforge::fmacdpp16<0>(v115_acc, v124_bc, v103_data);
          tensorforge::fmacdpp16<1>(v115_acc, v124_bc, v104_data);
          tensorforge::fmacdpp16<2>(v115_acc, v124_bc, v105_data);
          tensorforge::fmacdpp16<3>(v115_acc, v124_bc, v106_data);
          tensorforge::fmacdpp16<4>(v115_acc, v124_bc, v107_data);
          tensorforge::fmacdpp16<5>(v115_acc, v124_bc, v108_data);
          tensorforge::fmacdpp16<6>(v115_acc, v124_bc, v109_data);
          tensorforge::fmacdpp16<7>(v115_acc, v124_bc, v110_data);
          tensorforge::fmacdpp16<8>(v115_acc, v124_bc, v111_data);
          tensorforge::fmacdpp16<9>(v115_acc, v124_bc, v112_data);
          tensorforge::fmacdpp16<10>(v115_acc, v124_bc, v113_data);
          tensorforge::fmacdpp16<11>(v115_acc, v124_bc, v114_data);
          tensorforge::fmacdpp16<12>(v116_acc, v124_bc, v103_data);
          tensorforge::fmacdpp16<13>(v116_acc, v124_bc, v104_data);
          tensorforge::fmacdpp16<14>(v116_acc, v124_bc, v105_data);
          tensorforge::fmacdpp16<15>(v116_acc, v124_bc, v106_data);
          float v125_bc = tensorforge::broadcast<32, 16, 1>(v123_lin);
          tensorforge::fmacdpp16<0>(v116_acc, v125_bc, v107_data);
          tensorforge::fmacdpp16<1>(v116_acc, v125_bc, v108_data);
          tensorforge::fmacdpp16<2>(v116_acc, v125_bc, v109_data);
          tensorforge::fmacdpp16<3>(v116_acc, v125_bc, v110_data);
          tensorforge::fmacdpp16<4>(v116_acc, v125_bc, v111_data);
          tensorforge::fmacdpp16<5>(v116_acc, v125_bc, v112_data);
          tensorforge::fmacdpp16<6>(v116_acc, v125_bc, v113_data);
          tensorforge::fmacdpp16<7>(v116_acc, v125_bc, v114_data);
          tensorforge::fmacdpp16<8>(v117_acc, v125_bc, v103_data);
          tensorforge::fmacdpp16<9>(v117_acc, v125_bc, v104_data);
          tensorforge::fmacdpp16<10>(v117_acc, v125_bc, v105_data);
          tensorforge::fmacdpp16<11>(v117_acc, v125_bc, v106_data);
          tensorforge::fmacdpp16<12>(v117_acc, v125_bc, v107_data);
          tensorforge::fmacdpp16<13>(v117_acc, v125_bc, v108_data);
          tensorforge::fmacdpp16<14>(v117_acc, v125_bc, v109_data);
          tensorforge::fmacdpp16<15>(v117_acc, v125_bc, v110_data);
          float v126_lin = r4[1];
          float v127_bc = tensorforge::broadcast<32, 16, 0>(v126_lin);
          tensorforge::fmacdpp16<0>(v117_acc, v127_bc, v111_data);
          tensorforge::fmacdpp16<1>(v117_acc, v127_bc, v112_data);
          tensorforge::fmacdpp16<2>(v117_acc, v127_bc, v113_data);
          tensorforge::fmacdpp16<3>(v117_acc, v127_bc, v114_data);
          tensorforge::fmacdpp16<4>(v118_acc, v127_bc, v103_data);
          tensorforge::fmacdpp16<5>(v118_acc, v127_bc, v104_data);
          tensorforge::fmacdpp16<6>(v118_acc, v127_bc, v105_data);
          tensorforge::fmacdpp16<7>(v118_acc, v127_bc, v106_data);
          tensorforge::fmacdpp16<8>(v118_acc, v127_bc, v107_data);
          tensorforge::fmacdpp16<9>(v118_acc, v127_bc, v108_data);
          tensorforge::fmacdpp16<10>(v118_acc, v127_bc, v109_data);
          tensorforge::fmacdpp16<11>(v118_acc, v127_bc, v110_data);
          tensorforge::fmacdpp16<12>(v118_acc, v127_bc, v111_data);
          tensorforge::fmacdpp16<13>(v118_acc, v127_bc, v112_data);
          tensorforge::fmacdpp16<14>(v118_acc, v127_bc, v113_data);
          tensorforge::fmacdpp16<15>(v118_acc, v127_bc, v114_data);
          float v128_bc = tensorforge::broadcast<32, 16, 1>(v126_lin);
          tensorforge::fmacdpp16<0>(v119_acc, v128_bc, v103_data);
          tensorforge::fmacdpp16<1>(v119_acc, v128_bc, v104_data);
          tensorforge::fmacdpp16<2>(v119_acc, v128_bc, v105_data);
          tensorforge::fmacdpp16<3>(v119_acc, v128_bc, v106_data);
          tensorforge::fmacdpp16<4>(v119_acc, v128_bc, v107_data);
          tensorforge::fmacdpp16<5>(v119_acc, v128_bc, v108_data);
          tensorforge::fmacdpp16<6>(v119_acc, v128_bc, v109_data);
          tensorforge::fmacdpp16<7>(v119_acc, v128_bc, v110_data);
          tensorforge::fmacdpp16<8>(v119_acc, v128_bc, v111_data);
          tensorforge::fmacdpp16<9>(v119_acc, v128_bc, v112_data);
          tensorforge::fmacdpp16<10>(v119_acc, v128_bc, v113_data);
          tensorforge::fmacdpp16<11>(v119_acc, v128_bc, v114_data);
          tensorforge::fmacdpp16<12>(v120_acc, v128_bc, v103_data);
          tensorforge::fmacdpp16<13>(v120_acc, v128_bc, v104_data);
          tensorforge::fmacdpp16<14>(v120_acc, v128_bc, v105_data);
          tensorforge::fmacdpp16<15>(v120_acc, v128_bc, v106_data);
          float v129_lin = r4[2];
          float v130_bc = tensorforge::broadcast<32, 16, 0>(v129_lin);
          tensorforge::fmacdpp16<0>(v120_acc, v130_bc, v107_data);
          tensorforge::fmacdpp16<1>(v120_acc, v130_bc, v108_data);
          tensorforge::fmacdpp16<2>(v120_acc, v130_bc, v109_data);
          tensorforge::fmacdpp16<3>(v120_acc, v130_bc, v110_data);
          tensorforge::fmacdpp16<4>(v120_acc, v130_bc, v111_data);
          tensorforge::fmacdpp16<5>(v120_acc, v130_bc, v112_data);
          tensorforge::fmacdpp16<6>(v120_acc, v130_bc, v113_data);
          tensorforge::fmacdpp16<7>(v120_acc, v130_bc, v114_data);
          tensorforge::fmacdpp16<8>(v121_acc, v130_bc, v103_data);
          tensorforge::fmacdpp16<9>(v121_acc, v130_bc, v104_data);
          tensorforge::fmacdpp16<10>(v121_acc, v130_bc, v105_data);
          tensorforge::fmacdpp16<11>(v121_acc, v130_bc, v106_data);
          tensorforge::fmacdpp16<12>(v121_acc, v130_bc, v107_data);
          tensorforge::fmacdpp16<13>(v121_acc, v130_bc, v108_data);
          tensorforge::fmacdpp16<14>(v121_acc, v130_bc, v109_data);
          tensorforge::fmacdpp16<15>(v121_acc, v130_bc, v110_data);
          float v131_bc = tensorforge::broadcast<32, 16, 1>(v129_lin);
          tensorforge::fmacdpp16<0>(v121_acc, v131_bc, v111_data);
          tensorforge::fmacdpp16<1>(v121_acc, v131_bc, v112_data);
          tensorforge::fmacdpp16<2>(v121_acc, v131_bc, v113_data);
          tensorforge::fmacdpp16<3>(v121_acc, v131_bc, v114_data);
          tensorforge::fmacdpp16<4>(v122_acc, v131_bc, v103_data);
          tensorforge::fmacdpp16<5>(v122_acc, v131_bc, v104_data);
          tensorforge::fmacdpp16<6>(v122_acc, v131_bc, v105_data);
          tensorforge::fmacdpp16<7>(v122_acc, v131_bc, v106_data);
          tensorforge::fmacdpp16<8>(v122_acc, v131_bc, v107_data);
          tensorforge::fmacdpp16<9>(v122_acc, v131_bc, v108_data);
          tensorforge::fmacdpp16<10>(v122_acc, v131_bc, v109_data);
          tensorforge::fmacdpp16<11>(v122_acc, v131_bc, v110_data);
          tensorforge::fmacdpp16<12>(v122_acc, v131_bc, v111_data);
          tensorforge::fmacdpp16<13>(v122_acc, v131_bc, v112_data);
          tensorforge::fmacdpp16<14>(v122_acc, v131_bc, v113_data);
          tensorforge::fmacdpp16<15>(v122_acc, v131_bc, v114_data);
          ir5[0] = v115_acc;
          ir5[1] = v116_acc;
          ir5[2] = v117_acc;
          ir5[3] = v118_acc;
          ir5[4] = v119_acc;
          ir5[5] = v120_acc;
          ir5[6] = v121_acc;
          ir5[7] = v122_acc;
          // glb_m0 = store{r>g}(r5);
          int32_t v134_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v135_i0 = 0; v135_i0 < 1; ++v135_i0) {
            int32_t v144_lead = v134_lead + (v135_i0 * 32);
            #pragma unroll
            for (int32_t v136_i1 = 0; v136_i1 < 8; ++v136_i1) {
              int32_t v137_a = v135_i0 + v136_i1;
              float v139_data = r5[(v135_i0 + v136_i1)];
              int32_t v146_a = v144_lead + (v136_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v146_a], v139_data);
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
          float v147_data = r6[0];
          float v148_data = r6[1];
          float v149_data = r6[2];
          float v150_data = r6[3];
          float v151_data = r6[4];
          float v152_data = r6[5];
          float v153_data = r6[6];
          float v154_data = r6[7];
          float v155_data = r6[8];
          float v156_data = r6[9];
          float v157_data = r6[10];
          float v158_data = r6[11];
          float v159_acc{};
          float v160_acc{};
          float v161_acc{};
          float v162_acc{};
          float v163_acc{};
          float v164_acc{};
          float v165_acc{};
          float v166_acc{};
          float v167_lin = r7[0];
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
          float v170_lin = r7[1];
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
          float v173_lin = r7[2];
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
          ir8[0] = v159_acc;
          ir8[1] = v160_acc;
          ir8[2] = v161_acc;
          ir8[3] = v162_acc;
          ir8[4] = v163_acc;
          ir8[5] = v164_acc;
          ir8[6] = v165_acc;
          ir8[7] = v166_acc;
          // glb_m0 = store{r>g}(r8);
          int32_t v178_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v179_i0 = 0; v179_i0 < 1; ++v179_i0) {
            int32_t v188_lead = v178_lead + (v179_i0 * 32);
            #pragma unroll
            for (int32_t v180_i1 = 0; v180_i1 < 8; ++v180_i1) {
              int32_t v181_a = v179_i0 + v180_i1;
              float v183_data = r8[(v179_i0 + v180_i1)];
              int32_t v191_a = v188_lead + ((v180_i1 + 8) * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v191_a], v183_data);
            }
          }
          ;
        }
      }
    }
  }
}

