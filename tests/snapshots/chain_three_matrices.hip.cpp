// === base name ===
kernel_4da108b421

// === header ===
void launcher_kernel_4da108b421(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4da108b421(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4da108b421, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4da108b421), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_4da108b421, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4da108b421(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 56×9(56×9) {0..56}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 56×9(56×9) {0..56}×{0..9} strided
    // m3 56×56(56×56) {0..56}×{0..56} strided
    // t0 56×9(56×9) {0..56}×{0..9} pointer_based({0..56}×{0..9})[0, 1] = m0 56×9(56×9) {0..56}×{0..9} strided({0..56}×{0..9})[0, -1]×m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]
    // m2 56×9(56×9) {0..56}×{0..9} strided({0..56}×{0..9})[0, 1] = m3 56×56(56×56) {0..56}×{0..56} strided({0..56}×{0..56})[0, -1]×t0 56×9(56×9) {0..56}×{0..9} pointer_based({0..56}×{0..9})[-1, 1]
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
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 504 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 504 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 3136 + 0 + m3_extraOffset];
          float r0[18]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 9; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 56);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m0[v11_a]);
              }
              int32_t v14_a = v3_i0 + (v4_i1 * 2);
              r0[v14_a] = v12_data;
            }
          }
          if (v2_lead < 24) {
            int32_t v21_lead = v2_lead + 32_i32;
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 9; ++v16_i1) {
              int32_t v23_a = v21_lead + (v16_i1 * 56);
              float v24_data;
              {
                v24_data = __builtin_nontemporal_load(&glb_m0[v23_a]);
              }
              int32_t v26_a = 1 + (v16_i1 * 2);
              r0[v26_a] = v24_data;
            }
          }
          float r1[9]{};
          {
            // r1 = load{g>r}(glb_m1);
            float v0 = glb_m1[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v32 = glb_m1[32 + threadIdx.x * 1];
            r1[1] = v32;
            float v64 = glb_m1[64 + threadIdx.x * 1];
            r1[2] = v64;
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[112]{};
          // r3 = load{g>r}(glb_m3);
          int32_t v29_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
            int32_t v36_lead = v29_lead + (v30_i0 * 32);
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 56; ++v31_i1) {
              int32_t v38_a = v36_lead + (v31_i1 * 56);
              float v39_data;
              {
                v39_data = __builtin_nontemporal_load(&glb_m3[v38_a]);
              }
              int32_t v41_a = v30_i0 + (v31_i1 * 2);
              r3[v41_a] = v39_data;
            }
          }
          if (v29_lead < 24) {
            int32_t v48_lead = v29_lead + 32_i32;
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 56; ++v43_i1) {
              int32_t v50_a = v48_lead + (v43_i1 * 56);
              float v51_data;
              {
                v51_data = __builtin_nontemporal_load(&glb_m3[v50_a]);
              }
              int32_t v53_a = 1 + (v43_i1 * 2);
              r3[v53_a] = v51_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[18]{};
          // r2 = +(r0 * r1) + None
          // [(0, 56), (0, 9)] [(0, 9)]
          auto& ir2 = r2;
          float v54_data = r0[0];
          float v55_data = r0[1];
          float v56_data = r0[2];
          float v57_data = r0[3];
          float v58_data = r0[4];
          float v59_data = r0[5];
          float v60_data = r0[6];
          float v61_data = r0[7];
          float v62_data = r0[8];
          float v63_data = r0[9];
          float v64_data = r0[10];
          float v65_data = r0[11];
          float v66_data = r0[12];
          float v67_data = r0[13];
          float v68_data = r0[14];
          float v69_data = r0[15];
          float v70_data = r0[16];
          float v71_data = r0[17];
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
          float v87_acc{};
          float v88_acc{};
          float v89_acc{};
          float v90_lin = r1[0];
          float v91_bc = tensorforge::broadcast<32, 16, 0>(v90_lin);
          tensorforge::fmacdpp16<0>(v72_acc, v91_bc, v54_data);
          tensorforge::fmacdpp16<0>(v73_acc, v91_bc, v55_data);
          tensorforge::fmacdpp16<1>(v72_acc, v91_bc, v56_data);
          tensorforge::fmacdpp16<1>(v73_acc, v91_bc, v57_data);
          tensorforge::fmacdpp16<2>(v72_acc, v91_bc, v58_data);
          tensorforge::fmacdpp16<2>(v73_acc, v91_bc, v59_data);
          tensorforge::fmacdpp16<3>(v72_acc, v91_bc, v60_data);
          tensorforge::fmacdpp16<3>(v73_acc, v91_bc, v61_data);
          tensorforge::fmacdpp16<4>(v72_acc, v91_bc, v62_data);
          tensorforge::fmacdpp16<4>(v73_acc, v91_bc, v63_data);
          tensorforge::fmacdpp16<5>(v72_acc, v91_bc, v64_data);
          tensorforge::fmacdpp16<5>(v73_acc, v91_bc, v65_data);
          tensorforge::fmacdpp16<6>(v72_acc, v91_bc, v66_data);
          tensorforge::fmacdpp16<6>(v73_acc, v91_bc, v67_data);
          tensorforge::fmacdpp16<7>(v72_acc, v91_bc, v68_data);
          tensorforge::fmacdpp16<7>(v73_acc, v91_bc, v69_data);
          tensorforge::fmacdpp16<8>(v72_acc, v91_bc, v70_data);
          tensorforge::fmacdpp16<8>(v73_acc, v91_bc, v71_data);
          tensorforge::fmacdpp16<9>(v74_acc, v91_bc, v54_data);
          tensorforge::fmacdpp16<9>(v75_acc, v91_bc, v55_data);
          tensorforge::fmacdpp16<10>(v74_acc, v91_bc, v56_data);
          tensorforge::fmacdpp16<10>(v75_acc, v91_bc, v57_data);
          tensorforge::fmacdpp16<11>(v74_acc, v91_bc, v58_data);
          tensorforge::fmacdpp16<11>(v75_acc, v91_bc, v59_data);
          tensorforge::fmacdpp16<12>(v74_acc, v91_bc, v60_data);
          tensorforge::fmacdpp16<12>(v75_acc, v91_bc, v61_data);
          tensorforge::fmacdpp16<13>(v74_acc, v91_bc, v62_data);
          tensorforge::fmacdpp16<13>(v75_acc, v91_bc, v63_data);
          tensorforge::fmacdpp16<14>(v74_acc, v91_bc, v64_data);
          tensorforge::fmacdpp16<14>(v75_acc, v91_bc, v65_data);
          tensorforge::fmacdpp16<15>(v74_acc, v91_bc, v66_data);
          tensorforge::fmacdpp16<15>(v75_acc, v91_bc, v67_data);
          float v92_bc = tensorforge::broadcast<32, 16, 1>(v90_lin);
          tensorforge::fmacdpp16<0>(v74_acc, v92_bc, v68_data);
          tensorforge::fmacdpp16<0>(v75_acc, v92_bc, v69_data);
          tensorforge::fmacdpp16<1>(v74_acc, v92_bc, v70_data);
          tensorforge::fmacdpp16<1>(v75_acc, v92_bc, v71_data);
          tensorforge::fmacdpp16<2>(v76_acc, v92_bc, v54_data);
          tensorforge::fmacdpp16<2>(v77_acc, v92_bc, v55_data);
          tensorforge::fmacdpp16<3>(v76_acc, v92_bc, v56_data);
          tensorforge::fmacdpp16<3>(v77_acc, v92_bc, v57_data);
          tensorforge::fmacdpp16<4>(v76_acc, v92_bc, v58_data);
          tensorforge::fmacdpp16<4>(v77_acc, v92_bc, v59_data);
          tensorforge::fmacdpp16<5>(v76_acc, v92_bc, v60_data);
          tensorforge::fmacdpp16<5>(v77_acc, v92_bc, v61_data);
          tensorforge::fmacdpp16<6>(v76_acc, v92_bc, v62_data);
          tensorforge::fmacdpp16<6>(v77_acc, v92_bc, v63_data);
          tensorforge::fmacdpp16<7>(v76_acc, v92_bc, v64_data);
          tensorforge::fmacdpp16<7>(v77_acc, v92_bc, v65_data);
          tensorforge::fmacdpp16<8>(v76_acc, v92_bc, v66_data);
          tensorforge::fmacdpp16<8>(v77_acc, v92_bc, v67_data);
          tensorforge::fmacdpp16<9>(v76_acc, v92_bc, v68_data);
          tensorforge::fmacdpp16<9>(v77_acc, v92_bc, v69_data);
          tensorforge::fmacdpp16<10>(v76_acc, v92_bc, v70_data);
          tensorforge::fmacdpp16<10>(v77_acc, v92_bc, v71_data);
          tensorforge::fmacdpp16<11>(v78_acc, v92_bc, v54_data);
          tensorforge::fmacdpp16<11>(v79_acc, v92_bc, v55_data);
          tensorforge::fmacdpp16<12>(v78_acc, v92_bc, v56_data);
          tensorforge::fmacdpp16<12>(v79_acc, v92_bc, v57_data);
          tensorforge::fmacdpp16<13>(v78_acc, v92_bc, v58_data);
          tensorforge::fmacdpp16<13>(v79_acc, v92_bc, v59_data);
          tensorforge::fmacdpp16<14>(v78_acc, v92_bc, v60_data);
          tensorforge::fmacdpp16<14>(v79_acc, v92_bc, v61_data);
          tensorforge::fmacdpp16<15>(v78_acc, v92_bc, v62_data);
          tensorforge::fmacdpp16<15>(v79_acc, v92_bc, v63_data);
          float v93_lin = r1[1];
          float v94_bc = tensorforge::broadcast<32, 16, 0>(v93_lin);
          tensorforge::fmacdpp16<0>(v78_acc, v94_bc, v64_data);
          tensorforge::fmacdpp16<0>(v79_acc, v94_bc, v65_data);
          tensorforge::fmacdpp16<1>(v78_acc, v94_bc, v66_data);
          tensorforge::fmacdpp16<1>(v79_acc, v94_bc, v67_data);
          tensorforge::fmacdpp16<2>(v78_acc, v94_bc, v68_data);
          tensorforge::fmacdpp16<2>(v79_acc, v94_bc, v69_data);
          tensorforge::fmacdpp16<3>(v78_acc, v94_bc, v70_data);
          tensorforge::fmacdpp16<3>(v79_acc, v94_bc, v71_data);
          tensorforge::fmacdpp16<4>(v80_acc, v94_bc, v54_data);
          tensorforge::fmacdpp16<4>(v81_acc, v94_bc, v55_data);
          tensorforge::fmacdpp16<5>(v80_acc, v94_bc, v56_data);
          tensorforge::fmacdpp16<5>(v81_acc, v94_bc, v57_data);
          tensorforge::fmacdpp16<6>(v80_acc, v94_bc, v58_data);
          tensorforge::fmacdpp16<6>(v81_acc, v94_bc, v59_data);
          tensorforge::fmacdpp16<7>(v80_acc, v94_bc, v60_data);
          tensorforge::fmacdpp16<7>(v81_acc, v94_bc, v61_data);
          tensorforge::fmacdpp16<8>(v80_acc, v94_bc, v62_data);
          tensorforge::fmacdpp16<8>(v81_acc, v94_bc, v63_data);
          tensorforge::fmacdpp16<9>(v80_acc, v94_bc, v64_data);
          tensorforge::fmacdpp16<9>(v81_acc, v94_bc, v65_data);
          tensorforge::fmacdpp16<10>(v80_acc, v94_bc, v66_data);
          tensorforge::fmacdpp16<10>(v81_acc, v94_bc, v67_data);
          tensorforge::fmacdpp16<11>(v80_acc, v94_bc, v68_data);
          tensorforge::fmacdpp16<11>(v81_acc, v94_bc, v69_data);
          tensorforge::fmacdpp16<12>(v80_acc, v94_bc, v70_data);
          tensorforge::fmacdpp16<12>(v81_acc, v94_bc, v71_data);
          tensorforge::fmacdpp16<13>(v82_acc, v94_bc, v54_data);
          tensorforge::fmacdpp16<13>(v83_acc, v94_bc, v55_data);
          tensorforge::fmacdpp16<14>(v82_acc, v94_bc, v56_data);
          tensorforge::fmacdpp16<14>(v83_acc, v94_bc, v57_data);
          tensorforge::fmacdpp16<15>(v82_acc, v94_bc, v58_data);
          tensorforge::fmacdpp16<15>(v83_acc, v94_bc, v59_data);
          float v95_bc = tensorforge::broadcast<32, 16, 1>(v93_lin);
          tensorforge::fmacdpp16<0>(v82_acc, v95_bc, v60_data);
          tensorforge::fmacdpp16<0>(v83_acc, v95_bc, v61_data);
          tensorforge::fmacdpp16<1>(v82_acc, v95_bc, v62_data);
          tensorforge::fmacdpp16<1>(v83_acc, v95_bc, v63_data);
          tensorforge::fmacdpp16<2>(v82_acc, v95_bc, v64_data);
          tensorforge::fmacdpp16<2>(v83_acc, v95_bc, v65_data);
          tensorforge::fmacdpp16<3>(v82_acc, v95_bc, v66_data);
          tensorforge::fmacdpp16<3>(v83_acc, v95_bc, v67_data);
          tensorforge::fmacdpp16<4>(v82_acc, v95_bc, v68_data);
          tensorforge::fmacdpp16<4>(v83_acc, v95_bc, v69_data);
          tensorforge::fmacdpp16<5>(v82_acc, v95_bc, v70_data);
          tensorforge::fmacdpp16<5>(v83_acc, v95_bc, v71_data);
          tensorforge::fmacdpp16<6>(v84_acc, v95_bc, v54_data);
          tensorforge::fmacdpp16<6>(v85_acc, v95_bc, v55_data);
          tensorforge::fmacdpp16<7>(v84_acc, v95_bc, v56_data);
          tensorforge::fmacdpp16<7>(v85_acc, v95_bc, v57_data);
          tensorforge::fmacdpp16<8>(v84_acc, v95_bc, v58_data);
          tensorforge::fmacdpp16<8>(v85_acc, v95_bc, v59_data);
          tensorforge::fmacdpp16<9>(v84_acc, v95_bc, v60_data);
          tensorforge::fmacdpp16<9>(v85_acc, v95_bc, v61_data);
          tensorforge::fmacdpp16<10>(v84_acc, v95_bc, v62_data);
          tensorforge::fmacdpp16<10>(v85_acc, v95_bc, v63_data);
          tensorforge::fmacdpp16<11>(v84_acc, v95_bc, v64_data);
          tensorforge::fmacdpp16<11>(v85_acc, v95_bc, v65_data);
          tensorforge::fmacdpp16<12>(v84_acc, v95_bc, v66_data);
          tensorforge::fmacdpp16<12>(v85_acc, v95_bc, v67_data);
          tensorforge::fmacdpp16<13>(v84_acc, v95_bc, v68_data);
          tensorforge::fmacdpp16<13>(v85_acc, v95_bc, v69_data);
          tensorforge::fmacdpp16<14>(v84_acc, v95_bc, v70_data);
          tensorforge::fmacdpp16<14>(v85_acc, v95_bc, v71_data);
          tensorforge::fmacdpp16<15>(v86_acc, v95_bc, v54_data);
          tensorforge::fmacdpp16<15>(v87_acc, v95_bc, v55_data);
          float v96_lin = r1[2];
          float v97_bc = tensorforge::broadcast<32, 16, 0>(v96_lin);
          tensorforge::fmacdpp16<0>(v86_acc, v97_bc, v56_data);
          tensorforge::fmacdpp16<0>(v87_acc, v97_bc, v57_data);
          tensorforge::fmacdpp16<1>(v86_acc, v97_bc, v58_data);
          tensorforge::fmacdpp16<1>(v87_acc, v97_bc, v59_data);
          tensorforge::fmacdpp16<2>(v86_acc, v97_bc, v60_data);
          tensorforge::fmacdpp16<2>(v87_acc, v97_bc, v61_data);
          tensorforge::fmacdpp16<3>(v86_acc, v97_bc, v62_data);
          tensorforge::fmacdpp16<3>(v87_acc, v97_bc, v63_data);
          tensorforge::fmacdpp16<4>(v86_acc, v97_bc, v64_data);
          tensorforge::fmacdpp16<4>(v87_acc, v97_bc, v65_data);
          tensorforge::fmacdpp16<5>(v86_acc, v97_bc, v66_data);
          tensorforge::fmacdpp16<5>(v87_acc, v97_bc, v67_data);
          tensorforge::fmacdpp16<6>(v86_acc, v97_bc, v68_data);
          tensorforge::fmacdpp16<6>(v87_acc, v97_bc, v69_data);
          tensorforge::fmacdpp16<7>(v86_acc, v97_bc, v70_data);
          tensorforge::fmacdpp16<7>(v87_acc, v97_bc, v71_data);
          tensorforge::fmacdpp16<8>(v88_acc, v97_bc, v54_data);
          tensorforge::fmacdpp16<8>(v89_acc, v97_bc, v55_data);
          tensorforge::fmacdpp16<9>(v88_acc, v97_bc, v56_data);
          tensorforge::fmacdpp16<9>(v89_acc, v97_bc, v57_data);
          tensorforge::fmacdpp16<10>(v88_acc, v97_bc, v58_data);
          tensorforge::fmacdpp16<10>(v89_acc, v97_bc, v59_data);
          tensorforge::fmacdpp16<11>(v88_acc, v97_bc, v60_data);
          tensorforge::fmacdpp16<11>(v89_acc, v97_bc, v61_data);
          tensorforge::fmacdpp16<12>(v88_acc, v97_bc, v62_data);
          tensorforge::fmacdpp16<12>(v89_acc, v97_bc, v63_data);
          tensorforge::fmacdpp16<13>(v88_acc, v97_bc, v64_data);
          tensorforge::fmacdpp16<13>(v89_acc, v97_bc, v65_data);
          tensorforge::fmacdpp16<14>(v88_acc, v97_bc, v66_data);
          tensorforge::fmacdpp16<14>(v89_acc, v97_bc, v67_data);
          tensorforge::fmacdpp16<15>(v88_acc, v97_bc, v68_data);
          tensorforge::fmacdpp16<15>(v89_acc, v97_bc, v69_data);
          float v98_bc = tensorforge::broadcast<32, 16, 1>(v96_lin);
          tensorforge::fmacdpp16<0>(v88_acc, v98_bc, v70_data);
          tensorforge::fmacdpp16<0>(v89_acc, v98_bc, v71_data);
          ir2[0] = v72_acc;
          ir2[1] = v73_acc;
          ir2[2] = v74_acc;
          ir2[3] = v75_acc;
          ir2[4] = v76_acc;
          ir2[5] = v77_acc;
          ir2[6] = v78_acc;
          ir2[7] = v79_acc;
          ir2[8] = v80_acc;
          ir2[9] = v81_acc;
          ir2[10] = v82_acc;
          ir2[11] = v83_acc;
          ir2[12] = v84_acc;
          ir2[13] = v85_acc;
          ir2[14] = v86_acc;
          ir2[15] = v87_acc;
          ir2[16] = v88_acc;
          ir2[17] = v89_acc;
          // wait(r3 = load{g>r}(glb_m3););
          float r4[18]{};
          // r4 = +(r3 * r2) + None
          // [(0, 56), (0, 9)] [(0, 56)]
          auto& ir4 = r4;
          float v99_data = r2[0];
          float v100_data = r2[2];
          float v101_data = r2[4];
          float v102_data = r2[6];
          float v103_tp{};
          float v104_tp{};
          float v105_tp{};
          float v106_tp{};
          tensorforge::transpose4x4b32(v103_tp, v104_tp, v105_tp, v106_tp, v99_data, v100_data, v101_data, v102_data);
          float v107_data = r2[1];
          float v108_data = r2[3];
          float v109_data = r2[5];
          float v110_data = r2[7];
          float v111_tp{};
          float v112_tp{};
          float v113_tp{};
          float v114_tp{};
          tensorforge::transpose4x4b32(v111_tp, v112_tp, v113_tp, v114_tp, v107_data, v108_data, v109_data, v110_data);
          tensorforge::VectorT<float, 4> v115_acc{};
          float v116_data = r3[0];
          float v117_data = r3[2];
          float v118_data = r3[4];
          float v119_data = r3[6];
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v116_data, v115_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v117_data, v120_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v118_data, v121_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v119_data, v122_acc, 3, 0, 0);
          float v124_data = r3[8];
          float v125_data = r3[10];
          float v126_data = r3[12];
          float v127_data = r3[14];
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v124_data, v123_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v125_data, v128_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v126_data, v129_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v127_data, v130_acc, 3, 1, 0);
          float v132_data = r3[16];
          float v133_data = r3[18];
          float v134_data = r3[20];
          float v135_data = r3[22];
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v132_data, v131_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v133_data, v136_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v134_data, v137_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v135_data, v138_acc, 3, 2, 0);
          float v140_data = r3[24];
          float v141_data = r3[26];
          float v142_data = r3[28];
          float v143_data = r3[30];
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v140_data, v139_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v141_data, v144_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v142_data, v145_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v143_data, v146_acc, 3, 3, 0);
          float v148_data = r3[32];
          float v149_data = r3[34];
          float v150_data = r3[36];
          float v151_data = r3[38];
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v148_data, v147_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v149_data, v152_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v150_data, v153_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v151_data, v154_acc, 3, 4, 0);
          float v156_data = r3[40];
          float v157_data = r3[42];
          float v158_data = r3[44];
          float v159_data = r3[46];
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v156_data, v155_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v157_data, v160_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v158_data, v161_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v159_data, v162_acc, 3, 5, 0);
          float v164_data = r3[48];
          float v165_data = r3[50];
          float v166_data = r3[52];
          float v167_data = r3[54];
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v164_data, v163_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v165_data, v168_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v166_data, v169_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v167_data, v170_acc, 3, 6, 0);
          float v172_data = r3[56];
          float v173_data = r3[58];
          float v174_data = r3[60];
          float v175_data = r3[62];
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v172_data, v171_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v173_data, v176_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v174_data, v177_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v175_data, v178_acc, 3, 7, 0);
          float v180_data = r3[64];
          float v181_data = r3[66];
          float v182_data = r3[68];
          float v183_data = r3[70];
          tensorforge::VectorT<float, 4> v184_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v180_data, v179_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v185_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v181_data, v184_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v186_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v182_data, v185_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v187_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v183_data, v186_acc, 3, 0, 0);
          float v188_data = r3[72];
          float v189_data = r3[74];
          float v190_data = r3[76];
          float v191_data = r3[78];
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v188_data, v187_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v189_data, v192_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v190_data, v193_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v191_data, v194_acc, 3, 1, 0);
          float v196_data = r3[80];
          float v197_data = r3[82];
          float v198_data = r3[84];
          float v199_data = r3[86];
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v196_data, v195_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v197_data, v200_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v198_data, v201_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v199_data, v202_acc, 3, 2, 0);
          float v204_data = r3[88];
          float v205_data = r3[90];
          float v206_data = r3[92];
          float v207_data = r3[94];
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v204_data, v203_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v205_data, v208_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v206_data, v209_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v207_data, v210_acc, 3, 3, 0);
          float v212_data = r3[96];
          float v213_data = r3[98];
          float v214_data = r3[100];
          float v215_data = r3[102];
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v212_data, v211_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v213_data, v216_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v214_data, v217_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v215_data, v218_acc, 3, 4, 0);
          float v220_data = r3[104];
          float v221_data = r3[106];
          float v222_data = r3[108];
          float v223_data = r3[110];
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v220_data, v219_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v221_data, v224_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v222_data, v225_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v223_data, v226_acc, 3, 5, 0);
          ir4[0] = (v227_acc[0]);
          ir4[2] = (v227_acc[1]);
          ir4[4] = (v227_acc[2]);
          ir4[6] = (v227_acc[3]);
          tensorforge::VectorT<float, 4> v232_acc{};
          float v233_data = r3[1];
          float v234_data = r3[3];
          float v235_data = r3[5];
          float v236_data = r3[7];
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v233_data, v232_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v234_data, v237_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v235_data, v238_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v236_data, v239_acc, 3, 0, 0);
          float v241_data = r3[9];
          float v242_data = r3[11];
          float v243_data = r3[13];
          float v244_data = r3[15];
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v241_data, v240_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v242_data, v245_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v243_data, v246_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v244_data, v247_acc, 3, 1, 0);
          float v249_data = r3[17];
          float v250_data = r3[19];
          float v251_data = r3[21];
          float v252_data = r3[23];
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v249_data, v248_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v250_data, v253_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v251_data, v254_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v252_data, v255_acc, 3, 2, 0);
          float v257_data = r3[25];
          float v258_data = r3[27];
          float v259_data = r3[29];
          float v260_data = r3[31];
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v257_data, v256_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v258_data, v261_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v259_data, v262_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v260_data, v263_acc, 3, 3, 0);
          float v265_data = r3[33];
          float v266_data = r3[35];
          float v267_data = r3[37];
          float v268_data = r3[39];
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v265_data, v264_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v266_data, v269_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v267_data, v270_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v268_data, v271_acc, 3, 4, 0);
          float v273_data = r3[41];
          float v274_data = r3[43];
          float v275_data = r3[45];
          float v276_data = r3[47];
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v273_data, v272_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v274_data, v277_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v275_data, v278_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v276_data, v279_acc, 3, 5, 0);
          float v281_data = r3[49];
          float v282_data = r3[51];
          float v283_data = r3[53];
          float v284_data = r3[55];
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v281_data, v280_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v282_data, v285_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v283_data, v286_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v284_data, v287_acc, 3, 6, 0);
          float v289_data = r3[57];
          float v290_data = r3[59];
          float v291_data = r3[61];
          float v292_data = r3[63];
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v289_data, v288_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v290_data, v293_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v291_data, v294_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v292_data, v295_acc, 3, 7, 0);
          float v297_data = r3[65];
          float v298_data = r3[67];
          float v299_data = r3[69];
          float v300_data = r3[71];
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v297_data, v296_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v298_data, v301_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v299_data, v302_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v300_data, v303_acc, 3, 0, 0);
          float v305_data = r3[73];
          float v306_data = r3[75];
          float v307_data = r3[77];
          float v308_data = r3[79];
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v305_data, v304_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v306_data, v309_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v307_data, v310_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v308_data, v311_acc, 3, 1, 0);
          float v313_data = r3[81];
          float v314_data = r3[83];
          float v315_data = r3[85];
          float v316_data = r3[87];
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v313_data, v312_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v314_data, v317_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v315_data, v318_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v316_data, v319_acc, 3, 2, 0);
          float v321_data = r3[89];
          float v322_data = r3[91];
          float v323_data = r3[93];
          float v324_data = r3[95];
          tensorforge::VectorT<float, 4> v325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v321_data, v320_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v322_data, v325_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v327_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v323_data, v326_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v328_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v324_data, v327_acc, 3, 3, 0);
          float v329_data = r3[97];
          float v330_data = r3[99];
          float v331_data = r3[101];
          float v332_data = r3[103];
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v329_data, v328_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v330_data, v333_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v335_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v331_data, v334_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v336_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v332_data, v335_acc, 3, 4, 0);
          float v337_data = r3[105];
          float v338_data = r3[107];
          float v339_data = r3[109];
          float v340_data = r3[111];
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v111_tp, v337_data, v336_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v112_tp, v338_data, v341_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v113_tp, v339_data, v342_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v344_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v114_tp, v340_data, v343_acc, 3, 5, 0);
          ir4[1] = (v344_acc[0]);
          ir4[3] = (v344_acc[1]);
          ir4[5] = (v344_acc[2]);
          ir4[7] = (v344_acc[3]);
          float v349_data = r2[8];
          float v350_data = r2[10];
          float v351_data = r2[12];
          float v352_data = r2[14];
          float v353_tp{};
          float v354_tp{};
          float v355_tp{};
          float v356_tp{};
          tensorforge::transpose4x4b32(v353_tp, v354_tp, v355_tp, v356_tp, v349_data, v350_data, v351_data, v352_data);
          float v357_data = r2[9];
          float v358_data = r2[11];
          float v359_data = r2[13];
          float v360_data = r2[15];
          float v361_tp{};
          float v362_tp{};
          float v363_tp{};
          float v364_tp{};
          tensorforge::transpose4x4b32(v361_tp, v362_tp, v363_tp, v364_tp, v357_data, v358_data, v359_data, v360_data);
          tensorforge::VectorT<float, 4> v365_acc{};
          tensorforge::VectorT<float, 4> v370_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v116_data, v365_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v371_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v117_data, v370_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v372_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v118_data, v371_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v373_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v119_data, v372_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v124_data, v373_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v125_data, v378_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v380_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v126_data, v379_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v381_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v127_data, v380_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v132_data, v381_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v133_data, v386_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v134_data, v387_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v389_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v135_data, v388_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v140_data, v389_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v141_data, v394_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v142_data, v395_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v397_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v143_data, v396_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v148_data, v397_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v149_data, v402_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v150_data, v403_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v151_data, v404_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v156_data, v405_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v157_data, v410_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v158_data, v411_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v159_data, v412_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v164_data, v413_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v165_data, v418_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v166_data, v419_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v167_data, v420_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v172_data, v421_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v173_data, v426_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v174_data, v427_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v175_data, v428_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v180_data, v429_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v181_data, v434_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v182_data, v435_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v183_data, v436_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v442_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v188_data, v437_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v443_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v189_data, v442_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v190_data, v443_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v191_data, v444_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v450_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v196_data, v445_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v451_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v197_data, v450_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v198_data, v451_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v199_data, v452_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v458_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v204_data, v453_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v459_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v205_data, v458_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v206_data, v459_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v207_data, v460_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v466_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v212_data, v461_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v467_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v213_data, v466_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v468_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v214_data, v467_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v215_data, v468_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v474_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v220_data, v469_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v475_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v221_data, v474_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v476_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v222_data, v475_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v223_data, v476_acc, 3, 5, 0);
          ir4[8] = (v477_acc[0]);
          ir4[10] = (v477_acc[1]);
          ir4[12] = (v477_acc[2]);
          ir4[14] = (v477_acc[3]);
          tensorforge::VectorT<float, 4> v482_acc{};
          tensorforge::VectorT<float, 4> v487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v233_data, v482_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v234_data, v487_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v235_data, v488_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v236_data, v489_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v241_data, v490_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v242_data, v495_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v243_data, v496_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v244_data, v497_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v249_data, v498_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v250_data, v503_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v251_data, v504_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v252_data, v505_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v257_data, v506_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v258_data, v511_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v259_data, v512_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v260_data, v513_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v519_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v265_data, v514_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v520_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v266_data, v519_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v267_data, v520_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v268_data, v521_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v527_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v273_data, v522_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v528_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v274_data, v527_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v529_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v275_data, v528_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v530_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v276_data, v529_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v535_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v281_data, v530_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v536_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v282_data, v535_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v283_data, v536_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v284_data, v537_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v543_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v353_tp, v289_data, v538_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v544_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v354_tp, v290_data, v543_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v545_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v355_tp, v291_data, v544_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v546_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v356_tp, v292_data, v545_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v551_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v297_data, v546_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v552_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v298_data, v551_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v553_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v299_data, v552_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v554_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v300_data, v553_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v305_data, v554_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v306_data, v559_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v561_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v307_data, v560_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v562_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v308_data, v561_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v313_data, v562_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v314_data, v567_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v569_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v315_data, v568_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v570_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v316_data, v569_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v321_data, v570_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v322_data, v575_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v577_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v323_data, v576_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v578_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v324_data, v577_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v329_data, v578_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v330_data, v583_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v585_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v331_data, v584_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v586_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v332_data, v585_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v591_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v361_tp, v337_data, v586_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v592_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v362_tp, v338_data, v591_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v593_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v363_tp, v339_data, v592_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v594_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v364_tp, v340_data, v593_acc, 3, 5, 0);
          ir4[9] = (v594_acc[0]);
          ir4[11] = (v594_acc[1]);
          ir4[13] = (v594_acc[2]);
          ir4[15] = (v594_acc[3]);
          float v711_acc{};
          float v712_acc{};
          float v713_data = r2[16];
          float v714_data = r2[17];
          float v715_bc = tensorforge::broadcast<32, 16, 0>(v713_data);
          tensorforge::fmacdpp16<0>(v711_acc, v715_bc, v116_data);
          tensorforge::fmacdpp16<0>(v712_acc, v715_bc, v233_data);
          tensorforge::fmacdpp16<1>(v711_acc, v715_bc, v117_data);
          tensorforge::fmacdpp16<1>(v712_acc, v715_bc, v234_data);
          tensorforge::fmacdpp16<2>(v711_acc, v715_bc, v118_data);
          tensorforge::fmacdpp16<2>(v712_acc, v715_bc, v235_data);
          tensorforge::fmacdpp16<3>(v711_acc, v715_bc, v119_data);
          tensorforge::fmacdpp16<3>(v712_acc, v715_bc, v236_data);
          tensorforge::fmacdpp16<4>(v711_acc, v715_bc, v124_data);
          tensorforge::fmacdpp16<4>(v712_acc, v715_bc, v241_data);
          tensorforge::fmacdpp16<5>(v711_acc, v715_bc, v125_data);
          tensorforge::fmacdpp16<5>(v712_acc, v715_bc, v242_data);
          tensorforge::fmacdpp16<6>(v711_acc, v715_bc, v126_data);
          tensorforge::fmacdpp16<6>(v712_acc, v715_bc, v243_data);
          tensorforge::fmacdpp16<7>(v711_acc, v715_bc, v127_data);
          tensorforge::fmacdpp16<7>(v712_acc, v715_bc, v244_data);
          tensorforge::fmacdpp16<8>(v711_acc, v715_bc, v132_data);
          tensorforge::fmacdpp16<8>(v712_acc, v715_bc, v249_data);
          tensorforge::fmacdpp16<9>(v711_acc, v715_bc, v133_data);
          tensorforge::fmacdpp16<9>(v712_acc, v715_bc, v250_data);
          tensorforge::fmacdpp16<10>(v711_acc, v715_bc, v134_data);
          tensorforge::fmacdpp16<10>(v712_acc, v715_bc, v251_data);
          tensorforge::fmacdpp16<11>(v711_acc, v715_bc, v135_data);
          tensorforge::fmacdpp16<11>(v712_acc, v715_bc, v252_data);
          tensorforge::fmacdpp16<12>(v711_acc, v715_bc, v140_data);
          tensorforge::fmacdpp16<12>(v712_acc, v715_bc, v257_data);
          tensorforge::fmacdpp16<13>(v711_acc, v715_bc, v141_data);
          tensorforge::fmacdpp16<13>(v712_acc, v715_bc, v258_data);
          tensorforge::fmacdpp16<14>(v711_acc, v715_bc, v142_data);
          tensorforge::fmacdpp16<14>(v712_acc, v715_bc, v259_data);
          tensorforge::fmacdpp16<15>(v711_acc, v715_bc, v143_data);
          tensorforge::fmacdpp16<15>(v712_acc, v715_bc, v260_data);
          float v716_bc = tensorforge::broadcast<32, 16, 1>(v713_data);
          tensorforge::fmacdpp16<0>(v711_acc, v716_bc, v148_data);
          tensorforge::fmacdpp16<0>(v712_acc, v716_bc, v265_data);
          tensorforge::fmacdpp16<1>(v711_acc, v716_bc, v149_data);
          tensorforge::fmacdpp16<1>(v712_acc, v716_bc, v266_data);
          tensorforge::fmacdpp16<2>(v711_acc, v716_bc, v150_data);
          tensorforge::fmacdpp16<2>(v712_acc, v716_bc, v267_data);
          tensorforge::fmacdpp16<3>(v711_acc, v716_bc, v151_data);
          tensorforge::fmacdpp16<3>(v712_acc, v716_bc, v268_data);
          tensorforge::fmacdpp16<4>(v711_acc, v716_bc, v156_data);
          tensorforge::fmacdpp16<4>(v712_acc, v716_bc, v273_data);
          tensorforge::fmacdpp16<5>(v711_acc, v716_bc, v157_data);
          tensorforge::fmacdpp16<5>(v712_acc, v716_bc, v274_data);
          tensorforge::fmacdpp16<6>(v711_acc, v716_bc, v158_data);
          tensorforge::fmacdpp16<6>(v712_acc, v716_bc, v275_data);
          tensorforge::fmacdpp16<7>(v711_acc, v716_bc, v159_data);
          tensorforge::fmacdpp16<7>(v712_acc, v716_bc, v276_data);
          tensorforge::fmacdpp16<8>(v711_acc, v716_bc, v164_data);
          tensorforge::fmacdpp16<8>(v712_acc, v716_bc, v281_data);
          tensorforge::fmacdpp16<9>(v711_acc, v716_bc, v165_data);
          tensorforge::fmacdpp16<9>(v712_acc, v716_bc, v282_data);
          tensorforge::fmacdpp16<10>(v711_acc, v716_bc, v166_data);
          tensorforge::fmacdpp16<10>(v712_acc, v716_bc, v283_data);
          tensorforge::fmacdpp16<11>(v711_acc, v716_bc, v167_data);
          tensorforge::fmacdpp16<11>(v712_acc, v716_bc, v284_data);
          tensorforge::fmacdpp16<12>(v711_acc, v716_bc, v172_data);
          tensorforge::fmacdpp16<12>(v712_acc, v716_bc, v289_data);
          tensorforge::fmacdpp16<13>(v711_acc, v716_bc, v173_data);
          tensorforge::fmacdpp16<13>(v712_acc, v716_bc, v290_data);
          tensorforge::fmacdpp16<14>(v711_acc, v716_bc, v174_data);
          tensorforge::fmacdpp16<14>(v712_acc, v716_bc, v291_data);
          tensorforge::fmacdpp16<15>(v711_acc, v716_bc, v175_data);
          tensorforge::fmacdpp16<15>(v712_acc, v716_bc, v292_data);
          float v717_bc = tensorforge::broadcast<32, 16, 0>(v714_data);
          tensorforge::fmacdpp16<0>(v711_acc, v717_bc, v180_data);
          tensorforge::fmacdpp16<0>(v712_acc, v717_bc, v297_data);
          tensorforge::fmacdpp16<1>(v711_acc, v717_bc, v181_data);
          tensorforge::fmacdpp16<1>(v712_acc, v717_bc, v298_data);
          tensorforge::fmacdpp16<2>(v711_acc, v717_bc, v182_data);
          tensorforge::fmacdpp16<2>(v712_acc, v717_bc, v299_data);
          tensorforge::fmacdpp16<3>(v711_acc, v717_bc, v183_data);
          tensorforge::fmacdpp16<3>(v712_acc, v717_bc, v300_data);
          tensorforge::fmacdpp16<4>(v711_acc, v717_bc, v188_data);
          tensorforge::fmacdpp16<4>(v712_acc, v717_bc, v305_data);
          tensorforge::fmacdpp16<5>(v711_acc, v717_bc, v189_data);
          tensorforge::fmacdpp16<5>(v712_acc, v717_bc, v306_data);
          tensorforge::fmacdpp16<6>(v711_acc, v717_bc, v190_data);
          tensorforge::fmacdpp16<6>(v712_acc, v717_bc, v307_data);
          tensorforge::fmacdpp16<7>(v711_acc, v717_bc, v191_data);
          tensorforge::fmacdpp16<7>(v712_acc, v717_bc, v308_data);
          tensorforge::fmacdpp16<8>(v711_acc, v717_bc, v196_data);
          tensorforge::fmacdpp16<8>(v712_acc, v717_bc, v313_data);
          tensorforge::fmacdpp16<9>(v711_acc, v717_bc, v197_data);
          tensorforge::fmacdpp16<9>(v712_acc, v717_bc, v314_data);
          tensorforge::fmacdpp16<10>(v711_acc, v717_bc, v198_data);
          tensorforge::fmacdpp16<10>(v712_acc, v717_bc, v315_data);
          tensorforge::fmacdpp16<11>(v711_acc, v717_bc, v199_data);
          tensorforge::fmacdpp16<11>(v712_acc, v717_bc, v316_data);
          tensorforge::fmacdpp16<12>(v711_acc, v717_bc, v204_data);
          tensorforge::fmacdpp16<12>(v712_acc, v717_bc, v321_data);
          tensorforge::fmacdpp16<13>(v711_acc, v717_bc, v205_data);
          tensorforge::fmacdpp16<13>(v712_acc, v717_bc, v322_data);
          tensorforge::fmacdpp16<14>(v711_acc, v717_bc, v206_data);
          tensorforge::fmacdpp16<14>(v712_acc, v717_bc, v323_data);
          tensorforge::fmacdpp16<15>(v711_acc, v717_bc, v207_data);
          tensorforge::fmacdpp16<15>(v712_acc, v717_bc, v324_data);
          float v718_bc = tensorforge::broadcast<32, 16, 1>(v714_data);
          tensorforge::fmacdpp16<0>(v711_acc, v718_bc, v212_data);
          tensorforge::fmacdpp16<0>(v712_acc, v718_bc, v329_data);
          tensorforge::fmacdpp16<1>(v711_acc, v718_bc, v213_data);
          tensorforge::fmacdpp16<1>(v712_acc, v718_bc, v330_data);
          tensorforge::fmacdpp16<2>(v711_acc, v718_bc, v214_data);
          tensorforge::fmacdpp16<2>(v712_acc, v718_bc, v331_data);
          tensorforge::fmacdpp16<3>(v711_acc, v718_bc, v215_data);
          tensorforge::fmacdpp16<3>(v712_acc, v718_bc, v332_data);
          tensorforge::fmacdpp16<4>(v711_acc, v718_bc, v220_data);
          tensorforge::fmacdpp16<4>(v712_acc, v718_bc, v337_data);
          tensorforge::fmacdpp16<5>(v711_acc, v718_bc, v221_data);
          tensorforge::fmacdpp16<5>(v712_acc, v718_bc, v338_data);
          tensorforge::fmacdpp16<6>(v711_acc, v718_bc, v222_data);
          tensorforge::fmacdpp16<6>(v712_acc, v718_bc, v339_data);
          tensorforge::fmacdpp16<7>(v711_acc, v718_bc, v223_data);
          tensorforge::fmacdpp16<7>(v712_acc, v718_bc, v340_data);
          ir4[16] = v711_acc;
          ir4[17] = v712_acc;
          // glb_m2 = store{r>g}(r4);
          int32_t v721_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v722_i0 = 0; v722_i0 < 1; ++v722_i0) {
            int32_t v731_lead = v721_lead + (v722_i0 * 32);
            #pragma unroll
            for (int32_t v723_i1 = 0; v723_i1 < 9; ++v723_i1) {
              int32_t v725_a = v722_i0 + (v723_i1 * 2);
              float v726_data = r4[v725_a];
              int32_t v733_a = v731_lead + (v723_i1 * 56);
              glb_m2[v733_a] = v726_data;
            }
          }
          if (v721_lead < 24) {
            int32_t v743_lead = v721_lead + 32_i32;
            #pragma unroll
            for (int32_t v735_i1 = 0; v735_i1 < 9; ++v735_i1) {
              int32_t v737_a = 1 + (v735_i1 * 2);
              float v738_data = r4[v737_a];
              int32_t v745_a = v743_lead + (v735_i1 * 56);
              glb_m2[v745_a] = v738_data;
            }
          }
          ;
        }
      }
    }
  }
}

