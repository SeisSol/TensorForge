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
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v9_lead = v2_lead + (v3_i0 * 32);
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 13; ++v4_i1) {
              int32_t v11_a = v9_lead + (v4_i1 * 32);
              float v12_data;
              {
                v12_data = glb_m0[v11_a];
              }
              int32_t v13_a = v3_i0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          int32_t v16_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
            int32_t v23_lead = v16_lead + (v17_i0 * 32);
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              int32_t v25_a = v23_lead + (v18_i1 * 32);
              float v26_data;
              {
                v26_data = __builtin_nontemporal_load(&glb_m1[v25_a]);
              }
              int32_t v27_a = v17_i0 + v18_i1;
              r2[v27_a] = v26_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          auto& ir1 = r1;
          float v31_data = r0[0];
          float v32_data = ir1[0];
          ir1[0] = (v32_data + v31_data);
          float v34_data = r0[1];
          float v35_data = ir1[1];
          ir1[1] = (v35_data + v34_data);
          float v37_data = r0[2];
          float v38_data = ir1[2];
          ir1[2] = (v38_data + v37_data);
          float v40_data = r0[3];
          float v41_data = ir1[3];
          ir1[3] = (v41_data + v40_data);
          float v43_data = r0[4];
          float v44_data = ir1[4];
          ir1[4] = (v44_data + v43_data);
          float v46_data = r0[5];
          float v47_data = ir1[5];
          ir1[5] = (v47_data + v46_data);
          float v49_data = r0[6];
          float v50_data = ir1[6];
          ir1[6] = (v50_data + v49_data);
          float v52_data = r0[7];
          float v53_data = ir1[7];
          ir1[7] = (v53_data + v52_data);
          float v55_data = r0[8];
          float v56_data = ir1[8];
          ir1[8] = (v56_data + v55_data);
          float v58_data = r0[9];
          float v59_data = ir1[9];
          ir1[9] = (v59_data + v58_data);
          float v61_data = r0[10];
          float v62_data = ir1[10];
          ir1[10] = (v62_data + v61_data);
          float v64_data = r0[11];
          float v65_data = ir1[11];
          ir1[11] = (v65_data + v64_data);
          float v67_data = r0[12];
          float v68_data = ir1[12];
          ir1[12] = (v68_data + v67_data);
          float r3[13]{};
          {
            // r3 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r3[0] = v0;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r3[1] = v32;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r3[2] = v64;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r3[3] = v96;
            float v128 = glb_m2[128 + threadIdx.x * 1];
            r3[4] = v128;
          }
          // wait(r2 = load{g>r}(glb_m1););
          // wait(r3 = load{g>r}(glb_m2););
          float r4[13]{};
          {
            // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 32), (0, 13)] [(0, 12)]
            float ir4[13]{};
            float v70_data = r2[0];
            float v71_data = r2[1];
            float v72_data = r2[2];
            float v73_data = r2[3];
            float v74_data = r2[4];
            float v75_data = r2[5];
            float v76_data = r2[6];
            float v77_data = r2[7];
            float v78_data = r2[8];
            float v79_data = r2[9];
            float v80_data = r2[10];
            float v81_data = r2[11];
            float v82_acc{};
            float v83_acc{};
            float v84_acc{};
            float v85_acc{};
            float v86_acc{};
            float v87_acc{};
            float v88_acc{};
            float v89_acc{};
            float v90_acc{};
            float v91_acc{};
            float v92_acc{};
            float v93_acc{};
            float v94_acc{};
            float v95_lin = r3[0];
            float v96_bc = tensorforge::broadcast<32, 16, 0>(v95_lin);
            tensorforge::fmacdpp16<0>(v82_acc, v96_bc, v70_data);
            tensorforge::fmacdpp16<1>(v82_acc, v96_bc, v71_data);
            tensorforge::fmacdpp16<2>(v82_acc, v96_bc, v72_data);
            tensorforge::fmacdpp16<3>(v82_acc, v96_bc, v73_data);
            tensorforge::fmacdpp16<4>(v82_acc, v96_bc, v74_data);
            tensorforge::fmacdpp16<5>(v82_acc, v96_bc, v75_data);
            tensorforge::fmacdpp16<6>(v82_acc, v96_bc, v76_data);
            tensorforge::fmacdpp16<7>(v82_acc, v96_bc, v77_data);
            tensorforge::fmacdpp16<8>(v82_acc, v96_bc, v78_data);
            tensorforge::fmacdpp16<9>(v82_acc, v96_bc, v79_data);
            tensorforge::fmacdpp16<10>(v82_acc, v96_bc, v80_data);
            tensorforge::fmacdpp16<11>(v82_acc, v96_bc, v81_data);
            tensorforge::fmacdpp16<12>(v83_acc, v96_bc, v70_data);
            tensorforge::fmacdpp16<13>(v83_acc, v96_bc, v71_data);
            tensorforge::fmacdpp16<14>(v83_acc, v96_bc, v72_data);
            tensorforge::fmacdpp16<15>(v83_acc, v96_bc, v73_data);
            float v97_bc = tensorforge::broadcast<32, 16, 1>(v95_lin);
            tensorforge::fmacdpp16<0>(v83_acc, v97_bc, v74_data);
            tensorforge::fmacdpp16<1>(v83_acc, v97_bc, v75_data);
            tensorforge::fmacdpp16<2>(v83_acc, v97_bc, v76_data);
            tensorforge::fmacdpp16<3>(v83_acc, v97_bc, v77_data);
            tensorforge::fmacdpp16<4>(v83_acc, v97_bc, v78_data);
            tensorforge::fmacdpp16<5>(v83_acc, v97_bc, v79_data);
            tensorforge::fmacdpp16<6>(v83_acc, v97_bc, v80_data);
            tensorforge::fmacdpp16<7>(v83_acc, v97_bc, v81_data);
            tensorforge::fmacdpp16<8>(v84_acc, v97_bc, v70_data);
            tensorforge::fmacdpp16<9>(v84_acc, v97_bc, v71_data);
            tensorforge::fmacdpp16<10>(v84_acc, v97_bc, v72_data);
            tensorforge::fmacdpp16<11>(v84_acc, v97_bc, v73_data);
            tensorforge::fmacdpp16<12>(v84_acc, v97_bc, v74_data);
            tensorforge::fmacdpp16<13>(v84_acc, v97_bc, v75_data);
            tensorforge::fmacdpp16<14>(v84_acc, v97_bc, v76_data);
            tensorforge::fmacdpp16<15>(v84_acc, v97_bc, v77_data);
            float v98_lin = r3[1];
            float v99_bc = tensorforge::broadcast<32, 16, 0>(v98_lin);
            tensorforge::fmacdpp16<0>(v84_acc, v99_bc, v78_data);
            tensorforge::fmacdpp16<1>(v84_acc, v99_bc, v79_data);
            tensorforge::fmacdpp16<2>(v84_acc, v99_bc, v80_data);
            tensorforge::fmacdpp16<3>(v84_acc, v99_bc, v81_data);
            tensorforge::fmacdpp16<4>(v85_acc, v99_bc, v70_data);
            tensorforge::fmacdpp16<5>(v85_acc, v99_bc, v71_data);
            tensorforge::fmacdpp16<6>(v85_acc, v99_bc, v72_data);
            tensorforge::fmacdpp16<7>(v85_acc, v99_bc, v73_data);
            tensorforge::fmacdpp16<8>(v85_acc, v99_bc, v74_data);
            tensorforge::fmacdpp16<9>(v85_acc, v99_bc, v75_data);
            tensorforge::fmacdpp16<10>(v85_acc, v99_bc, v76_data);
            tensorforge::fmacdpp16<11>(v85_acc, v99_bc, v77_data);
            tensorforge::fmacdpp16<12>(v85_acc, v99_bc, v78_data);
            tensorforge::fmacdpp16<13>(v85_acc, v99_bc, v79_data);
            tensorforge::fmacdpp16<14>(v85_acc, v99_bc, v80_data);
            tensorforge::fmacdpp16<15>(v85_acc, v99_bc, v81_data);
            float v100_bc = tensorforge::broadcast<32, 16, 1>(v98_lin);
            tensorforge::fmacdpp16<0>(v86_acc, v100_bc, v70_data);
            tensorforge::fmacdpp16<1>(v86_acc, v100_bc, v71_data);
            tensorforge::fmacdpp16<2>(v86_acc, v100_bc, v72_data);
            tensorforge::fmacdpp16<3>(v86_acc, v100_bc, v73_data);
            tensorforge::fmacdpp16<4>(v86_acc, v100_bc, v74_data);
            tensorforge::fmacdpp16<5>(v86_acc, v100_bc, v75_data);
            tensorforge::fmacdpp16<6>(v86_acc, v100_bc, v76_data);
            tensorforge::fmacdpp16<7>(v86_acc, v100_bc, v77_data);
            tensorforge::fmacdpp16<8>(v86_acc, v100_bc, v78_data);
            tensorforge::fmacdpp16<9>(v86_acc, v100_bc, v79_data);
            tensorforge::fmacdpp16<10>(v86_acc, v100_bc, v80_data);
            tensorforge::fmacdpp16<11>(v86_acc, v100_bc, v81_data);
            tensorforge::fmacdpp16<12>(v87_acc, v100_bc, v70_data);
            tensorforge::fmacdpp16<13>(v87_acc, v100_bc, v71_data);
            tensorforge::fmacdpp16<14>(v87_acc, v100_bc, v72_data);
            tensorforge::fmacdpp16<15>(v87_acc, v100_bc, v73_data);
            float v101_lin = r3[2];
            float v102_bc = tensorforge::broadcast<32, 16, 0>(v101_lin);
            tensorforge::fmacdpp16<0>(v87_acc, v102_bc, v74_data);
            tensorforge::fmacdpp16<1>(v87_acc, v102_bc, v75_data);
            tensorforge::fmacdpp16<2>(v87_acc, v102_bc, v76_data);
            tensorforge::fmacdpp16<3>(v87_acc, v102_bc, v77_data);
            tensorforge::fmacdpp16<4>(v87_acc, v102_bc, v78_data);
            tensorforge::fmacdpp16<5>(v87_acc, v102_bc, v79_data);
            tensorforge::fmacdpp16<6>(v87_acc, v102_bc, v80_data);
            tensorforge::fmacdpp16<7>(v87_acc, v102_bc, v81_data);
            tensorforge::fmacdpp16<8>(v88_acc, v102_bc, v70_data);
            tensorforge::fmacdpp16<9>(v88_acc, v102_bc, v71_data);
            tensorforge::fmacdpp16<10>(v88_acc, v102_bc, v72_data);
            tensorforge::fmacdpp16<11>(v88_acc, v102_bc, v73_data);
            tensorforge::fmacdpp16<12>(v88_acc, v102_bc, v74_data);
            tensorforge::fmacdpp16<13>(v88_acc, v102_bc, v75_data);
            tensorforge::fmacdpp16<14>(v88_acc, v102_bc, v76_data);
            tensorforge::fmacdpp16<15>(v88_acc, v102_bc, v77_data);
            float v103_bc = tensorforge::broadcast<32, 16, 1>(v101_lin);
            tensorforge::fmacdpp16<0>(v88_acc, v103_bc, v78_data);
            tensorforge::fmacdpp16<1>(v88_acc, v103_bc, v79_data);
            tensorforge::fmacdpp16<2>(v88_acc, v103_bc, v80_data);
            tensorforge::fmacdpp16<3>(v88_acc, v103_bc, v81_data);
            tensorforge::fmacdpp16<4>(v89_acc, v103_bc, v70_data);
            tensorforge::fmacdpp16<5>(v89_acc, v103_bc, v71_data);
            tensorforge::fmacdpp16<6>(v89_acc, v103_bc, v72_data);
            tensorforge::fmacdpp16<7>(v89_acc, v103_bc, v73_data);
            tensorforge::fmacdpp16<8>(v89_acc, v103_bc, v74_data);
            tensorforge::fmacdpp16<9>(v89_acc, v103_bc, v75_data);
            tensorforge::fmacdpp16<10>(v89_acc, v103_bc, v76_data);
            tensorforge::fmacdpp16<11>(v89_acc, v103_bc, v77_data);
            tensorforge::fmacdpp16<12>(v89_acc, v103_bc, v78_data);
            tensorforge::fmacdpp16<13>(v89_acc, v103_bc, v79_data);
            tensorforge::fmacdpp16<14>(v89_acc, v103_bc, v80_data);
            tensorforge::fmacdpp16<15>(v89_acc, v103_bc, v81_data);
            float v104_lin = r3[3];
            float v105_bc = tensorforge::broadcast<32, 16, 0>(v104_lin);
            tensorforge::fmacdpp16<0>(v90_acc, v105_bc, v70_data);
            tensorforge::fmacdpp16<1>(v90_acc, v105_bc, v71_data);
            tensorforge::fmacdpp16<2>(v90_acc, v105_bc, v72_data);
            tensorforge::fmacdpp16<3>(v90_acc, v105_bc, v73_data);
            tensorforge::fmacdpp16<4>(v90_acc, v105_bc, v74_data);
            tensorforge::fmacdpp16<5>(v90_acc, v105_bc, v75_data);
            tensorforge::fmacdpp16<6>(v90_acc, v105_bc, v76_data);
            tensorforge::fmacdpp16<7>(v90_acc, v105_bc, v77_data);
            tensorforge::fmacdpp16<8>(v90_acc, v105_bc, v78_data);
            tensorforge::fmacdpp16<9>(v90_acc, v105_bc, v79_data);
            tensorforge::fmacdpp16<10>(v90_acc, v105_bc, v80_data);
            tensorforge::fmacdpp16<11>(v90_acc, v105_bc, v81_data);
            tensorforge::fmacdpp16<12>(v91_acc, v105_bc, v70_data);
            tensorforge::fmacdpp16<13>(v91_acc, v105_bc, v71_data);
            tensorforge::fmacdpp16<14>(v91_acc, v105_bc, v72_data);
            tensorforge::fmacdpp16<15>(v91_acc, v105_bc, v73_data);
            float v106_bc = tensorforge::broadcast<32, 16, 1>(v104_lin);
            tensorforge::fmacdpp16<0>(v91_acc, v106_bc, v74_data);
            tensorforge::fmacdpp16<1>(v91_acc, v106_bc, v75_data);
            tensorforge::fmacdpp16<2>(v91_acc, v106_bc, v76_data);
            tensorforge::fmacdpp16<3>(v91_acc, v106_bc, v77_data);
            tensorforge::fmacdpp16<4>(v91_acc, v106_bc, v78_data);
            tensorforge::fmacdpp16<5>(v91_acc, v106_bc, v79_data);
            tensorforge::fmacdpp16<6>(v91_acc, v106_bc, v80_data);
            tensorforge::fmacdpp16<7>(v91_acc, v106_bc, v81_data);
            tensorforge::fmacdpp16<8>(v92_acc, v106_bc, v70_data);
            tensorforge::fmacdpp16<9>(v92_acc, v106_bc, v71_data);
            tensorforge::fmacdpp16<10>(v92_acc, v106_bc, v72_data);
            tensorforge::fmacdpp16<11>(v92_acc, v106_bc, v73_data);
            tensorforge::fmacdpp16<12>(v92_acc, v106_bc, v74_data);
            tensorforge::fmacdpp16<13>(v92_acc, v106_bc, v75_data);
            tensorforge::fmacdpp16<14>(v92_acc, v106_bc, v76_data);
            tensorforge::fmacdpp16<15>(v92_acc, v106_bc, v77_data);
            float v107_lin = r3[4];
            float v108_bc = tensorforge::broadcast<32, 16, 0>(v107_lin);
            tensorforge::fmacdpp16<0>(v92_acc, v108_bc, v78_data);
            tensorforge::fmacdpp16<1>(v92_acc, v108_bc, v79_data);
            tensorforge::fmacdpp16<2>(v92_acc, v108_bc, v80_data);
            tensorforge::fmacdpp16<3>(v92_acc, v108_bc, v81_data);
            tensorforge::fmacdpp16<4>(v93_acc, v108_bc, v70_data);
            tensorforge::fmacdpp16<5>(v93_acc, v108_bc, v71_data);
            tensorforge::fmacdpp16<6>(v93_acc, v108_bc, v72_data);
            tensorforge::fmacdpp16<7>(v93_acc, v108_bc, v73_data);
            tensorforge::fmacdpp16<8>(v93_acc, v108_bc, v74_data);
            tensorforge::fmacdpp16<9>(v93_acc, v108_bc, v75_data);
            tensorforge::fmacdpp16<10>(v93_acc, v108_bc, v76_data);
            tensorforge::fmacdpp16<11>(v93_acc, v108_bc, v77_data);
            tensorforge::fmacdpp16<12>(v93_acc, v108_bc, v78_data);
            tensorforge::fmacdpp16<13>(v93_acc, v108_bc, v79_data);
            tensorforge::fmacdpp16<14>(v93_acc, v108_bc, v80_data);
            tensorforge::fmacdpp16<15>(v93_acc, v108_bc, v81_data);
            float v109_bc = tensorforge::broadcast<32, 16, 1>(v107_lin);
            tensorforge::fmacdpp16<0>(v94_acc, v109_bc, v70_data);
            tensorforge::fmacdpp16<1>(v94_acc, v109_bc, v71_data);
            tensorforge::fmacdpp16<2>(v94_acc, v109_bc, v72_data);
            tensorforge::fmacdpp16<3>(v94_acc, v109_bc, v73_data);
            tensorforge::fmacdpp16<4>(v94_acc, v109_bc, v74_data);
            tensorforge::fmacdpp16<5>(v94_acc, v109_bc, v75_data);
            tensorforge::fmacdpp16<6>(v94_acc, v109_bc, v76_data);
            tensorforge::fmacdpp16<7>(v94_acc, v109_bc, v77_data);
            tensorforge::fmacdpp16<8>(v94_acc, v109_bc, v78_data);
            tensorforge::fmacdpp16<9>(v94_acc, v109_bc, v79_data);
            tensorforge::fmacdpp16<10>(v94_acc, v109_bc, v80_data);
            tensorforge::fmacdpp16<11>(v94_acc, v109_bc, v81_data);
            ir4[0] = v82_acc;
            ir4[1] = v83_acc;
            ir4[2] = v84_acc;
            ir4[3] = v85_acc;
            ir4[4] = v86_acc;
            ir4[5] = v87_acc;
            ir4[6] = v88_acc;
            ir4[7] = v89_acc;
            ir4[8] = v90_acc;
            ir4[9] = v91_acc;
            ir4[10] = v92_acc;
            ir4[11] = v93_acc;
            ir4[12] = v94_acc;
            #pragma unroll
            for (int32_t v113_n0 = 0; v113_n0 < 1; ++v113_n0) {
              #pragma unroll
              for (int32_t v114_n1 = 0; v114_n1 < 13; ++v114_n1) {
                int32_t v115_a = v113_n0 + v114_n1;
                float v116_data = ir4[v115_a];
                int32_t v117_a = v113_n0 + v114_n1;
                float v118_data = r1[v117_a];
                int32_t v120_a = v113_n0 + v114_n1;
                r4[v120_a] = (v118_data + v116_data);
              }
            }
          }
          float r5[1]{};
          // r5 = +(r4) + None
          // [(0, 32), (0, 1)] []
          auto& ir5 = r5;
          float v124_data = r4[4];
          float v125_data = ir5[0];
          ir5[0] = (v125_data + v124_data);
          // glb_m0 = store{r>g}(r5);
          int32_t v129_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v130_i0 = 0; v130_i0 < 1; ++v130_i0) {
            int32_t v138_lead = v129_lead + (v130_i0 * 32);
            #pragma unroll
            for (int32_t v131_i1 = 0; v131_i1 < 1; ++v131_i1) {
              int32_t v132_a = v130_i0 + v131_i1;
              float v133_data = r5[v132_a];
              int32_t v141_a = v138_lead + ((v131_i1 + 4) * 32);
              glb_m0[v141_a] = v133_data;
            }
          }
          float r6[13]{};
          // r6 = load{g>r}(glb_m0);
          int32_t v144_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v145_i0 = 0; v145_i0 < 1; ++v145_i0) {
            int32_t v151_lead = v144_lead + (v145_i0 * 32);
            #pragma unroll
            for (int32_t v146_i1 = 0; v146_i1 < 13; ++v146_i1) {
              int32_t v153_a = v151_lead + (v146_i1 * 32);
              float v154_data;
              {
                v154_data = glb_m0[v153_a];
              }
              int32_t v155_a = v145_i0 + v146_i1;
              r6[v155_a] = v154_data;
            }
          }
          float r7[13]{};
          {
            // r7 = load{g>r}(glb_m4);
            float v0 = glb_m4[0 + threadIdx.x * 1];
            r7[0] = v0;
            float v32 = glb_m4[32 + threadIdx.x * 1];
            r7[1] = v32;
            float v64 = glb_m4[64 + threadIdx.x * 1];
            r7[2] = v64;
            float v96 = glb_m4[96 + threadIdx.x * 1];
            r7[3] = v96;
            float v128 = glb_m4[128 + threadIdx.x * 1];
            r7[4] = v128;
            float v160 = glb_m4[160 + threadIdx.x * 1];
            r7[5] = v160;
          }
          // wait(r6 = load{g>r}(glb_m0););
          // wait(r7 = load{g>r}(glb_m4););
          float r8[13]{};
          // r8 = +(r6 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          auto& ir8 = r8;
          float v156_data = r6[0];
          float v157_data = r6[1];
          float v158_data = r6[2];
          float v159_data = r6[3];
          float v160_data = r6[4];
          float v161_data = r6[5];
          float v162_data = r6[6];
          float v163_data = r6[7];
          float v164_data = r6[8];
          float v165_data = r6[9];
          float v166_data = r6[10];
          float v167_data = r6[11];
          float v168_data = r6[12];
          float v169_acc{};
          float v170_acc{};
          float v171_acc{};
          float v172_acc{};
          float v173_acc{};
          float v174_acc{};
          float v175_acc{};
          float v176_acc{};
          float v177_acc{};
          float v178_acc{};
          float v179_acc{};
          float v180_acc{};
          float v181_acc{};
          float v182_lin = r7[0];
          float v183_bc = tensorforge::broadcast<32, 16, 0>(v182_lin);
          tensorforge::fmacdpp16<0>(v169_acc, v183_bc, v156_data);
          tensorforge::fmacdpp16<1>(v169_acc, v183_bc, v157_data);
          tensorforge::fmacdpp16<2>(v169_acc, v183_bc, v158_data);
          tensorforge::fmacdpp16<3>(v169_acc, v183_bc, v159_data);
          tensorforge::fmacdpp16<4>(v169_acc, v183_bc, v160_data);
          tensorforge::fmacdpp16<5>(v169_acc, v183_bc, v161_data);
          tensorforge::fmacdpp16<6>(v169_acc, v183_bc, v162_data);
          tensorforge::fmacdpp16<7>(v169_acc, v183_bc, v163_data);
          tensorforge::fmacdpp16<8>(v169_acc, v183_bc, v164_data);
          tensorforge::fmacdpp16<9>(v169_acc, v183_bc, v165_data);
          tensorforge::fmacdpp16<10>(v169_acc, v183_bc, v166_data);
          tensorforge::fmacdpp16<11>(v169_acc, v183_bc, v167_data);
          tensorforge::fmacdpp16<12>(v169_acc, v183_bc, v168_data);
          tensorforge::fmacdpp16<13>(v170_acc, v183_bc, v156_data);
          tensorforge::fmacdpp16<14>(v170_acc, v183_bc, v157_data);
          tensorforge::fmacdpp16<15>(v170_acc, v183_bc, v158_data);
          float v184_bc = tensorforge::broadcast<32, 16, 1>(v182_lin);
          tensorforge::fmacdpp16<0>(v170_acc, v184_bc, v159_data);
          tensorforge::fmacdpp16<1>(v170_acc, v184_bc, v160_data);
          tensorforge::fmacdpp16<2>(v170_acc, v184_bc, v161_data);
          tensorforge::fmacdpp16<3>(v170_acc, v184_bc, v162_data);
          tensorforge::fmacdpp16<4>(v170_acc, v184_bc, v163_data);
          tensorforge::fmacdpp16<5>(v170_acc, v184_bc, v164_data);
          tensorforge::fmacdpp16<6>(v170_acc, v184_bc, v165_data);
          tensorforge::fmacdpp16<7>(v170_acc, v184_bc, v166_data);
          tensorforge::fmacdpp16<8>(v170_acc, v184_bc, v167_data);
          tensorforge::fmacdpp16<9>(v170_acc, v184_bc, v168_data);
          tensorforge::fmacdpp16<10>(v171_acc, v184_bc, v156_data);
          tensorforge::fmacdpp16<11>(v171_acc, v184_bc, v157_data);
          tensorforge::fmacdpp16<12>(v171_acc, v184_bc, v158_data);
          tensorforge::fmacdpp16<13>(v171_acc, v184_bc, v159_data);
          tensorforge::fmacdpp16<14>(v171_acc, v184_bc, v160_data);
          tensorforge::fmacdpp16<15>(v171_acc, v184_bc, v161_data);
          float v185_lin = r7[1];
          float v186_bc = tensorforge::broadcast<32, 16, 0>(v185_lin);
          tensorforge::fmacdpp16<0>(v171_acc, v186_bc, v162_data);
          tensorforge::fmacdpp16<1>(v171_acc, v186_bc, v163_data);
          tensorforge::fmacdpp16<2>(v171_acc, v186_bc, v164_data);
          tensorforge::fmacdpp16<3>(v171_acc, v186_bc, v165_data);
          tensorforge::fmacdpp16<4>(v171_acc, v186_bc, v166_data);
          tensorforge::fmacdpp16<5>(v171_acc, v186_bc, v167_data);
          tensorforge::fmacdpp16<6>(v171_acc, v186_bc, v168_data);
          tensorforge::fmacdpp16<7>(v172_acc, v186_bc, v156_data);
          tensorforge::fmacdpp16<8>(v172_acc, v186_bc, v157_data);
          tensorforge::fmacdpp16<9>(v172_acc, v186_bc, v158_data);
          tensorforge::fmacdpp16<10>(v172_acc, v186_bc, v159_data);
          tensorforge::fmacdpp16<11>(v172_acc, v186_bc, v160_data);
          tensorforge::fmacdpp16<12>(v172_acc, v186_bc, v161_data);
          tensorforge::fmacdpp16<13>(v172_acc, v186_bc, v162_data);
          tensorforge::fmacdpp16<14>(v172_acc, v186_bc, v163_data);
          tensorforge::fmacdpp16<15>(v172_acc, v186_bc, v164_data);
          float v187_bc = tensorforge::broadcast<32, 16, 1>(v185_lin);
          tensorforge::fmacdpp16<0>(v172_acc, v187_bc, v165_data);
          tensorforge::fmacdpp16<1>(v172_acc, v187_bc, v166_data);
          tensorforge::fmacdpp16<2>(v172_acc, v187_bc, v167_data);
          tensorforge::fmacdpp16<3>(v172_acc, v187_bc, v168_data);
          tensorforge::fmacdpp16<4>(v173_acc, v187_bc, v156_data);
          tensorforge::fmacdpp16<5>(v173_acc, v187_bc, v157_data);
          tensorforge::fmacdpp16<6>(v173_acc, v187_bc, v158_data);
          tensorforge::fmacdpp16<7>(v173_acc, v187_bc, v159_data);
          tensorforge::fmacdpp16<8>(v173_acc, v187_bc, v160_data);
          tensorforge::fmacdpp16<9>(v173_acc, v187_bc, v161_data);
          tensorforge::fmacdpp16<10>(v173_acc, v187_bc, v162_data);
          tensorforge::fmacdpp16<11>(v173_acc, v187_bc, v163_data);
          tensorforge::fmacdpp16<12>(v173_acc, v187_bc, v164_data);
          tensorforge::fmacdpp16<13>(v173_acc, v187_bc, v165_data);
          tensorforge::fmacdpp16<14>(v173_acc, v187_bc, v166_data);
          tensorforge::fmacdpp16<15>(v173_acc, v187_bc, v167_data);
          float v188_lin = r7[2];
          float v189_bc = tensorforge::broadcast<32, 16, 0>(v188_lin);
          tensorforge::fmacdpp16<0>(v173_acc, v189_bc, v168_data);
          tensorforge::fmacdpp16<1>(v174_acc, v189_bc, v156_data);
          tensorforge::fmacdpp16<2>(v174_acc, v189_bc, v157_data);
          tensorforge::fmacdpp16<3>(v174_acc, v189_bc, v158_data);
          tensorforge::fmacdpp16<4>(v174_acc, v189_bc, v159_data);
          tensorforge::fmacdpp16<5>(v174_acc, v189_bc, v160_data);
          tensorforge::fmacdpp16<6>(v174_acc, v189_bc, v161_data);
          tensorforge::fmacdpp16<7>(v174_acc, v189_bc, v162_data);
          tensorforge::fmacdpp16<8>(v174_acc, v189_bc, v163_data);
          tensorforge::fmacdpp16<9>(v174_acc, v189_bc, v164_data);
          tensorforge::fmacdpp16<10>(v174_acc, v189_bc, v165_data);
          tensorforge::fmacdpp16<11>(v174_acc, v189_bc, v166_data);
          tensorforge::fmacdpp16<12>(v174_acc, v189_bc, v167_data);
          tensorforge::fmacdpp16<13>(v174_acc, v189_bc, v168_data);
          tensorforge::fmacdpp16<14>(v175_acc, v189_bc, v156_data);
          tensorforge::fmacdpp16<15>(v175_acc, v189_bc, v157_data);
          float v190_bc = tensorforge::broadcast<32, 16, 1>(v188_lin);
          tensorforge::fmacdpp16<0>(v175_acc, v190_bc, v158_data);
          tensorforge::fmacdpp16<1>(v175_acc, v190_bc, v159_data);
          tensorforge::fmacdpp16<2>(v175_acc, v190_bc, v160_data);
          tensorforge::fmacdpp16<3>(v175_acc, v190_bc, v161_data);
          tensorforge::fmacdpp16<4>(v175_acc, v190_bc, v162_data);
          tensorforge::fmacdpp16<5>(v175_acc, v190_bc, v163_data);
          tensorforge::fmacdpp16<6>(v175_acc, v190_bc, v164_data);
          tensorforge::fmacdpp16<7>(v175_acc, v190_bc, v165_data);
          tensorforge::fmacdpp16<8>(v175_acc, v190_bc, v166_data);
          tensorforge::fmacdpp16<9>(v175_acc, v190_bc, v167_data);
          tensorforge::fmacdpp16<10>(v175_acc, v190_bc, v168_data);
          tensorforge::fmacdpp16<11>(v176_acc, v190_bc, v156_data);
          tensorforge::fmacdpp16<12>(v176_acc, v190_bc, v157_data);
          tensorforge::fmacdpp16<13>(v176_acc, v190_bc, v158_data);
          tensorforge::fmacdpp16<14>(v176_acc, v190_bc, v159_data);
          tensorforge::fmacdpp16<15>(v176_acc, v190_bc, v160_data);
          float v191_lin = r7[3];
          float v192_bc = tensorforge::broadcast<32, 16, 0>(v191_lin);
          tensorforge::fmacdpp16<0>(v176_acc, v192_bc, v161_data);
          tensorforge::fmacdpp16<1>(v176_acc, v192_bc, v162_data);
          tensorforge::fmacdpp16<2>(v176_acc, v192_bc, v163_data);
          tensorforge::fmacdpp16<3>(v176_acc, v192_bc, v164_data);
          tensorforge::fmacdpp16<4>(v176_acc, v192_bc, v165_data);
          tensorforge::fmacdpp16<5>(v176_acc, v192_bc, v166_data);
          tensorforge::fmacdpp16<6>(v176_acc, v192_bc, v167_data);
          tensorforge::fmacdpp16<7>(v176_acc, v192_bc, v168_data);
          tensorforge::fmacdpp16<8>(v177_acc, v192_bc, v156_data);
          tensorforge::fmacdpp16<9>(v177_acc, v192_bc, v157_data);
          tensorforge::fmacdpp16<10>(v177_acc, v192_bc, v158_data);
          tensorforge::fmacdpp16<11>(v177_acc, v192_bc, v159_data);
          tensorforge::fmacdpp16<12>(v177_acc, v192_bc, v160_data);
          tensorforge::fmacdpp16<13>(v177_acc, v192_bc, v161_data);
          tensorforge::fmacdpp16<14>(v177_acc, v192_bc, v162_data);
          tensorforge::fmacdpp16<15>(v177_acc, v192_bc, v163_data);
          float v193_bc = tensorforge::broadcast<32, 16, 1>(v191_lin);
          tensorforge::fmacdpp16<0>(v177_acc, v193_bc, v164_data);
          tensorforge::fmacdpp16<1>(v177_acc, v193_bc, v165_data);
          tensorforge::fmacdpp16<2>(v177_acc, v193_bc, v166_data);
          tensorforge::fmacdpp16<3>(v177_acc, v193_bc, v167_data);
          tensorforge::fmacdpp16<4>(v177_acc, v193_bc, v168_data);
          tensorforge::fmacdpp16<5>(v178_acc, v193_bc, v156_data);
          tensorforge::fmacdpp16<6>(v178_acc, v193_bc, v157_data);
          tensorforge::fmacdpp16<7>(v178_acc, v193_bc, v158_data);
          tensorforge::fmacdpp16<8>(v178_acc, v193_bc, v159_data);
          tensorforge::fmacdpp16<9>(v178_acc, v193_bc, v160_data);
          tensorforge::fmacdpp16<10>(v178_acc, v193_bc, v161_data);
          tensorforge::fmacdpp16<11>(v178_acc, v193_bc, v162_data);
          tensorforge::fmacdpp16<12>(v178_acc, v193_bc, v163_data);
          tensorforge::fmacdpp16<13>(v178_acc, v193_bc, v164_data);
          tensorforge::fmacdpp16<14>(v178_acc, v193_bc, v165_data);
          tensorforge::fmacdpp16<15>(v178_acc, v193_bc, v166_data);
          float v194_lin = r7[4];
          float v195_bc = tensorforge::broadcast<32, 16, 0>(v194_lin);
          tensorforge::fmacdpp16<0>(v178_acc, v195_bc, v167_data);
          tensorforge::fmacdpp16<1>(v178_acc, v195_bc, v168_data);
          tensorforge::fmacdpp16<2>(v179_acc, v195_bc, v156_data);
          tensorforge::fmacdpp16<3>(v179_acc, v195_bc, v157_data);
          tensorforge::fmacdpp16<4>(v179_acc, v195_bc, v158_data);
          tensorforge::fmacdpp16<5>(v179_acc, v195_bc, v159_data);
          tensorforge::fmacdpp16<6>(v179_acc, v195_bc, v160_data);
          tensorforge::fmacdpp16<7>(v179_acc, v195_bc, v161_data);
          tensorforge::fmacdpp16<8>(v179_acc, v195_bc, v162_data);
          tensorforge::fmacdpp16<9>(v179_acc, v195_bc, v163_data);
          tensorforge::fmacdpp16<10>(v179_acc, v195_bc, v164_data);
          tensorforge::fmacdpp16<11>(v179_acc, v195_bc, v165_data);
          tensorforge::fmacdpp16<12>(v179_acc, v195_bc, v166_data);
          tensorforge::fmacdpp16<13>(v179_acc, v195_bc, v167_data);
          tensorforge::fmacdpp16<14>(v179_acc, v195_bc, v168_data);
          tensorforge::fmacdpp16<15>(v180_acc, v195_bc, v156_data);
          float v196_bc = tensorforge::broadcast<32, 16, 1>(v194_lin);
          tensorforge::fmacdpp16<0>(v180_acc, v196_bc, v157_data);
          tensorforge::fmacdpp16<1>(v180_acc, v196_bc, v158_data);
          tensorforge::fmacdpp16<2>(v180_acc, v196_bc, v159_data);
          tensorforge::fmacdpp16<3>(v180_acc, v196_bc, v160_data);
          tensorforge::fmacdpp16<4>(v180_acc, v196_bc, v161_data);
          tensorforge::fmacdpp16<5>(v180_acc, v196_bc, v162_data);
          tensorforge::fmacdpp16<6>(v180_acc, v196_bc, v163_data);
          tensorforge::fmacdpp16<7>(v180_acc, v196_bc, v164_data);
          tensorforge::fmacdpp16<8>(v180_acc, v196_bc, v165_data);
          tensorforge::fmacdpp16<9>(v180_acc, v196_bc, v166_data);
          tensorforge::fmacdpp16<10>(v180_acc, v196_bc, v167_data);
          tensorforge::fmacdpp16<11>(v180_acc, v196_bc, v168_data);
          tensorforge::fmacdpp16<12>(v181_acc, v196_bc, v156_data);
          tensorforge::fmacdpp16<13>(v181_acc, v196_bc, v157_data);
          tensorforge::fmacdpp16<14>(v181_acc, v196_bc, v158_data);
          tensorforge::fmacdpp16<15>(v181_acc, v196_bc, v159_data);
          float v197_lin = r7[5];
          float v198_bc = tensorforge::broadcast<32, 16, 0>(v197_lin);
          tensorforge::fmacdpp16<0>(v181_acc, v198_bc, v160_data);
          tensorforge::fmacdpp16<1>(v181_acc, v198_bc, v161_data);
          tensorforge::fmacdpp16<2>(v181_acc, v198_bc, v162_data);
          tensorforge::fmacdpp16<3>(v181_acc, v198_bc, v163_data);
          tensorforge::fmacdpp16<4>(v181_acc, v198_bc, v164_data);
          tensorforge::fmacdpp16<5>(v181_acc, v198_bc, v165_data);
          tensorforge::fmacdpp16<6>(v181_acc, v198_bc, v166_data);
          tensorforge::fmacdpp16<7>(v181_acc, v198_bc, v167_data);
          tensorforge::fmacdpp16<8>(v181_acc, v198_bc, v168_data);
          ir8[0] = v169_acc;
          ir8[1] = v170_acc;
          ir8[2] = v171_acc;
          ir8[3] = v172_acc;
          ir8[4] = v173_acc;
          ir8[5] = v174_acc;
          ir8[6] = v175_acc;
          ir8[7] = v176_acc;
          ir8[8] = v177_acc;
          ir8[9] = v178_acc;
          ir8[10] = v179_acc;
          ir8[11] = v180_acc;
          ir8[12] = v181_acc;
          // glb_m3 = store{r>g}(r8);
          int32_t v201_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v202_i0 = 0; v202_i0 < 1; ++v202_i0) {
            int32_t v210_lead = v201_lead + (v202_i0 * 32);
            #pragma unroll
            for (int32_t v203_i1 = 0; v203_i1 < 13; ++v203_i1) {
              int32_t v204_a = v202_i0 + v203_i1;
              float v205_data = r8[v204_a];
              int32_t v212_a = v210_lead + (v203_i1 * 32);
              glb_m3[v212_a] = v205_data;
            }
          }
          ;
        }
      }
    }
  }
}

