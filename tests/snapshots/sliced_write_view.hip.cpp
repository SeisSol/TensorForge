// === base name ===
kernel_939857c66e

// === header ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_939857c66e, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_939857c66e), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_939857c66e, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v19_lead = v12_lead + (v13_i0 * 32);
            #pragma unroll
            for (int32_t v14_i1 = 10; v14_i1 < 13; ++v14_i1) {
              float v22_data = __builtin_nontemporal_load(&glb_m1[(v19_lead + (v14_i1 * 32))]);
              r0[(v13_i0 + (v14_i1 - 10))] = v22_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          float v26_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v26_lin;
          float v27_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v27_lin;
          float v28_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v28_lin;
          float v29_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v29_lin;
          float v30_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v30_lin;
          float v31_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v31_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[1]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float v33_data = r0[0];
          float v34_data = r0[1];
          float v35_data = r0[2];
          float v36_acc{};
          float v37_lin = r1[0];
          float v38_bc = tensorforge::broadcast<32, 16, 0>(v37_lin);
          tensorforge::fmacdpp16<0>(v36_acc, v38_bc, v33_data);
          tensorforge::fmacdpp16<1>(v36_acc, v38_bc, v34_data);
          tensorforge::fmacdpp16<2>(v36_acc, v38_bc, v35_data);
          r2[0] = v36_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v42_i0 = 0; v42_i0 < 1; ++v42_i0) {
            int32_t v50_lead = v12_lead + (v42_i0 * 32);
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 1; ++v43_i1) {
              float v45_data = r2[(v42_i0 + v43_i1)];
              glb_m0[(v50_lead + ((v43_i1 + 8) * 32))] = v45_data;
            }
          }
          float r3[13]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v58_i0 = 0; v58_i0 < 1; ++v58_i0) {
            int32_t v64_lead = v12_lead + (v58_i0 * 32);
            #pragma unroll
            for (int32_t v59_i1 = 0; v59_i1 < 13; ++v59_i1) {
              float v67_data = glb_m0[(v64_lead + (v59_i1 * 32))];
              r3[(v58_i0 + v59_i1)] = v67_data;
            }
          }
          float r4[13]{};
          // r4 = load{g>r}(glb_m4);
          float v70_lin = glb_m4[0 + threadIdx.x * 1];
          r4[0] = v70_lin;
          float v71_lin = glb_m4[32 + threadIdx.x * 1];
          r4[1] = v71_lin;
          float v72_lin = glb_m4[64 + threadIdx.x * 1];
          r4[2] = v72_lin;
          float v73_lin = glb_m4[96 + threadIdx.x * 1];
          r4[3] = v73_lin;
          float v74_lin = glb_m4[128 + threadIdx.x * 1];
          r4[4] = v74_lin;
          float v75_lin = glb_m4[160 + threadIdx.x * 1];
          r4[5] = v75_lin;
          // wait(r3 = load{g>r}(glb_m0););
          // wait(r4 = load{g>r}(glb_m4););
          float r5[13]{};
          // r5 = +(r3 * r4) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v77_data = r3[0];
          float v78_data = r3[1];
          float v79_data = r3[2];
          float v80_data = r3[3];
          float v81_data = r3[4];
          float v82_data = r3[5];
          float v83_data = r3[6];
          float v84_data = r3[7];
          float v85_data = r3[8];
          float v86_data = r3[9];
          float v87_data = r3[10];
          float v88_data = r3[11];
          float v89_data = r3[12];
          float v90_acc{};
          float v91_acc{};
          float v92_acc{};
          float v93_acc{};
          float v94_acc{};
          float v95_acc{};
          float v96_acc{};
          float v97_acc{};
          float v98_acc{};
          float v99_acc{};
          float v100_acc{};
          float v101_acc{};
          float v102_acc{};
          float v103_lin = r4[0];
          float v104_bc = tensorforge::broadcast<32, 16, 0>(v103_lin);
          tensorforge::fmacdpp16<0>(v90_acc, v104_bc, v77_data);
          tensorforge::fmacdpp16<1>(v90_acc, v104_bc, v78_data);
          tensorforge::fmacdpp16<2>(v90_acc, v104_bc, v79_data);
          tensorforge::fmacdpp16<3>(v90_acc, v104_bc, v80_data);
          tensorforge::fmacdpp16<4>(v90_acc, v104_bc, v81_data);
          tensorforge::fmacdpp16<5>(v90_acc, v104_bc, v82_data);
          tensorforge::fmacdpp16<6>(v90_acc, v104_bc, v83_data);
          tensorforge::fmacdpp16<7>(v90_acc, v104_bc, v84_data);
          tensorforge::fmacdpp16<8>(v90_acc, v104_bc, v85_data);
          tensorforge::fmacdpp16<9>(v90_acc, v104_bc, v86_data);
          tensorforge::fmacdpp16<10>(v90_acc, v104_bc, v87_data);
          tensorforge::fmacdpp16<11>(v90_acc, v104_bc, v88_data);
          tensorforge::fmacdpp16<12>(v90_acc, v104_bc, v89_data);
          tensorforge::fmacdpp16<13>(v91_acc, v104_bc, v77_data);
          tensorforge::fmacdpp16<14>(v91_acc, v104_bc, v78_data);
          tensorforge::fmacdpp16<15>(v91_acc, v104_bc, v79_data);
          float v105_bc = tensorforge::broadcast<32, 16, 1>(v103_lin);
          tensorforge::fmacdpp16<0>(v91_acc, v105_bc, v80_data);
          tensorforge::fmacdpp16<1>(v91_acc, v105_bc, v81_data);
          tensorforge::fmacdpp16<2>(v91_acc, v105_bc, v82_data);
          tensorforge::fmacdpp16<3>(v91_acc, v105_bc, v83_data);
          tensorforge::fmacdpp16<4>(v91_acc, v105_bc, v84_data);
          tensorforge::fmacdpp16<5>(v91_acc, v105_bc, v85_data);
          tensorforge::fmacdpp16<6>(v91_acc, v105_bc, v86_data);
          tensorforge::fmacdpp16<7>(v91_acc, v105_bc, v87_data);
          tensorforge::fmacdpp16<8>(v91_acc, v105_bc, v88_data);
          tensorforge::fmacdpp16<9>(v91_acc, v105_bc, v89_data);
          tensorforge::fmacdpp16<10>(v92_acc, v105_bc, v77_data);
          tensorforge::fmacdpp16<11>(v92_acc, v105_bc, v78_data);
          tensorforge::fmacdpp16<12>(v92_acc, v105_bc, v79_data);
          tensorforge::fmacdpp16<13>(v92_acc, v105_bc, v80_data);
          tensorforge::fmacdpp16<14>(v92_acc, v105_bc, v81_data);
          tensorforge::fmacdpp16<15>(v92_acc, v105_bc, v82_data);
          float v106_lin = r4[1];
          float v107_bc = tensorforge::broadcast<32, 16, 0>(v106_lin);
          tensorforge::fmacdpp16<0>(v92_acc, v107_bc, v83_data);
          tensorforge::fmacdpp16<1>(v92_acc, v107_bc, v84_data);
          tensorforge::fmacdpp16<2>(v92_acc, v107_bc, v85_data);
          tensorforge::fmacdpp16<3>(v92_acc, v107_bc, v86_data);
          tensorforge::fmacdpp16<4>(v92_acc, v107_bc, v87_data);
          tensorforge::fmacdpp16<5>(v92_acc, v107_bc, v88_data);
          tensorforge::fmacdpp16<6>(v92_acc, v107_bc, v89_data);
          tensorforge::fmacdpp16<7>(v93_acc, v107_bc, v77_data);
          tensorforge::fmacdpp16<8>(v93_acc, v107_bc, v78_data);
          tensorforge::fmacdpp16<9>(v93_acc, v107_bc, v79_data);
          tensorforge::fmacdpp16<10>(v93_acc, v107_bc, v80_data);
          tensorforge::fmacdpp16<11>(v93_acc, v107_bc, v81_data);
          tensorforge::fmacdpp16<12>(v93_acc, v107_bc, v82_data);
          tensorforge::fmacdpp16<13>(v93_acc, v107_bc, v83_data);
          tensorforge::fmacdpp16<14>(v93_acc, v107_bc, v84_data);
          tensorforge::fmacdpp16<15>(v93_acc, v107_bc, v85_data);
          float v108_bc = tensorforge::broadcast<32, 16, 1>(v106_lin);
          tensorforge::fmacdpp16<0>(v93_acc, v108_bc, v86_data);
          tensorforge::fmacdpp16<1>(v93_acc, v108_bc, v87_data);
          tensorforge::fmacdpp16<2>(v93_acc, v108_bc, v88_data);
          tensorforge::fmacdpp16<3>(v93_acc, v108_bc, v89_data);
          tensorforge::fmacdpp16<4>(v94_acc, v108_bc, v77_data);
          tensorforge::fmacdpp16<5>(v94_acc, v108_bc, v78_data);
          tensorforge::fmacdpp16<6>(v94_acc, v108_bc, v79_data);
          tensorforge::fmacdpp16<7>(v94_acc, v108_bc, v80_data);
          tensorforge::fmacdpp16<8>(v94_acc, v108_bc, v81_data);
          tensorforge::fmacdpp16<9>(v94_acc, v108_bc, v82_data);
          tensorforge::fmacdpp16<10>(v94_acc, v108_bc, v83_data);
          tensorforge::fmacdpp16<11>(v94_acc, v108_bc, v84_data);
          tensorforge::fmacdpp16<12>(v94_acc, v108_bc, v85_data);
          tensorforge::fmacdpp16<13>(v94_acc, v108_bc, v86_data);
          tensorforge::fmacdpp16<14>(v94_acc, v108_bc, v87_data);
          tensorforge::fmacdpp16<15>(v94_acc, v108_bc, v88_data);
          float v109_lin = r4[2];
          float v110_bc = tensorforge::broadcast<32, 16, 0>(v109_lin);
          tensorforge::fmacdpp16<0>(v94_acc, v110_bc, v89_data);
          tensorforge::fmacdpp16<1>(v95_acc, v110_bc, v77_data);
          tensorforge::fmacdpp16<2>(v95_acc, v110_bc, v78_data);
          tensorforge::fmacdpp16<3>(v95_acc, v110_bc, v79_data);
          tensorforge::fmacdpp16<4>(v95_acc, v110_bc, v80_data);
          tensorforge::fmacdpp16<5>(v95_acc, v110_bc, v81_data);
          tensorforge::fmacdpp16<6>(v95_acc, v110_bc, v82_data);
          tensorforge::fmacdpp16<7>(v95_acc, v110_bc, v83_data);
          tensorforge::fmacdpp16<8>(v95_acc, v110_bc, v84_data);
          tensorforge::fmacdpp16<9>(v95_acc, v110_bc, v85_data);
          tensorforge::fmacdpp16<10>(v95_acc, v110_bc, v86_data);
          tensorforge::fmacdpp16<11>(v95_acc, v110_bc, v87_data);
          tensorforge::fmacdpp16<12>(v95_acc, v110_bc, v88_data);
          tensorforge::fmacdpp16<13>(v95_acc, v110_bc, v89_data);
          tensorforge::fmacdpp16<14>(v96_acc, v110_bc, v77_data);
          tensorforge::fmacdpp16<15>(v96_acc, v110_bc, v78_data);
          float v111_bc = tensorforge::broadcast<32, 16, 1>(v109_lin);
          tensorforge::fmacdpp16<0>(v96_acc, v111_bc, v79_data);
          tensorforge::fmacdpp16<1>(v96_acc, v111_bc, v80_data);
          tensorforge::fmacdpp16<2>(v96_acc, v111_bc, v81_data);
          tensorforge::fmacdpp16<3>(v96_acc, v111_bc, v82_data);
          tensorforge::fmacdpp16<4>(v96_acc, v111_bc, v83_data);
          tensorforge::fmacdpp16<5>(v96_acc, v111_bc, v84_data);
          tensorforge::fmacdpp16<6>(v96_acc, v111_bc, v85_data);
          tensorforge::fmacdpp16<7>(v96_acc, v111_bc, v86_data);
          tensorforge::fmacdpp16<8>(v96_acc, v111_bc, v87_data);
          tensorforge::fmacdpp16<9>(v96_acc, v111_bc, v88_data);
          tensorforge::fmacdpp16<10>(v96_acc, v111_bc, v89_data);
          tensorforge::fmacdpp16<11>(v97_acc, v111_bc, v77_data);
          tensorforge::fmacdpp16<12>(v97_acc, v111_bc, v78_data);
          tensorforge::fmacdpp16<13>(v97_acc, v111_bc, v79_data);
          tensorforge::fmacdpp16<14>(v97_acc, v111_bc, v80_data);
          tensorforge::fmacdpp16<15>(v97_acc, v111_bc, v81_data);
          float v112_lin = r4[3];
          float v113_bc = tensorforge::broadcast<32, 16, 0>(v112_lin);
          tensorforge::fmacdpp16<0>(v97_acc, v113_bc, v82_data);
          tensorforge::fmacdpp16<1>(v97_acc, v113_bc, v83_data);
          tensorforge::fmacdpp16<2>(v97_acc, v113_bc, v84_data);
          tensorforge::fmacdpp16<3>(v97_acc, v113_bc, v85_data);
          tensorforge::fmacdpp16<4>(v97_acc, v113_bc, v86_data);
          tensorforge::fmacdpp16<5>(v97_acc, v113_bc, v87_data);
          tensorforge::fmacdpp16<6>(v97_acc, v113_bc, v88_data);
          tensorforge::fmacdpp16<7>(v97_acc, v113_bc, v89_data);
          tensorforge::fmacdpp16<8>(v98_acc, v113_bc, v77_data);
          tensorforge::fmacdpp16<9>(v98_acc, v113_bc, v78_data);
          tensorforge::fmacdpp16<10>(v98_acc, v113_bc, v79_data);
          tensorforge::fmacdpp16<11>(v98_acc, v113_bc, v80_data);
          tensorforge::fmacdpp16<12>(v98_acc, v113_bc, v81_data);
          tensorforge::fmacdpp16<13>(v98_acc, v113_bc, v82_data);
          tensorforge::fmacdpp16<14>(v98_acc, v113_bc, v83_data);
          tensorforge::fmacdpp16<15>(v98_acc, v113_bc, v84_data);
          float v114_bc = tensorforge::broadcast<32, 16, 1>(v112_lin);
          tensorforge::fmacdpp16<0>(v98_acc, v114_bc, v85_data);
          tensorforge::fmacdpp16<1>(v98_acc, v114_bc, v86_data);
          tensorforge::fmacdpp16<2>(v98_acc, v114_bc, v87_data);
          tensorforge::fmacdpp16<3>(v98_acc, v114_bc, v88_data);
          tensorforge::fmacdpp16<4>(v98_acc, v114_bc, v89_data);
          tensorforge::fmacdpp16<5>(v99_acc, v114_bc, v77_data);
          tensorforge::fmacdpp16<6>(v99_acc, v114_bc, v78_data);
          tensorforge::fmacdpp16<7>(v99_acc, v114_bc, v79_data);
          tensorforge::fmacdpp16<8>(v99_acc, v114_bc, v80_data);
          tensorforge::fmacdpp16<9>(v99_acc, v114_bc, v81_data);
          tensorforge::fmacdpp16<10>(v99_acc, v114_bc, v82_data);
          tensorforge::fmacdpp16<11>(v99_acc, v114_bc, v83_data);
          tensorforge::fmacdpp16<12>(v99_acc, v114_bc, v84_data);
          tensorforge::fmacdpp16<13>(v99_acc, v114_bc, v85_data);
          tensorforge::fmacdpp16<14>(v99_acc, v114_bc, v86_data);
          tensorforge::fmacdpp16<15>(v99_acc, v114_bc, v87_data);
          float v115_lin = r4[4];
          float v116_bc = tensorforge::broadcast<32, 16, 0>(v115_lin);
          tensorforge::fmacdpp16<0>(v99_acc, v116_bc, v88_data);
          tensorforge::fmacdpp16<1>(v99_acc, v116_bc, v89_data);
          tensorforge::fmacdpp16<2>(v100_acc, v116_bc, v77_data);
          tensorforge::fmacdpp16<3>(v100_acc, v116_bc, v78_data);
          tensorforge::fmacdpp16<4>(v100_acc, v116_bc, v79_data);
          tensorforge::fmacdpp16<5>(v100_acc, v116_bc, v80_data);
          tensorforge::fmacdpp16<6>(v100_acc, v116_bc, v81_data);
          tensorforge::fmacdpp16<7>(v100_acc, v116_bc, v82_data);
          tensorforge::fmacdpp16<8>(v100_acc, v116_bc, v83_data);
          tensorforge::fmacdpp16<9>(v100_acc, v116_bc, v84_data);
          tensorforge::fmacdpp16<10>(v100_acc, v116_bc, v85_data);
          tensorforge::fmacdpp16<11>(v100_acc, v116_bc, v86_data);
          tensorforge::fmacdpp16<12>(v100_acc, v116_bc, v87_data);
          tensorforge::fmacdpp16<13>(v100_acc, v116_bc, v88_data);
          tensorforge::fmacdpp16<14>(v100_acc, v116_bc, v89_data);
          tensorforge::fmacdpp16<15>(v101_acc, v116_bc, v77_data);
          float v117_bc = tensorforge::broadcast<32, 16, 1>(v115_lin);
          tensorforge::fmacdpp16<0>(v101_acc, v117_bc, v78_data);
          tensorforge::fmacdpp16<1>(v101_acc, v117_bc, v79_data);
          tensorforge::fmacdpp16<2>(v101_acc, v117_bc, v80_data);
          tensorforge::fmacdpp16<3>(v101_acc, v117_bc, v81_data);
          tensorforge::fmacdpp16<4>(v101_acc, v117_bc, v82_data);
          tensorforge::fmacdpp16<5>(v101_acc, v117_bc, v83_data);
          tensorforge::fmacdpp16<6>(v101_acc, v117_bc, v84_data);
          tensorforge::fmacdpp16<7>(v101_acc, v117_bc, v85_data);
          tensorforge::fmacdpp16<8>(v101_acc, v117_bc, v86_data);
          tensorforge::fmacdpp16<9>(v101_acc, v117_bc, v87_data);
          tensorforge::fmacdpp16<10>(v101_acc, v117_bc, v88_data);
          tensorforge::fmacdpp16<11>(v101_acc, v117_bc, v89_data);
          tensorforge::fmacdpp16<12>(v102_acc, v117_bc, v77_data);
          tensorforge::fmacdpp16<13>(v102_acc, v117_bc, v78_data);
          tensorforge::fmacdpp16<14>(v102_acc, v117_bc, v79_data);
          tensorforge::fmacdpp16<15>(v102_acc, v117_bc, v80_data);
          float v118_lin = r4[5];
          float v119_bc = tensorforge::broadcast<32, 16, 0>(v118_lin);
          tensorforge::fmacdpp16<0>(v102_acc, v119_bc, v81_data);
          tensorforge::fmacdpp16<1>(v102_acc, v119_bc, v82_data);
          tensorforge::fmacdpp16<2>(v102_acc, v119_bc, v83_data);
          tensorforge::fmacdpp16<3>(v102_acc, v119_bc, v84_data);
          tensorforge::fmacdpp16<4>(v102_acc, v119_bc, v85_data);
          tensorforge::fmacdpp16<5>(v102_acc, v119_bc, v86_data);
          tensorforge::fmacdpp16<6>(v102_acc, v119_bc, v87_data);
          tensorforge::fmacdpp16<7>(v102_acc, v119_bc, v88_data);
          tensorforge::fmacdpp16<8>(v102_acc, v119_bc, v89_data);
          r5[0] = v90_acc;
          r5[1] = v91_acc;
          r5[2] = v92_acc;
          r5[3] = v93_acc;
          r5[4] = v94_acc;
          r5[5] = v95_acc;
          r5[6] = v96_acc;
          r5[7] = v97_acc;
          r5[8] = v98_acc;
          r5[9] = v99_acc;
          r5[10] = v100_acc;
          r5[11] = v101_acc;
          r5[12] = v102_acc;
          // glb_m3 = store{r>g}(r5);
          #pragma unroll
          for (int32_t v123_i0 = 0; v123_i0 < 1; ++v123_i0) {
            int32_t v131_lead = v12_lead + (v123_i0 * 32);
            #pragma unroll
            for (int32_t v124_i1 = 0; v124_i1 < 13; ++v124_i1) {
              float v126_data = r5[(v123_i0 + v124_i1)];
              glb_m3[(v131_lead + (v124_i1 * 32))] = v126_data;
            }
          }
        }
      }
    }
  }
}

