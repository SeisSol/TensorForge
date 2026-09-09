// === base name ===
kernel_9a4dae70e8465cd8

// === header ===
void launcher_kernel_9a4dae70e8465cd8(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, float* m9, size_t m9_extraOffset, const float* m10, size_t m10_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_9a4dae70e8465cd8(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, float* m9, size_t m9_extraOffset, const float* m10, size_t m10_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9a4dae70e8465cd8, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_9a4dae70e8465cd8), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_9a4dae70e8465cd8, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  m9,  m9_extraOffset,  m10,  m10_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_9a4dae70e8465cd8(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, float* m9, size_t m9_extraOffset, const float* m10, size_t m10_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 13×13(13×13) {0..13}×{0..13} strided
    // m4 16×32(16×32) {0..16}×{0..32} strided
    // m5 13×13(13×13) {0..13}×{0..13} strided
    // m6 16×32(16×32) {0..16}×{0..32} strided
    // m7 13×13(13×13) {0..13}×{0..13} strided
    // m8 16×32(16×32) {0..16}×{0..32} strided
    // m9 32×13(32×13) {0..32}×{0..13} strided
    // m10 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    // t0 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m3 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..16}×{0..13})[0, 1] += m4 16×32(16×32) {0..16}×{0..32} strided({0..16}×{0..32})[0, -1]×t0 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[-1, 1]
    // t1 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m5 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..16}×{0..13})[0, 1] += m6 16×32(16×32) {0..16}×{0..32} strided({0..16}×{0..32})[0, -1]×t1 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[-1, 1]
    // t2 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m7 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..16}×{0..13})[0, 1] += m8 16×32(16×32) {0..16}×{0..32} strided({0..16}×{0..32})[0, -1]×t2 32×13(32×13) {0..32}×{0..13} pointer_based({0..32}×{0..13})[-1, 1]
    // m9 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m10 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
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
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 169 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 512 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 169 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[batchId0 * 512 + 0 + m6_extraOffset];
          const float *const __restrict__ glb_m7 = &m7[batchId0 * 169 + 0 + m7_extraOffset];
          const float *const __restrict__ glb_m8 = &m8[batchId0 * 512 + 0 + m8_extraOffset];
          float *const __restrict__ glb_m9 = &m9[batchId0 * 416 + 0 + m9_extraOffset];
          const float *const __restrict__ glb_m10 = &m10[batchId0 * 169 + 0 + m10_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v25_lead = v18_lead + (v19_i0 * 32);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 13; ++v20_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m1[(v25_lead + (v20_i1 * 32))]);
              r0[(v19_i0 + v20_i1)] = v28_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          if (v18_lead < 13) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 13; ++v35_i1) {
              float v43_data = __builtin_nontemporal_load(&glb_m2[(v18_lead + (v35_i1 * 13))]);
              r1[v35_i1] = v43_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r3[13]{};
          // r3 = load{g>r}(glb_m3);
          if (v18_lead < 13) {
            #pragma unroll
            for (int32_t v50_i1 = 0; v50_i1 < 13; ++v50_i1) {
              float v58_data = __builtin_nontemporal_load(&glb_m3[(v18_lead + (v50_i1 * 13))]);
              r3[v50_i1] = v58_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[13]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v61_data = r1[0];
          float v62_data = r1[1];
          float v63_data = r1[2];
          float v64_data = r1[3];
          float v65_tp{};
          float v66_tp{};
          float v67_tp{};
          float v68_tp{};
          tensorforge::transpose4x4b32(v65_tp, v66_tp, v67_tp, v68_tp, v61_data, v62_data, v63_data, v64_data);
          tensorforge::VectorT<float, 4> v69_acc{};
          float v70_data = r0[0];
          float v71_data = r0[1];
          float v72_data = r0[2];
          float v73_data = r0[3];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v70_data, v69_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v71_data, v74_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v72_data, v75_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v73_data, v76_acc, 3, 0, 0);
          float v78_data = r0[4];
          float v79_data = r0[5];
          float v80_data = r0[6];
          float v81_data = r0[7];
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v78_data, v77_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v79_data, v82_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v80_data, v83_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v81_data, v84_acc, 3, 1, 0);
          float v86_data = r0[8];
          float v87_data = r0[9];
          float v88_data = r0[10];
          float v89_data = r0[11];
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v86_data, v85_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v87_data, v90_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v88_data, v91_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v89_data, v92_acc, 3, 2, 0);
          float v94_data = r0[12];
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v94_data, v93_acc, 3, 3, 0);
          r2[0] = (v98_acc[0]);
          r2[1] = (v98_acc[1]);
          r2[2] = (v98_acc[2]);
          r2[3] = (v98_acc[3]);
          float v103_data = r1[4];
          float v104_data = r1[5];
          float v105_data = r1[6];
          float v106_data = r1[7];
          float v107_tp{};
          float v108_tp{};
          float v109_tp{};
          float v110_tp{};
          tensorforge::transpose4x4b32(v107_tp, v108_tp, v109_tp, v110_tp, v103_data, v104_data, v105_data, v106_data);
          tensorforge::VectorT<float, 4> v111_acc{};
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v70_data, v111_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v71_data, v116_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v72_data, v117_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v73_data, v118_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v78_data, v119_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v79_data, v124_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v80_data, v125_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v81_data, v126_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v86_data, v127_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v87_data, v132_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v88_data, v133_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v89_data, v134_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v94_data, v135_acc, 3, 3, 0);
          r2[4] = (v140_acc[0]);
          r2[5] = (v140_acc[1]);
          r2[6] = (v140_acc[2]);
          r2[7] = (v140_acc[3]);
          float v145_data = r1[8];
          float v146_data = r1[9];
          float v147_data = r1[10];
          float v148_data = r1[11];
          float v149_tp{};
          float v150_tp{};
          float v151_tp{};
          float v152_tp{};
          tensorforge::transpose4x4b32(v149_tp, v150_tp, v151_tp, v152_tp, v145_data, v146_data, v147_data, v148_data);
          tensorforge::VectorT<float, 4> v153_acc{};
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v70_data, v153_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v71_data, v158_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v72_data, v159_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v73_data, v160_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v78_data, v161_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v79_data, v166_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v80_data, v167_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v81_data, v168_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v86_data, v169_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v150_tp, v87_data, v174_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v151_tp, v88_data, v175_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v152_tp, v89_data, v176_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v182_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v149_tp, v94_data, v177_acc, 3, 3, 0);
          r2[8] = (v182_acc[0]);
          r2[9] = (v182_acc[1]);
          r2[10] = (v182_acc[2]);
          r2[11] = (v182_acc[3]);
          float v200_acc{};
          float v201_data = r1[12];
          float v202_bc = tensorforge::broadcast<32, 16, 0>(v201_data);
          tensorforge::fmacdpp16<0>(v200_acc, v202_bc, v70_data);
          tensorforge::fmacdpp16<1>(v200_acc, v202_bc, v71_data);
          tensorforge::fmacdpp16<2>(v200_acc, v202_bc, v72_data);
          tensorforge::fmacdpp16<3>(v200_acc, v202_bc, v73_data);
          tensorforge::fmacdpp16<4>(v200_acc, v202_bc, v78_data);
          tensorforge::fmacdpp16<5>(v200_acc, v202_bc, v79_data);
          tensorforge::fmacdpp16<6>(v200_acc, v202_bc, v80_data);
          tensorforge::fmacdpp16<7>(v200_acc, v202_bc, v81_data);
          tensorforge::fmacdpp16<8>(v200_acc, v202_bc, v86_data);
          tensorforge::fmacdpp16<9>(v200_acc, v202_bc, v87_data);
          tensorforge::fmacdpp16<10>(v200_acc, v202_bc, v88_data);
          tensorforge::fmacdpp16<11>(v200_acc, v202_bc, v89_data);
          tensorforge::fmacdpp16<12>(v200_acc, v202_bc, v94_data);
          r2[12] = v200_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v206_i0 = 0; v206_i0 < 1; ++v206_i0) {
            int32_t v214_lead = v18_lead + (v206_i0 * 32);
            #pragma unroll
            for (int32_t v207_i1 = 0; v207_i1 < 13; ++v207_i1) {
              float v209_data = r2[(v206_i0 + v207_i1)];
              glb_m0[(v214_lead + (v207_i1 * 32))] = v209_data;
            }
          }
          float r5[32]{};
          // r5 = load{g>r}(glb_m4);
          if (v18_lead < 16) {
            #pragma unroll
            for (int32_t v222_i1 = 0; v222_i1 < 32; ++v222_i1) {
              float v230_data = __builtin_nontemporal_load(&glb_m4[(v18_lead + (v222_i1 * 16))]);
              r5[v222_i1] = v230_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m3););
          float r4[13]{};
          // r4 = +(r0 * r3) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v233_data = r3[0];
          float v234_data = r3[1];
          float v235_data = r3[2];
          float v236_data = r3[3];
          float v237_tp{};
          float v238_tp{};
          float v239_tp{};
          float v240_tp{};
          tensorforge::transpose4x4b32(v237_tp, v238_tp, v239_tp, v240_tp, v233_data, v234_data, v235_data, v236_data);
          tensorforge::VectorT<float, 4> v241_acc{};
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v70_data, v241_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v71_data, v246_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v72_data, v247_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v73_data, v248_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v78_data, v249_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v79_data, v254_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v80_data, v255_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v81_data, v256_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v86_data, v257_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v87_data, v262_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v88_data, v263_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v89_data, v264_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v270_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v94_data, v265_acc, 3, 3, 0);
          r4[0] = (v270_acc[0]);
          r4[1] = (v270_acc[1]);
          r4[2] = (v270_acc[2]);
          r4[3] = (v270_acc[3]);
          float v275_data = r3[4];
          float v276_data = r3[5];
          float v277_data = r3[6];
          float v278_data = r3[7];
          float v279_tp{};
          float v280_tp{};
          float v281_tp{};
          float v282_tp{};
          tensorforge::transpose4x4b32(v279_tp, v280_tp, v281_tp, v282_tp, v275_data, v276_data, v277_data, v278_data);
          tensorforge::VectorT<float, 4> v283_acc{};
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v70_data, v283_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v71_data, v288_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v72_data, v289_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v73_data, v290_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v78_data, v291_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v79_data, v296_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v80_data, v297_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v81_data, v298_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v86_data, v299_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v280_tp, v87_data, v304_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v281_tp, v88_data, v305_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v282_tp, v89_data, v306_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v279_tp, v94_data, v307_acc, 3, 3, 0);
          r4[4] = (v312_acc[0]);
          r4[5] = (v312_acc[1]);
          r4[6] = (v312_acc[2]);
          r4[7] = (v312_acc[3]);
          float v317_data = r3[8];
          float v318_data = r3[9];
          float v319_data = r3[10];
          float v320_data = r3[11];
          float v321_tp{};
          float v322_tp{};
          float v323_tp{};
          float v324_tp{};
          tensorforge::transpose4x4b32(v321_tp, v322_tp, v323_tp, v324_tp, v317_data, v318_data, v319_data, v320_data);
          tensorforge::VectorT<float, 4> v325_acc{};
          tensorforge::VectorT<float, 4> v330_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v70_data, v325_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v71_data, v330_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v72_data, v331_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v73_data, v332_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v338_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v78_data, v333_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v79_data, v338_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v80_data, v339_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v81_data, v340_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v346_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v86_data, v341_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v322_tp, v87_data, v346_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v323_tp, v88_data, v347_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v324_tp, v89_data, v348_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v354_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v321_tp, v94_data, v349_acc, 3, 3, 0);
          r4[8] = (v354_acc[0]);
          r4[9] = (v354_acc[1]);
          r4[10] = (v354_acc[2]);
          r4[11] = (v354_acc[3]);
          float v372_acc{};
          float v373_data = r3[12];
          float v374_bc = tensorforge::broadcast<32, 16, 0>(v373_data);
          tensorforge::fmacdpp16<0>(v372_acc, v374_bc, v70_data);
          tensorforge::fmacdpp16<1>(v372_acc, v374_bc, v71_data);
          tensorforge::fmacdpp16<2>(v372_acc, v374_bc, v72_data);
          tensorforge::fmacdpp16<3>(v372_acc, v374_bc, v73_data);
          tensorforge::fmacdpp16<4>(v372_acc, v374_bc, v78_data);
          tensorforge::fmacdpp16<5>(v372_acc, v374_bc, v79_data);
          tensorforge::fmacdpp16<6>(v372_acc, v374_bc, v80_data);
          tensorforge::fmacdpp16<7>(v372_acc, v374_bc, v81_data);
          tensorforge::fmacdpp16<8>(v372_acc, v374_bc, v86_data);
          tensorforge::fmacdpp16<9>(v372_acc, v374_bc, v87_data);
          tensorforge::fmacdpp16<10>(v372_acc, v374_bc, v88_data);
          tensorforge::fmacdpp16<11>(v372_acc, v374_bc, v89_data);
          tensorforge::fmacdpp16<12>(v372_acc, v374_bc, v94_data);
          r4[12] = v372_acc;
          float r7[13]{};
          // r7 = load{g>r}(glb_m5);
          if (v18_lead < 13) {
            #pragma unroll
            for (int32_t v380_i1 = 0; v380_i1 < 13; ++v380_i1) {
              float v388_data = __builtin_nontemporal_load(&glb_m5[(v18_lead + (v380_i1 * 13))]);
              r7[v380_i1] = v388_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[13]{};
          // r6 = +(r5 * r4) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v391_data = r4[0];
          float v392_data = r4[1];
          float v393_data = r4[2];
          float v394_data = r4[3];
          float v395_tp{};
          float v396_tp{};
          float v397_tp{};
          float v398_tp{};
          tensorforge::transpose4x4b32(v395_tp, v396_tp, v397_tp, v398_tp, v391_data, v392_data, v393_data, v394_data);
          tensorforge::VectorT<float, 4> v399_acc{};
          float v400_data = r5[0];
          float v401_data = r5[1];
          float v402_data = r5[2];
          float v403_data = r5[3];
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v400_data, v399_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v405_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v401_data, v404_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v406_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v402_data, v405_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v407_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v403_data, v406_acc, 3, 0, 0);
          float v408_data = r5[4];
          float v409_data = r5[5];
          float v410_data = r5[6];
          float v411_data = r5[7];
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v408_data, v407_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v413_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v409_data, v412_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v410_data, v413_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v411_data, v414_acc, 3, 1, 0);
          float v416_data = r5[8];
          float v417_data = r5[9];
          float v418_data = r5[10];
          float v419_data = r5[11];
          tensorforge::VectorT<float, 4> v420_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v416_data, v415_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v421_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v417_data, v420_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v422_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v418_data, v421_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v423_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v419_data, v422_acc, 3, 2, 0);
          float v424_data = r5[12];
          float v425_data = r5[13];
          float v426_data = r5[14];
          float v427_data = r5[15];
          tensorforge::VectorT<float, 4> v428_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v424_data, v423_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v429_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v425_data, v428_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v426_data, v429_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v427_data, v430_acc, 3, 3, 0);
          float v432_data = r5[16];
          float v433_data = r5[17];
          float v434_data = r5[18];
          float v435_data = r5[19];
          tensorforge::VectorT<float, 4> v436_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v432_data, v431_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v437_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v433_data, v436_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v434_data, v437_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v435_data, v438_acc, 3, 4, 0);
          float v440_data = r5[20];
          float v441_data = r5[21];
          float v442_data = r5[22];
          float v443_data = r5[23];
          tensorforge::VectorT<float, 4> v444_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v440_data, v439_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v445_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v441_data, v444_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v442_data, v445_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v443_data, v446_acc, 3, 5, 0);
          float v448_data = r5[24];
          float v449_data = r5[25];
          float v450_data = r5[26];
          float v451_data = r5[27];
          tensorforge::VectorT<float, 4> v452_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v448_data, v447_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v449_data, v452_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v450_data, v453_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v451_data, v454_acc, 3, 6, 0);
          float v456_data = r5[28];
          float v457_data = r5[29];
          float v458_data = r5[30];
          float v459_data = r5[31];
          tensorforge::VectorT<float, 4> v460_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v395_tp, v456_data, v455_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v396_tp, v457_data, v460_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v397_tp, v458_data, v461_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v398_tp, v459_data, v462_acc, 3, 7, 0);
          r6[0] = (v463_acc[0]);
          r6[1] = (v463_acc[1]);
          r6[2] = (v463_acc[2]);
          r6[3] = (v463_acc[3]);
          float v468_data = r4[4];
          float v469_data = r4[5];
          float v470_data = r4[6];
          float v471_data = r4[7];
          float v472_tp{};
          float v473_tp{};
          float v474_tp{};
          float v475_tp{};
          tensorforge::transpose4x4b32(v472_tp, v473_tp, v474_tp, v475_tp, v468_data, v469_data, v470_data, v471_data);
          tensorforge::VectorT<float, 4> v476_acc{};
          tensorforge::VectorT<float, 4> v481_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v472_tp, v400_data, v476_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v482_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v473_tp, v401_data, v481_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v483_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v474_tp, v402_data, v482_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v484_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v403_data, v483_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v489_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v472_tp, v408_data, v484_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v490_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v473_tp, v409_data, v489_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v474_tp, v410_data, v490_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v411_data, v491_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v497_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v472_tp, v416_data, v492_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v498_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v473_tp, v417_data, v497_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v499_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v474_tp, v418_data, v498_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v500_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v419_data, v499_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v505_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v472_tp, v424_data, v500_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v506_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v473_tp, v425_data, v505_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v507_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v474_tp, v426_data, v506_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v508_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v427_data, v507_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v513_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v472_tp, v432_data, v508_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v514_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v473_tp, v433_data, v513_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v515_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v474_tp, v434_data, v514_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v516_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v435_data, v515_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v521_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v472_tp, v440_data, v516_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v522_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v473_tp, v441_data, v521_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v474_tp, v442_data, v522_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v443_data, v523_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v529_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v472_tp, v448_data, v524_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v530_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v473_tp, v449_data, v529_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v531_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v474_tp, v450_data, v530_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v532_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v451_data, v531_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v537_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v472_tp, v456_data, v532_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v538_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v473_tp, v457_data, v537_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v539_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v474_tp, v458_data, v538_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v540_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v475_tp, v459_data, v539_acc, 3, 7, 0);
          r6[4] = (v540_acc[0]);
          r6[5] = (v540_acc[1]);
          r6[6] = (v540_acc[2]);
          r6[7] = (v540_acc[3]);
          float v545_data = r4[8];
          float v546_data = r4[9];
          float v547_data = r4[10];
          float v548_data = r4[11];
          float v549_tp{};
          float v550_tp{};
          float v551_tp{};
          float v552_tp{};
          tensorforge::transpose4x4b32(v549_tp, v550_tp, v551_tp, v552_tp, v545_data, v546_data, v547_data, v548_data);
          tensorforge::VectorT<float, 4> v553_acc{};
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v400_data, v553_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v559_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v401_data, v558_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v560_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v402_data, v559_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v561_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v403_data, v560_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v566_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v408_data, v561_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v567_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v409_data, v566_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v568_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v410_data, v567_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v569_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v411_data, v568_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v574_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v416_data, v569_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v575_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v417_data, v574_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v576_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v418_data, v575_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v577_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v419_data, v576_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v582_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v424_data, v577_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v583_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v425_data, v582_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v584_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v426_data, v583_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v585_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v427_data, v584_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v590_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v432_data, v585_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v591_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v433_data, v590_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v592_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v434_data, v591_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v593_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v435_data, v592_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v598_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v440_data, v593_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v599_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v441_data, v598_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v600_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v442_data, v599_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v601_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v443_data, v600_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v606_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v448_data, v601_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v607_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v449_data, v606_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v608_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v450_data, v607_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v609_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v451_data, v608_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v549_tp, v456_data, v609_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v615_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v550_tp, v457_data, v614_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v616_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v551_tp, v458_data, v615_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v617_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v552_tp, v459_data, v616_acc, 3, 7, 0);
          r6[8] = (v617_acc[0]);
          r6[9] = (v617_acc[1]);
          r6[10] = (v617_acc[2]);
          r6[11] = (v617_acc[3]);
          float v654_acc{};
          float v655_data = r4[12];
          float v656_bc = tensorforge::broadcast<32, 16, 0>(v655_data);
          tensorforge::fmacdpp16<0>(v654_acc, v656_bc, v400_data);
          tensorforge::fmacdpp16<1>(v654_acc, v656_bc, v401_data);
          tensorforge::fmacdpp16<2>(v654_acc, v656_bc, v402_data);
          tensorforge::fmacdpp16<3>(v654_acc, v656_bc, v403_data);
          tensorforge::fmacdpp16<4>(v654_acc, v656_bc, v408_data);
          tensorforge::fmacdpp16<5>(v654_acc, v656_bc, v409_data);
          tensorforge::fmacdpp16<6>(v654_acc, v656_bc, v410_data);
          tensorforge::fmacdpp16<7>(v654_acc, v656_bc, v411_data);
          tensorforge::fmacdpp16<8>(v654_acc, v656_bc, v416_data);
          tensorforge::fmacdpp16<9>(v654_acc, v656_bc, v417_data);
          tensorforge::fmacdpp16<10>(v654_acc, v656_bc, v418_data);
          tensorforge::fmacdpp16<11>(v654_acc, v656_bc, v419_data);
          tensorforge::fmacdpp16<12>(v654_acc, v656_bc, v424_data);
          tensorforge::fmacdpp16<13>(v654_acc, v656_bc, v425_data);
          tensorforge::fmacdpp16<14>(v654_acc, v656_bc, v426_data);
          tensorforge::fmacdpp16<15>(v654_acc, v656_bc, v427_data);
          float v657_bc = tensorforge::broadcast<32, 16, 1>(v655_data);
          tensorforge::fmacdpp16<0>(v654_acc, v657_bc, v432_data);
          tensorforge::fmacdpp16<1>(v654_acc, v657_bc, v433_data);
          tensorforge::fmacdpp16<2>(v654_acc, v657_bc, v434_data);
          tensorforge::fmacdpp16<3>(v654_acc, v657_bc, v435_data);
          tensorforge::fmacdpp16<4>(v654_acc, v657_bc, v440_data);
          tensorforge::fmacdpp16<5>(v654_acc, v657_bc, v441_data);
          tensorforge::fmacdpp16<6>(v654_acc, v657_bc, v442_data);
          tensorforge::fmacdpp16<7>(v654_acc, v657_bc, v443_data);
          tensorforge::fmacdpp16<8>(v654_acc, v657_bc, v448_data);
          tensorforge::fmacdpp16<9>(v654_acc, v657_bc, v449_data);
          tensorforge::fmacdpp16<10>(v654_acc, v657_bc, v450_data);
          tensorforge::fmacdpp16<11>(v654_acc, v657_bc, v451_data);
          tensorforge::fmacdpp16<12>(v654_acc, v657_bc, v456_data);
          tensorforge::fmacdpp16<13>(v654_acc, v657_bc, v457_data);
          tensorforge::fmacdpp16<14>(v654_acc, v657_bc, v458_data);
          tensorforge::fmacdpp16<15>(v654_acc, v657_bc, v459_data);
          r6[12] = v654_acc;
          // glb_m0 = store{r>g}(r6);
          if (v18_lead < 16) {
            #pragma unroll
            for (int32_t v662_i1 = 0; v662_i1 < 13; ++v662_i1) {
              float v664_data = r6[v662_i1];
              int32_t v671_a = v18_lead + (v662_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v671_a], v664_data);
            }
          }
          float r9[32]{};
          // r9 = load{g>r}(glb_m6);
          if (v18_lead < 16) {
            #pragma unroll
            for (int32_t v677_i1 = 0; v677_i1 < 32; ++v677_i1) {
              float v685_data = __builtin_nontemporal_load(&glb_m6[(v18_lead + (v677_i1 * 16))]);
              r9[v677_i1] = v685_data;
            }
          }
          // wait(r7 = load{g>r}(glb_m5););
          float r8[13]{};
          // r8 = +(r0 * r7) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v688_data = r7[0];
          float v689_data = r7[1];
          float v690_data = r7[2];
          float v691_data = r7[3];
          float v692_tp{};
          float v693_tp{};
          float v694_tp{};
          float v695_tp{};
          tensorforge::transpose4x4b32(v692_tp, v693_tp, v694_tp, v695_tp, v688_data, v689_data, v690_data, v691_data);
          tensorforge::VectorT<float, 4> v696_acc{};
          tensorforge::VectorT<float, 4> v701_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v692_tp, v70_data, v696_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v702_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v693_tp, v71_data, v701_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v694_tp, v72_data, v702_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v704_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v695_tp, v73_data, v703_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v709_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v692_tp, v78_data, v704_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v710_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v693_tp, v79_data, v709_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v711_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v694_tp, v80_data, v710_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v712_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v695_tp, v81_data, v711_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v717_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v692_tp, v86_data, v712_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v718_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v693_tp, v87_data, v717_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v719_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v694_tp, v88_data, v718_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v720_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v695_tp, v89_data, v719_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v725_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v692_tp, v94_data, v720_acc, 3, 3, 0);
          r8[0] = (v725_acc[0]);
          r8[1] = (v725_acc[1]);
          r8[2] = (v725_acc[2]);
          r8[3] = (v725_acc[3]);
          float v730_data = r7[4];
          float v731_data = r7[5];
          float v732_data = r7[6];
          float v733_data = r7[7];
          float v734_tp{};
          float v735_tp{};
          float v736_tp{};
          float v737_tp{};
          tensorforge::transpose4x4b32(v734_tp, v735_tp, v736_tp, v737_tp, v730_data, v731_data, v732_data, v733_data);
          tensorforge::VectorT<float, 4> v738_acc{};
          tensorforge::VectorT<float, 4> v743_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v70_data, v738_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v744_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v71_data, v743_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v745_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v72_data, v744_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v746_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v73_data, v745_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v751_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v78_data, v746_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v752_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v79_data, v751_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v753_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v80_data, v752_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v754_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v81_data, v753_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v759_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v86_data, v754_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v760_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v735_tp, v87_data, v759_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v761_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v736_tp, v88_data, v760_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v762_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v737_tp, v89_data, v761_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v767_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v734_tp, v94_data, v762_acc, 3, 3, 0);
          r8[4] = (v767_acc[0]);
          r8[5] = (v767_acc[1]);
          r8[6] = (v767_acc[2]);
          r8[7] = (v767_acc[3]);
          float v772_data = r7[8];
          float v773_data = r7[9];
          float v774_data = r7[10];
          float v775_data = r7[11];
          float v776_tp{};
          float v777_tp{};
          float v778_tp{};
          float v779_tp{};
          tensorforge::transpose4x4b32(v776_tp, v777_tp, v778_tp, v779_tp, v772_data, v773_data, v774_data, v775_data);
          tensorforge::VectorT<float, 4> v780_acc{};
          tensorforge::VectorT<float, 4> v785_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v776_tp, v70_data, v780_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v786_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v777_tp, v71_data, v785_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v787_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v778_tp, v72_data, v786_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v788_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v779_tp, v73_data, v787_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v793_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v776_tp, v78_data, v788_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v794_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v777_tp, v79_data, v793_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v795_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v778_tp, v80_data, v794_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v796_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v779_tp, v81_data, v795_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v801_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v776_tp, v86_data, v796_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v802_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v777_tp, v87_data, v801_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v803_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v778_tp, v88_data, v802_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v804_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v779_tp, v89_data, v803_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v809_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v776_tp, v94_data, v804_acc, 3, 3, 0);
          r8[8] = (v809_acc[0]);
          r8[9] = (v809_acc[1]);
          r8[10] = (v809_acc[2]);
          r8[11] = (v809_acc[3]);
          float v827_acc{};
          float v828_data = r7[12];
          float v829_bc = tensorforge::broadcast<32, 16, 0>(v828_data);
          tensorforge::fmacdpp16<0>(v827_acc, v829_bc, v70_data);
          tensorforge::fmacdpp16<1>(v827_acc, v829_bc, v71_data);
          tensorforge::fmacdpp16<2>(v827_acc, v829_bc, v72_data);
          tensorforge::fmacdpp16<3>(v827_acc, v829_bc, v73_data);
          tensorforge::fmacdpp16<4>(v827_acc, v829_bc, v78_data);
          tensorforge::fmacdpp16<5>(v827_acc, v829_bc, v79_data);
          tensorforge::fmacdpp16<6>(v827_acc, v829_bc, v80_data);
          tensorforge::fmacdpp16<7>(v827_acc, v829_bc, v81_data);
          tensorforge::fmacdpp16<8>(v827_acc, v829_bc, v86_data);
          tensorforge::fmacdpp16<9>(v827_acc, v829_bc, v87_data);
          tensorforge::fmacdpp16<10>(v827_acc, v829_bc, v88_data);
          tensorforge::fmacdpp16<11>(v827_acc, v829_bc, v89_data);
          tensorforge::fmacdpp16<12>(v827_acc, v829_bc, v94_data);
          r8[12] = v827_acc;
          float r11[13]{};
          // r11 = load{g>r}(glb_m7);
          if (v18_lead < 13) {
            #pragma unroll
            for (int32_t v835_i1 = 0; v835_i1 < 13; ++v835_i1) {
              float v843_data = __builtin_nontemporal_load(&glb_m7[(v18_lead + (v835_i1 * 13))]);
              r11[v835_i1] = v843_data;
            }
          }
          // wait(r9 = load{g>r}(glb_m6););
          float r10[13]{};
          // r10 = +(r9 * r8) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v846_data = r8[0];
          float v847_data = r8[1];
          float v848_data = r8[2];
          float v849_data = r8[3];
          float v850_tp{};
          float v851_tp{};
          float v852_tp{};
          float v853_tp{};
          tensorforge::transpose4x4b32(v850_tp, v851_tp, v852_tp, v853_tp, v846_data, v847_data, v848_data, v849_data);
          tensorforge::VectorT<float, 4> v854_acc{};
          float v855_data = r9[0];
          float v856_data = r9[1];
          float v857_data = r9[2];
          float v858_data = r9[3];
          tensorforge::VectorT<float, 4> v859_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v855_data, v854_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v860_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v856_data, v859_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v861_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v857_data, v860_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v862_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v858_data, v861_acc, 3, 0, 0);
          float v863_data = r9[4];
          float v864_data = r9[5];
          float v865_data = r9[6];
          float v866_data = r9[7];
          tensorforge::VectorT<float, 4> v867_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v863_data, v862_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v868_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v864_data, v867_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v869_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v865_data, v868_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v870_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v866_data, v869_acc, 3, 1, 0);
          float v871_data = r9[8];
          float v872_data = r9[9];
          float v873_data = r9[10];
          float v874_data = r9[11];
          tensorforge::VectorT<float, 4> v875_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v871_data, v870_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v876_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v872_data, v875_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v877_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v873_data, v876_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v878_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v874_data, v877_acc, 3, 2, 0);
          float v879_data = r9[12];
          float v880_data = r9[13];
          float v881_data = r9[14];
          float v882_data = r9[15];
          tensorforge::VectorT<float, 4> v883_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v879_data, v878_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v884_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v880_data, v883_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v885_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v881_data, v884_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v886_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v882_data, v885_acc, 3, 3, 0);
          float v887_data = r9[16];
          float v888_data = r9[17];
          float v889_data = r9[18];
          float v890_data = r9[19];
          tensorforge::VectorT<float, 4> v891_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v887_data, v886_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v892_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v888_data, v891_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v893_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v889_data, v892_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v894_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v890_data, v893_acc, 3, 4, 0);
          float v895_data = r9[20];
          float v896_data = r9[21];
          float v897_data = r9[22];
          float v898_data = r9[23];
          tensorforge::VectorT<float, 4> v899_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v895_data, v894_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v900_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v896_data, v899_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v901_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v897_data, v900_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v902_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v898_data, v901_acc, 3, 5, 0);
          float v903_data = r9[24];
          float v904_data = r9[25];
          float v905_data = r9[26];
          float v906_data = r9[27];
          tensorforge::VectorT<float, 4> v907_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v903_data, v902_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v908_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v904_data, v907_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v909_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v905_data, v908_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v910_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v906_data, v909_acc, 3, 6, 0);
          float v911_data = r9[28];
          float v912_data = r9[29];
          float v913_data = r9[30];
          float v914_data = r9[31];
          tensorforge::VectorT<float, 4> v915_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v850_tp, v911_data, v910_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v916_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v851_tp, v912_data, v915_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v917_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v852_tp, v913_data, v916_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v918_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v853_tp, v914_data, v917_acc, 3, 7, 0);
          r10[0] = (v918_acc[0]);
          r10[1] = (v918_acc[1]);
          r10[2] = (v918_acc[2]);
          r10[3] = (v918_acc[3]);
          float v923_data = r8[4];
          float v924_data = r8[5];
          float v925_data = r8[6];
          float v926_data = r8[7];
          float v927_tp{};
          float v928_tp{};
          float v929_tp{};
          float v930_tp{};
          tensorforge::transpose4x4b32(v927_tp, v928_tp, v929_tp, v930_tp, v923_data, v924_data, v925_data, v926_data);
          tensorforge::VectorT<float, 4> v931_acc{};
          tensorforge::VectorT<float, 4> v936_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v927_tp, v855_data, v931_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v937_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v928_tp, v856_data, v936_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v938_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v929_tp, v857_data, v937_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v939_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v930_tp, v858_data, v938_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v944_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v927_tp, v863_data, v939_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v945_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v928_tp, v864_data, v944_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v946_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v929_tp, v865_data, v945_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v947_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v930_tp, v866_data, v946_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v952_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v927_tp, v871_data, v947_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v953_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v928_tp, v872_data, v952_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v954_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v929_tp, v873_data, v953_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v955_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v930_tp, v874_data, v954_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v960_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v927_tp, v879_data, v955_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v961_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v928_tp, v880_data, v960_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v962_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v929_tp, v881_data, v961_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v963_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v930_tp, v882_data, v962_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v968_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v927_tp, v887_data, v963_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v969_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v928_tp, v888_data, v968_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v970_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v929_tp, v889_data, v969_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v971_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v930_tp, v890_data, v970_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v976_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v927_tp, v895_data, v971_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v977_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v928_tp, v896_data, v976_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v978_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v929_tp, v897_data, v977_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v979_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v930_tp, v898_data, v978_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v984_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v927_tp, v903_data, v979_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v985_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v928_tp, v904_data, v984_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v986_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v929_tp, v905_data, v985_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v987_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v930_tp, v906_data, v986_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v992_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v927_tp, v911_data, v987_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v993_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v928_tp, v912_data, v992_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v994_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v929_tp, v913_data, v993_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v995_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v930_tp, v914_data, v994_acc, 3, 7, 0);
          r10[4] = (v995_acc[0]);
          r10[5] = (v995_acc[1]);
          r10[6] = (v995_acc[2]);
          r10[7] = (v995_acc[3]);
          float v1000_data = r8[8];
          float v1001_data = r8[9];
          float v1002_data = r8[10];
          float v1003_data = r8[11];
          float v1004_tp{};
          float v1005_tp{};
          float v1006_tp{};
          float v1007_tp{};
          tensorforge::transpose4x4b32(v1004_tp, v1005_tp, v1006_tp, v1007_tp, v1000_data, v1001_data, v1002_data, v1003_data);
          tensorforge::VectorT<float, 4> v1008_acc{};
          tensorforge::VectorT<float, 4> v1013_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1004_tp, v855_data, v1008_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1014_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1005_tp, v856_data, v1013_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1015_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1006_tp, v857_data, v1014_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1016_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1007_tp, v858_data, v1015_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1021_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1004_tp, v863_data, v1016_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1022_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1005_tp, v864_data, v1021_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1023_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1006_tp, v865_data, v1022_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1024_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1007_tp, v866_data, v1023_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1029_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1004_tp, v871_data, v1024_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1030_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1005_tp, v872_data, v1029_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1031_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1006_tp, v873_data, v1030_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1032_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1007_tp, v874_data, v1031_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1037_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1004_tp, v879_data, v1032_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1038_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1005_tp, v880_data, v1037_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1039_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1006_tp, v881_data, v1038_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1040_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1007_tp, v882_data, v1039_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1045_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1004_tp, v887_data, v1040_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1046_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1005_tp, v888_data, v1045_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1047_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1006_tp, v889_data, v1046_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1048_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1007_tp, v890_data, v1047_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1053_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1004_tp, v895_data, v1048_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1054_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1005_tp, v896_data, v1053_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1055_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1006_tp, v897_data, v1054_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1056_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1007_tp, v898_data, v1055_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1061_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1004_tp, v903_data, v1056_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1062_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1005_tp, v904_data, v1061_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1063_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1006_tp, v905_data, v1062_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1064_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1007_tp, v906_data, v1063_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1069_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1004_tp, v911_data, v1064_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1070_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1005_tp, v912_data, v1069_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1071_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1006_tp, v913_data, v1070_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1072_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1007_tp, v914_data, v1071_acc, 3, 7, 0);
          r10[8] = (v1072_acc[0]);
          r10[9] = (v1072_acc[1]);
          r10[10] = (v1072_acc[2]);
          r10[11] = (v1072_acc[3]);
          float v1109_acc{};
          float v1110_data = r8[12];
          float v1111_bc = tensorforge::broadcast<32, 16, 0>(v1110_data);
          tensorforge::fmacdpp16<0>(v1109_acc, v1111_bc, v855_data);
          tensorforge::fmacdpp16<1>(v1109_acc, v1111_bc, v856_data);
          tensorforge::fmacdpp16<2>(v1109_acc, v1111_bc, v857_data);
          tensorforge::fmacdpp16<3>(v1109_acc, v1111_bc, v858_data);
          tensorforge::fmacdpp16<4>(v1109_acc, v1111_bc, v863_data);
          tensorforge::fmacdpp16<5>(v1109_acc, v1111_bc, v864_data);
          tensorforge::fmacdpp16<6>(v1109_acc, v1111_bc, v865_data);
          tensorforge::fmacdpp16<7>(v1109_acc, v1111_bc, v866_data);
          tensorforge::fmacdpp16<8>(v1109_acc, v1111_bc, v871_data);
          tensorforge::fmacdpp16<9>(v1109_acc, v1111_bc, v872_data);
          tensorforge::fmacdpp16<10>(v1109_acc, v1111_bc, v873_data);
          tensorforge::fmacdpp16<11>(v1109_acc, v1111_bc, v874_data);
          tensorforge::fmacdpp16<12>(v1109_acc, v1111_bc, v879_data);
          tensorforge::fmacdpp16<13>(v1109_acc, v1111_bc, v880_data);
          tensorforge::fmacdpp16<14>(v1109_acc, v1111_bc, v881_data);
          tensorforge::fmacdpp16<15>(v1109_acc, v1111_bc, v882_data);
          float v1112_bc = tensorforge::broadcast<32, 16, 1>(v1110_data);
          tensorforge::fmacdpp16<0>(v1109_acc, v1112_bc, v887_data);
          tensorforge::fmacdpp16<1>(v1109_acc, v1112_bc, v888_data);
          tensorforge::fmacdpp16<2>(v1109_acc, v1112_bc, v889_data);
          tensorforge::fmacdpp16<3>(v1109_acc, v1112_bc, v890_data);
          tensorforge::fmacdpp16<4>(v1109_acc, v1112_bc, v895_data);
          tensorforge::fmacdpp16<5>(v1109_acc, v1112_bc, v896_data);
          tensorforge::fmacdpp16<6>(v1109_acc, v1112_bc, v897_data);
          tensorforge::fmacdpp16<7>(v1109_acc, v1112_bc, v898_data);
          tensorforge::fmacdpp16<8>(v1109_acc, v1112_bc, v903_data);
          tensorforge::fmacdpp16<9>(v1109_acc, v1112_bc, v904_data);
          tensorforge::fmacdpp16<10>(v1109_acc, v1112_bc, v905_data);
          tensorforge::fmacdpp16<11>(v1109_acc, v1112_bc, v906_data);
          tensorforge::fmacdpp16<12>(v1109_acc, v1112_bc, v911_data);
          tensorforge::fmacdpp16<13>(v1109_acc, v1112_bc, v912_data);
          tensorforge::fmacdpp16<14>(v1109_acc, v1112_bc, v913_data);
          tensorforge::fmacdpp16<15>(v1109_acc, v1112_bc, v914_data);
          r10[12] = v1109_acc;
          // glb_m0 = store{r>g}(r10);
          if (v18_lead < 16) {
            #pragma unroll
            for (int32_t v1117_i1 = 0; v1117_i1 < 13; ++v1117_i1) {
              float v1119_data = r10[v1117_i1];
              int32_t v1126_a = v18_lead + (v1117_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1126_a], v1119_data);
            }
          }
          float r13[32]{};
          // r13 = load{g>r}(glb_m8);
          if (v18_lead < 16) {
            #pragma unroll
            for (int32_t v1132_i1 = 0; v1132_i1 < 32; ++v1132_i1) {
              float v1140_data = __builtin_nontemporal_load(&glb_m8[(v18_lead + (v1132_i1 * 16))]);
              r13[v1132_i1] = v1140_data;
            }
          }
          // wait(r11 = load{g>r}(glb_m7););
          float r12[13]{};
          // r12 = +(r0 * r11) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1143_data = r11[0];
          float v1144_data = r11[1];
          float v1145_data = r11[2];
          float v1146_data = r11[3];
          float v1147_tp{};
          float v1148_tp{};
          float v1149_tp{};
          float v1150_tp{};
          tensorforge::transpose4x4b32(v1147_tp, v1148_tp, v1149_tp, v1150_tp, v1143_data, v1144_data, v1145_data, v1146_data);
          tensorforge::VectorT<float, 4> v1151_acc{};
          tensorforge::VectorT<float, 4> v1156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1147_tp, v70_data, v1151_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1148_tp, v71_data, v1156_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1149_tp, v72_data, v1157_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1150_tp, v73_data, v1158_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1147_tp, v78_data, v1159_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1148_tp, v79_data, v1164_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1149_tp, v80_data, v1165_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1150_tp, v81_data, v1166_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1147_tp, v86_data, v1167_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1148_tp, v87_data, v1172_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1149_tp, v88_data, v1173_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1150_tp, v89_data, v1174_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1147_tp, v94_data, v1175_acc, 3, 3, 0);
          r12[0] = (v1180_acc[0]);
          r12[1] = (v1180_acc[1]);
          r12[2] = (v1180_acc[2]);
          r12[3] = (v1180_acc[3]);
          float v1185_data = r11[4];
          float v1186_data = r11[5];
          float v1187_data = r11[6];
          float v1188_data = r11[7];
          float v1189_tp{};
          float v1190_tp{};
          float v1191_tp{};
          float v1192_tp{};
          tensorforge::transpose4x4b32(v1189_tp, v1190_tp, v1191_tp, v1192_tp, v1185_data, v1186_data, v1187_data, v1188_data);
          tensorforge::VectorT<float, 4> v1193_acc{};
          tensorforge::VectorT<float, 4> v1198_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1189_tp, v70_data, v1193_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1199_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1190_tp, v71_data, v1198_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1191_tp, v72_data, v1199_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1192_tp, v73_data, v1200_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1206_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1189_tp, v78_data, v1201_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1207_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1190_tp, v79_data, v1206_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1191_tp, v80_data, v1207_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1192_tp, v81_data, v1208_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1189_tp, v86_data, v1209_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1190_tp, v87_data, v1214_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1191_tp, v88_data, v1215_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1192_tp, v89_data, v1216_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1189_tp, v94_data, v1217_acc, 3, 3, 0);
          r12[4] = (v1222_acc[0]);
          r12[5] = (v1222_acc[1]);
          r12[6] = (v1222_acc[2]);
          r12[7] = (v1222_acc[3]);
          float v1227_data = r11[8];
          float v1228_data = r11[9];
          float v1229_data = r11[10];
          float v1230_data = r11[11];
          float v1231_tp{};
          float v1232_tp{};
          float v1233_tp{};
          float v1234_tp{};
          tensorforge::transpose4x4b32(v1231_tp, v1232_tp, v1233_tp, v1234_tp, v1227_data, v1228_data, v1229_data, v1230_data);
          tensorforge::VectorT<float, 4> v1235_acc{};
          tensorforge::VectorT<float, 4> v1240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1231_tp, v70_data, v1235_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1232_tp, v71_data, v1240_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1233_tp, v72_data, v1241_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1234_tp, v73_data, v1242_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1231_tp, v78_data, v1243_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1232_tp, v79_data, v1248_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1233_tp, v80_data, v1249_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1234_tp, v81_data, v1250_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1231_tp, v86_data, v1251_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1232_tp, v87_data, v1256_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1233_tp, v88_data, v1257_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1234_tp, v89_data, v1258_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1231_tp, v94_data, v1259_acc, 3, 3, 0);
          r12[8] = (v1264_acc[0]);
          r12[9] = (v1264_acc[1]);
          r12[10] = (v1264_acc[2]);
          r12[11] = (v1264_acc[3]);
          float v1282_acc{};
          float v1283_data = r11[12];
          float v1284_bc = tensorforge::broadcast<32, 16, 0>(v1283_data);
          tensorforge::fmacdpp16<0>(v1282_acc, v1284_bc, v70_data);
          tensorforge::fmacdpp16<1>(v1282_acc, v1284_bc, v71_data);
          tensorforge::fmacdpp16<2>(v1282_acc, v1284_bc, v72_data);
          tensorforge::fmacdpp16<3>(v1282_acc, v1284_bc, v73_data);
          tensorforge::fmacdpp16<4>(v1282_acc, v1284_bc, v78_data);
          tensorforge::fmacdpp16<5>(v1282_acc, v1284_bc, v79_data);
          tensorforge::fmacdpp16<6>(v1282_acc, v1284_bc, v80_data);
          tensorforge::fmacdpp16<7>(v1282_acc, v1284_bc, v81_data);
          tensorforge::fmacdpp16<8>(v1282_acc, v1284_bc, v86_data);
          tensorforge::fmacdpp16<9>(v1282_acc, v1284_bc, v87_data);
          tensorforge::fmacdpp16<10>(v1282_acc, v1284_bc, v88_data);
          tensorforge::fmacdpp16<11>(v1282_acc, v1284_bc, v89_data);
          tensorforge::fmacdpp16<12>(v1282_acc, v1284_bc, v94_data);
          r12[12] = v1282_acc;
          // wait(r13 = load{g>r}(glb_m8););
          float r14[13]{};
          // r14 = +(r13 * r12) + None
          // [(0, 16), (0, 13)] [(0, 32)]
          float v1286_data = r12[0];
          float v1287_data = r12[1];
          float v1288_data = r12[2];
          float v1289_data = r12[3];
          float v1290_tp{};
          float v1291_tp{};
          float v1292_tp{};
          float v1293_tp{};
          tensorforge::transpose4x4b32(v1290_tp, v1291_tp, v1292_tp, v1293_tp, v1286_data, v1287_data, v1288_data, v1289_data);
          tensorforge::VectorT<float, 4> v1294_acc{};
          float v1295_data = r13[0];
          float v1296_data = r13[1];
          float v1297_data = r13[2];
          float v1298_data = r13[3];
          tensorforge::VectorT<float, 4> v1299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1290_tp, v1295_data, v1294_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1291_tp, v1296_data, v1299_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1292_tp, v1297_data, v1300_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1293_tp, v1298_data, v1301_acc, 3, 0, 0);
          float v1303_data = r13[4];
          float v1304_data = r13[5];
          float v1305_data = r13[6];
          float v1306_data = r13[7];
          tensorforge::VectorT<float, 4> v1307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1290_tp, v1303_data, v1302_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1291_tp, v1304_data, v1307_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1292_tp, v1305_data, v1308_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1293_tp, v1306_data, v1309_acc, 3, 1, 0);
          float v1311_data = r13[8];
          float v1312_data = r13[9];
          float v1313_data = r13[10];
          float v1314_data = r13[11];
          tensorforge::VectorT<float, 4> v1315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1290_tp, v1311_data, v1310_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1291_tp, v1312_data, v1315_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1292_tp, v1313_data, v1316_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1293_tp, v1314_data, v1317_acc, 3, 2, 0);
          float v1319_data = r13[12];
          float v1320_data = r13[13];
          float v1321_data = r13[14];
          float v1322_data = r13[15];
          tensorforge::VectorT<float, 4> v1323_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1290_tp, v1319_data, v1318_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1324_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1291_tp, v1320_data, v1323_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1325_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1292_tp, v1321_data, v1324_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1326_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1293_tp, v1322_data, v1325_acc, 3, 3, 0);
          float v1327_data = r13[16];
          float v1328_data = r13[17];
          float v1329_data = r13[18];
          float v1330_data = r13[19];
          tensorforge::VectorT<float, 4> v1331_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1290_tp, v1327_data, v1326_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1332_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1291_tp, v1328_data, v1331_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1333_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1292_tp, v1329_data, v1332_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1334_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1293_tp, v1330_data, v1333_acc, 3, 4, 0);
          float v1335_data = r13[20];
          float v1336_data = r13[21];
          float v1337_data = r13[22];
          float v1338_data = r13[23];
          tensorforge::VectorT<float, 4> v1339_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1290_tp, v1335_data, v1334_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1291_tp, v1336_data, v1339_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1292_tp, v1337_data, v1340_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1293_tp, v1338_data, v1341_acc, 3, 5, 0);
          float v1343_data = r13[24];
          float v1344_data = r13[25];
          float v1345_data = r13[26];
          float v1346_data = r13[27];
          tensorforge::VectorT<float, 4> v1347_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1290_tp, v1343_data, v1342_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1291_tp, v1344_data, v1347_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1292_tp, v1345_data, v1348_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1293_tp, v1346_data, v1349_acc, 3, 6, 0);
          float v1351_data = r13[28];
          float v1352_data = r13[29];
          float v1353_data = r13[30];
          float v1354_data = r13[31];
          tensorforge::VectorT<float, 4> v1355_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1290_tp, v1351_data, v1350_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1291_tp, v1352_data, v1355_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1292_tp, v1353_data, v1356_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1293_tp, v1354_data, v1357_acc, 3, 7, 0);
          r14[0] = (v1358_acc[0]);
          r14[1] = (v1358_acc[1]);
          r14[2] = (v1358_acc[2]);
          r14[3] = (v1358_acc[3]);
          float v1363_data = r12[4];
          float v1364_data = r12[5];
          float v1365_data = r12[6];
          float v1366_data = r12[7];
          float v1367_tp{};
          float v1368_tp{};
          float v1369_tp{};
          float v1370_tp{};
          tensorforge::transpose4x4b32(v1367_tp, v1368_tp, v1369_tp, v1370_tp, v1363_data, v1364_data, v1365_data, v1366_data);
          tensorforge::VectorT<float, 4> v1371_acc{};
          tensorforge::VectorT<float, 4> v1376_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1367_tp, v1295_data, v1371_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1377_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1368_tp, v1296_data, v1376_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1378_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1369_tp, v1297_data, v1377_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1379_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1370_tp, v1298_data, v1378_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1367_tp, v1303_data, v1379_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1368_tp, v1304_data, v1384_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1369_tp, v1305_data, v1385_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1370_tp, v1306_data, v1386_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1392_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1367_tp, v1311_data, v1387_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1368_tp, v1312_data, v1392_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1369_tp, v1313_data, v1393_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1370_tp, v1314_data, v1394_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1400_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1367_tp, v1319_data, v1395_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1368_tp, v1320_data, v1400_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1369_tp, v1321_data, v1401_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1370_tp, v1322_data, v1402_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1408_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1367_tp, v1327_data, v1403_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1368_tp, v1328_data, v1408_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1369_tp, v1329_data, v1409_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1370_tp, v1330_data, v1410_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1367_tp, v1335_data, v1411_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1368_tp, v1336_data, v1416_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1418_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1369_tp, v1337_data, v1417_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1419_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1370_tp, v1338_data, v1418_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1424_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1367_tp, v1343_data, v1419_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1425_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1368_tp, v1344_data, v1424_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1426_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1369_tp, v1345_data, v1425_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1427_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1370_tp, v1346_data, v1426_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1367_tp, v1351_data, v1427_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1368_tp, v1352_data, v1432_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1434_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1369_tp, v1353_data, v1433_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1435_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1370_tp, v1354_data, v1434_acc, 3, 7, 0);
          r14[4] = (v1435_acc[0]);
          r14[5] = (v1435_acc[1]);
          r14[6] = (v1435_acc[2]);
          r14[7] = (v1435_acc[3]);
          float v1440_data = r12[8];
          float v1441_data = r12[9];
          float v1442_data = r12[10];
          float v1443_data = r12[11];
          float v1444_tp{};
          float v1445_tp{};
          float v1446_tp{};
          float v1447_tp{};
          tensorforge::transpose4x4b32(v1444_tp, v1445_tp, v1446_tp, v1447_tp, v1440_data, v1441_data, v1442_data, v1443_data);
          tensorforge::VectorT<float, 4> v1448_acc{};
          tensorforge::VectorT<float, 4> v1453_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1444_tp, v1295_data, v1448_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1445_tp, v1296_data, v1453_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1446_tp, v1297_data, v1454_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1447_tp, v1298_data, v1455_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1461_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1444_tp, v1303_data, v1456_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1462_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1445_tp, v1304_data, v1461_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1463_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1446_tp, v1305_data, v1462_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1464_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1447_tp, v1306_data, v1463_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1469_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1444_tp, v1311_data, v1464_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1470_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1445_tp, v1312_data, v1469_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1471_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1446_tp, v1313_data, v1470_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1472_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1447_tp, v1314_data, v1471_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1477_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1444_tp, v1319_data, v1472_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1478_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1445_tp, v1320_data, v1477_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1479_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1446_tp, v1321_data, v1478_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1480_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1447_tp, v1322_data, v1479_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v1485_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1444_tp, v1327_data, v1480_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1486_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1445_tp, v1328_data, v1485_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1487_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1446_tp, v1329_data, v1486_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1488_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1447_tp, v1330_data, v1487_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v1493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1444_tp, v1335_data, v1488_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1445_tp, v1336_data, v1493_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1495_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1446_tp, v1337_data, v1494_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1496_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1447_tp, v1338_data, v1495_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v1501_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1444_tp, v1343_data, v1496_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1502_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1445_tp, v1344_data, v1501_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1503_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1446_tp, v1345_data, v1502_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1504_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1447_tp, v1346_data, v1503_acc, 3, 6, 0);
          tensorforge::VectorT<float, 4> v1509_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1444_tp, v1351_data, v1504_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1510_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1445_tp, v1352_data, v1509_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1511_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1446_tp, v1353_data, v1510_acc, 3, 7, 0);
          tensorforge::VectorT<float, 4> v1512_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1447_tp, v1354_data, v1511_acc, 3, 7, 0);
          r14[8] = (v1512_acc[0]);
          r14[9] = (v1512_acc[1]);
          r14[10] = (v1512_acc[2]);
          r14[11] = (v1512_acc[3]);
          float v1549_acc{};
          float v1550_data = r12[12];
          float v1551_bc = tensorforge::broadcast<32, 16, 0>(v1550_data);
          tensorforge::fmacdpp16<0>(v1549_acc, v1551_bc, v1295_data);
          tensorforge::fmacdpp16<1>(v1549_acc, v1551_bc, v1296_data);
          tensorforge::fmacdpp16<2>(v1549_acc, v1551_bc, v1297_data);
          tensorforge::fmacdpp16<3>(v1549_acc, v1551_bc, v1298_data);
          tensorforge::fmacdpp16<4>(v1549_acc, v1551_bc, v1303_data);
          tensorforge::fmacdpp16<5>(v1549_acc, v1551_bc, v1304_data);
          tensorforge::fmacdpp16<6>(v1549_acc, v1551_bc, v1305_data);
          tensorforge::fmacdpp16<7>(v1549_acc, v1551_bc, v1306_data);
          tensorforge::fmacdpp16<8>(v1549_acc, v1551_bc, v1311_data);
          tensorforge::fmacdpp16<9>(v1549_acc, v1551_bc, v1312_data);
          tensorforge::fmacdpp16<10>(v1549_acc, v1551_bc, v1313_data);
          tensorforge::fmacdpp16<11>(v1549_acc, v1551_bc, v1314_data);
          tensorforge::fmacdpp16<12>(v1549_acc, v1551_bc, v1319_data);
          tensorforge::fmacdpp16<13>(v1549_acc, v1551_bc, v1320_data);
          tensorforge::fmacdpp16<14>(v1549_acc, v1551_bc, v1321_data);
          tensorforge::fmacdpp16<15>(v1549_acc, v1551_bc, v1322_data);
          float v1552_bc = tensorforge::broadcast<32, 16, 1>(v1550_data);
          tensorforge::fmacdpp16<0>(v1549_acc, v1552_bc, v1327_data);
          tensorforge::fmacdpp16<1>(v1549_acc, v1552_bc, v1328_data);
          tensorforge::fmacdpp16<2>(v1549_acc, v1552_bc, v1329_data);
          tensorforge::fmacdpp16<3>(v1549_acc, v1552_bc, v1330_data);
          tensorforge::fmacdpp16<4>(v1549_acc, v1552_bc, v1335_data);
          tensorforge::fmacdpp16<5>(v1549_acc, v1552_bc, v1336_data);
          tensorforge::fmacdpp16<6>(v1549_acc, v1552_bc, v1337_data);
          tensorforge::fmacdpp16<7>(v1549_acc, v1552_bc, v1338_data);
          tensorforge::fmacdpp16<8>(v1549_acc, v1552_bc, v1343_data);
          tensorforge::fmacdpp16<9>(v1549_acc, v1552_bc, v1344_data);
          tensorforge::fmacdpp16<10>(v1549_acc, v1552_bc, v1345_data);
          tensorforge::fmacdpp16<11>(v1549_acc, v1552_bc, v1346_data);
          tensorforge::fmacdpp16<12>(v1549_acc, v1552_bc, v1351_data);
          tensorforge::fmacdpp16<13>(v1549_acc, v1552_bc, v1352_data);
          tensorforge::fmacdpp16<14>(v1549_acc, v1552_bc, v1353_data);
          tensorforge::fmacdpp16<15>(v1549_acc, v1552_bc, v1354_data);
          r14[12] = v1549_acc;
          // glb_m0 = store{r>g}(r14);
          if (v18_lead < 16) {
            #pragma unroll
            for (int32_t v1557_i1 = 0; v1557_i1 < 13; ++v1557_i1) {
              float v1559_data = r14[v1557_i1];
              int32_t v1566_a = v18_lead + (v1557_i1 * 32);
              __builtin_amdgcn_global_atomic_fadd_f32(&glb_m0[v1566_a], v1559_data);
            }
          }
          float r15[13]{};
          // r15 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v1571_i0 = 0; v1571_i0 < 1; ++v1571_i0) {
            int32_t v1577_lead = v18_lead + (v1571_i0 * 32);
            #pragma unroll
            for (int32_t v1572_i1 = 0; v1572_i1 < 13; ++v1572_i1) {
              float v1580_data = glb_m0[(v1577_lead + (v1572_i1 * 32))];
              r15[(v1571_i0 + v1572_i1)] = v1580_data;
            }
          }
          float r16[13]{};
          // r16 = load{g>r}(glb_m10);
          if (v18_lead < 13) {
            #pragma unroll
            for (int32_t v1587_i1 = 0; v1587_i1 < 13; ++v1587_i1) {
              float v1595_data = __builtin_nontemporal_load(&glb_m10[(v18_lead + (v1587_i1 * 13))]);
              r16[v1587_i1] = v1595_data;
            }
          }
          // wait(r15 = load{g>r}(glb_m0););
          // wait(r16 = load{g>r}(glb_m10););
          float r17[13]{};
          // r17 = +(r15 * r16) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float v1598_data = r16[0];
          float v1599_data = r16[1];
          float v1600_data = r16[2];
          float v1601_data = r16[3];
          float v1602_tp{};
          float v1603_tp{};
          float v1604_tp{};
          float v1605_tp{};
          tensorforge::transpose4x4b32(v1602_tp, v1603_tp, v1604_tp, v1605_tp, v1598_data, v1599_data, v1600_data, v1601_data);
          tensorforge::VectorT<float, 4> v1606_acc{};
          float v1607_data = r15[0];
          float v1608_data = r15[1];
          float v1609_data = r15[2];
          float v1610_data = r15[3];
          tensorforge::VectorT<float, 4> v1611_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1602_tp, v1607_data, v1606_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1612_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1603_tp, v1608_data, v1611_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1613_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1604_tp, v1609_data, v1612_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1614_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1605_tp, v1610_data, v1613_acc, 3, 0, 0);
          float v1615_data = r15[4];
          float v1616_data = r15[5];
          float v1617_data = r15[6];
          float v1618_data = r15[7];
          tensorforge::VectorT<float, 4> v1619_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1602_tp, v1615_data, v1614_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1620_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1603_tp, v1616_data, v1619_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1621_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1604_tp, v1617_data, v1620_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1622_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1605_tp, v1618_data, v1621_acc, 3, 1, 0);
          float v1623_data = r15[8];
          float v1624_data = r15[9];
          float v1625_data = r15[10];
          float v1626_data = r15[11];
          tensorforge::VectorT<float, 4> v1627_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1602_tp, v1623_data, v1622_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1628_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1603_tp, v1624_data, v1627_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1629_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1604_tp, v1625_data, v1628_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1630_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1605_tp, v1626_data, v1629_acc, 3, 2, 0);
          float v1631_data = r15[12];
          tensorforge::VectorT<float, 4> v1635_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1602_tp, v1631_data, v1630_acc, 3, 3, 0);
          r17[0] = (v1635_acc[0]);
          r17[1] = (v1635_acc[1]);
          r17[2] = (v1635_acc[2]);
          r17[3] = (v1635_acc[3]);
          float v1640_data = r16[4];
          float v1641_data = r16[5];
          float v1642_data = r16[6];
          float v1643_data = r16[7];
          float v1644_tp{};
          float v1645_tp{};
          float v1646_tp{};
          float v1647_tp{};
          tensorforge::transpose4x4b32(v1644_tp, v1645_tp, v1646_tp, v1647_tp, v1640_data, v1641_data, v1642_data, v1643_data);
          tensorforge::VectorT<float, 4> v1648_acc{};
          tensorforge::VectorT<float, 4> v1653_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1644_tp, v1607_data, v1648_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1654_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1645_tp, v1608_data, v1653_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1655_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1646_tp, v1609_data, v1654_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1656_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1647_tp, v1610_data, v1655_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1661_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1644_tp, v1615_data, v1656_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1662_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1645_tp, v1616_data, v1661_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1663_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1646_tp, v1617_data, v1662_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1664_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1647_tp, v1618_data, v1663_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1669_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1644_tp, v1623_data, v1664_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1670_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1645_tp, v1624_data, v1669_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1671_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1646_tp, v1625_data, v1670_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1672_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1647_tp, v1626_data, v1671_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1677_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1644_tp, v1631_data, v1672_acc, 3, 3, 0);
          r17[4] = (v1677_acc[0]);
          r17[5] = (v1677_acc[1]);
          r17[6] = (v1677_acc[2]);
          r17[7] = (v1677_acc[3]);
          float v1682_data = r16[8];
          float v1683_data = r16[9];
          float v1684_data = r16[10];
          float v1685_data = r16[11];
          float v1686_tp{};
          float v1687_tp{};
          float v1688_tp{};
          float v1689_tp{};
          tensorforge::transpose4x4b32(v1686_tp, v1687_tp, v1688_tp, v1689_tp, v1682_data, v1683_data, v1684_data, v1685_data);
          tensorforge::VectorT<float, 4> v1690_acc{};
          tensorforge::VectorT<float, 4> v1695_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1686_tp, v1607_data, v1690_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1696_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1687_tp, v1608_data, v1695_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1697_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1688_tp, v1609_data, v1696_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1698_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1689_tp, v1610_data, v1697_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v1703_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1686_tp, v1615_data, v1698_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1704_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1687_tp, v1616_data, v1703_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1705_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1688_tp, v1617_data, v1704_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1706_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1689_tp, v1618_data, v1705_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v1711_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1686_tp, v1623_data, v1706_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1712_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1687_tp, v1624_data, v1711_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1713_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1688_tp, v1625_data, v1712_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1714_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1689_tp, v1626_data, v1713_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v1719_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v1686_tp, v1631_data, v1714_acc, 3, 3, 0);
          r17[8] = (v1719_acc[0]);
          r17[9] = (v1719_acc[1]);
          r17[10] = (v1719_acc[2]);
          r17[11] = (v1719_acc[3]);
          float v1737_acc{};
          float v1738_data = r16[12];
          float v1739_bc = tensorforge::broadcast<32, 16, 0>(v1738_data);
          tensorforge::fmacdpp16<0>(v1737_acc, v1739_bc, v1607_data);
          tensorforge::fmacdpp16<1>(v1737_acc, v1739_bc, v1608_data);
          tensorforge::fmacdpp16<2>(v1737_acc, v1739_bc, v1609_data);
          tensorforge::fmacdpp16<3>(v1737_acc, v1739_bc, v1610_data);
          tensorforge::fmacdpp16<4>(v1737_acc, v1739_bc, v1615_data);
          tensorforge::fmacdpp16<5>(v1737_acc, v1739_bc, v1616_data);
          tensorforge::fmacdpp16<6>(v1737_acc, v1739_bc, v1617_data);
          tensorforge::fmacdpp16<7>(v1737_acc, v1739_bc, v1618_data);
          tensorforge::fmacdpp16<8>(v1737_acc, v1739_bc, v1623_data);
          tensorforge::fmacdpp16<9>(v1737_acc, v1739_bc, v1624_data);
          tensorforge::fmacdpp16<10>(v1737_acc, v1739_bc, v1625_data);
          tensorforge::fmacdpp16<11>(v1737_acc, v1739_bc, v1626_data);
          tensorforge::fmacdpp16<12>(v1737_acc, v1739_bc, v1631_data);
          r17[12] = v1737_acc;
          // glb_m9 = store{r>g}(r17);
          #pragma unroll
          for (int32_t v1743_i0 = 0; v1743_i0 < 1; ++v1743_i0) {
            int32_t v1751_lead = v18_lead + (v1743_i0 * 32);
            #pragma unroll
            for (int32_t v1744_i1 = 0; v1744_i1 < 13; ++v1744_i1) {
              float v1746_data = r17[(v1743_i0 + v1744_i1)];
              glb_m9[(v1751_lead + (v1744_i1 * 32))] = v1746_data;
            }
          }
        }
      }
    }
  }
}

