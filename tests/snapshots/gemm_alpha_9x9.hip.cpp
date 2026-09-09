// === base name ===
kernel_708843a027e716b2

// === header ===
void launcher_kernel_708843a027e716b2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_708843a027e716b2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_708843a027e716b2, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_708843a027e716b2), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_708843a027e716b2, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_708843a027e716b2(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 9×9(9×9) {0..9}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 9×9(9×9) {0..9}×{0..9} strided
    // m3 ()  scalar
    // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 16;
          if (v13_lead < 9) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 9; ++v15_i1) {
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v13_lead + (v15_i1 * 9))]);
              r0[v15_i1] = v23_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m2);
          if (v13_lead < 9) {
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 9; ++v30_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m2[(v13_lead + (v30_i1 * 9))]);
              r1[v30_i1] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir2[9]{};
          float v42_data = r1[0];
          float v43_data = r1[1];
          float v44_data = r1[2];
          float v45_data = r1[3];
          float v46_tp{};
          float v47_tp{};
          float v48_tp{};
          float v49_tp{};
          tensorforge::transpose4x4b32(v46_tp, v47_tp, v48_tp, v49_tp, v42_data, v43_data, v44_data, v45_data);
          tensorforge::VectorT<float, 4> v50_acc{};
          float v51_data = r0[0];
          float v52_data = r0[1];
          float v53_data = r0[2];
          float v54_data = r0[3];
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v50_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
          float v59_data = r0[4];
          float v60_data = r0[5];
          float v61_data = r0[6];
          float v62_data = r0[7];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v58_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
          float v67_data = r0[8];
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v66_acc, 2, 2, 0);
          ir2[0] = (v71_acc[0]);
          ir2[1] = (v71_acc[1]);
          ir2[2] = (v71_acc[2]);
          ir2[3] = (v71_acc[3]);
          float v76_data = r1[4];
          float v77_data = r1[5];
          float v78_data = r1[6];
          float v79_data = r1[7];
          float v80_tp{};
          float v81_tp{};
          float v82_tp{};
          float v83_tp{};
          tensorforge::transpose4x4b32(v80_tp, v81_tp, v82_tp, v83_tp, v76_data, v77_data, v78_data, v79_data);
          tensorforge::VectorT<float, 4> v84_acc{};
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v51_data, v84_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v52_data, v89_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v53_data, v90_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v54_data, v91_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v59_data, v92_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v81_tp, v60_data, v97_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v61_data, v98_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v62_data, v99_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v80_tp, v67_data, v100_acc, 2, 2, 0);
          ir2[4] = (v105_acc[0]);
          ir2[5] = (v105_acc[1]);
          ir2[6] = (v105_acc[2]);
          ir2[7] = (v105_acc[3]);
          float v119_acc{};
          float v120_data = r1[8];
          tensorforge::fmacdpp16<0>(v119_acc, v120_data, v51_data);
          tensorforge::fmacdpp16<1>(v119_acc, v120_data, v52_data);
          tensorforge::fmacdpp16<2>(v119_acc, v120_data, v53_data);
          tensorforge::fmacdpp16<3>(v119_acc, v120_data, v54_data);
          tensorforge::fmacdpp16<4>(v119_acc, v120_data, v59_data);
          tensorforge::fmacdpp16<5>(v119_acc, v120_data, v60_data);
          tensorforge::fmacdpp16<6>(v119_acc, v120_data, v61_data);
          tensorforge::fmacdpp16<7>(v119_acc, v120_data, v62_data);
          tensorforge::fmacdpp16<8>(v119_acc, v120_data, v67_data);
          ir2[8] = v119_acc;
          if (v13_lead < 9) {
            #pragma unroll
            for (int32_t v126_n1 = 0; v126_n1 < 9; ++v126_n1) {
              float v128_data = ir2[v126_n1];
              r2[v126_n1] = (v128_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v13_lead < 9) {
            #pragma unroll
            for (int32_t v135_i1 = 0; v135_i1 < 9; ++v135_i1) {
              float v137_data = r2[v135_i1];
              glb_m0[(v13_lead + (v135_i1 * 9))] = v137_data;
            }
          }
        }
      }
    }
  }
}

