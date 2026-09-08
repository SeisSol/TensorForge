// === base name ===
kernel_8c9d1a8467

// === header ===
void launcher_kernel_8c9d1a8467(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8c9d1a8467(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8c9d1a8467, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_8c9d1a8467), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_8c9d1a8467, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8c9d1a8467(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // C = abs(M)
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v11_lead = threadIdx.x % 64;
          if (v11_lead < 8) {
            #pragma unroll
            for (int32_t v13_i1 = 0; v13_i1 < 8; ++v13_i1) {
              float v21_data = __builtin_nontemporal_load(&glb_m1[(v11_lead + (v13_i1 * 8))]);
              r0[v13_i1] = v21_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          if (v11_lead < 8) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
              float v36_data = __builtin_nontemporal_load(&glb_m2[(v11_lead + (v28_i1 * 8))]);
              r1[v28_i1] = v36_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v39_data = r1[0];
          float v40_data = r1[1];
          float v41_data = r1[2];
          float v42_data = r1[3];
          float v43_tp{};
          float v44_tp{};
          float v45_tp{};
          float v46_tp{};
          tensorforge::transpose4x4b32(v43_tp, v44_tp, v45_tp, v46_tp, v39_data, v40_data, v41_data, v42_data);
          tensorforge::VectorT<float, 4> v47_acc{};
          float v48_data = r0[0];
          float v49_data = r0[1];
          float v50_data = r0[2];
          float v51_data = r0[3];
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v48_data, v47_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v52_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v53_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v54_acc, 4, 0, 0);
          float v56_data = r0[4];
          float v57_data = r0[5];
          float v58_data = r0[6];
          float v59_data = r0[7];
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v56_data, v55_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v60_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v61_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v62_acc, 4, 1, 0);
          r2[0] = (v63_acc[0]);
          r2[1] = (v63_acc[1]);
          r2[2] = (v63_acc[2]);
          r2[3] = (v63_acc[3]);
          float v68_data = r1[4];
          float v69_data = r1[5];
          float v70_data = r1[6];
          float v71_data = r1[7];
          float v72_tp{};
          float v73_tp{};
          float v74_tp{};
          float v75_tp{};
          tensorforge::transpose4x4b32(v72_tp, v73_tp, v74_tp, v75_tp, v68_data, v69_data, v70_data, v71_data);
          tensorforge::VectorT<float, 4> v76_acc{};
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v48_data, v76_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v49_data, v81_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v50_data, v82_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v51_data, v83_acc, 4, 0, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v56_data, v84_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v57_data, v89_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v58_data, v90_acc, 4, 1, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v59_data, v91_acc, 4, 1, 0);
          r2[4] = (v92_acc[0]);
          r2[5] = (v92_acc[1]);
          r2[6] = (v92_acc[2]);
          r2[7] = (v92_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v11_lead < 8) {
            #pragma unroll
            for (int32_t v101_i1 = 0; v101_i1 < 8; ++v101_i1) {
              float v103_data = r2[v101_i1];
              glb_m0[(v11_lead + (v101_i1 * 8))] = v103_data;
            }
          }
          // glb_m3 = abs(glb_m0)
          if (v11_lead < 8) {
            #pragma unroll
            for (int32_t v115_k1 = 0; v115_k1 < 8; ++v115_k1) {
              int32_t v121_a = v115_k1 * 8;
              float v123_data = glb_m0[(v11_lead + v121_a)];
              glb_m3[(v11_lead + v121_a)] = (fabsf(v123_data));
            }
          }
        }
      }
    }
  }
}

