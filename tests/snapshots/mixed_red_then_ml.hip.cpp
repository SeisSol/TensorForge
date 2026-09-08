// === base name ===
kernel_49337a255f

// === header ===
void launcher_kernel_49337a255f(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_49337a255f(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_49337a255f, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_49337a255f), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_49337a255f, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_49337a255f(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // TMP = +(A, dims=[1])
    // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8(8) {0..8} pointer_based({0..8})[0]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1]
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
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          int32_t v10_lead = threadIdx.x % 32;
          if (v10_lead < 8) {
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
              float v20_data = __builtin_nontemporal_load(&glb_m2[(v10_lead + (v12_i1 * 8))]);
              r1[v12_i1] = v20_data;
            }
          }
          float r0[1]{};
          // r0 = +(glb_m0, dims=[1])
          if (v10_lead < 8) {
            float v28_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v27_r1 = 0; v27_r1 < 8; ++v27_r1) {
              float v36_data = glb_m0[(v10_lead + (v27_r1 * 8))];
              v28_acc0 = (v28_acc0 + v36_data);
            }
            r0[0] = v28_acc0;
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] []
          float v40_data = r1[0];
          float v41_data = r1[1];
          float v42_data = r1[2];
          float v43_data = r1[3];
          float v44_tp{};
          float v45_tp{};
          float v46_tp{};
          float v47_tp{};
          tensorforge::transpose4x4b32(v44_tp, v45_tp, v46_tp, v47_tp, v40_data, v41_data, v42_data, v43_data);
          tensorforge::VectorT<float, 4> v48_acc{};
          float v49_data = r0[0];
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v48_acc, 3, 0, 0);
          r2[0] = (v53_acc[0]);
          r2[1] = (v53_acc[1]);
          r2[2] = (v53_acc[2]);
          r2[3] = (v53_acc[3]);
          float v58_data = r1[4];
          float v59_data = r1[5];
          float v60_data = r1[6];
          float v61_data = r1[7];
          float v62_tp{};
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          tensorforge::transpose4x4b32(v62_tp, v63_tp, v64_tp, v65_tp, v58_data, v59_data, v60_data, v61_data);
          tensorforge::VectorT<float, 4> v66_acc{};
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v49_data, v66_acc, 3, 0, 0);
          r2[4] = (v71_acc[0]);
          r2[5] = (v71_acc[1]);
          r2[6] = (v71_acc[2]);
          r2[7] = (v71_acc[3]);
          // glb_m1 = store{r>g}(r2);
          if (v10_lead < 8) {
            #pragma unroll
            for (int32_t v80_i1 = 0; v80_i1 < 8; ++v80_i1) {
              float v82_data = r2[v80_i1];
              glb_m1[(v10_lead + (v80_i1 * 8))] = v82_data;
            }
          }
        }
      }
    }
  }
}

