// === base name ===
kernel_0b2fc070b9

// === header ===
void launcher_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (2, 128, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0b2fc070b9, block.x * block.y * block.z, 256 * sizeof(__float128)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_0b2fc070b9), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(__float128)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_0b2fc070b9, grid, block, 256 * sizeof(__float128), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 2×2(2×2) {0..2}×{0..2} strided
    // m1 2×2(2×2) {0..2}×{0..2} strided
    // m2 2×2(2×2) {0..2}×{0..2} strided
    // m0 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, 1] = m1 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, -1]×m2 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<__float128*>(totalShrMemPtr);
      __float128* localShrMem0 = &totalShrMem[2 * threadIdx.y + 0];
      __float128* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          __float128 *const __restrict__ glb_m0 = &m0[batchId0 * 4 + 0 + m0_extraOffset];
          const __float128 *const __restrict__ glb_m1 = &m1[batchId0 * 4 + 0 + m1_extraOffset];
          const __float128 *const __restrict__ glb_m2 = &m2[batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 2;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 2; ++v9_i1) {
              int32_t v15_a = v9_i1 * 2;
              int32_t v16_a = v14_lead + v15_a;
              __float128 v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + v9_i1)] = v24_data;
            }
          }
          __float128 r1[2]{};
          // r1 = load{g>r}(glb_m2);
          __float128 v27_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v27_lin;
          __float128 v28_lin = glb_m2[2 + threadIdx.x * 1];
          r1[1] = v28_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          __float128 r2[2]{};
          // r2 = +(r0 * r1) + None
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 v30_data = r0[0];
          __float128 v31_data = r0[1];
          __float128 v32_acc{};
          __float128 v33_acc{};
          __float128 v34_lin = r1[0];
          v32_acc += ((tensorforge::broadcast<2, 1, 0>(v34_lin)) * v30_data);
          v32_acc += ((tensorforge::broadcast<2, 1, 1>(v34_lin)) * v31_data);
          __float128 v39_lin = r1[1];
          v33_acc += ((tensorforge::broadcast<2, 1, 0>(v39_lin)) * v30_data);
          v33_acc += ((tensorforge::broadcast<2, 1, 1>(v39_lin)) * v31_data);
          r2[0] = v32_acc;
          r2[1] = v33_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v47_i0 = 0; v47_i0 < 1; ++v47_i0) {
            int32_t v56_lead = v7_lead + (v47_i0 * 2);
            #pragma unroll
            for (int32_t v48_i1 = 0; v48_i1 < 2; ++v48_i1) {
              int32_t v49_a = v47_i0 + v48_i1;
              __float128 v51_data = r2[(v47_i0 + v48_i1)];
              glb_m0[(v56_lead + (v48_i1 * 2))] = v51_data;
            }
          }
        }
      }
    }
  }
}

