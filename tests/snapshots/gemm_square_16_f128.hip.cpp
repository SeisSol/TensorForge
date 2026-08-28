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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          __float128 *const __restrict__ glb_m0 = &m0[batchId0 * 4 + 0 + m0_extraOffset];
          const __float128 *const __restrict__ glb_m1 = &m1[batchId0 * 4 + 0 + m1_extraOffset];
          const __float128 *const __restrict__ glb_m2 = &m2[batchId0 * 4 + 0 + m2_extraOffset];
          __float128 r0[2]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 2;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 2);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 2; ++v12_i1) {
              __float128 v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v12_i1 * 2))]);
              r0[(v11_i0 + v12_i1)] = v20_data;
            }
          }
          __float128 r1[2]{};
          // r1 = load{g>r}(glb_m2);
          __float128 v23_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v23_lin;
          __float128 v24_lin = glb_m2[2 + threadIdx.x * 1];
          r1[1] = v24_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          __float128 r2[2]{};
          // r2 = +(r0 * r1) + None
          // [(0, 2), (0, 2)] [(0, 2)]
          __float128 v26_data = r0[0];
          __float128 v27_data = r0[1];
          __float128 v28_acc{};
          __float128 v29_acc{};
          __float128 v30_lin = r1[0];
          v28_acc += ((tensorforge::broadcast<2, 1, 0>(v30_lin)) * v26_data);
          v28_acc += ((tensorforge::broadcast<2, 1, 1>(v30_lin)) * v27_data);
          __float128 v35_lin = r1[1];
          v29_acc += ((tensorforge::broadcast<2, 1, 0>(v35_lin)) * v26_data);
          v29_acc += ((tensorforge::broadcast<2, 1, 1>(v35_lin)) * v27_data);
          r2[0] = v28_acc;
          r2[1] = v29_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v43_i0 = 0; v43_i0 < 1; ++v43_i0) {
            int32_t v51_lead = v10_lead + (v43_i0 * 2);
            #pragma unroll
            for (int32_t v44_i1 = 0; v44_i1 < 2; ++v44_i1) {
              __float128 v46_data = r2[(v43_i0 + v44_i1)];
              glb_m0[(v51_lead + (v44_i1 * 2))] = v46_data;
            }
          }
        }
      }
    }
  }
}

