// === base name ===
kernel_75d3097b00

// === header ===
void launcher_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_75d3097b00, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_75d3097b00), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_75d3097b00, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 20×9(20×9) {0..20}×{0..9} strided
    // m1 1×20(1×20) {0..1}×{0..20} strided
    // m2 1×9(1×9) {0..1}×{0..9} strided
    // m0 20×9(20×9) {0..20}×{0..9} strided({0..20}×{0..9})[0, 1] = m1 1×20(1×20) {0..1}×{0..20} strided({0..1}×{0..20})[-1, 0]×m2 1×9(1×9) {0..1}×{0..9} strided({0..1}×{0..9})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 180 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 20 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 9 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 32;
          bool v4_g = v3_lead < 20;
          #pragma unroll
          for (int32_t v0_i0 = 0; v0_i0 < 1; ++v0_i0) {
            if (v4_g) {
              int32_t v10_a = v0_i0 + v3_lead;
              float v11_data;
              {
                v11_data = __builtin_nontemporal_load(&glb_m1[v10_a]);
              }
              int32_t v12_a = v0_i0 + 0;
              r0[v12_a] = v11_data;
            }
          }
          float r1[9]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 20), (0, 9)] [(0, 1)]
          auto& ir2 = r2;
          float v13_data = r0[0];
          float v14_acc{};
          float v15_acc{};
          float v16_acc{};
          float v17_acc{};
          float v18_acc{};
          float v19_acc{};
          float v20_acc{};
          float v21_acc{};
          float v22_acc{};
          float v23_lin = r1[0];
          float v24_bc = tensorforge::broadcast<32, 16, 0>(v23_lin);
          tensorforge::fmacdpp16<0>(v14_acc, v24_bc, v13_data);
          tensorforge::fmacdpp16<1>(v15_acc, v24_bc, v13_data);
          tensorforge::fmacdpp16<2>(v16_acc, v24_bc, v13_data);
          tensorforge::fmacdpp16<3>(v17_acc, v24_bc, v13_data);
          tensorforge::fmacdpp16<4>(v18_acc, v24_bc, v13_data);
          tensorforge::fmacdpp16<5>(v19_acc, v24_bc, v13_data);
          tensorforge::fmacdpp16<6>(v20_acc, v24_bc, v13_data);
          tensorforge::fmacdpp16<7>(v21_acc, v24_bc, v13_data);
          tensorforge::fmacdpp16<8>(v22_acc, v24_bc, v13_data);
          ir2[0] = v14_acc;
          ir2[1] = v15_acc;
          ir2[2] = v16_acc;
          ir2[3] = v17_acc;
          ir2[4] = v18_acc;
          ir2[5] = v19_acc;
          ir2[6] = v20_acc;
          ir2[7] = v21_acc;
          ir2[8] = v22_acc;
          // glb_m0 = store{r>g}(r2);
          int32_t v27_lead = threadIdx.x % 32;
          if (v27_lead < 20) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
              int32_t v30_a = 0 + v29_i1;
              float v32_data = r2[v29_i1];
              int32_t v39_a = v27_lead + (v29_i1 * 20);
              glb_m0[v39_a] = v32_data;
            }
          }
          ;
        }
      }
    }
  }
}

