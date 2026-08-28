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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 180 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 20 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 9 + 0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v11_lead = threadIdx.x % 32;
          bool v12_g = v11_lead < 20;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            if (v12_g) {
              float v19_data = __builtin_nontemporal_load(&glb_m1[(v8_i0 + v11_lead)]);
              r0[v8_i0] = v19_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m2);
          float v22_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v22_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 20), (0, 9)] [(0, 1)]
          float v24_data = r0[0];
          float v25_acc{};
          float v26_acc{};
          float v27_acc{};
          float v28_acc{};
          float v29_acc{};
          float v30_acc{};
          float v31_acc{};
          float v32_acc{};
          float v33_acc{};
          float v34_lin = r1[0];
          float v35_bc = tensorforge::broadcast<32, 16, 0>(v34_lin);
          tensorforge::fmacdpp16<0>(v25_acc, v35_bc, v24_data);
          tensorforge::fmacdpp16<1>(v26_acc, v35_bc, v24_data);
          tensorforge::fmacdpp16<2>(v27_acc, v35_bc, v24_data);
          tensorforge::fmacdpp16<3>(v28_acc, v35_bc, v24_data);
          tensorforge::fmacdpp16<4>(v29_acc, v35_bc, v24_data);
          tensorforge::fmacdpp16<5>(v30_acc, v35_bc, v24_data);
          tensorforge::fmacdpp16<6>(v31_acc, v35_bc, v24_data);
          tensorforge::fmacdpp16<7>(v32_acc, v35_bc, v24_data);
          tensorforge::fmacdpp16<8>(v33_acc, v35_bc, v24_data);
          r2[0] = v25_acc;
          r2[1] = v26_acc;
          r2[2] = v27_acc;
          r2[3] = v28_acc;
          r2[4] = v29_acc;
          r2[5] = v30_acc;
          r2[6] = v31_acc;
          r2[7] = v32_acc;
          r2[8] = v33_acc;
          // glb_m0 = store{r>g}(r2);
          int32_t v38_lead = threadIdx.x % 32;
          if (v38_lead < 20) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 9; ++v40_i1) {
              float v42_data = r2[v40_i1];
              glb_m0[(v38_lead + (v40_i1 * 20))] = v42_data;
            }
          }
        }
      }
    }
  }
}

