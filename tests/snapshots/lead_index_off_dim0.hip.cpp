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
          int32_t v24_lead = threadIdx.x % 32;
          if (v24_lead < 1) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 9; ++v26_i1) {
              float v33_data = __builtin_nontemporal_load(&glb_m2[(v24_lead + v26_i1)]);
              r1[v26_i1] = v33_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 20), (0, 9)] [(0, 1)]
          float v36_data = r1[0];
          float v37_data = r1[1];
          float v38_data = r1[2];
          float v39_data = r1[3];
          float v40_tp{};
          float v41_tp{};
          float v42_tp{};
          float v43_tp{};
          tensorforge::transpose4x4b32(v40_tp, v41_tp, v42_tp, v43_tp, v36_data, v37_data, v38_data, v39_data);
          tensorforge::VectorT<float, 4> v44_acc{};
          float v45_data = r0[0];
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v45_data, v44_acc, 3, 0, 0);
          r2[0] = (v49_acc[0]);
          r2[1] = (v49_acc[1]);
          r2[2] = (v49_acc[2]);
          r2[3] = (v49_acc[3]);
          float v54_data = r1[4];
          float v55_data = r1[5];
          float v56_data = r1[6];
          float v57_data = r1[7];
          float v58_tp{};
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          tensorforge::transpose4x4b32(v58_tp, v59_tp, v60_tp, v61_tp, v54_data, v55_data, v56_data, v57_data);
          tensorforge::VectorT<float, 4> v62_acc{};
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v45_data, v62_acc, 3, 0, 0);
          r2[4] = (v67_acc[0]);
          r2[5] = (v67_acc[1]);
          r2[6] = (v67_acc[2]);
          r2[7] = (v67_acc[3]);
          float v73_acc{};
          float v74_data = r1[8];
          tensorforge::fmacdpp16<0>(v73_acc, (tensorforge::broadcast<32, 16, 0>(v74_data)), v45_data);
          r2[8] = v73_acc;
          // glb_m0 = store{r>g}(r2);
          if (v24_lead < 20) {
            #pragma unroll
            for (int32_t v80_i1 = 0; v80_i1 < 9; ++v80_i1) {
              float v82_data = r2[v80_i1];
              glb_m0[(v24_lead + (v80_i1 * 20))] = v82_data;
            }
          }
        }
      }
    }
  }
}

