// === base name ===
kernel_8ccfcf95e3abbed4

// === header ===
void launcher_kernel_8ccfcf95e3abbed4(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8ccfcf95e3abbed4(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8ccfcf95e3abbed4, block.x * block.y * block.z, 0 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_8ccfcf95e3abbed4, cudaFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_8ccfcf95e3abbed4<<<grid,block,0 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_8ccfcf95e3abbed4(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // TMP = +(A, dims=[1])
    // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8(8) {0..8} pointer_based({0..8})[0]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v0_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v0_batchId0 < numElements0; v0_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1_ahead1 = v0_batchId0 + (gridDim.x * blockDim.y);
        size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v0_batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[v0_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v0_batchId0 * 64 + 0 + m2_extraOffset];
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          int32_t v14_lead = threadIdx.x % 32;
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
              float v24_data = __ldcg(&glb_m2[(v14_lead + (v16_i1 * 8))]);
              r1[v16_i1] = v24_data;
            }
          }
          float r0[1]{};
          // r0 = +(glb_m0, dims=[1])
          if (v14_lead < 8) {
            float v32_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v31_r1 = 0; v31_r1 < 8; ++v31_r1) {
              float v40_data = glb_m0[(v14_lead + (v31_r1 * 8))];
              v32_acc0 = (v32_acc0 + v40_data);
            }
            r0[0] = v32_acc0;
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] []
          float ir2[8]{};
          if (v14_lead < 8) {
            float v49_data = r0[0];
            float v50_data = r1[0];
            float v52_data = ir2[0];
            ir2[0] = (v52_data + (v49_data * v50_data));
            float v55_data = r1[1];
            float v57_data = ir2[1];
            ir2[1] = (v57_data + (v49_data * v55_data));
            float v60_data = r1[2];
            float v62_data = ir2[2];
            ir2[2] = (v62_data + (v49_data * v60_data));
            float v65_data = r1[3];
            float v67_data = ir2[3];
            ir2[3] = (v67_data + (v49_data * v65_data));
            float v70_data = r1[4];
            float v72_data = ir2[4];
            ir2[4] = (v72_data + (v49_data * v70_data));
            float v75_data = r1[5];
            float v77_data = ir2[5];
            ir2[5] = (v77_data + (v49_data * v75_data));
            float v80_data = r1[6];
            float v82_data = ir2[6];
            ir2[6] = (v82_data + (v49_data * v80_data));
            float v85_data = r1[7];
            float v87_data = ir2[7];
            ir2[7] = (v87_data + (v49_data * v85_data));
          }
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v93_n1 = 0; v93_n1 < 8; ++v93_n1) {
              float v95_data = ir2[v93_n1];
              r2[v93_n1] = v95_data;
            }
          }
          // glb_m1 = store{r>g}(r2);
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v101_i1 = 0; v101_i1 < 8; ++v101_i1) {
              float v103_data = r2[v101_i1];
              glb_m1[(v14_lead + (v101_i1 * 8))] = v103_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

