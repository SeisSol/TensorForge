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
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_49337a255f, block.x * block.y * block.z, 0 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_49337a255f, cudaFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_49337a255f<<<grid,block,0 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
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
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
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
              float v20_data = __ldcg(&glb_m2[(v10_lead + (v12_i1 * 8))]);
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
          float ir2[8]{};
          if (v10_lead < 8) {
            float v45_data = r0[0];
            float v46_data = r1[0];
            float v48_data = ir2[0];
            ir2[0] = (v48_data + (v45_data * v46_data));
            float v51_data = r1[1];
            float v53_data = ir2[1];
            ir2[1] = (v53_data + (v45_data * v51_data));
            float v56_data = r1[2];
            float v58_data = ir2[2];
            ir2[2] = (v58_data + (v45_data * v56_data));
            float v61_data = r1[3];
            float v63_data = ir2[3];
            ir2[3] = (v63_data + (v45_data * v61_data));
            float v66_data = r1[4];
            float v68_data = ir2[4];
            ir2[4] = (v68_data + (v45_data * v66_data));
            float v71_data = r1[5];
            float v73_data = ir2[5];
            ir2[5] = (v73_data + (v45_data * v71_data));
            float v76_data = r1[6];
            float v78_data = ir2[6];
            ir2[6] = (v78_data + (v45_data * v76_data));
            float v81_data = r1[7];
            float v83_data = ir2[7];
            ir2[7] = (v83_data + (v45_data * v81_data));
          }
          if (v10_lead < 8) {
            #pragma unroll
            for (int32_t v89_n1 = 0; v89_n1 < 8; ++v89_n1) {
              float v91_data = ir2[v89_n1];
              r2[v89_n1] = v91_data;
            }
          }
          // glb_m1 = store{r>g}(r2);
          if (v10_lead < 8) {
            #pragma unroll
            for (int32_t v97_i1 = 0; v97_i1 < 8; ++v97_i1) {
              float v99_data = r2[v97_i1];
              glb_m1[(v10_lead + (v97_i1 * 8))] = v99_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

