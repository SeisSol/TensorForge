// === base name ===
kernel_a7d5d30824

// === header ===
void launcher_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a7d5d30824, block.x * block.y * block.z, 256 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_a7d5d30824, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_a7d5d30824<<<grid,block,256 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16(16) {0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m0 16(16) {0..16} strided({0..16})[0] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 16;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v11_a = v5_i1 * 16;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __ldcg(&glb_m1[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] [(0, 16)]
          float ir1[1]{};
          float v27_data = r0[0];
          float v28_data = ir1[0];
          ir1[0] = (v28_data + v27_data);
          float v33_data = r0[1];
          float v34_data = ir1[0];
          ir1[0] = (v34_data + v33_data);
          float v39_data = r0[2];
          float v40_data = ir1[0];
          ir1[0] = (v40_data + v39_data);
          float v45_data = r0[3];
          float v46_data = ir1[0];
          ir1[0] = (v46_data + v45_data);
          float v51_data = r0[4];
          float v52_data = ir1[0];
          ir1[0] = (v52_data + v51_data);
          float v57_data = r0[5];
          float v58_data = ir1[0];
          ir1[0] = (v58_data + v57_data);
          float v63_data = r0[6];
          float v64_data = ir1[0];
          ir1[0] = (v64_data + v63_data);
          float v69_data = r0[7];
          float v70_data = ir1[0];
          ir1[0] = (v70_data + v69_data);
          float v75_data = r0[8];
          float v76_data = ir1[0];
          ir1[0] = (v76_data + v75_data);
          float v81_data = r0[9];
          float v82_data = ir1[0];
          ir1[0] = (v82_data + v81_data);
          float v87_data = r0[10];
          float v88_data = ir1[0];
          ir1[0] = (v88_data + v87_data);
          float v93_data = r0[11];
          float v94_data = ir1[0];
          ir1[0] = (v94_data + v93_data);
          float v99_data = r0[12];
          float v100_data = ir1[0];
          ir1[0] = (v100_data + v99_data);
          float v105_data = r0[13];
          float v106_data = ir1[0];
          ir1[0] = (v106_data + v105_data);
          float v111_data = r0[14];
          float v112_data = ir1[0];
          ir1[0] = (v112_data + v111_data);
          float v117_data = r0[15];
          float v118_data = ir1[0];
          ir1[0] = (v118_data + v117_data);
          #pragma unroll
          for (int32_t v123_n0 = 0; v123_n0 < 1; ++v123_n0) {
            float v124_data = ir1[v123_n0];
            r1[v123_n0] = v124_data;
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v128_i0 = 0; v128_i0 < 1; ++v128_i0) {
            float v129_data = r1[v128_i0];
            int32_t v134_lead = v3_lead + (v128_i0 * 16);
            glb_m0[v134_lead] = v129_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

