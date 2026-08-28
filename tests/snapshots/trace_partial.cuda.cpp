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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v9_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v10_i0 = 0; v10_i0 < 1; ++v10_i0) {
            int32_t v16_lead = v9_lead + (v10_i0 * 16);
            #pragma unroll
            for (int32_t v11_i1 = 0; v11_i1 < 16; ++v11_i1) {
              float v19_data = __ldcg(&glb_m1[(v16_lead + (v11_i1 * 16))]);
              r0[(v10_i0 + v11_i1)] = v19_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 16)] [(0, 16)]
          float ir1[1]{};
          float v26_data = r0[0];
          float v27_data = ir1[0];
          ir1[0] = (v27_data + v26_data);
          float v32_data = r0[1];
          float v33_data = ir1[0];
          ir1[0] = (v33_data + v32_data);
          float v38_data = r0[2];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + v38_data);
          float v44_data = r0[3];
          float v45_data = ir1[0];
          ir1[0] = (v45_data + v44_data);
          float v50_data = r0[4];
          float v51_data = ir1[0];
          ir1[0] = (v51_data + v50_data);
          float v56_data = r0[5];
          float v57_data = ir1[0];
          ir1[0] = (v57_data + v56_data);
          float v62_data = r0[6];
          float v63_data = ir1[0];
          ir1[0] = (v63_data + v62_data);
          float v68_data = r0[7];
          float v69_data = ir1[0];
          ir1[0] = (v69_data + v68_data);
          float v74_data = r0[8];
          float v75_data = ir1[0];
          ir1[0] = (v75_data + v74_data);
          float v80_data = r0[9];
          float v81_data = ir1[0];
          ir1[0] = (v81_data + v80_data);
          float v86_data = r0[10];
          float v87_data = ir1[0];
          ir1[0] = (v87_data + v86_data);
          float v92_data = r0[11];
          float v93_data = ir1[0];
          ir1[0] = (v93_data + v92_data);
          float v98_data = r0[12];
          float v99_data = ir1[0];
          ir1[0] = (v99_data + v98_data);
          float v104_data = r0[13];
          float v105_data = ir1[0];
          ir1[0] = (v105_data + v104_data);
          float v110_data = r0[14];
          float v111_data = ir1[0];
          ir1[0] = (v111_data + v110_data);
          float v116_data = r0[15];
          float v117_data = ir1[0];
          ir1[0] = (v117_data + v116_data);
          #pragma unroll
          for (int32_t v122_n0 = 0; v122_n0 < 1; ++v122_n0) {
            float v123_data = ir1[v122_n0];
            r1[v122_n0] = v123_data;
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v127_i0 = 0; v127_i0 < 1; ++v127_i0) {
            float v128_data = r1[v127_i0];
            glb_m0[(v9_lead + (v127_i0 * 16))] = v128_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

