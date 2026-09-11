// === base name ===
kernel_f5f3404f979dc447

// === header ===
void launcher_kernel_f5f3404f979dc447(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f5f3404f979dc447(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f5f3404f979dc447, block.x * block.y * block.z, 0 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_f5f3404f979dc447, cudaFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f5f3404f979dc447<<<grid,block,0 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_f5f3404f979dc447(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32(32) {0..32} pointer_based
    // m1 32×3(32×3) {0..32}×{0..3} pointer_based
    // m2 32×3(32×3) {0..32}×{0..3} pointer_based
    // t0 32(32) {0..32} strided({0..32})[0] = m0 32(32) {0..32} pointer_based({0..32})[0]
    // t1 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] = m1 32×3(32×3) {0..32}×{0..3} pointer_based({0..32}×{0..3})[0, 1]
    // t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] = t0 32(32) {0..32} strided({0..32})[0]
    // t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] += t1 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1]
    // m2 32×3(32×3) {0..32}×{0..3} pointer_based({0..32}×{0..3})[0, 1] = t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v0_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v0_batchId0 < numElements0; v0_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1_ahead1 = v0_batchId0 + (gridDim.x * blockDim.y);
        size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v0_batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v0_batchId0][0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v0_batchId0][0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v14_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
            float v21_data = __ldcg(&glb_m0[(v14_lead + (v15_i0 * 32))]);
            r0[v15_i0] = v21_data;
          }
          float r2[3]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v32_lead = v14_lead + (v26_i0 * 32);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 3; ++v27_i1) {
              float v35_data = __ldcg(&glb_m1[(v32_lead + (v27_i1 * 32))]);
              r2[(v26_i0 + v27_i1)] = v35_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 32)] []
          float v41_data = r0[0];
          float v42_data = r1[0];
          r1[0] = (v42_data + v41_data);
          // wait(r2 = load{g>r}(glb_m1););
          float r3[3]{};
          // r3 = +(r2) + None
          // [(0, 32), (0, 3)] []
          float v48_data = r2[0];
          float v49_data = r3[0];
          r3[0] = (v49_data + v48_data);
          float v51_data = r2[1];
          float v52_data = r3[1];
          r3[1] = (v52_data + v51_data);
          float v54_data = r2[2];
          float v55_data = r3[2];
          r3[2] = (v55_data + v54_data);
          float r4[3]{};
          // r4 = +(r1) + None
          // [(0, 32), (0, 3)] []
          float v61_data = r1[0];
          float v62_data = r4[0];
          r4[0] = (v62_data + v61_data);
          float v65_data = r4[1];
          r4[1] = (v65_data + v61_data);
          float v68_data = r4[2];
          r4[2] = (v68_data + v61_data);
          float r5[3]{};
          // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 3)] []
          float ir5[3]{};
          float v75_data = r3[0];
          float v76_data = ir5[0];
          ir5[0] = (v76_data + v75_data);
          float v78_data = r3[1];
          float v79_data = ir5[1];
          ir5[1] = (v79_data + v78_data);
          float v81_data = r3[2];
          float v82_data = ir5[2];
          ir5[2] = (v82_data + v81_data);
          #pragma unroll
          for (int32_t v87_n0 = 0; v87_n0 < 1; ++v87_n0) {
            #pragma unroll
            for (int32_t v88_n1 = 0; v88_n1 < 3; ++v88_n1) {
              int32_t v89_a = v87_n0 + v88_n1;
              float v90_data = ir5[v89_a];
              float v92_data = r4[v89_a];
              r5[v89_a] = (v92_data + v90_data);
            }
          }
          float r6[3]{};
          // r6 = +(r5) + None
          // [(0, 32), (0, 3)] []
          float ir6[3]{};
          float v100_data = r5[0];
          float v101_data = ir6[0];
          ir6[0] = (v101_data + v100_data);
          float v103_data = r5[1];
          float v104_data = ir6[1];
          ir6[1] = (v104_data + v103_data);
          float v106_data = r5[2];
          float v107_data = ir6[2];
          ir6[2] = (v107_data + v106_data);
          #pragma unroll
          for (int32_t v112_n0 = 0; v112_n0 < 1; ++v112_n0) {
            #pragma unroll
            for (int32_t v113_n1 = 0; v113_n1 < 3; ++v113_n1) {
              int32_t v114_a = v112_n0 + v113_n1;
              float v115_data = ir6[v114_a];
              r6[v114_a] = v115_data;
            }
          }
          // glb_m2 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v120_i0 = 0; v120_i0 < 1; ++v120_i0) {
            int32_t v128_lead = v14_lead + (v120_i0 * 32);
            #pragma unroll
            for (int32_t v121_i1 = 0; v121_i1 < 3; ++v121_i1) {
              float v123_data = r6[(v120_i0 + v121_i1)];
              glb_m2[(v128_lead + (v121_i1 * 32))] = v123_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

