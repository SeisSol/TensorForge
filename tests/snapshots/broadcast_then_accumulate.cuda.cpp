// === base name ===
kernel_7cc2a3c5b0

// === header ===
void launcher_kernel_7cc2a3c5b0(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7cc2a3c5b0(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7cc2a3c5b0, block.x * block.y * block.z, 0 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_7cc2a3c5b0, cudaFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_7cc2a3c5b0<<<grid,block,0 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7cc2a3c5b0(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
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
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v6_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v11_lead = v7_i0 * 32;
            int32_t v12_lead = v6_lead + v11_lead;
            float v18_data = __ldcg(&glb_m0[(v6_lead + v11_lead)]);
            r0[v7_i0] = v18_data;
          }
          float r2[3]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v28_lead = v23_i0 * 32;
            int32_t v29_lead = v6_lead + v28_lead;
            int32_t v36_lead = v6_lead + v28_lead;
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 3; ++v24_i1) {
              int32_t v30_a = v24_i1 * 32;
              int32_t v31_a = v29_lead + v30_a;
              float v39_data = __ldcg(&glb_m1[(v36_lead + v30_a)]);
              int32_t v40_a = v23_i0 + v24_i1;
              r2[v40_a] = v39_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 32)] []
          float v45_data = r0[0];
          float v46_data = r1[0];
          r1[0] = (v46_data + v45_data);
          // wait(r2 = load{g>r}(glb_m1););
          float r3[3]{};
          // r3 = +(r2) + None
          // [(0, 32), (0, 3)] []
          float v52_data = r2[0];
          float v53_data = r3[0];
          r3[0] = (v53_data + v52_data);
          float v55_data = r2[1];
          float v56_data = r3[1];
          r3[1] = (v56_data + v55_data);
          float v58_data = r2[2];
          float v59_data = r3[2];
          r3[2] = (v59_data + v58_data);
          float r4[3]{};
          // r4 = +(r1) + None
          // [(0, 32), (0, 3)] []
          float v65_data = r1[0];
          float v66_data = r4[0];
          r4[0] = (v66_data + v65_data);
          float v69_data = r4[1];
          r4[1] = (v69_data + v65_data);
          float v72_data = r4[2];
          r4[2] = (v72_data + v65_data);
          float r5[3]{};
          // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 3)] []
          float ir5[3]{};
          float v79_data = r3[0];
          float v80_data = ir5[0];
          ir5[0] = (v80_data + v79_data);
          float v82_data = r3[1];
          float v83_data = ir5[1];
          ir5[1] = (v83_data + v82_data);
          float v85_data = r3[2];
          float v86_data = ir5[2];
          ir5[2] = (v86_data + v85_data);
          #pragma unroll
          for (int32_t v91_n0 = 0; v91_n0 < 1; ++v91_n0) {
            #pragma unroll
            for (int32_t v92_n1 = 0; v92_n1 < 3; ++v92_n1) {
              int32_t v93_a = v91_n0 + v92_n1;
              int32_t v94_a = v91_n0 + v92_n1;
              float v95_data = ir5[v94_a];
              int32_t v96_a = v91_n0 + v92_n1;
              float v98_data = r4[v94_a];
              int32_t v100_a = v91_n0 + v92_n1;
              r5[v94_a] = (v98_data + v95_data);
            }
          }
          float r6[3]{};
          // r6 = +(r5) + None
          // [(0, 32), (0, 3)] []
          float ir6[3]{};
          float v107_data = r5[0];
          float v108_data = ir6[0];
          ir6[0] = (v108_data + v107_data);
          float v110_data = r5[1];
          float v111_data = ir6[1];
          ir6[1] = (v111_data + v110_data);
          float v113_data = r5[2];
          float v114_data = ir6[2];
          ir6[2] = (v114_data + v113_data);
          #pragma unroll
          for (int32_t v119_n0 = 0; v119_n0 < 1; ++v119_n0) {
            #pragma unroll
            for (int32_t v120_n1 = 0; v120_n1 < 3; ++v120_n1) {
              int32_t v121_a = v119_n0 + v120_n1;
              int32_t v122_a = v119_n0 + v120_n1;
              float v123_data = ir6[v122_a];
              int32_t v124_a = v119_n0 + v120_n1;
              r6[v122_a] = v123_data;
            }
          }
          // glb_m2 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v129_i0 = 0; v129_i0 < 1; ++v129_i0) {
            int32_t v138_lead = v6_lead + (v129_i0 * 32);
            #pragma unroll
            for (int32_t v130_i1 = 0; v130_i1 < 3; ++v130_i1) {
              int32_t v131_a = v129_i0 + v130_i1;
              float v133_data = r6[(v129_i0 + v130_i1)];
              int32_t v140_a = v138_lead + (v130_i1 * 32);
              glb_m2[v140_a] = v133_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

