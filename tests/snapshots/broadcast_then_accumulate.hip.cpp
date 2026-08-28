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
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7cc2a3c5b0, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_7cc2a3c5b0), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_7cc2a3c5b0, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
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
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[batchId0][0 + m1_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v7_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v12_lead = v8_i0 * 32;
            int32_t v13_lead = v7_lead + v12_lead;
            float v19_data = __builtin_nontemporal_load(&glb_m0[(v7_lead + v12_lead)]);
            r0[v8_i0] = v19_data;
          }
          float r2[3]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v29_lead = v24_i0 * 32;
            int32_t v30_lead = v7_lead + v29_lead;
            int32_t v37_lead = v7_lead + v29_lead;
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 3; ++v25_i1) {
              int32_t v31_a = v25_i1 * 32;
              int32_t v32_a = v30_lead + v31_a;
              float v40_data = __builtin_nontemporal_load(&glb_m1[(v37_lead + v31_a)]);
              r2[(v24_i0 + v25_i1)] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 32)] []
          float v46_data = r0[0];
          float v47_data = r1[0];
          r1[0] = (v47_data + v46_data);
          // wait(r2 = load{g>r}(glb_m1););
          float r3[3]{};
          // r3 = +(r2) + None
          // [(0, 32), (0, 3)] []
          float v53_data = r2[0];
          float v54_data = r3[0];
          r3[0] = (v54_data + v53_data);
          float v56_data = r2[1];
          float v57_data = r3[1];
          r3[1] = (v57_data + v56_data);
          float v59_data = r2[2];
          float v60_data = r3[2];
          r3[2] = (v60_data + v59_data);
          float r4[3]{};
          // r4 = +(r1) + None
          // [(0, 32), (0, 3)] []
          float v66_data = r1[0];
          float v67_data = r4[0];
          r4[0] = (v67_data + v66_data);
          float v70_data = r4[1];
          r4[1] = (v70_data + v66_data);
          float v73_data = r4[2];
          r4[2] = (v73_data + v66_data);
          float r5[3]{};
          // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 3)] []
          float ir5[3]{};
          float v80_data = r3[0];
          float v81_data = ir5[0];
          ir5[0] = (v81_data + v80_data);
          float v83_data = r3[1];
          float v84_data = ir5[1];
          ir5[1] = (v84_data + v83_data);
          float v86_data = r3[2];
          float v87_data = ir5[2];
          ir5[2] = (v87_data + v86_data);
          #pragma unroll
          for (int32_t v92_n0 = 0; v92_n0 < 1; ++v92_n0) {
            #pragma unroll
            for (int32_t v93_n1 = 0; v93_n1 < 3; ++v93_n1) {
              int32_t v94_a = v92_n0 + v93_n1;
              int32_t v95_a = v92_n0 + v93_n1;
              float v96_data = ir5[v95_a];
              int32_t v97_a = v92_n0 + v93_n1;
              float v99_data = r4[v95_a];
              r5[v95_a] = (v99_data + v96_data);
            }
          }
          float r6[3]{};
          // r6 = +(r5) + None
          // [(0, 32), (0, 3)] []
          float v106_data = r5[0];
          float v107_data = r6[0];
          r6[0] = (v107_data + v106_data);
          float v109_data = r5[1];
          float v110_data = r6[1];
          r6[1] = (v110_data + v109_data);
          float v112_data = r5[2];
          float v113_data = r6[2];
          r6[2] = (v113_data + v112_data);
          // glb_m2 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v118_i0 = 0; v118_i0 < 1; ++v118_i0) {
            int32_t v127_lead = v7_lead + (v118_i0 * 32);
            #pragma unroll
            for (int32_t v119_i1 = 0; v119_i1 < 3; ++v119_i1) {
              int32_t v120_a = v118_i0 + v119_i1;
              float v122_data = r6[(v118_i0 + v119_i1)];
              glb_m2[(v127_lead + (v119_i1 * 32))] = v122_data;
            }
          }
        }
      }
    }
  }
}

