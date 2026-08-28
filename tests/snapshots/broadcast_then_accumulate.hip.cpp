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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[batchId0][0 + m1_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          float r0[1]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v15_lead = v11_i0 * 32;
            int32_t v16_lead = v10_lead + v15_lead;
            float v22_data = __builtin_nontemporal_load(&glb_m0[(v10_lead + v15_lead)]);
            r0[v11_i0] = v22_data;
          }
          float r2[3]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
            int32_t v32_lead = v27_i0 * 32;
            int32_t v33_lead = v10_lead + v32_lead;
            int32_t v40_lead = v10_lead + v32_lead;
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 3; ++v28_i1) {
              int32_t v34_a = v28_i1 * 32;
              int32_t v35_a = v33_lead + v34_a;
              float v43_data = __builtin_nontemporal_load(&glb_m1[(v40_lead + v34_a)]);
              r2[(v27_i0 + v28_i1)] = v43_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 32)] []
          float v49_data = r0[0];
          float v50_data = r1[0];
          r1[0] = (v50_data + v49_data);
          // wait(r2 = load{g>r}(glb_m1););
          float r3[3]{};
          // r3 = +(r2) + None
          // [(0, 32), (0, 3)] []
          float v56_data = r2[0];
          float v57_data = r3[0];
          r3[0] = (v57_data + v56_data);
          float v59_data = r2[1];
          float v60_data = r3[1];
          r3[1] = (v60_data + v59_data);
          float v62_data = r2[2];
          float v63_data = r3[2];
          r3[2] = (v63_data + v62_data);
          float r4[3]{};
          // r4 = +(r1) + None
          // [(0, 32), (0, 3)] []
          float v69_data = r1[0];
          float v70_data = r4[0];
          r4[0] = (v70_data + v69_data);
          float v73_data = r4[1];
          r4[1] = (v73_data + v69_data);
          float v76_data = r4[2];
          r4[2] = (v76_data + v69_data);
          float r5[3]{};
          // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 3)] []
          float ir5[3]{};
          float v83_data = r3[0];
          float v84_data = ir5[0];
          ir5[0] = (v84_data + v83_data);
          float v86_data = r3[1];
          float v87_data = ir5[1];
          ir5[1] = (v87_data + v86_data);
          float v89_data = r3[2];
          float v90_data = ir5[2];
          ir5[2] = (v90_data + v89_data);
          #pragma unroll
          for (int32_t v95_n0 = 0; v95_n0 < 1; ++v95_n0) {
            #pragma unroll
            for (int32_t v96_n1 = 0; v96_n1 < 3; ++v96_n1) {
              int32_t v97_a = v95_n0 + v96_n1;
              int32_t v98_a = v95_n0 + v96_n1;
              float v99_data = ir5[v98_a];
              int32_t v100_a = v95_n0 + v96_n1;
              float v102_data = r4[v98_a];
              r5[v98_a] = (v102_data + v99_data);
            }
          }
          float r6[3]{};
          // r6 = +(r5) + None
          // [(0, 32), (0, 3)] []
          float v109_data = r5[0];
          float v110_data = r6[0];
          r6[0] = (v110_data + v109_data);
          float v112_data = r5[1];
          float v113_data = r6[1];
          r6[1] = (v113_data + v112_data);
          float v115_data = r5[2];
          float v116_data = r6[2];
          r6[2] = (v116_data + v115_data);
          // glb_m2 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v121_i0 = 0; v121_i0 < 1; ++v121_i0) {
            int32_t v130_lead = v10_lead + (v121_i0 * 32);
            #pragma unroll
            for (int32_t v122_i1 = 0; v122_i1 < 3; ++v122_i1) {
              int32_t v123_a = v121_i0 + v122_i1;
              float v125_data = r6[(v121_i0 + v122_i1)];
              glb_m2[(v130_lead + (v122_i1 * 32))] = v125_data;
            }
          }
        }
      }
    }
  }
}

