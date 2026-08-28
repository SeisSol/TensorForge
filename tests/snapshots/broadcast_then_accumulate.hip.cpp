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
            float v17_data = __builtin_nontemporal_load(&glb_m0[(v10_lead + (v11_i0 * 32))]);
            r0[v11_i0] = v17_data;
          }
          float r2[3]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v28_lead = v10_lead + (v22_i0 * 32);
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 3; ++v23_i1) {
              float v31_data = __builtin_nontemporal_load(&glb_m1[(v28_lead + (v23_i1 * 32))]);
              r2[(v22_i0 + v23_i1)] = v31_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[1]{};
          // r1 = +(r0) + None
          // [(0, 32)] []
          float v37_data = r0[0];
          float v38_data = r1[0];
          r1[0] = (v38_data + v37_data);
          // wait(r2 = load{g>r}(glb_m1););
          float r3[3]{};
          // r3 = +(r2) + None
          // [(0, 32), (0, 3)] []
          float v44_data = r2[0];
          float v45_data = r3[0];
          r3[0] = (v45_data + v44_data);
          float v47_data = r2[1];
          float v48_data = r3[1];
          r3[1] = (v48_data + v47_data);
          float v50_data = r2[2];
          float v51_data = r3[2];
          r3[2] = (v51_data + v50_data);
          float r4[3]{};
          // r4 = +(r1) + None
          // [(0, 32), (0, 3)] []
          float v57_data = r1[0];
          float v58_data = r4[0];
          r4[0] = (v58_data + v57_data);
          float v61_data = r4[1];
          r4[1] = (v61_data + v57_data);
          float v64_data = r4[2];
          r4[2] = (v64_data + v57_data);
          float r5[3]{};
          // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 3)] []
          float ir5[3]{};
          float v71_data = r3[0];
          float v72_data = ir5[0];
          ir5[0] = (v72_data + v71_data);
          float v74_data = r3[1];
          float v75_data = ir5[1];
          ir5[1] = (v75_data + v74_data);
          float v77_data = r3[2];
          float v78_data = ir5[2];
          ir5[2] = (v78_data + v77_data);
          #pragma unroll
          for (int32_t v83_n0 = 0; v83_n0 < 1; ++v83_n0) {
            #pragma unroll
            for (int32_t v84_n1 = 0; v84_n1 < 3; ++v84_n1) {
              int32_t v85_a = v83_n0 + v84_n1;
              float v86_data = ir5[v85_a];
              float v88_data = r4[v85_a];
              r5[v85_a] = (v88_data + v86_data);
            }
          }
          float r6[3]{};
          // r6 = +(r5) + None
          // [(0, 32), (0, 3)] []
          float v95_data = r5[0];
          float v96_data = r6[0];
          r6[0] = (v96_data + v95_data);
          float v98_data = r5[1];
          float v99_data = r6[1];
          r6[1] = (v99_data + v98_data);
          float v101_data = r5[2];
          float v102_data = r6[2];
          r6[2] = (v102_data + v101_data);
          // glb_m2 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v107_i0 = 0; v107_i0 < 1; ++v107_i0) {
            int32_t v115_lead = v10_lead + (v107_i0 * 32);
            #pragma unroll
            for (int32_t v108_i1 = 0; v108_i1 < 3; ++v108_i1) {
              float v110_data = r6[(v107_i0 + v108_i1)];
              glb_m2[(v115_lead + (v108_i1 * 32))] = v110_data;
            }
          }
        }
      }
    }
  }
}

