// === base name ===
kernel_87f2838a59

// === header ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_87f2838a59, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_87f2838a59), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_87f2838a59, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v10_off = (v2_lead + (v3_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v4_i1 = 8; v4_i1 < 24; ++v4_i1) {
              int32_t v12_a = v10_off + (v4_i1 * 32);
              float v13_data;
              {
                v13_data = __builtin_nontemporal_load(&glb_m1[v12_a]);
              }
              int32_t v15_a = v3_i0 + (v4_i1 - 8);
              r0[v15_a] = v13_data;
            }
          }
          float r1[8]{};
          {
            // r1 = load{g>r}(glb_m2);
            float v0 = glb_m2[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v16 = glb_m2[16 + threadIdx.x * 1];
            r1[1] = v16;
            float v32 = glb_m2[32 + threadIdx.x * 1];
            r1[2] = v32;
            float v48 = glb_m2[48 + threadIdx.x * 1];
            r1[3] = v48;
            float v64 = glb_m2[64 + threadIdx.x * 1];
            r1[4] = v64;
            float v80 = glb_m2[80 + threadIdx.x * 1];
            r1[5] = v80;
            float v96 = glb_m2[96 + threadIdx.x * 1];
            r1[6] = v96;
            float v112 = glb_m2[112 + threadIdx.x * 1];
            r1[7] = v112;
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          auto& ir2 = r2;
          float v16_data = r1[0];
          float v17_data = r1[1];
          float v18_data = r1[2];
          float v19_data = r1[3];
          float v20_tp{};
          float v21_tp{};
          float v22_tp{};
          float v23_tp{};
          tensorforge::transpose4x4b32(v20_tp, v21_tp, v22_tp, v23_tp, v16_data, v17_data, v18_data, v19_data);
          tensorforge::VectorT<float, 4> v24_acc{};
          float v25_data = r0[0];
          float v26_data = r0[1];
          float v27_data = r0[2];
          float v28_data = r0[3];
          tensorforge::VectorT<float, 4> v29_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v25_data, v24_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v30_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v26_data, v29_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v31_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v22_tp, v27_data, v30_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v32_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v28_data, v31_acc, 2, 0, 0);
          float v33_data = r0[4];
          float v34_data = r0[5];
          float v35_data = r0[6];
          float v36_data = r0[7];
          tensorforge::VectorT<float, 4> v37_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v33_data, v32_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v38_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v34_data, v37_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v39_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v22_tp, v35_data, v38_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v40_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v36_data, v39_acc, 2, 1, 0);
          float v41_data = r0[8];
          float v42_data = r0[9];
          float v43_data = r0[10];
          float v44_data = r0[11];
          tensorforge::VectorT<float, 4> v45_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v41_data, v40_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v42_data, v45_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v22_tp, v43_data, v46_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v44_data, v47_acc, 2, 2, 0);
          float v49_data = r0[12];
          float v50_data = r0[13];
          float v51_data = r0[14];
          float v52_data = r0[15];
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v20_tp, v49_data, v48_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v21_tp, v50_data, v53_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v22_tp, v51_data, v54_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v23_tp, v52_data, v55_acc, 2, 3, 0);
          ir2[0] = (v56_acc[0]);
          ir2[1] = (v56_acc[1]);
          ir2[2] = (v56_acc[2]);
          ir2[3] = (v56_acc[3]);
          float v61_data = r1[4];
          float v62_data = r1[5];
          float v63_data = r1[6];
          float v64_data = r1[7];
          float v65_tp{};
          float v66_tp{};
          float v67_tp{};
          float v68_tp{};
          tensorforge::transpose4x4b32(v65_tp, v66_tp, v67_tp, v68_tp, v61_data, v62_data, v63_data, v64_data);
          tensorforge::VectorT<float, 4> v69_acc{};
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v25_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v26_data, v74_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v27_data, v75_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v28_data, v76_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v33_data, v77_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v34_data, v82_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v35_data, v83_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v36_data, v84_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v41_data, v85_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v42_data, v90_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v43_data, v91_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v44_data, v92_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v49_data, v93_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v50_data, v98_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v51_data, v99_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v52_data, v100_acc, 2, 3, 0);
          ir2[4] = (v101_acc[0]);
          ir2[5] = (v101_acc[1]);
          ir2[6] = (v101_acc[2]);
          ir2[7] = (v101_acc[3]);
          // glb_m0 = store{r>g}(r2);
          int32_t v108_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v109_i0 = 0; v109_i0 < 1; ++v109_i0) {
            int32_t v117_lead = v108_lead + (v109_i0 * 16);
            #pragma unroll
            for (int32_t v110_i1 = 0; v110_i1 < 8; ++v110_i1) {
              int32_t v111_a = v109_i0 + v110_i1;
              float v112_data = r2[v111_a];
              int32_t v119_a = v117_lead + (v110_i1 * 16);
              glb_m0[v119_a] = v112_data;
            }
          }
          ;
        }
      }
    }
  }
}

