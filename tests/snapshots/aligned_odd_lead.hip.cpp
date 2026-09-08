// === base name ===
kernel_69f2bb9311

// === header ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_69f2bb9311, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_69f2bb9311), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_69f2bb9311, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 35×4(35×4) {0..35}×{0..4} strided
    // m1 35×8(35×8) {0..35}×{0..8} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 32);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v12_i1 * 35))]);
              r0[(v11_i0 + (v12_i1 * 2))] = v20_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v29_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m1[(v29_lead + (v24_i1 * 35))]);
              r0[(1 + (v24_i1 * 2))] = v32_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m2);
          if (v10_lead < 8) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 4; ++v40_i1) {
              float v48_data = __builtin_nontemporal_load(&glb_m2[(v10_lead + (v40_i1 * 8))]);
              r1[v40_i1] = v48_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 35), (0, 4)] [(0, 8)]
          float v51_data = r1[0];
          float v52_data = r1[1];
          float v53_data = r1[2];
          float v54_data = r1[3];
          float v55_tp{};
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          tensorforge::transpose4x4b32(v55_tp, v56_tp, v57_tp, v58_tp, v51_data, v52_data, v53_data, v54_data);
          tensorforge::VectorT<float, 4> v59_acc{};
          float v60_data = r0[0];
          float v61_data = r0[2];
          float v62_data = r0[4];
          float v63_data = r0[6];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v60_data, v59_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v61_data, v64_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v62_data, v65_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v63_data, v66_acc, 3, 0, 0);
          float v68_data = r0[8];
          float v69_data = r0[10];
          float v70_data = r0[12];
          float v71_data = r0[14];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v67_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 3, 1, 0);
          r2[0] = (v75_acc[0]);
          r2[2] = (v75_acc[1]);
          r2[4] = (v75_acc[2]);
          r2[6] = (v75_acc[3]);
          tensorforge::VectorT<float, 4> v80_acc{};
          float v81_data = r0[1];
          float v82_data = r0[3];
          float v83_data = r0[5];
          float v84_data = r0[7];
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v81_data, v80_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v82_data, v85_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v83_data, v86_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v84_data, v87_acc, 3, 0, 0);
          float v89_data = r0[9];
          float v90_data = r0[11];
          float v91_data = r0[13];
          float v92_data = r0[15];
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v89_data, v88_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v90_data, v93_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v91_data, v94_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v92_data, v95_acc, 3, 1, 0);
          r2[1] = (v96_acc[0]);
          r2[3] = (v96_acc[1]);
          r2[5] = (v96_acc[2]);
          r2[7] = (v96_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v104_i0 = 0; v104_i0 < 1; ++v104_i0) {
            int32_t v113_lead = v10_lead + (v104_i0 * 32);
            #pragma unroll
            for (int32_t v105_i1 = 0; v105_i1 < 4; ++v105_i1) {
              float v108_data = r2[(v104_i0 + (v105_i1 * 2))];
              glb_m0[(v113_lead + (v105_i1 * 35))] = v108_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v125_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v117_i1 = 0; v117_i1 < 4; ++v117_i1) {
              float v120_data = r2[(1 + (v117_i1 * 2))];
              glb_m0[(v125_lead + (v117_i1 * 35))] = v120_data;
            }
          }
        }
      }
    }
  }
}

