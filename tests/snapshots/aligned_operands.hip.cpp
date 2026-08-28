// === base name ===
kernel_21138a3fa2

// === header ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_21138a3fa2, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_21138a3fa2), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_21138a3fa2, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 16);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v12_i1 * 16))]);
              r0[(v11_i0 + v12_i1)] = v20_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          tensorforge::VectorT<float, 4> v23_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[0 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[0] = v23_lin;
          tensorforge::VectorT<float, 4> v24_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[64 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[4] = v24_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float v26_data = r1[0];
          float v27_data = r1[1];
          float v28_data = r1[2];
          float v29_data = r1[3];
          float v30_tp{};
          float v31_tp{};
          float v32_tp{};
          float v33_tp{};
          tensorforge::transpose4x4b32(v30_tp, v31_tp, v32_tp, v33_tp, v26_data, v27_data, v28_data, v29_data);
          tensorforge::VectorT<float, 4> v34_acc{};
          float v35_data = r0[0];
          float v36_data = r0[1];
          float v37_data = r0[2];
          float v38_data = r0[3];
          tensorforge::VectorT<float, 4> v39_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v35_data, v34_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v40_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v36_data, v39_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v41_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v37_data, v40_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v42_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v38_data, v41_acc, 2, 0, 0);
          float v43_data = r0[4];
          float v44_data = r0[5];
          float v45_data = r0[6];
          float v46_data = r0[7];
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v43_data, v42_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v44_data, v47_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v45_data, v48_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v46_data, v49_acc, 2, 1, 0);
          float v51_data = r0[8];
          float v52_data = r0[9];
          float v53_data = r0[10];
          float v54_data = r0[11];
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v51_data, v50_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v52_data, v55_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v53_data, v56_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v54_data, v57_acc, 2, 2, 0);
          float v59_data = r0[12];
          float v60_data = r0[13];
          float v61_data = r0[14];
          float v62_data = r0[15];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v30_tp, v59_data, v58_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v60_data, v63_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v61_data, v64_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v62_data, v65_acc, 2, 3, 0);
          r2[0] = (v66_acc[0]);
          r2[1] = (v66_acc[1]);
          r2[2] = (v66_acc[2]);
          r2[3] = (v66_acc[3]);
          float v71_data = r1[4];
          float v72_data = r1[5];
          float v73_data = r1[6];
          float v74_data = r1[7];
          float v75_tp{};
          float v76_tp{};
          float v77_tp{};
          float v78_tp{};
          tensorforge::transpose4x4b32(v75_tp, v76_tp, v77_tp, v78_tp, v71_data, v72_data, v73_data, v74_data);
          tensorforge::VectorT<float, 4> v79_acc{};
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v35_data, v79_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v36_data, v84_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v37_data, v85_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v38_data, v86_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v43_data, v87_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v44_data, v92_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v45_data, v93_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v46_data, v94_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v51_data, v95_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v52_data, v100_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v53_data, v101_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v54_data, v102_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v59_data, v103_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v76_tp, v60_data, v108_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v77_tp, v61_data, v109_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v78_tp, v62_data, v110_acc, 2, 3, 0);
          r2[4] = (v111_acc[0]);
          r2[5] = (v111_acc[1]);
          r2[6] = (v111_acc[2]);
          r2[7] = (v111_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v119_i0 = 0; v119_i0 < 1; ++v119_i0) {
            int32_t v127_lead = v10_lead + (v119_i0 * 16);
            #pragma unroll
            for (int32_t v120_i1 = 0; v120_i1 < 8; ++v120_i1) {
              float v122_data = r2[(v119_i0 + v120_i1)];
              glb_m0[(v127_lead + (v120_i1 * 16))] = v122_data;
            }
          }
        }
      }
    }
  }
}

