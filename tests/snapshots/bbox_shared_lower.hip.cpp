// === base name ===
kernel_4b59b6f027

// === header ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4b59b6f027, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4b59b6f027), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_4b59b6f027, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(12×8) {4..16}×{0..8} strided
    // m1 16×16(12×16) {4..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(12×8) {4..16}×{0..8} strided({4..16}×{0..8})[0, 1] = m1 16×16(12×16) {4..16}×{0..16} strided({4..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 12) {
            int32_t v19_a = (v10_lead + 4) - 4;
            int32_t v28_a = (v10_lead + 4) - 4;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              int32_t v20_a = v12_i1 * 12;
              int32_t v21_a = v19_a + v20_a;
              float v31_data = __builtin_nontemporal_load(&glb_m1[(v28_a + v20_a)]);
              r0[v12_i1] = v31_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v34_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v34_lin;
          float v35_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v35_lin;
          float v36_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v36_lin;
          float v37_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v37_lin;
          float v38_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v38_lin;
          float v39_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v39_lin;
          float v40_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v40_lin;
          float v41_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v41_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(16, 28), (0, 8)] [(0, 16)]
          float v43_data = r1[0];
          float v44_data = r1[1];
          float v45_data = r1[2];
          float v46_data = r1[3];
          float v47_tp{};
          float v48_tp{};
          float v49_tp{};
          float v50_tp{};
          tensorforge::transpose4x4b32(v47_tp, v48_tp, v49_tp, v50_tp, v43_data, v44_data, v45_data, v46_data);
          tensorforge::VectorT<float, 4> v51_acc{};
          float v52_data = r0[0];
          float v53_data = r0[1];
          float v54_data = r0[2];
          float v55_data = r0[3];
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v51_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 2, 0, 0);
          float v60_data = r0[4];
          float v61_data = r0[5];
          float v62_data = r0[6];
          float v63_data = r0[7];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v59_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 2, 1, 0);
          float v68_data = r0[8];
          float v69_data = r0[9];
          float v70_data = r0[10];
          float v71_data = r0[11];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v67_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 2, 2, 0);
          float v76_data = r0[12];
          float v77_data = r0[13];
          float v78_data = r0[14];
          float v79_data = r0[15];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v76_data, v75_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v80_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v78_data, v81_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v79_data, v82_acc, 2, 3, 0);
          r2[0] = (v83_acc[0]);
          r2[1] = (v83_acc[1]);
          r2[2] = (v83_acc[2]);
          r2[3] = (v83_acc[3]);
          float v88_data = r1[4];
          float v89_data = r1[5];
          float v90_data = r1[6];
          float v91_data = r1[7];
          float v92_tp{};
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          tensorforge::transpose4x4b32(v92_tp, v93_tp, v94_tp, v95_tp, v88_data, v89_data, v90_data, v91_data);
          tensorforge::VectorT<float, 4> v96_acc{};
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v52_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v53_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v54_data, v102_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v55_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v109_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v110_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v111_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v112_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v117_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v118_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v119_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v120_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v125_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v126_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v127_acc, 2, 3, 0);
          r2[4] = (v128_acc[0]);
          r2[5] = (v128_acc[1]);
          r2[6] = (v128_acc[2]);
          r2[7] = (v128_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v10_lead < 12) {
            int32_t v147_a = ((v10_lead + 16_i32) + -12) - 4;
            #pragma unroll
            for (int32_t v137_i1 = 0; v137_i1 < 8; ++v137_i1) {
              int32_t v138_a = 0 + v137_i1;
              float v140_data = r2[v137_i1];
              glb_m0[(v147_a + (v137_i1 * 12))] = v140_data;
            }
          }
        }
      }
    }
  }
}

