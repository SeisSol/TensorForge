// === base name ===
kernel_f61651fe59

// === header ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f61651fe59, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_f61651fe59), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_f61651fe59, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 32×16(12×16) {4..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(12×16) {4..16}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          if (v6_lead < 12) {
            int32_t v15_a = (v6_lead + 4) - 4;
            int32_t v24_a = (v6_lead + 4) - 4;
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 16; ++v8_i1) {
              int32_t v16_a = v8_i1 * 12;
              int32_t v17_a = v15_a + v16_a;
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v24_a + v16_a)]);
              int32_t v28_a = 0 + v8_i1;
              r0[v28_a] = v27_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v30_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v30_lin;
          float v31_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v31_lin;
          float v32_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v32_lin;
          float v33_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v33_lin;
          float v34_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v34_lin;
          float v35_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v35_lin;
          float v36_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v36_lin;
          float v37_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v37_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 16)]
          float v39_data = r1[0];
          float v40_data = r1[1];
          float v41_data = r1[2];
          float v42_data = r1[3];
          float v43_tp{};
          float v44_tp{};
          float v45_tp{};
          float v46_tp{};
          tensorforge::transpose4x4b32(v43_tp, v44_tp, v45_tp, v46_tp, v39_data, v40_data, v41_data, v42_data);
          tensorforge::VectorT<float, 4> v47_acc{};
          float v48_data = r0[0];
          float v49_data = r0[1];
          float v50_data = r0[2];
          float v51_data = r0[3];
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v48_data, v47_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v52_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v53_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v54_acc, 2, 0, 0);
          float v56_data = r0[4];
          float v57_data = r0[5];
          float v58_data = r0[6];
          float v59_data = r0[7];
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v56_data, v55_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v60_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v61_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v62_acc, 2, 1, 0);
          float v64_data = r0[8];
          float v65_data = r0[9];
          float v66_data = r0[10];
          float v67_data = r0[11];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v64_data, v63_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v65_data, v68_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v69_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v70_acc, 2, 2, 0);
          float v72_data = r0[12];
          float v73_data = r0[13];
          float v74_data = r0[14];
          float v75_data = r0[15];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v72_data, v71_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v73_data, v76_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v77_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v75_data, v78_acc, 2, 3, 0);
          r2[0] = (v79_acc[0]);
          r2[1] = (v79_acc[1]);
          r2[2] = (v79_acc[2]);
          r2[3] = (v79_acc[3]);
          float v84_data = r1[4];
          float v85_data = r1[5];
          float v86_data = r1[6];
          float v87_data = r1[7];
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          tensorforge::transpose4x4b32(v88_tp, v89_tp, v90_tp, v91_tp, v84_data, v85_data, v86_data, v87_data);
          tensorforge::VectorT<float, 4> v92_acc{};
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v48_data, v92_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v49_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v50_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v51_data, v99_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v56_data, v100_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v59_data, v107_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v108_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v113_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v114_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v67_data, v115_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v72_data, v116_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v121_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v122_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v75_data, v123_acc, 2, 3, 0);
          r2[4] = (v124_acc[0]);
          r2[5] = (v124_acc[1]);
          r2[6] = (v124_acc[2]);
          r2[7] = (v124_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v133_i1 = 0; v133_i1 < 8; ++v133_i1) {
              int32_t v134_a = 0 + v133_i1;
              float v136_data = r2[v133_i1];
              int32_t v143_a = v6_lead + (v133_i1 * 12);
              glb_m0[v143_a] = v136_data;
            }
          }
        }
      }
    }
  }
}

