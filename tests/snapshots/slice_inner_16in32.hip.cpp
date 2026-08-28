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
          int32_t v6_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
            int32_t v12_lead = v7_i0 * 16;
            int32_t v14_off = (v6_lead + v12_lead) + 8;
            int32_t v22_off = (v6_lead + v12_lead) + 8;
            #pragma unroll
            for (int32_t v8_i1 = 8; v8_i1 < 24; ++v8_i1) {
              int32_t v15_a = v8_i1 * 32;
              int32_t v16_a = v14_off + v15_a;
              float v25_data = __builtin_nontemporal_load(&glb_m1[(v22_off + v15_a)]);
              int32_t v27_a = v7_i0 + (v8_i1 - 8);
              r0[v27_a] = v25_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v29_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v29_lin;
          float v30_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v30_lin;
          float v31_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v31_lin;
          float v32_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v32_lin;
          float v33_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v33_lin;
          float v34_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v34_lin;
          float v35_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v35_lin;
          float v36_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v36_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float v38_data = r1[0];
          float v39_data = r1[1];
          float v40_data = r1[2];
          float v41_data = r1[3];
          float v42_tp{};
          float v43_tp{};
          float v44_tp{};
          float v45_tp{};
          tensorforge::transpose4x4b32(v42_tp, v43_tp, v44_tp, v45_tp, v38_data, v39_data, v40_data, v41_data);
          tensorforge::VectorT<float, 4> v46_acc{};
          float v47_data = r0[0];
          float v48_data = r0[1];
          float v49_data = r0[2];
          float v50_data = r0[3];
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v47_data, v46_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v48_data, v51_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v52_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v53_acc, 2, 0, 0);
          float v55_data = r0[4];
          float v56_data = r0[5];
          float v57_data = r0[6];
          float v58_data = r0[7];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v55_data, v54_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v56_data, v59_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v60_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v61_acc, 2, 1, 0);
          float v63_data = r0[8];
          float v64_data = r0[9];
          float v65_data = r0[10];
          float v66_data = r0[11];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v63_data, v62_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v64_data, v67_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v65_data, v68_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v69_acc, 2, 2, 0);
          float v71_data = r0[12];
          float v72_data = r0[13];
          float v73_data = r0[14];
          float v74_data = r0[15];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v71_data, v70_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v72_data, v75_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v73_data, v76_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v77_acc, 2, 3, 0);
          r2[0] = (v78_acc[0]);
          r2[1] = (v78_acc[1]);
          r2[2] = (v78_acc[2]);
          r2[3] = (v78_acc[3]);
          float v83_data = r1[4];
          float v84_data = r1[5];
          float v85_data = r1[6];
          float v86_data = r1[7];
          float v87_tp{};
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          tensorforge::transpose4x4b32(v87_tp, v88_tp, v89_tp, v90_tp, v83_data, v84_data, v85_data, v86_data);
          tensorforge::VectorT<float, 4> v91_acc{};
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v47_data, v91_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v48_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v49_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v50_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v99_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v56_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v107_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v112_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v113_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v114_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v115_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v72_data, v120_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v121_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v122_acc, 2, 3, 0);
          r2[4] = (v123_acc[0]);
          r2[5] = (v123_acc[1]);
          r2[6] = (v123_acc[2]);
          r2[7] = (v123_acc[3]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v131_i0 = 0; v131_i0 < 1; ++v131_i0) {
            int32_t v140_lead = v6_lead + (v131_i0 * 16);
            #pragma unroll
            for (int32_t v132_i1 = 0; v132_i1 < 8; ++v132_i1) {
              int32_t v133_a = v131_i0 + v132_i1;
              float v135_data = r2[(v131_i0 + v132_i1)];
              int32_t v142_a = v140_lead + (v132_i1 * 16);
              glb_m0[v142_a] = v135_data;
            }
          }
        }
      }
    }
  }
}

