// === base name ===
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ead773dd51, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_ead773dd51), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_ead773dd51, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 32×16(32×16) {0..32}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 12) {
            int32_t v18_off = v10_lead + 4;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              float v21_data = __builtin_nontemporal_load(&glb_m1[(v18_off + (v12_i1 * 32))]);
              r0[v12_i1] = v21_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v24_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v24_lin;
          float v25_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v25_lin;
          float v26_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v26_lin;
          float v27_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v27_lin;
          float v28_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v28_lin;
          float v29_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v29_lin;
          float v30_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v30_lin;
          float v31_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v31_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 8)] [(0, 16)]
          float v33_data = r1[0];
          float v34_data = r1[1];
          float v35_data = r1[2];
          float v36_data = r1[3];
          float v37_tp{};
          float v38_tp{};
          float v39_tp{};
          float v40_tp{};
          tensorforge::transpose4x4b32(v37_tp, v38_tp, v39_tp, v40_tp, v33_data, v34_data, v35_data, v36_data);
          tensorforge::VectorT<float, 4> v41_acc{};
          float v42_data = r0[0];
          float v43_data = r0[1];
          float v44_data = r0[2];
          float v45_data = r0[3];
          tensorforge::VectorT<float, 4> v46_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v42_data, v41_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v47_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v43_data, v46_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v44_data, v47_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v45_data, v48_acc, 2, 0, 0);
          float v50_data = r0[4];
          float v51_data = r0[5];
          float v52_data = r0[6];
          float v53_data = r0[7];
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v50_data, v49_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v51_data, v54_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v52_data, v55_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v53_data, v56_acc, 2, 1, 0);
          float v58_data = r0[8];
          float v59_data = r0[9];
          float v60_data = r0[10];
          float v61_data = r0[11];
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v58_data, v57_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v59_data, v62_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v60_data, v63_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v61_data, v64_acc, 2, 2, 0);
          float v66_data = r0[12];
          float v67_data = r0[13];
          float v68_data = r0[14];
          float v69_data = r0[15];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v66_data, v65_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v67_data, v70_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v39_tp, v68_data, v71_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v40_tp, v69_data, v72_acc, 2, 3, 0);
          r2[0] = (v73_acc[0]);
          r2[1] = (v73_acc[1]);
          r2[2] = (v73_acc[2]);
          r2[3] = (v73_acc[3]);
          float v78_data = r1[4];
          float v79_data = r1[5];
          float v80_data = r1[6];
          float v81_data = r1[7];
          float v82_tp{};
          float v83_tp{};
          float v84_tp{};
          float v85_tp{};
          tensorforge::transpose4x4b32(v82_tp, v83_tp, v84_tp, v85_tp, v78_data, v79_data, v80_data, v81_data);
          tensorforge::VectorT<float, 4> v86_acc{};
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v42_data, v86_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v43_data, v91_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v44_data, v92_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v45_data, v93_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v50_data, v94_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v51_data, v99_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v52_data, v100_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v53_data, v101_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v58_data, v102_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v59_data, v107_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v108_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v109_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v82_tp, v66_data, v110_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v67_data, v115_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v116_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v117_acc, 2, 3, 0);
          r2[4] = (v118_acc[0]);
          r2[5] = (v118_acc[1]);
          r2[6] = (v118_acc[2]);
          r2[7] = (v118_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v127_i1 = 0; v127_i1 < 8; ++v127_i1) {
              float v129_data = r2[v127_i1];
              glb_m0[(v10_lead + (v127_i1 * 12))] = v129_data;
            }
          }
        }
      }
    }
  }
}

