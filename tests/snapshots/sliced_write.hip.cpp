// === base name ===
kernel_b2bf69913aa4d94f

// === header ===
void launcher_kernel_b2bf69913aa4d94f(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_b2bf69913aa4d94f(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b2bf69913aa4d94f, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b2bf69913aa4d94f), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_b2bf69913aa4d94f, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b2bf69913aa4d94f(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{6..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{6..13})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 32);
            #pragma unroll
            for (int32_t v12_i1 = 10; v12_i1 < 13; ++v12_i1) {
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v12_i1 * 32))]);
              r0[(v11_i0 + (v12_i1 - 10))] = v20_data;
            }
          }
          float r1[7]{};
          // r1 = load{g>r}(glb_m2);
          if ((v10_lead >= 10) && (v10_lead < 13)) {
            #pragma unroll
            for (int32_t v30_i1 = 6; v30_i1 < 13; ++v30_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m2[(v10_lead + (v30_i1 * 13))]);
              r1[(v30_i1 - 6)] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[7]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (6, 13)] [(10, 13)]
          float v42_data = r1[0];
          float v43_data = r1[1];
          float v44_data = r1[2];
          float v45_data = r1[3];
          float v46_tp{};
          float v47_tp{};
          float v48_tp{};
          float v49_tp{};
          tensorforge::transpose4x4b32(v46_tp, v47_tp, v48_tp, v49_tp, v42_data, v43_data, v44_data, v45_data);
          tensorforge::VectorT<float, 4> v50_acc{};
          float v51_data = r0[0];
          float v52_data = r0[1];
          float v53_data = r0[2];
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v50_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v51_data, v57_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v52_data, v58_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v53_data, v63_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v51_data, v64_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v52_data, v65_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v53_data, v66_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v51_data, v71_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v52_data, v72_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v53_data, v73_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v74_acc, 3, 3, 0);
          r2[0] = (v79_acc[0]);
          r2[1] = (v79_acc[1]);
          r2[2] = (v79_acc[2]);
          r2[3] = (v79_acc[3]);
          float v84_data = r1[4];
          float v85_data = r1[5];
          float v86_data = r1[6];
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          float v91_tp{};
          tensorforge::transpose4x4b32(v88_tp, v89_tp, v90_tp, v91_tp, v84_data, v85_data, v86_data, 0.0f);
          tensorforge::VectorT<float, 4> v92_acc{};
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v51_data, v92_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v52_data, v97_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v53_data, v98_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v51_data, v99_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v52_data, v100_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v53_data, v105_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v51_data, v106_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v52_data, v107_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v53_data, v108_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v51_data, v113_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v52_data, v114_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v91_tp, v53_data, v115_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v51_data, v116_acc, 3, 3, 0);
          r2[4] = (v121_acc[0]);
          r2[5] = (v121_acc[1]);
          r2[6] = (v121_acc[2]);
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v128_i0 = 0; v128_i0 < 1; ++v128_i0) {
            int32_t v133_lead = v128_i0 * 32;
            glb_m0[(v10_lead + v133_lead)] = 0.0f;
            glb_m0[((v10_lead + v133_lead) + 32)] = 0.0f;
            glb_m0[((v10_lead + v133_lead) + 64)] = 0.0f;
            glb_m0[((v10_lead + v133_lead) + 96)] = 0.0f;
            glb_m0[((v10_lead + v133_lead) + 128)] = 0.0f;
            glb_m0[((v10_lead + v133_lead) + 160)] = 0.0f;
            float v172_data = r2[v128_i0];
            glb_m0[((v10_lead + v133_lead) + 192)] = v172_data;
            float v180_data = r2[(v128_i0 + 1)];
            glb_m0[((v10_lead + v133_lead) + 224)] = v180_data;
            float v188_data = r2[(v128_i0 + 2)];
            glb_m0[((v10_lead + v133_lead) + 256)] = v188_data;
            float v196_data = r2[(v128_i0 + 3)];
            glb_m0[((v10_lead + v133_lead) + 288)] = v196_data;
            float v204_data = r2[(v128_i0 + 4)];
            glb_m0[((v10_lead + v133_lead) + 320)] = v204_data;
            float v212_data = r2[(v128_i0 + 5)];
            glb_m0[((v10_lead + v133_lead) + 352)] = v212_data;
            float v220_data = r2[(v128_i0 + 6)];
            glb_m0[((v10_lead + v133_lead) + 384)] = v220_data;
          }
        }
      }
    }
  }
}

