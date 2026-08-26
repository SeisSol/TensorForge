// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08a27dccde, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_08a27dccde), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_08a27dccde, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 9×9(9×9) {0..9}×{0..9} strided
    // m1 9×9(9×9) {0..9}×{0..9} strided
    // m2 9×9(9×9) {0..9}×{0..9} strided
    // m3 ()  scalar
    // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 9; ++v5_i1) {
              int32_t v11_a = v5_i1 * 9;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[9]{};
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
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          {
            // r2 = +(r0 * r1) + None
            // [(0, 9), (0, 9)] [(0, 9)]
            float ir2[9]{};
            float v24_data = r0[0];
            float v25_data = r0[1];
            float v26_data = r0[2];
            float v27_data = r0[3];
            float v28_data = r0[4];
            float v29_data = r0[5];
            float v30_data = r0[6];
            float v31_data = r0[7];
            float v32_data = r0[8];
            float v33_acc{};
            float v34_acc{};
            float v35_acc{};
            float v36_acc{};
            float v37_acc{};
            float v38_acc{};
            float v39_acc{};
            float v40_acc{};
            float v41_acc{};
            float v42_lin = r1[0];
            tensorforge::fmacdpp16<0>(v33_acc, v42_lin, v24_data);
            tensorforge::fmacdpp16<1>(v33_acc, v42_lin, v25_data);
            tensorforge::fmacdpp16<2>(v33_acc, v42_lin, v26_data);
            tensorforge::fmacdpp16<3>(v33_acc, v42_lin, v27_data);
            tensorforge::fmacdpp16<4>(v33_acc, v42_lin, v28_data);
            tensorforge::fmacdpp16<5>(v33_acc, v42_lin, v29_data);
            tensorforge::fmacdpp16<6>(v33_acc, v42_lin, v30_data);
            tensorforge::fmacdpp16<7>(v33_acc, v42_lin, v31_data);
            tensorforge::fmacdpp16<8>(v33_acc, v42_lin, v32_data);
            tensorforge::fmacdpp16<9>(v34_acc, v42_lin, v24_data);
            tensorforge::fmacdpp16<10>(v34_acc, v42_lin, v25_data);
            tensorforge::fmacdpp16<11>(v34_acc, v42_lin, v26_data);
            tensorforge::fmacdpp16<12>(v34_acc, v42_lin, v27_data);
            tensorforge::fmacdpp16<13>(v34_acc, v42_lin, v28_data);
            tensorforge::fmacdpp16<14>(v34_acc, v42_lin, v29_data);
            tensorforge::fmacdpp16<15>(v34_acc, v42_lin, v30_data);
            float v43_lin = r1[1];
            tensorforge::fmacdpp16<0>(v34_acc, v43_lin, v31_data);
            tensorforge::fmacdpp16<1>(v34_acc, v43_lin, v32_data);
            tensorforge::fmacdpp16<2>(v35_acc, v43_lin, v24_data);
            tensorforge::fmacdpp16<3>(v35_acc, v43_lin, v25_data);
            tensorforge::fmacdpp16<4>(v35_acc, v43_lin, v26_data);
            tensorforge::fmacdpp16<5>(v35_acc, v43_lin, v27_data);
            tensorforge::fmacdpp16<6>(v35_acc, v43_lin, v28_data);
            tensorforge::fmacdpp16<7>(v35_acc, v43_lin, v29_data);
            tensorforge::fmacdpp16<8>(v35_acc, v43_lin, v30_data);
            tensorforge::fmacdpp16<9>(v35_acc, v43_lin, v31_data);
            tensorforge::fmacdpp16<10>(v35_acc, v43_lin, v32_data);
            tensorforge::fmacdpp16<11>(v36_acc, v43_lin, v24_data);
            tensorforge::fmacdpp16<12>(v36_acc, v43_lin, v25_data);
            tensorforge::fmacdpp16<13>(v36_acc, v43_lin, v26_data);
            tensorforge::fmacdpp16<14>(v36_acc, v43_lin, v27_data);
            tensorforge::fmacdpp16<15>(v36_acc, v43_lin, v28_data);
            float v44_lin = r1[2];
            tensorforge::fmacdpp16<0>(v36_acc, v44_lin, v29_data);
            tensorforge::fmacdpp16<1>(v36_acc, v44_lin, v30_data);
            tensorforge::fmacdpp16<2>(v36_acc, v44_lin, v31_data);
            tensorforge::fmacdpp16<3>(v36_acc, v44_lin, v32_data);
            tensorforge::fmacdpp16<4>(v37_acc, v44_lin, v24_data);
            tensorforge::fmacdpp16<5>(v37_acc, v44_lin, v25_data);
            tensorforge::fmacdpp16<6>(v37_acc, v44_lin, v26_data);
            tensorforge::fmacdpp16<7>(v37_acc, v44_lin, v27_data);
            tensorforge::fmacdpp16<8>(v37_acc, v44_lin, v28_data);
            tensorforge::fmacdpp16<9>(v37_acc, v44_lin, v29_data);
            tensorforge::fmacdpp16<10>(v37_acc, v44_lin, v30_data);
            tensorforge::fmacdpp16<11>(v37_acc, v44_lin, v31_data);
            tensorforge::fmacdpp16<12>(v37_acc, v44_lin, v32_data);
            tensorforge::fmacdpp16<13>(v38_acc, v44_lin, v24_data);
            tensorforge::fmacdpp16<14>(v38_acc, v44_lin, v25_data);
            tensorforge::fmacdpp16<15>(v38_acc, v44_lin, v26_data);
            float v45_lin = r1[3];
            tensorforge::fmacdpp16<0>(v38_acc, v45_lin, v27_data);
            tensorforge::fmacdpp16<1>(v38_acc, v45_lin, v28_data);
            tensorforge::fmacdpp16<2>(v38_acc, v45_lin, v29_data);
            tensorforge::fmacdpp16<3>(v38_acc, v45_lin, v30_data);
            tensorforge::fmacdpp16<4>(v38_acc, v45_lin, v31_data);
            tensorforge::fmacdpp16<5>(v38_acc, v45_lin, v32_data);
            tensorforge::fmacdpp16<6>(v39_acc, v45_lin, v24_data);
            tensorforge::fmacdpp16<7>(v39_acc, v45_lin, v25_data);
            tensorforge::fmacdpp16<8>(v39_acc, v45_lin, v26_data);
            tensorforge::fmacdpp16<9>(v39_acc, v45_lin, v27_data);
            tensorforge::fmacdpp16<10>(v39_acc, v45_lin, v28_data);
            tensorforge::fmacdpp16<11>(v39_acc, v45_lin, v29_data);
            tensorforge::fmacdpp16<12>(v39_acc, v45_lin, v30_data);
            tensorforge::fmacdpp16<13>(v39_acc, v45_lin, v31_data);
            tensorforge::fmacdpp16<14>(v39_acc, v45_lin, v32_data);
            tensorforge::fmacdpp16<15>(v40_acc, v45_lin, v24_data);
            float v46_lin = r1[4];
            tensorforge::fmacdpp16<0>(v40_acc, v46_lin, v25_data);
            tensorforge::fmacdpp16<1>(v40_acc, v46_lin, v26_data);
            tensorforge::fmacdpp16<2>(v40_acc, v46_lin, v27_data);
            tensorforge::fmacdpp16<3>(v40_acc, v46_lin, v28_data);
            tensorforge::fmacdpp16<4>(v40_acc, v46_lin, v29_data);
            tensorforge::fmacdpp16<5>(v40_acc, v46_lin, v30_data);
            tensorforge::fmacdpp16<6>(v40_acc, v46_lin, v31_data);
            tensorforge::fmacdpp16<7>(v40_acc, v46_lin, v32_data);
            tensorforge::fmacdpp16<8>(v41_acc, v46_lin, v24_data);
            tensorforge::fmacdpp16<9>(v41_acc, v46_lin, v25_data);
            tensorforge::fmacdpp16<10>(v41_acc, v46_lin, v26_data);
            tensorforge::fmacdpp16<11>(v41_acc, v46_lin, v27_data);
            tensorforge::fmacdpp16<12>(v41_acc, v46_lin, v28_data);
            tensorforge::fmacdpp16<13>(v41_acc, v46_lin, v29_data);
            tensorforge::fmacdpp16<14>(v41_acc, v46_lin, v30_data);
            tensorforge::fmacdpp16<15>(v41_acc, v46_lin, v31_data);
            float v47_lin = r1[5];
            tensorforge::fmacdpp16<0>(v41_acc, v47_lin, v32_data);
            ir2[0] = v33_acc;
            ir2[1] = v34_acc;
            ir2[2] = v35_acc;
            ir2[3] = v36_acc;
            ir2[4] = v37_acc;
            ir2[5] = v38_acc;
            ir2[6] = v39_acc;
            ir2[7] = v40_acc;
            ir2[8] = v41_acc;
            if (v3_lead < 9) {
              #pragma unroll
              for (int32_t v53_n1 = 0; v53_n1 < 9; ++v53_n1) {
                int32_t v54_a = 0 + v53_n1;
                float v56_data = ir2[v53_n1];
                int32_t v58_a = 0 + v53_n1;
                r2[v53_n1] = (v56_data * 13.0f);
              }
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v3_lead < 9) {
            #pragma unroll
            for (int32_t v64_i1 = 0; v64_i1 < 9; ++v64_i1) {
              int32_t v65_a = 0 + v64_i1;
              float v67_data = r2[v64_i1];
              int32_t v74_a = v3_lead + (v64_i1 * 9);
              glb_m0[v74_a] = v67_data;
            }
          }
          ;
        }
      }
    }
  }
}

