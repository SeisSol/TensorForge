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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 9) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 9; ++v4_i1) {
              int32_t v11_a = v2_lead + (v4_i1 * 9);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m1[v11_a]);
              }
              int32_t v13_a = 0 + v4_i1;
              r0[v13_a] = v12_data;
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
            float v14_data = r0[0];
            float v15_data = r0[1];
            float v16_data = r0[2];
            float v17_data = r0[3];
            float v18_data = r0[4];
            float v19_data = r0[5];
            float v20_data = r0[6];
            float v21_data = r0[7];
            float v22_data = r0[8];
            float v23_acc{};
            float v24_acc{};
            float v25_acc{};
            float v26_acc{};
            float v27_acc{};
            float v28_acc{};
            float v29_acc{};
            float v30_acc{};
            float v31_acc{};
            float v32_lin = r1[0];
            tensorforge::fmacdpp16<0>(v23_acc, v32_lin, v14_data);
            tensorforge::fmacdpp16<1>(v23_acc, v32_lin, v15_data);
            tensorforge::fmacdpp16<2>(v23_acc, v32_lin, v16_data);
            tensorforge::fmacdpp16<3>(v23_acc, v32_lin, v17_data);
            tensorforge::fmacdpp16<4>(v23_acc, v32_lin, v18_data);
            tensorforge::fmacdpp16<5>(v23_acc, v32_lin, v19_data);
            tensorforge::fmacdpp16<6>(v23_acc, v32_lin, v20_data);
            tensorforge::fmacdpp16<7>(v23_acc, v32_lin, v21_data);
            tensorforge::fmacdpp16<8>(v23_acc, v32_lin, v22_data);
            tensorforge::fmacdpp16<9>(v24_acc, v32_lin, v14_data);
            tensorforge::fmacdpp16<10>(v24_acc, v32_lin, v15_data);
            tensorforge::fmacdpp16<11>(v24_acc, v32_lin, v16_data);
            tensorforge::fmacdpp16<12>(v24_acc, v32_lin, v17_data);
            tensorforge::fmacdpp16<13>(v24_acc, v32_lin, v18_data);
            tensorforge::fmacdpp16<14>(v24_acc, v32_lin, v19_data);
            tensorforge::fmacdpp16<15>(v24_acc, v32_lin, v20_data);
            float v33_lin = r1[1];
            tensorforge::fmacdpp16<0>(v24_acc, v33_lin, v21_data);
            tensorforge::fmacdpp16<1>(v24_acc, v33_lin, v22_data);
            tensorforge::fmacdpp16<2>(v25_acc, v33_lin, v14_data);
            tensorforge::fmacdpp16<3>(v25_acc, v33_lin, v15_data);
            tensorforge::fmacdpp16<4>(v25_acc, v33_lin, v16_data);
            tensorforge::fmacdpp16<5>(v25_acc, v33_lin, v17_data);
            tensorforge::fmacdpp16<6>(v25_acc, v33_lin, v18_data);
            tensorforge::fmacdpp16<7>(v25_acc, v33_lin, v19_data);
            tensorforge::fmacdpp16<8>(v25_acc, v33_lin, v20_data);
            tensorforge::fmacdpp16<9>(v25_acc, v33_lin, v21_data);
            tensorforge::fmacdpp16<10>(v25_acc, v33_lin, v22_data);
            tensorforge::fmacdpp16<11>(v26_acc, v33_lin, v14_data);
            tensorforge::fmacdpp16<12>(v26_acc, v33_lin, v15_data);
            tensorforge::fmacdpp16<13>(v26_acc, v33_lin, v16_data);
            tensorforge::fmacdpp16<14>(v26_acc, v33_lin, v17_data);
            tensorforge::fmacdpp16<15>(v26_acc, v33_lin, v18_data);
            float v34_lin = r1[2];
            tensorforge::fmacdpp16<0>(v26_acc, v34_lin, v19_data);
            tensorforge::fmacdpp16<1>(v26_acc, v34_lin, v20_data);
            tensorforge::fmacdpp16<2>(v26_acc, v34_lin, v21_data);
            tensorforge::fmacdpp16<3>(v26_acc, v34_lin, v22_data);
            tensorforge::fmacdpp16<4>(v27_acc, v34_lin, v14_data);
            tensorforge::fmacdpp16<5>(v27_acc, v34_lin, v15_data);
            tensorforge::fmacdpp16<6>(v27_acc, v34_lin, v16_data);
            tensorforge::fmacdpp16<7>(v27_acc, v34_lin, v17_data);
            tensorforge::fmacdpp16<8>(v27_acc, v34_lin, v18_data);
            tensorforge::fmacdpp16<9>(v27_acc, v34_lin, v19_data);
            tensorforge::fmacdpp16<10>(v27_acc, v34_lin, v20_data);
            tensorforge::fmacdpp16<11>(v27_acc, v34_lin, v21_data);
            tensorforge::fmacdpp16<12>(v27_acc, v34_lin, v22_data);
            tensorforge::fmacdpp16<13>(v28_acc, v34_lin, v14_data);
            tensorforge::fmacdpp16<14>(v28_acc, v34_lin, v15_data);
            tensorforge::fmacdpp16<15>(v28_acc, v34_lin, v16_data);
            float v35_lin = r1[3];
            tensorforge::fmacdpp16<0>(v28_acc, v35_lin, v17_data);
            tensorforge::fmacdpp16<1>(v28_acc, v35_lin, v18_data);
            tensorforge::fmacdpp16<2>(v28_acc, v35_lin, v19_data);
            tensorforge::fmacdpp16<3>(v28_acc, v35_lin, v20_data);
            tensorforge::fmacdpp16<4>(v28_acc, v35_lin, v21_data);
            tensorforge::fmacdpp16<5>(v28_acc, v35_lin, v22_data);
            tensorforge::fmacdpp16<6>(v29_acc, v35_lin, v14_data);
            tensorforge::fmacdpp16<7>(v29_acc, v35_lin, v15_data);
            tensorforge::fmacdpp16<8>(v29_acc, v35_lin, v16_data);
            tensorforge::fmacdpp16<9>(v29_acc, v35_lin, v17_data);
            tensorforge::fmacdpp16<10>(v29_acc, v35_lin, v18_data);
            tensorforge::fmacdpp16<11>(v29_acc, v35_lin, v19_data);
            tensorforge::fmacdpp16<12>(v29_acc, v35_lin, v20_data);
            tensorforge::fmacdpp16<13>(v29_acc, v35_lin, v21_data);
            tensorforge::fmacdpp16<14>(v29_acc, v35_lin, v22_data);
            tensorforge::fmacdpp16<15>(v30_acc, v35_lin, v14_data);
            float v36_lin = r1[4];
            tensorforge::fmacdpp16<0>(v30_acc, v36_lin, v15_data);
            tensorforge::fmacdpp16<1>(v30_acc, v36_lin, v16_data);
            tensorforge::fmacdpp16<2>(v30_acc, v36_lin, v17_data);
            tensorforge::fmacdpp16<3>(v30_acc, v36_lin, v18_data);
            tensorforge::fmacdpp16<4>(v30_acc, v36_lin, v19_data);
            tensorforge::fmacdpp16<5>(v30_acc, v36_lin, v20_data);
            tensorforge::fmacdpp16<6>(v30_acc, v36_lin, v21_data);
            tensorforge::fmacdpp16<7>(v30_acc, v36_lin, v22_data);
            tensorforge::fmacdpp16<8>(v31_acc, v36_lin, v14_data);
            tensorforge::fmacdpp16<9>(v31_acc, v36_lin, v15_data);
            tensorforge::fmacdpp16<10>(v31_acc, v36_lin, v16_data);
            tensorforge::fmacdpp16<11>(v31_acc, v36_lin, v17_data);
            tensorforge::fmacdpp16<12>(v31_acc, v36_lin, v18_data);
            tensorforge::fmacdpp16<13>(v31_acc, v36_lin, v19_data);
            tensorforge::fmacdpp16<14>(v31_acc, v36_lin, v20_data);
            tensorforge::fmacdpp16<15>(v31_acc, v36_lin, v21_data);
            float v37_lin = r1[5];
            tensorforge::fmacdpp16<0>(v31_acc, v37_lin, v22_data);
            ir2[0] = v23_acc;
            ir2[1] = v24_acc;
            ir2[2] = v25_acc;
            ir2[3] = v26_acc;
            ir2[4] = v27_acc;
            ir2[5] = v28_acc;
            ir2[6] = v29_acc;
            ir2[7] = v30_acc;
            ir2[8] = v31_acc;
            float v38_data;
            {
              v38_data = 0.0f;
              v38_data = 13.0f;
            }
            if ((threadIdx.x % 16) < 9) {
              #pragma unroll
              for (int32_t v43_n1 = 0; v43_n1 < 9; ++v43_n1) {
                int32_t v44_a = 0 + v43_n1;
                float v46_data = ir2[v43_n1];
                int32_t v48_a = 0 + v43_n1;
                r2[v43_n1] = (v46_data * v38_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r2);
          int32_t v52_lead = threadIdx.x % 16;
          if (v52_lead < 9) {
            #pragma unroll
            for (int32_t v54_i1 = 0; v54_i1 < 9; ++v54_i1) {
              int32_t v55_a = 0 + v54_i1;
              float v57_data = r2[v54_i1];
              int32_t v64_a = v52_lead + (v54_i1 * 9);
              glb_m0[v64_a] = v57_data;
            }
          }
          ;
        }
      }
    }
  }
}

