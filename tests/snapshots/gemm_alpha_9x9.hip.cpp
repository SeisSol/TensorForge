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
          int32_t v7_lead = threadIdx.x % 16;
          if (v7_lead < 9) {
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 9; ++v9_i1) {
              int32_t v15_a = v9_i1 * 9;
              int32_t v16_a = v7_lead + v15_a;
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v7_lead + v15_a)]);
              r0[v9_i1] = v24_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m2);
          float v27_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v27_lin;
          float v28_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v28_lin;
          float v29_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v29_lin;
          float v30_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v30_lin;
          float v31_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v31_lin;
          float v32_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v32_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir2[9]{};
          float v35_data = r0[0];
          float v36_data = r0[1];
          float v37_data = r0[2];
          float v38_data = r0[3];
          float v39_data = r0[4];
          float v40_data = r0[5];
          float v41_data = r0[6];
          float v42_data = r0[7];
          float v43_data = r0[8];
          float v44_acc{};
          float v45_acc{};
          float v46_acc{};
          float v47_acc{};
          float v48_acc{};
          float v49_acc{};
          float v50_acc{};
          float v51_acc{};
          float v52_acc{};
          float v53_lin = r1[0];
          tensorforge::fmacdpp16<0>(v44_acc, v53_lin, v35_data);
          tensorforge::fmacdpp16<1>(v44_acc, v53_lin, v36_data);
          tensorforge::fmacdpp16<2>(v44_acc, v53_lin, v37_data);
          tensorforge::fmacdpp16<3>(v44_acc, v53_lin, v38_data);
          tensorforge::fmacdpp16<4>(v44_acc, v53_lin, v39_data);
          tensorforge::fmacdpp16<5>(v44_acc, v53_lin, v40_data);
          tensorforge::fmacdpp16<6>(v44_acc, v53_lin, v41_data);
          tensorforge::fmacdpp16<7>(v44_acc, v53_lin, v42_data);
          tensorforge::fmacdpp16<8>(v44_acc, v53_lin, v43_data);
          tensorforge::fmacdpp16<9>(v45_acc, v53_lin, v35_data);
          tensorforge::fmacdpp16<10>(v45_acc, v53_lin, v36_data);
          tensorforge::fmacdpp16<11>(v45_acc, v53_lin, v37_data);
          tensorforge::fmacdpp16<12>(v45_acc, v53_lin, v38_data);
          tensorforge::fmacdpp16<13>(v45_acc, v53_lin, v39_data);
          tensorforge::fmacdpp16<14>(v45_acc, v53_lin, v40_data);
          tensorforge::fmacdpp16<15>(v45_acc, v53_lin, v41_data);
          float v54_lin = r1[1];
          tensorforge::fmacdpp16<0>(v45_acc, v54_lin, v42_data);
          tensorforge::fmacdpp16<1>(v45_acc, v54_lin, v43_data);
          tensorforge::fmacdpp16<2>(v46_acc, v54_lin, v35_data);
          tensorforge::fmacdpp16<3>(v46_acc, v54_lin, v36_data);
          tensorforge::fmacdpp16<4>(v46_acc, v54_lin, v37_data);
          tensorforge::fmacdpp16<5>(v46_acc, v54_lin, v38_data);
          tensorforge::fmacdpp16<6>(v46_acc, v54_lin, v39_data);
          tensorforge::fmacdpp16<7>(v46_acc, v54_lin, v40_data);
          tensorforge::fmacdpp16<8>(v46_acc, v54_lin, v41_data);
          tensorforge::fmacdpp16<9>(v46_acc, v54_lin, v42_data);
          tensorforge::fmacdpp16<10>(v46_acc, v54_lin, v43_data);
          tensorforge::fmacdpp16<11>(v47_acc, v54_lin, v35_data);
          tensorforge::fmacdpp16<12>(v47_acc, v54_lin, v36_data);
          tensorforge::fmacdpp16<13>(v47_acc, v54_lin, v37_data);
          tensorforge::fmacdpp16<14>(v47_acc, v54_lin, v38_data);
          tensorforge::fmacdpp16<15>(v47_acc, v54_lin, v39_data);
          float v55_lin = r1[2];
          tensorforge::fmacdpp16<0>(v47_acc, v55_lin, v40_data);
          tensorforge::fmacdpp16<1>(v47_acc, v55_lin, v41_data);
          tensorforge::fmacdpp16<2>(v47_acc, v55_lin, v42_data);
          tensorforge::fmacdpp16<3>(v47_acc, v55_lin, v43_data);
          tensorforge::fmacdpp16<4>(v48_acc, v55_lin, v35_data);
          tensorforge::fmacdpp16<5>(v48_acc, v55_lin, v36_data);
          tensorforge::fmacdpp16<6>(v48_acc, v55_lin, v37_data);
          tensorforge::fmacdpp16<7>(v48_acc, v55_lin, v38_data);
          tensorforge::fmacdpp16<8>(v48_acc, v55_lin, v39_data);
          tensorforge::fmacdpp16<9>(v48_acc, v55_lin, v40_data);
          tensorforge::fmacdpp16<10>(v48_acc, v55_lin, v41_data);
          tensorforge::fmacdpp16<11>(v48_acc, v55_lin, v42_data);
          tensorforge::fmacdpp16<12>(v48_acc, v55_lin, v43_data);
          tensorforge::fmacdpp16<13>(v49_acc, v55_lin, v35_data);
          tensorforge::fmacdpp16<14>(v49_acc, v55_lin, v36_data);
          tensorforge::fmacdpp16<15>(v49_acc, v55_lin, v37_data);
          float v56_lin = r1[3];
          tensorforge::fmacdpp16<0>(v49_acc, v56_lin, v38_data);
          tensorforge::fmacdpp16<1>(v49_acc, v56_lin, v39_data);
          tensorforge::fmacdpp16<2>(v49_acc, v56_lin, v40_data);
          tensorforge::fmacdpp16<3>(v49_acc, v56_lin, v41_data);
          tensorforge::fmacdpp16<4>(v49_acc, v56_lin, v42_data);
          tensorforge::fmacdpp16<5>(v49_acc, v56_lin, v43_data);
          tensorforge::fmacdpp16<6>(v50_acc, v56_lin, v35_data);
          tensorforge::fmacdpp16<7>(v50_acc, v56_lin, v36_data);
          tensorforge::fmacdpp16<8>(v50_acc, v56_lin, v37_data);
          tensorforge::fmacdpp16<9>(v50_acc, v56_lin, v38_data);
          tensorforge::fmacdpp16<10>(v50_acc, v56_lin, v39_data);
          tensorforge::fmacdpp16<11>(v50_acc, v56_lin, v40_data);
          tensorforge::fmacdpp16<12>(v50_acc, v56_lin, v41_data);
          tensorforge::fmacdpp16<13>(v50_acc, v56_lin, v42_data);
          tensorforge::fmacdpp16<14>(v50_acc, v56_lin, v43_data);
          tensorforge::fmacdpp16<15>(v51_acc, v56_lin, v35_data);
          float v57_lin = r1[4];
          tensorforge::fmacdpp16<0>(v51_acc, v57_lin, v36_data);
          tensorforge::fmacdpp16<1>(v51_acc, v57_lin, v37_data);
          tensorforge::fmacdpp16<2>(v51_acc, v57_lin, v38_data);
          tensorforge::fmacdpp16<3>(v51_acc, v57_lin, v39_data);
          tensorforge::fmacdpp16<4>(v51_acc, v57_lin, v40_data);
          tensorforge::fmacdpp16<5>(v51_acc, v57_lin, v41_data);
          tensorforge::fmacdpp16<6>(v51_acc, v57_lin, v42_data);
          tensorforge::fmacdpp16<7>(v51_acc, v57_lin, v43_data);
          tensorforge::fmacdpp16<8>(v52_acc, v57_lin, v35_data);
          tensorforge::fmacdpp16<9>(v52_acc, v57_lin, v36_data);
          tensorforge::fmacdpp16<10>(v52_acc, v57_lin, v37_data);
          tensorforge::fmacdpp16<11>(v52_acc, v57_lin, v38_data);
          tensorforge::fmacdpp16<12>(v52_acc, v57_lin, v39_data);
          tensorforge::fmacdpp16<13>(v52_acc, v57_lin, v40_data);
          tensorforge::fmacdpp16<14>(v52_acc, v57_lin, v41_data);
          tensorforge::fmacdpp16<15>(v52_acc, v57_lin, v42_data);
          float v58_lin = r1[5];
          tensorforge::fmacdpp16<0>(v52_acc, v58_lin, v43_data);
          ir2[0] = v44_acc;
          ir2[1] = v45_acc;
          ir2[2] = v46_acc;
          ir2[3] = v47_acc;
          ir2[4] = v48_acc;
          ir2[5] = v49_acc;
          ir2[6] = v50_acc;
          ir2[7] = v51_acc;
          ir2[8] = v52_acc;
          if (v7_lead < 9) {
            #pragma unroll
            for (int32_t v64_n1 = 0; v64_n1 < 9; ++v64_n1) {
              int32_t v65_a = 0 + v64_n1;
              float v67_data = ir2[v64_n1];
              r2[v64_n1] = (v67_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v7_lead < 9) {
            #pragma unroll
            for (int32_t v74_i1 = 0; v74_i1 < 9; ++v74_i1) {
              int32_t v75_a = 0 + v74_i1;
              float v77_data = r2[v74_i1];
              glb_m0[(v7_lead + (v74_i1 * 9))] = v77_data;
            }
          }
        }
      }
    }
  }
}

