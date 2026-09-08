// === base name ===
kernel_4b748443ff

// === header ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4b748443ff, block.x * block.y * block.z, 512 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4b748443ff), hipFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_4b748443ff, grid, block, 512 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8(8) {0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // OUT = +(TMP, dims=[1])
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      __syncthreads();
      float* __restrict__ s0 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 8 + 0 + m2_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v14_lead = threadIdx.x % 32;
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m0[(v14_lead + (v16_i1 * 8))]);
              r0[v16_i1] = v24_data;
            }
          }
          float r1[8]{};
          // r1 = load{g>r}(glb_m1);
          float v27_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v27_lin;
          float v28_lin = glb_m1[32 + threadIdx.x * 1];
          r1[1] = v28_lin;
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m1););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v30_data = r0[0];
          float v31_data = r0[1];
          float v32_data = r0[2];
          float v33_data = r0[3];
          float v34_data = r0[4];
          float v35_data = r0[5];
          float v36_data = r0[6];
          float v37_data = r0[7];
          float v38_acc{};
          float v39_acc{};
          float v40_acc{};
          float v41_acc{};
          float v42_acc{};
          float v43_acc{};
          float v44_acc{};
          float v45_acc{};
          float v46_lin = r1[0];
          float v47_bc = tensorforge::broadcast<32, 16, 0>(v46_lin);
          tensorforge::fmacdpp16<0>(v38_acc, v47_bc, v30_data);
          tensorforge::fmacdpp16<1>(v38_acc, v47_bc, v31_data);
          tensorforge::fmacdpp16<2>(v38_acc, v47_bc, v32_data);
          tensorforge::fmacdpp16<3>(v38_acc, v47_bc, v33_data);
          tensorforge::fmacdpp16<4>(v38_acc, v47_bc, v34_data);
          tensorforge::fmacdpp16<5>(v38_acc, v47_bc, v35_data);
          tensorforge::fmacdpp16<6>(v38_acc, v47_bc, v36_data);
          tensorforge::fmacdpp16<7>(v38_acc, v47_bc, v37_data);
          tensorforge::fmacdpp16<8>(v39_acc, v47_bc, v30_data);
          tensorforge::fmacdpp16<9>(v39_acc, v47_bc, v31_data);
          tensorforge::fmacdpp16<10>(v39_acc, v47_bc, v32_data);
          tensorforge::fmacdpp16<11>(v39_acc, v47_bc, v33_data);
          tensorforge::fmacdpp16<12>(v39_acc, v47_bc, v34_data);
          tensorforge::fmacdpp16<13>(v39_acc, v47_bc, v35_data);
          tensorforge::fmacdpp16<14>(v39_acc, v47_bc, v36_data);
          tensorforge::fmacdpp16<15>(v39_acc, v47_bc, v37_data);
          float v48_bc = tensorforge::broadcast<32, 16, 1>(v46_lin);
          tensorforge::fmacdpp16<0>(v40_acc, v48_bc, v30_data);
          tensorforge::fmacdpp16<1>(v40_acc, v48_bc, v31_data);
          tensorforge::fmacdpp16<2>(v40_acc, v48_bc, v32_data);
          tensorforge::fmacdpp16<3>(v40_acc, v48_bc, v33_data);
          tensorforge::fmacdpp16<4>(v40_acc, v48_bc, v34_data);
          tensorforge::fmacdpp16<5>(v40_acc, v48_bc, v35_data);
          tensorforge::fmacdpp16<6>(v40_acc, v48_bc, v36_data);
          tensorforge::fmacdpp16<7>(v40_acc, v48_bc, v37_data);
          tensorforge::fmacdpp16<8>(v41_acc, v48_bc, v30_data);
          tensorforge::fmacdpp16<9>(v41_acc, v48_bc, v31_data);
          tensorforge::fmacdpp16<10>(v41_acc, v48_bc, v32_data);
          tensorforge::fmacdpp16<11>(v41_acc, v48_bc, v33_data);
          tensorforge::fmacdpp16<12>(v41_acc, v48_bc, v34_data);
          tensorforge::fmacdpp16<13>(v41_acc, v48_bc, v35_data);
          tensorforge::fmacdpp16<14>(v41_acc, v48_bc, v36_data);
          tensorforge::fmacdpp16<15>(v41_acc, v48_bc, v37_data);
          float v49_lin = r1[1];
          float v50_bc = tensorforge::broadcast<32, 16, 0>(v49_lin);
          tensorforge::fmacdpp16<0>(v42_acc, v50_bc, v30_data);
          tensorforge::fmacdpp16<1>(v42_acc, v50_bc, v31_data);
          tensorforge::fmacdpp16<2>(v42_acc, v50_bc, v32_data);
          tensorforge::fmacdpp16<3>(v42_acc, v50_bc, v33_data);
          tensorforge::fmacdpp16<4>(v42_acc, v50_bc, v34_data);
          tensorforge::fmacdpp16<5>(v42_acc, v50_bc, v35_data);
          tensorforge::fmacdpp16<6>(v42_acc, v50_bc, v36_data);
          tensorforge::fmacdpp16<7>(v42_acc, v50_bc, v37_data);
          tensorforge::fmacdpp16<8>(v43_acc, v50_bc, v30_data);
          tensorforge::fmacdpp16<9>(v43_acc, v50_bc, v31_data);
          tensorforge::fmacdpp16<10>(v43_acc, v50_bc, v32_data);
          tensorforge::fmacdpp16<11>(v43_acc, v50_bc, v33_data);
          tensorforge::fmacdpp16<12>(v43_acc, v50_bc, v34_data);
          tensorforge::fmacdpp16<13>(v43_acc, v50_bc, v35_data);
          tensorforge::fmacdpp16<14>(v43_acc, v50_bc, v36_data);
          tensorforge::fmacdpp16<15>(v43_acc, v50_bc, v37_data);
          float v51_bc = tensorforge::broadcast<32, 16, 1>(v49_lin);
          tensorforge::fmacdpp16<0>(v44_acc, v51_bc, v30_data);
          tensorforge::fmacdpp16<1>(v44_acc, v51_bc, v31_data);
          tensorforge::fmacdpp16<2>(v44_acc, v51_bc, v32_data);
          tensorforge::fmacdpp16<3>(v44_acc, v51_bc, v33_data);
          tensorforge::fmacdpp16<4>(v44_acc, v51_bc, v34_data);
          tensorforge::fmacdpp16<5>(v44_acc, v51_bc, v35_data);
          tensorforge::fmacdpp16<6>(v44_acc, v51_bc, v36_data);
          tensorforge::fmacdpp16<7>(v44_acc, v51_bc, v37_data);
          tensorforge::fmacdpp16<8>(v45_acc, v51_bc, v30_data);
          tensorforge::fmacdpp16<9>(v45_acc, v51_bc, v31_data);
          tensorforge::fmacdpp16<10>(v45_acc, v51_bc, v32_data);
          tensorforge::fmacdpp16<11>(v45_acc, v51_bc, v33_data);
          tensorforge::fmacdpp16<12>(v45_acc, v51_bc, v34_data);
          tensorforge::fmacdpp16<13>(v45_acc, v51_bc, v35_data);
          tensorforge::fmacdpp16<14>(v45_acc, v51_bc, v36_data);
          tensorforge::fmacdpp16<15>(v45_acc, v51_bc, v37_data);
          r2[0] = v38_acc;
          r2[1] = v39_acc;
          r2[2] = v40_acc;
          r2[3] = v41_acc;
          r2[4] = v42_acc;
          r2[5] = v43_acc;
          r2[6] = v44_acc;
          r2[7] = v45_acc;
          // s0 = store{r>s}(localShrMem0, r2);
          if (v14_lead < 8) {
            #pragma unroll
            for (int32_t v56_i1 = 0; v56_i1 < 8; ++v56_i1) {
              float v58_data = r2[v56_i1];
              int32_t v65_a = v14_lead + (v56_i1 * 8);
              s0[(v65_a ^ ((v65_a >> 5) & 31))] = v58_data;
            }
          }
          // glb_m2 = +(s0, dims=[1])
          if (v14_lead < 8) {
            float v74_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v73_r1 = 0; v73_r1 < 8; ++v73_r1) {
              int32_t v81_a = v14_lead + (v73_r1 * 8);
              float v85_data = s0[(v81_a ^ ((v81_a >> 5) & 31))];
              v74_acc0 = (v74_acc0 + v85_data);
            }
            glb_m2[v14_lead] = v74_acc0;
          }
        }
      }
    }
  }
}

