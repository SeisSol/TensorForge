// === base name ===
kernel_924fd3d329

// === header ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_924fd3d329, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_924fd3d329), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_924fd3d329, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×4(8×4) {0..8}×{0..4} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    // C = abs(TMP)
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 32 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 64;
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 8; ++v17_i1) {
              float v25_data = __builtin_nontemporal_load(&glb_m0[(v15_lead + (v17_i1 * 8))]);
              r0[v17_i1] = v25_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m1);
          float v28_lin = glb_m1[0 + threadIdx.x * 1];
          r1[0] = v28_lin;
          // wait(r0 = load{g>r}(glb_m0););
          float r3[4]{};
          // r3 = load{g>r}(glb_m2);
          float v30_lin = glb_m2[0 + threadIdx.x * 1];
          r3[0] = v30_lin;
          // wait(r1 = load{g>r}(glb_m1););
          float r2[4]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v32_data = r0[0];
          float v33_data = r0[1];
          float v34_data = r0[2];
          float v35_data = r0[3];
          float v36_data = r0[4];
          float v37_data = r0[5];
          float v38_data = r0[6];
          float v39_data = r0[7];
          float v40_acc{};
          float v41_acc{};
          float v42_acc{};
          float v43_acc{};
          float v44_lin = r1[0];
          float v45_bc = tensorforge::broadcast<64, 16, 0>(v44_lin);
          tensorforge::fmacdpp16<0>(v40_acc, v45_bc, v32_data);
          tensorforge::fmacdpp16<1>(v40_acc, v45_bc, v33_data);
          tensorforge::fmacdpp16<2>(v40_acc, v45_bc, v34_data);
          tensorforge::fmacdpp16<3>(v40_acc, v45_bc, v35_data);
          tensorforge::fmacdpp16<4>(v40_acc, v45_bc, v36_data);
          tensorforge::fmacdpp16<5>(v40_acc, v45_bc, v37_data);
          tensorforge::fmacdpp16<6>(v40_acc, v45_bc, v38_data);
          tensorforge::fmacdpp16<7>(v40_acc, v45_bc, v39_data);
          tensorforge::fmacdpp16<8>(v41_acc, v45_bc, v32_data);
          tensorforge::fmacdpp16<9>(v41_acc, v45_bc, v33_data);
          tensorforge::fmacdpp16<10>(v41_acc, v45_bc, v34_data);
          tensorforge::fmacdpp16<11>(v41_acc, v45_bc, v35_data);
          tensorforge::fmacdpp16<12>(v41_acc, v45_bc, v36_data);
          tensorforge::fmacdpp16<13>(v41_acc, v45_bc, v37_data);
          tensorforge::fmacdpp16<14>(v41_acc, v45_bc, v38_data);
          tensorforge::fmacdpp16<15>(v41_acc, v45_bc, v39_data);
          float v46_bc = tensorforge::broadcast<64, 16, 1>(v44_lin);
          tensorforge::fmacdpp16<0>(v42_acc, v46_bc, v32_data);
          tensorforge::fmacdpp16<1>(v42_acc, v46_bc, v33_data);
          tensorforge::fmacdpp16<2>(v42_acc, v46_bc, v34_data);
          tensorforge::fmacdpp16<3>(v42_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<4>(v42_acc, v46_bc, v36_data);
          tensorforge::fmacdpp16<5>(v42_acc, v46_bc, v37_data);
          tensorforge::fmacdpp16<6>(v42_acc, v46_bc, v38_data);
          tensorforge::fmacdpp16<7>(v42_acc, v46_bc, v39_data);
          tensorforge::fmacdpp16<8>(v43_acc, v46_bc, v32_data);
          tensorforge::fmacdpp16<9>(v43_acc, v46_bc, v33_data);
          tensorforge::fmacdpp16<10>(v43_acc, v46_bc, v34_data);
          tensorforge::fmacdpp16<11>(v43_acc, v46_bc, v35_data);
          tensorforge::fmacdpp16<12>(v43_acc, v46_bc, v36_data);
          tensorforge::fmacdpp16<13>(v43_acc, v46_bc, v37_data);
          tensorforge::fmacdpp16<14>(v43_acc, v46_bc, v38_data);
          tensorforge::fmacdpp16<15>(v43_acc, v46_bc, v39_data);
          r2[0] = v40_acc;
          r2[1] = v41_acc;
          r2[2] = v42_acc;
          r2[3] = v43_acc;
          // s0 = store{r>s}(localShrMem0, r2);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v51_i1 = 0; v51_i1 < 4; ++v51_i1) {
              float v53_data = r2[v51_i1];
              int32_t v60_a = v15_lead + (v51_i1 * 8);
              s0[(v60_a ^ ((v60_a >> 5) & 31))] = v53_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[4]{};
          // r4 = +(r0 * r3) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v73_acc{};
          float v74_acc{};
          float v75_acc{};
          float v76_acc{};
          float v77_lin = r3[0];
          float v78_bc = tensorforge::broadcast<64, 16, 0>(v77_lin);
          tensorforge::fmacdpp16<0>(v73_acc, v78_bc, v32_data);
          tensorforge::fmacdpp16<1>(v73_acc, v78_bc, v33_data);
          tensorforge::fmacdpp16<2>(v73_acc, v78_bc, v34_data);
          tensorforge::fmacdpp16<3>(v73_acc, v78_bc, v35_data);
          tensorforge::fmacdpp16<4>(v73_acc, v78_bc, v36_data);
          tensorforge::fmacdpp16<5>(v73_acc, v78_bc, v37_data);
          tensorforge::fmacdpp16<6>(v73_acc, v78_bc, v38_data);
          tensorforge::fmacdpp16<7>(v73_acc, v78_bc, v39_data);
          tensorforge::fmacdpp16<8>(v74_acc, v78_bc, v32_data);
          tensorforge::fmacdpp16<9>(v74_acc, v78_bc, v33_data);
          tensorforge::fmacdpp16<10>(v74_acc, v78_bc, v34_data);
          tensorforge::fmacdpp16<11>(v74_acc, v78_bc, v35_data);
          tensorforge::fmacdpp16<12>(v74_acc, v78_bc, v36_data);
          tensorforge::fmacdpp16<13>(v74_acc, v78_bc, v37_data);
          tensorforge::fmacdpp16<14>(v74_acc, v78_bc, v38_data);
          tensorforge::fmacdpp16<15>(v74_acc, v78_bc, v39_data);
          float v79_bc = tensorforge::broadcast<64, 16, 1>(v77_lin);
          tensorforge::fmacdpp16<0>(v75_acc, v79_bc, v32_data);
          tensorforge::fmacdpp16<1>(v75_acc, v79_bc, v33_data);
          tensorforge::fmacdpp16<2>(v75_acc, v79_bc, v34_data);
          tensorforge::fmacdpp16<3>(v75_acc, v79_bc, v35_data);
          tensorforge::fmacdpp16<4>(v75_acc, v79_bc, v36_data);
          tensorforge::fmacdpp16<5>(v75_acc, v79_bc, v37_data);
          tensorforge::fmacdpp16<6>(v75_acc, v79_bc, v38_data);
          tensorforge::fmacdpp16<7>(v75_acc, v79_bc, v39_data);
          tensorforge::fmacdpp16<8>(v76_acc, v79_bc, v32_data);
          tensorforge::fmacdpp16<9>(v76_acc, v79_bc, v33_data);
          tensorforge::fmacdpp16<10>(v76_acc, v79_bc, v34_data);
          tensorforge::fmacdpp16<11>(v76_acc, v79_bc, v35_data);
          tensorforge::fmacdpp16<12>(v76_acc, v79_bc, v36_data);
          tensorforge::fmacdpp16<13>(v76_acc, v79_bc, v37_data);
          tensorforge::fmacdpp16<14>(v76_acc, v79_bc, v38_data);
          tensorforge::fmacdpp16<15>(v76_acc, v79_bc, v39_data);
          r4[0] = v73_acc;
          r4[1] = v74_acc;
          r4[2] = v75_acc;
          r4[3] = v76_acc;
          // s0 = store{r>s}(localShrMem0, r4);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v84_i1 = 0; v84_i1 < 4; ++v84_i1) {
              float v86_data = r4[v84_i1];
              int32_t v94_a = v15_lead + ((v84_i1 + 4) * 8);
              s0[(v94_a ^ ((v94_a >> 5) & 31))] = v86_data;
            }
          }
          // glb_m3 = abs(s0)
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v102_k1 = 0; v102_k1 < 8; ++v102_k1) {
              int32_t v108_a = v102_k1 * 8;
              int32_t v109_a = v15_lead + v108_a;
              float v113_data = s0[(v109_a ^ ((v109_a >> 5) & 31))];
              glb_m3[(v15_lead + v108_a)] = (fabsf(v113_data));
            }
          }
        }
      }
    }
  }
}

