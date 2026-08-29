// === base name ===
kernel_a587425bdd

// === header ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (64, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a587425bdd, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_a587425bdd), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_a587425bdd, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // TMP = abs(A)
    // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
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
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
          float r1[8]{};
          // r1 = load{g>r}(glb_m2);
          float v8_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v8_lin;
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v12_lead = threadIdx.x % 64;
          if (v12_lead < 8) {
            #pragma unroll
            for (int32_t v14_k1 = 0; v14_k1 < 8; ++v14_k1) {
              float v22_data = glb_m0[(v12_lead + (v14_k1 * 8))];
              r0[v14_k1] = (fabsf(v22_data));
            }
          }
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v26_data = r0[0];
          float v27_data = r0[1];
          float v28_data = r0[2];
          float v29_data = r0[3];
          float v30_data = r0[4];
          float v31_data = r0[5];
          float v32_data = r0[6];
          float v33_data = r0[7];
          float v34_acc{};
          float v35_acc{};
          float v36_acc{};
          float v37_acc{};
          float v38_acc{};
          float v39_acc{};
          float v40_acc{};
          float v41_acc{};
          float v42_lin = r1[0];
          float v43_bc = tensorforge::broadcast<64, 16, 0>(v42_lin);
          tensorforge::fmacdpp16<0>(v34_acc, v43_bc, v26_data);
          tensorforge::fmacdpp16<1>(v34_acc, v43_bc, v27_data);
          tensorforge::fmacdpp16<2>(v34_acc, v43_bc, v28_data);
          tensorforge::fmacdpp16<3>(v34_acc, v43_bc, v29_data);
          tensorforge::fmacdpp16<4>(v34_acc, v43_bc, v30_data);
          tensorforge::fmacdpp16<5>(v34_acc, v43_bc, v31_data);
          tensorforge::fmacdpp16<6>(v34_acc, v43_bc, v32_data);
          tensorforge::fmacdpp16<7>(v34_acc, v43_bc, v33_data);
          tensorforge::fmacdpp16<8>(v35_acc, v43_bc, v26_data);
          tensorforge::fmacdpp16<9>(v35_acc, v43_bc, v27_data);
          tensorforge::fmacdpp16<10>(v35_acc, v43_bc, v28_data);
          tensorforge::fmacdpp16<11>(v35_acc, v43_bc, v29_data);
          tensorforge::fmacdpp16<12>(v35_acc, v43_bc, v30_data);
          tensorforge::fmacdpp16<13>(v35_acc, v43_bc, v31_data);
          tensorforge::fmacdpp16<14>(v35_acc, v43_bc, v32_data);
          tensorforge::fmacdpp16<15>(v35_acc, v43_bc, v33_data);
          float v44_bc = tensorforge::broadcast<64, 16, 1>(v42_lin);
          tensorforge::fmacdpp16<0>(v36_acc, v44_bc, v26_data);
          tensorforge::fmacdpp16<1>(v36_acc, v44_bc, v27_data);
          tensorforge::fmacdpp16<2>(v36_acc, v44_bc, v28_data);
          tensorforge::fmacdpp16<3>(v36_acc, v44_bc, v29_data);
          tensorforge::fmacdpp16<4>(v36_acc, v44_bc, v30_data);
          tensorforge::fmacdpp16<5>(v36_acc, v44_bc, v31_data);
          tensorforge::fmacdpp16<6>(v36_acc, v44_bc, v32_data);
          tensorforge::fmacdpp16<7>(v36_acc, v44_bc, v33_data);
          tensorforge::fmacdpp16<8>(v37_acc, v44_bc, v26_data);
          tensorforge::fmacdpp16<9>(v37_acc, v44_bc, v27_data);
          tensorforge::fmacdpp16<10>(v37_acc, v44_bc, v28_data);
          tensorforge::fmacdpp16<11>(v37_acc, v44_bc, v29_data);
          tensorforge::fmacdpp16<12>(v37_acc, v44_bc, v30_data);
          tensorforge::fmacdpp16<13>(v37_acc, v44_bc, v31_data);
          tensorforge::fmacdpp16<14>(v37_acc, v44_bc, v32_data);
          tensorforge::fmacdpp16<15>(v37_acc, v44_bc, v33_data);
          float v45_bc = tensorforge::broadcast<64, 16, 2>(v42_lin);
          tensorforge::fmacdpp16<0>(v38_acc, v45_bc, v26_data);
          tensorforge::fmacdpp16<1>(v38_acc, v45_bc, v27_data);
          tensorforge::fmacdpp16<2>(v38_acc, v45_bc, v28_data);
          tensorforge::fmacdpp16<3>(v38_acc, v45_bc, v29_data);
          tensorforge::fmacdpp16<4>(v38_acc, v45_bc, v30_data);
          tensorforge::fmacdpp16<5>(v38_acc, v45_bc, v31_data);
          tensorforge::fmacdpp16<6>(v38_acc, v45_bc, v32_data);
          tensorforge::fmacdpp16<7>(v38_acc, v45_bc, v33_data);
          tensorforge::fmacdpp16<8>(v39_acc, v45_bc, v26_data);
          tensorforge::fmacdpp16<9>(v39_acc, v45_bc, v27_data);
          tensorforge::fmacdpp16<10>(v39_acc, v45_bc, v28_data);
          tensorforge::fmacdpp16<11>(v39_acc, v45_bc, v29_data);
          tensorforge::fmacdpp16<12>(v39_acc, v45_bc, v30_data);
          tensorforge::fmacdpp16<13>(v39_acc, v45_bc, v31_data);
          tensorforge::fmacdpp16<14>(v39_acc, v45_bc, v32_data);
          tensorforge::fmacdpp16<15>(v39_acc, v45_bc, v33_data);
          float v46_bc = tensorforge::broadcast<64, 16, 3>(v42_lin);
          tensorforge::fmacdpp16<0>(v40_acc, v46_bc, v26_data);
          tensorforge::fmacdpp16<1>(v40_acc, v46_bc, v27_data);
          tensorforge::fmacdpp16<2>(v40_acc, v46_bc, v28_data);
          tensorforge::fmacdpp16<3>(v40_acc, v46_bc, v29_data);
          tensorforge::fmacdpp16<4>(v40_acc, v46_bc, v30_data);
          tensorforge::fmacdpp16<5>(v40_acc, v46_bc, v31_data);
          tensorforge::fmacdpp16<6>(v40_acc, v46_bc, v32_data);
          tensorforge::fmacdpp16<7>(v40_acc, v46_bc, v33_data);
          tensorforge::fmacdpp16<8>(v41_acc, v46_bc, v26_data);
          tensorforge::fmacdpp16<9>(v41_acc, v46_bc, v27_data);
          tensorforge::fmacdpp16<10>(v41_acc, v46_bc, v28_data);
          tensorforge::fmacdpp16<11>(v41_acc, v46_bc, v29_data);
          tensorforge::fmacdpp16<12>(v41_acc, v46_bc, v30_data);
          tensorforge::fmacdpp16<13>(v41_acc, v46_bc, v31_data);
          tensorforge::fmacdpp16<14>(v41_acc, v46_bc, v32_data);
          tensorforge::fmacdpp16<15>(v41_acc, v46_bc, v33_data);
          r2[0] = v34_acc;
          r2[1] = v35_acc;
          r2[2] = v36_acc;
          r2[3] = v37_acc;
          r2[4] = v38_acc;
          r2[5] = v39_acc;
          r2[6] = v40_acc;
          r2[7] = v41_acc;
          // glb_m1 = store{r>g}(r2);
          if (v12_lead < 8) {
            #pragma unroll
            for (int32_t v51_i1 = 0; v51_i1 < 8; ++v51_i1) {
              float v53_data = r2[v51_i1];
              glb_m1[(v12_lead + (v51_i1 * 8))] = v53_data;
            }
          }
        }
      }
    }
  }
}

