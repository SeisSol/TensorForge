// === base name ===
kernel_49acf988a6

// === header ===
void launcher_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_49acf988a6, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_49acf988a6), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_49acf988a6, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 10; v5_i1 < 13; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + v11_a)]);
              int32_t v22_a = v4_i0 + (v5_i1 - 10);
              r0[v22_a] = v20_data;
            }
          }
          float r1[13]{};
          // r1 = load{g>r}(glb_m2);
          float v24_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v24_lin;
          float v25_lin = glb_m2[32 + threadIdx.x * 1];
          r1[1] = v25_lin;
          float v26_lin = glb_m2[64 + threadIdx.x * 1];
          r1[2] = v26_lin;
          float v27_lin = glb_m2[96 + threadIdx.x * 1];
          r1[3] = v27_lin;
          float v28_lin = glb_m2[128 + threadIdx.x * 1];
          r1[4] = v28_lin;
          float v29_lin = glb_m2[160 + threadIdx.x * 1];
          r1[5] = v29_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[7]{};
          // r2 = +(r0 * r1) + None
          // [(0, 32), (6, 13)] [(10, 13)]
          auto& ir2 = r2;
          float v31_data = r0[0];
          float v32_data = r0[1];
          float v33_data = r0[2];
          float v34_acc{};
          float v35_acc{};
          float v36_acc{};
          float v37_acc{};
          float v38_acc{};
          float v39_acc{};
          float v40_acc{};
          float v41_lin = r1[0];
          float v42_bc = tensorforge::broadcast<32, 16, 0>(v41_lin);
          tensorforge::fmacdpp16<0>(v34_acc, v42_bc, v31_data);
          tensorforge::fmacdpp16<1>(v34_acc, v42_bc, v32_data);
          tensorforge::fmacdpp16<2>(v34_acc, v42_bc, v33_data);
          tensorforge::fmacdpp16<3>(v35_acc, v42_bc, v31_data);
          tensorforge::fmacdpp16<4>(v35_acc, v42_bc, v32_data);
          tensorforge::fmacdpp16<5>(v35_acc, v42_bc, v33_data);
          tensorforge::fmacdpp16<6>(v36_acc, v42_bc, v31_data);
          tensorforge::fmacdpp16<7>(v36_acc, v42_bc, v32_data);
          tensorforge::fmacdpp16<8>(v36_acc, v42_bc, v33_data);
          tensorforge::fmacdpp16<9>(v37_acc, v42_bc, v31_data);
          tensorforge::fmacdpp16<10>(v37_acc, v42_bc, v32_data);
          tensorforge::fmacdpp16<11>(v37_acc, v42_bc, v33_data);
          tensorforge::fmacdpp16<12>(v38_acc, v42_bc, v31_data);
          tensorforge::fmacdpp16<13>(v38_acc, v42_bc, v32_data);
          tensorforge::fmacdpp16<14>(v38_acc, v42_bc, v33_data);
          tensorforge::fmacdpp16<15>(v39_acc, v42_bc, v31_data);
          float v43_bc = tensorforge::broadcast<32, 16, 1>(v41_lin);
          tensorforge::fmacdpp16<0>(v39_acc, v43_bc, v32_data);
          tensorforge::fmacdpp16<1>(v39_acc, v43_bc, v33_data);
          tensorforge::fmacdpp16<2>(v40_acc, v43_bc, v31_data);
          tensorforge::fmacdpp16<3>(v40_acc, v43_bc, v32_data);
          tensorforge::fmacdpp16<4>(v40_acc, v43_bc, v33_data);
          ir2[0] = v34_acc;
          ir2[1] = v35_acc;
          ir2[2] = v36_acc;
          ir2[3] = v37_acc;
          ir2[4] = v38_acc;
          ir2[5] = v39_acc;
          ir2[6] = v40_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v47_i0 = 0; v47_i0 < 1; ++v47_i0) {
            int32_t v52_lead = v47_i0 * 32;
            int32_t v54_a = (v3_lead + v52_lead) + 0;
            glb_m0[v54_a] = 0.0f;
            int32_t v61_a = (v3_lead + v52_lead) + 32;
            glb_m0[v61_a] = 0.0f;
            int32_t v68_a = (v3_lead + v52_lead) + 64;
            glb_m0[v68_a] = 0.0f;
            int32_t v75_a = (v3_lead + v52_lead) + 96;
            glb_m0[v75_a] = 0.0f;
            int32_t v82_a = (v3_lead + v52_lead) + 128;
            glb_m0[v82_a] = 0.0f;
            int32_t v89_a = (v3_lead + v52_lead) + 160;
            glb_m0[v89_a] = 0.0f;
            int32_t v90_a = v47_i0 + 0;
            float v92_data = r2[v47_i0];
            int32_t v98_a = (v3_lead + v52_lead) + 192;
            glb_m0[v98_a] = v92_data;
            int32_t v99_a = v47_i0 + 1;
            float v101_data = r2[(v47_i0 + 1)];
            int32_t v107_a = (v3_lead + v52_lead) + 224;
            glb_m0[v107_a] = v101_data;
            int32_t v108_a = v47_i0 + 2;
            float v110_data = r2[(v47_i0 + 2)];
            int32_t v116_a = (v3_lead + v52_lead) + 256;
            glb_m0[v116_a] = v110_data;
            int32_t v117_a = v47_i0 + 3;
            float v119_data = r2[(v47_i0 + 3)];
            int32_t v125_a = (v3_lead + v52_lead) + 288;
            glb_m0[v125_a] = v119_data;
            int32_t v126_a = v47_i0 + 4;
            float v128_data = r2[(v47_i0 + 4)];
            int32_t v134_a = (v3_lead + v52_lead) + 320;
            glb_m0[v134_a] = v128_data;
            int32_t v135_a = v47_i0 + 5;
            float v137_data = r2[(v47_i0 + 5)];
            int32_t v143_a = (v3_lead + v52_lead) + 352;
            glb_m0[v143_a] = v137_data;
            int32_t v144_a = v47_i0 + 6;
            float v146_data = r2[(v47_i0 + 6)];
            int32_t v152_a = (v3_lead + v52_lead) + 384;
            glb_m0[v152_a] = v146_data;
          }
          ;
        }
      }
    }
  }
}

