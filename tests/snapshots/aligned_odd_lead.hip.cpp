// === base name ===
kernel_69f2bb9311

// === header ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_69f2bb9311, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_69f2bb9311), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_69f2bb9311, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 35×4(35×4) {0..35}×{0..4} strided
    // m1 35×8(35×8) {0..35}×{0..8} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 32);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v12_i1 * 35))]);
              r0[(v11_i0 + (v12_i1 * 2))] = v20_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v29_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m1[(v29_lead + (v24_i1 * 35))]);
              r0[(1 + (v24_i1 * 2))] = v32_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m2);
          float v36_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v36_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 35), (0, 4)] [(0, 8)]
          float v38_data = r0[0];
          float v39_data = r0[1];
          float v40_data = r0[2];
          float v41_data = r0[3];
          float v42_data = r0[4];
          float v43_data = r0[5];
          float v44_data = r0[6];
          float v45_data = r0[7];
          float v46_data = r0[8];
          float v47_data = r0[9];
          float v48_data = r0[10];
          float v49_data = r0[11];
          float v50_data = r0[12];
          float v51_data = r0[13];
          float v52_data = r0[14];
          float v53_data = r0[15];
          float v54_acc{};
          float v55_acc{};
          float v56_acc{};
          float v57_acc{};
          float v58_acc{};
          float v59_acc{};
          float v60_acc{};
          float v61_acc{};
          float v62_lin = r1[0];
          float v63_bc = tensorforge::broadcast<32, 16, 0>(v62_lin);
          tensorforge::fmacdpp16<0>(v54_acc, v63_bc, v38_data);
          tensorforge::fmacdpp16<0>(v55_acc, v63_bc, v39_data);
          tensorforge::fmacdpp16<1>(v54_acc, v63_bc, v40_data);
          tensorforge::fmacdpp16<1>(v55_acc, v63_bc, v41_data);
          tensorforge::fmacdpp16<2>(v54_acc, v63_bc, v42_data);
          tensorforge::fmacdpp16<2>(v55_acc, v63_bc, v43_data);
          tensorforge::fmacdpp16<3>(v54_acc, v63_bc, v44_data);
          tensorforge::fmacdpp16<3>(v55_acc, v63_bc, v45_data);
          tensorforge::fmacdpp16<4>(v54_acc, v63_bc, v46_data);
          tensorforge::fmacdpp16<4>(v55_acc, v63_bc, v47_data);
          tensorforge::fmacdpp16<5>(v54_acc, v63_bc, v48_data);
          tensorforge::fmacdpp16<5>(v55_acc, v63_bc, v49_data);
          tensorforge::fmacdpp16<6>(v54_acc, v63_bc, v50_data);
          tensorforge::fmacdpp16<6>(v55_acc, v63_bc, v51_data);
          tensorforge::fmacdpp16<7>(v54_acc, v63_bc, v52_data);
          tensorforge::fmacdpp16<7>(v55_acc, v63_bc, v53_data);
          tensorforge::fmacdpp16<8>(v56_acc, v63_bc, v38_data);
          tensorforge::fmacdpp16<8>(v57_acc, v63_bc, v39_data);
          tensorforge::fmacdpp16<9>(v56_acc, v63_bc, v40_data);
          tensorforge::fmacdpp16<9>(v57_acc, v63_bc, v41_data);
          tensorforge::fmacdpp16<10>(v56_acc, v63_bc, v42_data);
          tensorforge::fmacdpp16<10>(v57_acc, v63_bc, v43_data);
          tensorforge::fmacdpp16<11>(v56_acc, v63_bc, v44_data);
          tensorforge::fmacdpp16<11>(v57_acc, v63_bc, v45_data);
          tensorforge::fmacdpp16<12>(v56_acc, v63_bc, v46_data);
          tensorforge::fmacdpp16<12>(v57_acc, v63_bc, v47_data);
          tensorforge::fmacdpp16<13>(v56_acc, v63_bc, v48_data);
          tensorforge::fmacdpp16<13>(v57_acc, v63_bc, v49_data);
          tensorforge::fmacdpp16<14>(v56_acc, v63_bc, v50_data);
          tensorforge::fmacdpp16<14>(v57_acc, v63_bc, v51_data);
          tensorforge::fmacdpp16<15>(v56_acc, v63_bc, v52_data);
          tensorforge::fmacdpp16<15>(v57_acc, v63_bc, v53_data);
          float v64_bc = tensorforge::broadcast<32, 16, 1>(v62_lin);
          tensorforge::fmacdpp16<0>(v58_acc, v64_bc, v38_data);
          tensorforge::fmacdpp16<0>(v59_acc, v64_bc, v39_data);
          tensorforge::fmacdpp16<1>(v58_acc, v64_bc, v40_data);
          tensorforge::fmacdpp16<1>(v59_acc, v64_bc, v41_data);
          tensorforge::fmacdpp16<2>(v58_acc, v64_bc, v42_data);
          tensorforge::fmacdpp16<2>(v59_acc, v64_bc, v43_data);
          tensorforge::fmacdpp16<3>(v58_acc, v64_bc, v44_data);
          tensorforge::fmacdpp16<3>(v59_acc, v64_bc, v45_data);
          tensorforge::fmacdpp16<4>(v58_acc, v64_bc, v46_data);
          tensorforge::fmacdpp16<4>(v59_acc, v64_bc, v47_data);
          tensorforge::fmacdpp16<5>(v58_acc, v64_bc, v48_data);
          tensorforge::fmacdpp16<5>(v59_acc, v64_bc, v49_data);
          tensorforge::fmacdpp16<6>(v58_acc, v64_bc, v50_data);
          tensorforge::fmacdpp16<6>(v59_acc, v64_bc, v51_data);
          tensorforge::fmacdpp16<7>(v58_acc, v64_bc, v52_data);
          tensorforge::fmacdpp16<7>(v59_acc, v64_bc, v53_data);
          tensorforge::fmacdpp16<8>(v60_acc, v64_bc, v38_data);
          tensorforge::fmacdpp16<8>(v61_acc, v64_bc, v39_data);
          tensorforge::fmacdpp16<9>(v60_acc, v64_bc, v40_data);
          tensorforge::fmacdpp16<9>(v61_acc, v64_bc, v41_data);
          tensorforge::fmacdpp16<10>(v60_acc, v64_bc, v42_data);
          tensorforge::fmacdpp16<10>(v61_acc, v64_bc, v43_data);
          tensorforge::fmacdpp16<11>(v60_acc, v64_bc, v44_data);
          tensorforge::fmacdpp16<11>(v61_acc, v64_bc, v45_data);
          tensorforge::fmacdpp16<12>(v60_acc, v64_bc, v46_data);
          tensorforge::fmacdpp16<12>(v61_acc, v64_bc, v47_data);
          tensorforge::fmacdpp16<13>(v60_acc, v64_bc, v48_data);
          tensorforge::fmacdpp16<13>(v61_acc, v64_bc, v49_data);
          tensorforge::fmacdpp16<14>(v60_acc, v64_bc, v50_data);
          tensorforge::fmacdpp16<14>(v61_acc, v64_bc, v51_data);
          tensorforge::fmacdpp16<15>(v60_acc, v64_bc, v52_data);
          tensorforge::fmacdpp16<15>(v61_acc, v64_bc, v53_data);
          r2[0] = v54_acc;
          r2[1] = v55_acc;
          r2[2] = v56_acc;
          r2[3] = v57_acc;
          r2[4] = v58_acc;
          r2[5] = v59_acc;
          r2[6] = v60_acc;
          r2[7] = v61_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
            int32_t v77_lead = v10_lead + (v68_i0 * 32);
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 4; ++v69_i1) {
              float v72_data = r2[(v68_i0 + (v69_i1 * 2))];
              glb_m0[(v77_lead + (v69_i1 * 35))] = v72_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v89_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v81_i1 = 0; v81_i1 < 4; ++v81_i1) {
              float v84_data = r2[(1 + (v81_i1 * 2))];
              glb_m0[(v89_lead + (v81_i1 * 35))] = v84_data;
            }
          }
        }
      }
    }
  }
}

