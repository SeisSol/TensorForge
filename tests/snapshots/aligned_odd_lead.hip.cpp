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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v7_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
            int32_t v13_lead = v8_i0 * 32;
            int32_t v14_lead = v7_lead + v13_lead;
            int32_t v21_lead = v7_lead + v13_lead;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 8; ++v9_i1) {
              int32_t v15_a = v9_i1 * 35;
              int32_t v16_a = v14_lead + v15_a;
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v21_lead + v15_a)]);
              r0[(v8_i0 + (v9_i1 * 2))] = v24_data;
            }
          }
          if (v7_lead < 3) {
            int32_t v33_lead = v7_lead + 32_i32;
            int32_t v40_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
              int32_t v34_a = v28_i1 * 35;
              int32_t v35_a = v33_lead + v34_a;
              float v43_data = __builtin_nontemporal_load(&glb_m1[(v40_lead + v34_a)]);
              r0[(1 + (v28_i1 * 2))] = v43_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m2);
          float v47_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v47_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 35), (0, 4)] [(0, 8)]
          float v49_data = r0[0];
          float v50_data = r0[1];
          float v51_data = r0[2];
          float v52_data = r0[3];
          float v53_data = r0[4];
          float v54_data = r0[5];
          float v55_data = r0[6];
          float v56_data = r0[7];
          float v57_data = r0[8];
          float v58_data = r0[9];
          float v59_data = r0[10];
          float v60_data = r0[11];
          float v61_data = r0[12];
          float v62_data = r0[13];
          float v63_data = r0[14];
          float v64_data = r0[15];
          float v65_acc{};
          float v66_acc{};
          float v67_acc{};
          float v68_acc{};
          float v69_acc{};
          float v70_acc{};
          float v71_acc{};
          float v72_acc{};
          float v73_lin = r1[0];
          float v74_bc = tensorforge::broadcast<32, 16, 0>(v73_lin);
          tensorforge::fmacdpp16<0>(v65_acc, v74_bc, v49_data);
          tensorforge::fmacdpp16<0>(v66_acc, v74_bc, v50_data);
          tensorforge::fmacdpp16<1>(v65_acc, v74_bc, v51_data);
          tensorforge::fmacdpp16<1>(v66_acc, v74_bc, v52_data);
          tensorforge::fmacdpp16<2>(v65_acc, v74_bc, v53_data);
          tensorforge::fmacdpp16<2>(v66_acc, v74_bc, v54_data);
          tensorforge::fmacdpp16<3>(v65_acc, v74_bc, v55_data);
          tensorforge::fmacdpp16<3>(v66_acc, v74_bc, v56_data);
          tensorforge::fmacdpp16<4>(v65_acc, v74_bc, v57_data);
          tensorforge::fmacdpp16<4>(v66_acc, v74_bc, v58_data);
          tensorforge::fmacdpp16<5>(v65_acc, v74_bc, v59_data);
          tensorforge::fmacdpp16<5>(v66_acc, v74_bc, v60_data);
          tensorforge::fmacdpp16<6>(v65_acc, v74_bc, v61_data);
          tensorforge::fmacdpp16<6>(v66_acc, v74_bc, v62_data);
          tensorforge::fmacdpp16<7>(v65_acc, v74_bc, v63_data);
          tensorforge::fmacdpp16<7>(v66_acc, v74_bc, v64_data);
          tensorforge::fmacdpp16<8>(v67_acc, v74_bc, v49_data);
          tensorforge::fmacdpp16<8>(v68_acc, v74_bc, v50_data);
          tensorforge::fmacdpp16<9>(v67_acc, v74_bc, v51_data);
          tensorforge::fmacdpp16<9>(v68_acc, v74_bc, v52_data);
          tensorforge::fmacdpp16<10>(v67_acc, v74_bc, v53_data);
          tensorforge::fmacdpp16<10>(v68_acc, v74_bc, v54_data);
          tensorforge::fmacdpp16<11>(v67_acc, v74_bc, v55_data);
          tensorforge::fmacdpp16<11>(v68_acc, v74_bc, v56_data);
          tensorforge::fmacdpp16<12>(v67_acc, v74_bc, v57_data);
          tensorforge::fmacdpp16<12>(v68_acc, v74_bc, v58_data);
          tensorforge::fmacdpp16<13>(v67_acc, v74_bc, v59_data);
          tensorforge::fmacdpp16<13>(v68_acc, v74_bc, v60_data);
          tensorforge::fmacdpp16<14>(v67_acc, v74_bc, v61_data);
          tensorforge::fmacdpp16<14>(v68_acc, v74_bc, v62_data);
          tensorforge::fmacdpp16<15>(v67_acc, v74_bc, v63_data);
          tensorforge::fmacdpp16<15>(v68_acc, v74_bc, v64_data);
          float v75_bc = tensorforge::broadcast<32, 16, 1>(v73_lin);
          tensorforge::fmacdpp16<0>(v69_acc, v75_bc, v49_data);
          tensorforge::fmacdpp16<0>(v70_acc, v75_bc, v50_data);
          tensorforge::fmacdpp16<1>(v69_acc, v75_bc, v51_data);
          tensorforge::fmacdpp16<1>(v70_acc, v75_bc, v52_data);
          tensorforge::fmacdpp16<2>(v69_acc, v75_bc, v53_data);
          tensorforge::fmacdpp16<2>(v70_acc, v75_bc, v54_data);
          tensorforge::fmacdpp16<3>(v69_acc, v75_bc, v55_data);
          tensorforge::fmacdpp16<3>(v70_acc, v75_bc, v56_data);
          tensorforge::fmacdpp16<4>(v69_acc, v75_bc, v57_data);
          tensorforge::fmacdpp16<4>(v70_acc, v75_bc, v58_data);
          tensorforge::fmacdpp16<5>(v69_acc, v75_bc, v59_data);
          tensorforge::fmacdpp16<5>(v70_acc, v75_bc, v60_data);
          tensorforge::fmacdpp16<6>(v69_acc, v75_bc, v61_data);
          tensorforge::fmacdpp16<6>(v70_acc, v75_bc, v62_data);
          tensorforge::fmacdpp16<7>(v69_acc, v75_bc, v63_data);
          tensorforge::fmacdpp16<7>(v70_acc, v75_bc, v64_data);
          tensorforge::fmacdpp16<8>(v71_acc, v75_bc, v49_data);
          tensorforge::fmacdpp16<8>(v72_acc, v75_bc, v50_data);
          tensorforge::fmacdpp16<9>(v71_acc, v75_bc, v51_data);
          tensorforge::fmacdpp16<9>(v72_acc, v75_bc, v52_data);
          tensorforge::fmacdpp16<10>(v71_acc, v75_bc, v53_data);
          tensorforge::fmacdpp16<10>(v72_acc, v75_bc, v54_data);
          tensorforge::fmacdpp16<11>(v71_acc, v75_bc, v55_data);
          tensorforge::fmacdpp16<11>(v72_acc, v75_bc, v56_data);
          tensorforge::fmacdpp16<12>(v71_acc, v75_bc, v57_data);
          tensorforge::fmacdpp16<12>(v72_acc, v75_bc, v58_data);
          tensorforge::fmacdpp16<13>(v71_acc, v75_bc, v59_data);
          tensorforge::fmacdpp16<13>(v72_acc, v75_bc, v60_data);
          tensorforge::fmacdpp16<14>(v71_acc, v75_bc, v61_data);
          tensorforge::fmacdpp16<14>(v72_acc, v75_bc, v62_data);
          tensorforge::fmacdpp16<15>(v71_acc, v75_bc, v63_data);
          tensorforge::fmacdpp16<15>(v72_acc, v75_bc, v64_data);
          r2[0] = v65_acc;
          r2[1] = v66_acc;
          r2[2] = v67_acc;
          r2[3] = v68_acc;
          r2[4] = v69_acc;
          r2[5] = v70_acc;
          r2[6] = v71_acc;
          r2[7] = v72_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v79_i0 = 0; v79_i0 < 1; ++v79_i0) {
            int32_t v90_lead = v7_lead + (v79_i0 * 32);
            #pragma unroll
            for (int32_t v80_i1 = 0; v80_i1 < 4; ++v80_i1) {
              int32_t v81_a = v80_i1 * 2;
              int32_t v82_a = v79_i0 + v81_a;
              float v85_data = r2[(v79_i0 + v81_a)];
              glb_m0[(v90_lead + (v80_i1 * 35))] = v85_data;
            }
          }
          if (v7_lead < 3) {
            int32_t v104_lead = v7_lead + 32_i32;
            #pragma unroll
            for (int32_t v94_i1 = 0; v94_i1 < 4; ++v94_i1) {
              int32_t v95_a = v94_i1 * 2;
              int32_t v96_a = 1 + v95_a;
              float v99_data = r2[(1 + v95_a)];
              glb_m0[(v104_lead + (v94_i1 * 35))] = v99_data;
            }
          }
        }
      }
    }
  }
}

