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
            int32_t v16_lead = v11_i0 * 32;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
              int32_t v18_a = v12_i1 * 35;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + (v12_i1 * 2))] = v27_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v36_lead = v10_lead + 32_i32;
            int32_t v43_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
              int32_t v37_a = v31_i1 * 35;
              int32_t v38_a = v36_lead + v37_a;
              float v46_data = __builtin_nontemporal_load(&glb_m1[(v43_lead + v37_a)]);
              r0[(1 + (v31_i1 * 2))] = v46_data;
            }
          }
          float r1[4]{};
          // r1 = load{g>r}(glb_m2);
          float v50_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v50_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 35), (0, 4)] [(0, 8)]
          float v52_data = r0[0];
          float v53_data = r0[1];
          float v54_data = r0[2];
          float v55_data = r0[3];
          float v56_data = r0[4];
          float v57_data = r0[5];
          float v58_data = r0[6];
          float v59_data = r0[7];
          float v60_data = r0[8];
          float v61_data = r0[9];
          float v62_data = r0[10];
          float v63_data = r0[11];
          float v64_data = r0[12];
          float v65_data = r0[13];
          float v66_data = r0[14];
          float v67_data = r0[15];
          float v68_acc{};
          float v69_acc{};
          float v70_acc{};
          float v71_acc{};
          float v72_acc{};
          float v73_acc{};
          float v74_acc{};
          float v75_acc{};
          float v76_lin = r1[0];
          float v77_bc = tensorforge::broadcast<32, 16, 0>(v76_lin);
          tensorforge::fmacdpp16<0>(v68_acc, v77_bc, v52_data);
          tensorforge::fmacdpp16<0>(v69_acc, v77_bc, v53_data);
          tensorforge::fmacdpp16<1>(v68_acc, v77_bc, v54_data);
          tensorforge::fmacdpp16<1>(v69_acc, v77_bc, v55_data);
          tensorforge::fmacdpp16<2>(v68_acc, v77_bc, v56_data);
          tensorforge::fmacdpp16<2>(v69_acc, v77_bc, v57_data);
          tensorforge::fmacdpp16<3>(v68_acc, v77_bc, v58_data);
          tensorforge::fmacdpp16<3>(v69_acc, v77_bc, v59_data);
          tensorforge::fmacdpp16<4>(v68_acc, v77_bc, v60_data);
          tensorforge::fmacdpp16<4>(v69_acc, v77_bc, v61_data);
          tensorforge::fmacdpp16<5>(v68_acc, v77_bc, v62_data);
          tensorforge::fmacdpp16<5>(v69_acc, v77_bc, v63_data);
          tensorforge::fmacdpp16<6>(v68_acc, v77_bc, v64_data);
          tensorforge::fmacdpp16<6>(v69_acc, v77_bc, v65_data);
          tensorforge::fmacdpp16<7>(v68_acc, v77_bc, v66_data);
          tensorforge::fmacdpp16<7>(v69_acc, v77_bc, v67_data);
          tensorforge::fmacdpp16<8>(v70_acc, v77_bc, v52_data);
          tensorforge::fmacdpp16<8>(v71_acc, v77_bc, v53_data);
          tensorforge::fmacdpp16<9>(v70_acc, v77_bc, v54_data);
          tensorforge::fmacdpp16<9>(v71_acc, v77_bc, v55_data);
          tensorforge::fmacdpp16<10>(v70_acc, v77_bc, v56_data);
          tensorforge::fmacdpp16<10>(v71_acc, v77_bc, v57_data);
          tensorforge::fmacdpp16<11>(v70_acc, v77_bc, v58_data);
          tensorforge::fmacdpp16<11>(v71_acc, v77_bc, v59_data);
          tensorforge::fmacdpp16<12>(v70_acc, v77_bc, v60_data);
          tensorforge::fmacdpp16<12>(v71_acc, v77_bc, v61_data);
          tensorforge::fmacdpp16<13>(v70_acc, v77_bc, v62_data);
          tensorforge::fmacdpp16<13>(v71_acc, v77_bc, v63_data);
          tensorforge::fmacdpp16<14>(v70_acc, v77_bc, v64_data);
          tensorforge::fmacdpp16<14>(v71_acc, v77_bc, v65_data);
          tensorforge::fmacdpp16<15>(v70_acc, v77_bc, v66_data);
          tensorforge::fmacdpp16<15>(v71_acc, v77_bc, v67_data);
          float v78_bc = tensorforge::broadcast<32, 16, 1>(v76_lin);
          tensorforge::fmacdpp16<0>(v72_acc, v78_bc, v52_data);
          tensorforge::fmacdpp16<0>(v73_acc, v78_bc, v53_data);
          tensorforge::fmacdpp16<1>(v72_acc, v78_bc, v54_data);
          tensorforge::fmacdpp16<1>(v73_acc, v78_bc, v55_data);
          tensorforge::fmacdpp16<2>(v72_acc, v78_bc, v56_data);
          tensorforge::fmacdpp16<2>(v73_acc, v78_bc, v57_data);
          tensorforge::fmacdpp16<3>(v72_acc, v78_bc, v58_data);
          tensorforge::fmacdpp16<3>(v73_acc, v78_bc, v59_data);
          tensorforge::fmacdpp16<4>(v72_acc, v78_bc, v60_data);
          tensorforge::fmacdpp16<4>(v73_acc, v78_bc, v61_data);
          tensorforge::fmacdpp16<5>(v72_acc, v78_bc, v62_data);
          tensorforge::fmacdpp16<5>(v73_acc, v78_bc, v63_data);
          tensorforge::fmacdpp16<6>(v72_acc, v78_bc, v64_data);
          tensorforge::fmacdpp16<6>(v73_acc, v78_bc, v65_data);
          tensorforge::fmacdpp16<7>(v72_acc, v78_bc, v66_data);
          tensorforge::fmacdpp16<7>(v73_acc, v78_bc, v67_data);
          tensorforge::fmacdpp16<8>(v74_acc, v78_bc, v52_data);
          tensorforge::fmacdpp16<8>(v75_acc, v78_bc, v53_data);
          tensorforge::fmacdpp16<9>(v74_acc, v78_bc, v54_data);
          tensorforge::fmacdpp16<9>(v75_acc, v78_bc, v55_data);
          tensorforge::fmacdpp16<10>(v74_acc, v78_bc, v56_data);
          tensorforge::fmacdpp16<10>(v75_acc, v78_bc, v57_data);
          tensorforge::fmacdpp16<11>(v74_acc, v78_bc, v58_data);
          tensorforge::fmacdpp16<11>(v75_acc, v78_bc, v59_data);
          tensorforge::fmacdpp16<12>(v74_acc, v78_bc, v60_data);
          tensorforge::fmacdpp16<12>(v75_acc, v78_bc, v61_data);
          tensorforge::fmacdpp16<13>(v74_acc, v78_bc, v62_data);
          tensorforge::fmacdpp16<13>(v75_acc, v78_bc, v63_data);
          tensorforge::fmacdpp16<14>(v74_acc, v78_bc, v64_data);
          tensorforge::fmacdpp16<14>(v75_acc, v78_bc, v65_data);
          tensorforge::fmacdpp16<15>(v74_acc, v78_bc, v66_data);
          tensorforge::fmacdpp16<15>(v75_acc, v78_bc, v67_data);
          r2[0] = v68_acc;
          r2[1] = v69_acc;
          r2[2] = v70_acc;
          r2[3] = v71_acc;
          r2[4] = v72_acc;
          r2[5] = v73_acc;
          r2[6] = v74_acc;
          r2[7] = v75_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v82_i0 = 0; v82_i0 < 1; ++v82_i0) {
            int32_t v93_lead = v10_lead + (v82_i0 * 32);
            #pragma unroll
            for (int32_t v83_i1 = 0; v83_i1 < 4; ++v83_i1) {
              int32_t v84_a = v83_i1 * 2;
              int32_t v85_a = v82_i0 + v84_a;
              float v88_data = r2[(v82_i0 + v84_a)];
              glb_m0[(v93_lead + (v83_i1 * 35))] = v88_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v107_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v97_i1 = 0; v97_i1 < 4; ++v97_i1) {
              int32_t v98_a = v97_i1 * 2;
              int32_t v99_a = 1 + v98_a;
              float v102_data = r2[(1 + v98_a)];
              glb_m0[(v107_lead + (v97_i1 * 35))] = v102_data;
            }
          }
        }
      }
    }
  }
}

