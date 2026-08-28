// === base name ===
kernel_3d37ccf0b0

// === header ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3d37ccf0b0, block.x * block.y * block.z, 256 * sizeof(double)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_3d37ccf0b0), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(double)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_3d37ccf0b0, grid, block, 256 * sizeof(double), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v16_lead = v11_i0 * 16;
            int32_t v18_off = (v10_lead + v16_lead) + 8;
            int32_t v26_off = (v10_lead + v16_lead) + 8;
            #pragma unroll
            for (int32_t v12_i1 = 8; v12_i1 < 24; ++v12_i1) {
              int32_t v19_a = v12_i1 * 32;
              int32_t v20_a = v18_off + v19_a;
              double v29_data = __builtin_nontemporal_load(&glb_m1[(v26_off + v19_a)]);
              r0[(v11_i0 + (v12_i1 - 8))] = v29_data;
            }
          }
          double r1[8]{};
          // r1 = load{g>r}(glb_m2);
          double v33_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v33_lin;
          double v34_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v34_lin;
          double v35_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v35_lin;
          double v36_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v36_lin;
          double v37_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v37_lin;
          double v38_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v38_lin;
          double v39_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v39_lin;
          double v40_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v40_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          double v42_data = r0[0];
          double v43_data = r0[1];
          double v44_data = r0[2];
          double v45_data = r0[3];
          double v46_data = r0[4];
          double v47_data = r0[5];
          double v48_data = r0[6];
          double v49_data = r0[7];
          double v50_data = r0[8];
          double v51_data = r0[9];
          double v52_data = r0[10];
          double v53_data = r0[11];
          double v54_data = r0[12];
          double v55_data = r0[13];
          double v56_data = r0[14];
          double v57_data = r0[15];
          double v58_acc{};
          double v59_acc{};
          double v60_acc{};
          double v61_acc{};
          double v62_acc{};
          double v63_acc{};
          double v64_acc{};
          double v65_acc{};
          double v66_data = r1[0];
          double v67_data = r1[1];
          double v68_data = r1[2];
          double v69_data = r1[3];
          double v70_data = r1[4];
          double v71_data = r1[5];
          double v72_data = r1[6];
          double v73_data = r1[7];
          tensorforge::fmacdpp16<0>(v58_acc, v66_data, v42_data);
          tensorforge::fmacdpp16<1>(v58_acc, v66_data, v43_data);
          tensorforge::fmacdpp16<2>(v58_acc, v66_data, v44_data);
          tensorforge::fmacdpp16<3>(v58_acc, v66_data, v45_data);
          tensorforge::fmacdpp16<4>(v58_acc, v66_data, v46_data);
          tensorforge::fmacdpp16<5>(v58_acc, v66_data, v47_data);
          tensorforge::fmacdpp16<6>(v58_acc, v66_data, v48_data);
          tensorforge::fmacdpp16<7>(v58_acc, v66_data, v49_data);
          tensorforge::fmacdpp16<8>(v58_acc, v66_data, v50_data);
          tensorforge::fmacdpp16<9>(v58_acc, v66_data, v51_data);
          tensorforge::fmacdpp16<10>(v58_acc, v66_data, v52_data);
          tensorforge::fmacdpp16<11>(v58_acc, v66_data, v53_data);
          tensorforge::fmacdpp16<12>(v58_acc, v66_data, v54_data);
          tensorforge::fmacdpp16<13>(v58_acc, v66_data, v55_data);
          tensorforge::fmacdpp16<14>(v58_acc, v66_data, v56_data);
          tensorforge::fmacdpp16<15>(v58_acc, v66_data, v57_data);
          tensorforge::fmacdpp16<0>(v59_acc, v67_data, v42_data);
          tensorforge::fmacdpp16<1>(v59_acc, v67_data, v43_data);
          tensorforge::fmacdpp16<2>(v59_acc, v67_data, v44_data);
          tensorforge::fmacdpp16<3>(v59_acc, v67_data, v45_data);
          tensorforge::fmacdpp16<4>(v59_acc, v67_data, v46_data);
          tensorforge::fmacdpp16<5>(v59_acc, v67_data, v47_data);
          tensorforge::fmacdpp16<6>(v59_acc, v67_data, v48_data);
          tensorforge::fmacdpp16<7>(v59_acc, v67_data, v49_data);
          tensorforge::fmacdpp16<8>(v59_acc, v67_data, v50_data);
          tensorforge::fmacdpp16<9>(v59_acc, v67_data, v51_data);
          tensorforge::fmacdpp16<10>(v59_acc, v67_data, v52_data);
          tensorforge::fmacdpp16<11>(v59_acc, v67_data, v53_data);
          tensorforge::fmacdpp16<12>(v59_acc, v67_data, v54_data);
          tensorforge::fmacdpp16<13>(v59_acc, v67_data, v55_data);
          tensorforge::fmacdpp16<14>(v59_acc, v67_data, v56_data);
          tensorforge::fmacdpp16<15>(v59_acc, v67_data, v57_data);
          tensorforge::fmacdpp16<0>(v60_acc, v68_data, v42_data);
          tensorforge::fmacdpp16<1>(v60_acc, v68_data, v43_data);
          tensorforge::fmacdpp16<2>(v60_acc, v68_data, v44_data);
          tensorforge::fmacdpp16<3>(v60_acc, v68_data, v45_data);
          tensorforge::fmacdpp16<4>(v60_acc, v68_data, v46_data);
          tensorforge::fmacdpp16<5>(v60_acc, v68_data, v47_data);
          tensorforge::fmacdpp16<6>(v60_acc, v68_data, v48_data);
          tensorforge::fmacdpp16<7>(v60_acc, v68_data, v49_data);
          tensorforge::fmacdpp16<8>(v60_acc, v68_data, v50_data);
          tensorforge::fmacdpp16<9>(v60_acc, v68_data, v51_data);
          tensorforge::fmacdpp16<10>(v60_acc, v68_data, v52_data);
          tensorforge::fmacdpp16<11>(v60_acc, v68_data, v53_data);
          tensorforge::fmacdpp16<12>(v60_acc, v68_data, v54_data);
          tensorforge::fmacdpp16<13>(v60_acc, v68_data, v55_data);
          tensorforge::fmacdpp16<14>(v60_acc, v68_data, v56_data);
          tensorforge::fmacdpp16<15>(v60_acc, v68_data, v57_data);
          tensorforge::fmacdpp16<0>(v61_acc, v69_data, v42_data);
          tensorforge::fmacdpp16<1>(v61_acc, v69_data, v43_data);
          tensorforge::fmacdpp16<2>(v61_acc, v69_data, v44_data);
          tensorforge::fmacdpp16<3>(v61_acc, v69_data, v45_data);
          tensorforge::fmacdpp16<4>(v61_acc, v69_data, v46_data);
          tensorforge::fmacdpp16<5>(v61_acc, v69_data, v47_data);
          tensorforge::fmacdpp16<6>(v61_acc, v69_data, v48_data);
          tensorforge::fmacdpp16<7>(v61_acc, v69_data, v49_data);
          tensorforge::fmacdpp16<8>(v61_acc, v69_data, v50_data);
          tensorforge::fmacdpp16<9>(v61_acc, v69_data, v51_data);
          tensorforge::fmacdpp16<10>(v61_acc, v69_data, v52_data);
          tensorforge::fmacdpp16<11>(v61_acc, v69_data, v53_data);
          tensorforge::fmacdpp16<12>(v61_acc, v69_data, v54_data);
          tensorforge::fmacdpp16<13>(v61_acc, v69_data, v55_data);
          tensorforge::fmacdpp16<14>(v61_acc, v69_data, v56_data);
          tensorforge::fmacdpp16<15>(v61_acc, v69_data, v57_data);
          tensorforge::fmacdpp16<0>(v62_acc, v70_data, v42_data);
          tensorforge::fmacdpp16<1>(v62_acc, v70_data, v43_data);
          tensorforge::fmacdpp16<2>(v62_acc, v70_data, v44_data);
          tensorforge::fmacdpp16<3>(v62_acc, v70_data, v45_data);
          tensorforge::fmacdpp16<4>(v62_acc, v70_data, v46_data);
          tensorforge::fmacdpp16<5>(v62_acc, v70_data, v47_data);
          tensorforge::fmacdpp16<6>(v62_acc, v70_data, v48_data);
          tensorforge::fmacdpp16<7>(v62_acc, v70_data, v49_data);
          tensorforge::fmacdpp16<8>(v62_acc, v70_data, v50_data);
          tensorforge::fmacdpp16<9>(v62_acc, v70_data, v51_data);
          tensorforge::fmacdpp16<10>(v62_acc, v70_data, v52_data);
          tensorforge::fmacdpp16<11>(v62_acc, v70_data, v53_data);
          tensorforge::fmacdpp16<12>(v62_acc, v70_data, v54_data);
          tensorforge::fmacdpp16<13>(v62_acc, v70_data, v55_data);
          tensorforge::fmacdpp16<14>(v62_acc, v70_data, v56_data);
          tensorforge::fmacdpp16<15>(v62_acc, v70_data, v57_data);
          tensorforge::fmacdpp16<0>(v63_acc, v71_data, v42_data);
          tensorforge::fmacdpp16<1>(v63_acc, v71_data, v43_data);
          tensorforge::fmacdpp16<2>(v63_acc, v71_data, v44_data);
          tensorforge::fmacdpp16<3>(v63_acc, v71_data, v45_data);
          tensorforge::fmacdpp16<4>(v63_acc, v71_data, v46_data);
          tensorforge::fmacdpp16<5>(v63_acc, v71_data, v47_data);
          tensorforge::fmacdpp16<6>(v63_acc, v71_data, v48_data);
          tensorforge::fmacdpp16<7>(v63_acc, v71_data, v49_data);
          tensorforge::fmacdpp16<8>(v63_acc, v71_data, v50_data);
          tensorforge::fmacdpp16<9>(v63_acc, v71_data, v51_data);
          tensorforge::fmacdpp16<10>(v63_acc, v71_data, v52_data);
          tensorforge::fmacdpp16<11>(v63_acc, v71_data, v53_data);
          tensorforge::fmacdpp16<12>(v63_acc, v71_data, v54_data);
          tensorforge::fmacdpp16<13>(v63_acc, v71_data, v55_data);
          tensorforge::fmacdpp16<14>(v63_acc, v71_data, v56_data);
          tensorforge::fmacdpp16<15>(v63_acc, v71_data, v57_data);
          tensorforge::fmacdpp16<0>(v64_acc, v72_data, v42_data);
          tensorforge::fmacdpp16<1>(v64_acc, v72_data, v43_data);
          tensorforge::fmacdpp16<2>(v64_acc, v72_data, v44_data);
          tensorforge::fmacdpp16<3>(v64_acc, v72_data, v45_data);
          tensorforge::fmacdpp16<4>(v64_acc, v72_data, v46_data);
          tensorforge::fmacdpp16<5>(v64_acc, v72_data, v47_data);
          tensorforge::fmacdpp16<6>(v64_acc, v72_data, v48_data);
          tensorforge::fmacdpp16<7>(v64_acc, v72_data, v49_data);
          tensorforge::fmacdpp16<8>(v64_acc, v72_data, v50_data);
          tensorforge::fmacdpp16<9>(v64_acc, v72_data, v51_data);
          tensorforge::fmacdpp16<10>(v64_acc, v72_data, v52_data);
          tensorforge::fmacdpp16<11>(v64_acc, v72_data, v53_data);
          tensorforge::fmacdpp16<12>(v64_acc, v72_data, v54_data);
          tensorforge::fmacdpp16<13>(v64_acc, v72_data, v55_data);
          tensorforge::fmacdpp16<14>(v64_acc, v72_data, v56_data);
          tensorforge::fmacdpp16<15>(v64_acc, v72_data, v57_data);
          tensorforge::fmacdpp16<0>(v65_acc, v73_data, v42_data);
          tensorforge::fmacdpp16<1>(v65_acc, v73_data, v43_data);
          tensorforge::fmacdpp16<2>(v65_acc, v73_data, v44_data);
          tensorforge::fmacdpp16<3>(v65_acc, v73_data, v45_data);
          tensorforge::fmacdpp16<4>(v65_acc, v73_data, v46_data);
          tensorforge::fmacdpp16<5>(v65_acc, v73_data, v47_data);
          tensorforge::fmacdpp16<6>(v65_acc, v73_data, v48_data);
          tensorforge::fmacdpp16<7>(v65_acc, v73_data, v49_data);
          tensorforge::fmacdpp16<8>(v65_acc, v73_data, v50_data);
          tensorforge::fmacdpp16<9>(v65_acc, v73_data, v51_data);
          tensorforge::fmacdpp16<10>(v65_acc, v73_data, v52_data);
          tensorforge::fmacdpp16<11>(v65_acc, v73_data, v53_data);
          tensorforge::fmacdpp16<12>(v65_acc, v73_data, v54_data);
          tensorforge::fmacdpp16<13>(v65_acc, v73_data, v55_data);
          tensorforge::fmacdpp16<14>(v65_acc, v73_data, v56_data);
          tensorforge::fmacdpp16<15>(v65_acc, v73_data, v57_data);
          r2[0] = v58_acc;
          r2[1] = v59_acc;
          r2[2] = v60_acc;
          r2[3] = v61_acc;
          r2[4] = v62_acc;
          r2[5] = v63_acc;
          r2[6] = v64_acc;
          r2[7] = v65_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v77_i0 = 0; v77_i0 < 1; ++v77_i0) {
            int32_t v86_lead = v10_lead + (v77_i0 * 16);
            #pragma unroll
            for (int32_t v78_i1 = 0; v78_i1 < 8; ++v78_i1) {
              int32_t v79_a = v77_i0 + v78_i1;
              double v81_data = r2[(v77_i0 + v78_i1)];
              glb_m0[(v86_lead + (v78_i1 * 16))] = v81_data;
            }
          }
        }
      }
    }
  }
}

