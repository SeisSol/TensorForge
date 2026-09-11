// === base name ===
kernel_2972256972cd561c

// === header ===
void launcher_kernel_2972256972cd561c(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_2972256972cd561c(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_2972256972cd561c, block.x * block.y * block.z, 256 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_2972256972cd561c, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(double)));
          blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
        }
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_2972256972cd561c), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(double)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_2972256972cd561c, grid, block, 256 * sizeof(double), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_2972256972cd561c(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t v3_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v3_batchId0 < numElements0; v3_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v4_ahead1 = v3_batchId0 + (gridDim.x * blockDim.y);
        size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[v3_batchId0 * 128 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[v3_batchId0 * 1024 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[v3_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
            int32_t v25_off = (v17_lead + (v18_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v19_i1 = 8; v19_i1 < 24; ++v19_i1) {
              double v28_data = __builtin_nontemporal_load(&glb_m1[(v25_off + (v19_i1 * 32))]);
              r0[(v18_i0 + (v19_i1 - 8))] = v28_data;
            }
          }
          double r1[8]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
            int32_t v41_lead = v17_lead + (v35_i0 * 16);
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
              double v44_data = __builtin_nontemporal_load(&glb_m2[(v41_lead + (v36_i1 * 16))]);
              r1[(v35_i0 + v36_i1)] = v44_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          double r2[8]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          double v47_data = r0[0];
          double v48_data = r0[1];
          double v49_data = r0[2];
          double v50_data = r0[3];
          double v51_data = r0[4];
          double v52_data = r0[5];
          double v53_data = r0[6];
          double v54_data = r0[7];
          double v55_data = r0[8];
          double v56_data = r0[9];
          double v57_data = r0[10];
          double v58_data = r0[11];
          double v59_data = r0[12];
          double v60_data = r0[13];
          double v61_data = r0[14];
          double v62_data = r0[15];
          double v63_acc{};
          double v64_acc{};
          double v65_acc{};
          double v66_acc{};
          double v67_acc{};
          double v68_acc{};
          double v69_acc{};
          double v70_acc{};
          double v71_data = r1[0];
          double v72_data = r1[1];
          double v73_data = r1[2];
          double v74_data = r1[3];
          double v75_data = r1[4];
          double v76_data = r1[5];
          double v77_data = r1[6];
          double v78_data = r1[7];
          tensorforge::fmacdpp16<0>(v63_acc, v71_data, v47_data);
          tensorforge::fmacdpp16<1>(v63_acc, v71_data, v48_data);
          tensorforge::fmacdpp16<2>(v63_acc, v71_data, v49_data);
          tensorforge::fmacdpp16<3>(v63_acc, v71_data, v50_data);
          tensorforge::fmacdpp16<4>(v63_acc, v71_data, v51_data);
          tensorforge::fmacdpp16<5>(v63_acc, v71_data, v52_data);
          tensorforge::fmacdpp16<6>(v63_acc, v71_data, v53_data);
          tensorforge::fmacdpp16<7>(v63_acc, v71_data, v54_data);
          tensorforge::fmacdpp16<8>(v63_acc, v71_data, v55_data);
          tensorforge::fmacdpp16<9>(v63_acc, v71_data, v56_data);
          tensorforge::fmacdpp16<10>(v63_acc, v71_data, v57_data);
          tensorforge::fmacdpp16<11>(v63_acc, v71_data, v58_data);
          tensorforge::fmacdpp16<12>(v63_acc, v71_data, v59_data);
          tensorforge::fmacdpp16<13>(v63_acc, v71_data, v60_data);
          tensorforge::fmacdpp16<14>(v63_acc, v71_data, v61_data);
          tensorforge::fmacdpp16<15>(v63_acc, v71_data, v62_data);
          tensorforge::fmacdpp16<0>(v64_acc, v72_data, v47_data);
          tensorforge::fmacdpp16<1>(v64_acc, v72_data, v48_data);
          tensorforge::fmacdpp16<2>(v64_acc, v72_data, v49_data);
          tensorforge::fmacdpp16<3>(v64_acc, v72_data, v50_data);
          tensorforge::fmacdpp16<4>(v64_acc, v72_data, v51_data);
          tensorforge::fmacdpp16<5>(v64_acc, v72_data, v52_data);
          tensorforge::fmacdpp16<6>(v64_acc, v72_data, v53_data);
          tensorforge::fmacdpp16<7>(v64_acc, v72_data, v54_data);
          tensorforge::fmacdpp16<8>(v64_acc, v72_data, v55_data);
          tensorforge::fmacdpp16<9>(v64_acc, v72_data, v56_data);
          tensorforge::fmacdpp16<10>(v64_acc, v72_data, v57_data);
          tensorforge::fmacdpp16<11>(v64_acc, v72_data, v58_data);
          tensorforge::fmacdpp16<12>(v64_acc, v72_data, v59_data);
          tensorforge::fmacdpp16<13>(v64_acc, v72_data, v60_data);
          tensorforge::fmacdpp16<14>(v64_acc, v72_data, v61_data);
          tensorforge::fmacdpp16<15>(v64_acc, v72_data, v62_data);
          tensorforge::fmacdpp16<0>(v65_acc, v73_data, v47_data);
          tensorforge::fmacdpp16<1>(v65_acc, v73_data, v48_data);
          tensorforge::fmacdpp16<2>(v65_acc, v73_data, v49_data);
          tensorforge::fmacdpp16<3>(v65_acc, v73_data, v50_data);
          tensorforge::fmacdpp16<4>(v65_acc, v73_data, v51_data);
          tensorforge::fmacdpp16<5>(v65_acc, v73_data, v52_data);
          tensorforge::fmacdpp16<6>(v65_acc, v73_data, v53_data);
          tensorforge::fmacdpp16<7>(v65_acc, v73_data, v54_data);
          tensorforge::fmacdpp16<8>(v65_acc, v73_data, v55_data);
          tensorforge::fmacdpp16<9>(v65_acc, v73_data, v56_data);
          tensorforge::fmacdpp16<10>(v65_acc, v73_data, v57_data);
          tensorforge::fmacdpp16<11>(v65_acc, v73_data, v58_data);
          tensorforge::fmacdpp16<12>(v65_acc, v73_data, v59_data);
          tensorforge::fmacdpp16<13>(v65_acc, v73_data, v60_data);
          tensorforge::fmacdpp16<14>(v65_acc, v73_data, v61_data);
          tensorforge::fmacdpp16<15>(v65_acc, v73_data, v62_data);
          tensorforge::fmacdpp16<0>(v66_acc, v74_data, v47_data);
          tensorforge::fmacdpp16<1>(v66_acc, v74_data, v48_data);
          tensorforge::fmacdpp16<2>(v66_acc, v74_data, v49_data);
          tensorforge::fmacdpp16<3>(v66_acc, v74_data, v50_data);
          tensorforge::fmacdpp16<4>(v66_acc, v74_data, v51_data);
          tensorforge::fmacdpp16<5>(v66_acc, v74_data, v52_data);
          tensorforge::fmacdpp16<6>(v66_acc, v74_data, v53_data);
          tensorforge::fmacdpp16<7>(v66_acc, v74_data, v54_data);
          tensorforge::fmacdpp16<8>(v66_acc, v74_data, v55_data);
          tensorforge::fmacdpp16<9>(v66_acc, v74_data, v56_data);
          tensorforge::fmacdpp16<10>(v66_acc, v74_data, v57_data);
          tensorforge::fmacdpp16<11>(v66_acc, v74_data, v58_data);
          tensorforge::fmacdpp16<12>(v66_acc, v74_data, v59_data);
          tensorforge::fmacdpp16<13>(v66_acc, v74_data, v60_data);
          tensorforge::fmacdpp16<14>(v66_acc, v74_data, v61_data);
          tensorforge::fmacdpp16<15>(v66_acc, v74_data, v62_data);
          tensorforge::fmacdpp16<0>(v67_acc, v75_data, v47_data);
          tensorforge::fmacdpp16<1>(v67_acc, v75_data, v48_data);
          tensorforge::fmacdpp16<2>(v67_acc, v75_data, v49_data);
          tensorforge::fmacdpp16<3>(v67_acc, v75_data, v50_data);
          tensorforge::fmacdpp16<4>(v67_acc, v75_data, v51_data);
          tensorforge::fmacdpp16<5>(v67_acc, v75_data, v52_data);
          tensorforge::fmacdpp16<6>(v67_acc, v75_data, v53_data);
          tensorforge::fmacdpp16<7>(v67_acc, v75_data, v54_data);
          tensorforge::fmacdpp16<8>(v67_acc, v75_data, v55_data);
          tensorforge::fmacdpp16<9>(v67_acc, v75_data, v56_data);
          tensorforge::fmacdpp16<10>(v67_acc, v75_data, v57_data);
          tensorforge::fmacdpp16<11>(v67_acc, v75_data, v58_data);
          tensorforge::fmacdpp16<12>(v67_acc, v75_data, v59_data);
          tensorforge::fmacdpp16<13>(v67_acc, v75_data, v60_data);
          tensorforge::fmacdpp16<14>(v67_acc, v75_data, v61_data);
          tensorforge::fmacdpp16<15>(v67_acc, v75_data, v62_data);
          tensorforge::fmacdpp16<0>(v68_acc, v76_data, v47_data);
          tensorforge::fmacdpp16<1>(v68_acc, v76_data, v48_data);
          tensorforge::fmacdpp16<2>(v68_acc, v76_data, v49_data);
          tensorforge::fmacdpp16<3>(v68_acc, v76_data, v50_data);
          tensorforge::fmacdpp16<4>(v68_acc, v76_data, v51_data);
          tensorforge::fmacdpp16<5>(v68_acc, v76_data, v52_data);
          tensorforge::fmacdpp16<6>(v68_acc, v76_data, v53_data);
          tensorforge::fmacdpp16<7>(v68_acc, v76_data, v54_data);
          tensorforge::fmacdpp16<8>(v68_acc, v76_data, v55_data);
          tensorforge::fmacdpp16<9>(v68_acc, v76_data, v56_data);
          tensorforge::fmacdpp16<10>(v68_acc, v76_data, v57_data);
          tensorforge::fmacdpp16<11>(v68_acc, v76_data, v58_data);
          tensorforge::fmacdpp16<12>(v68_acc, v76_data, v59_data);
          tensorforge::fmacdpp16<13>(v68_acc, v76_data, v60_data);
          tensorforge::fmacdpp16<14>(v68_acc, v76_data, v61_data);
          tensorforge::fmacdpp16<15>(v68_acc, v76_data, v62_data);
          tensorforge::fmacdpp16<0>(v69_acc, v77_data, v47_data);
          tensorforge::fmacdpp16<1>(v69_acc, v77_data, v48_data);
          tensorforge::fmacdpp16<2>(v69_acc, v77_data, v49_data);
          tensorforge::fmacdpp16<3>(v69_acc, v77_data, v50_data);
          tensorforge::fmacdpp16<4>(v69_acc, v77_data, v51_data);
          tensorforge::fmacdpp16<5>(v69_acc, v77_data, v52_data);
          tensorforge::fmacdpp16<6>(v69_acc, v77_data, v53_data);
          tensorforge::fmacdpp16<7>(v69_acc, v77_data, v54_data);
          tensorforge::fmacdpp16<8>(v69_acc, v77_data, v55_data);
          tensorforge::fmacdpp16<9>(v69_acc, v77_data, v56_data);
          tensorforge::fmacdpp16<10>(v69_acc, v77_data, v57_data);
          tensorforge::fmacdpp16<11>(v69_acc, v77_data, v58_data);
          tensorforge::fmacdpp16<12>(v69_acc, v77_data, v59_data);
          tensorforge::fmacdpp16<13>(v69_acc, v77_data, v60_data);
          tensorforge::fmacdpp16<14>(v69_acc, v77_data, v61_data);
          tensorforge::fmacdpp16<15>(v69_acc, v77_data, v62_data);
          tensorforge::fmacdpp16<0>(v70_acc, v78_data, v47_data);
          tensorforge::fmacdpp16<1>(v70_acc, v78_data, v48_data);
          tensorforge::fmacdpp16<2>(v70_acc, v78_data, v49_data);
          tensorforge::fmacdpp16<3>(v70_acc, v78_data, v50_data);
          tensorforge::fmacdpp16<4>(v70_acc, v78_data, v51_data);
          tensorforge::fmacdpp16<5>(v70_acc, v78_data, v52_data);
          tensorforge::fmacdpp16<6>(v70_acc, v78_data, v53_data);
          tensorforge::fmacdpp16<7>(v70_acc, v78_data, v54_data);
          tensorforge::fmacdpp16<8>(v70_acc, v78_data, v55_data);
          tensorforge::fmacdpp16<9>(v70_acc, v78_data, v56_data);
          tensorforge::fmacdpp16<10>(v70_acc, v78_data, v57_data);
          tensorforge::fmacdpp16<11>(v70_acc, v78_data, v58_data);
          tensorforge::fmacdpp16<12>(v70_acc, v78_data, v59_data);
          tensorforge::fmacdpp16<13>(v70_acc, v78_data, v60_data);
          tensorforge::fmacdpp16<14>(v70_acc, v78_data, v61_data);
          tensorforge::fmacdpp16<15>(v70_acc, v78_data, v62_data);
          r2[0] = v63_acc;
          r2[1] = v64_acc;
          r2[2] = v65_acc;
          r2[3] = v66_acc;
          r2[4] = v67_acc;
          r2[5] = v68_acc;
          r2[6] = v69_acc;
          r2[7] = v70_acc;
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v82_i0 = 0; v82_i0 < 1; ++v82_i0) {
            int32_t v90_lead = v17_lead + (v82_i0 * 16);
            #pragma unroll
            for (int32_t v83_i1 = 0; v83_i1 < 8; ++v83_i1) {
              double v85_data = r2[(v82_i0 + v83_i1)];
              glb_m0[(v90_lead + (v83_i1 * 16))] = v85_data;
            }
          }
        }
      }
    }
  }
}

