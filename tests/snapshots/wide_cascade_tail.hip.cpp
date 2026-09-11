// === base name ===
kernel_469c60117a00780e

// === header ===
void launcher_kernel_469c60117a00780e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_469c60117a00780e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_469c60117a00780e, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_469c60117a00780e, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_469c60117a00780e), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_469c60117a00780e, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_469c60117a00780e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 24×9(24×9) {0..24}×{0..9} strided
    // m1 24×24(24×24) {0..24}×{0..24} strided
    // m2 24×9(24×9) {0..24}×{0..9} strided
    // m0 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[0, 1] = m1 24×24(24×24) {0..24}×{0..24} strided({0..24}×{0..24})[0, -1]×m2 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t v0_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v0_batchId0 < numElements0; v0_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1_ahead1 = v0_batchId0 + (gridDim.x * blockDim.y);
        size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v0_batchId0 * 216 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 576 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v0_batchId0 * 216 + 0 + m2_extraOffset];
          float r0[24]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v14_lead = threadIdx.x % 32;
          if (v14_lead < 24) {
            #pragma unroll
            for (int32_t v16_i1 = 0; v16_i1 < 24; ++v16_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v14_lead + (v16_i1 * 24))]);
              r0[v16_i1] = v24_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m2);
          if (v14_lead < 24) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m2[(v14_lead + (v31_i1 * 24))]);
              r1[v31_i1] = v39_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 24), (0, 9)] [(0, 24)]
          float v42_data = r1[0];
          float v43_data = r1[1];
          float v44_data = r1[2];
          float v45_data = r1[3];
          float v46_tp{};
          float v47_tp{};
          float v48_tp{};
          float v49_tp{};
          tensorforge::transpose4x4b32(v46_tp, v47_tp, v48_tp, v49_tp, v42_data, v43_data, v44_data, v45_data);
          tensorforge::VectorT<float, 4> v50_acc{};
          float v51_data = r0[0];
          float v52_data = r0[1];
          float v53_data = r0[2];
          float v54_data = r0[3];
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v50_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 3, 0, 0);
          float v59_data = r0[4];
          float v60_data = r0[5];
          float v61_data = r0[6];
          float v62_data = r0[7];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v58_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 3, 1, 0);
          float v67_data = r0[8];
          float v68_data = r0[9];
          float v69_data = r0[10];
          float v70_data = r0[11];
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v66_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v71_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 3, 2, 0);
          float v75_data = r0[12];
          float v76_data = r0[13];
          float v77_data = r0[14];
          float v78_data = r0[15];
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v75_data, v74_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v76_data, v79_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v80_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v78_data, v81_acc, 3, 3, 0);
          float v83_data = r0[16];
          float v84_data = r0[17];
          float v85_data = r0[18];
          float v86_data = r0[19];
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v83_data, v82_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v84_data, v87_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v85_data, v88_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v86_data, v89_acc, 3, 4, 0);
          float v91_data = r0[20];
          float v92_data = r0[21];
          float v93_data = r0[22];
          float v94_data = r0[23];
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v91_data, v90_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v92_data, v95_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v93_data, v96_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v94_data, v97_acc, 3, 5, 0);
          r2[0] = (v98_acc[0]);
          r2[1] = (v98_acc[1]);
          r2[2] = (v98_acc[2]);
          r2[3] = (v98_acc[3]);
          float v103_data = r1[4];
          float v104_data = r1[5];
          float v105_data = r1[6];
          float v106_data = r1[7];
          float v107_tp{};
          float v108_tp{};
          float v109_tp{};
          float v110_tp{};
          tensorforge::transpose4x4b32(v107_tp, v108_tp, v109_tp, v110_tp, v103_data, v104_data, v105_data, v106_data);
          tensorforge::VectorT<float, 4> v111_acc{};
          tensorforge::VectorT<float, 4> v116_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v51_data, v111_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v52_data, v116_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v53_data, v117_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v54_data, v118_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v124_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v59_data, v119_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v60_data, v124_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v61_data, v125_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v62_data, v126_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v67_data, v127_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v68_data, v132_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v69_data, v133_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v70_data, v134_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v75_data, v135_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v76_data, v140_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v77_data, v141_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v78_data, v142_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v83_data, v143_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v84_data, v148_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v85_data, v149_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v86_data, v150_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v107_tp, v91_data, v151_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v108_tp, v92_data, v156_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v109_tp, v93_data, v157_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v110_tp, v94_data, v158_acc, 3, 5, 0);
          r2[4] = (v159_acc[0]);
          r2[5] = (v159_acc[1]);
          r2[6] = (v159_acc[2]);
          r2[7] = (v159_acc[3]);
          float v188_acc{};
          float v189_data = r1[8];
          float v190_bc = tensorforge::broadcast<32, 16, 0>(v189_data);
          tensorforge::fmacdpp16<0>(v188_acc, v190_bc, v51_data);
          tensorforge::fmacdpp16<1>(v188_acc, v190_bc, v52_data);
          tensorforge::fmacdpp16<2>(v188_acc, v190_bc, v53_data);
          tensorforge::fmacdpp16<3>(v188_acc, v190_bc, v54_data);
          tensorforge::fmacdpp16<4>(v188_acc, v190_bc, v59_data);
          tensorforge::fmacdpp16<5>(v188_acc, v190_bc, v60_data);
          tensorforge::fmacdpp16<6>(v188_acc, v190_bc, v61_data);
          tensorforge::fmacdpp16<7>(v188_acc, v190_bc, v62_data);
          tensorforge::fmacdpp16<8>(v188_acc, v190_bc, v67_data);
          tensorforge::fmacdpp16<9>(v188_acc, v190_bc, v68_data);
          tensorforge::fmacdpp16<10>(v188_acc, v190_bc, v69_data);
          tensorforge::fmacdpp16<11>(v188_acc, v190_bc, v70_data);
          tensorforge::fmacdpp16<12>(v188_acc, v190_bc, v75_data);
          tensorforge::fmacdpp16<13>(v188_acc, v190_bc, v76_data);
          tensorforge::fmacdpp16<14>(v188_acc, v190_bc, v77_data);
          tensorforge::fmacdpp16<15>(v188_acc, v190_bc, v78_data);
          float v191_bc = tensorforge::broadcast<32, 16, 1>(v189_data);
          tensorforge::fmacdpp16<0>(v188_acc, v191_bc, v83_data);
          tensorforge::fmacdpp16<1>(v188_acc, v191_bc, v84_data);
          tensorforge::fmacdpp16<2>(v188_acc, v191_bc, v85_data);
          tensorforge::fmacdpp16<3>(v188_acc, v191_bc, v86_data);
          tensorforge::fmacdpp16<4>(v188_acc, v191_bc, v91_data);
          tensorforge::fmacdpp16<5>(v188_acc, v191_bc, v92_data);
          tensorforge::fmacdpp16<6>(v188_acc, v191_bc, v93_data);
          tensorforge::fmacdpp16<7>(v188_acc, v191_bc, v94_data);
          r2[8] = v188_acc;
          // glb_m0 = store{r>g}(r2);
          if (v14_lead < 24) {
            #pragma unroll
            for (int32_t v196_i1 = 0; v196_i1 < 9; ++v196_i1) {
              float v198_data = r2[v196_i1];
              glb_m0[(v14_lead + (v196_i1 * 24))] = v198_data;
            }
          }
        }
      }
    }
  }
}

