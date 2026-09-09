// === base name ===
kernel_d12819472a51972d

// === header ===
void launcher_kernel_d12819472a51972d(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_d12819472a51972d(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d12819472a51972d, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_d12819472a51972d), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_d12819472a51972d, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_d12819472a51972d(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 216 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 576 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 216 + 0 + m2_extraOffset];
          float r0[24]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          if (v10_lead < 24) {
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 24; ++v12_i1) {
              float v20_data = __builtin_nontemporal_load(&glb_m1[(v10_lead + (v12_i1 * 24))]);
              r0[v12_i1] = v20_data;
            }
          }
          float r1[9]{};
          // r1 = load{g>r}(glb_m2);
          if (v10_lead < 24) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
              float v35_data = __builtin_nontemporal_load(&glb_m2[(v10_lead + (v27_i1 * 24))]);
              r1[v27_i1] = v35_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 24), (0, 9)] [(0, 24)]
          float v38_data = r1[0];
          float v39_data = r1[1];
          float v40_data = r1[2];
          float v41_data = r1[3];
          float v42_tp{};
          float v43_tp{};
          float v44_tp{};
          float v45_tp{};
          tensorforge::transpose4x4b32(v42_tp, v43_tp, v44_tp, v45_tp, v38_data, v39_data, v40_data, v41_data);
          tensorforge::VectorT<float, 4> v46_acc{};
          float v47_data = r0[0];
          float v48_data = r0[1];
          float v49_data = r0[2];
          float v50_data = r0[3];
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v47_data, v46_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v52_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v48_data, v51_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v53_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v49_data, v52_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v54_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v50_data, v53_acc, 3, 0, 0);
          float v55_data = r0[4];
          float v56_data = r0[5];
          float v57_data = r0[6];
          float v58_data = r0[7];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v55_data, v54_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v56_data, v59_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v57_data, v60_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v58_data, v61_acc, 3, 1, 0);
          float v63_data = r0[8];
          float v64_data = r0[9];
          float v65_data = r0[10];
          float v66_data = r0[11];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v63_data, v62_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v64_data, v67_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v65_data, v68_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v66_data, v69_acc, 3, 2, 0);
          float v71_data = r0[12];
          float v72_data = r0[13];
          float v73_data = r0[14];
          float v74_data = r0[15];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v71_data, v70_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v72_data, v75_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v73_data, v76_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v74_data, v77_acc, 3, 3, 0);
          float v79_data = r0[16];
          float v80_data = r0[17];
          float v81_data = r0[18];
          float v82_data = r0[19];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v79_data, v78_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v80_data, v83_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v81_data, v84_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v82_data, v85_acc, 3, 4, 0);
          float v87_data = r0[20];
          float v88_data = r0[21];
          float v89_data = r0[22];
          float v90_data = r0[23];
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v42_tp, v87_data, v86_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v88_data, v91_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, v89_data, v92_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v45_tp, v90_data, v93_acc, 3, 5, 0);
          r2[0] = (v94_acc[0]);
          r2[1] = (v94_acc[1]);
          r2[2] = (v94_acc[2]);
          r2[3] = (v94_acc[3]);
          float v99_data = r1[4];
          float v100_data = r1[5];
          float v101_data = r1[6];
          float v102_data = r1[7];
          float v103_tp{};
          float v104_tp{};
          float v105_tp{};
          float v106_tp{};
          tensorforge::transpose4x4b32(v103_tp, v104_tp, v105_tp, v106_tp, v99_data, v100_data, v101_data, v102_data);
          tensorforge::VectorT<float, 4> v107_acc{};
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v47_data, v107_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v48_data, v112_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v49_data, v113_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v50_data, v114_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v55_data, v115_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v56_data, v120_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v122_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v57_data, v121_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v58_data, v122_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v63_data, v123_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v64_data, v128_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v65_data, v129_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v66_data, v130_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v71_data, v131_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v72_data, v136_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v73_data, v137_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v74_data, v138_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v79_data, v139_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v80_data, v144_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v81_data, v145_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v82_data, v146_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v103_tp, v87_data, v147_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v104_tp, v88_data, v152_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v105_tp, v89_data, v153_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v106_tp, v90_data, v154_acc, 3, 5, 0);
          r2[4] = (v155_acc[0]);
          r2[5] = (v155_acc[1]);
          r2[6] = (v155_acc[2]);
          r2[7] = (v155_acc[3]);
          float v184_acc{};
          float v185_data = r1[8];
          float v186_bc = tensorforge::broadcast<32, 16, 0>(v185_data);
          tensorforge::fmacdpp16<0>(v184_acc, v186_bc, v47_data);
          tensorforge::fmacdpp16<1>(v184_acc, v186_bc, v48_data);
          tensorforge::fmacdpp16<2>(v184_acc, v186_bc, v49_data);
          tensorforge::fmacdpp16<3>(v184_acc, v186_bc, v50_data);
          tensorforge::fmacdpp16<4>(v184_acc, v186_bc, v55_data);
          tensorforge::fmacdpp16<5>(v184_acc, v186_bc, v56_data);
          tensorforge::fmacdpp16<6>(v184_acc, v186_bc, v57_data);
          tensorforge::fmacdpp16<7>(v184_acc, v186_bc, v58_data);
          tensorforge::fmacdpp16<8>(v184_acc, v186_bc, v63_data);
          tensorforge::fmacdpp16<9>(v184_acc, v186_bc, v64_data);
          tensorforge::fmacdpp16<10>(v184_acc, v186_bc, v65_data);
          tensorforge::fmacdpp16<11>(v184_acc, v186_bc, v66_data);
          tensorforge::fmacdpp16<12>(v184_acc, v186_bc, v71_data);
          tensorforge::fmacdpp16<13>(v184_acc, v186_bc, v72_data);
          tensorforge::fmacdpp16<14>(v184_acc, v186_bc, v73_data);
          tensorforge::fmacdpp16<15>(v184_acc, v186_bc, v74_data);
          float v187_bc = tensorforge::broadcast<32, 16, 1>(v185_data);
          tensorforge::fmacdpp16<0>(v184_acc, v187_bc, v79_data);
          tensorforge::fmacdpp16<1>(v184_acc, v187_bc, v80_data);
          tensorforge::fmacdpp16<2>(v184_acc, v187_bc, v81_data);
          tensorforge::fmacdpp16<3>(v184_acc, v187_bc, v82_data);
          tensorforge::fmacdpp16<4>(v184_acc, v187_bc, v87_data);
          tensorforge::fmacdpp16<5>(v184_acc, v187_bc, v88_data);
          tensorforge::fmacdpp16<6>(v184_acc, v187_bc, v89_data);
          tensorforge::fmacdpp16<7>(v184_acc, v187_bc, v90_data);
          r2[8] = v184_acc;
          // glb_m0 = store{r>g}(r2);
          if (v10_lead < 24) {
            #pragma unroll
            for (int32_t v192_i1 = 0; v192_i1 < 9; ++v192_i1) {
              float v194_data = r2[v192_i1];
              glb_m0[(v10_lead + (v192_i1 * 24))] = v194_data;
            }
          }
        }
      }
    }
  }
}

