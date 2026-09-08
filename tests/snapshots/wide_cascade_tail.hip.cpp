// === base name ===
kernel_8162b17515

// === header ===
void launcher_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8162b17515, block.x * block.y * block.z, 0 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_8162b17515), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_8162b17515, grid, block, 0 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
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
          tensorforge::VectorT<float, 4> v23_lin = *(tensorforge::VectorT<float, 4>*)&glb_m2[0 + threadIdx.x * 4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[0] = v23_lin;
          tensorforge::VectorT<float, 2> v24_lin = *(tensorforge::VectorT<float, 2>*)&glb_m2[128 + threadIdx.x * 2];
          *(tensorforge::VectorRelaxedT<float, 2>*)&r1[4] = v24_lin;
          float v25_lin = glb_m2[192 + threadIdx.x * 1];
          r1[6] = v25_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[9]{};
          // r2 = +(r0 * r1) + None
          // [(0, 24), (0, 9)] [(0, 24)]
          float v27_data = r1[0];
          float v28_data = r1[1];
          float v29_data = r1[2];
          float v30_data = r1[3];
          float v31_tp{};
          float v32_tp{};
          float v33_tp{};
          float v34_tp{};
          tensorforge::transpose4x4b32(v31_tp, v32_tp, v33_tp, v34_tp, v27_data, v28_data, v29_data, v30_data);
          tensorforge::VectorT<float, 4> v35_acc{};
          float v36_data = r0[0];
          float v37_data = r0[1];
          float v38_data = r0[2];
          float v39_data = r0[3];
          tensorforge::VectorT<float, 4> v40_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v36_data, v35_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v41_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v37_data, v40_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v42_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v38_data, v41_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v39_data, v42_acc, 3, 0, 0);
          float v44_data = r0[4];
          float v45_data = r0[5];
          float v46_data = r0[6];
          float v47_data = r0[7];
          tensorforge::VectorT<float, 4> v48_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v44_data, v43_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v45_data, v48_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v46_data, v49_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v51_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v47_data, v50_acc, 3, 1, 0);
          float v52_data = r0[8];
          float v53_data = r0[9];
          float v54_data = r0[10];
          float v55_data = r0[11];
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v52_data, v51_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v53_data, v56_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v54_data, v57_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v55_data, v58_acc, 3, 2, 0);
          float v60_data = r0[12];
          float v61_data = r0[13];
          float v62_data = r0[14];
          float v63_data = r0[15];
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v60_data, v59_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v61_data, v64_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v62_data, v65_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v63_data, v66_acc, 3, 3, 0);
          float v68_data = r0[16];
          float v69_data = r0[17];
          float v70_data = r0[18];
          float v71_data = r0[19];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v68_data, v67_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v69_data, v72_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v70_data, v73_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v71_data, v74_acc, 3, 4, 0);
          float v76_data = r0[20];
          float v77_data = r0[21];
          float v78_data = r0[22];
          float v79_data = r0[23];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v31_tp, v76_data, v75_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v77_data, v80_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v78_data, v81_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v79_data, v82_acc, 3, 5, 0);
          r2[0] = (v83_acc[0]);
          r2[1] = (v83_acc[1]);
          r2[2] = (v83_acc[2]);
          r2[3] = (v83_acc[3]);
          float v88_data = r1[4];
          float v89_data = r1[5];
          float v90_data = r1[6];
          float v91_data = r1[7];
          float v92_tp{};
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          tensorforge::transpose4x4b32(v92_tp, v93_tp, v94_tp, v95_tp, v88_data, v89_data, v90_data, v91_data);
          tensorforge::VectorT<float, 4> v96_acc{};
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v36_data, v96_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v37_data, v101_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v38_data, v102_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v39_data, v103_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v44_data, v104_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v45_data, v109_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v46_data, v110_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v47_data, v111_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v52_data, v112_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v53_data, v117_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v54_data, v118_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v55_data, v119_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v60_data, v120_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v125_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v126_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v127_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v68_data, v128_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v133_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v134_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v135_acc, 3, 4, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v92_tp, v76_data, v136_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v141_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v142_acc, 3, 5, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v143_acc, 3, 5, 0);
          r2[4] = (v144_acc[0]);
          r2[5] = (v144_acc[1]);
          r2[6] = (v144_acc[2]);
          r2[7] = (v144_acc[3]);
          float v173_acc{};
          float v174_data = r1[8];
          float v175_bc = tensorforge::broadcast<32, 16, 0>(v174_data);
          tensorforge::fmacdpp16<0>(v173_acc, v175_bc, v36_data);
          tensorforge::fmacdpp16<1>(v173_acc, v175_bc, v37_data);
          tensorforge::fmacdpp16<2>(v173_acc, v175_bc, v38_data);
          tensorforge::fmacdpp16<3>(v173_acc, v175_bc, v39_data);
          tensorforge::fmacdpp16<4>(v173_acc, v175_bc, v44_data);
          tensorforge::fmacdpp16<5>(v173_acc, v175_bc, v45_data);
          tensorforge::fmacdpp16<6>(v173_acc, v175_bc, v46_data);
          tensorforge::fmacdpp16<7>(v173_acc, v175_bc, v47_data);
          tensorforge::fmacdpp16<8>(v173_acc, v175_bc, v52_data);
          tensorforge::fmacdpp16<9>(v173_acc, v175_bc, v53_data);
          tensorforge::fmacdpp16<10>(v173_acc, v175_bc, v54_data);
          tensorforge::fmacdpp16<11>(v173_acc, v175_bc, v55_data);
          tensorforge::fmacdpp16<12>(v173_acc, v175_bc, v60_data);
          tensorforge::fmacdpp16<13>(v173_acc, v175_bc, v61_data);
          tensorforge::fmacdpp16<14>(v173_acc, v175_bc, v62_data);
          tensorforge::fmacdpp16<15>(v173_acc, v175_bc, v63_data);
          float v176_bc = tensorforge::broadcast<32, 16, 1>(v174_data);
          tensorforge::fmacdpp16<0>(v173_acc, v176_bc, v68_data);
          tensorforge::fmacdpp16<1>(v173_acc, v176_bc, v69_data);
          tensorforge::fmacdpp16<2>(v173_acc, v176_bc, v70_data);
          tensorforge::fmacdpp16<3>(v173_acc, v176_bc, v71_data);
          tensorforge::fmacdpp16<4>(v173_acc, v176_bc, v76_data);
          tensorforge::fmacdpp16<5>(v173_acc, v176_bc, v77_data);
          tensorforge::fmacdpp16<6>(v173_acc, v176_bc, v78_data);
          tensorforge::fmacdpp16<7>(v173_acc, v176_bc, v79_data);
          r2[8] = v173_acc;
          // glb_m0 = store{r>g}(r2);
          if (v10_lead < 24) {
            #pragma unroll
            for (int32_t v181_i1 = 0; v181_i1 < 9; ++v181_i1) {
              float v183_data = r2[v181_i1];
              glb_m0[(v10_lead + (v181_i1 * 24))] = v183_data;
            }
          }
        }
      }
    }
  }
}

