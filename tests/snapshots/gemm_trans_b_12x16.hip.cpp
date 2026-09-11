// === base name ===
kernel_56a240d2c4bfed19

// === header ===
void launcher_kernel_56a240d2c4bfed19(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_56a240d2c4bfed19(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_56a240d2c4bfed19, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_56a240d2c4bfed19, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_56a240d2c4bfed19), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_56a240d2c4bfed19, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_56a240d2c4bfed19(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 12×16(12×16) {0..12}×{0..16} strided
    // m1 12×20(12×20) {0..12}×{0..20} strided
    // m2 16×20(16×20) {0..16}×{0..20} strided
    // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 12×20(12×20) {0..12}×{0..20} strided({0..12}×{0..20})[0, -1]×m2 16×20(16×20) {0..16}×{0..20} strided({0..16}×{0..20})[1, -1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t v3_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v3_batchId0 < numElements0; v3_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v4_ahead1 = v3_batchId0 + (gridDim.x * blockDim.y);
        size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v3_batchId0 * 192 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v3_batchId0 * 240 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v3_batchId0 * 320 + 0 + m2_extraOffset];
          float r0[20]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 16;
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 20; ++v19_i1) {
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v19_i1 * 12))]);
              r0[v19_i1] = v27_data;
            }
          }
          float r1[20]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
            int32_t v39_lead = v17_lead + (v33_i0 * 16);
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 20; ++v34_i1) {
              float v42_data = __builtin_nontemporal_load(&glb_m2[(v39_lead + (v34_i1 * 16))]);
              r1[(v33_i0 + v34_i1)] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v45_data = r1[0];
          float v53_tp{};
          float v54_tp{};
          float v55_tp{};
          float v56_tp{};
          tensorforge::transpose4x4b32(v53_tp, v54_tp, v55_tp, v56_tp, (tensorforge::broadcast<16, 1, 0>(v45_data)), (tensorforge::broadcast<16, 1, 1>(v45_data)), (tensorforge::broadcast<16, 1, 2>(v45_data)), (tensorforge::broadcast<16, 1, 3>(v45_data)));
          float v57_data = r1[1];
          float v65_tp{};
          float v66_tp{};
          float v67_tp{};
          float v68_tp{};
          tensorforge::transpose4x4b32(v65_tp, v66_tp, v67_tp, v68_tp, (tensorforge::broadcast<16, 1, 0>(v57_data)), (tensorforge::broadcast<16, 1, 1>(v57_data)), (tensorforge::broadcast<16, 1, 2>(v57_data)), (tensorforge::broadcast<16, 1, 3>(v57_data)));
          tensorforge::VectorT<float, 4> v69_acc{};
          float v70_data = r0[0];
          float v71_data = r0[1];
          float v72_data = r0[2];
          float v73_data = r0[3];
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v70_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v71_data, v74_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v72_data, v75_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v73_data, v76_acc, 2, 0, 0);
          float v78_data = r0[4];
          float v79_data = r0[5];
          float v80_data = r0[6];
          float v81_data = r0[7];
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v78_data, v77_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v79_data, v82_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v80_data, v83_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v81_data, v84_acc, 2, 1, 0);
          float v86_data = r0[8];
          float v87_data = r0[9];
          float v88_data = r0[10];
          float v89_data = r0[11];
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v86_data, v85_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v87_data, v90_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v88_data, v91_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v89_data, v92_acc, 2, 2, 0);
          float v94_data = r0[12];
          float v95_data = r0[13];
          float v96_data = r0[14];
          float v97_data = r0[15];
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v94_data, v93_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v54_tp, v95_data, v98_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v96_data, v99_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v97_data, v100_acc, 2, 3, 0);
          float v102_data = r0[16];
          float v103_data = r0[17];
          float v104_data = r0[18];
          float v105_data = r0[19];
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v102_data, v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v103_data, v106_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v104_data, v107_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v105_data, v108_acc, 2, 0, 0);
          r2[0] = (v109_acc[0]);
          r2[1] = (v109_acc[1]);
          r2[2] = (v109_acc[2]);
          r2[3] = (v109_acc[3]);
          float v122_tp{};
          float v123_tp{};
          float v124_tp{};
          float v125_tp{};
          tensorforge::transpose4x4b32(v122_tp, v123_tp, v124_tp, v125_tp, (tensorforge::broadcast<16, 1, 4>(v45_data)), (tensorforge::broadcast<16, 1, 5>(v45_data)), (tensorforge::broadcast<16, 1, 6>(v45_data)), (tensorforge::broadcast<16, 1, 7>(v45_data)));
          float v134_tp{};
          float v135_tp{};
          float v136_tp{};
          float v137_tp{};
          tensorforge::transpose4x4b32(v134_tp, v135_tp, v136_tp, v137_tp, (tensorforge::broadcast<16, 1, 4>(v57_data)), (tensorforge::broadcast<16, 1, 5>(v57_data)), (tensorforge::broadcast<16, 1, 6>(v57_data)), (tensorforge::broadcast<16, 1, 7>(v57_data)));
          tensorforge::VectorT<float, 4> v138_acc{};
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v70_data, v138_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v71_data, v143_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v72_data, v144_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v73_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v78_data, v146_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v79_data, v151_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v80_data, v152_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v81_data, v153_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v86_data, v154_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v87_data, v159_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v88_data, v160_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v89_data, v161_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v94_data, v162_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v95_data, v167_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v96_data, v168_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v97_data, v169_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v175_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v134_tp, v102_data, v170_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v176_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v135_tp, v103_data, v175_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v104_data, v176_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v105_data, v177_acc, 2, 0, 0);
          r2[4] = (v178_acc[0]);
          r2[5] = (v178_acc[1]);
          r2[6] = (v178_acc[2]);
          r2[7] = (v178_acc[3]);
          float v191_tp{};
          float v192_tp{};
          float v193_tp{};
          float v194_tp{};
          tensorforge::transpose4x4b32(v191_tp, v192_tp, v193_tp, v194_tp, (tensorforge::broadcast<16, 1, 8>(v45_data)), (tensorforge::broadcast<16, 1, 9>(v45_data)), (tensorforge::broadcast<16, 1, 10>(v45_data)), (tensorforge::broadcast<16, 1, 11>(v45_data)));
          float v203_tp{};
          float v204_tp{};
          float v205_tp{};
          float v206_tp{};
          tensorforge::transpose4x4b32(v203_tp, v204_tp, v205_tp, v206_tp, (tensorforge::broadcast<16, 1, 8>(v57_data)), (tensorforge::broadcast<16, 1, 9>(v57_data)), (tensorforge::broadcast<16, 1, 10>(v57_data)), (tensorforge::broadcast<16, 1, 11>(v57_data)));
          tensorforge::VectorT<float, 4> v207_acc{};
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v70_data, v207_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v71_data, v212_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v72_data, v213_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v73_data, v214_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v78_data, v215_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v79_data, v220_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v80_data, v221_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v81_data, v222_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v86_data, v223_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v87_data, v228_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v88_data, v229_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v89_data, v230_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v236_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v191_tp, v94_data, v231_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v192_tp, v95_data, v236_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v96_data, v237_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v97_data, v238_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v203_tp, v102_data, v239_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v204_tp, v103_data, v244_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v104_data, v245_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v105_data, v246_acc, 2, 0, 0);
          r2[8] = (v247_acc[0]);
          r2[9] = (v247_acc[1]);
          r2[10] = (v247_acc[2]);
          r2[11] = (v247_acc[3]);
          float v260_tp{};
          float v261_tp{};
          float v262_tp{};
          float v263_tp{};
          tensorforge::transpose4x4b32(v260_tp, v261_tp, v262_tp, v263_tp, (tensorforge::broadcast<16, 1, 12>(v45_data)), (tensorforge::broadcast<16, 1, 13>(v45_data)), (tensorforge::broadcast<16, 1, 14>(v45_data)), (tensorforge::broadcast<16, 1, 15>(v45_data)));
          float v272_tp{};
          float v273_tp{};
          float v274_tp{};
          float v275_tp{};
          tensorforge::transpose4x4b32(v272_tp, v273_tp, v274_tp, v275_tp, (tensorforge::broadcast<16, 1, 12>(v57_data)), (tensorforge::broadcast<16, 1, 13>(v57_data)), (tensorforge::broadcast<16, 1, 14>(v57_data)), (tensorforge::broadcast<16, 1, 15>(v57_data)));
          tensorforge::VectorT<float, 4> v276_acc{};
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v70_data, v276_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v71_data, v281_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v72_data, v282_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v73_data, v283_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v78_data, v284_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v79_data, v289_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v80_data, v290_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v81_data, v291_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v86_data, v292_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v87_data, v297_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v88_data, v298_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v89_data, v299_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v260_tp, v94_data, v300_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v261_tp, v95_data, v305_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v96_data, v306_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v97_data, v307_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v272_tp, v102_data, v308_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v103_data, v313_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v104_data, v314_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v105_data, v315_acc, 2, 0, 0);
          r2[12] = (v316_acc[0]);
          r2[13] = (v316_acc[1]);
          r2[14] = (v316_acc[2]);
          r2[15] = (v316_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v325_i1 = 0; v325_i1 < 16; ++v325_i1) {
              float v327_data = r2[v325_i1];
              glb_m0[(v17_lead + (v325_i1 * 12))] = v327_data;
            }
          }
        }
      }
    }
  }
}

