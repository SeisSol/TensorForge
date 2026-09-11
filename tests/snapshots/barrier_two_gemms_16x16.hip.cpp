// === base name ===
kernel_65cc1128377a0ab3

// === header ===
void launcher_kernel_65cc1128377a0ab3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 = nullptr, unsigned* flags1 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_65cc1128377a0ab3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_65cc1128377a0ab3, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  dim3 grid (gridsize, 1, 1);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_65cc1128377a0ab3), hipFuncAttributeMaxDynamicSharedMemorySize, 0 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  
    auto args = tensorforge::argsPtrs( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  numElements1,  flags0 ,  flags1 );
    hipLaunchCooperativeKernel(kernel_kernel_65cc1128377a0ab3, grid, block, args.data(), 0 * sizeof(float), stream);
  ;
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_65cc1128377a0ab3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned* flags0 , unsigned* flags1 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m3 16×16(16×16) {0..16}×{0..16} strided
    // m4 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    // barrier
    // m3 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m4 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v0_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v0_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v0_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v0_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v0_batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v16_lead = threadIdx.x % 32;
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
              float v26_data = __builtin_nontemporal_load(&glb_m1[(v16_lead + (v18_i1 * 16))]);
              r0[v18_i1] = v26_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 16; ++v33_i1) {
              float v41_data = __builtin_nontemporal_load(&glb_m2[(v16_lead + (v33_i1 * 16))]);
              r1[v33_i1] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v44_data = r1[0];
          float v45_data = r1[1];
          float v46_data = r1[2];
          float v47_data = r1[3];
          float v48_tp{};
          float v49_tp{};
          float v50_tp{};
          float v51_tp{};
          tensorforge::transpose4x4b32(v48_tp, v49_tp, v50_tp, v51_tp, v44_data, v45_data, v46_data, v47_data);
          tensorforge::VectorT<float, 4> v52_acc{};
          float v53_data = r0[0];
          float v54_data = r0[1];
          float v55_data = r0[2];
          float v56_data = r0[3];
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v52_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v58_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v59_acc, 3, 0, 0);
          float v61_data = r0[4];
          float v62_data = r0[5];
          float v63_data = r0[6];
          float v64_data = r0[7];
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v60_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v66_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 3, 1, 0);
          float v69_data = r0[8];
          float v70_data = r0[9];
          float v71_data = r0[10];
          float v72_data = r0[11];
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v68_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v74_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 3, 2, 0);
          float v77_data = r0[12];
          float v78_data = r0[13];
          float v79_data = r0[14];
          float v80_data = r0[15];
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v77_data, v76_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v78_data, v81_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v79_data, v82_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v80_data, v83_acc, 3, 3, 0);
          r2[0] = (v84_acc[0]);
          r2[1] = (v84_acc[1]);
          r2[2] = (v84_acc[2]);
          r2[3] = (v84_acc[3]);
          float v89_data = r1[4];
          float v90_data = r1[5];
          float v91_data = r1[6];
          float v92_data = r1[7];
          float v93_tp{};
          float v94_tp{};
          float v95_tp{};
          float v96_tp{};
          tensorforge::transpose4x4b32(v93_tp, v94_tp, v95_tp, v96_tp, v89_data, v90_data, v91_data, v92_data);
          tensorforge::VectorT<float, 4> v97_acc{};
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v53_data, v97_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v54_data, v102_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v55_data, v103_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v56_data, v104_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v61_data, v105_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v62_data, v110_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v63_data, v111_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v64_data, v112_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v118_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v69_data, v113_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v119_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v70_data, v118_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v120_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v71_data, v119_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v72_data, v120_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v126_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v93_tp, v77_data, v121_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v94_tp, v78_data, v126_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v128_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v95_tp, v79_data, v127_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v96_tp, v80_data, v128_acc, 3, 3, 0);
          r2[4] = (v129_acc[0]);
          r2[5] = (v129_acc[1]);
          r2[6] = (v129_acc[2]);
          r2[7] = (v129_acc[3]);
          float v134_data = r1[8];
          float v135_data = r1[9];
          float v136_data = r1[10];
          float v137_data = r1[11];
          float v138_tp{};
          float v139_tp{};
          float v140_tp{};
          float v141_tp{};
          tensorforge::transpose4x4b32(v138_tp, v139_tp, v140_tp, v141_tp, v134_data, v135_data, v136_data, v137_data);
          tensorforge::VectorT<float, 4> v142_acc{};
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v53_data, v142_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v54_data, v147_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v55_data, v148_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v56_data, v149_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v61_data, v150_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v62_data, v155_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v63_data, v156_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v64_data, v157_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v69_data, v158_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v70_data, v163_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v71_data, v164_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v72_data, v165_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v77_data, v166_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v78_data, v171_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v140_tp, v79_data, v172_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v141_tp, v80_data, v173_acc, 3, 3, 0);
          r2[8] = (v174_acc[0]);
          r2[9] = (v174_acc[1]);
          r2[10] = (v174_acc[2]);
          r2[11] = (v174_acc[3]);
          float v179_data = r1[12];
          float v180_data = r1[13];
          float v181_data = r1[14];
          float v182_data = r1[15];
          float v183_tp{};
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          tensorforge::transpose4x4b32(v183_tp, v184_tp, v185_tp, v186_tp, v179_data, v180_data, v181_data, v182_data);
          tensorforge::VectorT<float, 4> v187_acc{};
          tensorforge::VectorT<float, 4> v192_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v53_data, v187_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v193_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v54_data, v192_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v55_data, v193_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v56_data, v194_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v200_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v61_data, v195_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v201_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v62_data, v200_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v63_data, v201_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v64_data, v202_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v69_data, v203_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v70_data, v208_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v71_data, v209_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v72_data, v210_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v77_data, v211_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, v78_data, v216_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v79_data, v217_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v80_data, v218_acc, 3, 3, 0);
          r2[12] = (v219_acc[0]);
          r2[13] = (v219_acc[1]);
          r2[14] = (v219_acc[2]);
          r2[15] = (v219_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v16_lead < 16) {
            #pragma unroll
            for (int32_t v228_i1 = 0; v228_i1 < 16; ++v228_i1) {
              float v230_data = r2[v228_i1];
              glb_m0[(v16_lead + (v228_i1 * 16))] = v230_data;
            }
          }
        }
      }
    }
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements1 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      __syncthreads();
      cooperative_groups::this_grid().sync();
      for (size_t v238_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v238_batchId0 < numElements1; v238_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v239_ahead1 = v238_batchId0 + (gridDim.x * blockDim.y);
        size_t v241_batchId1 = (v239_ahead1 < numElements1) ? v239_ahead1 : v238_batchId0;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[v238_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v238_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v238_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v238_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v238_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v238_batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v254_lead = threadIdx.x % 32;
          if (v254_lead < 16) {
            #pragma unroll
            for (int32_t v256_i1 = 0; v256_i1 < 16; ++v256_i1) {
              float v264_data = __builtin_nontemporal_load(&glb_m0[(v254_lead + (v256_i1 * 16))]);
              r0[v256_i1] = v264_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m4);
          if (v254_lead < 16) {
            #pragma unroll
            for (int32_t v271_i1 = 0; v271_i1 < 16; ++v271_i1) {
              float v279_data = __builtin_nontemporal_load(&glb_m4[(v254_lead + (v271_i1 * 16))]);
              r1[v271_i1] = v279_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v282_data = r1[0];
          float v283_data = r1[1];
          float v284_data = r1[2];
          float v285_data = r1[3];
          float v286_tp{};
          float v287_tp{};
          float v288_tp{};
          float v289_tp{};
          tensorforge::transpose4x4b32(v286_tp, v287_tp, v288_tp, v289_tp, v282_data, v283_data, v284_data, v285_data);
          tensorforge::VectorT<float, 4> v290_acc{};
          float v291_data = r0[0];
          float v292_data = r0[1];
          float v293_data = r0[2];
          float v294_data = r0[3];
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v291_data, v290_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v292_data, v295_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v297_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v293_data, v296_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v294_data, v297_acc, 3, 0, 0);
          float v299_data = r0[4];
          float v300_data = r0[5];
          float v301_data = r0[6];
          float v302_data = r0[7];
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v299_data, v298_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v300_data, v303_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v301_data, v304_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v302_data, v305_acc, 3, 1, 0);
          float v307_data = r0[8];
          float v308_data = r0[9];
          float v309_data = r0[10];
          float v310_data = r0[11];
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v307_data, v306_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v308_data, v311_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v313_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v309_data, v312_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v314_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v310_data, v313_acc, 3, 2, 0);
          float v315_data = r0[12];
          float v316_data = r0[13];
          float v317_data = r0[14];
          float v318_data = r0[15];
          tensorforge::VectorT<float, 4> v319_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v286_tp, v315_data, v314_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v320_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v287_tp, v316_data, v319_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v321_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v288_tp, v317_data, v320_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v322_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v289_tp, v318_data, v321_acc, 3, 3, 0);
          r2[0] = (v322_acc[0]);
          r2[1] = (v322_acc[1]);
          r2[2] = (v322_acc[2]);
          r2[3] = (v322_acc[3]);
          float v327_data = r1[4];
          float v328_data = r1[5];
          float v329_data = r1[6];
          float v330_data = r1[7];
          float v331_tp{};
          float v332_tp{};
          float v333_tp{};
          float v334_tp{};
          tensorforge::transpose4x4b32(v331_tp, v332_tp, v333_tp, v334_tp, v327_data, v328_data, v329_data, v330_data);
          tensorforge::VectorT<float, 4> v335_acc{};
          tensorforge::VectorT<float, 4> v340_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v291_data, v335_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v341_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v292_data, v340_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v342_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v293_data, v341_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v343_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v294_data, v342_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v348_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v299_data, v343_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v349_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v300_data, v348_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v301_data, v349_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v302_data, v350_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v356_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v307_data, v351_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v357_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v308_data, v356_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v358_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v309_data, v357_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v359_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v310_data, v358_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v364_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v331_tp, v315_data, v359_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v365_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v332_tp, v316_data, v364_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v366_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v333_tp, v317_data, v365_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v367_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v334_tp, v318_data, v366_acc, 3, 3, 0);
          r2[4] = (v367_acc[0]);
          r2[5] = (v367_acc[1]);
          r2[6] = (v367_acc[2]);
          r2[7] = (v367_acc[3]);
          float v372_data = r1[8];
          float v373_data = r1[9];
          float v374_data = r1[10];
          float v375_data = r1[11];
          float v376_tp{};
          float v377_tp{};
          float v378_tp{};
          float v379_tp{};
          tensorforge::transpose4x4b32(v376_tp, v377_tp, v378_tp, v379_tp, v372_data, v373_data, v374_data, v375_data);
          tensorforge::VectorT<float, 4> v380_acc{};
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v291_data, v380_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v386_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v292_data, v385_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v387_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v293_data, v386_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v388_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v294_data, v387_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v393_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v299_data, v388_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v394_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v300_data, v393_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v395_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v301_data, v394_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v396_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v302_data, v395_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v401_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v307_data, v396_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v402_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v308_data, v401_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v403_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v309_data, v402_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v404_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v310_data, v403_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v409_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v376_tp, v315_data, v404_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v410_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v377_tp, v316_data, v409_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v411_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v378_tp, v317_data, v410_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v412_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v379_tp, v318_data, v411_acc, 3, 3, 0);
          r2[8] = (v412_acc[0]);
          r2[9] = (v412_acc[1]);
          r2[10] = (v412_acc[2]);
          r2[11] = (v412_acc[3]);
          float v417_data = r1[12];
          float v418_data = r1[13];
          float v419_data = r1[14];
          float v420_data = r1[15];
          float v421_tp{};
          float v422_tp{};
          float v423_tp{};
          float v424_tp{};
          tensorforge::transpose4x4b32(v421_tp, v422_tp, v423_tp, v424_tp, v417_data, v418_data, v419_data, v420_data);
          tensorforge::VectorT<float, 4> v425_acc{};
          tensorforge::VectorT<float, 4> v430_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v291_data, v425_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v431_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v292_data, v430_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v432_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v293_data, v431_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v433_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v294_data, v432_acc, 3, 0, 0);
          tensorforge::VectorT<float, 4> v438_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v299_data, v433_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v439_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v300_data, v438_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v440_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v301_data, v439_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v441_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v302_data, v440_acc, 3, 1, 0);
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v307_data, v441_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v308_data, v446_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v309_data, v447_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v310_data, v448_acc, 3, 2, 0);
          tensorforge::VectorT<float, 4> v454_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v421_tp, v315_data, v449_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v455_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v422_tp, v316_data, v454_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v456_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v423_tp, v317_data, v455_acc, 3, 3, 0);
          tensorforge::VectorT<float, 4> v457_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v424_tp, v318_data, v456_acc, 3, 3, 0);
          r2[12] = (v457_acc[0]);
          r2[13] = (v457_acc[1]);
          r2[14] = (v457_acc[2]);
          r2[15] = (v457_acc[3]);
          // glb_m3 = store{r>g}(r2);
          if (v254_lead < 16) {
            #pragma unroll
            for (int32_t v466_i1 = 0; v466_i1 < 16; ++v466_i1) {
              float v468_data = r2[v466_i1];
              glb_m3[(v254_lead + (v466_i1 * 16))] = v468_data;
            }
          }
        }
      }
    }
  }
}

