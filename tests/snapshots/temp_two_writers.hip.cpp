// === base name ===
kernel_3e24e7feaf

// === header ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3e24e7feaf, block.x * block.y * block.z, 3328 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_3e24e7feaf), hipFuncAttributeMaxDynamicSharedMemorySize, 3328 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_3e24e7feaf, grid, block, 3328 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×32(6×12) {0..6}×{0..12} strided
    // m1 32×32(12×12) {0..12}×{0..12} strided
    // m2 32×32(6×12) {0..6}×{0..12} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // m4 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
              int32_t v11_a = v5_i1 * 6;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m0[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r1[12]{};
          {
            // r1 = load{g>r}(glb_m1);
            float v0 = glb_m1[0 + threadIdx.x * 1];
            r1[0] = v0;
            float v16 = glb_m1[16 + threadIdx.x * 1];
            r1[1] = v16;
            float v32 = glb_m1[32 + threadIdx.x * 1];
            r1[2] = v32;
            float v48 = glb_m1[48 + threadIdx.x * 1];
            r1[3] = v48;
            float v64 = glb_m1[64 + threadIdx.x * 1];
            r1[4] = v64;
            float v80 = glb_m1[80 + threadIdx.x * 1];
            r1[5] = v80;
            float v96 = glb_m1[96 + threadIdx.x * 1];
            r1[6] = v96;
            float v112 = glb_m1[112 + threadIdx.x * 1];
            r1[7] = v112;
            float v128 = glb_m1[128 + threadIdx.x * 1];
            r1[8] = v128;
            float v144 = glb_m1[144 + threadIdx.x * 1];
            r1[9] = v144;
            float v160 = glb_m1[160 + threadIdx.x * 1];
            r1[10] = v160;
            float v176 = glb_m1[176 + threadIdx.x * 1];
            r1[11] = v176;
            float v192 = glb_m1[192 + threadIdx.x * 1];
            r1[12] = v192;
            float v208 = glb_m1[208 + threadIdx.x * 1];
            r1[13] = v208;
            float v224 = glb_m1[224 + threadIdx.x * 1];
            r1[14] = v224;
            float v240 = glb_m1[240 + threadIdx.x * 1];
            r1[15] = v240;
            float v256 = glb_m1[256 + threadIdx.x * 1];
            r1[16] = v256;
            float v272 = glb_m1[272 + threadIdx.x * 1];
            r1[17] = v272;
            float v288 = glb_m1[288 + threadIdx.x * 1];
            r1[18] = v288;
            float v304 = glb_m1[304 + threadIdx.x * 1];
            r1[19] = v304;
            float v320 = glb_m1[320 + threadIdx.x * 1];
            r1[20] = v320;
            float v336 = glb_m1[336 + threadIdx.x * 1];
            r1[21] = v336;
            float v352 = glb_m1[352 + threadIdx.x * 1];
            r1[22] = v352;
            float v368 = glb_m1[368 + threadIdx.x * 1];
            r1[23] = v368;
            float v384 = glb_m1[384 + threadIdx.x * 1];
            r1[24] = v384;
            float v400 = glb_m1[400 + threadIdx.x * 1];
            r1[25] = v400;
            float v416 = glb_m1[416 + threadIdx.x * 1];
            r1[26] = v416;
            float v432 = glb_m1[432 + threadIdx.x * 1];
            r1[27] = v432;
            float v448 = glb_m1[448 + threadIdx.x * 1];
            r1[28] = v448;
            float v464 = glb_m1[464 + threadIdx.x * 1];
            r1[29] = v464;
            float v480 = glb_m1[480 + threadIdx.x * 1];
            r1[30] = v480;
            float v496 = glb_m1[496 + threadIdx.x * 1];
            r1[31] = v496;
            float v512 = glb_m1[512 + threadIdx.x * 1];
            r1[32] = v512;
            float v528 = glb_m1[528 + threadIdx.x * 1];
            r1[33] = v528;
            float v544 = glb_m1[544 + threadIdx.x * 1];
            r1[34] = v544;
            float v560 = glb_m1[560 + threadIdx.x * 1];
            r1[35] = v560;
            float v576 = glb_m1[576 + threadIdx.x * 1];
            r1[36] = v576;
            float v592 = glb_m1[592 + threadIdx.x * 1];
            r1[37] = v592;
            float v608 = glb_m1[608 + threadIdx.x * 1];
            r1[38] = v608;
            float v624 = glb_m1[624 + threadIdx.x * 1];
            r1[39] = v624;
            float v640 = glb_m1[640 + threadIdx.x * 1];
            r1[40] = v640;
            float v656 = glb_m1[656 + threadIdx.x * 1];
            r1[41] = v656;
            float v672 = glb_m1[672 + threadIdx.x * 1];
            r1[42] = v672;
            float v688 = glb_m1[688 + threadIdx.x * 1];
            r1[43] = v688;
            float v704 = glb_m1[704 + threadIdx.x * 1];
            r1[44] = v704;
            float v720 = glb_m1[720 + threadIdx.x * 1];
            r1[45] = v720;
            float v736 = glb_m1[736 + threadIdx.x * 1];
            r1[46] = v736;
            float v752 = glb_m1[752 + threadIdx.x * 1];
            r1[47] = v752;
            float v768 = glb_m1[768 + threadIdx.x * 1];
            r1[48] = v768;
            float v784 = glb_m1[784 + threadIdx.x * 1];
            r1[49] = v784;
            float v800 = glb_m1[800 + threadIdx.x * 1];
            r1[50] = v800;
            float v816 = glb_m1[816 + threadIdx.x * 1];
            r1[51] = v816;
            float v832 = glb_m1[832 + threadIdx.x * 1];
            r1[52] = v832;
            float v848 = glb_m1[848 + threadIdx.x * 1];
            r1[53] = v848;
            float v864 = glb_m1[864 + threadIdx.x * 1];
            r1[54] = v864;
            float v880 = glb_m1[880 + threadIdx.x * 1];
            r1[55] = v880;
            float v896 = glb_m1[896 + threadIdx.x * 1];
            r1[56] = v896;
            float v912 = glb_m1[912 + threadIdx.x * 1];
            r1[57] = v912;
            float v928 = glb_m1[928 + threadIdx.x * 1];
            r1[58] = v928;
            float v944 = glb_m1[944 + threadIdx.x * 1];
            r1[59] = v944;
            float v960 = glb_m1[960 + threadIdx.x * 1];
            r1[60] = v960;
            float v976 = glb_m1[976 + threadIdx.x * 1];
            r1[61] = v976;
            float v992 = glb_m1[992 + threadIdx.x * 1];
            r1[62] = v992;
            float v1008 = glb_m1[1008 + threadIdx.x * 1];
            r1[63] = v1008;
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r3[12]{};
          // r3 = load{g>r}(glb_m2);
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              int32_t v34_a = v28_i1 * 6;
              int32_t v35_a = v3_lead + v34_a;
              float v43_data = __builtin_nontemporal_load(&glb_m2[(v3_lead + v34_a)]);
              int32_t v44_a = 0 + v28_i1;
              r3[v44_a] = v43_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir2 = r2;
          float v46_data = r1[0];
          float v47_data = r1[1];
          float v48_data = r1[2];
          float v49_data = r1[3];
          float v50_tp{};
          float v51_tp{};
          float v52_tp{};
          float v53_tp{};
          tensorforge::transpose4x4b32(v50_tp, v51_tp, v52_tp, v53_tp, v46_data, v47_data, v48_data, v49_data);
          tensorforge::VectorT<float, 4> v54_acc{};
          float v55_data = r0[0];
          float v56_data = r0[1];
          float v57_data = r0[2];
          float v58_data = r0[3];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v55_data, v54_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v60_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v56_data, v59_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v57_data, v60_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v62_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v58_data, v61_acc, 2, 0, 0);
          float v63_data = r0[4];
          float v64_data = r0[5];
          float v65_data = r0[6];
          float v66_data = r0[7];
          tensorforge::VectorT<float, 4> v67_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v63_data, v62_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v64_data, v67_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v65_data, v68_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v66_data, v69_acc, 2, 1, 0);
          float v71_data = r0[8];
          float v72_data = r0[9];
          float v73_data = r0[10];
          float v74_data = r0[11];
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v71_data, v70_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v72_data, v75_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v73_data, v76_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v53_tp, v74_data, v77_acc, 2, 2, 0);
          ir2[0] = (v78_acc[0]);
          ir2[1] = (v78_acc[1]);
          ir2[2] = (v78_acc[2]);
          ir2[3] = (v78_acc[3]);
          float v83_data = r1[4];
          float v84_data = r1[5];
          float v85_data = r1[6];
          float v86_data = r1[7];
          float v87_tp{};
          float v88_tp{};
          float v89_tp{};
          float v90_tp{};
          tensorforge::transpose4x4b32(v87_tp, v88_tp, v89_tp, v90_tp, v83_data, v84_data, v85_data, v86_data);
          tensorforge::VectorT<float, 4> v91_acc{};
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v55_data, v91_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v56_data, v96_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v57_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v58_data, v98_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v63_data, v99_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v64_data, v104_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v65_data, v105_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v66_data, v106_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v112_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v87_tp, v71_data, v107_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v88_tp, v72_data, v112_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v114_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v89_tp, v73_data, v113_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v90_tp, v74_data, v114_acc, 2, 2, 0);
          ir2[4] = (v115_acc[0]);
          ir2[5] = (v115_acc[1]);
          ir2[6] = (v115_acc[2]);
          ir2[7] = (v115_acc[3]);
          float v120_data = r1[8];
          float v121_data = r1[9];
          float v122_data = r1[10];
          float v123_data = r1[11];
          float v124_tp{};
          float v125_tp{};
          float v126_tp{};
          float v127_tp{};
          tensorforge::transpose4x4b32(v124_tp, v125_tp, v126_tp, v127_tp, v120_data, v121_data, v122_data, v123_data);
          tensorforge::VectorT<float, 4> v128_acc{};
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v55_data, v128_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v56_data, v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v57_data, v134_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v58_data, v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v63_data, v136_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v64_data, v141_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v65_data, v142_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v66_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v71_data, v144_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v72_data, v149_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v73_data, v150_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v74_data, v151_acc, 2, 2, 0);
          ir2[8] = (v152_acc[0]);
          ir2[9] = (v152_acc[1]);
          ir2[10] = (v152_acc[2]);
          ir2[11] = (v152_acc[3]);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r2);
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v161_i1 = 0; v161_i1 < 12; ++v161_i1) {
              int32_t v162_a = 0 + v161_i1;
              float v164_data = r2[v161_i1];
              int32_t v171_a = v3_lead + (v161_i1 * 12);
              s0[v171_a] = v164_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v177_i1 = 0; v177_i1 < 12; ++v177_i1) {
              int32_t v183_a = v177_i1 * 12;
              int32_t v184_a = v3_lead + v183_a;
              float v192_data = __builtin_nontemporal_load(&glb_m4[(v3_lead + v183_a)]);
              int32_t v193_a = 0 + v177_i1;
              r5[v193_a] = v192_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir4 = r4;
          float v195_data = r1[0];
          float v196_data = r1[1];
          float v197_data = r1[2];
          float v198_data = r1[3];
          float v199_tp{};
          float v200_tp{};
          float v201_tp{};
          float v202_tp{};
          tensorforge::transpose4x4b32(v199_tp, v200_tp, v201_tp, v202_tp, v195_data, v196_data, v197_data, v198_data);
          tensorforge::VectorT<float, 4> v203_acc{};
          float v204_data = r3[0];
          float v205_data = r3[1];
          float v206_data = r3[2];
          float v207_data = r3[3];
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v204_data, v203_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v205_data, v208_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v206_data, v209_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v207_data, v210_acc, 2, 0, 0);
          float v212_data = r3[4];
          float v213_data = r3[5];
          float v214_data = r3[6];
          float v215_data = r3[7];
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v212_data, v211_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v213_data, v216_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v214_data, v217_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v215_data, v218_acc, 2, 1, 0);
          float v220_data = r3[8];
          float v221_data = r3[9];
          float v222_data = r3[10];
          float v223_data = r3[11];
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v220_data, v219_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v221_data, v224_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v222_data, v225_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v223_data, v226_acc, 2, 2, 0);
          ir4[0] = (v227_acc[0]);
          ir4[1] = (v227_acc[1]);
          ir4[2] = (v227_acc[2]);
          ir4[3] = (v227_acc[3]);
          float v232_data = r1[4];
          float v233_data = r1[5];
          float v234_data = r1[6];
          float v235_data = r1[7];
          float v236_tp{};
          float v237_tp{};
          float v238_tp{};
          float v239_tp{};
          tensorforge::transpose4x4b32(v236_tp, v237_tp, v238_tp, v239_tp, v232_data, v233_data, v234_data, v235_data);
          tensorforge::VectorT<float, 4> v240_acc{};
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v204_data, v240_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v205_data, v245_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v206_data, v246_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v207_data, v247_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v253_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v212_data, v248_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v254_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v213_data, v253_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v214_data, v254_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v215_data, v255_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v236_tp, v220_data, v256_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v262_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v237_tp, v221_data, v261_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v222_data, v262_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v223_data, v263_acc, 2, 2, 0);
          ir4[4] = (v264_acc[0]);
          ir4[5] = (v264_acc[1]);
          ir4[6] = (v264_acc[2]);
          ir4[7] = (v264_acc[3]);
          float v269_data = r1[8];
          float v270_data = r1[9];
          float v271_data = r1[10];
          float v272_data = r1[11];
          float v273_tp{};
          float v274_tp{};
          float v275_tp{};
          float v276_tp{};
          tensorforge::transpose4x4b32(v273_tp, v274_tp, v275_tp, v276_tp, v269_data, v270_data, v271_data, v272_data);
          tensorforge::VectorT<float, 4> v277_acc{};
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v204_data, v277_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v205_data, v282_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v206_data, v283_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v207_data, v284_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v212_data, v285_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v213_data, v290_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v214_data, v291_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v215_data, v292_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v298_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v273_tp, v220_data, v293_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v221_data, v298_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v222_data, v299_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v223_data, v300_acc, 2, 2, 0);
          ir4[8] = (v301_acc[0]);
          ir4[9] = (v301_acc[1]);
          ir4[10] = (v301_acc[2]);
          ir4[11] = (v301_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v3_lead < 6) {
            int32_t v319_off = v3_lead + 6;
            #pragma unroll
            for (int32_t v310_i1 = 0; v310_i1 < 12; ++v310_i1) {
              int32_t v311_a = 0 + v310_i1;
              float v313_data = r4[v310_i1];
              int32_t v321_a = v319_off + (v310_i1 * 12);
              s0[v321_a] = v313_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          ;
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          auto& ir6 = r6;
          float v323_data = r5[0];
          float v324_data = r5[1];
          float v325_data = r5[2];
          float v326_data = r5[3];
          float v327_data = r5[4];
          float v328_data = r5[5];
          float v329_data = r5[6];
          float v330_data = r5[7];
          float v331_data = r5[8];
          float v332_data = r5[9];
          float v333_data = r5[10];
          float v334_data = r5[11];
          float v335_acc{};
          float v336_acc{};
          float v337_acc{};
          float v338_acc{};
          float v339_acc{};
          float v340_acc{};
          float v341_acc{};
          float v342_acc{};
          float v343_acc{};
          float v344_acc{};
          float v345_acc{};
          float v346_acc{};
          float v347_lin = s0[0 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v335_acc, v347_lin, v323_data);
          tensorforge::fmacdpp16<1>(v335_acc, v347_lin, v324_data);
          tensorforge::fmacdpp16<2>(v335_acc, v347_lin, v325_data);
          tensorforge::fmacdpp16<3>(v335_acc, v347_lin, v326_data);
          tensorforge::fmacdpp16<4>(v335_acc, v347_lin, v327_data);
          tensorforge::fmacdpp16<5>(v335_acc, v347_lin, v328_data);
          tensorforge::fmacdpp16<6>(v335_acc, v347_lin, v329_data);
          tensorforge::fmacdpp16<7>(v335_acc, v347_lin, v330_data);
          tensorforge::fmacdpp16<8>(v335_acc, v347_lin, v331_data);
          tensorforge::fmacdpp16<9>(v335_acc, v347_lin, v332_data);
          tensorforge::fmacdpp16<10>(v335_acc, v347_lin, v333_data);
          tensorforge::fmacdpp16<11>(v335_acc, v347_lin, v334_data);
          tensorforge::fmacdpp16<12>(v336_acc, v347_lin, v323_data);
          tensorforge::fmacdpp16<13>(v336_acc, v347_lin, v324_data);
          tensorforge::fmacdpp16<14>(v336_acc, v347_lin, v325_data);
          tensorforge::fmacdpp16<15>(v336_acc, v347_lin, v326_data);
          float v348_lin = s0[16 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v336_acc, v348_lin, v327_data);
          tensorforge::fmacdpp16<1>(v336_acc, v348_lin, v328_data);
          tensorforge::fmacdpp16<2>(v336_acc, v348_lin, v329_data);
          tensorforge::fmacdpp16<3>(v336_acc, v348_lin, v330_data);
          tensorforge::fmacdpp16<4>(v336_acc, v348_lin, v331_data);
          tensorforge::fmacdpp16<5>(v336_acc, v348_lin, v332_data);
          tensorforge::fmacdpp16<6>(v336_acc, v348_lin, v333_data);
          tensorforge::fmacdpp16<7>(v336_acc, v348_lin, v334_data);
          tensorforge::fmacdpp16<8>(v337_acc, v348_lin, v323_data);
          tensorforge::fmacdpp16<9>(v337_acc, v348_lin, v324_data);
          tensorforge::fmacdpp16<10>(v337_acc, v348_lin, v325_data);
          tensorforge::fmacdpp16<11>(v337_acc, v348_lin, v326_data);
          tensorforge::fmacdpp16<12>(v337_acc, v348_lin, v327_data);
          tensorforge::fmacdpp16<13>(v337_acc, v348_lin, v328_data);
          tensorforge::fmacdpp16<14>(v337_acc, v348_lin, v329_data);
          tensorforge::fmacdpp16<15>(v337_acc, v348_lin, v330_data);
          float v349_lin = s0[32 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v337_acc, v349_lin, v331_data);
          tensorforge::fmacdpp16<1>(v337_acc, v349_lin, v332_data);
          tensorforge::fmacdpp16<2>(v337_acc, v349_lin, v333_data);
          tensorforge::fmacdpp16<3>(v337_acc, v349_lin, v334_data);
          tensorforge::fmacdpp16<4>(v338_acc, v349_lin, v323_data);
          tensorforge::fmacdpp16<5>(v338_acc, v349_lin, v324_data);
          tensorforge::fmacdpp16<6>(v338_acc, v349_lin, v325_data);
          tensorforge::fmacdpp16<7>(v338_acc, v349_lin, v326_data);
          tensorforge::fmacdpp16<8>(v338_acc, v349_lin, v327_data);
          tensorforge::fmacdpp16<9>(v338_acc, v349_lin, v328_data);
          tensorforge::fmacdpp16<10>(v338_acc, v349_lin, v329_data);
          tensorforge::fmacdpp16<11>(v338_acc, v349_lin, v330_data);
          tensorforge::fmacdpp16<12>(v338_acc, v349_lin, v331_data);
          tensorforge::fmacdpp16<13>(v338_acc, v349_lin, v332_data);
          tensorforge::fmacdpp16<14>(v338_acc, v349_lin, v333_data);
          tensorforge::fmacdpp16<15>(v338_acc, v349_lin, v334_data);
          float v350_lin = s0[48 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v339_acc, v350_lin, v323_data);
          tensorforge::fmacdpp16<1>(v339_acc, v350_lin, v324_data);
          tensorforge::fmacdpp16<2>(v339_acc, v350_lin, v325_data);
          tensorforge::fmacdpp16<3>(v339_acc, v350_lin, v326_data);
          tensorforge::fmacdpp16<4>(v339_acc, v350_lin, v327_data);
          tensorforge::fmacdpp16<5>(v339_acc, v350_lin, v328_data);
          tensorforge::fmacdpp16<6>(v339_acc, v350_lin, v329_data);
          tensorforge::fmacdpp16<7>(v339_acc, v350_lin, v330_data);
          tensorforge::fmacdpp16<8>(v339_acc, v350_lin, v331_data);
          tensorforge::fmacdpp16<9>(v339_acc, v350_lin, v332_data);
          tensorforge::fmacdpp16<10>(v339_acc, v350_lin, v333_data);
          tensorforge::fmacdpp16<11>(v339_acc, v350_lin, v334_data);
          tensorforge::fmacdpp16<12>(v340_acc, v350_lin, v323_data);
          tensorforge::fmacdpp16<13>(v340_acc, v350_lin, v324_data);
          tensorforge::fmacdpp16<14>(v340_acc, v350_lin, v325_data);
          tensorforge::fmacdpp16<15>(v340_acc, v350_lin, v326_data);
          float v351_lin = s0[64 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v340_acc, v351_lin, v327_data);
          tensorforge::fmacdpp16<1>(v340_acc, v351_lin, v328_data);
          tensorforge::fmacdpp16<2>(v340_acc, v351_lin, v329_data);
          tensorforge::fmacdpp16<3>(v340_acc, v351_lin, v330_data);
          tensorforge::fmacdpp16<4>(v340_acc, v351_lin, v331_data);
          tensorforge::fmacdpp16<5>(v340_acc, v351_lin, v332_data);
          tensorforge::fmacdpp16<6>(v340_acc, v351_lin, v333_data);
          tensorforge::fmacdpp16<7>(v340_acc, v351_lin, v334_data);
          tensorforge::fmacdpp16<8>(v341_acc, v351_lin, v323_data);
          tensorforge::fmacdpp16<9>(v341_acc, v351_lin, v324_data);
          tensorforge::fmacdpp16<10>(v341_acc, v351_lin, v325_data);
          tensorforge::fmacdpp16<11>(v341_acc, v351_lin, v326_data);
          tensorforge::fmacdpp16<12>(v341_acc, v351_lin, v327_data);
          tensorforge::fmacdpp16<13>(v341_acc, v351_lin, v328_data);
          tensorforge::fmacdpp16<14>(v341_acc, v351_lin, v329_data);
          tensorforge::fmacdpp16<15>(v341_acc, v351_lin, v330_data);
          float v352_lin = s0[80 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v341_acc, v352_lin, v331_data);
          tensorforge::fmacdpp16<1>(v341_acc, v352_lin, v332_data);
          tensorforge::fmacdpp16<2>(v341_acc, v352_lin, v333_data);
          tensorforge::fmacdpp16<3>(v341_acc, v352_lin, v334_data);
          tensorforge::fmacdpp16<4>(v342_acc, v352_lin, v323_data);
          tensorforge::fmacdpp16<5>(v342_acc, v352_lin, v324_data);
          tensorforge::fmacdpp16<6>(v342_acc, v352_lin, v325_data);
          tensorforge::fmacdpp16<7>(v342_acc, v352_lin, v326_data);
          tensorforge::fmacdpp16<8>(v342_acc, v352_lin, v327_data);
          tensorforge::fmacdpp16<9>(v342_acc, v352_lin, v328_data);
          tensorforge::fmacdpp16<10>(v342_acc, v352_lin, v329_data);
          tensorforge::fmacdpp16<11>(v342_acc, v352_lin, v330_data);
          tensorforge::fmacdpp16<12>(v342_acc, v352_lin, v331_data);
          tensorforge::fmacdpp16<13>(v342_acc, v352_lin, v332_data);
          tensorforge::fmacdpp16<14>(v342_acc, v352_lin, v333_data);
          tensorforge::fmacdpp16<15>(v342_acc, v352_lin, v334_data);
          float v353_lin = s0[96 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v343_acc, v353_lin, v323_data);
          tensorforge::fmacdpp16<1>(v343_acc, v353_lin, v324_data);
          tensorforge::fmacdpp16<2>(v343_acc, v353_lin, v325_data);
          tensorforge::fmacdpp16<3>(v343_acc, v353_lin, v326_data);
          tensorforge::fmacdpp16<4>(v343_acc, v353_lin, v327_data);
          tensorforge::fmacdpp16<5>(v343_acc, v353_lin, v328_data);
          tensorforge::fmacdpp16<6>(v343_acc, v353_lin, v329_data);
          tensorforge::fmacdpp16<7>(v343_acc, v353_lin, v330_data);
          tensorforge::fmacdpp16<8>(v343_acc, v353_lin, v331_data);
          tensorforge::fmacdpp16<9>(v343_acc, v353_lin, v332_data);
          tensorforge::fmacdpp16<10>(v343_acc, v353_lin, v333_data);
          tensorforge::fmacdpp16<11>(v343_acc, v353_lin, v334_data);
          tensorforge::fmacdpp16<12>(v344_acc, v353_lin, v323_data);
          tensorforge::fmacdpp16<13>(v344_acc, v353_lin, v324_data);
          tensorforge::fmacdpp16<14>(v344_acc, v353_lin, v325_data);
          tensorforge::fmacdpp16<15>(v344_acc, v353_lin, v326_data);
          float v354_lin = s0[112 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v344_acc, v354_lin, v327_data);
          tensorforge::fmacdpp16<1>(v344_acc, v354_lin, v328_data);
          tensorforge::fmacdpp16<2>(v344_acc, v354_lin, v329_data);
          tensorforge::fmacdpp16<3>(v344_acc, v354_lin, v330_data);
          tensorforge::fmacdpp16<4>(v344_acc, v354_lin, v331_data);
          tensorforge::fmacdpp16<5>(v344_acc, v354_lin, v332_data);
          tensorforge::fmacdpp16<6>(v344_acc, v354_lin, v333_data);
          tensorforge::fmacdpp16<7>(v344_acc, v354_lin, v334_data);
          tensorforge::fmacdpp16<8>(v345_acc, v354_lin, v323_data);
          tensorforge::fmacdpp16<9>(v345_acc, v354_lin, v324_data);
          tensorforge::fmacdpp16<10>(v345_acc, v354_lin, v325_data);
          tensorforge::fmacdpp16<11>(v345_acc, v354_lin, v326_data);
          tensorforge::fmacdpp16<12>(v345_acc, v354_lin, v327_data);
          tensorforge::fmacdpp16<13>(v345_acc, v354_lin, v328_data);
          tensorforge::fmacdpp16<14>(v345_acc, v354_lin, v329_data);
          tensorforge::fmacdpp16<15>(v345_acc, v354_lin, v330_data);
          float v355_lin = s0[128 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v345_acc, v355_lin, v331_data);
          tensorforge::fmacdpp16<1>(v345_acc, v355_lin, v332_data);
          tensorforge::fmacdpp16<2>(v345_acc, v355_lin, v333_data);
          tensorforge::fmacdpp16<3>(v345_acc, v355_lin, v334_data);
          tensorforge::fmacdpp16<4>(v346_acc, v355_lin, v323_data);
          tensorforge::fmacdpp16<5>(v346_acc, v355_lin, v324_data);
          tensorforge::fmacdpp16<6>(v346_acc, v355_lin, v325_data);
          tensorforge::fmacdpp16<7>(v346_acc, v355_lin, v326_data);
          tensorforge::fmacdpp16<8>(v346_acc, v355_lin, v327_data);
          tensorforge::fmacdpp16<9>(v346_acc, v355_lin, v328_data);
          tensorforge::fmacdpp16<10>(v346_acc, v355_lin, v329_data);
          tensorforge::fmacdpp16<11>(v346_acc, v355_lin, v330_data);
          tensorforge::fmacdpp16<12>(v346_acc, v355_lin, v331_data);
          tensorforge::fmacdpp16<13>(v346_acc, v355_lin, v332_data);
          tensorforge::fmacdpp16<14>(v346_acc, v355_lin, v333_data);
          tensorforge::fmacdpp16<15>(v346_acc, v355_lin, v334_data);
          ir6[0] = v335_acc;
          ir6[1] = v336_acc;
          ir6[2] = v337_acc;
          ir6[3] = v338_acc;
          ir6[4] = v339_acc;
          ir6[5] = v340_acc;
          ir6[6] = v341_acc;
          ir6[7] = v342_acc;
          ir6[8] = v343_acc;
          ir6[9] = v344_acc;
          ir6[10] = v345_acc;
          ir6[11] = v346_acc;
          // glb_m3 = store{r>g}(r6);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v360_i1 = 0; v360_i1 < 12; ++v360_i1) {
              int32_t v361_a = 0 + v360_i1;
              float v363_data = r6[v360_i1];
              int32_t v370_a = v3_lead + (v360_i1 * 12);
              glb_m3[v370_a] = v363_data;
            }
          }
          ;
        }
      }
    }
  }
}

