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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 6) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 12; ++v4_i1) {
              int32_t v10_a = v4_i1 * 6;
              int32_t v11_a = v2_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m0[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
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
          if (v2_lead < 6) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              int32_t v31_a = v25_i1 * 6;
              int32_t v32_a = v2_lead + v31_a;
              float v40_data = __builtin_nontemporal_load(&glb_m2[(v2_lead + v31_a)]);
              int32_t v41_a = 0 + v25_i1;
              r3[v41_a] = v40_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = +(r0 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir2 = r2;
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
          tensorforge::VectorT<float, 4> v55_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v51_data, v50_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v56_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v52_data, v55_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v53_data, v56_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v58_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v54_data, v57_acc, 2, 0, 0);
          float v59_data = r0[4];
          float v60_data = r0[5];
          float v61_data = r0[6];
          float v62_data = r0[7];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v59_data, v58_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v64_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v60_data, v63_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v65_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v61_data, v64_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v66_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v62_data, v65_acc, 2, 1, 0);
          float v67_data = r0[8];
          float v68_data = r0[9];
          float v69_data = r0[10];
          float v70_data = r0[11];
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v67_data, v66_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v68_data, v71_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v69_data, v72_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v70_data, v73_acc, 2, 2, 0);
          ir2[0] = (v74_acc[0]);
          ir2[1] = (v74_acc[1]);
          ir2[2] = (v74_acc[2]);
          ir2[3] = (v74_acc[3]);
          float v79_data = r1[4];
          float v80_data = r1[5];
          float v81_data = r1[6];
          float v82_data = r1[7];
          float v83_tp{};
          float v84_tp{};
          float v85_tp{};
          float v86_tp{};
          tensorforge::transpose4x4b32(v83_tp, v84_tp, v85_tp, v86_tp, v79_data, v80_data, v81_data, v82_data);
          tensorforge::VectorT<float, 4> v87_acc{};
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v51_data, v87_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v52_data, v92_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v53_data, v93_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v54_data, v94_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v59_data, v95_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v60_data, v100_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v61_data, v101_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v62_data, v102_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v83_tp, v67_data, v103_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v84_tp, v68_data, v108_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v85_tp, v69_data, v109_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v86_tp, v70_data, v110_acc, 2, 2, 0);
          ir2[4] = (v111_acc[0]);
          ir2[5] = (v111_acc[1]);
          ir2[6] = (v111_acc[2]);
          ir2[7] = (v111_acc[3]);
          float v116_data = r1[8];
          float v117_data = r1[9];
          float v118_data = r1[10];
          float v119_data = r1[11];
          float v120_tp{};
          float v121_tp{};
          float v122_tp{};
          float v123_tp{};
          tensorforge::transpose4x4b32(v120_tp, v121_tp, v122_tp, v123_tp, v116_data, v117_data, v118_data, v119_data);
          tensorforge::VectorT<float, 4> v124_acc{};
          tensorforge::VectorT<float, 4> v129_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v51_data, v124_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v130_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v52_data, v129_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v53_data, v130_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v54_data, v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v59_data, v132_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v138_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v60_data, v137_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v61_data, v138_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v62_data, v139_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v67_data, v140_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v68_data, v145_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v122_tp, v69_data, v146_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v123_tp, v70_data, v147_acc, 2, 2, 0);
          ir2[8] = (v148_acc[0]);
          ir2[9] = (v148_acc[1]);
          ir2[10] = (v148_acc[2]);
          ir2[11] = (v148_acc[3]);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r2);
          if (v2_lead < 6) {
            #pragma unroll
            for (int32_t v157_i1 = 0; v157_i1 < 12; ++v157_i1) {
              int32_t v158_a = 0 + v157_i1;
              float v160_data = r2[v157_i1];
              int32_t v167_a = v2_lead + (v157_i1 * 12);
              s0[v167_a] = v160_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v172_i1 = 0; v172_i1 < 12; ++v172_i1) {
              int32_t v178_a = v172_i1 * 12;
              int32_t v179_a = v2_lead + v178_a;
              float v187_data = __builtin_nontemporal_load(&glb_m4[(v2_lead + v178_a)]);
              int32_t v188_a = 0 + v172_i1;
              r5[v188_a] = v187_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m2););
          float r4[12]{};
          // r4 = +(r3 * r1) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir4 = r4;
          float v189_data = r1[0];
          float v190_data = r1[1];
          float v191_data = r1[2];
          float v192_data = r1[3];
          float v193_tp{};
          float v194_tp{};
          float v195_tp{};
          float v196_tp{};
          tensorforge::transpose4x4b32(v193_tp, v194_tp, v195_tp, v196_tp, v189_data, v190_data, v191_data, v192_data);
          tensorforge::VectorT<float, 4> v197_acc{};
          float v198_data = r3[0];
          float v199_data = r3[1];
          float v200_data = r3[2];
          float v201_data = r3[3];
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v198_data, v197_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v199_data, v202_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v200_data, v203_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v201_data, v204_acc, 2, 0, 0);
          float v206_data = r3[4];
          float v207_data = r3[5];
          float v208_data = r3[6];
          float v209_data = r3[7];
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v206_data, v205_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v207_data, v210_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v208_data, v211_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v209_data, v212_acc, 2, 1, 0);
          float v214_data = r3[8];
          float v215_data = r3[9];
          float v216_data = r3[10];
          float v217_data = r3[11];
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v214_data, v213_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v215_data, v218_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v216_data, v219_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v217_data, v220_acc, 2, 2, 0);
          ir4[0] = (v221_acc[0]);
          ir4[1] = (v221_acc[1]);
          ir4[2] = (v221_acc[2]);
          ir4[3] = (v221_acc[3]);
          float v226_data = r1[4];
          float v227_data = r1[5];
          float v228_data = r1[6];
          float v229_data = r1[7];
          float v230_tp{};
          float v231_tp{};
          float v232_tp{};
          float v233_tp{};
          tensorforge::transpose4x4b32(v230_tp, v231_tp, v232_tp, v233_tp, v226_data, v227_data, v228_data, v229_data);
          tensorforge::VectorT<float, 4> v234_acc{};
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v198_data, v234_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v199_data, v239_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v200_data, v240_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v201_data, v241_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v206_data, v242_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v207_data, v247_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v208_data, v248_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v250_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v209_data, v249_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v230_tp, v214_data, v250_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v231_tp, v215_data, v255_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v232_tp, v216_data, v256_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v233_tp, v217_data, v257_acc, 2, 2, 0);
          ir4[4] = (v258_acc[0]);
          ir4[5] = (v258_acc[1]);
          ir4[6] = (v258_acc[2]);
          ir4[7] = (v258_acc[3]);
          float v263_data = r1[8];
          float v264_data = r1[9];
          float v265_data = r1[10];
          float v266_data = r1[11];
          float v267_tp{};
          float v268_tp{};
          float v269_tp{};
          float v270_tp{};
          tensorforge::transpose4x4b32(v267_tp, v268_tp, v269_tp, v270_tp, v263_data, v264_data, v265_data, v266_data);
          tensorforge::VectorT<float, 4> v271_acc{};
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v198_data, v271_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v199_data, v276_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v200_data, v277_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v201_data, v278_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v206_data, v279_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v207_data, v284_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v208_data, v285_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v209_data, v286_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v267_tp, v214_data, v287_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v215_data, v292_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v216_data, v293_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v217_data, v294_acc, 2, 2, 0);
          ir4[8] = (v295_acc[0]);
          ir4[9] = (v295_acc[1]);
          ir4[10] = (v295_acc[2]);
          ir4[11] = (v295_acc[3]);
          // s0 = store{r>s}(localShrMem0, r4);
          if (v2_lead < 6) {
            int32_t v313_off = v2_lead + 6;
            #pragma unroll
            for (int32_t v304_i1 = 0; v304_i1 < 12; ++v304_i1) {
              int32_t v305_a = 0 + v304_i1;
              float v307_data = r4[v304_i1];
              int32_t v315_a = v313_off + (v304_i1 * 12);
              s0[v315_a] = v307_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          ;
          // r6 = +(r5 * s0) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          auto& ir6 = r6;
          float v316_data = r5[0];
          float v317_data = r5[1];
          float v318_data = r5[2];
          float v319_data = r5[3];
          float v320_data = r5[4];
          float v321_data = r5[5];
          float v322_data = r5[6];
          float v323_data = r5[7];
          float v324_data = r5[8];
          float v325_data = r5[9];
          float v326_data = r5[10];
          float v327_data = r5[11];
          float v328_acc{};
          float v329_acc{};
          float v330_acc{};
          float v331_acc{};
          float v332_acc{};
          float v333_acc{};
          float v334_acc{};
          float v335_acc{};
          float v336_acc{};
          float v337_acc{};
          float v338_acc{};
          float v339_acc{};
          float v340_lin = s0[0 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v328_acc, v340_lin, v316_data);
          tensorforge::fmacdpp16<1>(v328_acc, v340_lin, v317_data);
          tensorforge::fmacdpp16<2>(v328_acc, v340_lin, v318_data);
          tensorforge::fmacdpp16<3>(v328_acc, v340_lin, v319_data);
          tensorforge::fmacdpp16<4>(v328_acc, v340_lin, v320_data);
          tensorforge::fmacdpp16<5>(v328_acc, v340_lin, v321_data);
          tensorforge::fmacdpp16<6>(v328_acc, v340_lin, v322_data);
          tensorforge::fmacdpp16<7>(v328_acc, v340_lin, v323_data);
          tensorforge::fmacdpp16<8>(v328_acc, v340_lin, v324_data);
          tensorforge::fmacdpp16<9>(v328_acc, v340_lin, v325_data);
          tensorforge::fmacdpp16<10>(v328_acc, v340_lin, v326_data);
          tensorforge::fmacdpp16<11>(v328_acc, v340_lin, v327_data);
          tensorforge::fmacdpp16<12>(v329_acc, v340_lin, v316_data);
          tensorforge::fmacdpp16<13>(v329_acc, v340_lin, v317_data);
          tensorforge::fmacdpp16<14>(v329_acc, v340_lin, v318_data);
          tensorforge::fmacdpp16<15>(v329_acc, v340_lin, v319_data);
          float v341_lin = s0[16 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v329_acc, v341_lin, v320_data);
          tensorforge::fmacdpp16<1>(v329_acc, v341_lin, v321_data);
          tensorforge::fmacdpp16<2>(v329_acc, v341_lin, v322_data);
          tensorforge::fmacdpp16<3>(v329_acc, v341_lin, v323_data);
          tensorforge::fmacdpp16<4>(v329_acc, v341_lin, v324_data);
          tensorforge::fmacdpp16<5>(v329_acc, v341_lin, v325_data);
          tensorforge::fmacdpp16<6>(v329_acc, v341_lin, v326_data);
          tensorforge::fmacdpp16<7>(v329_acc, v341_lin, v327_data);
          tensorforge::fmacdpp16<8>(v330_acc, v341_lin, v316_data);
          tensorforge::fmacdpp16<9>(v330_acc, v341_lin, v317_data);
          tensorforge::fmacdpp16<10>(v330_acc, v341_lin, v318_data);
          tensorforge::fmacdpp16<11>(v330_acc, v341_lin, v319_data);
          tensorforge::fmacdpp16<12>(v330_acc, v341_lin, v320_data);
          tensorforge::fmacdpp16<13>(v330_acc, v341_lin, v321_data);
          tensorforge::fmacdpp16<14>(v330_acc, v341_lin, v322_data);
          tensorforge::fmacdpp16<15>(v330_acc, v341_lin, v323_data);
          float v342_lin = s0[32 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v330_acc, v342_lin, v324_data);
          tensorforge::fmacdpp16<1>(v330_acc, v342_lin, v325_data);
          tensorforge::fmacdpp16<2>(v330_acc, v342_lin, v326_data);
          tensorforge::fmacdpp16<3>(v330_acc, v342_lin, v327_data);
          tensorforge::fmacdpp16<4>(v331_acc, v342_lin, v316_data);
          tensorforge::fmacdpp16<5>(v331_acc, v342_lin, v317_data);
          tensorforge::fmacdpp16<6>(v331_acc, v342_lin, v318_data);
          tensorforge::fmacdpp16<7>(v331_acc, v342_lin, v319_data);
          tensorforge::fmacdpp16<8>(v331_acc, v342_lin, v320_data);
          tensorforge::fmacdpp16<9>(v331_acc, v342_lin, v321_data);
          tensorforge::fmacdpp16<10>(v331_acc, v342_lin, v322_data);
          tensorforge::fmacdpp16<11>(v331_acc, v342_lin, v323_data);
          tensorforge::fmacdpp16<12>(v331_acc, v342_lin, v324_data);
          tensorforge::fmacdpp16<13>(v331_acc, v342_lin, v325_data);
          tensorforge::fmacdpp16<14>(v331_acc, v342_lin, v326_data);
          tensorforge::fmacdpp16<15>(v331_acc, v342_lin, v327_data);
          float v343_lin = s0[48 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v332_acc, v343_lin, v316_data);
          tensorforge::fmacdpp16<1>(v332_acc, v343_lin, v317_data);
          tensorforge::fmacdpp16<2>(v332_acc, v343_lin, v318_data);
          tensorforge::fmacdpp16<3>(v332_acc, v343_lin, v319_data);
          tensorforge::fmacdpp16<4>(v332_acc, v343_lin, v320_data);
          tensorforge::fmacdpp16<5>(v332_acc, v343_lin, v321_data);
          tensorforge::fmacdpp16<6>(v332_acc, v343_lin, v322_data);
          tensorforge::fmacdpp16<7>(v332_acc, v343_lin, v323_data);
          tensorforge::fmacdpp16<8>(v332_acc, v343_lin, v324_data);
          tensorforge::fmacdpp16<9>(v332_acc, v343_lin, v325_data);
          tensorforge::fmacdpp16<10>(v332_acc, v343_lin, v326_data);
          tensorforge::fmacdpp16<11>(v332_acc, v343_lin, v327_data);
          tensorforge::fmacdpp16<12>(v333_acc, v343_lin, v316_data);
          tensorforge::fmacdpp16<13>(v333_acc, v343_lin, v317_data);
          tensorforge::fmacdpp16<14>(v333_acc, v343_lin, v318_data);
          tensorforge::fmacdpp16<15>(v333_acc, v343_lin, v319_data);
          float v344_lin = s0[64 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v333_acc, v344_lin, v320_data);
          tensorforge::fmacdpp16<1>(v333_acc, v344_lin, v321_data);
          tensorforge::fmacdpp16<2>(v333_acc, v344_lin, v322_data);
          tensorforge::fmacdpp16<3>(v333_acc, v344_lin, v323_data);
          tensorforge::fmacdpp16<4>(v333_acc, v344_lin, v324_data);
          tensorforge::fmacdpp16<5>(v333_acc, v344_lin, v325_data);
          tensorforge::fmacdpp16<6>(v333_acc, v344_lin, v326_data);
          tensorforge::fmacdpp16<7>(v333_acc, v344_lin, v327_data);
          tensorforge::fmacdpp16<8>(v334_acc, v344_lin, v316_data);
          tensorforge::fmacdpp16<9>(v334_acc, v344_lin, v317_data);
          tensorforge::fmacdpp16<10>(v334_acc, v344_lin, v318_data);
          tensorforge::fmacdpp16<11>(v334_acc, v344_lin, v319_data);
          tensorforge::fmacdpp16<12>(v334_acc, v344_lin, v320_data);
          tensorforge::fmacdpp16<13>(v334_acc, v344_lin, v321_data);
          tensorforge::fmacdpp16<14>(v334_acc, v344_lin, v322_data);
          tensorforge::fmacdpp16<15>(v334_acc, v344_lin, v323_data);
          float v345_lin = s0[80 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v334_acc, v345_lin, v324_data);
          tensorforge::fmacdpp16<1>(v334_acc, v345_lin, v325_data);
          tensorforge::fmacdpp16<2>(v334_acc, v345_lin, v326_data);
          tensorforge::fmacdpp16<3>(v334_acc, v345_lin, v327_data);
          tensorforge::fmacdpp16<4>(v335_acc, v345_lin, v316_data);
          tensorforge::fmacdpp16<5>(v335_acc, v345_lin, v317_data);
          tensorforge::fmacdpp16<6>(v335_acc, v345_lin, v318_data);
          tensorforge::fmacdpp16<7>(v335_acc, v345_lin, v319_data);
          tensorforge::fmacdpp16<8>(v335_acc, v345_lin, v320_data);
          tensorforge::fmacdpp16<9>(v335_acc, v345_lin, v321_data);
          tensorforge::fmacdpp16<10>(v335_acc, v345_lin, v322_data);
          tensorforge::fmacdpp16<11>(v335_acc, v345_lin, v323_data);
          tensorforge::fmacdpp16<12>(v335_acc, v345_lin, v324_data);
          tensorforge::fmacdpp16<13>(v335_acc, v345_lin, v325_data);
          tensorforge::fmacdpp16<14>(v335_acc, v345_lin, v326_data);
          tensorforge::fmacdpp16<15>(v335_acc, v345_lin, v327_data);
          float v346_lin = s0[96 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v336_acc, v346_lin, v316_data);
          tensorforge::fmacdpp16<1>(v336_acc, v346_lin, v317_data);
          tensorforge::fmacdpp16<2>(v336_acc, v346_lin, v318_data);
          tensorforge::fmacdpp16<3>(v336_acc, v346_lin, v319_data);
          tensorforge::fmacdpp16<4>(v336_acc, v346_lin, v320_data);
          tensorforge::fmacdpp16<5>(v336_acc, v346_lin, v321_data);
          tensorforge::fmacdpp16<6>(v336_acc, v346_lin, v322_data);
          tensorforge::fmacdpp16<7>(v336_acc, v346_lin, v323_data);
          tensorforge::fmacdpp16<8>(v336_acc, v346_lin, v324_data);
          tensorforge::fmacdpp16<9>(v336_acc, v346_lin, v325_data);
          tensorforge::fmacdpp16<10>(v336_acc, v346_lin, v326_data);
          tensorforge::fmacdpp16<11>(v336_acc, v346_lin, v327_data);
          tensorforge::fmacdpp16<12>(v337_acc, v346_lin, v316_data);
          tensorforge::fmacdpp16<13>(v337_acc, v346_lin, v317_data);
          tensorforge::fmacdpp16<14>(v337_acc, v346_lin, v318_data);
          tensorforge::fmacdpp16<15>(v337_acc, v346_lin, v319_data);
          float v347_lin = s0[112 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v337_acc, v347_lin, v320_data);
          tensorforge::fmacdpp16<1>(v337_acc, v347_lin, v321_data);
          tensorforge::fmacdpp16<2>(v337_acc, v347_lin, v322_data);
          tensorforge::fmacdpp16<3>(v337_acc, v347_lin, v323_data);
          tensorforge::fmacdpp16<4>(v337_acc, v347_lin, v324_data);
          tensorforge::fmacdpp16<5>(v337_acc, v347_lin, v325_data);
          tensorforge::fmacdpp16<6>(v337_acc, v347_lin, v326_data);
          tensorforge::fmacdpp16<7>(v337_acc, v347_lin, v327_data);
          tensorforge::fmacdpp16<8>(v338_acc, v347_lin, v316_data);
          tensorforge::fmacdpp16<9>(v338_acc, v347_lin, v317_data);
          tensorforge::fmacdpp16<10>(v338_acc, v347_lin, v318_data);
          tensorforge::fmacdpp16<11>(v338_acc, v347_lin, v319_data);
          tensorforge::fmacdpp16<12>(v338_acc, v347_lin, v320_data);
          tensorforge::fmacdpp16<13>(v338_acc, v347_lin, v321_data);
          tensorforge::fmacdpp16<14>(v338_acc, v347_lin, v322_data);
          tensorforge::fmacdpp16<15>(v338_acc, v347_lin, v323_data);
          float v348_lin = s0[128 + threadIdx.x * 1];
          tensorforge::fmacdpp16<0>(v338_acc, v348_lin, v324_data);
          tensorforge::fmacdpp16<1>(v338_acc, v348_lin, v325_data);
          tensorforge::fmacdpp16<2>(v338_acc, v348_lin, v326_data);
          tensorforge::fmacdpp16<3>(v338_acc, v348_lin, v327_data);
          tensorforge::fmacdpp16<4>(v339_acc, v348_lin, v316_data);
          tensorforge::fmacdpp16<5>(v339_acc, v348_lin, v317_data);
          tensorforge::fmacdpp16<6>(v339_acc, v348_lin, v318_data);
          tensorforge::fmacdpp16<7>(v339_acc, v348_lin, v319_data);
          tensorforge::fmacdpp16<8>(v339_acc, v348_lin, v320_data);
          tensorforge::fmacdpp16<9>(v339_acc, v348_lin, v321_data);
          tensorforge::fmacdpp16<10>(v339_acc, v348_lin, v322_data);
          tensorforge::fmacdpp16<11>(v339_acc, v348_lin, v323_data);
          tensorforge::fmacdpp16<12>(v339_acc, v348_lin, v324_data);
          tensorforge::fmacdpp16<13>(v339_acc, v348_lin, v325_data);
          tensorforge::fmacdpp16<14>(v339_acc, v348_lin, v326_data);
          tensorforge::fmacdpp16<15>(v339_acc, v348_lin, v327_data);
          ir6[0] = v328_acc;
          ir6[1] = v329_acc;
          ir6[2] = v330_acc;
          ir6[3] = v331_acc;
          ir6[4] = v332_acc;
          ir6[5] = v333_acc;
          ir6[6] = v334_acc;
          ir6[7] = v335_acc;
          ir6[8] = v336_acc;
          ir6[9] = v337_acc;
          ir6[10] = v338_acc;
          ir6[11] = v339_acc;
          // glb_m3 = store{r>g}(r6);
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v353_i1 = 0; v353_i1 < 12; ++v353_i1) {
              int32_t v354_a = 0 + v353_i1;
              float v356_data = r6[v353_i1];
              int32_t v363_a = v2_lead + (v353_i1 * 12);
              glb_m3[v363_a] = v356_data;
            }
          }
          ;
        }
      }
    }
  }
}

