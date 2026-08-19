// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08703cce1d, block.x * block.y * block.z, 256 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_08703cce1d), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_08703cce1d, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×32(12×6) {0..12}×{0..6} strided
    // m1 32×32(6×6) {0..6}×{0..6} strided
    // m2 32×32(12×6) {0..12}×{0..6} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
    // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 6; ++v4_i1) {
              int32_t v11_a = v2_lead + (v4_i1 * 12);
              float v12_data;
              {
                v12_data = __builtin_nontemporal_load(&glb_m0[v11_a]);
              }
              int32_t v13_a = 0 + v4_i1;
              r0[v13_a] = v12_data;
            }
          }
          float r1[6]{};
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
          // r3 = load{g>r}(glb_m3);
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              int32_t v25_a = v16_lead + (v18_i1 * 12);
              float v26_data;
              {
                v26_data = __builtin_nontemporal_load(&glb_m3[v25_a]);
              }
              int32_t v27_a = 0 + v18_i1;
              r3[v27_a] = v26_data;
            }
          }
          // wait(r1 = load{g>r}(glb_m1););
          float r2[6]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          auto& ir2 = r2;
          float v28_data = r1[0];
          float v29_data = r1[1];
          float v30_data = r1[2];
          float v31_data = r1[3];
          float v32_tp{};
          float v33_tp{};
          float v34_tp{};
          float v35_tp{};
          tensorforge::transpose4x4b32(v32_tp, v33_tp, v34_tp, v35_tp, v28_data, v29_data, v30_data, v31_data);
          tensorforge::VectorT<float, 4> v36_acc{};
          float v37_data = r0[0];
          float v38_data = r0[1];
          float v39_data = r0[2];
          float v40_data = r0[3];
          tensorforge::VectorT<float, 4> v41_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v37_data, v36_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v42_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v38_data, v41_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v43_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v34_tp, v39_data, v42_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v44_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v40_data, v43_acc, 2, 0, 0);
          float v45_data = r0[4];
          float v46_data = r0[5];
          tensorforge::VectorT<float, 4> v49_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v32_tp, v45_data, v44_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v50_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v33_tp, v46_data, v49_acc, 2, 1, 0);
          ir2[0] = (v50_acc[0]);
          ir2[1] = (v50_acc[1]);
          ir2[2] = (v50_acc[2]);
          ir2[3] = (v50_acc[3]);
          float v55_data = r1[4];
          float v56_data = r1[5];
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          tensorforge::transpose4x4b32(v59_tp, v60_tp, v61_tp, v62_tp, v55_data, v56_data, 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v63_acc{};
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v37_data, v63_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v38_data, v68_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v39_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v40_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, v45_data, v71_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, v46_data, v76_acc, 2, 1, 0);
          ir2[4] = (v77_acc[0]);
          ir2[5] = (v77_acc[1]);
          // wait(r3 = load{g>r}(glb_m3););
          float r4[6]{};
          // r4 = +(r3 * r2) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          auto& ir4 = r4;
          float v80_data = r3[0];
          float v81_data = r3[1];
          float v82_data = r3[2];
          float v83_data = r3[3];
          float v84_data = r3[4];
          float v85_data = r3[5];
          float v86_data = r3[6];
          float v87_data = r3[7];
          float v88_data = r3[8];
          float v89_data = r3[9];
          float v90_data = r3[10];
          float v91_data = r3[11];
          float v92_acc{};
          float v93_acc{};
          float v94_acc{};
          float v95_acc{};
          float v96_acc{};
          float v97_acc{};
          float v98_lin = r2[0];
          tensorforge::fmacdpp16<0>(v92_acc, v98_lin, v80_data);
          tensorforge::fmacdpp16<1>(v92_acc, v98_lin, v81_data);
          tensorforge::fmacdpp16<2>(v92_acc, v98_lin, v82_data);
          tensorforge::fmacdpp16<3>(v92_acc, v98_lin, v83_data);
          tensorforge::fmacdpp16<4>(v92_acc, v98_lin, v84_data);
          tensorforge::fmacdpp16<5>(v92_acc, v98_lin, v85_data);
          tensorforge::fmacdpp16<6>(v92_acc, v98_lin, v86_data);
          tensorforge::fmacdpp16<7>(v92_acc, v98_lin, v87_data);
          tensorforge::fmacdpp16<8>(v92_acc, v98_lin, v88_data);
          tensorforge::fmacdpp16<9>(v92_acc, v98_lin, v89_data);
          tensorforge::fmacdpp16<10>(v92_acc, v98_lin, v90_data);
          tensorforge::fmacdpp16<11>(v92_acc, v98_lin, v91_data);
          tensorforge::fmacdpp16<12>(v93_acc, v98_lin, v80_data);
          tensorforge::fmacdpp16<13>(v93_acc, v98_lin, v81_data);
          tensorforge::fmacdpp16<14>(v93_acc, v98_lin, v82_data);
          tensorforge::fmacdpp16<15>(v93_acc, v98_lin, v83_data);
          float v99_lin = r2[1];
          tensorforge::fmacdpp16<0>(v93_acc, v99_lin, v84_data);
          tensorforge::fmacdpp16<1>(v93_acc, v99_lin, v85_data);
          tensorforge::fmacdpp16<2>(v93_acc, v99_lin, v86_data);
          tensorforge::fmacdpp16<3>(v93_acc, v99_lin, v87_data);
          tensorforge::fmacdpp16<4>(v93_acc, v99_lin, v88_data);
          tensorforge::fmacdpp16<5>(v93_acc, v99_lin, v89_data);
          tensorforge::fmacdpp16<6>(v93_acc, v99_lin, v90_data);
          tensorforge::fmacdpp16<7>(v93_acc, v99_lin, v91_data);
          tensorforge::fmacdpp16<8>(v94_acc, v99_lin, v80_data);
          tensorforge::fmacdpp16<9>(v94_acc, v99_lin, v81_data);
          tensorforge::fmacdpp16<10>(v94_acc, v99_lin, v82_data);
          tensorforge::fmacdpp16<11>(v94_acc, v99_lin, v83_data);
          tensorforge::fmacdpp16<12>(v94_acc, v99_lin, v84_data);
          tensorforge::fmacdpp16<13>(v94_acc, v99_lin, v85_data);
          tensorforge::fmacdpp16<14>(v94_acc, v99_lin, v86_data);
          tensorforge::fmacdpp16<15>(v94_acc, v99_lin, v87_data);
          float v100_lin = r2[2];
          tensorforge::fmacdpp16<0>(v94_acc, v100_lin, v88_data);
          tensorforge::fmacdpp16<1>(v94_acc, v100_lin, v89_data);
          tensorforge::fmacdpp16<2>(v94_acc, v100_lin, v90_data);
          tensorforge::fmacdpp16<3>(v94_acc, v100_lin, v91_data);
          tensorforge::fmacdpp16<4>(v95_acc, v100_lin, v80_data);
          tensorforge::fmacdpp16<5>(v95_acc, v100_lin, v81_data);
          tensorforge::fmacdpp16<6>(v95_acc, v100_lin, v82_data);
          tensorforge::fmacdpp16<7>(v95_acc, v100_lin, v83_data);
          tensorforge::fmacdpp16<8>(v95_acc, v100_lin, v84_data);
          tensorforge::fmacdpp16<9>(v95_acc, v100_lin, v85_data);
          tensorforge::fmacdpp16<10>(v95_acc, v100_lin, v86_data);
          tensorforge::fmacdpp16<11>(v95_acc, v100_lin, v87_data);
          tensorforge::fmacdpp16<12>(v95_acc, v100_lin, v88_data);
          tensorforge::fmacdpp16<13>(v95_acc, v100_lin, v89_data);
          tensorforge::fmacdpp16<14>(v95_acc, v100_lin, v90_data);
          tensorforge::fmacdpp16<15>(v95_acc, v100_lin, v91_data);
          float v101_lin = r2[3];
          tensorforge::fmacdpp16<0>(v96_acc, v101_lin, v80_data);
          tensorforge::fmacdpp16<1>(v96_acc, v101_lin, v81_data);
          tensorforge::fmacdpp16<2>(v96_acc, v101_lin, v82_data);
          tensorforge::fmacdpp16<3>(v96_acc, v101_lin, v83_data);
          tensorforge::fmacdpp16<4>(v96_acc, v101_lin, v84_data);
          tensorforge::fmacdpp16<5>(v96_acc, v101_lin, v85_data);
          tensorforge::fmacdpp16<6>(v96_acc, v101_lin, v86_data);
          tensorforge::fmacdpp16<7>(v96_acc, v101_lin, v87_data);
          tensorforge::fmacdpp16<8>(v96_acc, v101_lin, v88_data);
          tensorforge::fmacdpp16<9>(v96_acc, v101_lin, v89_data);
          tensorforge::fmacdpp16<10>(v96_acc, v101_lin, v90_data);
          tensorforge::fmacdpp16<11>(v96_acc, v101_lin, v91_data);
          tensorforge::fmacdpp16<12>(v97_acc, v101_lin, v80_data);
          tensorforge::fmacdpp16<13>(v97_acc, v101_lin, v81_data);
          tensorforge::fmacdpp16<14>(v97_acc, v101_lin, v82_data);
          tensorforge::fmacdpp16<15>(v97_acc, v101_lin, v83_data);
          float v102_lin = r2[4];
          tensorforge::fmacdpp16<0>(v97_acc, v102_lin, v84_data);
          tensorforge::fmacdpp16<1>(v97_acc, v102_lin, v85_data);
          tensorforge::fmacdpp16<2>(v97_acc, v102_lin, v86_data);
          tensorforge::fmacdpp16<3>(v97_acc, v102_lin, v87_data);
          tensorforge::fmacdpp16<4>(v97_acc, v102_lin, v88_data);
          tensorforge::fmacdpp16<5>(v97_acc, v102_lin, v89_data);
          tensorforge::fmacdpp16<6>(v97_acc, v102_lin, v90_data);
          tensorforge::fmacdpp16<7>(v97_acc, v102_lin, v91_data);
          ir4[0] = v92_acc;
          ir4[1] = v93_acc;
          ir4[2] = v94_acc;
          ir4[3] = v95_acc;
          ir4[4] = v96_acc;
          ir4[5] = v97_acc;
          // glb_m2 = store{r>g}(r4);
          int32_t v105_lead = threadIdx.x % 16;
          if (v105_lead < 12) {
            #pragma unroll
            for (int32_t v107_i1 = 0; v107_i1 < 6; ++v107_i1) {
              int32_t v108_a = 0 + v107_i1;
              float v109_data = r4[v108_a];
              int32_t v116_a = v105_lead + (v107_i1 * 12);
              glb_m2[v116_a] = v109_data;
            }
          }
          ;
        }
      }
    }
  }
}

