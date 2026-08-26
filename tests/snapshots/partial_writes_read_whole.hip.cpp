// === base name ===
kernel_7ab185b978

// === header ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7ab185b978, block.x * block.y * block.z, 2560 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_7ab185b978), hipFuncAttributeMaxDynamicSharedMemorySize, 2560 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_7ab185b978, grid, block, 2560 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×9(32×9) {0..32}×{0..9} pointer_based
    // m1 16×9(16×9) {0..16}×{0..9} pointer_based
    // m2 16×9(16×9) {0..16}×{0..9} pointer_based
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based
    // m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[320 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[320];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[batchId0][0 + m1_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          auto glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[batchId0][0 + m3_extraOffset];
          auto glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v2_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 32;
            int32_t v9_lead = v2_lead + v8_lead;
            int32_t v16_lead = v2_lead + v8_lead;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 9; ++v4_i1) {
              int32_t v10_a = v4_i1 * 32;
              int32_t v11_a = v9_lead + v10_a;
              float v19_data = __builtin_nontemporal_load(&glb_m0[(v16_lead + v10_a)]);
              int32_t v20_a = v3_i0 + v4_i1;
              r0[v20_a] = v19_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v2_lead < 16) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
              int32_t v31_a = v25_i1 * 16;
              int32_t v32_a = v2_lead + v31_a;
              float v40_data = __builtin_nontemporal_load(&glb_m1[(v2_lead + v31_a)]);
              int32_t v41_a = 0 + v25_i1;
              r2[v41_a] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v45_data = r0[0];
          float v46_data = ir1[0];
          ir1[0] = (v46_data + v45_data);
          float v48_data = r0[1];
          float v49_data = ir1[1];
          ir1[1] = (v49_data + v48_data);
          float v51_data = r0[2];
          float v52_data = ir1[2];
          ir1[2] = (v52_data + v51_data);
          float v54_data = r0[3];
          float v55_data = ir1[3];
          ir1[3] = (v55_data + v54_data);
          float v57_data = r0[4];
          float v58_data = ir1[4];
          ir1[4] = (v58_data + v57_data);
          float v60_data = r0[5];
          float v61_data = ir1[5];
          ir1[5] = (v61_data + v60_data);
          float v63_data = r0[6];
          float v64_data = ir1[6];
          ir1[6] = (v64_data + v63_data);
          float v66_data = r0[7];
          float v67_data = ir1[7];
          ir1[7] = (v67_data + v66_data);
          float v69_data = r0[8];
          float v70_data = ir1[8];
          ir1[8] = (v70_data + v69_data);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v75_i0 = 0; v75_i0 < 1; ++v75_i0) {
            int32_t v84_lead = v2_lead + (v75_i0 * 32);
            #pragma unroll
            for (int32_t v76_i1 = 0; v76_i1 < 9; ++v76_i1) {
              int32_t v77_a = v75_i0 + v76_i1;
              float v79_data = r1[(v75_i0 + v76_i1)];
              int32_t v86_a = v84_lead + (v76_i1 * 32);
              s0[v86_a] = v79_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v2_lead < 16) {
            #pragma unroll
            for (int32_t v91_i1 = 0; v91_i1 < 9; ++v91_i1) {
              int32_t v97_a = v91_i1 * 16;
              int32_t v98_a = v2_lead + v97_a;
              float v106_data = __builtin_nontemporal_load(&glb_m2[(v2_lead + v97_a)]);
              int32_t v107_a = 0 + v91_i1;
              r4[v107_a] = v106_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          auto& ir3 = r3;
          if (v2_lead < 16) {
            float v112_data = r2[0];
            float v113_data = ir3[0];
            ir3[0] = (v113_data + v112_data);
            float v115_data = r2[1];
            float v116_data = ir3[1];
            ir3[1] = (v116_data + v115_data);
            float v118_data = r2[2];
            float v119_data = ir3[2];
            ir3[2] = (v119_data + v118_data);
            float v121_data = r2[3];
            float v122_data = ir3[3];
            ir3[3] = (v122_data + v121_data);
            float v124_data = r2[4];
            float v125_data = ir3[4];
            ir3[4] = (v125_data + v124_data);
            float v127_data = r2[5];
            float v128_data = ir3[5];
            ir3[5] = (v128_data + v127_data);
            float v130_data = r2[6];
            float v131_data = ir3[6];
            ir3[6] = (v131_data + v130_data);
            float v133_data = r2[7];
            float v134_data = ir3[7];
            ir3[7] = (v134_data + v133_data);
            float v136_data = r2[8];
            float v137_data = ir3[8];
            ir3[8] = (v137_data + v136_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v2_lead < 16) {
            #pragma unroll
            for (int32_t v143_i1 = 0; v143_i1 < 9; ++v143_i1) {
              int32_t v144_a = 0 + v143_i1;
              float v146_data = r3[v143_i1];
              int32_t v153_a = v2_lead + (v143_i1 * 32);
              s0[v153_a] = v146_data;
            }
          }
          float r6[9]{};
          {
            // r6 = load{g>r}(glb_m4);
            float v0 = glb_m4[0 + threadIdx.x * 1];
            r6[0] = v0;
            float v32 = glb_m4[32 + threadIdx.x * 1];
            r6[1] = v32;
            float v64 = glb_m4[64 + threadIdx.x * 1];
            r6[2] = v64;
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          auto& ir5 = r5;
          if (v2_lead < 16) {
            float v158_data = r4[0];
            float v159_data = ir5[0];
            ir5[0] = (v159_data + v158_data);
            float v161_data = r4[1];
            float v162_data = ir5[1];
            ir5[1] = (v162_data + v161_data);
            float v164_data = r4[2];
            float v165_data = ir5[2];
            ir5[2] = (v165_data + v164_data);
            float v167_data = r4[3];
            float v168_data = ir5[3];
            ir5[3] = (v168_data + v167_data);
            float v170_data = r4[4];
            float v171_data = ir5[4];
            ir5[4] = (v171_data + v170_data);
            float v173_data = r4[5];
            float v174_data = ir5[5];
            ir5[5] = (v174_data + v173_data);
            float v176_data = r4[6];
            float v177_data = ir5[6];
            ir5[6] = (v177_data + v176_data);
            float v179_data = r4[7];
            float v180_data = ir5[7];
            ir5[7] = (v180_data + v179_data);
            float v182_data = r4[8];
            float v183_data = ir5[8];
            ir5[8] = (v183_data + v182_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v2_lead < 16) {
            #pragma unroll
            for (int32_t v189_i1 = 0; v189_i1 < 9; ++v189_i1) {
              int32_t v190_a = 0 + v189_i1;
              float v192_data = r5[v189_i1];
              int32_t v199_a = v2_lead + (v189_i1 * 32);
              s0[v199_a] = v192_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          ;
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          auto& ir7 = r7;
          int32_t v205_a = v2_lead + 0;
          float v212_data = s0[v2_lead];
          int32_t v218_a = v2_lead + 32;
          float v225_data = s0[(v2_lead + 32)];
          int32_t v231_a = v2_lead + 64;
          float v238_data = s0[(v2_lead + 64)];
          int32_t v244_a = v2_lead + 96;
          float v251_data = s0[(v2_lead + 96)];
          int32_t v257_a = v2_lead + 128;
          float v264_data = s0[(v2_lead + 128)];
          int32_t v270_a = v2_lead + 160;
          float v277_data = s0[(v2_lead + 160)];
          int32_t v283_a = v2_lead + 192;
          float v290_data = s0[(v2_lead + 192)];
          int32_t v296_a = v2_lead + 224;
          float v303_data = s0[(v2_lead + 224)];
          int32_t v309_a = v2_lead + 256;
          float v316_data = s0[(v2_lead + 256)];
          float v317_acc{};
          float v318_acc{};
          float v319_acc{};
          float v320_acc{};
          float v321_acc{};
          float v322_acc{};
          float v323_acc{};
          float v324_acc{};
          float v325_acc{};
          float v326_lin = r6[0];
          float v327_bc = tensorforge::broadcast<32, 16, 0>(v326_lin);
          tensorforge::fmacdpp16<0>(v317_acc, v327_bc, v212_data);
          tensorforge::fmacdpp16<1>(v317_acc, v327_bc, v225_data);
          tensorforge::fmacdpp16<2>(v317_acc, v327_bc, v238_data);
          tensorforge::fmacdpp16<3>(v317_acc, v327_bc, v251_data);
          tensorforge::fmacdpp16<4>(v317_acc, v327_bc, v264_data);
          tensorforge::fmacdpp16<5>(v317_acc, v327_bc, v277_data);
          tensorforge::fmacdpp16<6>(v317_acc, v327_bc, v290_data);
          tensorforge::fmacdpp16<7>(v317_acc, v327_bc, v303_data);
          tensorforge::fmacdpp16<8>(v317_acc, v327_bc, v316_data);
          tensorforge::fmacdpp16<9>(v318_acc, v327_bc, v212_data);
          tensorforge::fmacdpp16<10>(v318_acc, v327_bc, v225_data);
          tensorforge::fmacdpp16<11>(v318_acc, v327_bc, v238_data);
          tensorforge::fmacdpp16<12>(v318_acc, v327_bc, v251_data);
          tensorforge::fmacdpp16<13>(v318_acc, v327_bc, v264_data);
          tensorforge::fmacdpp16<14>(v318_acc, v327_bc, v277_data);
          tensorforge::fmacdpp16<15>(v318_acc, v327_bc, v290_data);
          float v328_bc = tensorforge::broadcast<32, 16, 1>(v326_lin);
          tensorforge::fmacdpp16<0>(v318_acc, v328_bc, v303_data);
          tensorforge::fmacdpp16<1>(v318_acc, v328_bc, v316_data);
          tensorforge::fmacdpp16<2>(v319_acc, v328_bc, v212_data);
          tensorforge::fmacdpp16<3>(v319_acc, v328_bc, v225_data);
          tensorforge::fmacdpp16<4>(v319_acc, v328_bc, v238_data);
          tensorforge::fmacdpp16<5>(v319_acc, v328_bc, v251_data);
          tensorforge::fmacdpp16<6>(v319_acc, v328_bc, v264_data);
          tensorforge::fmacdpp16<7>(v319_acc, v328_bc, v277_data);
          tensorforge::fmacdpp16<8>(v319_acc, v328_bc, v290_data);
          tensorforge::fmacdpp16<9>(v319_acc, v328_bc, v303_data);
          tensorforge::fmacdpp16<10>(v319_acc, v328_bc, v316_data);
          tensorforge::fmacdpp16<11>(v320_acc, v328_bc, v212_data);
          tensorforge::fmacdpp16<12>(v320_acc, v328_bc, v225_data);
          tensorforge::fmacdpp16<13>(v320_acc, v328_bc, v238_data);
          tensorforge::fmacdpp16<14>(v320_acc, v328_bc, v251_data);
          tensorforge::fmacdpp16<15>(v320_acc, v328_bc, v264_data);
          float v329_lin = r6[1];
          float v330_bc = tensorforge::broadcast<32, 16, 0>(v329_lin);
          tensorforge::fmacdpp16<0>(v320_acc, v330_bc, v277_data);
          tensorforge::fmacdpp16<1>(v320_acc, v330_bc, v290_data);
          tensorforge::fmacdpp16<2>(v320_acc, v330_bc, v303_data);
          tensorforge::fmacdpp16<3>(v320_acc, v330_bc, v316_data);
          tensorforge::fmacdpp16<4>(v321_acc, v330_bc, v212_data);
          tensorforge::fmacdpp16<5>(v321_acc, v330_bc, v225_data);
          tensorforge::fmacdpp16<6>(v321_acc, v330_bc, v238_data);
          tensorforge::fmacdpp16<7>(v321_acc, v330_bc, v251_data);
          tensorforge::fmacdpp16<8>(v321_acc, v330_bc, v264_data);
          tensorforge::fmacdpp16<9>(v321_acc, v330_bc, v277_data);
          tensorforge::fmacdpp16<10>(v321_acc, v330_bc, v290_data);
          tensorforge::fmacdpp16<11>(v321_acc, v330_bc, v303_data);
          tensorforge::fmacdpp16<12>(v321_acc, v330_bc, v316_data);
          tensorforge::fmacdpp16<13>(v322_acc, v330_bc, v212_data);
          tensorforge::fmacdpp16<14>(v322_acc, v330_bc, v225_data);
          tensorforge::fmacdpp16<15>(v322_acc, v330_bc, v238_data);
          float v331_bc = tensorforge::broadcast<32, 16, 1>(v329_lin);
          tensorforge::fmacdpp16<0>(v322_acc, v331_bc, v251_data);
          tensorforge::fmacdpp16<1>(v322_acc, v331_bc, v264_data);
          tensorforge::fmacdpp16<2>(v322_acc, v331_bc, v277_data);
          tensorforge::fmacdpp16<3>(v322_acc, v331_bc, v290_data);
          tensorforge::fmacdpp16<4>(v322_acc, v331_bc, v303_data);
          tensorforge::fmacdpp16<5>(v322_acc, v331_bc, v316_data);
          tensorforge::fmacdpp16<6>(v323_acc, v331_bc, v212_data);
          tensorforge::fmacdpp16<7>(v323_acc, v331_bc, v225_data);
          tensorforge::fmacdpp16<8>(v323_acc, v331_bc, v238_data);
          tensorforge::fmacdpp16<9>(v323_acc, v331_bc, v251_data);
          tensorforge::fmacdpp16<10>(v323_acc, v331_bc, v264_data);
          tensorforge::fmacdpp16<11>(v323_acc, v331_bc, v277_data);
          tensorforge::fmacdpp16<12>(v323_acc, v331_bc, v290_data);
          tensorforge::fmacdpp16<13>(v323_acc, v331_bc, v303_data);
          tensorforge::fmacdpp16<14>(v323_acc, v331_bc, v316_data);
          tensorforge::fmacdpp16<15>(v324_acc, v331_bc, v212_data);
          float v332_lin = r6[2];
          float v333_bc = tensorforge::broadcast<32, 16, 0>(v332_lin);
          tensorforge::fmacdpp16<0>(v324_acc, v333_bc, v225_data);
          tensorforge::fmacdpp16<1>(v324_acc, v333_bc, v238_data);
          tensorforge::fmacdpp16<2>(v324_acc, v333_bc, v251_data);
          tensorforge::fmacdpp16<3>(v324_acc, v333_bc, v264_data);
          tensorforge::fmacdpp16<4>(v324_acc, v333_bc, v277_data);
          tensorforge::fmacdpp16<5>(v324_acc, v333_bc, v290_data);
          tensorforge::fmacdpp16<6>(v324_acc, v333_bc, v303_data);
          tensorforge::fmacdpp16<7>(v324_acc, v333_bc, v316_data);
          tensorforge::fmacdpp16<8>(v325_acc, v333_bc, v212_data);
          tensorforge::fmacdpp16<9>(v325_acc, v333_bc, v225_data);
          tensorforge::fmacdpp16<10>(v325_acc, v333_bc, v238_data);
          tensorforge::fmacdpp16<11>(v325_acc, v333_bc, v251_data);
          tensorforge::fmacdpp16<12>(v325_acc, v333_bc, v264_data);
          tensorforge::fmacdpp16<13>(v325_acc, v333_bc, v277_data);
          tensorforge::fmacdpp16<14>(v325_acc, v333_bc, v290_data);
          tensorforge::fmacdpp16<15>(v325_acc, v333_bc, v303_data);
          tensorforge::fmacdpp16<0>(v325_acc, (tensorforge::broadcast<32, 16, 1>(v332_lin)), v316_data);
          ir7[0] = v317_acc;
          ir7[1] = v318_acc;
          ir7[2] = v319_acc;
          ir7[3] = v320_acc;
          ir7[4] = v321_acc;
          ir7[5] = v322_acc;
          ir7[6] = v323_acc;
          ir7[7] = v324_acc;
          ir7[8] = v325_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v338_i0 = 0; v338_i0 < 1; ++v338_i0) {
            int32_t v347_lead = v2_lead + (v338_i0 * 32);
            #pragma unroll
            for (int32_t v339_i1 = 0; v339_i1 < 9; ++v339_i1) {
              int32_t v340_a = v338_i0 + v339_i1;
              float v342_data = r7[(v338_i0 + v339_i1)];
              int32_t v349_a = v347_lead + (v339_i1 * 32);
              glb_m3[v349_a] = v342_data;
            }
          }
          ;
        }
      }
    }
  }
}

