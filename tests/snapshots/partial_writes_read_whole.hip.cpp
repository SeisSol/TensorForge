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
          int32_t v3_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
            int32_t v9_lead = v4_i0 * 32;
            int32_t v10_lead = v3_lead + v9_lead;
            int32_t v17_lead = v3_lead + v9_lead;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 9; ++v5_i1) {
              int32_t v11_a = v5_i1 * 32;
              int32_t v12_a = v10_lead + v11_a;
              float v20_data = __builtin_nontemporal_load(&glb_m0[(v17_lead + v11_a)]);
              int32_t v21_a = v4_i0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
              int32_t v33_a = v27_i1 * 16;
              int32_t v34_a = v3_lead + v33_a;
              float v42_data = __builtin_nontemporal_load(&glb_m1[(v3_lead + v33_a)]);
              int32_t v43_a = 0 + v27_i1;
              r2[v43_a] = v42_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v48_data = r0[0];
          float v49_data = ir1[0];
          ir1[0] = (v49_data + v48_data);
          float v51_data = r0[1];
          float v52_data = ir1[1];
          ir1[1] = (v52_data + v51_data);
          float v54_data = r0[2];
          float v55_data = ir1[2];
          ir1[2] = (v55_data + v54_data);
          float v57_data = r0[3];
          float v58_data = ir1[3];
          ir1[3] = (v58_data + v57_data);
          float v60_data = r0[4];
          float v61_data = ir1[4];
          ir1[4] = (v61_data + v60_data);
          float v63_data = r0[5];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + v63_data);
          float v66_data = r0[6];
          float v67_data = ir1[6];
          ir1[6] = (v67_data + v66_data);
          float v69_data = r0[7];
          float v70_data = ir1[7];
          ir1[7] = (v70_data + v69_data);
          float v72_data = r0[8];
          float v73_data = ir1[8];
          ir1[8] = (v73_data + v72_data);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v78_i0 = 0; v78_i0 < 1; ++v78_i0) {
            int32_t v87_lead = v3_lead + (v78_i0 * 32);
            #pragma unroll
            for (int32_t v79_i1 = 0; v79_i1 < 9; ++v79_i1) {
              int32_t v80_a = v78_i0 + v79_i1;
              float v82_data = r1[(v78_i0 + v79_i1)];
              int32_t v89_a = v87_lead + (v79_i1 * 32);
              s0[v89_a] = v82_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v95_i1 = 0; v95_i1 < 9; ++v95_i1) {
              int32_t v101_a = v95_i1 * 16;
              int32_t v102_a = v3_lead + v101_a;
              float v110_data = __builtin_nontemporal_load(&glb_m2[(v3_lead + v101_a)]);
              int32_t v111_a = 0 + v95_i1;
              r4[v111_a] = v110_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          auto& ir3 = r3;
          if (v3_lead < 16) {
            float v117_data = r2[0];
            float v118_data = ir3[0];
            ir3[0] = (v118_data + v117_data);
            float v120_data = r2[1];
            float v121_data = ir3[1];
            ir3[1] = (v121_data + v120_data);
            float v123_data = r2[2];
            float v124_data = ir3[2];
            ir3[2] = (v124_data + v123_data);
            float v126_data = r2[3];
            float v127_data = ir3[3];
            ir3[3] = (v127_data + v126_data);
            float v129_data = r2[4];
            float v130_data = ir3[4];
            ir3[4] = (v130_data + v129_data);
            float v132_data = r2[5];
            float v133_data = ir3[5];
            ir3[5] = (v133_data + v132_data);
            float v135_data = r2[6];
            float v136_data = ir3[6];
            ir3[6] = (v136_data + v135_data);
            float v138_data = r2[7];
            float v139_data = ir3[7];
            ir3[7] = (v139_data + v138_data);
            float v141_data = r2[8];
            float v142_data = ir3[8];
            ir3[8] = (v142_data + v141_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v148_i1 = 0; v148_i1 < 9; ++v148_i1) {
              int32_t v149_a = 0 + v148_i1;
              float v151_data = r3[v148_i1];
              int32_t v158_a = v3_lead + (v148_i1 * 32);
              s0[v158_a] = v151_data;
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
          if (v3_lead < 16) {
            float v165_data = r4[0];
            float v166_data = ir5[0];
            ir5[0] = (v166_data + v165_data);
            float v168_data = r4[1];
            float v169_data = ir5[1];
            ir5[1] = (v169_data + v168_data);
            float v171_data = r4[2];
            float v172_data = ir5[2];
            ir5[2] = (v172_data + v171_data);
            float v174_data = r4[3];
            float v175_data = ir5[3];
            ir5[3] = (v175_data + v174_data);
            float v177_data = r4[4];
            float v178_data = ir5[4];
            ir5[4] = (v178_data + v177_data);
            float v180_data = r4[5];
            float v181_data = ir5[5];
            ir5[5] = (v181_data + v180_data);
            float v183_data = r4[6];
            float v184_data = ir5[6];
            ir5[6] = (v184_data + v183_data);
            float v186_data = r4[7];
            float v187_data = ir5[7];
            ir5[7] = (v187_data + v186_data);
            float v189_data = r4[8];
            float v190_data = ir5[8];
            ir5[8] = (v190_data + v189_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v3_lead < 16) {
            #pragma unroll
            for (int32_t v196_i1 = 0; v196_i1 < 9; ++v196_i1) {
              int32_t v197_a = 0 + v196_i1;
              float v199_data = r5[v196_i1];
              int32_t v206_a = v3_lead + (v196_i1 * 32);
              s0[v206_a] = v199_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          ;
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          auto& ir7 = r7;
          int32_t v213_a = v3_lead + 0;
          float v220_data = s0[v3_lead];
          int32_t v226_a = v3_lead + 32;
          float v233_data = s0[(v3_lead + 32)];
          int32_t v239_a = v3_lead + 64;
          float v246_data = s0[(v3_lead + 64)];
          int32_t v252_a = v3_lead + 96;
          float v259_data = s0[(v3_lead + 96)];
          int32_t v265_a = v3_lead + 128;
          float v272_data = s0[(v3_lead + 128)];
          int32_t v278_a = v3_lead + 160;
          float v285_data = s0[(v3_lead + 160)];
          int32_t v291_a = v3_lead + 192;
          float v298_data = s0[(v3_lead + 192)];
          int32_t v304_a = v3_lead + 224;
          float v311_data = s0[(v3_lead + 224)];
          int32_t v317_a = v3_lead + 256;
          float v324_data = s0[(v3_lead + 256)];
          float v325_acc{};
          float v326_acc{};
          float v327_acc{};
          float v328_acc{};
          float v329_acc{};
          float v330_acc{};
          float v331_acc{};
          float v332_acc{};
          float v333_acc{};
          float v334_lin = r6[0];
          float v335_bc = tensorforge::broadcast<32, 16, 0>(v334_lin);
          tensorforge::fmacdpp16<0>(v325_acc, v335_bc, v220_data);
          tensorforge::fmacdpp16<1>(v325_acc, v335_bc, v233_data);
          tensorforge::fmacdpp16<2>(v325_acc, v335_bc, v246_data);
          tensorforge::fmacdpp16<3>(v325_acc, v335_bc, v259_data);
          tensorforge::fmacdpp16<4>(v325_acc, v335_bc, v272_data);
          tensorforge::fmacdpp16<5>(v325_acc, v335_bc, v285_data);
          tensorforge::fmacdpp16<6>(v325_acc, v335_bc, v298_data);
          tensorforge::fmacdpp16<7>(v325_acc, v335_bc, v311_data);
          tensorforge::fmacdpp16<8>(v325_acc, v335_bc, v324_data);
          tensorforge::fmacdpp16<9>(v326_acc, v335_bc, v220_data);
          tensorforge::fmacdpp16<10>(v326_acc, v335_bc, v233_data);
          tensorforge::fmacdpp16<11>(v326_acc, v335_bc, v246_data);
          tensorforge::fmacdpp16<12>(v326_acc, v335_bc, v259_data);
          tensorforge::fmacdpp16<13>(v326_acc, v335_bc, v272_data);
          tensorforge::fmacdpp16<14>(v326_acc, v335_bc, v285_data);
          tensorforge::fmacdpp16<15>(v326_acc, v335_bc, v298_data);
          float v336_bc = tensorforge::broadcast<32, 16, 1>(v334_lin);
          tensorforge::fmacdpp16<0>(v326_acc, v336_bc, v311_data);
          tensorforge::fmacdpp16<1>(v326_acc, v336_bc, v324_data);
          tensorforge::fmacdpp16<2>(v327_acc, v336_bc, v220_data);
          tensorforge::fmacdpp16<3>(v327_acc, v336_bc, v233_data);
          tensorforge::fmacdpp16<4>(v327_acc, v336_bc, v246_data);
          tensorforge::fmacdpp16<5>(v327_acc, v336_bc, v259_data);
          tensorforge::fmacdpp16<6>(v327_acc, v336_bc, v272_data);
          tensorforge::fmacdpp16<7>(v327_acc, v336_bc, v285_data);
          tensorforge::fmacdpp16<8>(v327_acc, v336_bc, v298_data);
          tensorforge::fmacdpp16<9>(v327_acc, v336_bc, v311_data);
          tensorforge::fmacdpp16<10>(v327_acc, v336_bc, v324_data);
          tensorforge::fmacdpp16<11>(v328_acc, v336_bc, v220_data);
          tensorforge::fmacdpp16<12>(v328_acc, v336_bc, v233_data);
          tensorforge::fmacdpp16<13>(v328_acc, v336_bc, v246_data);
          tensorforge::fmacdpp16<14>(v328_acc, v336_bc, v259_data);
          tensorforge::fmacdpp16<15>(v328_acc, v336_bc, v272_data);
          float v337_lin = r6[1];
          float v338_bc = tensorforge::broadcast<32, 16, 0>(v337_lin);
          tensorforge::fmacdpp16<0>(v328_acc, v338_bc, v285_data);
          tensorforge::fmacdpp16<1>(v328_acc, v338_bc, v298_data);
          tensorforge::fmacdpp16<2>(v328_acc, v338_bc, v311_data);
          tensorforge::fmacdpp16<3>(v328_acc, v338_bc, v324_data);
          tensorforge::fmacdpp16<4>(v329_acc, v338_bc, v220_data);
          tensorforge::fmacdpp16<5>(v329_acc, v338_bc, v233_data);
          tensorforge::fmacdpp16<6>(v329_acc, v338_bc, v246_data);
          tensorforge::fmacdpp16<7>(v329_acc, v338_bc, v259_data);
          tensorforge::fmacdpp16<8>(v329_acc, v338_bc, v272_data);
          tensorforge::fmacdpp16<9>(v329_acc, v338_bc, v285_data);
          tensorforge::fmacdpp16<10>(v329_acc, v338_bc, v298_data);
          tensorforge::fmacdpp16<11>(v329_acc, v338_bc, v311_data);
          tensorforge::fmacdpp16<12>(v329_acc, v338_bc, v324_data);
          tensorforge::fmacdpp16<13>(v330_acc, v338_bc, v220_data);
          tensorforge::fmacdpp16<14>(v330_acc, v338_bc, v233_data);
          tensorforge::fmacdpp16<15>(v330_acc, v338_bc, v246_data);
          float v339_bc = tensorforge::broadcast<32, 16, 1>(v337_lin);
          tensorforge::fmacdpp16<0>(v330_acc, v339_bc, v259_data);
          tensorforge::fmacdpp16<1>(v330_acc, v339_bc, v272_data);
          tensorforge::fmacdpp16<2>(v330_acc, v339_bc, v285_data);
          tensorforge::fmacdpp16<3>(v330_acc, v339_bc, v298_data);
          tensorforge::fmacdpp16<4>(v330_acc, v339_bc, v311_data);
          tensorforge::fmacdpp16<5>(v330_acc, v339_bc, v324_data);
          tensorforge::fmacdpp16<6>(v331_acc, v339_bc, v220_data);
          tensorforge::fmacdpp16<7>(v331_acc, v339_bc, v233_data);
          tensorforge::fmacdpp16<8>(v331_acc, v339_bc, v246_data);
          tensorforge::fmacdpp16<9>(v331_acc, v339_bc, v259_data);
          tensorforge::fmacdpp16<10>(v331_acc, v339_bc, v272_data);
          tensorforge::fmacdpp16<11>(v331_acc, v339_bc, v285_data);
          tensorforge::fmacdpp16<12>(v331_acc, v339_bc, v298_data);
          tensorforge::fmacdpp16<13>(v331_acc, v339_bc, v311_data);
          tensorforge::fmacdpp16<14>(v331_acc, v339_bc, v324_data);
          tensorforge::fmacdpp16<15>(v332_acc, v339_bc, v220_data);
          float v340_lin = r6[2];
          float v341_bc = tensorforge::broadcast<32, 16, 0>(v340_lin);
          tensorforge::fmacdpp16<0>(v332_acc, v341_bc, v233_data);
          tensorforge::fmacdpp16<1>(v332_acc, v341_bc, v246_data);
          tensorforge::fmacdpp16<2>(v332_acc, v341_bc, v259_data);
          tensorforge::fmacdpp16<3>(v332_acc, v341_bc, v272_data);
          tensorforge::fmacdpp16<4>(v332_acc, v341_bc, v285_data);
          tensorforge::fmacdpp16<5>(v332_acc, v341_bc, v298_data);
          tensorforge::fmacdpp16<6>(v332_acc, v341_bc, v311_data);
          tensorforge::fmacdpp16<7>(v332_acc, v341_bc, v324_data);
          tensorforge::fmacdpp16<8>(v333_acc, v341_bc, v220_data);
          tensorforge::fmacdpp16<9>(v333_acc, v341_bc, v233_data);
          tensorforge::fmacdpp16<10>(v333_acc, v341_bc, v246_data);
          tensorforge::fmacdpp16<11>(v333_acc, v341_bc, v259_data);
          tensorforge::fmacdpp16<12>(v333_acc, v341_bc, v272_data);
          tensorforge::fmacdpp16<13>(v333_acc, v341_bc, v285_data);
          tensorforge::fmacdpp16<14>(v333_acc, v341_bc, v298_data);
          tensorforge::fmacdpp16<15>(v333_acc, v341_bc, v311_data);
          tensorforge::fmacdpp16<0>(v333_acc, (tensorforge::broadcast<32, 16, 1>(v340_lin)), v324_data);
          ir7[0] = v325_acc;
          ir7[1] = v326_acc;
          ir7[2] = v327_acc;
          ir7[3] = v328_acc;
          ir7[4] = v329_acc;
          ir7[5] = v330_acc;
          ir7[6] = v331_acc;
          ir7[7] = v332_acc;
          ir7[8] = v333_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v346_i0 = 0; v346_i0 < 1; ++v346_i0) {
            int32_t v355_lead = v3_lead + (v346_i0 * 32);
            #pragma unroll
            for (int32_t v347_i1 = 0; v347_i1 < 9; ++v347_i1) {
              int32_t v348_a = v346_i0 + v347_i1;
              float v350_data = r7[(v346_i0 + v347_i1)];
              int32_t v357_a = v355_lead + (v347_i1 * 32);
              glb_m3[v357_a] = v350_data;
            }
          }
          ;
        }
      }
    }
  }
}

