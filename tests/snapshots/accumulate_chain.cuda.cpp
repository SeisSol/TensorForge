// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8a03a3cd0d, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_8a03a3cd0d, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_8a03a3cd0d<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×12(12×12) {0..12}×{0..12} strided
    // m2 12×8(12×8) {0..12}×{0..8} strided
    // m3 12×12(12×12) {0..12}×{0..12} strided
    // m4 12×8(12×8) {0..12}×{0..8} strided
    // m5 12×12(12×12) {0..12}×{0..12} strided
    // m6 12×8(12×8) {0..12}×{0..8} strided
    // m7 12×12(12×12) {0..12}×{0..12} strided
    // m8 12×8(12×8) {0..12}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
          const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
          const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
              int32_t v24_a = v18_i1 * 12;
              int32_t v25_a = v16_lead + v24_a;
              float v33_data = __ldcg(&glb_m1[(v16_lead + v24_a)]);
              r0[v18_i1] = v33_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v42_i1 = 0; v42_i1 < 12; ++v42_i1) {
              int32_t v48_a = v42_i1 * 12;
              int32_t v49_a = v16_lead + v48_a;
              float v57_data = __ldcg(&glb_m3[(v16_lead + v48_a)]);
              r2[v42_i1] = v57_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v16_lead < 12) {
            float v65_data = r0[0];
            float v66_data = s0[0];
            float v68_data = ir1[0];
            ir1[0] = (v68_data + (v65_data * v66_data));
            float v71_data = s0[12];
            float v73_data = ir1[1];
            ir1[1] = (v73_data + (v65_data * v71_data));
            float v76_data = s0[24];
            float v78_data = ir1[2];
            ir1[2] = (v78_data + (v65_data * v76_data));
            float v81_data = s0[36];
            float v83_data = ir1[3];
            ir1[3] = (v83_data + (v65_data * v81_data));
            float v86_data = s0[48];
            float v88_data = ir1[4];
            ir1[4] = (v88_data + (v65_data * v86_data));
            float v91_data = s0[60];
            float v93_data = ir1[5];
            ir1[5] = (v93_data + (v65_data * v91_data));
            float v96_data = s0[72];
            float v98_data = ir1[6];
            ir1[6] = (v98_data + (v65_data * v96_data));
            float v101_data = s0[84];
            float v103_data = ir1[7];
            ir1[7] = (v103_data + (v65_data * v101_data));
          }
          if (v16_lead < 12) {
            float v109_data = r0[1];
            float v110_data = s0[1];
            float v112_data = ir1[0];
            ir1[0] = (v112_data + (v109_data * v110_data));
            float v115_data = s0[13];
            float v117_data = ir1[1];
            ir1[1] = (v117_data + (v109_data * v115_data));
            float v120_data = s0[25];
            float v122_data = ir1[2];
            ir1[2] = (v122_data + (v109_data * v120_data));
            float v125_data = s0[37];
            float v127_data = ir1[3];
            ir1[3] = (v127_data + (v109_data * v125_data));
            float v130_data = s0[49];
            float v132_data = ir1[4];
            ir1[4] = (v132_data + (v109_data * v130_data));
            float v135_data = s0[61];
            float v137_data = ir1[5];
            ir1[5] = (v137_data + (v109_data * v135_data));
            float v140_data = s0[73];
            float v142_data = ir1[6];
            ir1[6] = (v142_data + (v109_data * v140_data));
            float v145_data = s0[85];
            float v147_data = ir1[7];
            ir1[7] = (v147_data + (v109_data * v145_data));
          }
          if (v16_lead < 12) {
            float v153_data = r0[2];
            float v154_data = s0[2];
            float v156_data = ir1[0];
            ir1[0] = (v156_data + (v153_data * v154_data));
            float v159_data = s0[14];
            float v161_data = ir1[1];
            ir1[1] = (v161_data + (v153_data * v159_data));
            float v164_data = s0[26];
            float v166_data = ir1[2];
            ir1[2] = (v166_data + (v153_data * v164_data));
            float v169_data = s0[38];
            float v171_data = ir1[3];
            ir1[3] = (v171_data + (v153_data * v169_data));
            float v174_data = s0[50];
            float v176_data = ir1[4];
            ir1[4] = (v176_data + (v153_data * v174_data));
            float v179_data = s0[62];
            float v181_data = ir1[5];
            ir1[5] = (v181_data + (v153_data * v179_data));
            float v184_data = s0[74];
            float v186_data = ir1[6];
            ir1[6] = (v186_data + (v153_data * v184_data));
            float v189_data = s0[86];
            float v191_data = ir1[7];
            ir1[7] = (v191_data + (v153_data * v189_data));
          }
          if (v16_lead < 12) {
            float v197_data = r0[3];
            float v198_data = s0[3];
            float v200_data = ir1[0];
            ir1[0] = (v200_data + (v197_data * v198_data));
            float v203_data = s0[15];
            float v205_data = ir1[1];
            ir1[1] = (v205_data + (v197_data * v203_data));
            float v208_data = s0[27];
            float v210_data = ir1[2];
            ir1[2] = (v210_data + (v197_data * v208_data));
            float v213_data = s0[39];
            float v215_data = ir1[3];
            ir1[3] = (v215_data + (v197_data * v213_data));
            float v218_data = s0[51];
            float v220_data = ir1[4];
            ir1[4] = (v220_data + (v197_data * v218_data));
            float v223_data = s0[63];
            float v225_data = ir1[5];
            ir1[5] = (v225_data + (v197_data * v223_data));
            float v228_data = s0[75];
            float v230_data = ir1[6];
            ir1[6] = (v230_data + (v197_data * v228_data));
            float v233_data = s0[87];
            float v235_data = ir1[7];
            ir1[7] = (v235_data + (v197_data * v233_data));
          }
          if (v16_lead < 12) {
            float v241_data = r0[4];
            float v242_data = s0[4];
            float v244_data = ir1[0];
            ir1[0] = (v244_data + (v241_data * v242_data));
            float v247_data = s0[16];
            float v249_data = ir1[1];
            ir1[1] = (v249_data + (v241_data * v247_data));
            float v252_data = s0[28];
            float v254_data = ir1[2];
            ir1[2] = (v254_data + (v241_data * v252_data));
            float v257_data = s0[40];
            float v259_data = ir1[3];
            ir1[3] = (v259_data + (v241_data * v257_data));
            float v262_data = s0[52];
            float v264_data = ir1[4];
            ir1[4] = (v264_data + (v241_data * v262_data));
            float v267_data = s0[64];
            float v269_data = ir1[5];
            ir1[5] = (v269_data + (v241_data * v267_data));
            float v272_data = s0[76];
            float v274_data = ir1[6];
            ir1[6] = (v274_data + (v241_data * v272_data));
            float v277_data = s0[88];
            float v279_data = ir1[7];
            ir1[7] = (v279_data + (v241_data * v277_data));
          }
          if (v16_lead < 12) {
            float v285_data = r0[5];
            float v286_data = s0[5];
            float v288_data = ir1[0];
            ir1[0] = (v288_data + (v285_data * v286_data));
            float v291_data = s0[17];
            float v293_data = ir1[1];
            ir1[1] = (v293_data + (v285_data * v291_data));
            float v296_data = s0[29];
            float v298_data = ir1[2];
            ir1[2] = (v298_data + (v285_data * v296_data));
            float v301_data = s0[41];
            float v303_data = ir1[3];
            ir1[3] = (v303_data + (v285_data * v301_data));
            float v306_data = s0[53];
            float v308_data = ir1[4];
            ir1[4] = (v308_data + (v285_data * v306_data));
            float v311_data = s0[65];
            float v313_data = ir1[5];
            ir1[5] = (v313_data + (v285_data * v311_data));
            float v316_data = s0[77];
            float v318_data = ir1[6];
            ir1[6] = (v318_data + (v285_data * v316_data));
            float v321_data = s0[89];
            float v323_data = ir1[7];
            ir1[7] = (v323_data + (v285_data * v321_data));
          }
          if (v16_lead < 12) {
            float v329_data = r0[6];
            float v330_data = s0[6];
            float v332_data = ir1[0];
            ir1[0] = (v332_data + (v329_data * v330_data));
            float v335_data = s0[18];
            float v337_data = ir1[1];
            ir1[1] = (v337_data + (v329_data * v335_data));
            float v340_data = s0[30];
            float v342_data = ir1[2];
            ir1[2] = (v342_data + (v329_data * v340_data));
            float v345_data = s0[42];
            float v347_data = ir1[3];
            ir1[3] = (v347_data + (v329_data * v345_data));
            float v350_data = s0[54];
            float v352_data = ir1[4];
            ir1[4] = (v352_data + (v329_data * v350_data));
            float v355_data = s0[66];
            float v357_data = ir1[5];
            ir1[5] = (v357_data + (v329_data * v355_data));
            float v360_data = s0[78];
            float v362_data = ir1[6];
            ir1[6] = (v362_data + (v329_data * v360_data));
            float v365_data = s0[90];
            float v367_data = ir1[7];
            ir1[7] = (v367_data + (v329_data * v365_data));
          }
          if (v16_lead < 12) {
            float v373_data = r0[7];
            float v374_data = s0[7];
            float v376_data = ir1[0];
            ir1[0] = (v376_data + (v373_data * v374_data));
            float v379_data = s0[19];
            float v381_data = ir1[1];
            ir1[1] = (v381_data + (v373_data * v379_data));
            float v384_data = s0[31];
            float v386_data = ir1[2];
            ir1[2] = (v386_data + (v373_data * v384_data));
            float v389_data = s0[43];
            float v391_data = ir1[3];
            ir1[3] = (v391_data + (v373_data * v389_data));
            float v394_data = s0[55];
            float v396_data = ir1[4];
            ir1[4] = (v396_data + (v373_data * v394_data));
            float v399_data = s0[67];
            float v401_data = ir1[5];
            ir1[5] = (v401_data + (v373_data * v399_data));
            float v404_data = s0[79];
            float v406_data = ir1[6];
            ir1[6] = (v406_data + (v373_data * v404_data));
            float v409_data = s0[91];
            float v411_data = ir1[7];
            ir1[7] = (v411_data + (v373_data * v409_data));
          }
          if (v16_lead < 12) {
            float v417_data = r0[8];
            float v418_data = s0[8];
            float v420_data = ir1[0];
            ir1[0] = (v420_data + (v417_data * v418_data));
            float v423_data = s0[20];
            float v425_data = ir1[1];
            ir1[1] = (v425_data + (v417_data * v423_data));
            float v428_data = s0[32];
            float v430_data = ir1[2];
            ir1[2] = (v430_data + (v417_data * v428_data));
            float v433_data = s0[44];
            float v435_data = ir1[3];
            ir1[3] = (v435_data + (v417_data * v433_data));
            float v438_data = s0[56];
            float v440_data = ir1[4];
            ir1[4] = (v440_data + (v417_data * v438_data));
            float v443_data = s0[68];
            float v445_data = ir1[5];
            ir1[5] = (v445_data + (v417_data * v443_data));
            float v448_data = s0[80];
            float v450_data = ir1[6];
            ir1[6] = (v450_data + (v417_data * v448_data));
            float v453_data = s0[92];
            float v455_data = ir1[7];
            ir1[7] = (v455_data + (v417_data * v453_data));
          }
          if (v16_lead < 12) {
            float v461_data = r0[9];
            float v462_data = s0[9];
            float v464_data = ir1[0];
            ir1[0] = (v464_data + (v461_data * v462_data));
            float v467_data = s0[21];
            float v469_data = ir1[1];
            ir1[1] = (v469_data + (v461_data * v467_data));
            float v472_data = s0[33];
            float v474_data = ir1[2];
            ir1[2] = (v474_data + (v461_data * v472_data));
            float v477_data = s0[45];
            float v479_data = ir1[3];
            ir1[3] = (v479_data + (v461_data * v477_data));
            float v482_data = s0[57];
            float v484_data = ir1[4];
            ir1[4] = (v484_data + (v461_data * v482_data));
            float v487_data = s0[69];
            float v489_data = ir1[5];
            ir1[5] = (v489_data + (v461_data * v487_data));
            float v492_data = s0[81];
            float v494_data = ir1[6];
            ir1[6] = (v494_data + (v461_data * v492_data));
            float v497_data = s0[93];
            float v499_data = ir1[7];
            ir1[7] = (v499_data + (v461_data * v497_data));
          }
          if (v16_lead < 12) {
            float v505_data = r0[10];
            float v506_data = s0[10];
            float v508_data = ir1[0];
            ir1[0] = (v508_data + (v505_data * v506_data));
            float v511_data = s0[22];
            float v513_data = ir1[1];
            ir1[1] = (v513_data + (v505_data * v511_data));
            float v516_data = s0[34];
            float v518_data = ir1[2];
            ir1[2] = (v518_data + (v505_data * v516_data));
            float v521_data = s0[46];
            float v523_data = ir1[3];
            ir1[3] = (v523_data + (v505_data * v521_data));
            float v526_data = s0[58];
            float v528_data = ir1[4];
            ir1[4] = (v528_data + (v505_data * v526_data));
            float v531_data = s0[70];
            float v533_data = ir1[5];
            ir1[5] = (v533_data + (v505_data * v531_data));
            float v536_data = s0[82];
            float v538_data = ir1[6];
            ir1[6] = (v538_data + (v505_data * v536_data));
            float v541_data = s0[94];
            float v543_data = ir1[7];
            ir1[7] = (v543_data + (v505_data * v541_data));
          }
          if (v16_lead < 12) {
            float v549_data = r0[11];
            float v550_data = s0[11];
            float v552_data = ir1[0];
            ir1[0] = (v552_data + (v549_data * v550_data));
            float v555_data = s0[23];
            float v557_data = ir1[1];
            ir1[1] = (v557_data + (v549_data * v555_data));
            float v560_data = s0[35];
            float v562_data = ir1[2];
            ir1[2] = (v562_data + (v549_data * v560_data));
            float v565_data = s0[47];
            float v567_data = ir1[3];
            ir1[3] = (v567_data + (v549_data * v565_data));
            float v570_data = s0[59];
            float v572_data = ir1[4];
            ir1[4] = (v572_data + (v549_data * v570_data));
            float v575_data = s0[71];
            float v577_data = ir1[5];
            ir1[5] = (v577_data + (v549_data * v575_data));
            float v580_data = s0[83];
            float v582_data = ir1[6];
            ir1[6] = (v582_data + (v549_data * v580_data));
            float v585_data = s0[95];
            float v587_data = ir1[7];
            ir1[7] = (v587_data + (v549_data * v585_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v593_n1 = 0; v593_n1 < 8; ++v593_n1) {
              int32_t v594_a = 0 + v593_n1;
              float v596_data = ir1[v593_n1];
              r1[v593_n1] = v596_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v605_i1 = 0; v605_i1 < 12; ++v605_i1) {
              int32_t v611_a = v605_i1 * 12;
              int32_t v612_a = v16_lead + v611_a;
              float v620_data = __ldcg(&glb_m5[(v16_lead + v611_a)]);
              r4[v605_i1] = v620_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v16_lead < 12) {
            float v628_data = r2[0];
            float v629_data = s1[0];
            float v631_data = ir3[0];
            ir3[0] = (v631_data + (v628_data * v629_data));
            float v634_data = s1[12];
            float v636_data = ir3[1];
            ir3[1] = (v636_data + (v628_data * v634_data));
            float v639_data = s1[24];
            float v641_data = ir3[2];
            ir3[2] = (v641_data + (v628_data * v639_data));
            float v644_data = s1[36];
            float v646_data = ir3[3];
            ir3[3] = (v646_data + (v628_data * v644_data));
            float v649_data = s1[48];
            float v651_data = ir3[4];
            ir3[4] = (v651_data + (v628_data * v649_data));
            float v654_data = s1[60];
            float v656_data = ir3[5];
            ir3[5] = (v656_data + (v628_data * v654_data));
            float v659_data = s1[72];
            float v661_data = ir3[6];
            ir3[6] = (v661_data + (v628_data * v659_data));
            float v664_data = s1[84];
            float v666_data = ir3[7];
            ir3[7] = (v666_data + (v628_data * v664_data));
          }
          if (v16_lead < 12) {
            float v672_data = r2[1];
            float v673_data = s1[1];
            float v675_data = ir3[0];
            ir3[0] = (v675_data + (v672_data * v673_data));
            float v678_data = s1[13];
            float v680_data = ir3[1];
            ir3[1] = (v680_data + (v672_data * v678_data));
            float v683_data = s1[25];
            float v685_data = ir3[2];
            ir3[2] = (v685_data + (v672_data * v683_data));
            float v688_data = s1[37];
            float v690_data = ir3[3];
            ir3[3] = (v690_data + (v672_data * v688_data));
            float v693_data = s1[49];
            float v695_data = ir3[4];
            ir3[4] = (v695_data + (v672_data * v693_data));
            float v698_data = s1[61];
            float v700_data = ir3[5];
            ir3[5] = (v700_data + (v672_data * v698_data));
            float v703_data = s1[73];
            float v705_data = ir3[6];
            ir3[6] = (v705_data + (v672_data * v703_data));
            float v708_data = s1[85];
            float v710_data = ir3[7];
            ir3[7] = (v710_data + (v672_data * v708_data));
          }
          if (v16_lead < 12) {
            float v716_data = r2[2];
            float v717_data = s1[2];
            float v719_data = ir3[0];
            ir3[0] = (v719_data + (v716_data * v717_data));
            float v722_data = s1[14];
            float v724_data = ir3[1];
            ir3[1] = (v724_data + (v716_data * v722_data));
            float v727_data = s1[26];
            float v729_data = ir3[2];
            ir3[2] = (v729_data + (v716_data * v727_data));
            float v732_data = s1[38];
            float v734_data = ir3[3];
            ir3[3] = (v734_data + (v716_data * v732_data));
            float v737_data = s1[50];
            float v739_data = ir3[4];
            ir3[4] = (v739_data + (v716_data * v737_data));
            float v742_data = s1[62];
            float v744_data = ir3[5];
            ir3[5] = (v744_data + (v716_data * v742_data));
            float v747_data = s1[74];
            float v749_data = ir3[6];
            ir3[6] = (v749_data + (v716_data * v747_data));
            float v752_data = s1[86];
            float v754_data = ir3[7];
            ir3[7] = (v754_data + (v716_data * v752_data));
          }
          if (v16_lead < 12) {
            float v760_data = r2[3];
            float v761_data = s1[3];
            float v763_data = ir3[0];
            ir3[0] = (v763_data + (v760_data * v761_data));
            float v766_data = s1[15];
            float v768_data = ir3[1];
            ir3[1] = (v768_data + (v760_data * v766_data));
            float v771_data = s1[27];
            float v773_data = ir3[2];
            ir3[2] = (v773_data + (v760_data * v771_data));
            float v776_data = s1[39];
            float v778_data = ir3[3];
            ir3[3] = (v778_data + (v760_data * v776_data));
            float v781_data = s1[51];
            float v783_data = ir3[4];
            ir3[4] = (v783_data + (v760_data * v781_data));
            float v786_data = s1[63];
            float v788_data = ir3[5];
            ir3[5] = (v788_data + (v760_data * v786_data));
            float v791_data = s1[75];
            float v793_data = ir3[6];
            ir3[6] = (v793_data + (v760_data * v791_data));
            float v796_data = s1[87];
            float v798_data = ir3[7];
            ir3[7] = (v798_data + (v760_data * v796_data));
          }
          if (v16_lead < 12) {
            float v804_data = r2[4];
            float v805_data = s1[4];
            float v807_data = ir3[0];
            ir3[0] = (v807_data + (v804_data * v805_data));
            float v810_data = s1[16];
            float v812_data = ir3[1];
            ir3[1] = (v812_data + (v804_data * v810_data));
            float v815_data = s1[28];
            float v817_data = ir3[2];
            ir3[2] = (v817_data + (v804_data * v815_data));
            float v820_data = s1[40];
            float v822_data = ir3[3];
            ir3[3] = (v822_data + (v804_data * v820_data));
            float v825_data = s1[52];
            float v827_data = ir3[4];
            ir3[4] = (v827_data + (v804_data * v825_data));
            float v830_data = s1[64];
            float v832_data = ir3[5];
            ir3[5] = (v832_data + (v804_data * v830_data));
            float v835_data = s1[76];
            float v837_data = ir3[6];
            ir3[6] = (v837_data + (v804_data * v835_data));
            float v840_data = s1[88];
            float v842_data = ir3[7];
            ir3[7] = (v842_data + (v804_data * v840_data));
          }
          if (v16_lead < 12) {
            float v848_data = r2[5];
            float v849_data = s1[5];
            float v851_data = ir3[0];
            ir3[0] = (v851_data + (v848_data * v849_data));
            float v854_data = s1[17];
            float v856_data = ir3[1];
            ir3[1] = (v856_data + (v848_data * v854_data));
            float v859_data = s1[29];
            float v861_data = ir3[2];
            ir3[2] = (v861_data + (v848_data * v859_data));
            float v864_data = s1[41];
            float v866_data = ir3[3];
            ir3[3] = (v866_data + (v848_data * v864_data));
            float v869_data = s1[53];
            float v871_data = ir3[4];
            ir3[4] = (v871_data + (v848_data * v869_data));
            float v874_data = s1[65];
            float v876_data = ir3[5];
            ir3[5] = (v876_data + (v848_data * v874_data));
            float v879_data = s1[77];
            float v881_data = ir3[6];
            ir3[6] = (v881_data + (v848_data * v879_data));
            float v884_data = s1[89];
            float v886_data = ir3[7];
            ir3[7] = (v886_data + (v848_data * v884_data));
          }
          if (v16_lead < 12) {
            float v892_data = r2[6];
            float v893_data = s1[6];
            float v895_data = ir3[0];
            ir3[0] = (v895_data + (v892_data * v893_data));
            float v898_data = s1[18];
            float v900_data = ir3[1];
            ir3[1] = (v900_data + (v892_data * v898_data));
            float v903_data = s1[30];
            float v905_data = ir3[2];
            ir3[2] = (v905_data + (v892_data * v903_data));
            float v908_data = s1[42];
            float v910_data = ir3[3];
            ir3[3] = (v910_data + (v892_data * v908_data));
            float v913_data = s1[54];
            float v915_data = ir3[4];
            ir3[4] = (v915_data + (v892_data * v913_data));
            float v918_data = s1[66];
            float v920_data = ir3[5];
            ir3[5] = (v920_data + (v892_data * v918_data));
            float v923_data = s1[78];
            float v925_data = ir3[6];
            ir3[6] = (v925_data + (v892_data * v923_data));
            float v928_data = s1[90];
            float v930_data = ir3[7];
            ir3[7] = (v930_data + (v892_data * v928_data));
          }
          if (v16_lead < 12) {
            float v936_data = r2[7];
            float v937_data = s1[7];
            float v939_data = ir3[0];
            ir3[0] = (v939_data + (v936_data * v937_data));
            float v942_data = s1[19];
            float v944_data = ir3[1];
            ir3[1] = (v944_data + (v936_data * v942_data));
            float v947_data = s1[31];
            float v949_data = ir3[2];
            ir3[2] = (v949_data + (v936_data * v947_data));
            float v952_data = s1[43];
            float v954_data = ir3[3];
            ir3[3] = (v954_data + (v936_data * v952_data));
            float v957_data = s1[55];
            float v959_data = ir3[4];
            ir3[4] = (v959_data + (v936_data * v957_data));
            float v962_data = s1[67];
            float v964_data = ir3[5];
            ir3[5] = (v964_data + (v936_data * v962_data));
            float v967_data = s1[79];
            float v969_data = ir3[6];
            ir3[6] = (v969_data + (v936_data * v967_data));
            float v972_data = s1[91];
            float v974_data = ir3[7];
            ir3[7] = (v974_data + (v936_data * v972_data));
          }
          if (v16_lead < 12) {
            float v980_data = r2[8];
            float v981_data = s1[8];
            float v983_data = ir3[0];
            ir3[0] = (v983_data + (v980_data * v981_data));
            float v986_data = s1[20];
            float v988_data = ir3[1];
            ir3[1] = (v988_data + (v980_data * v986_data));
            float v991_data = s1[32];
            float v993_data = ir3[2];
            ir3[2] = (v993_data + (v980_data * v991_data));
            float v996_data = s1[44];
            float v998_data = ir3[3];
            ir3[3] = (v998_data + (v980_data * v996_data));
            float v1001_data = s1[56];
            float v1003_data = ir3[4];
            ir3[4] = (v1003_data + (v980_data * v1001_data));
            float v1006_data = s1[68];
            float v1008_data = ir3[5];
            ir3[5] = (v1008_data + (v980_data * v1006_data));
            float v1011_data = s1[80];
            float v1013_data = ir3[6];
            ir3[6] = (v1013_data + (v980_data * v1011_data));
            float v1016_data = s1[92];
            float v1018_data = ir3[7];
            ir3[7] = (v1018_data + (v980_data * v1016_data));
          }
          if (v16_lead < 12) {
            float v1024_data = r2[9];
            float v1025_data = s1[9];
            float v1027_data = ir3[0];
            ir3[0] = (v1027_data + (v1024_data * v1025_data));
            float v1030_data = s1[21];
            float v1032_data = ir3[1];
            ir3[1] = (v1032_data + (v1024_data * v1030_data));
            float v1035_data = s1[33];
            float v1037_data = ir3[2];
            ir3[2] = (v1037_data + (v1024_data * v1035_data));
            float v1040_data = s1[45];
            float v1042_data = ir3[3];
            ir3[3] = (v1042_data + (v1024_data * v1040_data));
            float v1045_data = s1[57];
            float v1047_data = ir3[4];
            ir3[4] = (v1047_data + (v1024_data * v1045_data));
            float v1050_data = s1[69];
            float v1052_data = ir3[5];
            ir3[5] = (v1052_data + (v1024_data * v1050_data));
            float v1055_data = s1[81];
            float v1057_data = ir3[6];
            ir3[6] = (v1057_data + (v1024_data * v1055_data));
            float v1060_data = s1[93];
            float v1062_data = ir3[7];
            ir3[7] = (v1062_data + (v1024_data * v1060_data));
          }
          if (v16_lead < 12) {
            float v1068_data = r2[10];
            float v1069_data = s1[10];
            float v1071_data = ir3[0];
            ir3[0] = (v1071_data + (v1068_data * v1069_data));
            float v1074_data = s1[22];
            float v1076_data = ir3[1];
            ir3[1] = (v1076_data + (v1068_data * v1074_data));
            float v1079_data = s1[34];
            float v1081_data = ir3[2];
            ir3[2] = (v1081_data + (v1068_data * v1079_data));
            float v1084_data = s1[46];
            float v1086_data = ir3[3];
            ir3[3] = (v1086_data + (v1068_data * v1084_data));
            float v1089_data = s1[58];
            float v1091_data = ir3[4];
            ir3[4] = (v1091_data + (v1068_data * v1089_data));
            float v1094_data = s1[70];
            float v1096_data = ir3[5];
            ir3[5] = (v1096_data + (v1068_data * v1094_data));
            float v1099_data = s1[82];
            float v1101_data = ir3[6];
            ir3[6] = (v1101_data + (v1068_data * v1099_data));
            float v1104_data = s1[94];
            float v1106_data = ir3[7];
            ir3[7] = (v1106_data + (v1068_data * v1104_data));
          }
          if (v16_lead < 12) {
            float v1112_data = r2[11];
            float v1113_data = s1[11];
            float v1115_data = ir3[0];
            ir3[0] = (v1115_data + (v1112_data * v1113_data));
            float v1118_data = s1[23];
            float v1120_data = ir3[1];
            ir3[1] = (v1120_data + (v1112_data * v1118_data));
            float v1123_data = s1[35];
            float v1125_data = ir3[2];
            ir3[2] = (v1125_data + (v1112_data * v1123_data));
            float v1128_data = s1[47];
            float v1130_data = ir3[3];
            ir3[3] = (v1130_data + (v1112_data * v1128_data));
            float v1133_data = s1[59];
            float v1135_data = ir3[4];
            ir3[4] = (v1135_data + (v1112_data * v1133_data));
            float v1138_data = s1[71];
            float v1140_data = ir3[5];
            ir3[5] = (v1140_data + (v1112_data * v1138_data));
            float v1143_data = s1[83];
            float v1145_data = ir3[6];
            ir3[6] = (v1145_data + (v1112_data * v1143_data));
            float v1148_data = s1[95];
            float v1150_data = ir3[7];
            ir3[7] = (v1150_data + (v1112_data * v1148_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v1156_n1 = 0; v1156_n1 < 8; ++v1156_n1) {
              int32_t v1157_a = 0 + v1156_n1;
              float v1159_data = ir3[v1156_n1];
              int32_t v1160_a = 0 + v1156_n1;
              float v1162_data = r1[v1156_n1];
              r3[v1156_n1] = (v1162_data + v1159_data);
            }
          }
          __syncwarp();
          float* __restrict__ s2 = &localShrMem0[0];
          // s2 = load{g>s}(glb_m6[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v1172_i1 = 0; v1172_i1 < 12; ++v1172_i1) {
              int32_t v1178_a = v1172_i1 * 12;
              int32_t v1179_a = v16_lead + v1178_a;
              float v1187_data = __ldcg(&glb_m7[(v16_lead + v1178_a)]);
              r6[v1172_i1] = v1187_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          float r5[8]{};
          __syncwarp();
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v16_lead < 12) {
            float v1195_data = r4[0];
            float v1196_data = s2[0];
            float v1198_data = ir5[0];
            ir5[0] = (v1198_data + (v1195_data * v1196_data));
            float v1201_data = s2[12];
            float v1203_data = ir5[1];
            ir5[1] = (v1203_data + (v1195_data * v1201_data));
            float v1206_data = s2[24];
            float v1208_data = ir5[2];
            ir5[2] = (v1208_data + (v1195_data * v1206_data));
            float v1211_data = s2[36];
            float v1213_data = ir5[3];
            ir5[3] = (v1213_data + (v1195_data * v1211_data));
            float v1216_data = s2[48];
            float v1218_data = ir5[4];
            ir5[4] = (v1218_data + (v1195_data * v1216_data));
            float v1221_data = s2[60];
            float v1223_data = ir5[5];
            ir5[5] = (v1223_data + (v1195_data * v1221_data));
            float v1226_data = s2[72];
            float v1228_data = ir5[6];
            ir5[6] = (v1228_data + (v1195_data * v1226_data));
            float v1231_data = s2[84];
            float v1233_data = ir5[7];
            ir5[7] = (v1233_data + (v1195_data * v1231_data));
          }
          if (v16_lead < 12) {
            float v1239_data = r4[1];
            float v1240_data = s2[1];
            float v1242_data = ir5[0];
            ir5[0] = (v1242_data + (v1239_data * v1240_data));
            float v1245_data = s2[13];
            float v1247_data = ir5[1];
            ir5[1] = (v1247_data + (v1239_data * v1245_data));
            float v1250_data = s2[25];
            float v1252_data = ir5[2];
            ir5[2] = (v1252_data + (v1239_data * v1250_data));
            float v1255_data = s2[37];
            float v1257_data = ir5[3];
            ir5[3] = (v1257_data + (v1239_data * v1255_data));
            float v1260_data = s2[49];
            float v1262_data = ir5[4];
            ir5[4] = (v1262_data + (v1239_data * v1260_data));
            float v1265_data = s2[61];
            float v1267_data = ir5[5];
            ir5[5] = (v1267_data + (v1239_data * v1265_data));
            float v1270_data = s2[73];
            float v1272_data = ir5[6];
            ir5[6] = (v1272_data + (v1239_data * v1270_data));
            float v1275_data = s2[85];
            float v1277_data = ir5[7];
            ir5[7] = (v1277_data + (v1239_data * v1275_data));
          }
          if (v16_lead < 12) {
            float v1283_data = r4[2];
            float v1284_data = s2[2];
            float v1286_data = ir5[0];
            ir5[0] = (v1286_data + (v1283_data * v1284_data));
            float v1289_data = s2[14];
            float v1291_data = ir5[1];
            ir5[1] = (v1291_data + (v1283_data * v1289_data));
            float v1294_data = s2[26];
            float v1296_data = ir5[2];
            ir5[2] = (v1296_data + (v1283_data * v1294_data));
            float v1299_data = s2[38];
            float v1301_data = ir5[3];
            ir5[3] = (v1301_data + (v1283_data * v1299_data));
            float v1304_data = s2[50];
            float v1306_data = ir5[4];
            ir5[4] = (v1306_data + (v1283_data * v1304_data));
            float v1309_data = s2[62];
            float v1311_data = ir5[5];
            ir5[5] = (v1311_data + (v1283_data * v1309_data));
            float v1314_data = s2[74];
            float v1316_data = ir5[6];
            ir5[6] = (v1316_data + (v1283_data * v1314_data));
            float v1319_data = s2[86];
            float v1321_data = ir5[7];
            ir5[7] = (v1321_data + (v1283_data * v1319_data));
          }
          if (v16_lead < 12) {
            float v1327_data = r4[3];
            float v1328_data = s2[3];
            float v1330_data = ir5[0];
            ir5[0] = (v1330_data + (v1327_data * v1328_data));
            float v1333_data = s2[15];
            float v1335_data = ir5[1];
            ir5[1] = (v1335_data + (v1327_data * v1333_data));
            float v1338_data = s2[27];
            float v1340_data = ir5[2];
            ir5[2] = (v1340_data + (v1327_data * v1338_data));
            float v1343_data = s2[39];
            float v1345_data = ir5[3];
            ir5[3] = (v1345_data + (v1327_data * v1343_data));
            float v1348_data = s2[51];
            float v1350_data = ir5[4];
            ir5[4] = (v1350_data + (v1327_data * v1348_data));
            float v1353_data = s2[63];
            float v1355_data = ir5[5];
            ir5[5] = (v1355_data + (v1327_data * v1353_data));
            float v1358_data = s2[75];
            float v1360_data = ir5[6];
            ir5[6] = (v1360_data + (v1327_data * v1358_data));
            float v1363_data = s2[87];
            float v1365_data = ir5[7];
            ir5[7] = (v1365_data + (v1327_data * v1363_data));
          }
          if (v16_lead < 12) {
            float v1371_data = r4[4];
            float v1372_data = s2[4];
            float v1374_data = ir5[0];
            ir5[0] = (v1374_data + (v1371_data * v1372_data));
            float v1377_data = s2[16];
            float v1379_data = ir5[1];
            ir5[1] = (v1379_data + (v1371_data * v1377_data));
            float v1382_data = s2[28];
            float v1384_data = ir5[2];
            ir5[2] = (v1384_data + (v1371_data * v1382_data));
            float v1387_data = s2[40];
            float v1389_data = ir5[3];
            ir5[3] = (v1389_data + (v1371_data * v1387_data));
            float v1392_data = s2[52];
            float v1394_data = ir5[4];
            ir5[4] = (v1394_data + (v1371_data * v1392_data));
            float v1397_data = s2[64];
            float v1399_data = ir5[5];
            ir5[5] = (v1399_data + (v1371_data * v1397_data));
            float v1402_data = s2[76];
            float v1404_data = ir5[6];
            ir5[6] = (v1404_data + (v1371_data * v1402_data));
            float v1407_data = s2[88];
            float v1409_data = ir5[7];
            ir5[7] = (v1409_data + (v1371_data * v1407_data));
          }
          if (v16_lead < 12) {
            float v1415_data = r4[5];
            float v1416_data = s2[5];
            float v1418_data = ir5[0];
            ir5[0] = (v1418_data + (v1415_data * v1416_data));
            float v1421_data = s2[17];
            float v1423_data = ir5[1];
            ir5[1] = (v1423_data + (v1415_data * v1421_data));
            float v1426_data = s2[29];
            float v1428_data = ir5[2];
            ir5[2] = (v1428_data + (v1415_data * v1426_data));
            float v1431_data = s2[41];
            float v1433_data = ir5[3];
            ir5[3] = (v1433_data + (v1415_data * v1431_data));
            float v1436_data = s2[53];
            float v1438_data = ir5[4];
            ir5[4] = (v1438_data + (v1415_data * v1436_data));
            float v1441_data = s2[65];
            float v1443_data = ir5[5];
            ir5[5] = (v1443_data + (v1415_data * v1441_data));
            float v1446_data = s2[77];
            float v1448_data = ir5[6];
            ir5[6] = (v1448_data + (v1415_data * v1446_data));
            float v1451_data = s2[89];
            float v1453_data = ir5[7];
            ir5[7] = (v1453_data + (v1415_data * v1451_data));
          }
          if (v16_lead < 12) {
            float v1459_data = r4[6];
            float v1460_data = s2[6];
            float v1462_data = ir5[0];
            ir5[0] = (v1462_data + (v1459_data * v1460_data));
            float v1465_data = s2[18];
            float v1467_data = ir5[1];
            ir5[1] = (v1467_data + (v1459_data * v1465_data));
            float v1470_data = s2[30];
            float v1472_data = ir5[2];
            ir5[2] = (v1472_data + (v1459_data * v1470_data));
            float v1475_data = s2[42];
            float v1477_data = ir5[3];
            ir5[3] = (v1477_data + (v1459_data * v1475_data));
            float v1480_data = s2[54];
            float v1482_data = ir5[4];
            ir5[4] = (v1482_data + (v1459_data * v1480_data));
            float v1485_data = s2[66];
            float v1487_data = ir5[5];
            ir5[5] = (v1487_data + (v1459_data * v1485_data));
            float v1490_data = s2[78];
            float v1492_data = ir5[6];
            ir5[6] = (v1492_data + (v1459_data * v1490_data));
            float v1495_data = s2[90];
            float v1497_data = ir5[7];
            ir5[7] = (v1497_data + (v1459_data * v1495_data));
          }
          if (v16_lead < 12) {
            float v1503_data = r4[7];
            float v1504_data = s2[7];
            float v1506_data = ir5[0];
            ir5[0] = (v1506_data + (v1503_data * v1504_data));
            float v1509_data = s2[19];
            float v1511_data = ir5[1];
            ir5[1] = (v1511_data + (v1503_data * v1509_data));
            float v1514_data = s2[31];
            float v1516_data = ir5[2];
            ir5[2] = (v1516_data + (v1503_data * v1514_data));
            float v1519_data = s2[43];
            float v1521_data = ir5[3];
            ir5[3] = (v1521_data + (v1503_data * v1519_data));
            float v1524_data = s2[55];
            float v1526_data = ir5[4];
            ir5[4] = (v1526_data + (v1503_data * v1524_data));
            float v1529_data = s2[67];
            float v1531_data = ir5[5];
            ir5[5] = (v1531_data + (v1503_data * v1529_data));
            float v1534_data = s2[79];
            float v1536_data = ir5[6];
            ir5[6] = (v1536_data + (v1503_data * v1534_data));
            float v1539_data = s2[91];
            float v1541_data = ir5[7];
            ir5[7] = (v1541_data + (v1503_data * v1539_data));
          }
          if (v16_lead < 12) {
            float v1547_data = r4[8];
            float v1548_data = s2[8];
            float v1550_data = ir5[0];
            ir5[0] = (v1550_data + (v1547_data * v1548_data));
            float v1553_data = s2[20];
            float v1555_data = ir5[1];
            ir5[1] = (v1555_data + (v1547_data * v1553_data));
            float v1558_data = s2[32];
            float v1560_data = ir5[2];
            ir5[2] = (v1560_data + (v1547_data * v1558_data));
            float v1563_data = s2[44];
            float v1565_data = ir5[3];
            ir5[3] = (v1565_data + (v1547_data * v1563_data));
            float v1568_data = s2[56];
            float v1570_data = ir5[4];
            ir5[4] = (v1570_data + (v1547_data * v1568_data));
            float v1573_data = s2[68];
            float v1575_data = ir5[5];
            ir5[5] = (v1575_data + (v1547_data * v1573_data));
            float v1578_data = s2[80];
            float v1580_data = ir5[6];
            ir5[6] = (v1580_data + (v1547_data * v1578_data));
            float v1583_data = s2[92];
            float v1585_data = ir5[7];
            ir5[7] = (v1585_data + (v1547_data * v1583_data));
          }
          if (v16_lead < 12) {
            float v1591_data = r4[9];
            float v1592_data = s2[9];
            float v1594_data = ir5[0];
            ir5[0] = (v1594_data + (v1591_data * v1592_data));
            float v1597_data = s2[21];
            float v1599_data = ir5[1];
            ir5[1] = (v1599_data + (v1591_data * v1597_data));
            float v1602_data = s2[33];
            float v1604_data = ir5[2];
            ir5[2] = (v1604_data + (v1591_data * v1602_data));
            float v1607_data = s2[45];
            float v1609_data = ir5[3];
            ir5[3] = (v1609_data + (v1591_data * v1607_data));
            float v1612_data = s2[57];
            float v1614_data = ir5[4];
            ir5[4] = (v1614_data + (v1591_data * v1612_data));
            float v1617_data = s2[69];
            float v1619_data = ir5[5];
            ir5[5] = (v1619_data + (v1591_data * v1617_data));
            float v1622_data = s2[81];
            float v1624_data = ir5[6];
            ir5[6] = (v1624_data + (v1591_data * v1622_data));
            float v1627_data = s2[93];
            float v1629_data = ir5[7];
            ir5[7] = (v1629_data + (v1591_data * v1627_data));
          }
          if (v16_lead < 12) {
            float v1635_data = r4[10];
            float v1636_data = s2[10];
            float v1638_data = ir5[0];
            ir5[0] = (v1638_data + (v1635_data * v1636_data));
            float v1641_data = s2[22];
            float v1643_data = ir5[1];
            ir5[1] = (v1643_data + (v1635_data * v1641_data));
            float v1646_data = s2[34];
            float v1648_data = ir5[2];
            ir5[2] = (v1648_data + (v1635_data * v1646_data));
            float v1651_data = s2[46];
            float v1653_data = ir5[3];
            ir5[3] = (v1653_data + (v1635_data * v1651_data));
            float v1656_data = s2[58];
            float v1658_data = ir5[4];
            ir5[4] = (v1658_data + (v1635_data * v1656_data));
            float v1661_data = s2[70];
            float v1663_data = ir5[5];
            ir5[5] = (v1663_data + (v1635_data * v1661_data));
            float v1666_data = s2[82];
            float v1668_data = ir5[6];
            ir5[6] = (v1668_data + (v1635_data * v1666_data));
            float v1671_data = s2[94];
            float v1673_data = ir5[7];
            ir5[7] = (v1673_data + (v1635_data * v1671_data));
          }
          if (v16_lead < 12) {
            float v1679_data = r4[11];
            float v1680_data = s2[11];
            float v1682_data = ir5[0];
            ir5[0] = (v1682_data + (v1679_data * v1680_data));
            float v1685_data = s2[23];
            float v1687_data = ir5[1];
            ir5[1] = (v1687_data + (v1679_data * v1685_data));
            float v1690_data = s2[35];
            float v1692_data = ir5[2];
            ir5[2] = (v1692_data + (v1679_data * v1690_data));
            float v1695_data = s2[47];
            float v1697_data = ir5[3];
            ir5[3] = (v1697_data + (v1679_data * v1695_data));
            float v1700_data = s2[59];
            float v1702_data = ir5[4];
            ir5[4] = (v1702_data + (v1679_data * v1700_data));
            float v1705_data = s2[71];
            float v1707_data = ir5[5];
            ir5[5] = (v1707_data + (v1679_data * v1705_data));
            float v1710_data = s2[83];
            float v1712_data = ir5[6];
            ir5[6] = (v1712_data + (v1679_data * v1710_data));
            float v1715_data = s2[95];
            float v1717_data = ir5[7];
            ir5[7] = (v1717_data + (v1679_data * v1715_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v1723_n1 = 0; v1723_n1 < 8; ++v1723_n1) {
              int32_t v1724_a = 0 + v1723_n1;
              float v1726_data = ir5[v1723_n1];
              int32_t v1727_a = 0 + v1723_n1;
              float v1729_data = r3[v1723_n1];
              r5[v1723_n1] = (v1729_data + v1726_data);
            }
          }
          __syncwarp();
          float* __restrict__ s3 = &localShrMem0[0];
          // s3 = load{g>s}(glb_m8[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r6 = load{g>r}(glb_m7););
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          __pipeline_wait_prior(0);
          float r7[8]{};
          __syncwarp();
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v16_lead < 12) {
            float v1740_data = r6[0];
            float v1741_data = s3[0];
            float v1743_data = ir7[0];
            ir7[0] = (v1743_data + (v1740_data * v1741_data));
            float v1746_data = s3[12];
            float v1748_data = ir7[1];
            ir7[1] = (v1748_data + (v1740_data * v1746_data));
            float v1751_data = s3[24];
            float v1753_data = ir7[2];
            ir7[2] = (v1753_data + (v1740_data * v1751_data));
            float v1756_data = s3[36];
            float v1758_data = ir7[3];
            ir7[3] = (v1758_data + (v1740_data * v1756_data));
            float v1761_data = s3[48];
            float v1763_data = ir7[4];
            ir7[4] = (v1763_data + (v1740_data * v1761_data));
            float v1766_data = s3[60];
            float v1768_data = ir7[5];
            ir7[5] = (v1768_data + (v1740_data * v1766_data));
            float v1771_data = s3[72];
            float v1773_data = ir7[6];
            ir7[6] = (v1773_data + (v1740_data * v1771_data));
            float v1776_data = s3[84];
            float v1778_data = ir7[7];
            ir7[7] = (v1778_data + (v1740_data * v1776_data));
          }
          if (v16_lead < 12) {
            float v1784_data = r6[1];
            float v1785_data = s3[1];
            float v1787_data = ir7[0];
            ir7[0] = (v1787_data + (v1784_data * v1785_data));
            float v1790_data = s3[13];
            float v1792_data = ir7[1];
            ir7[1] = (v1792_data + (v1784_data * v1790_data));
            float v1795_data = s3[25];
            float v1797_data = ir7[2];
            ir7[2] = (v1797_data + (v1784_data * v1795_data));
            float v1800_data = s3[37];
            float v1802_data = ir7[3];
            ir7[3] = (v1802_data + (v1784_data * v1800_data));
            float v1805_data = s3[49];
            float v1807_data = ir7[4];
            ir7[4] = (v1807_data + (v1784_data * v1805_data));
            float v1810_data = s3[61];
            float v1812_data = ir7[5];
            ir7[5] = (v1812_data + (v1784_data * v1810_data));
            float v1815_data = s3[73];
            float v1817_data = ir7[6];
            ir7[6] = (v1817_data + (v1784_data * v1815_data));
            float v1820_data = s3[85];
            float v1822_data = ir7[7];
            ir7[7] = (v1822_data + (v1784_data * v1820_data));
          }
          if (v16_lead < 12) {
            float v1828_data = r6[2];
            float v1829_data = s3[2];
            float v1831_data = ir7[0];
            ir7[0] = (v1831_data + (v1828_data * v1829_data));
            float v1834_data = s3[14];
            float v1836_data = ir7[1];
            ir7[1] = (v1836_data + (v1828_data * v1834_data));
            float v1839_data = s3[26];
            float v1841_data = ir7[2];
            ir7[2] = (v1841_data + (v1828_data * v1839_data));
            float v1844_data = s3[38];
            float v1846_data = ir7[3];
            ir7[3] = (v1846_data + (v1828_data * v1844_data));
            float v1849_data = s3[50];
            float v1851_data = ir7[4];
            ir7[4] = (v1851_data + (v1828_data * v1849_data));
            float v1854_data = s3[62];
            float v1856_data = ir7[5];
            ir7[5] = (v1856_data + (v1828_data * v1854_data));
            float v1859_data = s3[74];
            float v1861_data = ir7[6];
            ir7[6] = (v1861_data + (v1828_data * v1859_data));
            float v1864_data = s3[86];
            float v1866_data = ir7[7];
            ir7[7] = (v1866_data + (v1828_data * v1864_data));
          }
          if (v16_lead < 12) {
            float v1872_data = r6[3];
            float v1873_data = s3[3];
            float v1875_data = ir7[0];
            ir7[0] = (v1875_data + (v1872_data * v1873_data));
            float v1878_data = s3[15];
            float v1880_data = ir7[1];
            ir7[1] = (v1880_data + (v1872_data * v1878_data));
            float v1883_data = s3[27];
            float v1885_data = ir7[2];
            ir7[2] = (v1885_data + (v1872_data * v1883_data));
            float v1888_data = s3[39];
            float v1890_data = ir7[3];
            ir7[3] = (v1890_data + (v1872_data * v1888_data));
            float v1893_data = s3[51];
            float v1895_data = ir7[4];
            ir7[4] = (v1895_data + (v1872_data * v1893_data));
            float v1898_data = s3[63];
            float v1900_data = ir7[5];
            ir7[5] = (v1900_data + (v1872_data * v1898_data));
            float v1903_data = s3[75];
            float v1905_data = ir7[6];
            ir7[6] = (v1905_data + (v1872_data * v1903_data));
            float v1908_data = s3[87];
            float v1910_data = ir7[7];
            ir7[7] = (v1910_data + (v1872_data * v1908_data));
          }
          if (v16_lead < 12) {
            float v1916_data = r6[4];
            float v1917_data = s3[4];
            float v1919_data = ir7[0];
            ir7[0] = (v1919_data + (v1916_data * v1917_data));
            float v1922_data = s3[16];
            float v1924_data = ir7[1];
            ir7[1] = (v1924_data + (v1916_data * v1922_data));
            float v1927_data = s3[28];
            float v1929_data = ir7[2];
            ir7[2] = (v1929_data + (v1916_data * v1927_data));
            float v1932_data = s3[40];
            float v1934_data = ir7[3];
            ir7[3] = (v1934_data + (v1916_data * v1932_data));
            float v1937_data = s3[52];
            float v1939_data = ir7[4];
            ir7[4] = (v1939_data + (v1916_data * v1937_data));
            float v1942_data = s3[64];
            float v1944_data = ir7[5];
            ir7[5] = (v1944_data + (v1916_data * v1942_data));
            float v1947_data = s3[76];
            float v1949_data = ir7[6];
            ir7[6] = (v1949_data + (v1916_data * v1947_data));
            float v1952_data = s3[88];
            float v1954_data = ir7[7];
            ir7[7] = (v1954_data + (v1916_data * v1952_data));
          }
          if (v16_lead < 12) {
            float v1960_data = r6[5];
            float v1961_data = s3[5];
            float v1963_data = ir7[0];
            ir7[0] = (v1963_data + (v1960_data * v1961_data));
            float v1966_data = s3[17];
            float v1968_data = ir7[1];
            ir7[1] = (v1968_data + (v1960_data * v1966_data));
            float v1971_data = s3[29];
            float v1973_data = ir7[2];
            ir7[2] = (v1973_data + (v1960_data * v1971_data));
            float v1976_data = s3[41];
            float v1978_data = ir7[3];
            ir7[3] = (v1978_data + (v1960_data * v1976_data));
            float v1981_data = s3[53];
            float v1983_data = ir7[4];
            ir7[4] = (v1983_data + (v1960_data * v1981_data));
            float v1986_data = s3[65];
            float v1988_data = ir7[5];
            ir7[5] = (v1988_data + (v1960_data * v1986_data));
            float v1991_data = s3[77];
            float v1993_data = ir7[6];
            ir7[6] = (v1993_data + (v1960_data * v1991_data));
            float v1996_data = s3[89];
            float v1998_data = ir7[7];
            ir7[7] = (v1998_data + (v1960_data * v1996_data));
          }
          if (v16_lead < 12) {
            float v2004_data = r6[6];
            float v2005_data = s3[6];
            float v2007_data = ir7[0];
            ir7[0] = (v2007_data + (v2004_data * v2005_data));
            float v2010_data = s3[18];
            float v2012_data = ir7[1];
            ir7[1] = (v2012_data + (v2004_data * v2010_data));
            float v2015_data = s3[30];
            float v2017_data = ir7[2];
            ir7[2] = (v2017_data + (v2004_data * v2015_data));
            float v2020_data = s3[42];
            float v2022_data = ir7[3];
            ir7[3] = (v2022_data + (v2004_data * v2020_data));
            float v2025_data = s3[54];
            float v2027_data = ir7[4];
            ir7[4] = (v2027_data + (v2004_data * v2025_data));
            float v2030_data = s3[66];
            float v2032_data = ir7[5];
            ir7[5] = (v2032_data + (v2004_data * v2030_data));
            float v2035_data = s3[78];
            float v2037_data = ir7[6];
            ir7[6] = (v2037_data + (v2004_data * v2035_data));
            float v2040_data = s3[90];
            float v2042_data = ir7[7];
            ir7[7] = (v2042_data + (v2004_data * v2040_data));
          }
          if (v16_lead < 12) {
            float v2048_data = r6[7];
            float v2049_data = s3[7];
            float v2051_data = ir7[0];
            ir7[0] = (v2051_data + (v2048_data * v2049_data));
            float v2054_data = s3[19];
            float v2056_data = ir7[1];
            ir7[1] = (v2056_data + (v2048_data * v2054_data));
            float v2059_data = s3[31];
            float v2061_data = ir7[2];
            ir7[2] = (v2061_data + (v2048_data * v2059_data));
            float v2064_data = s3[43];
            float v2066_data = ir7[3];
            ir7[3] = (v2066_data + (v2048_data * v2064_data));
            float v2069_data = s3[55];
            float v2071_data = ir7[4];
            ir7[4] = (v2071_data + (v2048_data * v2069_data));
            float v2074_data = s3[67];
            float v2076_data = ir7[5];
            ir7[5] = (v2076_data + (v2048_data * v2074_data));
            float v2079_data = s3[79];
            float v2081_data = ir7[6];
            ir7[6] = (v2081_data + (v2048_data * v2079_data));
            float v2084_data = s3[91];
            float v2086_data = ir7[7];
            ir7[7] = (v2086_data + (v2048_data * v2084_data));
          }
          if (v16_lead < 12) {
            float v2092_data = r6[8];
            float v2093_data = s3[8];
            float v2095_data = ir7[0];
            ir7[0] = (v2095_data + (v2092_data * v2093_data));
            float v2098_data = s3[20];
            float v2100_data = ir7[1];
            ir7[1] = (v2100_data + (v2092_data * v2098_data));
            float v2103_data = s3[32];
            float v2105_data = ir7[2];
            ir7[2] = (v2105_data + (v2092_data * v2103_data));
            float v2108_data = s3[44];
            float v2110_data = ir7[3];
            ir7[3] = (v2110_data + (v2092_data * v2108_data));
            float v2113_data = s3[56];
            float v2115_data = ir7[4];
            ir7[4] = (v2115_data + (v2092_data * v2113_data));
            float v2118_data = s3[68];
            float v2120_data = ir7[5];
            ir7[5] = (v2120_data + (v2092_data * v2118_data));
            float v2123_data = s3[80];
            float v2125_data = ir7[6];
            ir7[6] = (v2125_data + (v2092_data * v2123_data));
            float v2128_data = s3[92];
            float v2130_data = ir7[7];
            ir7[7] = (v2130_data + (v2092_data * v2128_data));
          }
          if (v16_lead < 12) {
            float v2136_data = r6[9];
            float v2137_data = s3[9];
            float v2139_data = ir7[0];
            ir7[0] = (v2139_data + (v2136_data * v2137_data));
            float v2142_data = s3[21];
            float v2144_data = ir7[1];
            ir7[1] = (v2144_data + (v2136_data * v2142_data));
            float v2147_data = s3[33];
            float v2149_data = ir7[2];
            ir7[2] = (v2149_data + (v2136_data * v2147_data));
            float v2152_data = s3[45];
            float v2154_data = ir7[3];
            ir7[3] = (v2154_data + (v2136_data * v2152_data));
            float v2157_data = s3[57];
            float v2159_data = ir7[4];
            ir7[4] = (v2159_data + (v2136_data * v2157_data));
            float v2162_data = s3[69];
            float v2164_data = ir7[5];
            ir7[5] = (v2164_data + (v2136_data * v2162_data));
            float v2167_data = s3[81];
            float v2169_data = ir7[6];
            ir7[6] = (v2169_data + (v2136_data * v2167_data));
            float v2172_data = s3[93];
            float v2174_data = ir7[7];
            ir7[7] = (v2174_data + (v2136_data * v2172_data));
          }
          if (v16_lead < 12) {
            float v2180_data = r6[10];
            float v2181_data = s3[10];
            float v2183_data = ir7[0];
            ir7[0] = (v2183_data + (v2180_data * v2181_data));
            float v2186_data = s3[22];
            float v2188_data = ir7[1];
            ir7[1] = (v2188_data + (v2180_data * v2186_data));
            float v2191_data = s3[34];
            float v2193_data = ir7[2];
            ir7[2] = (v2193_data + (v2180_data * v2191_data));
            float v2196_data = s3[46];
            float v2198_data = ir7[3];
            ir7[3] = (v2198_data + (v2180_data * v2196_data));
            float v2201_data = s3[58];
            float v2203_data = ir7[4];
            ir7[4] = (v2203_data + (v2180_data * v2201_data));
            float v2206_data = s3[70];
            float v2208_data = ir7[5];
            ir7[5] = (v2208_data + (v2180_data * v2206_data));
            float v2211_data = s3[82];
            float v2213_data = ir7[6];
            ir7[6] = (v2213_data + (v2180_data * v2211_data));
            float v2216_data = s3[94];
            float v2218_data = ir7[7];
            ir7[7] = (v2218_data + (v2180_data * v2216_data));
          }
          if (v16_lead < 12) {
            float v2224_data = r6[11];
            float v2225_data = s3[11];
            float v2227_data = ir7[0];
            ir7[0] = (v2227_data + (v2224_data * v2225_data));
            float v2230_data = s3[23];
            float v2232_data = ir7[1];
            ir7[1] = (v2232_data + (v2224_data * v2230_data));
            float v2235_data = s3[35];
            float v2237_data = ir7[2];
            ir7[2] = (v2237_data + (v2224_data * v2235_data));
            float v2240_data = s3[47];
            float v2242_data = ir7[3];
            ir7[3] = (v2242_data + (v2224_data * v2240_data));
            float v2245_data = s3[59];
            float v2247_data = ir7[4];
            ir7[4] = (v2247_data + (v2224_data * v2245_data));
            float v2250_data = s3[71];
            float v2252_data = ir7[5];
            ir7[5] = (v2252_data + (v2224_data * v2250_data));
            float v2255_data = s3[83];
            float v2257_data = ir7[6];
            ir7[6] = (v2257_data + (v2224_data * v2255_data));
            float v2260_data = s3[95];
            float v2262_data = ir7[7];
            ir7[7] = (v2262_data + (v2224_data * v2260_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v2268_n1 = 0; v2268_n1 < 8; ++v2268_n1) {
              int32_t v2269_a = 0 + v2268_n1;
              float v2271_data = ir7[v2268_n1];
              int32_t v2272_a = 0 + v2268_n1;
              float v2274_data = r5[v2268_n1];
              r7[v2268_n1] = (v2274_data + v2271_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v2281_i1 = 0; v2281_i1 < 8; ++v2281_i1) {
              int32_t v2282_a = 0 + v2281_i1;
              float v2284_data = r7[v2281_i1];
              glb_m0[(v16_lead + (v2281_i1 * 12))] = v2284_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

