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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
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
          int32_t v12_lead = threadIdx.x % 16;
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 12; ++v14_i1) {
              int32_t v20_a = v14_i1 * 12;
              int32_t v21_a = v12_lead + v20_a;
              float v29_data = __ldcg(&glb_m1[(v12_lead + v20_a)]);
              int32_t v30_a = 0 + v14_i1;
              r0[v30_a] = v29_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
              int32_t v43_a = v37_i1 * 12;
              int32_t v44_a = v12_lead + v43_a;
              float v52_data = __ldcg(&glb_m3[(v12_lead + v43_a)]);
              int32_t v53_a = 0 + v37_i1;
              r2[v53_a] = v52_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          if (v12_lead < 12) {
            float v60_data = r0[0];
            float v61_data = s0[0];
            float v63_data = ir1[0];
            ir1[0] = (v63_data + (v60_data * v61_data));
            float v66_data = s0[12];
            float v68_data = ir1[1];
            ir1[1] = (v68_data + (v60_data * v66_data));
            float v71_data = s0[24];
            float v73_data = ir1[2];
            ir1[2] = (v73_data + (v60_data * v71_data));
            float v76_data = s0[36];
            float v78_data = ir1[3];
            ir1[3] = (v78_data + (v60_data * v76_data));
            float v81_data = s0[48];
            float v83_data = ir1[4];
            ir1[4] = (v83_data + (v60_data * v81_data));
            float v86_data = s0[60];
            float v88_data = ir1[5];
            ir1[5] = (v88_data + (v60_data * v86_data));
            float v91_data = s0[72];
            float v93_data = ir1[6];
            ir1[6] = (v93_data + (v60_data * v91_data));
            float v96_data = s0[84];
            float v98_data = ir1[7];
            ir1[7] = (v98_data + (v60_data * v96_data));
          }
          if (v12_lead < 12) {
            float v104_data = r0[1];
            float v105_data = s0[1];
            float v107_data = ir1[0];
            ir1[0] = (v107_data + (v104_data * v105_data));
            float v110_data = s0[13];
            float v112_data = ir1[1];
            ir1[1] = (v112_data + (v104_data * v110_data));
            float v115_data = s0[25];
            float v117_data = ir1[2];
            ir1[2] = (v117_data + (v104_data * v115_data));
            float v120_data = s0[37];
            float v122_data = ir1[3];
            ir1[3] = (v122_data + (v104_data * v120_data));
            float v125_data = s0[49];
            float v127_data = ir1[4];
            ir1[4] = (v127_data + (v104_data * v125_data));
            float v130_data = s0[61];
            float v132_data = ir1[5];
            ir1[5] = (v132_data + (v104_data * v130_data));
            float v135_data = s0[73];
            float v137_data = ir1[6];
            ir1[6] = (v137_data + (v104_data * v135_data));
            float v140_data = s0[85];
            float v142_data = ir1[7];
            ir1[7] = (v142_data + (v104_data * v140_data));
          }
          if (v12_lead < 12) {
            float v148_data = r0[2];
            float v149_data = s0[2];
            float v151_data = ir1[0];
            ir1[0] = (v151_data + (v148_data * v149_data));
            float v154_data = s0[14];
            float v156_data = ir1[1];
            ir1[1] = (v156_data + (v148_data * v154_data));
            float v159_data = s0[26];
            float v161_data = ir1[2];
            ir1[2] = (v161_data + (v148_data * v159_data));
            float v164_data = s0[38];
            float v166_data = ir1[3];
            ir1[3] = (v166_data + (v148_data * v164_data));
            float v169_data = s0[50];
            float v171_data = ir1[4];
            ir1[4] = (v171_data + (v148_data * v169_data));
            float v174_data = s0[62];
            float v176_data = ir1[5];
            ir1[5] = (v176_data + (v148_data * v174_data));
            float v179_data = s0[74];
            float v181_data = ir1[6];
            ir1[6] = (v181_data + (v148_data * v179_data));
            float v184_data = s0[86];
            float v186_data = ir1[7];
            ir1[7] = (v186_data + (v148_data * v184_data));
          }
          if (v12_lead < 12) {
            float v192_data = r0[3];
            float v193_data = s0[3];
            float v195_data = ir1[0];
            ir1[0] = (v195_data + (v192_data * v193_data));
            float v198_data = s0[15];
            float v200_data = ir1[1];
            ir1[1] = (v200_data + (v192_data * v198_data));
            float v203_data = s0[27];
            float v205_data = ir1[2];
            ir1[2] = (v205_data + (v192_data * v203_data));
            float v208_data = s0[39];
            float v210_data = ir1[3];
            ir1[3] = (v210_data + (v192_data * v208_data));
            float v213_data = s0[51];
            float v215_data = ir1[4];
            ir1[4] = (v215_data + (v192_data * v213_data));
            float v218_data = s0[63];
            float v220_data = ir1[5];
            ir1[5] = (v220_data + (v192_data * v218_data));
            float v223_data = s0[75];
            float v225_data = ir1[6];
            ir1[6] = (v225_data + (v192_data * v223_data));
            float v228_data = s0[87];
            float v230_data = ir1[7];
            ir1[7] = (v230_data + (v192_data * v228_data));
          }
          if (v12_lead < 12) {
            float v236_data = r0[4];
            float v237_data = s0[4];
            float v239_data = ir1[0];
            ir1[0] = (v239_data + (v236_data * v237_data));
            float v242_data = s0[16];
            float v244_data = ir1[1];
            ir1[1] = (v244_data + (v236_data * v242_data));
            float v247_data = s0[28];
            float v249_data = ir1[2];
            ir1[2] = (v249_data + (v236_data * v247_data));
            float v252_data = s0[40];
            float v254_data = ir1[3];
            ir1[3] = (v254_data + (v236_data * v252_data));
            float v257_data = s0[52];
            float v259_data = ir1[4];
            ir1[4] = (v259_data + (v236_data * v257_data));
            float v262_data = s0[64];
            float v264_data = ir1[5];
            ir1[5] = (v264_data + (v236_data * v262_data));
            float v267_data = s0[76];
            float v269_data = ir1[6];
            ir1[6] = (v269_data + (v236_data * v267_data));
            float v272_data = s0[88];
            float v274_data = ir1[7];
            ir1[7] = (v274_data + (v236_data * v272_data));
          }
          if (v12_lead < 12) {
            float v280_data = r0[5];
            float v281_data = s0[5];
            float v283_data = ir1[0];
            ir1[0] = (v283_data + (v280_data * v281_data));
            float v286_data = s0[17];
            float v288_data = ir1[1];
            ir1[1] = (v288_data + (v280_data * v286_data));
            float v291_data = s0[29];
            float v293_data = ir1[2];
            ir1[2] = (v293_data + (v280_data * v291_data));
            float v296_data = s0[41];
            float v298_data = ir1[3];
            ir1[3] = (v298_data + (v280_data * v296_data));
            float v301_data = s0[53];
            float v303_data = ir1[4];
            ir1[4] = (v303_data + (v280_data * v301_data));
            float v306_data = s0[65];
            float v308_data = ir1[5];
            ir1[5] = (v308_data + (v280_data * v306_data));
            float v311_data = s0[77];
            float v313_data = ir1[6];
            ir1[6] = (v313_data + (v280_data * v311_data));
            float v316_data = s0[89];
            float v318_data = ir1[7];
            ir1[7] = (v318_data + (v280_data * v316_data));
          }
          if (v12_lead < 12) {
            float v324_data = r0[6];
            float v325_data = s0[6];
            float v327_data = ir1[0];
            ir1[0] = (v327_data + (v324_data * v325_data));
            float v330_data = s0[18];
            float v332_data = ir1[1];
            ir1[1] = (v332_data + (v324_data * v330_data));
            float v335_data = s0[30];
            float v337_data = ir1[2];
            ir1[2] = (v337_data + (v324_data * v335_data));
            float v340_data = s0[42];
            float v342_data = ir1[3];
            ir1[3] = (v342_data + (v324_data * v340_data));
            float v345_data = s0[54];
            float v347_data = ir1[4];
            ir1[4] = (v347_data + (v324_data * v345_data));
            float v350_data = s0[66];
            float v352_data = ir1[5];
            ir1[5] = (v352_data + (v324_data * v350_data));
            float v355_data = s0[78];
            float v357_data = ir1[6];
            ir1[6] = (v357_data + (v324_data * v355_data));
            float v360_data = s0[90];
            float v362_data = ir1[7];
            ir1[7] = (v362_data + (v324_data * v360_data));
          }
          if (v12_lead < 12) {
            float v368_data = r0[7];
            float v369_data = s0[7];
            float v371_data = ir1[0];
            ir1[0] = (v371_data + (v368_data * v369_data));
            float v374_data = s0[19];
            float v376_data = ir1[1];
            ir1[1] = (v376_data + (v368_data * v374_data));
            float v379_data = s0[31];
            float v381_data = ir1[2];
            ir1[2] = (v381_data + (v368_data * v379_data));
            float v384_data = s0[43];
            float v386_data = ir1[3];
            ir1[3] = (v386_data + (v368_data * v384_data));
            float v389_data = s0[55];
            float v391_data = ir1[4];
            ir1[4] = (v391_data + (v368_data * v389_data));
            float v394_data = s0[67];
            float v396_data = ir1[5];
            ir1[5] = (v396_data + (v368_data * v394_data));
            float v399_data = s0[79];
            float v401_data = ir1[6];
            ir1[6] = (v401_data + (v368_data * v399_data));
            float v404_data = s0[91];
            float v406_data = ir1[7];
            ir1[7] = (v406_data + (v368_data * v404_data));
          }
          if (v12_lead < 12) {
            float v412_data = r0[8];
            float v413_data = s0[8];
            float v415_data = ir1[0];
            ir1[0] = (v415_data + (v412_data * v413_data));
            float v418_data = s0[20];
            float v420_data = ir1[1];
            ir1[1] = (v420_data + (v412_data * v418_data));
            float v423_data = s0[32];
            float v425_data = ir1[2];
            ir1[2] = (v425_data + (v412_data * v423_data));
            float v428_data = s0[44];
            float v430_data = ir1[3];
            ir1[3] = (v430_data + (v412_data * v428_data));
            float v433_data = s0[56];
            float v435_data = ir1[4];
            ir1[4] = (v435_data + (v412_data * v433_data));
            float v438_data = s0[68];
            float v440_data = ir1[5];
            ir1[5] = (v440_data + (v412_data * v438_data));
            float v443_data = s0[80];
            float v445_data = ir1[6];
            ir1[6] = (v445_data + (v412_data * v443_data));
            float v448_data = s0[92];
            float v450_data = ir1[7];
            ir1[7] = (v450_data + (v412_data * v448_data));
          }
          if (v12_lead < 12) {
            float v456_data = r0[9];
            float v457_data = s0[9];
            float v459_data = ir1[0];
            ir1[0] = (v459_data + (v456_data * v457_data));
            float v462_data = s0[21];
            float v464_data = ir1[1];
            ir1[1] = (v464_data + (v456_data * v462_data));
            float v467_data = s0[33];
            float v469_data = ir1[2];
            ir1[2] = (v469_data + (v456_data * v467_data));
            float v472_data = s0[45];
            float v474_data = ir1[3];
            ir1[3] = (v474_data + (v456_data * v472_data));
            float v477_data = s0[57];
            float v479_data = ir1[4];
            ir1[4] = (v479_data + (v456_data * v477_data));
            float v482_data = s0[69];
            float v484_data = ir1[5];
            ir1[5] = (v484_data + (v456_data * v482_data));
            float v487_data = s0[81];
            float v489_data = ir1[6];
            ir1[6] = (v489_data + (v456_data * v487_data));
            float v492_data = s0[93];
            float v494_data = ir1[7];
            ir1[7] = (v494_data + (v456_data * v492_data));
          }
          if (v12_lead < 12) {
            float v500_data = r0[10];
            float v501_data = s0[10];
            float v503_data = ir1[0];
            ir1[0] = (v503_data + (v500_data * v501_data));
            float v506_data = s0[22];
            float v508_data = ir1[1];
            ir1[1] = (v508_data + (v500_data * v506_data));
            float v511_data = s0[34];
            float v513_data = ir1[2];
            ir1[2] = (v513_data + (v500_data * v511_data));
            float v516_data = s0[46];
            float v518_data = ir1[3];
            ir1[3] = (v518_data + (v500_data * v516_data));
            float v521_data = s0[58];
            float v523_data = ir1[4];
            ir1[4] = (v523_data + (v500_data * v521_data));
            float v526_data = s0[70];
            float v528_data = ir1[5];
            ir1[5] = (v528_data + (v500_data * v526_data));
            float v531_data = s0[82];
            float v533_data = ir1[6];
            ir1[6] = (v533_data + (v500_data * v531_data));
            float v536_data = s0[94];
            float v538_data = ir1[7];
            ir1[7] = (v538_data + (v500_data * v536_data));
          }
          if (v12_lead < 12) {
            float v544_data = r0[11];
            float v545_data = s0[11];
            float v547_data = ir1[0];
            ir1[0] = (v547_data + (v544_data * v545_data));
            float v550_data = s0[23];
            float v552_data = ir1[1];
            ir1[1] = (v552_data + (v544_data * v550_data));
            float v555_data = s0[35];
            float v557_data = ir1[2];
            ir1[2] = (v557_data + (v544_data * v555_data));
            float v560_data = s0[47];
            float v562_data = ir1[3];
            ir1[3] = (v562_data + (v544_data * v560_data));
            float v565_data = s0[59];
            float v567_data = ir1[4];
            ir1[4] = (v567_data + (v544_data * v565_data));
            float v570_data = s0[71];
            float v572_data = ir1[5];
            ir1[5] = (v572_data + (v544_data * v570_data));
            float v575_data = s0[83];
            float v577_data = ir1[6];
            ir1[6] = (v577_data + (v544_data * v575_data));
            float v580_data = s0[95];
            float v582_data = ir1[7];
            ir1[7] = (v582_data + (v544_data * v580_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v588_n1 = 0; v588_n1 < 8; ++v588_n1) {
              int32_t v589_a = 0 + v588_n1;
              float v591_data = ir1[v588_n1];
              int32_t v592_a = 0 + v588_n1;
              r1[v588_n1] = v591_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          {
            // s1 = load{g>s}(glb_m4[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v600_i1 = 0; v600_i1 < 12; ++v600_i1) {
              int32_t v606_a = v600_i1 * 12;
              int32_t v607_a = v12_lead + v606_a;
              float v615_data = __ldcg(&glb_m5[(v12_lead + v606_a)]);
              int32_t v616_a = 0 + v600_i1;
              r4[v616_a] = v615_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          if (v12_lead < 12) {
            float v623_data = r2[0];
            float v624_data = s1[0];
            float v626_data = ir3[0];
            ir3[0] = (v626_data + (v623_data * v624_data));
            float v629_data = s1[12];
            float v631_data = ir3[1];
            ir3[1] = (v631_data + (v623_data * v629_data));
            float v634_data = s1[24];
            float v636_data = ir3[2];
            ir3[2] = (v636_data + (v623_data * v634_data));
            float v639_data = s1[36];
            float v641_data = ir3[3];
            ir3[3] = (v641_data + (v623_data * v639_data));
            float v644_data = s1[48];
            float v646_data = ir3[4];
            ir3[4] = (v646_data + (v623_data * v644_data));
            float v649_data = s1[60];
            float v651_data = ir3[5];
            ir3[5] = (v651_data + (v623_data * v649_data));
            float v654_data = s1[72];
            float v656_data = ir3[6];
            ir3[6] = (v656_data + (v623_data * v654_data));
            float v659_data = s1[84];
            float v661_data = ir3[7];
            ir3[7] = (v661_data + (v623_data * v659_data));
          }
          if (v12_lead < 12) {
            float v667_data = r2[1];
            float v668_data = s1[1];
            float v670_data = ir3[0];
            ir3[0] = (v670_data + (v667_data * v668_data));
            float v673_data = s1[13];
            float v675_data = ir3[1];
            ir3[1] = (v675_data + (v667_data * v673_data));
            float v678_data = s1[25];
            float v680_data = ir3[2];
            ir3[2] = (v680_data + (v667_data * v678_data));
            float v683_data = s1[37];
            float v685_data = ir3[3];
            ir3[3] = (v685_data + (v667_data * v683_data));
            float v688_data = s1[49];
            float v690_data = ir3[4];
            ir3[4] = (v690_data + (v667_data * v688_data));
            float v693_data = s1[61];
            float v695_data = ir3[5];
            ir3[5] = (v695_data + (v667_data * v693_data));
            float v698_data = s1[73];
            float v700_data = ir3[6];
            ir3[6] = (v700_data + (v667_data * v698_data));
            float v703_data = s1[85];
            float v705_data = ir3[7];
            ir3[7] = (v705_data + (v667_data * v703_data));
          }
          if (v12_lead < 12) {
            float v711_data = r2[2];
            float v712_data = s1[2];
            float v714_data = ir3[0];
            ir3[0] = (v714_data + (v711_data * v712_data));
            float v717_data = s1[14];
            float v719_data = ir3[1];
            ir3[1] = (v719_data + (v711_data * v717_data));
            float v722_data = s1[26];
            float v724_data = ir3[2];
            ir3[2] = (v724_data + (v711_data * v722_data));
            float v727_data = s1[38];
            float v729_data = ir3[3];
            ir3[3] = (v729_data + (v711_data * v727_data));
            float v732_data = s1[50];
            float v734_data = ir3[4];
            ir3[4] = (v734_data + (v711_data * v732_data));
            float v737_data = s1[62];
            float v739_data = ir3[5];
            ir3[5] = (v739_data + (v711_data * v737_data));
            float v742_data = s1[74];
            float v744_data = ir3[6];
            ir3[6] = (v744_data + (v711_data * v742_data));
            float v747_data = s1[86];
            float v749_data = ir3[7];
            ir3[7] = (v749_data + (v711_data * v747_data));
          }
          if (v12_lead < 12) {
            float v755_data = r2[3];
            float v756_data = s1[3];
            float v758_data = ir3[0];
            ir3[0] = (v758_data + (v755_data * v756_data));
            float v761_data = s1[15];
            float v763_data = ir3[1];
            ir3[1] = (v763_data + (v755_data * v761_data));
            float v766_data = s1[27];
            float v768_data = ir3[2];
            ir3[2] = (v768_data + (v755_data * v766_data));
            float v771_data = s1[39];
            float v773_data = ir3[3];
            ir3[3] = (v773_data + (v755_data * v771_data));
            float v776_data = s1[51];
            float v778_data = ir3[4];
            ir3[4] = (v778_data + (v755_data * v776_data));
            float v781_data = s1[63];
            float v783_data = ir3[5];
            ir3[5] = (v783_data + (v755_data * v781_data));
            float v786_data = s1[75];
            float v788_data = ir3[6];
            ir3[6] = (v788_data + (v755_data * v786_data));
            float v791_data = s1[87];
            float v793_data = ir3[7];
            ir3[7] = (v793_data + (v755_data * v791_data));
          }
          if (v12_lead < 12) {
            float v799_data = r2[4];
            float v800_data = s1[4];
            float v802_data = ir3[0];
            ir3[0] = (v802_data + (v799_data * v800_data));
            float v805_data = s1[16];
            float v807_data = ir3[1];
            ir3[1] = (v807_data + (v799_data * v805_data));
            float v810_data = s1[28];
            float v812_data = ir3[2];
            ir3[2] = (v812_data + (v799_data * v810_data));
            float v815_data = s1[40];
            float v817_data = ir3[3];
            ir3[3] = (v817_data + (v799_data * v815_data));
            float v820_data = s1[52];
            float v822_data = ir3[4];
            ir3[4] = (v822_data + (v799_data * v820_data));
            float v825_data = s1[64];
            float v827_data = ir3[5];
            ir3[5] = (v827_data + (v799_data * v825_data));
            float v830_data = s1[76];
            float v832_data = ir3[6];
            ir3[6] = (v832_data + (v799_data * v830_data));
            float v835_data = s1[88];
            float v837_data = ir3[7];
            ir3[7] = (v837_data + (v799_data * v835_data));
          }
          if (v12_lead < 12) {
            float v843_data = r2[5];
            float v844_data = s1[5];
            float v846_data = ir3[0];
            ir3[0] = (v846_data + (v843_data * v844_data));
            float v849_data = s1[17];
            float v851_data = ir3[1];
            ir3[1] = (v851_data + (v843_data * v849_data));
            float v854_data = s1[29];
            float v856_data = ir3[2];
            ir3[2] = (v856_data + (v843_data * v854_data));
            float v859_data = s1[41];
            float v861_data = ir3[3];
            ir3[3] = (v861_data + (v843_data * v859_data));
            float v864_data = s1[53];
            float v866_data = ir3[4];
            ir3[4] = (v866_data + (v843_data * v864_data));
            float v869_data = s1[65];
            float v871_data = ir3[5];
            ir3[5] = (v871_data + (v843_data * v869_data));
            float v874_data = s1[77];
            float v876_data = ir3[6];
            ir3[6] = (v876_data + (v843_data * v874_data));
            float v879_data = s1[89];
            float v881_data = ir3[7];
            ir3[7] = (v881_data + (v843_data * v879_data));
          }
          if (v12_lead < 12) {
            float v887_data = r2[6];
            float v888_data = s1[6];
            float v890_data = ir3[0];
            ir3[0] = (v890_data + (v887_data * v888_data));
            float v893_data = s1[18];
            float v895_data = ir3[1];
            ir3[1] = (v895_data + (v887_data * v893_data));
            float v898_data = s1[30];
            float v900_data = ir3[2];
            ir3[2] = (v900_data + (v887_data * v898_data));
            float v903_data = s1[42];
            float v905_data = ir3[3];
            ir3[3] = (v905_data + (v887_data * v903_data));
            float v908_data = s1[54];
            float v910_data = ir3[4];
            ir3[4] = (v910_data + (v887_data * v908_data));
            float v913_data = s1[66];
            float v915_data = ir3[5];
            ir3[5] = (v915_data + (v887_data * v913_data));
            float v918_data = s1[78];
            float v920_data = ir3[6];
            ir3[6] = (v920_data + (v887_data * v918_data));
            float v923_data = s1[90];
            float v925_data = ir3[7];
            ir3[7] = (v925_data + (v887_data * v923_data));
          }
          if (v12_lead < 12) {
            float v931_data = r2[7];
            float v932_data = s1[7];
            float v934_data = ir3[0];
            ir3[0] = (v934_data + (v931_data * v932_data));
            float v937_data = s1[19];
            float v939_data = ir3[1];
            ir3[1] = (v939_data + (v931_data * v937_data));
            float v942_data = s1[31];
            float v944_data = ir3[2];
            ir3[2] = (v944_data + (v931_data * v942_data));
            float v947_data = s1[43];
            float v949_data = ir3[3];
            ir3[3] = (v949_data + (v931_data * v947_data));
            float v952_data = s1[55];
            float v954_data = ir3[4];
            ir3[4] = (v954_data + (v931_data * v952_data));
            float v957_data = s1[67];
            float v959_data = ir3[5];
            ir3[5] = (v959_data + (v931_data * v957_data));
            float v962_data = s1[79];
            float v964_data = ir3[6];
            ir3[6] = (v964_data + (v931_data * v962_data));
            float v967_data = s1[91];
            float v969_data = ir3[7];
            ir3[7] = (v969_data + (v931_data * v967_data));
          }
          if (v12_lead < 12) {
            float v975_data = r2[8];
            float v976_data = s1[8];
            float v978_data = ir3[0];
            ir3[0] = (v978_data + (v975_data * v976_data));
            float v981_data = s1[20];
            float v983_data = ir3[1];
            ir3[1] = (v983_data + (v975_data * v981_data));
            float v986_data = s1[32];
            float v988_data = ir3[2];
            ir3[2] = (v988_data + (v975_data * v986_data));
            float v991_data = s1[44];
            float v993_data = ir3[3];
            ir3[3] = (v993_data + (v975_data * v991_data));
            float v996_data = s1[56];
            float v998_data = ir3[4];
            ir3[4] = (v998_data + (v975_data * v996_data));
            float v1001_data = s1[68];
            float v1003_data = ir3[5];
            ir3[5] = (v1003_data + (v975_data * v1001_data));
            float v1006_data = s1[80];
            float v1008_data = ir3[6];
            ir3[6] = (v1008_data + (v975_data * v1006_data));
            float v1011_data = s1[92];
            float v1013_data = ir3[7];
            ir3[7] = (v1013_data + (v975_data * v1011_data));
          }
          if (v12_lead < 12) {
            float v1019_data = r2[9];
            float v1020_data = s1[9];
            float v1022_data = ir3[0];
            ir3[0] = (v1022_data + (v1019_data * v1020_data));
            float v1025_data = s1[21];
            float v1027_data = ir3[1];
            ir3[1] = (v1027_data + (v1019_data * v1025_data));
            float v1030_data = s1[33];
            float v1032_data = ir3[2];
            ir3[2] = (v1032_data + (v1019_data * v1030_data));
            float v1035_data = s1[45];
            float v1037_data = ir3[3];
            ir3[3] = (v1037_data + (v1019_data * v1035_data));
            float v1040_data = s1[57];
            float v1042_data = ir3[4];
            ir3[4] = (v1042_data + (v1019_data * v1040_data));
            float v1045_data = s1[69];
            float v1047_data = ir3[5];
            ir3[5] = (v1047_data + (v1019_data * v1045_data));
            float v1050_data = s1[81];
            float v1052_data = ir3[6];
            ir3[6] = (v1052_data + (v1019_data * v1050_data));
            float v1055_data = s1[93];
            float v1057_data = ir3[7];
            ir3[7] = (v1057_data + (v1019_data * v1055_data));
          }
          if (v12_lead < 12) {
            float v1063_data = r2[10];
            float v1064_data = s1[10];
            float v1066_data = ir3[0];
            ir3[0] = (v1066_data + (v1063_data * v1064_data));
            float v1069_data = s1[22];
            float v1071_data = ir3[1];
            ir3[1] = (v1071_data + (v1063_data * v1069_data));
            float v1074_data = s1[34];
            float v1076_data = ir3[2];
            ir3[2] = (v1076_data + (v1063_data * v1074_data));
            float v1079_data = s1[46];
            float v1081_data = ir3[3];
            ir3[3] = (v1081_data + (v1063_data * v1079_data));
            float v1084_data = s1[58];
            float v1086_data = ir3[4];
            ir3[4] = (v1086_data + (v1063_data * v1084_data));
            float v1089_data = s1[70];
            float v1091_data = ir3[5];
            ir3[5] = (v1091_data + (v1063_data * v1089_data));
            float v1094_data = s1[82];
            float v1096_data = ir3[6];
            ir3[6] = (v1096_data + (v1063_data * v1094_data));
            float v1099_data = s1[94];
            float v1101_data = ir3[7];
            ir3[7] = (v1101_data + (v1063_data * v1099_data));
          }
          if (v12_lead < 12) {
            float v1107_data = r2[11];
            float v1108_data = s1[11];
            float v1110_data = ir3[0];
            ir3[0] = (v1110_data + (v1107_data * v1108_data));
            float v1113_data = s1[23];
            float v1115_data = ir3[1];
            ir3[1] = (v1115_data + (v1107_data * v1113_data));
            float v1118_data = s1[35];
            float v1120_data = ir3[2];
            ir3[2] = (v1120_data + (v1107_data * v1118_data));
            float v1123_data = s1[47];
            float v1125_data = ir3[3];
            ir3[3] = (v1125_data + (v1107_data * v1123_data));
            float v1128_data = s1[59];
            float v1130_data = ir3[4];
            ir3[4] = (v1130_data + (v1107_data * v1128_data));
            float v1133_data = s1[71];
            float v1135_data = ir3[5];
            ir3[5] = (v1135_data + (v1107_data * v1133_data));
            float v1138_data = s1[83];
            float v1140_data = ir3[6];
            ir3[6] = (v1140_data + (v1107_data * v1138_data));
            float v1143_data = s1[95];
            float v1145_data = ir3[7];
            ir3[7] = (v1145_data + (v1107_data * v1143_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v1151_n1 = 0; v1151_n1 < 8; ++v1151_n1) {
              int32_t v1152_a = 0 + v1151_n1;
              float v1154_data = ir3[v1151_n1];
              int32_t v1155_a = 0 + v1151_n1;
              float v1157_data = r1[v1151_n1];
              int32_t v1159_a = 0 + v1151_n1;
              r3[v1151_n1] = (v1157_data + v1154_data);
            }
          }
          __syncwarp();
          float* __restrict__ s2 = &localShrMem0[0];
          {
            // s2 = load{g>s}(glb_m6[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r4 = load{g>r}(glb_m5););
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v1167_i1 = 0; v1167_i1 < 12; ++v1167_i1) {
              int32_t v1173_a = v1167_i1 * 12;
              int32_t v1174_a = v12_lead + v1173_a;
              float v1182_data = __ldcg(&glb_m7[(v12_lead + v1173_a)]);
              int32_t v1183_a = 0 + v1167_i1;
              r6[v1183_a] = v1182_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r5[8]{};
          __syncwarp();
          // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          if (v12_lead < 12) {
            float v1190_data = r4[0];
            float v1191_data = s2[0];
            float v1193_data = ir5[0];
            ir5[0] = (v1193_data + (v1190_data * v1191_data));
            float v1196_data = s2[12];
            float v1198_data = ir5[1];
            ir5[1] = (v1198_data + (v1190_data * v1196_data));
            float v1201_data = s2[24];
            float v1203_data = ir5[2];
            ir5[2] = (v1203_data + (v1190_data * v1201_data));
            float v1206_data = s2[36];
            float v1208_data = ir5[3];
            ir5[3] = (v1208_data + (v1190_data * v1206_data));
            float v1211_data = s2[48];
            float v1213_data = ir5[4];
            ir5[4] = (v1213_data + (v1190_data * v1211_data));
            float v1216_data = s2[60];
            float v1218_data = ir5[5];
            ir5[5] = (v1218_data + (v1190_data * v1216_data));
            float v1221_data = s2[72];
            float v1223_data = ir5[6];
            ir5[6] = (v1223_data + (v1190_data * v1221_data));
            float v1226_data = s2[84];
            float v1228_data = ir5[7];
            ir5[7] = (v1228_data + (v1190_data * v1226_data));
          }
          if (v12_lead < 12) {
            float v1234_data = r4[1];
            float v1235_data = s2[1];
            float v1237_data = ir5[0];
            ir5[0] = (v1237_data + (v1234_data * v1235_data));
            float v1240_data = s2[13];
            float v1242_data = ir5[1];
            ir5[1] = (v1242_data + (v1234_data * v1240_data));
            float v1245_data = s2[25];
            float v1247_data = ir5[2];
            ir5[2] = (v1247_data + (v1234_data * v1245_data));
            float v1250_data = s2[37];
            float v1252_data = ir5[3];
            ir5[3] = (v1252_data + (v1234_data * v1250_data));
            float v1255_data = s2[49];
            float v1257_data = ir5[4];
            ir5[4] = (v1257_data + (v1234_data * v1255_data));
            float v1260_data = s2[61];
            float v1262_data = ir5[5];
            ir5[5] = (v1262_data + (v1234_data * v1260_data));
            float v1265_data = s2[73];
            float v1267_data = ir5[6];
            ir5[6] = (v1267_data + (v1234_data * v1265_data));
            float v1270_data = s2[85];
            float v1272_data = ir5[7];
            ir5[7] = (v1272_data + (v1234_data * v1270_data));
          }
          if (v12_lead < 12) {
            float v1278_data = r4[2];
            float v1279_data = s2[2];
            float v1281_data = ir5[0];
            ir5[0] = (v1281_data + (v1278_data * v1279_data));
            float v1284_data = s2[14];
            float v1286_data = ir5[1];
            ir5[1] = (v1286_data + (v1278_data * v1284_data));
            float v1289_data = s2[26];
            float v1291_data = ir5[2];
            ir5[2] = (v1291_data + (v1278_data * v1289_data));
            float v1294_data = s2[38];
            float v1296_data = ir5[3];
            ir5[3] = (v1296_data + (v1278_data * v1294_data));
            float v1299_data = s2[50];
            float v1301_data = ir5[4];
            ir5[4] = (v1301_data + (v1278_data * v1299_data));
            float v1304_data = s2[62];
            float v1306_data = ir5[5];
            ir5[5] = (v1306_data + (v1278_data * v1304_data));
            float v1309_data = s2[74];
            float v1311_data = ir5[6];
            ir5[6] = (v1311_data + (v1278_data * v1309_data));
            float v1314_data = s2[86];
            float v1316_data = ir5[7];
            ir5[7] = (v1316_data + (v1278_data * v1314_data));
          }
          if (v12_lead < 12) {
            float v1322_data = r4[3];
            float v1323_data = s2[3];
            float v1325_data = ir5[0];
            ir5[0] = (v1325_data + (v1322_data * v1323_data));
            float v1328_data = s2[15];
            float v1330_data = ir5[1];
            ir5[1] = (v1330_data + (v1322_data * v1328_data));
            float v1333_data = s2[27];
            float v1335_data = ir5[2];
            ir5[2] = (v1335_data + (v1322_data * v1333_data));
            float v1338_data = s2[39];
            float v1340_data = ir5[3];
            ir5[3] = (v1340_data + (v1322_data * v1338_data));
            float v1343_data = s2[51];
            float v1345_data = ir5[4];
            ir5[4] = (v1345_data + (v1322_data * v1343_data));
            float v1348_data = s2[63];
            float v1350_data = ir5[5];
            ir5[5] = (v1350_data + (v1322_data * v1348_data));
            float v1353_data = s2[75];
            float v1355_data = ir5[6];
            ir5[6] = (v1355_data + (v1322_data * v1353_data));
            float v1358_data = s2[87];
            float v1360_data = ir5[7];
            ir5[7] = (v1360_data + (v1322_data * v1358_data));
          }
          if (v12_lead < 12) {
            float v1366_data = r4[4];
            float v1367_data = s2[4];
            float v1369_data = ir5[0];
            ir5[0] = (v1369_data + (v1366_data * v1367_data));
            float v1372_data = s2[16];
            float v1374_data = ir5[1];
            ir5[1] = (v1374_data + (v1366_data * v1372_data));
            float v1377_data = s2[28];
            float v1379_data = ir5[2];
            ir5[2] = (v1379_data + (v1366_data * v1377_data));
            float v1382_data = s2[40];
            float v1384_data = ir5[3];
            ir5[3] = (v1384_data + (v1366_data * v1382_data));
            float v1387_data = s2[52];
            float v1389_data = ir5[4];
            ir5[4] = (v1389_data + (v1366_data * v1387_data));
            float v1392_data = s2[64];
            float v1394_data = ir5[5];
            ir5[5] = (v1394_data + (v1366_data * v1392_data));
            float v1397_data = s2[76];
            float v1399_data = ir5[6];
            ir5[6] = (v1399_data + (v1366_data * v1397_data));
            float v1402_data = s2[88];
            float v1404_data = ir5[7];
            ir5[7] = (v1404_data + (v1366_data * v1402_data));
          }
          if (v12_lead < 12) {
            float v1410_data = r4[5];
            float v1411_data = s2[5];
            float v1413_data = ir5[0];
            ir5[0] = (v1413_data + (v1410_data * v1411_data));
            float v1416_data = s2[17];
            float v1418_data = ir5[1];
            ir5[1] = (v1418_data + (v1410_data * v1416_data));
            float v1421_data = s2[29];
            float v1423_data = ir5[2];
            ir5[2] = (v1423_data + (v1410_data * v1421_data));
            float v1426_data = s2[41];
            float v1428_data = ir5[3];
            ir5[3] = (v1428_data + (v1410_data * v1426_data));
            float v1431_data = s2[53];
            float v1433_data = ir5[4];
            ir5[4] = (v1433_data + (v1410_data * v1431_data));
            float v1436_data = s2[65];
            float v1438_data = ir5[5];
            ir5[5] = (v1438_data + (v1410_data * v1436_data));
            float v1441_data = s2[77];
            float v1443_data = ir5[6];
            ir5[6] = (v1443_data + (v1410_data * v1441_data));
            float v1446_data = s2[89];
            float v1448_data = ir5[7];
            ir5[7] = (v1448_data + (v1410_data * v1446_data));
          }
          if (v12_lead < 12) {
            float v1454_data = r4[6];
            float v1455_data = s2[6];
            float v1457_data = ir5[0];
            ir5[0] = (v1457_data + (v1454_data * v1455_data));
            float v1460_data = s2[18];
            float v1462_data = ir5[1];
            ir5[1] = (v1462_data + (v1454_data * v1460_data));
            float v1465_data = s2[30];
            float v1467_data = ir5[2];
            ir5[2] = (v1467_data + (v1454_data * v1465_data));
            float v1470_data = s2[42];
            float v1472_data = ir5[3];
            ir5[3] = (v1472_data + (v1454_data * v1470_data));
            float v1475_data = s2[54];
            float v1477_data = ir5[4];
            ir5[4] = (v1477_data + (v1454_data * v1475_data));
            float v1480_data = s2[66];
            float v1482_data = ir5[5];
            ir5[5] = (v1482_data + (v1454_data * v1480_data));
            float v1485_data = s2[78];
            float v1487_data = ir5[6];
            ir5[6] = (v1487_data + (v1454_data * v1485_data));
            float v1490_data = s2[90];
            float v1492_data = ir5[7];
            ir5[7] = (v1492_data + (v1454_data * v1490_data));
          }
          if (v12_lead < 12) {
            float v1498_data = r4[7];
            float v1499_data = s2[7];
            float v1501_data = ir5[0];
            ir5[0] = (v1501_data + (v1498_data * v1499_data));
            float v1504_data = s2[19];
            float v1506_data = ir5[1];
            ir5[1] = (v1506_data + (v1498_data * v1504_data));
            float v1509_data = s2[31];
            float v1511_data = ir5[2];
            ir5[2] = (v1511_data + (v1498_data * v1509_data));
            float v1514_data = s2[43];
            float v1516_data = ir5[3];
            ir5[3] = (v1516_data + (v1498_data * v1514_data));
            float v1519_data = s2[55];
            float v1521_data = ir5[4];
            ir5[4] = (v1521_data + (v1498_data * v1519_data));
            float v1524_data = s2[67];
            float v1526_data = ir5[5];
            ir5[5] = (v1526_data + (v1498_data * v1524_data));
            float v1529_data = s2[79];
            float v1531_data = ir5[6];
            ir5[6] = (v1531_data + (v1498_data * v1529_data));
            float v1534_data = s2[91];
            float v1536_data = ir5[7];
            ir5[7] = (v1536_data + (v1498_data * v1534_data));
          }
          if (v12_lead < 12) {
            float v1542_data = r4[8];
            float v1543_data = s2[8];
            float v1545_data = ir5[0];
            ir5[0] = (v1545_data + (v1542_data * v1543_data));
            float v1548_data = s2[20];
            float v1550_data = ir5[1];
            ir5[1] = (v1550_data + (v1542_data * v1548_data));
            float v1553_data = s2[32];
            float v1555_data = ir5[2];
            ir5[2] = (v1555_data + (v1542_data * v1553_data));
            float v1558_data = s2[44];
            float v1560_data = ir5[3];
            ir5[3] = (v1560_data + (v1542_data * v1558_data));
            float v1563_data = s2[56];
            float v1565_data = ir5[4];
            ir5[4] = (v1565_data + (v1542_data * v1563_data));
            float v1568_data = s2[68];
            float v1570_data = ir5[5];
            ir5[5] = (v1570_data + (v1542_data * v1568_data));
            float v1573_data = s2[80];
            float v1575_data = ir5[6];
            ir5[6] = (v1575_data + (v1542_data * v1573_data));
            float v1578_data = s2[92];
            float v1580_data = ir5[7];
            ir5[7] = (v1580_data + (v1542_data * v1578_data));
          }
          if (v12_lead < 12) {
            float v1586_data = r4[9];
            float v1587_data = s2[9];
            float v1589_data = ir5[0];
            ir5[0] = (v1589_data + (v1586_data * v1587_data));
            float v1592_data = s2[21];
            float v1594_data = ir5[1];
            ir5[1] = (v1594_data + (v1586_data * v1592_data));
            float v1597_data = s2[33];
            float v1599_data = ir5[2];
            ir5[2] = (v1599_data + (v1586_data * v1597_data));
            float v1602_data = s2[45];
            float v1604_data = ir5[3];
            ir5[3] = (v1604_data + (v1586_data * v1602_data));
            float v1607_data = s2[57];
            float v1609_data = ir5[4];
            ir5[4] = (v1609_data + (v1586_data * v1607_data));
            float v1612_data = s2[69];
            float v1614_data = ir5[5];
            ir5[5] = (v1614_data + (v1586_data * v1612_data));
            float v1617_data = s2[81];
            float v1619_data = ir5[6];
            ir5[6] = (v1619_data + (v1586_data * v1617_data));
            float v1622_data = s2[93];
            float v1624_data = ir5[7];
            ir5[7] = (v1624_data + (v1586_data * v1622_data));
          }
          if (v12_lead < 12) {
            float v1630_data = r4[10];
            float v1631_data = s2[10];
            float v1633_data = ir5[0];
            ir5[0] = (v1633_data + (v1630_data * v1631_data));
            float v1636_data = s2[22];
            float v1638_data = ir5[1];
            ir5[1] = (v1638_data + (v1630_data * v1636_data));
            float v1641_data = s2[34];
            float v1643_data = ir5[2];
            ir5[2] = (v1643_data + (v1630_data * v1641_data));
            float v1646_data = s2[46];
            float v1648_data = ir5[3];
            ir5[3] = (v1648_data + (v1630_data * v1646_data));
            float v1651_data = s2[58];
            float v1653_data = ir5[4];
            ir5[4] = (v1653_data + (v1630_data * v1651_data));
            float v1656_data = s2[70];
            float v1658_data = ir5[5];
            ir5[5] = (v1658_data + (v1630_data * v1656_data));
            float v1661_data = s2[82];
            float v1663_data = ir5[6];
            ir5[6] = (v1663_data + (v1630_data * v1661_data));
            float v1666_data = s2[94];
            float v1668_data = ir5[7];
            ir5[7] = (v1668_data + (v1630_data * v1666_data));
          }
          if (v12_lead < 12) {
            float v1674_data = r4[11];
            float v1675_data = s2[11];
            float v1677_data = ir5[0];
            ir5[0] = (v1677_data + (v1674_data * v1675_data));
            float v1680_data = s2[23];
            float v1682_data = ir5[1];
            ir5[1] = (v1682_data + (v1674_data * v1680_data));
            float v1685_data = s2[35];
            float v1687_data = ir5[2];
            ir5[2] = (v1687_data + (v1674_data * v1685_data));
            float v1690_data = s2[47];
            float v1692_data = ir5[3];
            ir5[3] = (v1692_data + (v1674_data * v1690_data));
            float v1695_data = s2[59];
            float v1697_data = ir5[4];
            ir5[4] = (v1697_data + (v1674_data * v1695_data));
            float v1700_data = s2[71];
            float v1702_data = ir5[5];
            ir5[5] = (v1702_data + (v1674_data * v1700_data));
            float v1705_data = s2[83];
            float v1707_data = ir5[6];
            ir5[6] = (v1707_data + (v1674_data * v1705_data));
            float v1710_data = s2[95];
            float v1712_data = ir5[7];
            ir5[7] = (v1712_data + (v1674_data * v1710_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v1718_n1 = 0; v1718_n1 < 8; ++v1718_n1) {
              int32_t v1719_a = 0 + v1718_n1;
              float v1721_data = ir5[v1718_n1];
              int32_t v1722_a = 0 + v1718_n1;
              float v1724_data = r3[v1718_n1];
              int32_t v1726_a = 0 + v1718_n1;
              r5[v1718_n1] = (v1724_data + v1721_data);
            }
          }
          __syncwarp();
          float* __restrict__ s3 = &localShrMem0[0];
          {
            // s3 = load{g>s}(glb_m8[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 6; i += 1) {
              cuda::memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r6 = load{g>r}(glb_m7););
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r7[8]{};
          __syncwarp();
          // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          if (v12_lead < 12) {
            float v1735_data = r6[0];
            float v1736_data = s3[0];
            float v1738_data = ir7[0];
            ir7[0] = (v1738_data + (v1735_data * v1736_data));
            float v1741_data = s3[12];
            float v1743_data = ir7[1];
            ir7[1] = (v1743_data + (v1735_data * v1741_data));
            float v1746_data = s3[24];
            float v1748_data = ir7[2];
            ir7[2] = (v1748_data + (v1735_data * v1746_data));
            float v1751_data = s3[36];
            float v1753_data = ir7[3];
            ir7[3] = (v1753_data + (v1735_data * v1751_data));
            float v1756_data = s3[48];
            float v1758_data = ir7[4];
            ir7[4] = (v1758_data + (v1735_data * v1756_data));
            float v1761_data = s3[60];
            float v1763_data = ir7[5];
            ir7[5] = (v1763_data + (v1735_data * v1761_data));
            float v1766_data = s3[72];
            float v1768_data = ir7[6];
            ir7[6] = (v1768_data + (v1735_data * v1766_data));
            float v1771_data = s3[84];
            float v1773_data = ir7[7];
            ir7[7] = (v1773_data + (v1735_data * v1771_data));
          }
          if (v12_lead < 12) {
            float v1779_data = r6[1];
            float v1780_data = s3[1];
            float v1782_data = ir7[0];
            ir7[0] = (v1782_data + (v1779_data * v1780_data));
            float v1785_data = s3[13];
            float v1787_data = ir7[1];
            ir7[1] = (v1787_data + (v1779_data * v1785_data));
            float v1790_data = s3[25];
            float v1792_data = ir7[2];
            ir7[2] = (v1792_data + (v1779_data * v1790_data));
            float v1795_data = s3[37];
            float v1797_data = ir7[3];
            ir7[3] = (v1797_data + (v1779_data * v1795_data));
            float v1800_data = s3[49];
            float v1802_data = ir7[4];
            ir7[4] = (v1802_data + (v1779_data * v1800_data));
            float v1805_data = s3[61];
            float v1807_data = ir7[5];
            ir7[5] = (v1807_data + (v1779_data * v1805_data));
            float v1810_data = s3[73];
            float v1812_data = ir7[6];
            ir7[6] = (v1812_data + (v1779_data * v1810_data));
            float v1815_data = s3[85];
            float v1817_data = ir7[7];
            ir7[7] = (v1817_data + (v1779_data * v1815_data));
          }
          if (v12_lead < 12) {
            float v1823_data = r6[2];
            float v1824_data = s3[2];
            float v1826_data = ir7[0];
            ir7[0] = (v1826_data + (v1823_data * v1824_data));
            float v1829_data = s3[14];
            float v1831_data = ir7[1];
            ir7[1] = (v1831_data + (v1823_data * v1829_data));
            float v1834_data = s3[26];
            float v1836_data = ir7[2];
            ir7[2] = (v1836_data + (v1823_data * v1834_data));
            float v1839_data = s3[38];
            float v1841_data = ir7[3];
            ir7[3] = (v1841_data + (v1823_data * v1839_data));
            float v1844_data = s3[50];
            float v1846_data = ir7[4];
            ir7[4] = (v1846_data + (v1823_data * v1844_data));
            float v1849_data = s3[62];
            float v1851_data = ir7[5];
            ir7[5] = (v1851_data + (v1823_data * v1849_data));
            float v1854_data = s3[74];
            float v1856_data = ir7[6];
            ir7[6] = (v1856_data + (v1823_data * v1854_data));
            float v1859_data = s3[86];
            float v1861_data = ir7[7];
            ir7[7] = (v1861_data + (v1823_data * v1859_data));
          }
          if (v12_lead < 12) {
            float v1867_data = r6[3];
            float v1868_data = s3[3];
            float v1870_data = ir7[0];
            ir7[0] = (v1870_data + (v1867_data * v1868_data));
            float v1873_data = s3[15];
            float v1875_data = ir7[1];
            ir7[1] = (v1875_data + (v1867_data * v1873_data));
            float v1878_data = s3[27];
            float v1880_data = ir7[2];
            ir7[2] = (v1880_data + (v1867_data * v1878_data));
            float v1883_data = s3[39];
            float v1885_data = ir7[3];
            ir7[3] = (v1885_data + (v1867_data * v1883_data));
            float v1888_data = s3[51];
            float v1890_data = ir7[4];
            ir7[4] = (v1890_data + (v1867_data * v1888_data));
            float v1893_data = s3[63];
            float v1895_data = ir7[5];
            ir7[5] = (v1895_data + (v1867_data * v1893_data));
            float v1898_data = s3[75];
            float v1900_data = ir7[6];
            ir7[6] = (v1900_data + (v1867_data * v1898_data));
            float v1903_data = s3[87];
            float v1905_data = ir7[7];
            ir7[7] = (v1905_data + (v1867_data * v1903_data));
          }
          if (v12_lead < 12) {
            float v1911_data = r6[4];
            float v1912_data = s3[4];
            float v1914_data = ir7[0];
            ir7[0] = (v1914_data + (v1911_data * v1912_data));
            float v1917_data = s3[16];
            float v1919_data = ir7[1];
            ir7[1] = (v1919_data + (v1911_data * v1917_data));
            float v1922_data = s3[28];
            float v1924_data = ir7[2];
            ir7[2] = (v1924_data + (v1911_data * v1922_data));
            float v1927_data = s3[40];
            float v1929_data = ir7[3];
            ir7[3] = (v1929_data + (v1911_data * v1927_data));
            float v1932_data = s3[52];
            float v1934_data = ir7[4];
            ir7[4] = (v1934_data + (v1911_data * v1932_data));
            float v1937_data = s3[64];
            float v1939_data = ir7[5];
            ir7[5] = (v1939_data + (v1911_data * v1937_data));
            float v1942_data = s3[76];
            float v1944_data = ir7[6];
            ir7[6] = (v1944_data + (v1911_data * v1942_data));
            float v1947_data = s3[88];
            float v1949_data = ir7[7];
            ir7[7] = (v1949_data + (v1911_data * v1947_data));
          }
          if (v12_lead < 12) {
            float v1955_data = r6[5];
            float v1956_data = s3[5];
            float v1958_data = ir7[0];
            ir7[0] = (v1958_data + (v1955_data * v1956_data));
            float v1961_data = s3[17];
            float v1963_data = ir7[1];
            ir7[1] = (v1963_data + (v1955_data * v1961_data));
            float v1966_data = s3[29];
            float v1968_data = ir7[2];
            ir7[2] = (v1968_data + (v1955_data * v1966_data));
            float v1971_data = s3[41];
            float v1973_data = ir7[3];
            ir7[3] = (v1973_data + (v1955_data * v1971_data));
            float v1976_data = s3[53];
            float v1978_data = ir7[4];
            ir7[4] = (v1978_data + (v1955_data * v1976_data));
            float v1981_data = s3[65];
            float v1983_data = ir7[5];
            ir7[5] = (v1983_data + (v1955_data * v1981_data));
            float v1986_data = s3[77];
            float v1988_data = ir7[6];
            ir7[6] = (v1988_data + (v1955_data * v1986_data));
            float v1991_data = s3[89];
            float v1993_data = ir7[7];
            ir7[7] = (v1993_data + (v1955_data * v1991_data));
          }
          if (v12_lead < 12) {
            float v1999_data = r6[6];
            float v2000_data = s3[6];
            float v2002_data = ir7[0];
            ir7[0] = (v2002_data + (v1999_data * v2000_data));
            float v2005_data = s3[18];
            float v2007_data = ir7[1];
            ir7[1] = (v2007_data + (v1999_data * v2005_data));
            float v2010_data = s3[30];
            float v2012_data = ir7[2];
            ir7[2] = (v2012_data + (v1999_data * v2010_data));
            float v2015_data = s3[42];
            float v2017_data = ir7[3];
            ir7[3] = (v2017_data + (v1999_data * v2015_data));
            float v2020_data = s3[54];
            float v2022_data = ir7[4];
            ir7[4] = (v2022_data + (v1999_data * v2020_data));
            float v2025_data = s3[66];
            float v2027_data = ir7[5];
            ir7[5] = (v2027_data + (v1999_data * v2025_data));
            float v2030_data = s3[78];
            float v2032_data = ir7[6];
            ir7[6] = (v2032_data + (v1999_data * v2030_data));
            float v2035_data = s3[90];
            float v2037_data = ir7[7];
            ir7[7] = (v2037_data + (v1999_data * v2035_data));
          }
          if (v12_lead < 12) {
            float v2043_data = r6[7];
            float v2044_data = s3[7];
            float v2046_data = ir7[0];
            ir7[0] = (v2046_data + (v2043_data * v2044_data));
            float v2049_data = s3[19];
            float v2051_data = ir7[1];
            ir7[1] = (v2051_data + (v2043_data * v2049_data));
            float v2054_data = s3[31];
            float v2056_data = ir7[2];
            ir7[2] = (v2056_data + (v2043_data * v2054_data));
            float v2059_data = s3[43];
            float v2061_data = ir7[3];
            ir7[3] = (v2061_data + (v2043_data * v2059_data));
            float v2064_data = s3[55];
            float v2066_data = ir7[4];
            ir7[4] = (v2066_data + (v2043_data * v2064_data));
            float v2069_data = s3[67];
            float v2071_data = ir7[5];
            ir7[5] = (v2071_data + (v2043_data * v2069_data));
            float v2074_data = s3[79];
            float v2076_data = ir7[6];
            ir7[6] = (v2076_data + (v2043_data * v2074_data));
            float v2079_data = s3[91];
            float v2081_data = ir7[7];
            ir7[7] = (v2081_data + (v2043_data * v2079_data));
          }
          if (v12_lead < 12) {
            float v2087_data = r6[8];
            float v2088_data = s3[8];
            float v2090_data = ir7[0];
            ir7[0] = (v2090_data + (v2087_data * v2088_data));
            float v2093_data = s3[20];
            float v2095_data = ir7[1];
            ir7[1] = (v2095_data + (v2087_data * v2093_data));
            float v2098_data = s3[32];
            float v2100_data = ir7[2];
            ir7[2] = (v2100_data + (v2087_data * v2098_data));
            float v2103_data = s3[44];
            float v2105_data = ir7[3];
            ir7[3] = (v2105_data + (v2087_data * v2103_data));
            float v2108_data = s3[56];
            float v2110_data = ir7[4];
            ir7[4] = (v2110_data + (v2087_data * v2108_data));
            float v2113_data = s3[68];
            float v2115_data = ir7[5];
            ir7[5] = (v2115_data + (v2087_data * v2113_data));
            float v2118_data = s3[80];
            float v2120_data = ir7[6];
            ir7[6] = (v2120_data + (v2087_data * v2118_data));
            float v2123_data = s3[92];
            float v2125_data = ir7[7];
            ir7[7] = (v2125_data + (v2087_data * v2123_data));
          }
          if (v12_lead < 12) {
            float v2131_data = r6[9];
            float v2132_data = s3[9];
            float v2134_data = ir7[0];
            ir7[0] = (v2134_data + (v2131_data * v2132_data));
            float v2137_data = s3[21];
            float v2139_data = ir7[1];
            ir7[1] = (v2139_data + (v2131_data * v2137_data));
            float v2142_data = s3[33];
            float v2144_data = ir7[2];
            ir7[2] = (v2144_data + (v2131_data * v2142_data));
            float v2147_data = s3[45];
            float v2149_data = ir7[3];
            ir7[3] = (v2149_data + (v2131_data * v2147_data));
            float v2152_data = s3[57];
            float v2154_data = ir7[4];
            ir7[4] = (v2154_data + (v2131_data * v2152_data));
            float v2157_data = s3[69];
            float v2159_data = ir7[5];
            ir7[5] = (v2159_data + (v2131_data * v2157_data));
            float v2162_data = s3[81];
            float v2164_data = ir7[6];
            ir7[6] = (v2164_data + (v2131_data * v2162_data));
            float v2167_data = s3[93];
            float v2169_data = ir7[7];
            ir7[7] = (v2169_data + (v2131_data * v2167_data));
          }
          if (v12_lead < 12) {
            float v2175_data = r6[10];
            float v2176_data = s3[10];
            float v2178_data = ir7[0];
            ir7[0] = (v2178_data + (v2175_data * v2176_data));
            float v2181_data = s3[22];
            float v2183_data = ir7[1];
            ir7[1] = (v2183_data + (v2175_data * v2181_data));
            float v2186_data = s3[34];
            float v2188_data = ir7[2];
            ir7[2] = (v2188_data + (v2175_data * v2186_data));
            float v2191_data = s3[46];
            float v2193_data = ir7[3];
            ir7[3] = (v2193_data + (v2175_data * v2191_data));
            float v2196_data = s3[58];
            float v2198_data = ir7[4];
            ir7[4] = (v2198_data + (v2175_data * v2196_data));
            float v2201_data = s3[70];
            float v2203_data = ir7[5];
            ir7[5] = (v2203_data + (v2175_data * v2201_data));
            float v2206_data = s3[82];
            float v2208_data = ir7[6];
            ir7[6] = (v2208_data + (v2175_data * v2206_data));
            float v2211_data = s3[94];
            float v2213_data = ir7[7];
            ir7[7] = (v2213_data + (v2175_data * v2211_data));
          }
          if (v12_lead < 12) {
            float v2219_data = r6[11];
            float v2220_data = s3[11];
            float v2222_data = ir7[0];
            ir7[0] = (v2222_data + (v2219_data * v2220_data));
            float v2225_data = s3[23];
            float v2227_data = ir7[1];
            ir7[1] = (v2227_data + (v2219_data * v2225_data));
            float v2230_data = s3[35];
            float v2232_data = ir7[2];
            ir7[2] = (v2232_data + (v2219_data * v2230_data));
            float v2235_data = s3[47];
            float v2237_data = ir7[3];
            ir7[3] = (v2237_data + (v2219_data * v2235_data));
            float v2240_data = s3[59];
            float v2242_data = ir7[4];
            ir7[4] = (v2242_data + (v2219_data * v2240_data));
            float v2245_data = s3[71];
            float v2247_data = ir7[5];
            ir7[5] = (v2247_data + (v2219_data * v2245_data));
            float v2250_data = s3[83];
            float v2252_data = ir7[6];
            ir7[6] = (v2252_data + (v2219_data * v2250_data));
            float v2255_data = s3[95];
            float v2257_data = ir7[7];
            ir7[7] = (v2257_data + (v2219_data * v2255_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2263_n1 = 0; v2263_n1 < 8; ++v2263_n1) {
              int32_t v2264_a = 0 + v2263_n1;
              float v2266_data = ir7[v2263_n1];
              int32_t v2267_a = 0 + v2263_n1;
              float v2269_data = r5[v2263_n1];
              int32_t v2271_a = 0 + v2263_n1;
              r7[v2263_n1] = (v2269_data + v2266_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2277_i1 = 0; v2277_i1 < 8; ++v2277_i1) {
              int32_t v2278_a = 0 + v2277_i1;
              float v2280_data = r7[v2277_i1];
              int32_t v2287_a = v12_lead + (v2277_i1 * 12);
              glb_m0[v2287_a] = v2280_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

