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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __ldcg(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
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
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              int32_t v33_a = v27_i1 * 12;
              int32_t v34_a = v3_lead + v33_a;
              float v42_data = __ldcg(&glb_m3[(v3_lead + v33_a)]);
              int32_t v43_a = 0 + v27_i1;
              r2[v43_a] = v42_data;
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
          if (v3_lead < 12) {
            float v50_data = r0[0];
            float v51_data = s0[0];
            float v53_data = ir1[0];
            ir1[0] = (v53_data + (v50_data * v51_data));
            float v56_data = s0[12];
            float v58_data = ir1[1];
            ir1[1] = (v58_data + (v50_data * v56_data));
            float v61_data = s0[24];
            float v63_data = ir1[2];
            ir1[2] = (v63_data + (v50_data * v61_data));
            float v66_data = s0[36];
            float v68_data = ir1[3];
            ir1[3] = (v68_data + (v50_data * v66_data));
            float v71_data = s0[48];
            float v73_data = ir1[4];
            ir1[4] = (v73_data + (v50_data * v71_data));
            float v76_data = s0[60];
            float v78_data = ir1[5];
            ir1[5] = (v78_data + (v50_data * v76_data));
            float v81_data = s0[72];
            float v83_data = ir1[6];
            ir1[6] = (v83_data + (v50_data * v81_data));
            float v86_data = s0[84];
            float v88_data = ir1[7];
            ir1[7] = (v88_data + (v50_data * v86_data));
          }
          if (v3_lead < 12) {
            float v94_data = r0[1];
            float v95_data = s0[1];
            float v97_data = ir1[0];
            ir1[0] = (v97_data + (v94_data * v95_data));
            float v100_data = s0[13];
            float v102_data = ir1[1];
            ir1[1] = (v102_data + (v94_data * v100_data));
            float v105_data = s0[25];
            float v107_data = ir1[2];
            ir1[2] = (v107_data + (v94_data * v105_data));
            float v110_data = s0[37];
            float v112_data = ir1[3];
            ir1[3] = (v112_data + (v94_data * v110_data));
            float v115_data = s0[49];
            float v117_data = ir1[4];
            ir1[4] = (v117_data + (v94_data * v115_data));
            float v120_data = s0[61];
            float v122_data = ir1[5];
            ir1[5] = (v122_data + (v94_data * v120_data));
            float v125_data = s0[73];
            float v127_data = ir1[6];
            ir1[6] = (v127_data + (v94_data * v125_data));
            float v130_data = s0[85];
            float v132_data = ir1[7];
            ir1[7] = (v132_data + (v94_data * v130_data));
          }
          if (v3_lead < 12) {
            float v138_data = r0[2];
            float v139_data = s0[2];
            float v141_data = ir1[0];
            ir1[0] = (v141_data + (v138_data * v139_data));
            float v144_data = s0[14];
            float v146_data = ir1[1];
            ir1[1] = (v146_data + (v138_data * v144_data));
            float v149_data = s0[26];
            float v151_data = ir1[2];
            ir1[2] = (v151_data + (v138_data * v149_data));
            float v154_data = s0[38];
            float v156_data = ir1[3];
            ir1[3] = (v156_data + (v138_data * v154_data));
            float v159_data = s0[50];
            float v161_data = ir1[4];
            ir1[4] = (v161_data + (v138_data * v159_data));
            float v164_data = s0[62];
            float v166_data = ir1[5];
            ir1[5] = (v166_data + (v138_data * v164_data));
            float v169_data = s0[74];
            float v171_data = ir1[6];
            ir1[6] = (v171_data + (v138_data * v169_data));
            float v174_data = s0[86];
            float v176_data = ir1[7];
            ir1[7] = (v176_data + (v138_data * v174_data));
          }
          if (v3_lead < 12) {
            float v182_data = r0[3];
            float v183_data = s0[3];
            float v185_data = ir1[0];
            ir1[0] = (v185_data + (v182_data * v183_data));
            float v188_data = s0[15];
            float v190_data = ir1[1];
            ir1[1] = (v190_data + (v182_data * v188_data));
            float v193_data = s0[27];
            float v195_data = ir1[2];
            ir1[2] = (v195_data + (v182_data * v193_data));
            float v198_data = s0[39];
            float v200_data = ir1[3];
            ir1[3] = (v200_data + (v182_data * v198_data));
            float v203_data = s0[51];
            float v205_data = ir1[4];
            ir1[4] = (v205_data + (v182_data * v203_data));
            float v208_data = s0[63];
            float v210_data = ir1[5];
            ir1[5] = (v210_data + (v182_data * v208_data));
            float v213_data = s0[75];
            float v215_data = ir1[6];
            ir1[6] = (v215_data + (v182_data * v213_data));
            float v218_data = s0[87];
            float v220_data = ir1[7];
            ir1[7] = (v220_data + (v182_data * v218_data));
          }
          if (v3_lead < 12) {
            float v226_data = r0[4];
            float v227_data = s0[4];
            float v229_data = ir1[0];
            ir1[0] = (v229_data + (v226_data * v227_data));
            float v232_data = s0[16];
            float v234_data = ir1[1];
            ir1[1] = (v234_data + (v226_data * v232_data));
            float v237_data = s0[28];
            float v239_data = ir1[2];
            ir1[2] = (v239_data + (v226_data * v237_data));
            float v242_data = s0[40];
            float v244_data = ir1[3];
            ir1[3] = (v244_data + (v226_data * v242_data));
            float v247_data = s0[52];
            float v249_data = ir1[4];
            ir1[4] = (v249_data + (v226_data * v247_data));
            float v252_data = s0[64];
            float v254_data = ir1[5];
            ir1[5] = (v254_data + (v226_data * v252_data));
            float v257_data = s0[76];
            float v259_data = ir1[6];
            ir1[6] = (v259_data + (v226_data * v257_data));
            float v262_data = s0[88];
            float v264_data = ir1[7];
            ir1[7] = (v264_data + (v226_data * v262_data));
          }
          if (v3_lead < 12) {
            float v270_data = r0[5];
            float v271_data = s0[5];
            float v273_data = ir1[0];
            ir1[0] = (v273_data + (v270_data * v271_data));
            float v276_data = s0[17];
            float v278_data = ir1[1];
            ir1[1] = (v278_data + (v270_data * v276_data));
            float v281_data = s0[29];
            float v283_data = ir1[2];
            ir1[2] = (v283_data + (v270_data * v281_data));
            float v286_data = s0[41];
            float v288_data = ir1[3];
            ir1[3] = (v288_data + (v270_data * v286_data));
            float v291_data = s0[53];
            float v293_data = ir1[4];
            ir1[4] = (v293_data + (v270_data * v291_data));
            float v296_data = s0[65];
            float v298_data = ir1[5];
            ir1[5] = (v298_data + (v270_data * v296_data));
            float v301_data = s0[77];
            float v303_data = ir1[6];
            ir1[6] = (v303_data + (v270_data * v301_data));
            float v306_data = s0[89];
            float v308_data = ir1[7];
            ir1[7] = (v308_data + (v270_data * v306_data));
          }
          if (v3_lead < 12) {
            float v314_data = r0[6];
            float v315_data = s0[6];
            float v317_data = ir1[0];
            ir1[0] = (v317_data + (v314_data * v315_data));
            float v320_data = s0[18];
            float v322_data = ir1[1];
            ir1[1] = (v322_data + (v314_data * v320_data));
            float v325_data = s0[30];
            float v327_data = ir1[2];
            ir1[2] = (v327_data + (v314_data * v325_data));
            float v330_data = s0[42];
            float v332_data = ir1[3];
            ir1[3] = (v332_data + (v314_data * v330_data));
            float v335_data = s0[54];
            float v337_data = ir1[4];
            ir1[4] = (v337_data + (v314_data * v335_data));
            float v340_data = s0[66];
            float v342_data = ir1[5];
            ir1[5] = (v342_data + (v314_data * v340_data));
            float v345_data = s0[78];
            float v347_data = ir1[6];
            ir1[6] = (v347_data + (v314_data * v345_data));
            float v350_data = s0[90];
            float v352_data = ir1[7];
            ir1[7] = (v352_data + (v314_data * v350_data));
          }
          if (v3_lead < 12) {
            float v358_data = r0[7];
            float v359_data = s0[7];
            float v361_data = ir1[0];
            ir1[0] = (v361_data + (v358_data * v359_data));
            float v364_data = s0[19];
            float v366_data = ir1[1];
            ir1[1] = (v366_data + (v358_data * v364_data));
            float v369_data = s0[31];
            float v371_data = ir1[2];
            ir1[2] = (v371_data + (v358_data * v369_data));
            float v374_data = s0[43];
            float v376_data = ir1[3];
            ir1[3] = (v376_data + (v358_data * v374_data));
            float v379_data = s0[55];
            float v381_data = ir1[4];
            ir1[4] = (v381_data + (v358_data * v379_data));
            float v384_data = s0[67];
            float v386_data = ir1[5];
            ir1[5] = (v386_data + (v358_data * v384_data));
            float v389_data = s0[79];
            float v391_data = ir1[6];
            ir1[6] = (v391_data + (v358_data * v389_data));
            float v394_data = s0[91];
            float v396_data = ir1[7];
            ir1[7] = (v396_data + (v358_data * v394_data));
          }
          if (v3_lead < 12) {
            float v402_data = r0[8];
            float v403_data = s0[8];
            float v405_data = ir1[0];
            ir1[0] = (v405_data + (v402_data * v403_data));
            float v408_data = s0[20];
            float v410_data = ir1[1];
            ir1[1] = (v410_data + (v402_data * v408_data));
            float v413_data = s0[32];
            float v415_data = ir1[2];
            ir1[2] = (v415_data + (v402_data * v413_data));
            float v418_data = s0[44];
            float v420_data = ir1[3];
            ir1[3] = (v420_data + (v402_data * v418_data));
            float v423_data = s0[56];
            float v425_data = ir1[4];
            ir1[4] = (v425_data + (v402_data * v423_data));
            float v428_data = s0[68];
            float v430_data = ir1[5];
            ir1[5] = (v430_data + (v402_data * v428_data));
            float v433_data = s0[80];
            float v435_data = ir1[6];
            ir1[6] = (v435_data + (v402_data * v433_data));
            float v438_data = s0[92];
            float v440_data = ir1[7];
            ir1[7] = (v440_data + (v402_data * v438_data));
          }
          if (v3_lead < 12) {
            float v446_data = r0[9];
            float v447_data = s0[9];
            float v449_data = ir1[0];
            ir1[0] = (v449_data + (v446_data * v447_data));
            float v452_data = s0[21];
            float v454_data = ir1[1];
            ir1[1] = (v454_data + (v446_data * v452_data));
            float v457_data = s0[33];
            float v459_data = ir1[2];
            ir1[2] = (v459_data + (v446_data * v457_data));
            float v462_data = s0[45];
            float v464_data = ir1[3];
            ir1[3] = (v464_data + (v446_data * v462_data));
            float v467_data = s0[57];
            float v469_data = ir1[4];
            ir1[4] = (v469_data + (v446_data * v467_data));
            float v472_data = s0[69];
            float v474_data = ir1[5];
            ir1[5] = (v474_data + (v446_data * v472_data));
            float v477_data = s0[81];
            float v479_data = ir1[6];
            ir1[6] = (v479_data + (v446_data * v477_data));
            float v482_data = s0[93];
            float v484_data = ir1[7];
            ir1[7] = (v484_data + (v446_data * v482_data));
          }
          if (v3_lead < 12) {
            float v490_data = r0[10];
            float v491_data = s0[10];
            float v493_data = ir1[0];
            ir1[0] = (v493_data + (v490_data * v491_data));
            float v496_data = s0[22];
            float v498_data = ir1[1];
            ir1[1] = (v498_data + (v490_data * v496_data));
            float v501_data = s0[34];
            float v503_data = ir1[2];
            ir1[2] = (v503_data + (v490_data * v501_data));
            float v506_data = s0[46];
            float v508_data = ir1[3];
            ir1[3] = (v508_data + (v490_data * v506_data));
            float v511_data = s0[58];
            float v513_data = ir1[4];
            ir1[4] = (v513_data + (v490_data * v511_data));
            float v516_data = s0[70];
            float v518_data = ir1[5];
            ir1[5] = (v518_data + (v490_data * v516_data));
            float v521_data = s0[82];
            float v523_data = ir1[6];
            ir1[6] = (v523_data + (v490_data * v521_data));
            float v526_data = s0[94];
            float v528_data = ir1[7];
            ir1[7] = (v528_data + (v490_data * v526_data));
          }
          if (v3_lead < 12) {
            float v534_data = r0[11];
            float v535_data = s0[11];
            float v537_data = ir1[0];
            ir1[0] = (v537_data + (v534_data * v535_data));
            float v540_data = s0[23];
            float v542_data = ir1[1];
            ir1[1] = (v542_data + (v534_data * v540_data));
            float v545_data = s0[35];
            float v547_data = ir1[2];
            ir1[2] = (v547_data + (v534_data * v545_data));
            float v550_data = s0[47];
            float v552_data = ir1[3];
            ir1[3] = (v552_data + (v534_data * v550_data));
            float v555_data = s0[59];
            float v557_data = ir1[4];
            ir1[4] = (v557_data + (v534_data * v555_data));
            float v560_data = s0[71];
            float v562_data = ir1[5];
            ir1[5] = (v562_data + (v534_data * v560_data));
            float v565_data = s0[83];
            float v567_data = ir1[6];
            ir1[6] = (v567_data + (v534_data * v565_data));
            float v570_data = s0[95];
            float v572_data = ir1[7];
            ir1[7] = (v572_data + (v534_data * v570_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v578_n1 = 0; v578_n1 < 8; ++v578_n1) {
              int32_t v579_a = 0 + v578_n1;
              float v581_data = ir1[v578_n1];
              int32_t v582_a = 0 + v578_n1;
              r1[v578_n1] = v581_data;
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
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v589_i1 = 0; v589_i1 < 12; ++v589_i1) {
              int32_t v595_a = v589_i1 * 12;
              int32_t v596_a = v3_lead + v595_a;
              float v604_data = __ldcg(&glb_m5[(v3_lead + v595_a)]);
              int32_t v605_a = 0 + v589_i1;
              r4[v605_a] = v604_data;
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
          if (v3_lead < 12) {
            float v612_data = r2[0];
            float v613_data = s1[0];
            float v615_data = ir3[0];
            ir3[0] = (v615_data + (v612_data * v613_data));
            float v618_data = s1[12];
            float v620_data = ir3[1];
            ir3[1] = (v620_data + (v612_data * v618_data));
            float v623_data = s1[24];
            float v625_data = ir3[2];
            ir3[2] = (v625_data + (v612_data * v623_data));
            float v628_data = s1[36];
            float v630_data = ir3[3];
            ir3[3] = (v630_data + (v612_data * v628_data));
            float v633_data = s1[48];
            float v635_data = ir3[4];
            ir3[4] = (v635_data + (v612_data * v633_data));
            float v638_data = s1[60];
            float v640_data = ir3[5];
            ir3[5] = (v640_data + (v612_data * v638_data));
            float v643_data = s1[72];
            float v645_data = ir3[6];
            ir3[6] = (v645_data + (v612_data * v643_data));
            float v648_data = s1[84];
            float v650_data = ir3[7];
            ir3[7] = (v650_data + (v612_data * v648_data));
          }
          if (v3_lead < 12) {
            float v656_data = r2[1];
            float v657_data = s1[1];
            float v659_data = ir3[0];
            ir3[0] = (v659_data + (v656_data * v657_data));
            float v662_data = s1[13];
            float v664_data = ir3[1];
            ir3[1] = (v664_data + (v656_data * v662_data));
            float v667_data = s1[25];
            float v669_data = ir3[2];
            ir3[2] = (v669_data + (v656_data * v667_data));
            float v672_data = s1[37];
            float v674_data = ir3[3];
            ir3[3] = (v674_data + (v656_data * v672_data));
            float v677_data = s1[49];
            float v679_data = ir3[4];
            ir3[4] = (v679_data + (v656_data * v677_data));
            float v682_data = s1[61];
            float v684_data = ir3[5];
            ir3[5] = (v684_data + (v656_data * v682_data));
            float v687_data = s1[73];
            float v689_data = ir3[6];
            ir3[6] = (v689_data + (v656_data * v687_data));
            float v692_data = s1[85];
            float v694_data = ir3[7];
            ir3[7] = (v694_data + (v656_data * v692_data));
          }
          if (v3_lead < 12) {
            float v700_data = r2[2];
            float v701_data = s1[2];
            float v703_data = ir3[0];
            ir3[0] = (v703_data + (v700_data * v701_data));
            float v706_data = s1[14];
            float v708_data = ir3[1];
            ir3[1] = (v708_data + (v700_data * v706_data));
            float v711_data = s1[26];
            float v713_data = ir3[2];
            ir3[2] = (v713_data + (v700_data * v711_data));
            float v716_data = s1[38];
            float v718_data = ir3[3];
            ir3[3] = (v718_data + (v700_data * v716_data));
            float v721_data = s1[50];
            float v723_data = ir3[4];
            ir3[4] = (v723_data + (v700_data * v721_data));
            float v726_data = s1[62];
            float v728_data = ir3[5];
            ir3[5] = (v728_data + (v700_data * v726_data));
            float v731_data = s1[74];
            float v733_data = ir3[6];
            ir3[6] = (v733_data + (v700_data * v731_data));
            float v736_data = s1[86];
            float v738_data = ir3[7];
            ir3[7] = (v738_data + (v700_data * v736_data));
          }
          if (v3_lead < 12) {
            float v744_data = r2[3];
            float v745_data = s1[3];
            float v747_data = ir3[0];
            ir3[0] = (v747_data + (v744_data * v745_data));
            float v750_data = s1[15];
            float v752_data = ir3[1];
            ir3[1] = (v752_data + (v744_data * v750_data));
            float v755_data = s1[27];
            float v757_data = ir3[2];
            ir3[2] = (v757_data + (v744_data * v755_data));
            float v760_data = s1[39];
            float v762_data = ir3[3];
            ir3[3] = (v762_data + (v744_data * v760_data));
            float v765_data = s1[51];
            float v767_data = ir3[4];
            ir3[4] = (v767_data + (v744_data * v765_data));
            float v770_data = s1[63];
            float v772_data = ir3[5];
            ir3[5] = (v772_data + (v744_data * v770_data));
            float v775_data = s1[75];
            float v777_data = ir3[6];
            ir3[6] = (v777_data + (v744_data * v775_data));
            float v780_data = s1[87];
            float v782_data = ir3[7];
            ir3[7] = (v782_data + (v744_data * v780_data));
          }
          if (v3_lead < 12) {
            float v788_data = r2[4];
            float v789_data = s1[4];
            float v791_data = ir3[0];
            ir3[0] = (v791_data + (v788_data * v789_data));
            float v794_data = s1[16];
            float v796_data = ir3[1];
            ir3[1] = (v796_data + (v788_data * v794_data));
            float v799_data = s1[28];
            float v801_data = ir3[2];
            ir3[2] = (v801_data + (v788_data * v799_data));
            float v804_data = s1[40];
            float v806_data = ir3[3];
            ir3[3] = (v806_data + (v788_data * v804_data));
            float v809_data = s1[52];
            float v811_data = ir3[4];
            ir3[4] = (v811_data + (v788_data * v809_data));
            float v814_data = s1[64];
            float v816_data = ir3[5];
            ir3[5] = (v816_data + (v788_data * v814_data));
            float v819_data = s1[76];
            float v821_data = ir3[6];
            ir3[6] = (v821_data + (v788_data * v819_data));
            float v824_data = s1[88];
            float v826_data = ir3[7];
            ir3[7] = (v826_data + (v788_data * v824_data));
          }
          if (v3_lead < 12) {
            float v832_data = r2[5];
            float v833_data = s1[5];
            float v835_data = ir3[0];
            ir3[0] = (v835_data + (v832_data * v833_data));
            float v838_data = s1[17];
            float v840_data = ir3[1];
            ir3[1] = (v840_data + (v832_data * v838_data));
            float v843_data = s1[29];
            float v845_data = ir3[2];
            ir3[2] = (v845_data + (v832_data * v843_data));
            float v848_data = s1[41];
            float v850_data = ir3[3];
            ir3[3] = (v850_data + (v832_data * v848_data));
            float v853_data = s1[53];
            float v855_data = ir3[4];
            ir3[4] = (v855_data + (v832_data * v853_data));
            float v858_data = s1[65];
            float v860_data = ir3[5];
            ir3[5] = (v860_data + (v832_data * v858_data));
            float v863_data = s1[77];
            float v865_data = ir3[6];
            ir3[6] = (v865_data + (v832_data * v863_data));
            float v868_data = s1[89];
            float v870_data = ir3[7];
            ir3[7] = (v870_data + (v832_data * v868_data));
          }
          if (v3_lead < 12) {
            float v876_data = r2[6];
            float v877_data = s1[6];
            float v879_data = ir3[0];
            ir3[0] = (v879_data + (v876_data * v877_data));
            float v882_data = s1[18];
            float v884_data = ir3[1];
            ir3[1] = (v884_data + (v876_data * v882_data));
            float v887_data = s1[30];
            float v889_data = ir3[2];
            ir3[2] = (v889_data + (v876_data * v887_data));
            float v892_data = s1[42];
            float v894_data = ir3[3];
            ir3[3] = (v894_data + (v876_data * v892_data));
            float v897_data = s1[54];
            float v899_data = ir3[4];
            ir3[4] = (v899_data + (v876_data * v897_data));
            float v902_data = s1[66];
            float v904_data = ir3[5];
            ir3[5] = (v904_data + (v876_data * v902_data));
            float v907_data = s1[78];
            float v909_data = ir3[6];
            ir3[6] = (v909_data + (v876_data * v907_data));
            float v912_data = s1[90];
            float v914_data = ir3[7];
            ir3[7] = (v914_data + (v876_data * v912_data));
          }
          if (v3_lead < 12) {
            float v920_data = r2[7];
            float v921_data = s1[7];
            float v923_data = ir3[0];
            ir3[0] = (v923_data + (v920_data * v921_data));
            float v926_data = s1[19];
            float v928_data = ir3[1];
            ir3[1] = (v928_data + (v920_data * v926_data));
            float v931_data = s1[31];
            float v933_data = ir3[2];
            ir3[2] = (v933_data + (v920_data * v931_data));
            float v936_data = s1[43];
            float v938_data = ir3[3];
            ir3[3] = (v938_data + (v920_data * v936_data));
            float v941_data = s1[55];
            float v943_data = ir3[4];
            ir3[4] = (v943_data + (v920_data * v941_data));
            float v946_data = s1[67];
            float v948_data = ir3[5];
            ir3[5] = (v948_data + (v920_data * v946_data));
            float v951_data = s1[79];
            float v953_data = ir3[6];
            ir3[6] = (v953_data + (v920_data * v951_data));
            float v956_data = s1[91];
            float v958_data = ir3[7];
            ir3[7] = (v958_data + (v920_data * v956_data));
          }
          if (v3_lead < 12) {
            float v964_data = r2[8];
            float v965_data = s1[8];
            float v967_data = ir3[0];
            ir3[0] = (v967_data + (v964_data * v965_data));
            float v970_data = s1[20];
            float v972_data = ir3[1];
            ir3[1] = (v972_data + (v964_data * v970_data));
            float v975_data = s1[32];
            float v977_data = ir3[2];
            ir3[2] = (v977_data + (v964_data * v975_data));
            float v980_data = s1[44];
            float v982_data = ir3[3];
            ir3[3] = (v982_data + (v964_data * v980_data));
            float v985_data = s1[56];
            float v987_data = ir3[4];
            ir3[4] = (v987_data + (v964_data * v985_data));
            float v990_data = s1[68];
            float v992_data = ir3[5];
            ir3[5] = (v992_data + (v964_data * v990_data));
            float v995_data = s1[80];
            float v997_data = ir3[6];
            ir3[6] = (v997_data + (v964_data * v995_data));
            float v1000_data = s1[92];
            float v1002_data = ir3[7];
            ir3[7] = (v1002_data + (v964_data * v1000_data));
          }
          if (v3_lead < 12) {
            float v1008_data = r2[9];
            float v1009_data = s1[9];
            float v1011_data = ir3[0];
            ir3[0] = (v1011_data + (v1008_data * v1009_data));
            float v1014_data = s1[21];
            float v1016_data = ir3[1];
            ir3[1] = (v1016_data + (v1008_data * v1014_data));
            float v1019_data = s1[33];
            float v1021_data = ir3[2];
            ir3[2] = (v1021_data + (v1008_data * v1019_data));
            float v1024_data = s1[45];
            float v1026_data = ir3[3];
            ir3[3] = (v1026_data + (v1008_data * v1024_data));
            float v1029_data = s1[57];
            float v1031_data = ir3[4];
            ir3[4] = (v1031_data + (v1008_data * v1029_data));
            float v1034_data = s1[69];
            float v1036_data = ir3[5];
            ir3[5] = (v1036_data + (v1008_data * v1034_data));
            float v1039_data = s1[81];
            float v1041_data = ir3[6];
            ir3[6] = (v1041_data + (v1008_data * v1039_data));
            float v1044_data = s1[93];
            float v1046_data = ir3[7];
            ir3[7] = (v1046_data + (v1008_data * v1044_data));
          }
          if (v3_lead < 12) {
            float v1052_data = r2[10];
            float v1053_data = s1[10];
            float v1055_data = ir3[0];
            ir3[0] = (v1055_data + (v1052_data * v1053_data));
            float v1058_data = s1[22];
            float v1060_data = ir3[1];
            ir3[1] = (v1060_data + (v1052_data * v1058_data));
            float v1063_data = s1[34];
            float v1065_data = ir3[2];
            ir3[2] = (v1065_data + (v1052_data * v1063_data));
            float v1068_data = s1[46];
            float v1070_data = ir3[3];
            ir3[3] = (v1070_data + (v1052_data * v1068_data));
            float v1073_data = s1[58];
            float v1075_data = ir3[4];
            ir3[4] = (v1075_data + (v1052_data * v1073_data));
            float v1078_data = s1[70];
            float v1080_data = ir3[5];
            ir3[5] = (v1080_data + (v1052_data * v1078_data));
            float v1083_data = s1[82];
            float v1085_data = ir3[6];
            ir3[6] = (v1085_data + (v1052_data * v1083_data));
            float v1088_data = s1[94];
            float v1090_data = ir3[7];
            ir3[7] = (v1090_data + (v1052_data * v1088_data));
          }
          if (v3_lead < 12) {
            float v1096_data = r2[11];
            float v1097_data = s1[11];
            float v1099_data = ir3[0];
            ir3[0] = (v1099_data + (v1096_data * v1097_data));
            float v1102_data = s1[23];
            float v1104_data = ir3[1];
            ir3[1] = (v1104_data + (v1096_data * v1102_data));
            float v1107_data = s1[35];
            float v1109_data = ir3[2];
            ir3[2] = (v1109_data + (v1096_data * v1107_data));
            float v1112_data = s1[47];
            float v1114_data = ir3[3];
            ir3[3] = (v1114_data + (v1096_data * v1112_data));
            float v1117_data = s1[59];
            float v1119_data = ir3[4];
            ir3[4] = (v1119_data + (v1096_data * v1117_data));
            float v1122_data = s1[71];
            float v1124_data = ir3[5];
            ir3[5] = (v1124_data + (v1096_data * v1122_data));
            float v1127_data = s1[83];
            float v1129_data = ir3[6];
            ir3[6] = (v1129_data + (v1096_data * v1127_data));
            float v1132_data = s1[95];
            float v1134_data = ir3[7];
            ir3[7] = (v1134_data + (v1096_data * v1132_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v1140_n1 = 0; v1140_n1 < 8; ++v1140_n1) {
              int32_t v1141_a = 0 + v1140_n1;
              float v1143_data = ir3[v1140_n1];
              int32_t v1144_a = 0 + v1140_n1;
              float v1146_data = r1[v1140_n1];
              int32_t v1148_a = 0 + v1140_n1;
              r3[v1140_n1] = (v1146_data + v1143_data);
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
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v1155_i1 = 0; v1155_i1 < 12; ++v1155_i1) {
              int32_t v1161_a = v1155_i1 * 12;
              int32_t v1162_a = v3_lead + v1161_a;
              float v1170_data = __ldcg(&glb_m7[(v3_lead + v1161_a)]);
              int32_t v1171_a = 0 + v1155_i1;
              r6[v1171_a] = v1170_data;
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
          if (v3_lead < 12) {
            float v1178_data = r4[0];
            float v1179_data = s2[0];
            float v1181_data = ir5[0];
            ir5[0] = (v1181_data + (v1178_data * v1179_data));
            float v1184_data = s2[12];
            float v1186_data = ir5[1];
            ir5[1] = (v1186_data + (v1178_data * v1184_data));
            float v1189_data = s2[24];
            float v1191_data = ir5[2];
            ir5[2] = (v1191_data + (v1178_data * v1189_data));
            float v1194_data = s2[36];
            float v1196_data = ir5[3];
            ir5[3] = (v1196_data + (v1178_data * v1194_data));
            float v1199_data = s2[48];
            float v1201_data = ir5[4];
            ir5[4] = (v1201_data + (v1178_data * v1199_data));
            float v1204_data = s2[60];
            float v1206_data = ir5[5];
            ir5[5] = (v1206_data + (v1178_data * v1204_data));
            float v1209_data = s2[72];
            float v1211_data = ir5[6];
            ir5[6] = (v1211_data + (v1178_data * v1209_data));
            float v1214_data = s2[84];
            float v1216_data = ir5[7];
            ir5[7] = (v1216_data + (v1178_data * v1214_data));
          }
          if (v3_lead < 12) {
            float v1222_data = r4[1];
            float v1223_data = s2[1];
            float v1225_data = ir5[0];
            ir5[0] = (v1225_data + (v1222_data * v1223_data));
            float v1228_data = s2[13];
            float v1230_data = ir5[1];
            ir5[1] = (v1230_data + (v1222_data * v1228_data));
            float v1233_data = s2[25];
            float v1235_data = ir5[2];
            ir5[2] = (v1235_data + (v1222_data * v1233_data));
            float v1238_data = s2[37];
            float v1240_data = ir5[3];
            ir5[3] = (v1240_data + (v1222_data * v1238_data));
            float v1243_data = s2[49];
            float v1245_data = ir5[4];
            ir5[4] = (v1245_data + (v1222_data * v1243_data));
            float v1248_data = s2[61];
            float v1250_data = ir5[5];
            ir5[5] = (v1250_data + (v1222_data * v1248_data));
            float v1253_data = s2[73];
            float v1255_data = ir5[6];
            ir5[6] = (v1255_data + (v1222_data * v1253_data));
            float v1258_data = s2[85];
            float v1260_data = ir5[7];
            ir5[7] = (v1260_data + (v1222_data * v1258_data));
          }
          if (v3_lead < 12) {
            float v1266_data = r4[2];
            float v1267_data = s2[2];
            float v1269_data = ir5[0];
            ir5[0] = (v1269_data + (v1266_data * v1267_data));
            float v1272_data = s2[14];
            float v1274_data = ir5[1];
            ir5[1] = (v1274_data + (v1266_data * v1272_data));
            float v1277_data = s2[26];
            float v1279_data = ir5[2];
            ir5[2] = (v1279_data + (v1266_data * v1277_data));
            float v1282_data = s2[38];
            float v1284_data = ir5[3];
            ir5[3] = (v1284_data + (v1266_data * v1282_data));
            float v1287_data = s2[50];
            float v1289_data = ir5[4];
            ir5[4] = (v1289_data + (v1266_data * v1287_data));
            float v1292_data = s2[62];
            float v1294_data = ir5[5];
            ir5[5] = (v1294_data + (v1266_data * v1292_data));
            float v1297_data = s2[74];
            float v1299_data = ir5[6];
            ir5[6] = (v1299_data + (v1266_data * v1297_data));
            float v1302_data = s2[86];
            float v1304_data = ir5[7];
            ir5[7] = (v1304_data + (v1266_data * v1302_data));
          }
          if (v3_lead < 12) {
            float v1310_data = r4[3];
            float v1311_data = s2[3];
            float v1313_data = ir5[0];
            ir5[0] = (v1313_data + (v1310_data * v1311_data));
            float v1316_data = s2[15];
            float v1318_data = ir5[1];
            ir5[1] = (v1318_data + (v1310_data * v1316_data));
            float v1321_data = s2[27];
            float v1323_data = ir5[2];
            ir5[2] = (v1323_data + (v1310_data * v1321_data));
            float v1326_data = s2[39];
            float v1328_data = ir5[3];
            ir5[3] = (v1328_data + (v1310_data * v1326_data));
            float v1331_data = s2[51];
            float v1333_data = ir5[4];
            ir5[4] = (v1333_data + (v1310_data * v1331_data));
            float v1336_data = s2[63];
            float v1338_data = ir5[5];
            ir5[5] = (v1338_data + (v1310_data * v1336_data));
            float v1341_data = s2[75];
            float v1343_data = ir5[6];
            ir5[6] = (v1343_data + (v1310_data * v1341_data));
            float v1346_data = s2[87];
            float v1348_data = ir5[7];
            ir5[7] = (v1348_data + (v1310_data * v1346_data));
          }
          if (v3_lead < 12) {
            float v1354_data = r4[4];
            float v1355_data = s2[4];
            float v1357_data = ir5[0];
            ir5[0] = (v1357_data + (v1354_data * v1355_data));
            float v1360_data = s2[16];
            float v1362_data = ir5[1];
            ir5[1] = (v1362_data + (v1354_data * v1360_data));
            float v1365_data = s2[28];
            float v1367_data = ir5[2];
            ir5[2] = (v1367_data + (v1354_data * v1365_data));
            float v1370_data = s2[40];
            float v1372_data = ir5[3];
            ir5[3] = (v1372_data + (v1354_data * v1370_data));
            float v1375_data = s2[52];
            float v1377_data = ir5[4];
            ir5[4] = (v1377_data + (v1354_data * v1375_data));
            float v1380_data = s2[64];
            float v1382_data = ir5[5];
            ir5[5] = (v1382_data + (v1354_data * v1380_data));
            float v1385_data = s2[76];
            float v1387_data = ir5[6];
            ir5[6] = (v1387_data + (v1354_data * v1385_data));
            float v1390_data = s2[88];
            float v1392_data = ir5[7];
            ir5[7] = (v1392_data + (v1354_data * v1390_data));
          }
          if (v3_lead < 12) {
            float v1398_data = r4[5];
            float v1399_data = s2[5];
            float v1401_data = ir5[0];
            ir5[0] = (v1401_data + (v1398_data * v1399_data));
            float v1404_data = s2[17];
            float v1406_data = ir5[1];
            ir5[1] = (v1406_data + (v1398_data * v1404_data));
            float v1409_data = s2[29];
            float v1411_data = ir5[2];
            ir5[2] = (v1411_data + (v1398_data * v1409_data));
            float v1414_data = s2[41];
            float v1416_data = ir5[3];
            ir5[3] = (v1416_data + (v1398_data * v1414_data));
            float v1419_data = s2[53];
            float v1421_data = ir5[4];
            ir5[4] = (v1421_data + (v1398_data * v1419_data));
            float v1424_data = s2[65];
            float v1426_data = ir5[5];
            ir5[5] = (v1426_data + (v1398_data * v1424_data));
            float v1429_data = s2[77];
            float v1431_data = ir5[6];
            ir5[6] = (v1431_data + (v1398_data * v1429_data));
            float v1434_data = s2[89];
            float v1436_data = ir5[7];
            ir5[7] = (v1436_data + (v1398_data * v1434_data));
          }
          if (v3_lead < 12) {
            float v1442_data = r4[6];
            float v1443_data = s2[6];
            float v1445_data = ir5[0];
            ir5[0] = (v1445_data + (v1442_data * v1443_data));
            float v1448_data = s2[18];
            float v1450_data = ir5[1];
            ir5[1] = (v1450_data + (v1442_data * v1448_data));
            float v1453_data = s2[30];
            float v1455_data = ir5[2];
            ir5[2] = (v1455_data + (v1442_data * v1453_data));
            float v1458_data = s2[42];
            float v1460_data = ir5[3];
            ir5[3] = (v1460_data + (v1442_data * v1458_data));
            float v1463_data = s2[54];
            float v1465_data = ir5[4];
            ir5[4] = (v1465_data + (v1442_data * v1463_data));
            float v1468_data = s2[66];
            float v1470_data = ir5[5];
            ir5[5] = (v1470_data + (v1442_data * v1468_data));
            float v1473_data = s2[78];
            float v1475_data = ir5[6];
            ir5[6] = (v1475_data + (v1442_data * v1473_data));
            float v1478_data = s2[90];
            float v1480_data = ir5[7];
            ir5[7] = (v1480_data + (v1442_data * v1478_data));
          }
          if (v3_lead < 12) {
            float v1486_data = r4[7];
            float v1487_data = s2[7];
            float v1489_data = ir5[0];
            ir5[0] = (v1489_data + (v1486_data * v1487_data));
            float v1492_data = s2[19];
            float v1494_data = ir5[1];
            ir5[1] = (v1494_data + (v1486_data * v1492_data));
            float v1497_data = s2[31];
            float v1499_data = ir5[2];
            ir5[2] = (v1499_data + (v1486_data * v1497_data));
            float v1502_data = s2[43];
            float v1504_data = ir5[3];
            ir5[3] = (v1504_data + (v1486_data * v1502_data));
            float v1507_data = s2[55];
            float v1509_data = ir5[4];
            ir5[4] = (v1509_data + (v1486_data * v1507_data));
            float v1512_data = s2[67];
            float v1514_data = ir5[5];
            ir5[5] = (v1514_data + (v1486_data * v1512_data));
            float v1517_data = s2[79];
            float v1519_data = ir5[6];
            ir5[6] = (v1519_data + (v1486_data * v1517_data));
            float v1522_data = s2[91];
            float v1524_data = ir5[7];
            ir5[7] = (v1524_data + (v1486_data * v1522_data));
          }
          if (v3_lead < 12) {
            float v1530_data = r4[8];
            float v1531_data = s2[8];
            float v1533_data = ir5[0];
            ir5[0] = (v1533_data + (v1530_data * v1531_data));
            float v1536_data = s2[20];
            float v1538_data = ir5[1];
            ir5[1] = (v1538_data + (v1530_data * v1536_data));
            float v1541_data = s2[32];
            float v1543_data = ir5[2];
            ir5[2] = (v1543_data + (v1530_data * v1541_data));
            float v1546_data = s2[44];
            float v1548_data = ir5[3];
            ir5[3] = (v1548_data + (v1530_data * v1546_data));
            float v1551_data = s2[56];
            float v1553_data = ir5[4];
            ir5[4] = (v1553_data + (v1530_data * v1551_data));
            float v1556_data = s2[68];
            float v1558_data = ir5[5];
            ir5[5] = (v1558_data + (v1530_data * v1556_data));
            float v1561_data = s2[80];
            float v1563_data = ir5[6];
            ir5[6] = (v1563_data + (v1530_data * v1561_data));
            float v1566_data = s2[92];
            float v1568_data = ir5[7];
            ir5[7] = (v1568_data + (v1530_data * v1566_data));
          }
          if (v3_lead < 12) {
            float v1574_data = r4[9];
            float v1575_data = s2[9];
            float v1577_data = ir5[0];
            ir5[0] = (v1577_data + (v1574_data * v1575_data));
            float v1580_data = s2[21];
            float v1582_data = ir5[1];
            ir5[1] = (v1582_data + (v1574_data * v1580_data));
            float v1585_data = s2[33];
            float v1587_data = ir5[2];
            ir5[2] = (v1587_data + (v1574_data * v1585_data));
            float v1590_data = s2[45];
            float v1592_data = ir5[3];
            ir5[3] = (v1592_data + (v1574_data * v1590_data));
            float v1595_data = s2[57];
            float v1597_data = ir5[4];
            ir5[4] = (v1597_data + (v1574_data * v1595_data));
            float v1600_data = s2[69];
            float v1602_data = ir5[5];
            ir5[5] = (v1602_data + (v1574_data * v1600_data));
            float v1605_data = s2[81];
            float v1607_data = ir5[6];
            ir5[6] = (v1607_data + (v1574_data * v1605_data));
            float v1610_data = s2[93];
            float v1612_data = ir5[7];
            ir5[7] = (v1612_data + (v1574_data * v1610_data));
          }
          if (v3_lead < 12) {
            float v1618_data = r4[10];
            float v1619_data = s2[10];
            float v1621_data = ir5[0];
            ir5[0] = (v1621_data + (v1618_data * v1619_data));
            float v1624_data = s2[22];
            float v1626_data = ir5[1];
            ir5[1] = (v1626_data + (v1618_data * v1624_data));
            float v1629_data = s2[34];
            float v1631_data = ir5[2];
            ir5[2] = (v1631_data + (v1618_data * v1629_data));
            float v1634_data = s2[46];
            float v1636_data = ir5[3];
            ir5[3] = (v1636_data + (v1618_data * v1634_data));
            float v1639_data = s2[58];
            float v1641_data = ir5[4];
            ir5[4] = (v1641_data + (v1618_data * v1639_data));
            float v1644_data = s2[70];
            float v1646_data = ir5[5];
            ir5[5] = (v1646_data + (v1618_data * v1644_data));
            float v1649_data = s2[82];
            float v1651_data = ir5[6];
            ir5[6] = (v1651_data + (v1618_data * v1649_data));
            float v1654_data = s2[94];
            float v1656_data = ir5[7];
            ir5[7] = (v1656_data + (v1618_data * v1654_data));
          }
          if (v3_lead < 12) {
            float v1662_data = r4[11];
            float v1663_data = s2[11];
            float v1665_data = ir5[0];
            ir5[0] = (v1665_data + (v1662_data * v1663_data));
            float v1668_data = s2[23];
            float v1670_data = ir5[1];
            ir5[1] = (v1670_data + (v1662_data * v1668_data));
            float v1673_data = s2[35];
            float v1675_data = ir5[2];
            ir5[2] = (v1675_data + (v1662_data * v1673_data));
            float v1678_data = s2[47];
            float v1680_data = ir5[3];
            ir5[3] = (v1680_data + (v1662_data * v1678_data));
            float v1683_data = s2[59];
            float v1685_data = ir5[4];
            ir5[4] = (v1685_data + (v1662_data * v1683_data));
            float v1688_data = s2[71];
            float v1690_data = ir5[5];
            ir5[5] = (v1690_data + (v1662_data * v1688_data));
            float v1693_data = s2[83];
            float v1695_data = ir5[6];
            ir5[6] = (v1695_data + (v1662_data * v1693_data));
            float v1698_data = s2[95];
            float v1700_data = ir5[7];
            ir5[7] = (v1700_data + (v1662_data * v1698_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v1706_n1 = 0; v1706_n1 < 8; ++v1706_n1) {
              int32_t v1707_a = 0 + v1706_n1;
              float v1709_data = ir5[v1706_n1];
              int32_t v1710_a = 0 + v1706_n1;
              float v1712_data = r3[v1706_n1];
              int32_t v1714_a = 0 + v1706_n1;
              r5[v1706_n1] = (v1712_data + v1709_data);
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
          if (v3_lead < 12) {
            float v1722_data = r6[0];
            float v1723_data = s3[0];
            float v1725_data = ir7[0];
            ir7[0] = (v1725_data + (v1722_data * v1723_data));
            float v1728_data = s3[12];
            float v1730_data = ir7[1];
            ir7[1] = (v1730_data + (v1722_data * v1728_data));
            float v1733_data = s3[24];
            float v1735_data = ir7[2];
            ir7[2] = (v1735_data + (v1722_data * v1733_data));
            float v1738_data = s3[36];
            float v1740_data = ir7[3];
            ir7[3] = (v1740_data + (v1722_data * v1738_data));
            float v1743_data = s3[48];
            float v1745_data = ir7[4];
            ir7[4] = (v1745_data + (v1722_data * v1743_data));
            float v1748_data = s3[60];
            float v1750_data = ir7[5];
            ir7[5] = (v1750_data + (v1722_data * v1748_data));
            float v1753_data = s3[72];
            float v1755_data = ir7[6];
            ir7[6] = (v1755_data + (v1722_data * v1753_data));
            float v1758_data = s3[84];
            float v1760_data = ir7[7];
            ir7[7] = (v1760_data + (v1722_data * v1758_data));
          }
          if (v3_lead < 12) {
            float v1766_data = r6[1];
            float v1767_data = s3[1];
            float v1769_data = ir7[0];
            ir7[0] = (v1769_data + (v1766_data * v1767_data));
            float v1772_data = s3[13];
            float v1774_data = ir7[1];
            ir7[1] = (v1774_data + (v1766_data * v1772_data));
            float v1777_data = s3[25];
            float v1779_data = ir7[2];
            ir7[2] = (v1779_data + (v1766_data * v1777_data));
            float v1782_data = s3[37];
            float v1784_data = ir7[3];
            ir7[3] = (v1784_data + (v1766_data * v1782_data));
            float v1787_data = s3[49];
            float v1789_data = ir7[4];
            ir7[4] = (v1789_data + (v1766_data * v1787_data));
            float v1792_data = s3[61];
            float v1794_data = ir7[5];
            ir7[5] = (v1794_data + (v1766_data * v1792_data));
            float v1797_data = s3[73];
            float v1799_data = ir7[6];
            ir7[6] = (v1799_data + (v1766_data * v1797_data));
            float v1802_data = s3[85];
            float v1804_data = ir7[7];
            ir7[7] = (v1804_data + (v1766_data * v1802_data));
          }
          if (v3_lead < 12) {
            float v1810_data = r6[2];
            float v1811_data = s3[2];
            float v1813_data = ir7[0];
            ir7[0] = (v1813_data + (v1810_data * v1811_data));
            float v1816_data = s3[14];
            float v1818_data = ir7[1];
            ir7[1] = (v1818_data + (v1810_data * v1816_data));
            float v1821_data = s3[26];
            float v1823_data = ir7[2];
            ir7[2] = (v1823_data + (v1810_data * v1821_data));
            float v1826_data = s3[38];
            float v1828_data = ir7[3];
            ir7[3] = (v1828_data + (v1810_data * v1826_data));
            float v1831_data = s3[50];
            float v1833_data = ir7[4];
            ir7[4] = (v1833_data + (v1810_data * v1831_data));
            float v1836_data = s3[62];
            float v1838_data = ir7[5];
            ir7[5] = (v1838_data + (v1810_data * v1836_data));
            float v1841_data = s3[74];
            float v1843_data = ir7[6];
            ir7[6] = (v1843_data + (v1810_data * v1841_data));
            float v1846_data = s3[86];
            float v1848_data = ir7[7];
            ir7[7] = (v1848_data + (v1810_data * v1846_data));
          }
          if (v3_lead < 12) {
            float v1854_data = r6[3];
            float v1855_data = s3[3];
            float v1857_data = ir7[0];
            ir7[0] = (v1857_data + (v1854_data * v1855_data));
            float v1860_data = s3[15];
            float v1862_data = ir7[1];
            ir7[1] = (v1862_data + (v1854_data * v1860_data));
            float v1865_data = s3[27];
            float v1867_data = ir7[2];
            ir7[2] = (v1867_data + (v1854_data * v1865_data));
            float v1870_data = s3[39];
            float v1872_data = ir7[3];
            ir7[3] = (v1872_data + (v1854_data * v1870_data));
            float v1875_data = s3[51];
            float v1877_data = ir7[4];
            ir7[4] = (v1877_data + (v1854_data * v1875_data));
            float v1880_data = s3[63];
            float v1882_data = ir7[5];
            ir7[5] = (v1882_data + (v1854_data * v1880_data));
            float v1885_data = s3[75];
            float v1887_data = ir7[6];
            ir7[6] = (v1887_data + (v1854_data * v1885_data));
            float v1890_data = s3[87];
            float v1892_data = ir7[7];
            ir7[7] = (v1892_data + (v1854_data * v1890_data));
          }
          if (v3_lead < 12) {
            float v1898_data = r6[4];
            float v1899_data = s3[4];
            float v1901_data = ir7[0];
            ir7[0] = (v1901_data + (v1898_data * v1899_data));
            float v1904_data = s3[16];
            float v1906_data = ir7[1];
            ir7[1] = (v1906_data + (v1898_data * v1904_data));
            float v1909_data = s3[28];
            float v1911_data = ir7[2];
            ir7[2] = (v1911_data + (v1898_data * v1909_data));
            float v1914_data = s3[40];
            float v1916_data = ir7[3];
            ir7[3] = (v1916_data + (v1898_data * v1914_data));
            float v1919_data = s3[52];
            float v1921_data = ir7[4];
            ir7[4] = (v1921_data + (v1898_data * v1919_data));
            float v1924_data = s3[64];
            float v1926_data = ir7[5];
            ir7[5] = (v1926_data + (v1898_data * v1924_data));
            float v1929_data = s3[76];
            float v1931_data = ir7[6];
            ir7[6] = (v1931_data + (v1898_data * v1929_data));
            float v1934_data = s3[88];
            float v1936_data = ir7[7];
            ir7[7] = (v1936_data + (v1898_data * v1934_data));
          }
          if (v3_lead < 12) {
            float v1942_data = r6[5];
            float v1943_data = s3[5];
            float v1945_data = ir7[0];
            ir7[0] = (v1945_data + (v1942_data * v1943_data));
            float v1948_data = s3[17];
            float v1950_data = ir7[1];
            ir7[1] = (v1950_data + (v1942_data * v1948_data));
            float v1953_data = s3[29];
            float v1955_data = ir7[2];
            ir7[2] = (v1955_data + (v1942_data * v1953_data));
            float v1958_data = s3[41];
            float v1960_data = ir7[3];
            ir7[3] = (v1960_data + (v1942_data * v1958_data));
            float v1963_data = s3[53];
            float v1965_data = ir7[4];
            ir7[4] = (v1965_data + (v1942_data * v1963_data));
            float v1968_data = s3[65];
            float v1970_data = ir7[5];
            ir7[5] = (v1970_data + (v1942_data * v1968_data));
            float v1973_data = s3[77];
            float v1975_data = ir7[6];
            ir7[6] = (v1975_data + (v1942_data * v1973_data));
            float v1978_data = s3[89];
            float v1980_data = ir7[7];
            ir7[7] = (v1980_data + (v1942_data * v1978_data));
          }
          if (v3_lead < 12) {
            float v1986_data = r6[6];
            float v1987_data = s3[6];
            float v1989_data = ir7[0];
            ir7[0] = (v1989_data + (v1986_data * v1987_data));
            float v1992_data = s3[18];
            float v1994_data = ir7[1];
            ir7[1] = (v1994_data + (v1986_data * v1992_data));
            float v1997_data = s3[30];
            float v1999_data = ir7[2];
            ir7[2] = (v1999_data + (v1986_data * v1997_data));
            float v2002_data = s3[42];
            float v2004_data = ir7[3];
            ir7[3] = (v2004_data + (v1986_data * v2002_data));
            float v2007_data = s3[54];
            float v2009_data = ir7[4];
            ir7[4] = (v2009_data + (v1986_data * v2007_data));
            float v2012_data = s3[66];
            float v2014_data = ir7[5];
            ir7[5] = (v2014_data + (v1986_data * v2012_data));
            float v2017_data = s3[78];
            float v2019_data = ir7[6];
            ir7[6] = (v2019_data + (v1986_data * v2017_data));
            float v2022_data = s3[90];
            float v2024_data = ir7[7];
            ir7[7] = (v2024_data + (v1986_data * v2022_data));
          }
          if (v3_lead < 12) {
            float v2030_data = r6[7];
            float v2031_data = s3[7];
            float v2033_data = ir7[0];
            ir7[0] = (v2033_data + (v2030_data * v2031_data));
            float v2036_data = s3[19];
            float v2038_data = ir7[1];
            ir7[1] = (v2038_data + (v2030_data * v2036_data));
            float v2041_data = s3[31];
            float v2043_data = ir7[2];
            ir7[2] = (v2043_data + (v2030_data * v2041_data));
            float v2046_data = s3[43];
            float v2048_data = ir7[3];
            ir7[3] = (v2048_data + (v2030_data * v2046_data));
            float v2051_data = s3[55];
            float v2053_data = ir7[4];
            ir7[4] = (v2053_data + (v2030_data * v2051_data));
            float v2056_data = s3[67];
            float v2058_data = ir7[5];
            ir7[5] = (v2058_data + (v2030_data * v2056_data));
            float v2061_data = s3[79];
            float v2063_data = ir7[6];
            ir7[6] = (v2063_data + (v2030_data * v2061_data));
            float v2066_data = s3[91];
            float v2068_data = ir7[7];
            ir7[7] = (v2068_data + (v2030_data * v2066_data));
          }
          if (v3_lead < 12) {
            float v2074_data = r6[8];
            float v2075_data = s3[8];
            float v2077_data = ir7[0];
            ir7[0] = (v2077_data + (v2074_data * v2075_data));
            float v2080_data = s3[20];
            float v2082_data = ir7[1];
            ir7[1] = (v2082_data + (v2074_data * v2080_data));
            float v2085_data = s3[32];
            float v2087_data = ir7[2];
            ir7[2] = (v2087_data + (v2074_data * v2085_data));
            float v2090_data = s3[44];
            float v2092_data = ir7[3];
            ir7[3] = (v2092_data + (v2074_data * v2090_data));
            float v2095_data = s3[56];
            float v2097_data = ir7[4];
            ir7[4] = (v2097_data + (v2074_data * v2095_data));
            float v2100_data = s3[68];
            float v2102_data = ir7[5];
            ir7[5] = (v2102_data + (v2074_data * v2100_data));
            float v2105_data = s3[80];
            float v2107_data = ir7[6];
            ir7[6] = (v2107_data + (v2074_data * v2105_data));
            float v2110_data = s3[92];
            float v2112_data = ir7[7];
            ir7[7] = (v2112_data + (v2074_data * v2110_data));
          }
          if (v3_lead < 12) {
            float v2118_data = r6[9];
            float v2119_data = s3[9];
            float v2121_data = ir7[0];
            ir7[0] = (v2121_data + (v2118_data * v2119_data));
            float v2124_data = s3[21];
            float v2126_data = ir7[1];
            ir7[1] = (v2126_data + (v2118_data * v2124_data));
            float v2129_data = s3[33];
            float v2131_data = ir7[2];
            ir7[2] = (v2131_data + (v2118_data * v2129_data));
            float v2134_data = s3[45];
            float v2136_data = ir7[3];
            ir7[3] = (v2136_data + (v2118_data * v2134_data));
            float v2139_data = s3[57];
            float v2141_data = ir7[4];
            ir7[4] = (v2141_data + (v2118_data * v2139_data));
            float v2144_data = s3[69];
            float v2146_data = ir7[5];
            ir7[5] = (v2146_data + (v2118_data * v2144_data));
            float v2149_data = s3[81];
            float v2151_data = ir7[6];
            ir7[6] = (v2151_data + (v2118_data * v2149_data));
            float v2154_data = s3[93];
            float v2156_data = ir7[7];
            ir7[7] = (v2156_data + (v2118_data * v2154_data));
          }
          if (v3_lead < 12) {
            float v2162_data = r6[10];
            float v2163_data = s3[10];
            float v2165_data = ir7[0];
            ir7[0] = (v2165_data + (v2162_data * v2163_data));
            float v2168_data = s3[22];
            float v2170_data = ir7[1];
            ir7[1] = (v2170_data + (v2162_data * v2168_data));
            float v2173_data = s3[34];
            float v2175_data = ir7[2];
            ir7[2] = (v2175_data + (v2162_data * v2173_data));
            float v2178_data = s3[46];
            float v2180_data = ir7[3];
            ir7[3] = (v2180_data + (v2162_data * v2178_data));
            float v2183_data = s3[58];
            float v2185_data = ir7[4];
            ir7[4] = (v2185_data + (v2162_data * v2183_data));
            float v2188_data = s3[70];
            float v2190_data = ir7[5];
            ir7[5] = (v2190_data + (v2162_data * v2188_data));
            float v2193_data = s3[82];
            float v2195_data = ir7[6];
            ir7[6] = (v2195_data + (v2162_data * v2193_data));
            float v2198_data = s3[94];
            float v2200_data = ir7[7];
            ir7[7] = (v2200_data + (v2162_data * v2198_data));
          }
          if (v3_lead < 12) {
            float v2206_data = r6[11];
            float v2207_data = s3[11];
            float v2209_data = ir7[0];
            ir7[0] = (v2209_data + (v2206_data * v2207_data));
            float v2212_data = s3[23];
            float v2214_data = ir7[1];
            ir7[1] = (v2214_data + (v2206_data * v2212_data));
            float v2217_data = s3[35];
            float v2219_data = ir7[2];
            ir7[2] = (v2219_data + (v2206_data * v2217_data));
            float v2222_data = s3[47];
            float v2224_data = ir7[3];
            ir7[3] = (v2224_data + (v2206_data * v2222_data));
            float v2227_data = s3[59];
            float v2229_data = ir7[4];
            ir7[4] = (v2229_data + (v2206_data * v2227_data));
            float v2232_data = s3[71];
            float v2234_data = ir7[5];
            ir7[5] = (v2234_data + (v2206_data * v2232_data));
            float v2237_data = s3[83];
            float v2239_data = ir7[6];
            ir7[6] = (v2239_data + (v2206_data * v2237_data));
            float v2242_data = s3[95];
            float v2244_data = ir7[7];
            ir7[7] = (v2244_data + (v2206_data * v2242_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v2250_n1 = 0; v2250_n1 < 8; ++v2250_n1) {
              int32_t v2251_a = 0 + v2250_n1;
              float v2253_data = ir7[v2250_n1];
              int32_t v2254_a = 0 + v2250_n1;
              float v2256_data = r5[v2250_n1];
              int32_t v2258_a = 0 + v2250_n1;
              r7[v2250_n1] = (v2256_data + v2253_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v2264_i1 = 0; v2264_i1 < 8; ++v2264_i1) {
              int32_t v2265_a = 0 + v2264_i1;
              float v2267_data = r7[v2264_i1];
              int32_t v2274_a = v3_lead + (v2264_i1 * 12);
              glb_m0[v2274_a] = v2267_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

