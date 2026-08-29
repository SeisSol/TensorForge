// === base name ===
kernel_609dd06e89

// === header ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_609dd06e89, block.x * block.y * block.z, 512 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_609dd06e89, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_609dd06e89<<<grid,block,512 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // m4 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // C = abs(TMP)
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 32;
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 8; ++v17_i1) {
              float v25_data = __ldcg(&glb_m0[(v15_lead + (v17_i1 * 8))]);
              r0[v17_i1] = v25_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[8]{};
          // r2 = load{g>r}(glb_m2);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
              float v43_data = __ldcg(&glb_m2[(v15_lead + (v35_i1 * 8))]);
              r2[v35_i1] = v43_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          if (v15_lead < 8) {
            float v50_data = r0[0];
            float v51_data = s0[0];
            float v53_data = r1[0];
            r1[0] = (v53_data + (v50_data * v51_data));
            float v56_data = s0[8];
            float v58_data = r1[1];
            r1[1] = (v58_data + (v50_data * v56_data));
            float v61_data = s0[16];
            float v63_data = r1[2];
            r1[2] = (v63_data + (v50_data * v61_data));
            float v66_data = s0[24];
            float v68_data = r1[3];
            r1[3] = (v68_data + (v50_data * v66_data));
            float v71_data = s0[33];
            float v73_data = r1[4];
            r1[4] = (v73_data + (v50_data * v71_data));
            float v76_data = s0[41];
            float v78_data = r1[5];
            r1[5] = (v78_data + (v50_data * v76_data));
            float v81_data = s0[49];
            float v83_data = r1[6];
            r1[6] = (v83_data + (v50_data * v81_data));
            float v86_data = s0[57];
            float v88_data = r1[7];
            r1[7] = (v88_data + (v50_data * v86_data));
          }
          if (v15_lead < 8) {
            float v94_data = r0[1];
            float v95_data = s0[1];
            float v97_data = r1[0];
            r1[0] = (v97_data + (v94_data * v95_data));
            float v100_data = s0[9];
            float v102_data = r1[1];
            r1[1] = (v102_data + (v94_data * v100_data));
            float v105_data = s0[17];
            float v107_data = r1[2];
            r1[2] = (v107_data + (v94_data * v105_data));
            float v110_data = s0[25];
            float v112_data = r1[3];
            r1[3] = (v112_data + (v94_data * v110_data));
            float v115_data = s0[32];
            float v117_data = r1[4];
            r1[4] = (v117_data + (v94_data * v115_data));
            float v120_data = s0[40];
            float v122_data = r1[5];
            r1[5] = (v122_data + (v94_data * v120_data));
            float v125_data = s0[48];
            float v127_data = r1[6];
            r1[6] = (v127_data + (v94_data * v125_data));
            float v130_data = s0[56];
            float v132_data = r1[7];
            r1[7] = (v132_data + (v94_data * v130_data));
          }
          if (v15_lead < 8) {
            float v138_data = r0[2];
            float v139_data = s0[2];
            float v141_data = r1[0];
            r1[0] = (v141_data + (v138_data * v139_data));
            float v144_data = s0[10];
            float v146_data = r1[1];
            r1[1] = (v146_data + (v138_data * v144_data));
            float v149_data = s0[18];
            float v151_data = r1[2];
            r1[2] = (v151_data + (v138_data * v149_data));
            float v154_data = s0[26];
            float v156_data = r1[3];
            r1[3] = (v156_data + (v138_data * v154_data));
            float v159_data = s0[35];
            float v161_data = r1[4];
            r1[4] = (v161_data + (v138_data * v159_data));
            float v164_data = s0[43];
            float v166_data = r1[5];
            r1[5] = (v166_data + (v138_data * v164_data));
            float v169_data = s0[51];
            float v171_data = r1[6];
            r1[6] = (v171_data + (v138_data * v169_data));
            float v174_data = s0[59];
            float v176_data = r1[7];
            r1[7] = (v176_data + (v138_data * v174_data));
          }
          if (v15_lead < 8) {
            float v182_data = r0[3];
            float v183_data = s0[3];
            float v185_data = r1[0];
            r1[0] = (v185_data + (v182_data * v183_data));
            float v188_data = s0[11];
            float v190_data = r1[1];
            r1[1] = (v190_data + (v182_data * v188_data));
            float v193_data = s0[19];
            float v195_data = r1[2];
            r1[2] = (v195_data + (v182_data * v193_data));
            float v198_data = s0[27];
            float v200_data = r1[3];
            r1[3] = (v200_data + (v182_data * v198_data));
            float v203_data = s0[34];
            float v205_data = r1[4];
            r1[4] = (v205_data + (v182_data * v203_data));
            float v208_data = s0[42];
            float v210_data = r1[5];
            r1[5] = (v210_data + (v182_data * v208_data));
            float v213_data = s0[50];
            float v215_data = r1[6];
            r1[6] = (v215_data + (v182_data * v213_data));
            float v218_data = s0[58];
            float v220_data = r1[7];
            r1[7] = (v220_data + (v182_data * v218_data));
          }
          if (v15_lead < 8) {
            float v226_data = r0[4];
            float v227_data = s0[4];
            float v229_data = r1[0];
            r1[0] = (v229_data + (v226_data * v227_data));
            float v232_data = s0[12];
            float v234_data = r1[1];
            r1[1] = (v234_data + (v226_data * v232_data));
            float v237_data = s0[20];
            float v239_data = r1[2];
            r1[2] = (v239_data + (v226_data * v237_data));
            float v242_data = s0[28];
            float v244_data = r1[3];
            r1[3] = (v244_data + (v226_data * v242_data));
            float v247_data = s0[37];
            float v249_data = r1[4];
            r1[4] = (v249_data + (v226_data * v247_data));
            float v252_data = s0[45];
            float v254_data = r1[5];
            r1[5] = (v254_data + (v226_data * v252_data));
            float v257_data = s0[53];
            float v259_data = r1[6];
            r1[6] = (v259_data + (v226_data * v257_data));
            float v262_data = s0[61];
            float v264_data = r1[7];
            r1[7] = (v264_data + (v226_data * v262_data));
          }
          if (v15_lead < 8) {
            float v270_data = r0[5];
            float v271_data = s0[5];
            float v273_data = r1[0];
            r1[0] = (v273_data + (v270_data * v271_data));
            float v276_data = s0[13];
            float v278_data = r1[1];
            r1[1] = (v278_data + (v270_data * v276_data));
            float v281_data = s0[21];
            float v283_data = r1[2];
            r1[2] = (v283_data + (v270_data * v281_data));
            float v286_data = s0[29];
            float v288_data = r1[3];
            r1[3] = (v288_data + (v270_data * v286_data));
            float v291_data = s0[36];
            float v293_data = r1[4];
            r1[4] = (v293_data + (v270_data * v291_data));
            float v296_data = s0[44];
            float v298_data = r1[5];
            r1[5] = (v298_data + (v270_data * v296_data));
            float v301_data = s0[52];
            float v303_data = r1[6];
            r1[6] = (v303_data + (v270_data * v301_data));
            float v306_data = s0[60];
            float v308_data = r1[7];
            r1[7] = (v308_data + (v270_data * v306_data));
          }
          if (v15_lead < 8) {
            float v314_data = r0[6];
            float v315_data = s0[6];
            float v317_data = r1[0];
            r1[0] = (v317_data + (v314_data * v315_data));
            float v320_data = s0[14];
            float v322_data = r1[1];
            r1[1] = (v322_data + (v314_data * v320_data));
            float v325_data = s0[22];
            float v327_data = r1[2];
            r1[2] = (v327_data + (v314_data * v325_data));
            float v330_data = s0[30];
            float v332_data = r1[3];
            r1[3] = (v332_data + (v314_data * v330_data));
            float v335_data = s0[39];
            float v337_data = r1[4];
            r1[4] = (v337_data + (v314_data * v335_data));
            float v340_data = s0[47];
            float v342_data = r1[5];
            r1[5] = (v342_data + (v314_data * v340_data));
            float v345_data = s0[55];
            float v347_data = r1[6];
            r1[6] = (v347_data + (v314_data * v345_data));
            float v350_data = s0[63];
            float v352_data = r1[7];
            r1[7] = (v352_data + (v314_data * v350_data));
          }
          if (v15_lead < 8) {
            float v358_data = r0[7];
            float v359_data = s0[7];
            float v361_data = r1[0];
            r1[0] = (v361_data + (v358_data * v359_data));
            float v364_data = s0[15];
            float v366_data = r1[1];
            r1[1] = (v366_data + (v358_data * v364_data));
            float v369_data = s0[23];
            float v371_data = r1[2];
            r1[2] = (v371_data + (v358_data * v369_data));
            float v374_data = s0[31];
            float v376_data = r1[3];
            r1[3] = (v376_data + (v358_data * v374_data));
            float v379_data = s0[38];
            float v381_data = r1[4];
            r1[4] = (v381_data + (v358_data * v379_data));
            float v384_data = s0[46];
            float v386_data = r1[5];
            r1[5] = (v386_data + (v358_data * v384_data));
            float v389_data = s0[54];
            float v391_data = r1[6];
            r1[6] = (v391_data + (v358_data * v389_data));
            float v394_data = s0[62];
            float v396_data = r1[7];
            r1[7] = (v396_data + (v358_data * v394_data));
          }
          __syncwarp();
          float* __restrict__ s2 = &localShrMem0[0];
          // s2 = load{g>s}(glb_m3[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m3[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 32], &glb_m3[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m2););
          // wait(s2 = load{g>s}(glb_m3[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir3[8]{};
          if (v15_lead < 8) {
            float v407_data = r2[0];
            float v408_data = s2[0];
            float v410_data = ir3[0];
            ir3[0] = (v410_data + (v407_data * v408_data));
            float v413_data = s2[8];
            float v415_data = ir3[1];
            ir3[1] = (v415_data + (v407_data * v413_data));
            float v418_data = s2[16];
            float v420_data = ir3[2];
            ir3[2] = (v420_data + (v407_data * v418_data));
            float v423_data = s2[24];
            float v425_data = ir3[3];
            ir3[3] = (v425_data + (v407_data * v423_data));
            float v428_data = s2[33];
            float v430_data = ir3[4];
            ir3[4] = (v430_data + (v407_data * v428_data));
            float v433_data = s2[41];
            float v435_data = ir3[5];
            ir3[5] = (v435_data + (v407_data * v433_data));
            float v438_data = s2[49];
            float v440_data = ir3[6];
            ir3[6] = (v440_data + (v407_data * v438_data));
            float v443_data = s2[57];
            float v445_data = ir3[7];
            ir3[7] = (v445_data + (v407_data * v443_data));
          }
          if (v15_lead < 8) {
            float v451_data = r2[1];
            float v452_data = s2[1];
            float v454_data = ir3[0];
            ir3[0] = (v454_data + (v451_data * v452_data));
            float v457_data = s2[9];
            float v459_data = ir3[1];
            ir3[1] = (v459_data + (v451_data * v457_data));
            float v462_data = s2[17];
            float v464_data = ir3[2];
            ir3[2] = (v464_data + (v451_data * v462_data));
            float v467_data = s2[25];
            float v469_data = ir3[3];
            ir3[3] = (v469_data + (v451_data * v467_data));
            float v472_data = s2[32];
            float v474_data = ir3[4];
            ir3[4] = (v474_data + (v451_data * v472_data));
            float v477_data = s2[40];
            float v479_data = ir3[5];
            ir3[5] = (v479_data + (v451_data * v477_data));
            float v482_data = s2[48];
            float v484_data = ir3[6];
            ir3[6] = (v484_data + (v451_data * v482_data));
            float v487_data = s2[56];
            float v489_data = ir3[7];
            ir3[7] = (v489_data + (v451_data * v487_data));
          }
          if (v15_lead < 8) {
            float v495_data = r2[2];
            float v496_data = s2[2];
            float v498_data = ir3[0];
            ir3[0] = (v498_data + (v495_data * v496_data));
            float v501_data = s2[10];
            float v503_data = ir3[1];
            ir3[1] = (v503_data + (v495_data * v501_data));
            float v506_data = s2[18];
            float v508_data = ir3[2];
            ir3[2] = (v508_data + (v495_data * v506_data));
            float v511_data = s2[26];
            float v513_data = ir3[3];
            ir3[3] = (v513_data + (v495_data * v511_data));
            float v516_data = s2[35];
            float v518_data = ir3[4];
            ir3[4] = (v518_data + (v495_data * v516_data));
            float v521_data = s2[43];
            float v523_data = ir3[5];
            ir3[5] = (v523_data + (v495_data * v521_data));
            float v526_data = s2[51];
            float v528_data = ir3[6];
            ir3[6] = (v528_data + (v495_data * v526_data));
            float v531_data = s2[59];
            float v533_data = ir3[7];
            ir3[7] = (v533_data + (v495_data * v531_data));
          }
          if (v15_lead < 8) {
            float v539_data = r2[3];
            float v540_data = s2[3];
            float v542_data = ir3[0];
            ir3[0] = (v542_data + (v539_data * v540_data));
            float v545_data = s2[11];
            float v547_data = ir3[1];
            ir3[1] = (v547_data + (v539_data * v545_data));
            float v550_data = s2[19];
            float v552_data = ir3[2];
            ir3[2] = (v552_data + (v539_data * v550_data));
            float v555_data = s2[27];
            float v557_data = ir3[3];
            ir3[3] = (v557_data + (v539_data * v555_data));
            float v560_data = s2[34];
            float v562_data = ir3[4];
            ir3[4] = (v562_data + (v539_data * v560_data));
            float v565_data = s2[42];
            float v567_data = ir3[5];
            ir3[5] = (v567_data + (v539_data * v565_data));
            float v570_data = s2[50];
            float v572_data = ir3[6];
            ir3[6] = (v572_data + (v539_data * v570_data));
            float v575_data = s2[58];
            float v577_data = ir3[7];
            ir3[7] = (v577_data + (v539_data * v575_data));
          }
          if (v15_lead < 8) {
            float v583_data = r2[4];
            float v584_data = s2[4];
            float v586_data = ir3[0];
            ir3[0] = (v586_data + (v583_data * v584_data));
            float v589_data = s2[12];
            float v591_data = ir3[1];
            ir3[1] = (v591_data + (v583_data * v589_data));
            float v594_data = s2[20];
            float v596_data = ir3[2];
            ir3[2] = (v596_data + (v583_data * v594_data));
            float v599_data = s2[28];
            float v601_data = ir3[3];
            ir3[3] = (v601_data + (v583_data * v599_data));
            float v604_data = s2[37];
            float v606_data = ir3[4];
            ir3[4] = (v606_data + (v583_data * v604_data));
            float v609_data = s2[45];
            float v611_data = ir3[5];
            ir3[5] = (v611_data + (v583_data * v609_data));
            float v614_data = s2[53];
            float v616_data = ir3[6];
            ir3[6] = (v616_data + (v583_data * v614_data));
            float v619_data = s2[61];
            float v621_data = ir3[7];
            ir3[7] = (v621_data + (v583_data * v619_data));
          }
          if (v15_lead < 8) {
            float v627_data = r2[5];
            float v628_data = s2[5];
            float v630_data = ir3[0];
            ir3[0] = (v630_data + (v627_data * v628_data));
            float v633_data = s2[13];
            float v635_data = ir3[1];
            ir3[1] = (v635_data + (v627_data * v633_data));
            float v638_data = s2[21];
            float v640_data = ir3[2];
            ir3[2] = (v640_data + (v627_data * v638_data));
            float v643_data = s2[29];
            float v645_data = ir3[3];
            ir3[3] = (v645_data + (v627_data * v643_data));
            float v648_data = s2[36];
            float v650_data = ir3[4];
            ir3[4] = (v650_data + (v627_data * v648_data));
            float v653_data = s2[44];
            float v655_data = ir3[5];
            ir3[5] = (v655_data + (v627_data * v653_data));
            float v658_data = s2[52];
            float v660_data = ir3[6];
            ir3[6] = (v660_data + (v627_data * v658_data));
            float v663_data = s2[60];
            float v665_data = ir3[7];
            ir3[7] = (v665_data + (v627_data * v663_data));
          }
          if (v15_lead < 8) {
            float v671_data = r2[6];
            float v672_data = s2[6];
            float v674_data = ir3[0];
            ir3[0] = (v674_data + (v671_data * v672_data));
            float v677_data = s2[14];
            float v679_data = ir3[1];
            ir3[1] = (v679_data + (v671_data * v677_data));
            float v682_data = s2[22];
            float v684_data = ir3[2];
            ir3[2] = (v684_data + (v671_data * v682_data));
            float v687_data = s2[30];
            float v689_data = ir3[3];
            ir3[3] = (v689_data + (v671_data * v687_data));
            float v692_data = s2[39];
            float v694_data = ir3[4];
            ir3[4] = (v694_data + (v671_data * v692_data));
            float v697_data = s2[47];
            float v699_data = ir3[5];
            ir3[5] = (v699_data + (v671_data * v697_data));
            float v702_data = s2[55];
            float v704_data = ir3[6];
            ir3[6] = (v704_data + (v671_data * v702_data));
            float v707_data = s2[63];
            float v709_data = ir3[7];
            ir3[7] = (v709_data + (v671_data * v707_data));
          }
          if (v15_lead < 8) {
            float v715_data = r2[7];
            float v716_data = s2[7];
            float v718_data = ir3[0];
            ir3[0] = (v718_data + (v715_data * v716_data));
            float v721_data = s2[15];
            float v723_data = ir3[1];
            ir3[1] = (v723_data + (v715_data * v721_data));
            float v726_data = s2[23];
            float v728_data = ir3[2];
            ir3[2] = (v728_data + (v715_data * v726_data));
            float v731_data = s2[31];
            float v733_data = ir3[3];
            ir3[3] = (v733_data + (v715_data * v731_data));
            float v736_data = s2[38];
            float v738_data = ir3[4];
            ir3[4] = (v738_data + (v715_data * v736_data));
            float v741_data = s2[46];
            float v743_data = ir3[5];
            ir3[5] = (v743_data + (v715_data * v741_data));
            float v746_data = s2[54];
            float v748_data = ir3[6];
            ir3[6] = (v748_data + (v715_data * v746_data));
            float v751_data = s2[62];
            float v753_data = ir3[7];
            ir3[7] = (v753_data + (v715_data * v751_data));
          }
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v759_n1 = 0; v759_n1 < 8; ++v759_n1) {
              float v761_data = ir3[v759_n1];
              float v763_data = r1[v759_n1];
              r3[v759_n1] = (v763_data + v761_data);
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r3);
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v771_i1 = 0; v771_i1 < 8; ++v771_i1) {
              float v773_data = r3[v771_i1];
              int32_t v780_a = v15_lead + (v771_i1 * 8);
              s1[(v780_a ^ ((v780_a >> 5) & 31))] = v773_data;
            }
          }
          __syncwarp();
          // glb_m4 = abs(s1)
          if (v15_lead < 8) {
            #pragma unroll
            for (int32_t v788_k1 = 0; v788_k1 < 8; ++v788_k1) {
              int32_t v794_a = v788_k1 * 8;
              int32_t v795_a = v15_lead + v794_a;
              float v799_data = s1[(v795_a ^ ((v795_a >> 5) & 31))];
              glb_m4[(v15_lead + v794_a)] = (fabsf(v799_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

