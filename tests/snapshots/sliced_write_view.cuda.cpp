// === base name ===
kernel_939857c66e

// === header ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_939857c66e, block.x * block.y * block.z, 1536 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_939857c66e, cudaFuncAttributeMaxDynamicSharedMemorySize, 1536 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_939857c66e<<<grid,block,1536 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
    // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v15_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
            int32_t v22_lead = v15_lead + (v16_i0 * 32);
            #pragma unroll
            for (int32_t v17_i1 = 10; v17_i1 < 13; ++v17_i1) {
              float v25_data = __ldcg(&glb_m1[(v22_lead + (v17_i1 * 32))]);
              r0[(v16_i0 + (v17_i1 - 10))] = v25_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
            __pipeline_commit();
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[1]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v36_data = r0[0];
          float v37_data = s0[114];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v44_data = r0[1];
          float v45_data = s0[115];
          float v47_data = ir1[0];
          ir1[0] = (v47_data + (v44_data * v45_data));
          float v52_data = r0[2];
          float v53_data = s0[116];
          float v55_data = ir1[0];
          ir1[0] = (v55_data + (v52_data * v53_data));
          #pragma unroll
          for (int32_t v60_n0 = 0; v60_n0 < 1; ++v60_n0) {
            #pragma unroll
            for (int32_t v61_n1 = 0; v61_n1 < 1; ++v61_n1) {
              int32_t v62_a = v60_n0 + v61_n1;
              float v63_data = ir1[v62_a];
              r1[v62_a] = v63_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
            int32_t v76_lead = v15_lead + (v68_i0 * 32);
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 1; ++v69_i1) {
              float v71_data = r1[(v68_i0 + v69_i1)];
              glb_m0[(v76_lead + ((v69_i1 + 8) * 32))] = v71_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v84_i0 = 0; v84_i0 < 1; ++v84_i0) {
            int32_t v90_lead = v15_lead + (v84_i0 * 32);
            #pragma unroll
            for (int32_t v85_i1 = 0; v85_i1 < 13; ++v85_i1) {
              float v93_data = glb_m0[(v90_lead + (v85_i1 * 32))];
              r2[(v84_i0 + v85_i1)] = v93_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 32], 4);
            __pipeline_commit();
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 160], &glb_m4[0 + 0 + 1 * threadIdx.x + 160], 4);
            __pipeline_commit();
          }
          // wait(r2 = load{g>r}(glb_m0););
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[13]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v103_data = r2[0];
          float v104_data = s1[0];
          float v106_data = ir3[0];
          ir3[0] = (v106_data + (v103_data * v104_data));
          float v109_data = s1[13];
          float v111_data = ir3[1];
          ir3[1] = (v111_data + (v103_data * v109_data));
          float v114_data = s1[26];
          float v116_data = ir3[2];
          ir3[2] = (v116_data + (v103_data * v114_data));
          float v119_data = s1[39];
          float v121_data = ir3[3];
          ir3[3] = (v121_data + (v103_data * v119_data));
          float v124_data = s1[52];
          float v126_data = ir3[4];
          ir3[4] = (v126_data + (v103_data * v124_data));
          float v129_data = s1[65];
          float v131_data = ir3[5];
          ir3[5] = (v131_data + (v103_data * v129_data));
          float v134_data = s1[78];
          float v136_data = ir3[6];
          ir3[6] = (v136_data + (v103_data * v134_data));
          float v139_data = s1[91];
          float v141_data = ir3[7];
          ir3[7] = (v141_data + (v103_data * v139_data));
          float v144_data = s1[104];
          float v146_data = ir3[8];
          ir3[8] = (v146_data + (v103_data * v144_data));
          float v149_data = s1[117];
          float v151_data = ir3[9];
          ir3[9] = (v151_data + (v103_data * v149_data));
          float v154_data = s1[130];
          float v156_data = ir3[10];
          ir3[10] = (v156_data + (v103_data * v154_data));
          float v159_data = s1[143];
          float v161_data = ir3[11];
          ir3[11] = (v161_data + (v103_data * v159_data));
          float v164_data = s1[156];
          float v166_data = ir3[12];
          ir3[12] = (v166_data + (v103_data * v164_data));
          float v171_data = r2[1];
          float v172_data = s1[1];
          float v174_data = ir3[0];
          ir3[0] = (v174_data + (v171_data * v172_data));
          float v177_data = s1[14];
          float v179_data = ir3[1];
          ir3[1] = (v179_data + (v171_data * v177_data));
          float v182_data = s1[27];
          float v184_data = ir3[2];
          ir3[2] = (v184_data + (v171_data * v182_data));
          float v187_data = s1[40];
          float v189_data = ir3[3];
          ir3[3] = (v189_data + (v171_data * v187_data));
          float v192_data = s1[53];
          float v194_data = ir3[4];
          ir3[4] = (v194_data + (v171_data * v192_data));
          float v197_data = s1[66];
          float v199_data = ir3[5];
          ir3[5] = (v199_data + (v171_data * v197_data));
          float v202_data = s1[79];
          float v204_data = ir3[6];
          ir3[6] = (v204_data + (v171_data * v202_data));
          float v207_data = s1[92];
          float v209_data = ir3[7];
          ir3[7] = (v209_data + (v171_data * v207_data));
          float v212_data = s1[105];
          float v214_data = ir3[8];
          ir3[8] = (v214_data + (v171_data * v212_data));
          float v217_data = s1[118];
          float v219_data = ir3[9];
          ir3[9] = (v219_data + (v171_data * v217_data));
          float v222_data = s1[131];
          float v224_data = ir3[10];
          ir3[10] = (v224_data + (v171_data * v222_data));
          float v227_data = s1[144];
          float v229_data = ir3[11];
          ir3[11] = (v229_data + (v171_data * v227_data));
          float v232_data = s1[157];
          float v234_data = ir3[12];
          ir3[12] = (v234_data + (v171_data * v232_data));
          float v239_data = r2[2];
          float v240_data = s1[2];
          float v242_data = ir3[0];
          ir3[0] = (v242_data + (v239_data * v240_data));
          float v245_data = s1[15];
          float v247_data = ir3[1];
          ir3[1] = (v247_data + (v239_data * v245_data));
          float v250_data = s1[28];
          float v252_data = ir3[2];
          ir3[2] = (v252_data + (v239_data * v250_data));
          float v255_data = s1[41];
          float v257_data = ir3[3];
          ir3[3] = (v257_data + (v239_data * v255_data));
          float v260_data = s1[54];
          float v262_data = ir3[4];
          ir3[4] = (v262_data + (v239_data * v260_data));
          float v265_data = s1[67];
          float v267_data = ir3[5];
          ir3[5] = (v267_data + (v239_data * v265_data));
          float v270_data = s1[80];
          float v272_data = ir3[6];
          ir3[6] = (v272_data + (v239_data * v270_data));
          float v275_data = s1[93];
          float v277_data = ir3[7];
          ir3[7] = (v277_data + (v239_data * v275_data));
          float v280_data = s1[106];
          float v282_data = ir3[8];
          ir3[8] = (v282_data + (v239_data * v280_data));
          float v285_data = s1[119];
          float v287_data = ir3[9];
          ir3[9] = (v287_data + (v239_data * v285_data));
          float v290_data = s1[132];
          float v292_data = ir3[10];
          ir3[10] = (v292_data + (v239_data * v290_data));
          float v295_data = s1[145];
          float v297_data = ir3[11];
          ir3[11] = (v297_data + (v239_data * v295_data));
          float v300_data = s1[158];
          float v302_data = ir3[12];
          ir3[12] = (v302_data + (v239_data * v300_data));
          float v307_data = r2[3];
          float v308_data = s1[3];
          float v310_data = ir3[0];
          ir3[0] = (v310_data + (v307_data * v308_data));
          float v313_data = s1[16];
          float v315_data = ir3[1];
          ir3[1] = (v315_data + (v307_data * v313_data));
          float v318_data = s1[29];
          float v320_data = ir3[2];
          ir3[2] = (v320_data + (v307_data * v318_data));
          float v323_data = s1[42];
          float v325_data = ir3[3];
          ir3[3] = (v325_data + (v307_data * v323_data));
          float v328_data = s1[55];
          float v330_data = ir3[4];
          ir3[4] = (v330_data + (v307_data * v328_data));
          float v333_data = s1[68];
          float v335_data = ir3[5];
          ir3[5] = (v335_data + (v307_data * v333_data));
          float v338_data = s1[81];
          float v340_data = ir3[6];
          ir3[6] = (v340_data + (v307_data * v338_data));
          float v343_data = s1[94];
          float v345_data = ir3[7];
          ir3[7] = (v345_data + (v307_data * v343_data));
          float v348_data = s1[107];
          float v350_data = ir3[8];
          ir3[8] = (v350_data + (v307_data * v348_data));
          float v353_data = s1[120];
          float v355_data = ir3[9];
          ir3[9] = (v355_data + (v307_data * v353_data));
          float v358_data = s1[133];
          float v360_data = ir3[10];
          ir3[10] = (v360_data + (v307_data * v358_data));
          float v363_data = s1[146];
          float v365_data = ir3[11];
          ir3[11] = (v365_data + (v307_data * v363_data));
          float v368_data = s1[159];
          float v370_data = ir3[12];
          ir3[12] = (v370_data + (v307_data * v368_data));
          float v375_data = r2[4];
          float v376_data = s1[4];
          float v378_data = ir3[0];
          ir3[0] = (v378_data + (v375_data * v376_data));
          float v381_data = s1[17];
          float v383_data = ir3[1];
          ir3[1] = (v383_data + (v375_data * v381_data));
          float v386_data = s1[30];
          float v388_data = ir3[2];
          ir3[2] = (v388_data + (v375_data * v386_data));
          float v391_data = s1[43];
          float v393_data = ir3[3];
          ir3[3] = (v393_data + (v375_data * v391_data));
          float v396_data = s1[56];
          float v398_data = ir3[4];
          ir3[4] = (v398_data + (v375_data * v396_data));
          float v401_data = s1[69];
          float v403_data = ir3[5];
          ir3[5] = (v403_data + (v375_data * v401_data));
          float v406_data = s1[82];
          float v408_data = ir3[6];
          ir3[6] = (v408_data + (v375_data * v406_data));
          float v411_data = s1[95];
          float v413_data = ir3[7];
          ir3[7] = (v413_data + (v375_data * v411_data));
          float v416_data = s1[108];
          float v418_data = ir3[8];
          ir3[8] = (v418_data + (v375_data * v416_data));
          float v421_data = s1[121];
          float v423_data = ir3[9];
          ir3[9] = (v423_data + (v375_data * v421_data));
          float v426_data = s1[134];
          float v428_data = ir3[10];
          ir3[10] = (v428_data + (v375_data * v426_data));
          float v431_data = s1[147];
          float v433_data = ir3[11];
          ir3[11] = (v433_data + (v375_data * v431_data));
          float v436_data = s1[160];
          float v438_data = ir3[12];
          ir3[12] = (v438_data + (v375_data * v436_data));
          float v443_data = r2[5];
          float v444_data = s1[5];
          float v446_data = ir3[0];
          ir3[0] = (v446_data + (v443_data * v444_data));
          float v449_data = s1[18];
          float v451_data = ir3[1];
          ir3[1] = (v451_data + (v443_data * v449_data));
          float v454_data = s1[31];
          float v456_data = ir3[2];
          ir3[2] = (v456_data + (v443_data * v454_data));
          float v459_data = s1[44];
          float v461_data = ir3[3];
          ir3[3] = (v461_data + (v443_data * v459_data));
          float v464_data = s1[57];
          float v466_data = ir3[4];
          ir3[4] = (v466_data + (v443_data * v464_data));
          float v469_data = s1[70];
          float v471_data = ir3[5];
          ir3[5] = (v471_data + (v443_data * v469_data));
          float v474_data = s1[83];
          float v476_data = ir3[6];
          ir3[6] = (v476_data + (v443_data * v474_data));
          float v479_data = s1[96];
          float v481_data = ir3[7];
          ir3[7] = (v481_data + (v443_data * v479_data));
          float v484_data = s1[109];
          float v486_data = ir3[8];
          ir3[8] = (v486_data + (v443_data * v484_data));
          float v489_data = s1[122];
          float v491_data = ir3[9];
          ir3[9] = (v491_data + (v443_data * v489_data));
          float v494_data = s1[135];
          float v496_data = ir3[10];
          ir3[10] = (v496_data + (v443_data * v494_data));
          float v499_data = s1[148];
          float v501_data = ir3[11];
          ir3[11] = (v501_data + (v443_data * v499_data));
          float v504_data = s1[161];
          float v506_data = ir3[12];
          ir3[12] = (v506_data + (v443_data * v504_data));
          float v511_data = r2[6];
          float v512_data = s1[6];
          float v514_data = ir3[0];
          ir3[0] = (v514_data + (v511_data * v512_data));
          float v517_data = s1[19];
          float v519_data = ir3[1];
          ir3[1] = (v519_data + (v511_data * v517_data));
          float v522_data = s1[32];
          float v524_data = ir3[2];
          ir3[2] = (v524_data + (v511_data * v522_data));
          float v527_data = s1[45];
          float v529_data = ir3[3];
          ir3[3] = (v529_data + (v511_data * v527_data));
          float v532_data = s1[58];
          float v534_data = ir3[4];
          ir3[4] = (v534_data + (v511_data * v532_data));
          float v537_data = s1[71];
          float v539_data = ir3[5];
          ir3[5] = (v539_data + (v511_data * v537_data));
          float v542_data = s1[84];
          float v544_data = ir3[6];
          ir3[6] = (v544_data + (v511_data * v542_data));
          float v547_data = s1[97];
          float v549_data = ir3[7];
          ir3[7] = (v549_data + (v511_data * v547_data));
          float v552_data = s1[110];
          float v554_data = ir3[8];
          ir3[8] = (v554_data + (v511_data * v552_data));
          float v557_data = s1[123];
          float v559_data = ir3[9];
          ir3[9] = (v559_data + (v511_data * v557_data));
          float v562_data = s1[136];
          float v564_data = ir3[10];
          ir3[10] = (v564_data + (v511_data * v562_data));
          float v567_data = s1[149];
          float v569_data = ir3[11];
          ir3[11] = (v569_data + (v511_data * v567_data));
          float v572_data = s1[162];
          float v574_data = ir3[12];
          ir3[12] = (v574_data + (v511_data * v572_data));
          float v579_data = r2[7];
          float v580_data = s1[7];
          float v582_data = ir3[0];
          ir3[0] = (v582_data + (v579_data * v580_data));
          float v585_data = s1[20];
          float v587_data = ir3[1];
          ir3[1] = (v587_data + (v579_data * v585_data));
          float v590_data = s1[33];
          float v592_data = ir3[2];
          ir3[2] = (v592_data + (v579_data * v590_data));
          float v595_data = s1[46];
          float v597_data = ir3[3];
          ir3[3] = (v597_data + (v579_data * v595_data));
          float v600_data = s1[59];
          float v602_data = ir3[4];
          ir3[4] = (v602_data + (v579_data * v600_data));
          float v605_data = s1[72];
          float v607_data = ir3[5];
          ir3[5] = (v607_data + (v579_data * v605_data));
          float v610_data = s1[85];
          float v612_data = ir3[6];
          ir3[6] = (v612_data + (v579_data * v610_data));
          float v615_data = s1[98];
          float v617_data = ir3[7];
          ir3[7] = (v617_data + (v579_data * v615_data));
          float v620_data = s1[111];
          float v622_data = ir3[8];
          ir3[8] = (v622_data + (v579_data * v620_data));
          float v625_data = s1[124];
          float v627_data = ir3[9];
          ir3[9] = (v627_data + (v579_data * v625_data));
          float v630_data = s1[137];
          float v632_data = ir3[10];
          ir3[10] = (v632_data + (v579_data * v630_data));
          float v635_data = s1[150];
          float v637_data = ir3[11];
          ir3[11] = (v637_data + (v579_data * v635_data));
          float v640_data = s1[163];
          float v642_data = ir3[12];
          ir3[12] = (v642_data + (v579_data * v640_data));
          float v647_data = r2[8];
          float v648_data = s1[8];
          float v650_data = ir3[0];
          ir3[0] = (v650_data + (v647_data * v648_data));
          float v653_data = s1[21];
          float v655_data = ir3[1];
          ir3[1] = (v655_data + (v647_data * v653_data));
          float v658_data = s1[34];
          float v660_data = ir3[2];
          ir3[2] = (v660_data + (v647_data * v658_data));
          float v663_data = s1[47];
          float v665_data = ir3[3];
          ir3[3] = (v665_data + (v647_data * v663_data));
          float v668_data = s1[60];
          float v670_data = ir3[4];
          ir3[4] = (v670_data + (v647_data * v668_data));
          float v673_data = s1[73];
          float v675_data = ir3[5];
          ir3[5] = (v675_data + (v647_data * v673_data));
          float v678_data = s1[86];
          float v680_data = ir3[6];
          ir3[6] = (v680_data + (v647_data * v678_data));
          float v683_data = s1[99];
          float v685_data = ir3[7];
          ir3[7] = (v685_data + (v647_data * v683_data));
          float v688_data = s1[112];
          float v690_data = ir3[8];
          ir3[8] = (v690_data + (v647_data * v688_data));
          float v693_data = s1[125];
          float v695_data = ir3[9];
          ir3[9] = (v695_data + (v647_data * v693_data));
          float v698_data = s1[138];
          float v700_data = ir3[10];
          ir3[10] = (v700_data + (v647_data * v698_data));
          float v703_data = s1[151];
          float v705_data = ir3[11];
          ir3[11] = (v705_data + (v647_data * v703_data));
          float v708_data = s1[164];
          float v710_data = ir3[12];
          ir3[12] = (v710_data + (v647_data * v708_data));
          float v715_data = r2[9];
          float v716_data = s1[9];
          float v718_data = ir3[0];
          ir3[0] = (v718_data + (v715_data * v716_data));
          float v721_data = s1[22];
          float v723_data = ir3[1];
          ir3[1] = (v723_data + (v715_data * v721_data));
          float v726_data = s1[35];
          float v728_data = ir3[2];
          ir3[2] = (v728_data + (v715_data * v726_data));
          float v731_data = s1[48];
          float v733_data = ir3[3];
          ir3[3] = (v733_data + (v715_data * v731_data));
          float v736_data = s1[61];
          float v738_data = ir3[4];
          ir3[4] = (v738_data + (v715_data * v736_data));
          float v741_data = s1[74];
          float v743_data = ir3[5];
          ir3[5] = (v743_data + (v715_data * v741_data));
          float v746_data = s1[87];
          float v748_data = ir3[6];
          ir3[6] = (v748_data + (v715_data * v746_data));
          float v751_data = s1[100];
          float v753_data = ir3[7];
          ir3[7] = (v753_data + (v715_data * v751_data));
          float v756_data = s1[113];
          float v758_data = ir3[8];
          ir3[8] = (v758_data + (v715_data * v756_data));
          float v761_data = s1[126];
          float v763_data = ir3[9];
          ir3[9] = (v763_data + (v715_data * v761_data));
          float v766_data = s1[139];
          float v768_data = ir3[10];
          ir3[10] = (v768_data + (v715_data * v766_data));
          float v771_data = s1[152];
          float v773_data = ir3[11];
          ir3[11] = (v773_data + (v715_data * v771_data));
          float v776_data = s1[165];
          float v778_data = ir3[12];
          ir3[12] = (v778_data + (v715_data * v776_data));
          float v783_data = r2[10];
          float v784_data = s1[10];
          float v786_data = ir3[0];
          ir3[0] = (v786_data + (v783_data * v784_data));
          float v789_data = s1[23];
          float v791_data = ir3[1];
          ir3[1] = (v791_data + (v783_data * v789_data));
          float v794_data = s1[36];
          float v796_data = ir3[2];
          ir3[2] = (v796_data + (v783_data * v794_data));
          float v799_data = s1[49];
          float v801_data = ir3[3];
          ir3[3] = (v801_data + (v783_data * v799_data));
          float v804_data = s1[62];
          float v806_data = ir3[4];
          ir3[4] = (v806_data + (v783_data * v804_data));
          float v809_data = s1[75];
          float v811_data = ir3[5];
          ir3[5] = (v811_data + (v783_data * v809_data));
          float v814_data = s1[88];
          float v816_data = ir3[6];
          ir3[6] = (v816_data + (v783_data * v814_data));
          float v819_data = s1[101];
          float v821_data = ir3[7];
          ir3[7] = (v821_data + (v783_data * v819_data));
          float v824_data = s1[114];
          float v826_data = ir3[8];
          ir3[8] = (v826_data + (v783_data * v824_data));
          float v829_data = s1[127];
          float v831_data = ir3[9];
          ir3[9] = (v831_data + (v783_data * v829_data));
          float v834_data = s1[140];
          float v836_data = ir3[10];
          ir3[10] = (v836_data + (v783_data * v834_data));
          float v839_data = s1[153];
          float v841_data = ir3[11];
          ir3[11] = (v841_data + (v783_data * v839_data));
          float v844_data = s1[166];
          float v846_data = ir3[12];
          ir3[12] = (v846_data + (v783_data * v844_data));
          float v851_data = r2[11];
          float v852_data = s1[11];
          float v854_data = ir3[0];
          ir3[0] = (v854_data + (v851_data * v852_data));
          float v857_data = s1[24];
          float v859_data = ir3[1];
          ir3[1] = (v859_data + (v851_data * v857_data));
          float v862_data = s1[37];
          float v864_data = ir3[2];
          ir3[2] = (v864_data + (v851_data * v862_data));
          float v867_data = s1[50];
          float v869_data = ir3[3];
          ir3[3] = (v869_data + (v851_data * v867_data));
          float v872_data = s1[63];
          float v874_data = ir3[4];
          ir3[4] = (v874_data + (v851_data * v872_data));
          float v877_data = s1[76];
          float v879_data = ir3[5];
          ir3[5] = (v879_data + (v851_data * v877_data));
          float v882_data = s1[89];
          float v884_data = ir3[6];
          ir3[6] = (v884_data + (v851_data * v882_data));
          float v887_data = s1[102];
          float v889_data = ir3[7];
          ir3[7] = (v889_data + (v851_data * v887_data));
          float v892_data = s1[115];
          float v894_data = ir3[8];
          ir3[8] = (v894_data + (v851_data * v892_data));
          float v897_data = s1[128];
          float v899_data = ir3[9];
          ir3[9] = (v899_data + (v851_data * v897_data));
          float v902_data = s1[141];
          float v904_data = ir3[10];
          ir3[10] = (v904_data + (v851_data * v902_data));
          float v907_data = s1[154];
          float v909_data = ir3[11];
          ir3[11] = (v909_data + (v851_data * v907_data));
          float v912_data = s1[167];
          float v914_data = ir3[12];
          ir3[12] = (v914_data + (v851_data * v912_data));
          float v919_data = r2[12];
          float v920_data = s1[12];
          float v922_data = ir3[0];
          ir3[0] = (v922_data + (v919_data * v920_data));
          float v925_data = s1[25];
          float v927_data = ir3[1];
          ir3[1] = (v927_data + (v919_data * v925_data));
          float v930_data = s1[38];
          float v932_data = ir3[2];
          ir3[2] = (v932_data + (v919_data * v930_data));
          float v935_data = s1[51];
          float v937_data = ir3[3];
          ir3[3] = (v937_data + (v919_data * v935_data));
          float v940_data = s1[64];
          float v942_data = ir3[4];
          ir3[4] = (v942_data + (v919_data * v940_data));
          float v945_data = s1[77];
          float v947_data = ir3[5];
          ir3[5] = (v947_data + (v919_data * v945_data));
          float v950_data = s1[90];
          float v952_data = ir3[6];
          ir3[6] = (v952_data + (v919_data * v950_data));
          float v955_data = s1[103];
          float v957_data = ir3[7];
          ir3[7] = (v957_data + (v919_data * v955_data));
          float v960_data = s1[116];
          float v962_data = ir3[8];
          ir3[8] = (v962_data + (v919_data * v960_data));
          float v965_data = s1[129];
          float v967_data = ir3[9];
          ir3[9] = (v967_data + (v919_data * v965_data));
          float v970_data = s1[142];
          float v972_data = ir3[10];
          ir3[10] = (v972_data + (v919_data * v970_data));
          float v975_data = s1[155];
          float v977_data = ir3[11];
          ir3[11] = (v977_data + (v919_data * v975_data));
          float v980_data = s1[168];
          float v982_data = ir3[12];
          ir3[12] = (v982_data + (v919_data * v980_data));
          #pragma unroll
          for (int32_t v987_n0 = 0; v987_n0 < 1; ++v987_n0) {
            #pragma unroll
            for (int32_t v988_n1 = 0; v988_n1 < 13; ++v988_n1) {
              int32_t v989_a = v987_n0 + v988_n1;
              float v990_data = ir3[v989_a];
              r3[v989_a] = v990_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v995_i0 = 0; v995_i0 < 1; ++v995_i0) {
            int32_t v1003_lead = v15_lead + (v995_i0 * 32);
            #pragma unroll
            for (int32_t v996_i1 = 0; v996_i1 < 13; ++v996_i1) {
              float v998_data = r3[(v995_i0 + v996_i1)];
              glb_m3[(v1003_lead + (v996_i1 * 32))] = v998_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

