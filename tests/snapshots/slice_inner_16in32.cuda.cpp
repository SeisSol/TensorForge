// === base name ===
kernel_87f2838a59

// === header ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_87f2838a59, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_87f2838a59, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_87f2838a59<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v10_off = (v2_lead + (v3_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v4_i1 = 8; v4_i1 < 24; ++v4_i1) {
              int32_t v12_a = v10_off + (v4_i1 * 32);
              float v13_data;
              {
                v13_data = __ldcg(&glb_m1[v12_a]);
              }
              int32_t v15_a = v3_i0 + (v4_i1 - 8);
              r0[v15_a] = v13_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[8]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 8)] [(0, 16)]
            float ir1[8]{};
            float v19_data = r0[0];
            float v20_data = s0[0];
            float v22_data = ir1[0];
            ir1[0] = (v22_data + (v19_data * v20_data));
            float v25_data = s0[16];
            float v27_data = ir1[1];
            ir1[1] = (v27_data + (v19_data * v25_data));
            float v30_data = s0[32];
            float v32_data = ir1[2];
            ir1[2] = (v32_data + (v19_data * v30_data));
            float v35_data = s0[48];
            float v37_data = ir1[3];
            ir1[3] = (v37_data + (v19_data * v35_data));
            float v40_data = s0[64];
            float v42_data = ir1[4];
            ir1[4] = (v42_data + (v19_data * v40_data));
            float v45_data = s0[80];
            float v47_data = ir1[5];
            ir1[5] = (v47_data + (v19_data * v45_data));
            float v50_data = s0[96];
            float v52_data = ir1[6];
            ir1[6] = (v52_data + (v19_data * v50_data));
            float v55_data = s0[112];
            float v57_data = ir1[7];
            ir1[7] = (v57_data + (v19_data * v55_data));
            float v62_data = r0[1];
            float v63_data = s0[1];
            float v65_data = ir1[0];
            ir1[0] = (v65_data + (v62_data * v63_data));
            float v68_data = s0[17];
            float v70_data = ir1[1];
            ir1[1] = (v70_data + (v62_data * v68_data));
            float v73_data = s0[33];
            float v75_data = ir1[2];
            ir1[2] = (v75_data + (v62_data * v73_data));
            float v78_data = s0[49];
            float v80_data = ir1[3];
            ir1[3] = (v80_data + (v62_data * v78_data));
            float v83_data = s0[65];
            float v85_data = ir1[4];
            ir1[4] = (v85_data + (v62_data * v83_data));
            float v88_data = s0[81];
            float v90_data = ir1[5];
            ir1[5] = (v90_data + (v62_data * v88_data));
            float v93_data = s0[97];
            float v95_data = ir1[6];
            ir1[6] = (v95_data + (v62_data * v93_data));
            float v98_data = s0[113];
            float v100_data = ir1[7];
            ir1[7] = (v100_data + (v62_data * v98_data));
            float v105_data = r0[2];
            float v106_data = s0[2];
            float v108_data = ir1[0];
            ir1[0] = (v108_data + (v105_data * v106_data));
            float v111_data = s0[18];
            float v113_data = ir1[1];
            ir1[1] = (v113_data + (v105_data * v111_data));
            float v116_data = s0[34];
            float v118_data = ir1[2];
            ir1[2] = (v118_data + (v105_data * v116_data));
            float v121_data = s0[50];
            float v123_data = ir1[3];
            ir1[3] = (v123_data + (v105_data * v121_data));
            float v126_data = s0[66];
            float v128_data = ir1[4];
            ir1[4] = (v128_data + (v105_data * v126_data));
            float v131_data = s0[82];
            float v133_data = ir1[5];
            ir1[5] = (v133_data + (v105_data * v131_data));
            float v136_data = s0[98];
            float v138_data = ir1[6];
            ir1[6] = (v138_data + (v105_data * v136_data));
            float v141_data = s0[114];
            float v143_data = ir1[7];
            ir1[7] = (v143_data + (v105_data * v141_data));
            float v148_data = r0[3];
            float v149_data = s0[3];
            float v151_data = ir1[0];
            ir1[0] = (v151_data + (v148_data * v149_data));
            float v154_data = s0[19];
            float v156_data = ir1[1];
            ir1[1] = (v156_data + (v148_data * v154_data));
            float v159_data = s0[35];
            float v161_data = ir1[2];
            ir1[2] = (v161_data + (v148_data * v159_data));
            float v164_data = s0[51];
            float v166_data = ir1[3];
            ir1[3] = (v166_data + (v148_data * v164_data));
            float v169_data = s0[67];
            float v171_data = ir1[4];
            ir1[4] = (v171_data + (v148_data * v169_data));
            float v174_data = s0[83];
            float v176_data = ir1[5];
            ir1[5] = (v176_data + (v148_data * v174_data));
            float v179_data = s0[99];
            float v181_data = ir1[6];
            ir1[6] = (v181_data + (v148_data * v179_data));
            float v184_data = s0[115];
            float v186_data = ir1[7];
            ir1[7] = (v186_data + (v148_data * v184_data));
            float v191_data = r0[4];
            float v192_data = s0[4];
            float v194_data = ir1[0];
            ir1[0] = (v194_data + (v191_data * v192_data));
            float v197_data = s0[20];
            float v199_data = ir1[1];
            ir1[1] = (v199_data + (v191_data * v197_data));
            float v202_data = s0[36];
            float v204_data = ir1[2];
            ir1[2] = (v204_data + (v191_data * v202_data));
            float v207_data = s0[52];
            float v209_data = ir1[3];
            ir1[3] = (v209_data + (v191_data * v207_data));
            float v212_data = s0[68];
            float v214_data = ir1[4];
            ir1[4] = (v214_data + (v191_data * v212_data));
            float v217_data = s0[84];
            float v219_data = ir1[5];
            ir1[5] = (v219_data + (v191_data * v217_data));
            float v222_data = s0[100];
            float v224_data = ir1[6];
            ir1[6] = (v224_data + (v191_data * v222_data));
            float v227_data = s0[116];
            float v229_data = ir1[7];
            ir1[7] = (v229_data + (v191_data * v227_data));
            float v234_data = r0[5];
            float v235_data = s0[5];
            float v237_data = ir1[0];
            ir1[0] = (v237_data + (v234_data * v235_data));
            float v240_data = s0[21];
            float v242_data = ir1[1];
            ir1[1] = (v242_data + (v234_data * v240_data));
            float v245_data = s0[37];
            float v247_data = ir1[2];
            ir1[2] = (v247_data + (v234_data * v245_data));
            float v250_data = s0[53];
            float v252_data = ir1[3];
            ir1[3] = (v252_data + (v234_data * v250_data));
            float v255_data = s0[69];
            float v257_data = ir1[4];
            ir1[4] = (v257_data + (v234_data * v255_data));
            float v260_data = s0[85];
            float v262_data = ir1[5];
            ir1[5] = (v262_data + (v234_data * v260_data));
            float v265_data = s0[101];
            float v267_data = ir1[6];
            ir1[6] = (v267_data + (v234_data * v265_data));
            float v270_data = s0[117];
            float v272_data = ir1[7];
            ir1[7] = (v272_data + (v234_data * v270_data));
            float v277_data = r0[6];
            float v278_data = s0[6];
            float v280_data = ir1[0];
            ir1[0] = (v280_data + (v277_data * v278_data));
            float v283_data = s0[22];
            float v285_data = ir1[1];
            ir1[1] = (v285_data + (v277_data * v283_data));
            float v288_data = s0[38];
            float v290_data = ir1[2];
            ir1[2] = (v290_data + (v277_data * v288_data));
            float v293_data = s0[54];
            float v295_data = ir1[3];
            ir1[3] = (v295_data + (v277_data * v293_data));
            float v298_data = s0[70];
            float v300_data = ir1[4];
            ir1[4] = (v300_data + (v277_data * v298_data));
            float v303_data = s0[86];
            float v305_data = ir1[5];
            ir1[5] = (v305_data + (v277_data * v303_data));
            float v308_data = s0[102];
            float v310_data = ir1[6];
            ir1[6] = (v310_data + (v277_data * v308_data));
            float v313_data = s0[118];
            float v315_data = ir1[7];
            ir1[7] = (v315_data + (v277_data * v313_data));
            float v320_data = r0[7];
            float v321_data = s0[7];
            float v323_data = ir1[0];
            ir1[0] = (v323_data + (v320_data * v321_data));
            float v326_data = s0[23];
            float v328_data = ir1[1];
            ir1[1] = (v328_data + (v320_data * v326_data));
            float v331_data = s0[39];
            float v333_data = ir1[2];
            ir1[2] = (v333_data + (v320_data * v331_data));
            float v336_data = s0[55];
            float v338_data = ir1[3];
            ir1[3] = (v338_data + (v320_data * v336_data));
            float v341_data = s0[71];
            float v343_data = ir1[4];
            ir1[4] = (v343_data + (v320_data * v341_data));
            float v346_data = s0[87];
            float v348_data = ir1[5];
            ir1[5] = (v348_data + (v320_data * v346_data));
            float v351_data = s0[103];
            float v353_data = ir1[6];
            ir1[6] = (v353_data + (v320_data * v351_data));
            float v356_data = s0[119];
            float v358_data = ir1[7];
            ir1[7] = (v358_data + (v320_data * v356_data));
            float v363_data = r0[8];
            float v364_data = s0[8];
            float v366_data = ir1[0];
            ir1[0] = (v366_data + (v363_data * v364_data));
            float v369_data = s0[24];
            float v371_data = ir1[1];
            ir1[1] = (v371_data + (v363_data * v369_data));
            float v374_data = s0[40];
            float v376_data = ir1[2];
            ir1[2] = (v376_data + (v363_data * v374_data));
            float v379_data = s0[56];
            float v381_data = ir1[3];
            ir1[3] = (v381_data + (v363_data * v379_data));
            float v384_data = s0[72];
            float v386_data = ir1[4];
            ir1[4] = (v386_data + (v363_data * v384_data));
            float v389_data = s0[88];
            float v391_data = ir1[5];
            ir1[5] = (v391_data + (v363_data * v389_data));
            float v394_data = s0[104];
            float v396_data = ir1[6];
            ir1[6] = (v396_data + (v363_data * v394_data));
            float v399_data = s0[120];
            float v401_data = ir1[7];
            ir1[7] = (v401_data + (v363_data * v399_data));
            float v406_data = r0[9];
            float v407_data = s0[9];
            float v409_data = ir1[0];
            ir1[0] = (v409_data + (v406_data * v407_data));
            float v412_data = s0[25];
            float v414_data = ir1[1];
            ir1[1] = (v414_data + (v406_data * v412_data));
            float v417_data = s0[41];
            float v419_data = ir1[2];
            ir1[2] = (v419_data + (v406_data * v417_data));
            float v422_data = s0[57];
            float v424_data = ir1[3];
            ir1[3] = (v424_data + (v406_data * v422_data));
            float v427_data = s0[73];
            float v429_data = ir1[4];
            ir1[4] = (v429_data + (v406_data * v427_data));
            float v432_data = s0[89];
            float v434_data = ir1[5];
            ir1[5] = (v434_data + (v406_data * v432_data));
            float v437_data = s0[105];
            float v439_data = ir1[6];
            ir1[6] = (v439_data + (v406_data * v437_data));
            float v442_data = s0[121];
            float v444_data = ir1[7];
            ir1[7] = (v444_data + (v406_data * v442_data));
            float v449_data = r0[10];
            float v450_data = s0[10];
            float v452_data = ir1[0];
            ir1[0] = (v452_data + (v449_data * v450_data));
            float v455_data = s0[26];
            float v457_data = ir1[1];
            ir1[1] = (v457_data + (v449_data * v455_data));
            float v460_data = s0[42];
            float v462_data = ir1[2];
            ir1[2] = (v462_data + (v449_data * v460_data));
            float v465_data = s0[58];
            float v467_data = ir1[3];
            ir1[3] = (v467_data + (v449_data * v465_data));
            float v470_data = s0[74];
            float v472_data = ir1[4];
            ir1[4] = (v472_data + (v449_data * v470_data));
            float v475_data = s0[90];
            float v477_data = ir1[5];
            ir1[5] = (v477_data + (v449_data * v475_data));
            float v480_data = s0[106];
            float v482_data = ir1[6];
            ir1[6] = (v482_data + (v449_data * v480_data));
            float v485_data = s0[122];
            float v487_data = ir1[7];
            ir1[7] = (v487_data + (v449_data * v485_data));
            float v492_data = r0[11];
            float v493_data = s0[11];
            float v495_data = ir1[0];
            ir1[0] = (v495_data + (v492_data * v493_data));
            float v498_data = s0[27];
            float v500_data = ir1[1];
            ir1[1] = (v500_data + (v492_data * v498_data));
            float v503_data = s0[43];
            float v505_data = ir1[2];
            ir1[2] = (v505_data + (v492_data * v503_data));
            float v508_data = s0[59];
            float v510_data = ir1[3];
            ir1[3] = (v510_data + (v492_data * v508_data));
            float v513_data = s0[75];
            float v515_data = ir1[4];
            ir1[4] = (v515_data + (v492_data * v513_data));
            float v518_data = s0[91];
            float v520_data = ir1[5];
            ir1[5] = (v520_data + (v492_data * v518_data));
            float v523_data = s0[107];
            float v525_data = ir1[6];
            ir1[6] = (v525_data + (v492_data * v523_data));
            float v528_data = s0[123];
            float v530_data = ir1[7];
            ir1[7] = (v530_data + (v492_data * v528_data));
            float v535_data = r0[12];
            float v536_data = s0[12];
            float v538_data = ir1[0];
            ir1[0] = (v538_data + (v535_data * v536_data));
            float v541_data = s0[28];
            float v543_data = ir1[1];
            ir1[1] = (v543_data + (v535_data * v541_data));
            float v546_data = s0[44];
            float v548_data = ir1[2];
            ir1[2] = (v548_data + (v535_data * v546_data));
            float v551_data = s0[60];
            float v553_data = ir1[3];
            ir1[3] = (v553_data + (v535_data * v551_data));
            float v556_data = s0[76];
            float v558_data = ir1[4];
            ir1[4] = (v558_data + (v535_data * v556_data));
            float v561_data = s0[92];
            float v563_data = ir1[5];
            ir1[5] = (v563_data + (v535_data * v561_data));
            float v566_data = s0[108];
            float v568_data = ir1[6];
            ir1[6] = (v568_data + (v535_data * v566_data));
            float v571_data = s0[124];
            float v573_data = ir1[7];
            ir1[7] = (v573_data + (v535_data * v571_data));
            float v578_data = r0[13];
            float v579_data = s0[13];
            float v581_data = ir1[0];
            ir1[0] = (v581_data + (v578_data * v579_data));
            float v584_data = s0[29];
            float v586_data = ir1[1];
            ir1[1] = (v586_data + (v578_data * v584_data));
            float v589_data = s0[45];
            float v591_data = ir1[2];
            ir1[2] = (v591_data + (v578_data * v589_data));
            float v594_data = s0[61];
            float v596_data = ir1[3];
            ir1[3] = (v596_data + (v578_data * v594_data));
            float v599_data = s0[77];
            float v601_data = ir1[4];
            ir1[4] = (v601_data + (v578_data * v599_data));
            float v604_data = s0[93];
            float v606_data = ir1[5];
            ir1[5] = (v606_data + (v578_data * v604_data));
            float v609_data = s0[109];
            float v611_data = ir1[6];
            ir1[6] = (v611_data + (v578_data * v609_data));
            float v614_data = s0[125];
            float v616_data = ir1[7];
            ir1[7] = (v616_data + (v578_data * v614_data));
            float v621_data = r0[14];
            float v622_data = s0[14];
            float v624_data = ir1[0];
            ir1[0] = (v624_data + (v621_data * v622_data));
            float v627_data = s0[30];
            float v629_data = ir1[1];
            ir1[1] = (v629_data + (v621_data * v627_data));
            float v632_data = s0[46];
            float v634_data = ir1[2];
            ir1[2] = (v634_data + (v621_data * v632_data));
            float v637_data = s0[62];
            float v639_data = ir1[3];
            ir1[3] = (v639_data + (v621_data * v637_data));
            float v642_data = s0[78];
            float v644_data = ir1[4];
            ir1[4] = (v644_data + (v621_data * v642_data));
            float v647_data = s0[94];
            float v649_data = ir1[5];
            ir1[5] = (v649_data + (v621_data * v647_data));
            float v652_data = s0[110];
            float v654_data = ir1[6];
            ir1[6] = (v654_data + (v621_data * v652_data));
            float v657_data = s0[126];
            float v659_data = ir1[7];
            ir1[7] = (v659_data + (v621_data * v657_data));
            float v664_data = r0[15];
            float v665_data = s0[15];
            float v667_data = ir1[0];
            ir1[0] = (v667_data + (v664_data * v665_data));
            float v670_data = s0[31];
            float v672_data = ir1[1];
            ir1[1] = (v672_data + (v664_data * v670_data));
            float v675_data = s0[47];
            float v677_data = ir1[2];
            ir1[2] = (v677_data + (v664_data * v675_data));
            float v680_data = s0[63];
            float v682_data = ir1[3];
            ir1[3] = (v682_data + (v664_data * v680_data));
            float v685_data = s0[79];
            float v687_data = ir1[4];
            ir1[4] = (v687_data + (v664_data * v685_data));
            float v690_data = s0[95];
            float v692_data = ir1[5];
            ir1[5] = (v692_data + (v664_data * v690_data));
            float v695_data = s0[111];
            float v697_data = ir1[6];
            ir1[6] = (v697_data + (v664_data * v695_data));
            float v700_data = s0[127];
            float v702_data = ir1[7];
            ir1[7] = (v702_data + (v664_data * v700_data));
            #pragma unroll
            for (int32_t v707_n0 = 0; v707_n0 < 1; ++v707_n0) {
              #pragma unroll
              for (int32_t v708_n1 = 0; v708_n1 < 8; ++v708_n1) {
                int32_t v709_a = v707_n0 + v708_n1;
                float v711_data = ir1[(v707_n0 + v708_n1)];
                int32_t v712_a = v707_n0 + v708_n1;
                r1[v712_a] = v711_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v715_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v716_i0 = 0; v716_i0 < 1; ++v716_i0) {
            int32_t v725_lead = v715_lead + (v716_i0 * 16);
            #pragma unroll
            for (int32_t v717_i1 = 0; v717_i1 < 8; ++v717_i1) {
              int32_t v718_a = v716_i0 + v717_i1;
              float v720_data = r1[(v716_i0 + v717_i1)];
              int32_t v727_a = v725_lead + (v717_i1 * 16);
              glb_m0[v727_a] = v720_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

