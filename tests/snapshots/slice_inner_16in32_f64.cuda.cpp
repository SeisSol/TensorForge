// === base name ===
kernel_3d37ccf0b0

// === header ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3d37ccf0b0, block.x * block.y * block.z, 2304 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_3d37ccf0b0, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3d37ccf0b0<<<grid,block,2304 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3d37ccf0b0(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v10_off = (v2_lead + (v3_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v4_i1 = 8; v4_i1 < 24; ++v4_i1) {
              int32_t v12_a = v10_off + (v4_i1 * 32);
              double v13_data;
              {
                v13_data = __ldcg(&glb_m1[v12_a]);
              }
              int32_t v15_a = v3_i0 + (v4_i1 - 8);
              r0[v15_a] = v13_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<8>(8), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          double r1[8]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 8)] [(0, 16)]
            double ir1[8]{};
            double v19_data = r0[0];
            double v20_data = s0[0];
            double v22_data = ir1[0];
            ir1[0] = (v22_data + (v19_data * v20_data));
            double v25_data = s0[16];
            double v27_data = ir1[1];
            ir1[1] = (v27_data + (v19_data * v25_data));
            double v30_data = s0[32];
            double v32_data = ir1[2];
            ir1[2] = (v32_data + (v19_data * v30_data));
            double v35_data = s0[48];
            double v37_data = ir1[3];
            ir1[3] = (v37_data + (v19_data * v35_data));
            double v40_data = s0[64];
            double v42_data = ir1[4];
            ir1[4] = (v42_data + (v19_data * v40_data));
            double v45_data = s0[80];
            double v47_data = ir1[5];
            ir1[5] = (v47_data + (v19_data * v45_data));
            double v50_data = s0[96];
            double v52_data = ir1[6];
            ir1[6] = (v52_data + (v19_data * v50_data));
            double v55_data = s0[112];
            double v57_data = ir1[7];
            ir1[7] = (v57_data + (v19_data * v55_data));
            double v62_data = r0[1];
            double v63_data = s0[1];
            double v65_data = ir1[0];
            ir1[0] = (v65_data + (v62_data * v63_data));
            double v68_data = s0[17];
            double v70_data = ir1[1];
            ir1[1] = (v70_data + (v62_data * v68_data));
            double v73_data = s0[33];
            double v75_data = ir1[2];
            ir1[2] = (v75_data + (v62_data * v73_data));
            double v78_data = s0[49];
            double v80_data = ir1[3];
            ir1[3] = (v80_data + (v62_data * v78_data));
            double v83_data = s0[65];
            double v85_data = ir1[4];
            ir1[4] = (v85_data + (v62_data * v83_data));
            double v88_data = s0[81];
            double v90_data = ir1[5];
            ir1[5] = (v90_data + (v62_data * v88_data));
            double v93_data = s0[97];
            double v95_data = ir1[6];
            ir1[6] = (v95_data + (v62_data * v93_data));
            double v98_data = s0[113];
            double v100_data = ir1[7];
            ir1[7] = (v100_data + (v62_data * v98_data));
            double v105_data = r0[2];
            double v106_data = s0[2];
            double v108_data = ir1[0];
            ir1[0] = (v108_data + (v105_data * v106_data));
            double v111_data = s0[18];
            double v113_data = ir1[1];
            ir1[1] = (v113_data + (v105_data * v111_data));
            double v116_data = s0[34];
            double v118_data = ir1[2];
            ir1[2] = (v118_data + (v105_data * v116_data));
            double v121_data = s0[50];
            double v123_data = ir1[3];
            ir1[3] = (v123_data + (v105_data * v121_data));
            double v126_data = s0[66];
            double v128_data = ir1[4];
            ir1[4] = (v128_data + (v105_data * v126_data));
            double v131_data = s0[82];
            double v133_data = ir1[5];
            ir1[5] = (v133_data + (v105_data * v131_data));
            double v136_data = s0[98];
            double v138_data = ir1[6];
            ir1[6] = (v138_data + (v105_data * v136_data));
            double v141_data = s0[114];
            double v143_data = ir1[7];
            ir1[7] = (v143_data + (v105_data * v141_data));
            double v148_data = r0[3];
            double v149_data = s0[3];
            double v151_data = ir1[0];
            ir1[0] = (v151_data + (v148_data * v149_data));
            double v154_data = s0[19];
            double v156_data = ir1[1];
            ir1[1] = (v156_data + (v148_data * v154_data));
            double v159_data = s0[35];
            double v161_data = ir1[2];
            ir1[2] = (v161_data + (v148_data * v159_data));
            double v164_data = s0[51];
            double v166_data = ir1[3];
            ir1[3] = (v166_data + (v148_data * v164_data));
            double v169_data = s0[67];
            double v171_data = ir1[4];
            ir1[4] = (v171_data + (v148_data * v169_data));
            double v174_data = s0[83];
            double v176_data = ir1[5];
            ir1[5] = (v176_data + (v148_data * v174_data));
            double v179_data = s0[99];
            double v181_data = ir1[6];
            ir1[6] = (v181_data + (v148_data * v179_data));
            double v184_data = s0[115];
            double v186_data = ir1[7];
            ir1[7] = (v186_data + (v148_data * v184_data));
            double v191_data = r0[4];
            double v192_data = s0[4];
            double v194_data = ir1[0];
            ir1[0] = (v194_data + (v191_data * v192_data));
            double v197_data = s0[20];
            double v199_data = ir1[1];
            ir1[1] = (v199_data + (v191_data * v197_data));
            double v202_data = s0[36];
            double v204_data = ir1[2];
            ir1[2] = (v204_data + (v191_data * v202_data));
            double v207_data = s0[52];
            double v209_data = ir1[3];
            ir1[3] = (v209_data + (v191_data * v207_data));
            double v212_data = s0[68];
            double v214_data = ir1[4];
            ir1[4] = (v214_data + (v191_data * v212_data));
            double v217_data = s0[84];
            double v219_data = ir1[5];
            ir1[5] = (v219_data + (v191_data * v217_data));
            double v222_data = s0[100];
            double v224_data = ir1[6];
            ir1[6] = (v224_data + (v191_data * v222_data));
            double v227_data = s0[116];
            double v229_data = ir1[7];
            ir1[7] = (v229_data + (v191_data * v227_data));
            double v234_data = r0[5];
            double v235_data = s0[5];
            double v237_data = ir1[0];
            ir1[0] = (v237_data + (v234_data * v235_data));
            double v240_data = s0[21];
            double v242_data = ir1[1];
            ir1[1] = (v242_data + (v234_data * v240_data));
            double v245_data = s0[37];
            double v247_data = ir1[2];
            ir1[2] = (v247_data + (v234_data * v245_data));
            double v250_data = s0[53];
            double v252_data = ir1[3];
            ir1[3] = (v252_data + (v234_data * v250_data));
            double v255_data = s0[69];
            double v257_data = ir1[4];
            ir1[4] = (v257_data + (v234_data * v255_data));
            double v260_data = s0[85];
            double v262_data = ir1[5];
            ir1[5] = (v262_data + (v234_data * v260_data));
            double v265_data = s0[101];
            double v267_data = ir1[6];
            ir1[6] = (v267_data + (v234_data * v265_data));
            double v270_data = s0[117];
            double v272_data = ir1[7];
            ir1[7] = (v272_data + (v234_data * v270_data));
            double v277_data = r0[6];
            double v278_data = s0[6];
            double v280_data = ir1[0];
            ir1[0] = (v280_data + (v277_data * v278_data));
            double v283_data = s0[22];
            double v285_data = ir1[1];
            ir1[1] = (v285_data + (v277_data * v283_data));
            double v288_data = s0[38];
            double v290_data = ir1[2];
            ir1[2] = (v290_data + (v277_data * v288_data));
            double v293_data = s0[54];
            double v295_data = ir1[3];
            ir1[3] = (v295_data + (v277_data * v293_data));
            double v298_data = s0[70];
            double v300_data = ir1[4];
            ir1[4] = (v300_data + (v277_data * v298_data));
            double v303_data = s0[86];
            double v305_data = ir1[5];
            ir1[5] = (v305_data + (v277_data * v303_data));
            double v308_data = s0[102];
            double v310_data = ir1[6];
            ir1[6] = (v310_data + (v277_data * v308_data));
            double v313_data = s0[118];
            double v315_data = ir1[7];
            ir1[7] = (v315_data + (v277_data * v313_data));
            double v320_data = r0[7];
            double v321_data = s0[7];
            double v323_data = ir1[0];
            ir1[0] = (v323_data + (v320_data * v321_data));
            double v326_data = s0[23];
            double v328_data = ir1[1];
            ir1[1] = (v328_data + (v320_data * v326_data));
            double v331_data = s0[39];
            double v333_data = ir1[2];
            ir1[2] = (v333_data + (v320_data * v331_data));
            double v336_data = s0[55];
            double v338_data = ir1[3];
            ir1[3] = (v338_data + (v320_data * v336_data));
            double v341_data = s0[71];
            double v343_data = ir1[4];
            ir1[4] = (v343_data + (v320_data * v341_data));
            double v346_data = s0[87];
            double v348_data = ir1[5];
            ir1[5] = (v348_data + (v320_data * v346_data));
            double v351_data = s0[103];
            double v353_data = ir1[6];
            ir1[6] = (v353_data + (v320_data * v351_data));
            double v356_data = s0[119];
            double v358_data = ir1[7];
            ir1[7] = (v358_data + (v320_data * v356_data));
            double v363_data = r0[8];
            double v364_data = s0[8];
            double v366_data = ir1[0];
            ir1[0] = (v366_data + (v363_data * v364_data));
            double v369_data = s0[24];
            double v371_data = ir1[1];
            ir1[1] = (v371_data + (v363_data * v369_data));
            double v374_data = s0[40];
            double v376_data = ir1[2];
            ir1[2] = (v376_data + (v363_data * v374_data));
            double v379_data = s0[56];
            double v381_data = ir1[3];
            ir1[3] = (v381_data + (v363_data * v379_data));
            double v384_data = s0[72];
            double v386_data = ir1[4];
            ir1[4] = (v386_data + (v363_data * v384_data));
            double v389_data = s0[88];
            double v391_data = ir1[5];
            ir1[5] = (v391_data + (v363_data * v389_data));
            double v394_data = s0[104];
            double v396_data = ir1[6];
            ir1[6] = (v396_data + (v363_data * v394_data));
            double v399_data = s0[120];
            double v401_data = ir1[7];
            ir1[7] = (v401_data + (v363_data * v399_data));
            double v406_data = r0[9];
            double v407_data = s0[9];
            double v409_data = ir1[0];
            ir1[0] = (v409_data + (v406_data * v407_data));
            double v412_data = s0[25];
            double v414_data = ir1[1];
            ir1[1] = (v414_data + (v406_data * v412_data));
            double v417_data = s0[41];
            double v419_data = ir1[2];
            ir1[2] = (v419_data + (v406_data * v417_data));
            double v422_data = s0[57];
            double v424_data = ir1[3];
            ir1[3] = (v424_data + (v406_data * v422_data));
            double v427_data = s0[73];
            double v429_data = ir1[4];
            ir1[4] = (v429_data + (v406_data * v427_data));
            double v432_data = s0[89];
            double v434_data = ir1[5];
            ir1[5] = (v434_data + (v406_data * v432_data));
            double v437_data = s0[105];
            double v439_data = ir1[6];
            ir1[6] = (v439_data + (v406_data * v437_data));
            double v442_data = s0[121];
            double v444_data = ir1[7];
            ir1[7] = (v444_data + (v406_data * v442_data));
            double v449_data = r0[10];
            double v450_data = s0[10];
            double v452_data = ir1[0];
            ir1[0] = (v452_data + (v449_data * v450_data));
            double v455_data = s0[26];
            double v457_data = ir1[1];
            ir1[1] = (v457_data + (v449_data * v455_data));
            double v460_data = s0[42];
            double v462_data = ir1[2];
            ir1[2] = (v462_data + (v449_data * v460_data));
            double v465_data = s0[58];
            double v467_data = ir1[3];
            ir1[3] = (v467_data + (v449_data * v465_data));
            double v470_data = s0[74];
            double v472_data = ir1[4];
            ir1[4] = (v472_data + (v449_data * v470_data));
            double v475_data = s0[90];
            double v477_data = ir1[5];
            ir1[5] = (v477_data + (v449_data * v475_data));
            double v480_data = s0[106];
            double v482_data = ir1[6];
            ir1[6] = (v482_data + (v449_data * v480_data));
            double v485_data = s0[122];
            double v487_data = ir1[7];
            ir1[7] = (v487_data + (v449_data * v485_data));
            double v492_data = r0[11];
            double v493_data = s0[11];
            double v495_data = ir1[0];
            ir1[0] = (v495_data + (v492_data * v493_data));
            double v498_data = s0[27];
            double v500_data = ir1[1];
            ir1[1] = (v500_data + (v492_data * v498_data));
            double v503_data = s0[43];
            double v505_data = ir1[2];
            ir1[2] = (v505_data + (v492_data * v503_data));
            double v508_data = s0[59];
            double v510_data = ir1[3];
            ir1[3] = (v510_data + (v492_data * v508_data));
            double v513_data = s0[75];
            double v515_data = ir1[4];
            ir1[4] = (v515_data + (v492_data * v513_data));
            double v518_data = s0[91];
            double v520_data = ir1[5];
            ir1[5] = (v520_data + (v492_data * v518_data));
            double v523_data = s0[107];
            double v525_data = ir1[6];
            ir1[6] = (v525_data + (v492_data * v523_data));
            double v528_data = s0[123];
            double v530_data = ir1[7];
            ir1[7] = (v530_data + (v492_data * v528_data));
            double v535_data = r0[12];
            double v536_data = s0[12];
            double v538_data = ir1[0];
            ir1[0] = (v538_data + (v535_data * v536_data));
            double v541_data = s0[28];
            double v543_data = ir1[1];
            ir1[1] = (v543_data + (v535_data * v541_data));
            double v546_data = s0[44];
            double v548_data = ir1[2];
            ir1[2] = (v548_data + (v535_data * v546_data));
            double v551_data = s0[60];
            double v553_data = ir1[3];
            ir1[3] = (v553_data + (v535_data * v551_data));
            double v556_data = s0[76];
            double v558_data = ir1[4];
            ir1[4] = (v558_data + (v535_data * v556_data));
            double v561_data = s0[92];
            double v563_data = ir1[5];
            ir1[5] = (v563_data + (v535_data * v561_data));
            double v566_data = s0[108];
            double v568_data = ir1[6];
            ir1[6] = (v568_data + (v535_data * v566_data));
            double v571_data = s0[124];
            double v573_data = ir1[7];
            ir1[7] = (v573_data + (v535_data * v571_data));
            double v578_data = r0[13];
            double v579_data = s0[13];
            double v581_data = ir1[0];
            ir1[0] = (v581_data + (v578_data * v579_data));
            double v584_data = s0[29];
            double v586_data = ir1[1];
            ir1[1] = (v586_data + (v578_data * v584_data));
            double v589_data = s0[45];
            double v591_data = ir1[2];
            ir1[2] = (v591_data + (v578_data * v589_data));
            double v594_data = s0[61];
            double v596_data = ir1[3];
            ir1[3] = (v596_data + (v578_data * v594_data));
            double v599_data = s0[77];
            double v601_data = ir1[4];
            ir1[4] = (v601_data + (v578_data * v599_data));
            double v604_data = s0[93];
            double v606_data = ir1[5];
            ir1[5] = (v606_data + (v578_data * v604_data));
            double v609_data = s0[109];
            double v611_data = ir1[6];
            ir1[6] = (v611_data + (v578_data * v609_data));
            double v614_data = s0[125];
            double v616_data = ir1[7];
            ir1[7] = (v616_data + (v578_data * v614_data));
            double v621_data = r0[14];
            double v622_data = s0[14];
            double v624_data = ir1[0];
            ir1[0] = (v624_data + (v621_data * v622_data));
            double v627_data = s0[30];
            double v629_data = ir1[1];
            ir1[1] = (v629_data + (v621_data * v627_data));
            double v632_data = s0[46];
            double v634_data = ir1[2];
            ir1[2] = (v634_data + (v621_data * v632_data));
            double v637_data = s0[62];
            double v639_data = ir1[3];
            ir1[3] = (v639_data + (v621_data * v637_data));
            double v642_data = s0[78];
            double v644_data = ir1[4];
            ir1[4] = (v644_data + (v621_data * v642_data));
            double v647_data = s0[94];
            double v649_data = ir1[5];
            ir1[5] = (v649_data + (v621_data * v647_data));
            double v652_data = s0[110];
            double v654_data = ir1[6];
            ir1[6] = (v654_data + (v621_data * v652_data));
            double v657_data = s0[126];
            double v659_data = ir1[7];
            ir1[7] = (v659_data + (v621_data * v657_data));
            double v664_data = r0[15];
            double v665_data = s0[15];
            double v667_data = ir1[0];
            ir1[0] = (v667_data + (v664_data * v665_data));
            double v670_data = s0[31];
            double v672_data = ir1[1];
            ir1[1] = (v672_data + (v664_data * v670_data));
            double v675_data = s0[47];
            double v677_data = ir1[2];
            ir1[2] = (v677_data + (v664_data * v675_data));
            double v680_data = s0[63];
            double v682_data = ir1[3];
            ir1[3] = (v682_data + (v664_data * v680_data));
            double v685_data = s0[79];
            double v687_data = ir1[4];
            ir1[4] = (v687_data + (v664_data * v685_data));
            double v690_data = s0[95];
            double v692_data = ir1[5];
            ir1[5] = (v692_data + (v664_data * v690_data));
            double v695_data = s0[111];
            double v697_data = ir1[6];
            ir1[6] = (v697_data + (v664_data * v695_data));
            double v700_data = s0[127];
            double v702_data = ir1[7];
            ir1[7] = (v702_data + (v664_data * v700_data));
            #pragma unroll
            for (int32_t v707_n0 = 0; v707_n0 < 1; ++v707_n0) {
              #pragma unroll
              for (int32_t v708_n1 = 0; v708_n1 < 8; ++v708_n1) {
                int32_t v709_a = v707_n0 + v708_n1;
                double v710_data = ir1[v709_a];
                int32_t v711_a = v707_n0 + v708_n1;
                r1[v711_a] = v710_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v714_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v715_i0 = 0; v715_i0 < 1; ++v715_i0) {
            int32_t v723_lead = v714_lead + (v715_i0 * 16);
            #pragma unroll
            for (int32_t v716_i1 = 0; v716_i1 < 8; ++v716_i1) {
              int32_t v717_a = v715_i0 + v716_i1;
              double v718_data = r1[v717_a];
              int32_t v725_a = v723_lead + (v716_i1 * 16);
              glb_m0[v725_a] = v718_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

