// === base name ===
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ead773dd51, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_ead773dd51, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ead773dd51<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 32×16(32×16) {0..32}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            int32_t v10_off = v2_lead + 4;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v12_a = v10_off + (v4_i1 * 32);
              float v13_data;
              {
                v13_data = __ldcg(&glb_m1[v12_a]);
              }
              int32_t v14_a = 0 + v4_i1;
              r0[v14_a] = v13_data;
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
            // [(0, 12), (0, 8)] [(0, 16)]
            float ir1[8]{};
            int32_t v17_lead = threadIdx.x % 16;
            if (v17_lead < 12) {
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
            }
            if (v17_lead < 12) {
              float v63_data = r0[1];
              float v64_data = s0[1];
              float v66_data = ir1[0];
              ir1[0] = (v66_data + (v63_data * v64_data));
              float v69_data = s0[17];
              float v71_data = ir1[1];
              ir1[1] = (v71_data + (v63_data * v69_data));
              float v74_data = s0[33];
              float v76_data = ir1[2];
              ir1[2] = (v76_data + (v63_data * v74_data));
              float v79_data = s0[49];
              float v81_data = ir1[3];
              ir1[3] = (v81_data + (v63_data * v79_data));
              float v84_data = s0[65];
              float v86_data = ir1[4];
              ir1[4] = (v86_data + (v63_data * v84_data));
              float v89_data = s0[81];
              float v91_data = ir1[5];
              ir1[5] = (v91_data + (v63_data * v89_data));
              float v94_data = s0[97];
              float v96_data = ir1[6];
              ir1[6] = (v96_data + (v63_data * v94_data));
              float v99_data = s0[113];
              float v101_data = ir1[7];
              ir1[7] = (v101_data + (v63_data * v99_data));
            }
            if (v17_lead < 12) {
              float v107_data = r0[2];
              float v108_data = s0[2];
              float v110_data = ir1[0];
              ir1[0] = (v110_data + (v107_data * v108_data));
              float v113_data = s0[18];
              float v115_data = ir1[1];
              ir1[1] = (v115_data + (v107_data * v113_data));
              float v118_data = s0[34];
              float v120_data = ir1[2];
              ir1[2] = (v120_data + (v107_data * v118_data));
              float v123_data = s0[50];
              float v125_data = ir1[3];
              ir1[3] = (v125_data + (v107_data * v123_data));
              float v128_data = s0[66];
              float v130_data = ir1[4];
              ir1[4] = (v130_data + (v107_data * v128_data));
              float v133_data = s0[82];
              float v135_data = ir1[5];
              ir1[5] = (v135_data + (v107_data * v133_data));
              float v138_data = s0[98];
              float v140_data = ir1[6];
              ir1[6] = (v140_data + (v107_data * v138_data));
              float v143_data = s0[114];
              float v145_data = ir1[7];
              ir1[7] = (v145_data + (v107_data * v143_data));
            }
            if (v17_lead < 12) {
              float v151_data = r0[3];
              float v152_data = s0[3];
              float v154_data = ir1[0];
              ir1[0] = (v154_data + (v151_data * v152_data));
              float v157_data = s0[19];
              float v159_data = ir1[1];
              ir1[1] = (v159_data + (v151_data * v157_data));
              float v162_data = s0[35];
              float v164_data = ir1[2];
              ir1[2] = (v164_data + (v151_data * v162_data));
              float v167_data = s0[51];
              float v169_data = ir1[3];
              ir1[3] = (v169_data + (v151_data * v167_data));
              float v172_data = s0[67];
              float v174_data = ir1[4];
              ir1[4] = (v174_data + (v151_data * v172_data));
              float v177_data = s0[83];
              float v179_data = ir1[5];
              ir1[5] = (v179_data + (v151_data * v177_data));
              float v182_data = s0[99];
              float v184_data = ir1[6];
              ir1[6] = (v184_data + (v151_data * v182_data));
              float v187_data = s0[115];
              float v189_data = ir1[7];
              ir1[7] = (v189_data + (v151_data * v187_data));
            }
            if (v17_lead < 12) {
              float v195_data = r0[4];
              float v196_data = s0[4];
              float v198_data = ir1[0];
              ir1[0] = (v198_data + (v195_data * v196_data));
              float v201_data = s0[20];
              float v203_data = ir1[1];
              ir1[1] = (v203_data + (v195_data * v201_data));
              float v206_data = s0[36];
              float v208_data = ir1[2];
              ir1[2] = (v208_data + (v195_data * v206_data));
              float v211_data = s0[52];
              float v213_data = ir1[3];
              ir1[3] = (v213_data + (v195_data * v211_data));
              float v216_data = s0[68];
              float v218_data = ir1[4];
              ir1[4] = (v218_data + (v195_data * v216_data));
              float v221_data = s0[84];
              float v223_data = ir1[5];
              ir1[5] = (v223_data + (v195_data * v221_data));
              float v226_data = s0[100];
              float v228_data = ir1[6];
              ir1[6] = (v228_data + (v195_data * v226_data));
              float v231_data = s0[116];
              float v233_data = ir1[7];
              ir1[7] = (v233_data + (v195_data * v231_data));
            }
            if (v17_lead < 12) {
              float v239_data = r0[5];
              float v240_data = s0[5];
              float v242_data = ir1[0];
              ir1[0] = (v242_data + (v239_data * v240_data));
              float v245_data = s0[21];
              float v247_data = ir1[1];
              ir1[1] = (v247_data + (v239_data * v245_data));
              float v250_data = s0[37];
              float v252_data = ir1[2];
              ir1[2] = (v252_data + (v239_data * v250_data));
              float v255_data = s0[53];
              float v257_data = ir1[3];
              ir1[3] = (v257_data + (v239_data * v255_data));
              float v260_data = s0[69];
              float v262_data = ir1[4];
              ir1[4] = (v262_data + (v239_data * v260_data));
              float v265_data = s0[85];
              float v267_data = ir1[5];
              ir1[5] = (v267_data + (v239_data * v265_data));
              float v270_data = s0[101];
              float v272_data = ir1[6];
              ir1[6] = (v272_data + (v239_data * v270_data));
              float v275_data = s0[117];
              float v277_data = ir1[7];
              ir1[7] = (v277_data + (v239_data * v275_data));
            }
            if (v17_lead < 12) {
              float v283_data = r0[6];
              float v284_data = s0[6];
              float v286_data = ir1[0];
              ir1[0] = (v286_data + (v283_data * v284_data));
              float v289_data = s0[22];
              float v291_data = ir1[1];
              ir1[1] = (v291_data + (v283_data * v289_data));
              float v294_data = s0[38];
              float v296_data = ir1[2];
              ir1[2] = (v296_data + (v283_data * v294_data));
              float v299_data = s0[54];
              float v301_data = ir1[3];
              ir1[3] = (v301_data + (v283_data * v299_data));
              float v304_data = s0[70];
              float v306_data = ir1[4];
              ir1[4] = (v306_data + (v283_data * v304_data));
              float v309_data = s0[86];
              float v311_data = ir1[5];
              ir1[5] = (v311_data + (v283_data * v309_data));
              float v314_data = s0[102];
              float v316_data = ir1[6];
              ir1[6] = (v316_data + (v283_data * v314_data));
              float v319_data = s0[118];
              float v321_data = ir1[7];
              ir1[7] = (v321_data + (v283_data * v319_data));
            }
            if (v17_lead < 12) {
              float v327_data = r0[7];
              float v328_data = s0[7];
              float v330_data = ir1[0];
              ir1[0] = (v330_data + (v327_data * v328_data));
              float v333_data = s0[23];
              float v335_data = ir1[1];
              ir1[1] = (v335_data + (v327_data * v333_data));
              float v338_data = s0[39];
              float v340_data = ir1[2];
              ir1[2] = (v340_data + (v327_data * v338_data));
              float v343_data = s0[55];
              float v345_data = ir1[3];
              ir1[3] = (v345_data + (v327_data * v343_data));
              float v348_data = s0[71];
              float v350_data = ir1[4];
              ir1[4] = (v350_data + (v327_data * v348_data));
              float v353_data = s0[87];
              float v355_data = ir1[5];
              ir1[5] = (v355_data + (v327_data * v353_data));
              float v358_data = s0[103];
              float v360_data = ir1[6];
              ir1[6] = (v360_data + (v327_data * v358_data));
              float v363_data = s0[119];
              float v365_data = ir1[7];
              ir1[7] = (v365_data + (v327_data * v363_data));
            }
            if (v17_lead < 12) {
              float v371_data = r0[8];
              float v372_data = s0[8];
              float v374_data = ir1[0];
              ir1[0] = (v374_data + (v371_data * v372_data));
              float v377_data = s0[24];
              float v379_data = ir1[1];
              ir1[1] = (v379_data + (v371_data * v377_data));
              float v382_data = s0[40];
              float v384_data = ir1[2];
              ir1[2] = (v384_data + (v371_data * v382_data));
              float v387_data = s0[56];
              float v389_data = ir1[3];
              ir1[3] = (v389_data + (v371_data * v387_data));
              float v392_data = s0[72];
              float v394_data = ir1[4];
              ir1[4] = (v394_data + (v371_data * v392_data));
              float v397_data = s0[88];
              float v399_data = ir1[5];
              ir1[5] = (v399_data + (v371_data * v397_data));
              float v402_data = s0[104];
              float v404_data = ir1[6];
              ir1[6] = (v404_data + (v371_data * v402_data));
              float v407_data = s0[120];
              float v409_data = ir1[7];
              ir1[7] = (v409_data + (v371_data * v407_data));
            }
            if (v17_lead < 12) {
              float v415_data = r0[9];
              float v416_data = s0[9];
              float v418_data = ir1[0];
              ir1[0] = (v418_data + (v415_data * v416_data));
              float v421_data = s0[25];
              float v423_data = ir1[1];
              ir1[1] = (v423_data + (v415_data * v421_data));
              float v426_data = s0[41];
              float v428_data = ir1[2];
              ir1[2] = (v428_data + (v415_data * v426_data));
              float v431_data = s0[57];
              float v433_data = ir1[3];
              ir1[3] = (v433_data + (v415_data * v431_data));
              float v436_data = s0[73];
              float v438_data = ir1[4];
              ir1[4] = (v438_data + (v415_data * v436_data));
              float v441_data = s0[89];
              float v443_data = ir1[5];
              ir1[5] = (v443_data + (v415_data * v441_data));
              float v446_data = s0[105];
              float v448_data = ir1[6];
              ir1[6] = (v448_data + (v415_data * v446_data));
              float v451_data = s0[121];
              float v453_data = ir1[7];
              ir1[7] = (v453_data + (v415_data * v451_data));
            }
            if (v17_lead < 12) {
              float v459_data = r0[10];
              float v460_data = s0[10];
              float v462_data = ir1[0];
              ir1[0] = (v462_data + (v459_data * v460_data));
              float v465_data = s0[26];
              float v467_data = ir1[1];
              ir1[1] = (v467_data + (v459_data * v465_data));
              float v470_data = s0[42];
              float v472_data = ir1[2];
              ir1[2] = (v472_data + (v459_data * v470_data));
              float v475_data = s0[58];
              float v477_data = ir1[3];
              ir1[3] = (v477_data + (v459_data * v475_data));
              float v480_data = s0[74];
              float v482_data = ir1[4];
              ir1[4] = (v482_data + (v459_data * v480_data));
              float v485_data = s0[90];
              float v487_data = ir1[5];
              ir1[5] = (v487_data + (v459_data * v485_data));
              float v490_data = s0[106];
              float v492_data = ir1[6];
              ir1[6] = (v492_data + (v459_data * v490_data));
              float v495_data = s0[122];
              float v497_data = ir1[7];
              ir1[7] = (v497_data + (v459_data * v495_data));
            }
            if (v17_lead < 12) {
              float v503_data = r0[11];
              float v504_data = s0[11];
              float v506_data = ir1[0];
              ir1[0] = (v506_data + (v503_data * v504_data));
              float v509_data = s0[27];
              float v511_data = ir1[1];
              ir1[1] = (v511_data + (v503_data * v509_data));
              float v514_data = s0[43];
              float v516_data = ir1[2];
              ir1[2] = (v516_data + (v503_data * v514_data));
              float v519_data = s0[59];
              float v521_data = ir1[3];
              ir1[3] = (v521_data + (v503_data * v519_data));
              float v524_data = s0[75];
              float v526_data = ir1[4];
              ir1[4] = (v526_data + (v503_data * v524_data));
              float v529_data = s0[91];
              float v531_data = ir1[5];
              ir1[5] = (v531_data + (v503_data * v529_data));
              float v534_data = s0[107];
              float v536_data = ir1[6];
              ir1[6] = (v536_data + (v503_data * v534_data));
              float v539_data = s0[123];
              float v541_data = ir1[7];
              ir1[7] = (v541_data + (v503_data * v539_data));
            }
            if (v17_lead < 12) {
              float v547_data = r0[12];
              float v548_data = s0[12];
              float v550_data = ir1[0];
              ir1[0] = (v550_data + (v547_data * v548_data));
              float v553_data = s0[28];
              float v555_data = ir1[1];
              ir1[1] = (v555_data + (v547_data * v553_data));
              float v558_data = s0[44];
              float v560_data = ir1[2];
              ir1[2] = (v560_data + (v547_data * v558_data));
              float v563_data = s0[60];
              float v565_data = ir1[3];
              ir1[3] = (v565_data + (v547_data * v563_data));
              float v568_data = s0[76];
              float v570_data = ir1[4];
              ir1[4] = (v570_data + (v547_data * v568_data));
              float v573_data = s0[92];
              float v575_data = ir1[5];
              ir1[5] = (v575_data + (v547_data * v573_data));
              float v578_data = s0[108];
              float v580_data = ir1[6];
              ir1[6] = (v580_data + (v547_data * v578_data));
              float v583_data = s0[124];
              float v585_data = ir1[7];
              ir1[7] = (v585_data + (v547_data * v583_data));
            }
            if (v17_lead < 12) {
              float v591_data = r0[13];
              float v592_data = s0[13];
              float v594_data = ir1[0];
              ir1[0] = (v594_data + (v591_data * v592_data));
              float v597_data = s0[29];
              float v599_data = ir1[1];
              ir1[1] = (v599_data + (v591_data * v597_data));
              float v602_data = s0[45];
              float v604_data = ir1[2];
              ir1[2] = (v604_data + (v591_data * v602_data));
              float v607_data = s0[61];
              float v609_data = ir1[3];
              ir1[3] = (v609_data + (v591_data * v607_data));
              float v612_data = s0[77];
              float v614_data = ir1[4];
              ir1[4] = (v614_data + (v591_data * v612_data));
              float v617_data = s0[93];
              float v619_data = ir1[5];
              ir1[5] = (v619_data + (v591_data * v617_data));
              float v622_data = s0[109];
              float v624_data = ir1[6];
              ir1[6] = (v624_data + (v591_data * v622_data));
              float v627_data = s0[125];
              float v629_data = ir1[7];
              ir1[7] = (v629_data + (v591_data * v627_data));
            }
            if (v17_lead < 12) {
              float v635_data = r0[14];
              float v636_data = s0[14];
              float v638_data = ir1[0];
              ir1[0] = (v638_data + (v635_data * v636_data));
              float v641_data = s0[30];
              float v643_data = ir1[1];
              ir1[1] = (v643_data + (v635_data * v641_data));
              float v646_data = s0[46];
              float v648_data = ir1[2];
              ir1[2] = (v648_data + (v635_data * v646_data));
              float v651_data = s0[62];
              float v653_data = ir1[3];
              ir1[3] = (v653_data + (v635_data * v651_data));
              float v656_data = s0[78];
              float v658_data = ir1[4];
              ir1[4] = (v658_data + (v635_data * v656_data));
              float v661_data = s0[94];
              float v663_data = ir1[5];
              ir1[5] = (v663_data + (v635_data * v661_data));
              float v666_data = s0[110];
              float v668_data = ir1[6];
              ir1[6] = (v668_data + (v635_data * v666_data));
              float v671_data = s0[126];
              float v673_data = ir1[7];
              ir1[7] = (v673_data + (v635_data * v671_data));
            }
            if (v17_lead < 12) {
              float v679_data = r0[15];
              float v680_data = s0[15];
              float v682_data = ir1[0];
              ir1[0] = (v682_data + (v679_data * v680_data));
              float v685_data = s0[31];
              float v687_data = ir1[1];
              ir1[1] = (v687_data + (v679_data * v685_data));
              float v690_data = s0[47];
              float v692_data = ir1[2];
              ir1[2] = (v692_data + (v679_data * v690_data));
              float v695_data = s0[63];
              float v697_data = ir1[3];
              ir1[3] = (v697_data + (v679_data * v695_data));
              float v700_data = s0[79];
              float v702_data = ir1[4];
              ir1[4] = (v702_data + (v679_data * v700_data));
              float v705_data = s0[95];
              float v707_data = ir1[5];
              ir1[5] = (v707_data + (v679_data * v705_data));
              float v710_data = s0[111];
              float v712_data = ir1[6];
              ir1[6] = (v712_data + (v679_data * v710_data));
              float v715_data = s0[127];
              float v717_data = ir1[7];
              ir1[7] = (v717_data + (v679_data * v715_data));
            }
            if (v17_lead < 12) {
              #pragma unroll
              for (int32_t v723_n1 = 0; v723_n1 < 8; ++v723_n1) {
                int32_t v724_a = 0 + v723_n1;
                float v726_data = ir1[v723_n1];
                int32_t v727_a = 0 + v723_n1;
                r1[v727_a] = v726_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          int32_t v730_lead = threadIdx.x % 16;
          if (v730_lead < 12) {
            #pragma unroll
            for (int32_t v732_i1 = 0; v732_i1 < 8; ++v732_i1) {
              int32_t v733_a = 0 + v732_i1;
              float v735_data = r1[v732_i1];
              int32_t v742_a = v730_lead + (v732_i1 * 12);
              glb_m0[v742_a] = v735_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

