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
            int32_t v18_off = v2_lead + 4;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v11_a = v4_i1 * 32;
              int32_t v12_a = v10_off + v11_a;
              float v21_data = __ldcg(&glb_m1[(v18_off + v11_a)]);
              int32_t v22_a = 0 + v4_i1;
              r0[v22_a] = v21_data;
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
            if (v2_lead < 12) {
              float v27_data = r0[0];
              float v28_data = s0[0];
              float v30_data = ir1[0];
              ir1[0] = (v30_data + (v27_data * v28_data));
              float v33_data = s0[16];
              float v35_data = ir1[1];
              ir1[1] = (v35_data + (v27_data * v33_data));
              float v38_data = s0[32];
              float v40_data = ir1[2];
              ir1[2] = (v40_data + (v27_data * v38_data));
              float v43_data = s0[48];
              float v45_data = ir1[3];
              ir1[3] = (v45_data + (v27_data * v43_data));
              float v48_data = s0[64];
              float v50_data = ir1[4];
              ir1[4] = (v50_data + (v27_data * v48_data));
              float v53_data = s0[80];
              float v55_data = ir1[5];
              ir1[5] = (v55_data + (v27_data * v53_data));
              float v58_data = s0[96];
              float v60_data = ir1[6];
              ir1[6] = (v60_data + (v27_data * v58_data));
              float v63_data = s0[112];
              float v65_data = ir1[7];
              ir1[7] = (v65_data + (v27_data * v63_data));
            }
            if (v2_lead < 12) {
              float v71_data = r0[1];
              float v72_data = s0[1];
              float v74_data = ir1[0];
              ir1[0] = (v74_data + (v71_data * v72_data));
              float v77_data = s0[17];
              float v79_data = ir1[1];
              ir1[1] = (v79_data + (v71_data * v77_data));
              float v82_data = s0[33];
              float v84_data = ir1[2];
              ir1[2] = (v84_data + (v71_data * v82_data));
              float v87_data = s0[49];
              float v89_data = ir1[3];
              ir1[3] = (v89_data + (v71_data * v87_data));
              float v92_data = s0[65];
              float v94_data = ir1[4];
              ir1[4] = (v94_data + (v71_data * v92_data));
              float v97_data = s0[81];
              float v99_data = ir1[5];
              ir1[5] = (v99_data + (v71_data * v97_data));
              float v102_data = s0[97];
              float v104_data = ir1[6];
              ir1[6] = (v104_data + (v71_data * v102_data));
              float v107_data = s0[113];
              float v109_data = ir1[7];
              ir1[7] = (v109_data + (v71_data * v107_data));
            }
            if (v2_lead < 12) {
              float v115_data = r0[2];
              float v116_data = s0[2];
              float v118_data = ir1[0];
              ir1[0] = (v118_data + (v115_data * v116_data));
              float v121_data = s0[18];
              float v123_data = ir1[1];
              ir1[1] = (v123_data + (v115_data * v121_data));
              float v126_data = s0[34];
              float v128_data = ir1[2];
              ir1[2] = (v128_data + (v115_data * v126_data));
              float v131_data = s0[50];
              float v133_data = ir1[3];
              ir1[3] = (v133_data + (v115_data * v131_data));
              float v136_data = s0[66];
              float v138_data = ir1[4];
              ir1[4] = (v138_data + (v115_data * v136_data));
              float v141_data = s0[82];
              float v143_data = ir1[5];
              ir1[5] = (v143_data + (v115_data * v141_data));
              float v146_data = s0[98];
              float v148_data = ir1[6];
              ir1[6] = (v148_data + (v115_data * v146_data));
              float v151_data = s0[114];
              float v153_data = ir1[7];
              ir1[7] = (v153_data + (v115_data * v151_data));
            }
            if (v2_lead < 12) {
              float v159_data = r0[3];
              float v160_data = s0[3];
              float v162_data = ir1[0];
              ir1[0] = (v162_data + (v159_data * v160_data));
              float v165_data = s0[19];
              float v167_data = ir1[1];
              ir1[1] = (v167_data + (v159_data * v165_data));
              float v170_data = s0[35];
              float v172_data = ir1[2];
              ir1[2] = (v172_data + (v159_data * v170_data));
              float v175_data = s0[51];
              float v177_data = ir1[3];
              ir1[3] = (v177_data + (v159_data * v175_data));
              float v180_data = s0[67];
              float v182_data = ir1[4];
              ir1[4] = (v182_data + (v159_data * v180_data));
              float v185_data = s0[83];
              float v187_data = ir1[5];
              ir1[5] = (v187_data + (v159_data * v185_data));
              float v190_data = s0[99];
              float v192_data = ir1[6];
              ir1[6] = (v192_data + (v159_data * v190_data));
              float v195_data = s0[115];
              float v197_data = ir1[7];
              ir1[7] = (v197_data + (v159_data * v195_data));
            }
            if (v2_lead < 12) {
              float v203_data = r0[4];
              float v204_data = s0[4];
              float v206_data = ir1[0];
              ir1[0] = (v206_data + (v203_data * v204_data));
              float v209_data = s0[20];
              float v211_data = ir1[1];
              ir1[1] = (v211_data + (v203_data * v209_data));
              float v214_data = s0[36];
              float v216_data = ir1[2];
              ir1[2] = (v216_data + (v203_data * v214_data));
              float v219_data = s0[52];
              float v221_data = ir1[3];
              ir1[3] = (v221_data + (v203_data * v219_data));
              float v224_data = s0[68];
              float v226_data = ir1[4];
              ir1[4] = (v226_data + (v203_data * v224_data));
              float v229_data = s0[84];
              float v231_data = ir1[5];
              ir1[5] = (v231_data + (v203_data * v229_data));
              float v234_data = s0[100];
              float v236_data = ir1[6];
              ir1[6] = (v236_data + (v203_data * v234_data));
              float v239_data = s0[116];
              float v241_data = ir1[7];
              ir1[7] = (v241_data + (v203_data * v239_data));
            }
            if (v2_lead < 12) {
              float v247_data = r0[5];
              float v248_data = s0[5];
              float v250_data = ir1[0];
              ir1[0] = (v250_data + (v247_data * v248_data));
              float v253_data = s0[21];
              float v255_data = ir1[1];
              ir1[1] = (v255_data + (v247_data * v253_data));
              float v258_data = s0[37];
              float v260_data = ir1[2];
              ir1[2] = (v260_data + (v247_data * v258_data));
              float v263_data = s0[53];
              float v265_data = ir1[3];
              ir1[3] = (v265_data + (v247_data * v263_data));
              float v268_data = s0[69];
              float v270_data = ir1[4];
              ir1[4] = (v270_data + (v247_data * v268_data));
              float v273_data = s0[85];
              float v275_data = ir1[5];
              ir1[5] = (v275_data + (v247_data * v273_data));
              float v278_data = s0[101];
              float v280_data = ir1[6];
              ir1[6] = (v280_data + (v247_data * v278_data));
              float v283_data = s0[117];
              float v285_data = ir1[7];
              ir1[7] = (v285_data + (v247_data * v283_data));
            }
            if (v2_lead < 12) {
              float v291_data = r0[6];
              float v292_data = s0[6];
              float v294_data = ir1[0];
              ir1[0] = (v294_data + (v291_data * v292_data));
              float v297_data = s0[22];
              float v299_data = ir1[1];
              ir1[1] = (v299_data + (v291_data * v297_data));
              float v302_data = s0[38];
              float v304_data = ir1[2];
              ir1[2] = (v304_data + (v291_data * v302_data));
              float v307_data = s0[54];
              float v309_data = ir1[3];
              ir1[3] = (v309_data + (v291_data * v307_data));
              float v312_data = s0[70];
              float v314_data = ir1[4];
              ir1[4] = (v314_data + (v291_data * v312_data));
              float v317_data = s0[86];
              float v319_data = ir1[5];
              ir1[5] = (v319_data + (v291_data * v317_data));
              float v322_data = s0[102];
              float v324_data = ir1[6];
              ir1[6] = (v324_data + (v291_data * v322_data));
              float v327_data = s0[118];
              float v329_data = ir1[7];
              ir1[7] = (v329_data + (v291_data * v327_data));
            }
            if (v2_lead < 12) {
              float v335_data = r0[7];
              float v336_data = s0[7];
              float v338_data = ir1[0];
              ir1[0] = (v338_data + (v335_data * v336_data));
              float v341_data = s0[23];
              float v343_data = ir1[1];
              ir1[1] = (v343_data + (v335_data * v341_data));
              float v346_data = s0[39];
              float v348_data = ir1[2];
              ir1[2] = (v348_data + (v335_data * v346_data));
              float v351_data = s0[55];
              float v353_data = ir1[3];
              ir1[3] = (v353_data + (v335_data * v351_data));
              float v356_data = s0[71];
              float v358_data = ir1[4];
              ir1[4] = (v358_data + (v335_data * v356_data));
              float v361_data = s0[87];
              float v363_data = ir1[5];
              ir1[5] = (v363_data + (v335_data * v361_data));
              float v366_data = s0[103];
              float v368_data = ir1[6];
              ir1[6] = (v368_data + (v335_data * v366_data));
              float v371_data = s0[119];
              float v373_data = ir1[7];
              ir1[7] = (v373_data + (v335_data * v371_data));
            }
            if (v2_lead < 12) {
              float v379_data = r0[8];
              float v380_data = s0[8];
              float v382_data = ir1[0];
              ir1[0] = (v382_data + (v379_data * v380_data));
              float v385_data = s0[24];
              float v387_data = ir1[1];
              ir1[1] = (v387_data + (v379_data * v385_data));
              float v390_data = s0[40];
              float v392_data = ir1[2];
              ir1[2] = (v392_data + (v379_data * v390_data));
              float v395_data = s0[56];
              float v397_data = ir1[3];
              ir1[3] = (v397_data + (v379_data * v395_data));
              float v400_data = s0[72];
              float v402_data = ir1[4];
              ir1[4] = (v402_data + (v379_data * v400_data));
              float v405_data = s0[88];
              float v407_data = ir1[5];
              ir1[5] = (v407_data + (v379_data * v405_data));
              float v410_data = s0[104];
              float v412_data = ir1[6];
              ir1[6] = (v412_data + (v379_data * v410_data));
              float v415_data = s0[120];
              float v417_data = ir1[7];
              ir1[7] = (v417_data + (v379_data * v415_data));
            }
            if (v2_lead < 12) {
              float v423_data = r0[9];
              float v424_data = s0[9];
              float v426_data = ir1[0];
              ir1[0] = (v426_data + (v423_data * v424_data));
              float v429_data = s0[25];
              float v431_data = ir1[1];
              ir1[1] = (v431_data + (v423_data * v429_data));
              float v434_data = s0[41];
              float v436_data = ir1[2];
              ir1[2] = (v436_data + (v423_data * v434_data));
              float v439_data = s0[57];
              float v441_data = ir1[3];
              ir1[3] = (v441_data + (v423_data * v439_data));
              float v444_data = s0[73];
              float v446_data = ir1[4];
              ir1[4] = (v446_data + (v423_data * v444_data));
              float v449_data = s0[89];
              float v451_data = ir1[5];
              ir1[5] = (v451_data + (v423_data * v449_data));
              float v454_data = s0[105];
              float v456_data = ir1[6];
              ir1[6] = (v456_data + (v423_data * v454_data));
              float v459_data = s0[121];
              float v461_data = ir1[7];
              ir1[7] = (v461_data + (v423_data * v459_data));
            }
            if (v2_lead < 12) {
              float v467_data = r0[10];
              float v468_data = s0[10];
              float v470_data = ir1[0];
              ir1[0] = (v470_data + (v467_data * v468_data));
              float v473_data = s0[26];
              float v475_data = ir1[1];
              ir1[1] = (v475_data + (v467_data * v473_data));
              float v478_data = s0[42];
              float v480_data = ir1[2];
              ir1[2] = (v480_data + (v467_data * v478_data));
              float v483_data = s0[58];
              float v485_data = ir1[3];
              ir1[3] = (v485_data + (v467_data * v483_data));
              float v488_data = s0[74];
              float v490_data = ir1[4];
              ir1[4] = (v490_data + (v467_data * v488_data));
              float v493_data = s0[90];
              float v495_data = ir1[5];
              ir1[5] = (v495_data + (v467_data * v493_data));
              float v498_data = s0[106];
              float v500_data = ir1[6];
              ir1[6] = (v500_data + (v467_data * v498_data));
              float v503_data = s0[122];
              float v505_data = ir1[7];
              ir1[7] = (v505_data + (v467_data * v503_data));
            }
            if (v2_lead < 12) {
              float v511_data = r0[11];
              float v512_data = s0[11];
              float v514_data = ir1[0];
              ir1[0] = (v514_data + (v511_data * v512_data));
              float v517_data = s0[27];
              float v519_data = ir1[1];
              ir1[1] = (v519_data + (v511_data * v517_data));
              float v522_data = s0[43];
              float v524_data = ir1[2];
              ir1[2] = (v524_data + (v511_data * v522_data));
              float v527_data = s0[59];
              float v529_data = ir1[3];
              ir1[3] = (v529_data + (v511_data * v527_data));
              float v532_data = s0[75];
              float v534_data = ir1[4];
              ir1[4] = (v534_data + (v511_data * v532_data));
              float v537_data = s0[91];
              float v539_data = ir1[5];
              ir1[5] = (v539_data + (v511_data * v537_data));
              float v542_data = s0[107];
              float v544_data = ir1[6];
              ir1[6] = (v544_data + (v511_data * v542_data));
              float v547_data = s0[123];
              float v549_data = ir1[7];
              ir1[7] = (v549_data + (v511_data * v547_data));
            }
            if (v2_lead < 12) {
              float v555_data = r0[12];
              float v556_data = s0[12];
              float v558_data = ir1[0];
              ir1[0] = (v558_data + (v555_data * v556_data));
              float v561_data = s0[28];
              float v563_data = ir1[1];
              ir1[1] = (v563_data + (v555_data * v561_data));
              float v566_data = s0[44];
              float v568_data = ir1[2];
              ir1[2] = (v568_data + (v555_data * v566_data));
              float v571_data = s0[60];
              float v573_data = ir1[3];
              ir1[3] = (v573_data + (v555_data * v571_data));
              float v576_data = s0[76];
              float v578_data = ir1[4];
              ir1[4] = (v578_data + (v555_data * v576_data));
              float v581_data = s0[92];
              float v583_data = ir1[5];
              ir1[5] = (v583_data + (v555_data * v581_data));
              float v586_data = s0[108];
              float v588_data = ir1[6];
              ir1[6] = (v588_data + (v555_data * v586_data));
              float v591_data = s0[124];
              float v593_data = ir1[7];
              ir1[7] = (v593_data + (v555_data * v591_data));
            }
            if (v2_lead < 12) {
              float v599_data = r0[13];
              float v600_data = s0[13];
              float v602_data = ir1[0];
              ir1[0] = (v602_data + (v599_data * v600_data));
              float v605_data = s0[29];
              float v607_data = ir1[1];
              ir1[1] = (v607_data + (v599_data * v605_data));
              float v610_data = s0[45];
              float v612_data = ir1[2];
              ir1[2] = (v612_data + (v599_data * v610_data));
              float v615_data = s0[61];
              float v617_data = ir1[3];
              ir1[3] = (v617_data + (v599_data * v615_data));
              float v620_data = s0[77];
              float v622_data = ir1[4];
              ir1[4] = (v622_data + (v599_data * v620_data));
              float v625_data = s0[93];
              float v627_data = ir1[5];
              ir1[5] = (v627_data + (v599_data * v625_data));
              float v630_data = s0[109];
              float v632_data = ir1[6];
              ir1[6] = (v632_data + (v599_data * v630_data));
              float v635_data = s0[125];
              float v637_data = ir1[7];
              ir1[7] = (v637_data + (v599_data * v635_data));
            }
            if (v2_lead < 12) {
              float v643_data = r0[14];
              float v644_data = s0[14];
              float v646_data = ir1[0];
              ir1[0] = (v646_data + (v643_data * v644_data));
              float v649_data = s0[30];
              float v651_data = ir1[1];
              ir1[1] = (v651_data + (v643_data * v649_data));
              float v654_data = s0[46];
              float v656_data = ir1[2];
              ir1[2] = (v656_data + (v643_data * v654_data));
              float v659_data = s0[62];
              float v661_data = ir1[3];
              ir1[3] = (v661_data + (v643_data * v659_data));
              float v664_data = s0[78];
              float v666_data = ir1[4];
              ir1[4] = (v666_data + (v643_data * v664_data));
              float v669_data = s0[94];
              float v671_data = ir1[5];
              ir1[5] = (v671_data + (v643_data * v669_data));
              float v674_data = s0[110];
              float v676_data = ir1[6];
              ir1[6] = (v676_data + (v643_data * v674_data));
              float v679_data = s0[126];
              float v681_data = ir1[7];
              ir1[7] = (v681_data + (v643_data * v679_data));
            }
            if (v2_lead < 12) {
              float v687_data = r0[15];
              float v688_data = s0[15];
              float v690_data = ir1[0];
              ir1[0] = (v690_data + (v687_data * v688_data));
              float v693_data = s0[31];
              float v695_data = ir1[1];
              ir1[1] = (v695_data + (v687_data * v693_data));
              float v698_data = s0[47];
              float v700_data = ir1[2];
              ir1[2] = (v700_data + (v687_data * v698_data));
              float v703_data = s0[63];
              float v705_data = ir1[3];
              ir1[3] = (v705_data + (v687_data * v703_data));
              float v708_data = s0[79];
              float v710_data = ir1[4];
              ir1[4] = (v710_data + (v687_data * v708_data));
              float v713_data = s0[95];
              float v715_data = ir1[5];
              ir1[5] = (v715_data + (v687_data * v713_data));
              float v718_data = s0[111];
              float v720_data = ir1[6];
              ir1[6] = (v720_data + (v687_data * v718_data));
              float v723_data = s0[127];
              float v725_data = ir1[7];
              ir1[7] = (v725_data + (v687_data * v723_data));
            }
            if (v2_lead < 12) {
              #pragma unroll
              for (int32_t v731_n1 = 0; v731_n1 < 8; ++v731_n1) {
                int32_t v732_a = 0 + v731_n1;
                float v734_data = ir1[v731_n1];
                int32_t v735_a = 0 + v731_n1;
                r1[v731_n1] = v734_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v741_i1 = 0; v741_i1 < 8; ++v741_i1) {
              int32_t v742_a = 0 + v741_i1;
              float v744_data = r1[v741_i1];
              int32_t v751_a = v2_lead + (v741_i1 * 12);
              glb_m0[v751_a] = v744_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

