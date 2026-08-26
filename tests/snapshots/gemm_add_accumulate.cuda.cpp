// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5e7da3148f, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_5e7da3148f, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_5e7da3148f<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×16(12×16) {0..12}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
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
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
              int32_t v33_a = v27_i1 * 12;
              int32_t v34_a = v3_lead + v33_a;
              float v42_data = glb_m0[(v3_lead + v33_a)];
              int32_t v43_a = 0 + v27_i1;
              r1[v43_a] = v42_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          // wait(r1 = load{g>r}(glb_m0););
          float r2[8]{};
          __syncwarp();
          {
            // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 16)]
            float ir2[8]{};
            if (v3_lead < 12) {
              float v49_data = r0[0];
              float v50_data = s0[0];
              float v52_data = ir2[0];
              ir2[0] = (v52_data + (v49_data * v50_data));
              float v55_data = s0[16];
              float v57_data = ir2[1];
              ir2[1] = (v57_data + (v49_data * v55_data));
              float v60_data = s0[32];
              float v62_data = ir2[2];
              ir2[2] = (v62_data + (v49_data * v60_data));
              float v65_data = s0[48];
              float v67_data = ir2[3];
              ir2[3] = (v67_data + (v49_data * v65_data));
              float v70_data = s0[64];
              float v72_data = ir2[4];
              ir2[4] = (v72_data + (v49_data * v70_data));
              float v75_data = s0[80];
              float v77_data = ir2[5];
              ir2[5] = (v77_data + (v49_data * v75_data));
              float v80_data = s0[96];
              float v82_data = ir2[6];
              ir2[6] = (v82_data + (v49_data * v80_data));
              float v85_data = s0[112];
              float v87_data = ir2[7];
              ir2[7] = (v87_data + (v49_data * v85_data));
            }
            if (v3_lead < 12) {
              float v93_data = r0[1];
              float v94_data = s0[1];
              float v96_data = ir2[0];
              ir2[0] = (v96_data + (v93_data * v94_data));
              float v99_data = s0[17];
              float v101_data = ir2[1];
              ir2[1] = (v101_data + (v93_data * v99_data));
              float v104_data = s0[33];
              float v106_data = ir2[2];
              ir2[2] = (v106_data + (v93_data * v104_data));
              float v109_data = s0[49];
              float v111_data = ir2[3];
              ir2[3] = (v111_data + (v93_data * v109_data));
              float v114_data = s0[65];
              float v116_data = ir2[4];
              ir2[4] = (v116_data + (v93_data * v114_data));
              float v119_data = s0[81];
              float v121_data = ir2[5];
              ir2[5] = (v121_data + (v93_data * v119_data));
              float v124_data = s0[97];
              float v126_data = ir2[6];
              ir2[6] = (v126_data + (v93_data * v124_data));
              float v129_data = s0[113];
              float v131_data = ir2[7];
              ir2[7] = (v131_data + (v93_data * v129_data));
            }
            if (v3_lead < 12) {
              float v137_data = r0[2];
              float v138_data = s0[2];
              float v140_data = ir2[0];
              ir2[0] = (v140_data + (v137_data * v138_data));
              float v143_data = s0[18];
              float v145_data = ir2[1];
              ir2[1] = (v145_data + (v137_data * v143_data));
              float v148_data = s0[34];
              float v150_data = ir2[2];
              ir2[2] = (v150_data + (v137_data * v148_data));
              float v153_data = s0[50];
              float v155_data = ir2[3];
              ir2[3] = (v155_data + (v137_data * v153_data));
              float v158_data = s0[66];
              float v160_data = ir2[4];
              ir2[4] = (v160_data + (v137_data * v158_data));
              float v163_data = s0[82];
              float v165_data = ir2[5];
              ir2[5] = (v165_data + (v137_data * v163_data));
              float v168_data = s0[98];
              float v170_data = ir2[6];
              ir2[6] = (v170_data + (v137_data * v168_data));
              float v173_data = s0[114];
              float v175_data = ir2[7];
              ir2[7] = (v175_data + (v137_data * v173_data));
            }
            if (v3_lead < 12) {
              float v181_data = r0[3];
              float v182_data = s0[3];
              float v184_data = ir2[0];
              ir2[0] = (v184_data + (v181_data * v182_data));
              float v187_data = s0[19];
              float v189_data = ir2[1];
              ir2[1] = (v189_data + (v181_data * v187_data));
              float v192_data = s0[35];
              float v194_data = ir2[2];
              ir2[2] = (v194_data + (v181_data * v192_data));
              float v197_data = s0[51];
              float v199_data = ir2[3];
              ir2[3] = (v199_data + (v181_data * v197_data));
              float v202_data = s0[67];
              float v204_data = ir2[4];
              ir2[4] = (v204_data + (v181_data * v202_data));
              float v207_data = s0[83];
              float v209_data = ir2[5];
              ir2[5] = (v209_data + (v181_data * v207_data));
              float v212_data = s0[99];
              float v214_data = ir2[6];
              ir2[6] = (v214_data + (v181_data * v212_data));
              float v217_data = s0[115];
              float v219_data = ir2[7];
              ir2[7] = (v219_data + (v181_data * v217_data));
            }
            if (v3_lead < 12) {
              float v225_data = r0[4];
              float v226_data = s0[4];
              float v228_data = ir2[0];
              ir2[0] = (v228_data + (v225_data * v226_data));
              float v231_data = s0[20];
              float v233_data = ir2[1];
              ir2[1] = (v233_data + (v225_data * v231_data));
              float v236_data = s0[36];
              float v238_data = ir2[2];
              ir2[2] = (v238_data + (v225_data * v236_data));
              float v241_data = s0[52];
              float v243_data = ir2[3];
              ir2[3] = (v243_data + (v225_data * v241_data));
              float v246_data = s0[68];
              float v248_data = ir2[4];
              ir2[4] = (v248_data + (v225_data * v246_data));
              float v251_data = s0[84];
              float v253_data = ir2[5];
              ir2[5] = (v253_data + (v225_data * v251_data));
              float v256_data = s0[100];
              float v258_data = ir2[6];
              ir2[6] = (v258_data + (v225_data * v256_data));
              float v261_data = s0[116];
              float v263_data = ir2[7];
              ir2[7] = (v263_data + (v225_data * v261_data));
            }
            if (v3_lead < 12) {
              float v269_data = r0[5];
              float v270_data = s0[5];
              float v272_data = ir2[0];
              ir2[0] = (v272_data + (v269_data * v270_data));
              float v275_data = s0[21];
              float v277_data = ir2[1];
              ir2[1] = (v277_data + (v269_data * v275_data));
              float v280_data = s0[37];
              float v282_data = ir2[2];
              ir2[2] = (v282_data + (v269_data * v280_data));
              float v285_data = s0[53];
              float v287_data = ir2[3];
              ir2[3] = (v287_data + (v269_data * v285_data));
              float v290_data = s0[69];
              float v292_data = ir2[4];
              ir2[4] = (v292_data + (v269_data * v290_data));
              float v295_data = s0[85];
              float v297_data = ir2[5];
              ir2[5] = (v297_data + (v269_data * v295_data));
              float v300_data = s0[101];
              float v302_data = ir2[6];
              ir2[6] = (v302_data + (v269_data * v300_data));
              float v305_data = s0[117];
              float v307_data = ir2[7];
              ir2[7] = (v307_data + (v269_data * v305_data));
            }
            if (v3_lead < 12) {
              float v313_data = r0[6];
              float v314_data = s0[6];
              float v316_data = ir2[0];
              ir2[0] = (v316_data + (v313_data * v314_data));
              float v319_data = s0[22];
              float v321_data = ir2[1];
              ir2[1] = (v321_data + (v313_data * v319_data));
              float v324_data = s0[38];
              float v326_data = ir2[2];
              ir2[2] = (v326_data + (v313_data * v324_data));
              float v329_data = s0[54];
              float v331_data = ir2[3];
              ir2[3] = (v331_data + (v313_data * v329_data));
              float v334_data = s0[70];
              float v336_data = ir2[4];
              ir2[4] = (v336_data + (v313_data * v334_data));
              float v339_data = s0[86];
              float v341_data = ir2[5];
              ir2[5] = (v341_data + (v313_data * v339_data));
              float v344_data = s0[102];
              float v346_data = ir2[6];
              ir2[6] = (v346_data + (v313_data * v344_data));
              float v349_data = s0[118];
              float v351_data = ir2[7];
              ir2[7] = (v351_data + (v313_data * v349_data));
            }
            if (v3_lead < 12) {
              float v357_data = r0[7];
              float v358_data = s0[7];
              float v360_data = ir2[0];
              ir2[0] = (v360_data + (v357_data * v358_data));
              float v363_data = s0[23];
              float v365_data = ir2[1];
              ir2[1] = (v365_data + (v357_data * v363_data));
              float v368_data = s0[39];
              float v370_data = ir2[2];
              ir2[2] = (v370_data + (v357_data * v368_data));
              float v373_data = s0[55];
              float v375_data = ir2[3];
              ir2[3] = (v375_data + (v357_data * v373_data));
              float v378_data = s0[71];
              float v380_data = ir2[4];
              ir2[4] = (v380_data + (v357_data * v378_data));
              float v383_data = s0[87];
              float v385_data = ir2[5];
              ir2[5] = (v385_data + (v357_data * v383_data));
              float v388_data = s0[103];
              float v390_data = ir2[6];
              ir2[6] = (v390_data + (v357_data * v388_data));
              float v393_data = s0[119];
              float v395_data = ir2[7];
              ir2[7] = (v395_data + (v357_data * v393_data));
            }
            if (v3_lead < 12) {
              float v401_data = r0[8];
              float v402_data = s0[8];
              float v404_data = ir2[0];
              ir2[0] = (v404_data + (v401_data * v402_data));
              float v407_data = s0[24];
              float v409_data = ir2[1];
              ir2[1] = (v409_data + (v401_data * v407_data));
              float v412_data = s0[40];
              float v414_data = ir2[2];
              ir2[2] = (v414_data + (v401_data * v412_data));
              float v417_data = s0[56];
              float v419_data = ir2[3];
              ir2[3] = (v419_data + (v401_data * v417_data));
              float v422_data = s0[72];
              float v424_data = ir2[4];
              ir2[4] = (v424_data + (v401_data * v422_data));
              float v427_data = s0[88];
              float v429_data = ir2[5];
              ir2[5] = (v429_data + (v401_data * v427_data));
              float v432_data = s0[104];
              float v434_data = ir2[6];
              ir2[6] = (v434_data + (v401_data * v432_data));
              float v437_data = s0[120];
              float v439_data = ir2[7];
              ir2[7] = (v439_data + (v401_data * v437_data));
            }
            if (v3_lead < 12) {
              float v445_data = r0[9];
              float v446_data = s0[9];
              float v448_data = ir2[0];
              ir2[0] = (v448_data + (v445_data * v446_data));
              float v451_data = s0[25];
              float v453_data = ir2[1];
              ir2[1] = (v453_data + (v445_data * v451_data));
              float v456_data = s0[41];
              float v458_data = ir2[2];
              ir2[2] = (v458_data + (v445_data * v456_data));
              float v461_data = s0[57];
              float v463_data = ir2[3];
              ir2[3] = (v463_data + (v445_data * v461_data));
              float v466_data = s0[73];
              float v468_data = ir2[4];
              ir2[4] = (v468_data + (v445_data * v466_data));
              float v471_data = s0[89];
              float v473_data = ir2[5];
              ir2[5] = (v473_data + (v445_data * v471_data));
              float v476_data = s0[105];
              float v478_data = ir2[6];
              ir2[6] = (v478_data + (v445_data * v476_data));
              float v481_data = s0[121];
              float v483_data = ir2[7];
              ir2[7] = (v483_data + (v445_data * v481_data));
            }
            if (v3_lead < 12) {
              float v489_data = r0[10];
              float v490_data = s0[10];
              float v492_data = ir2[0];
              ir2[0] = (v492_data + (v489_data * v490_data));
              float v495_data = s0[26];
              float v497_data = ir2[1];
              ir2[1] = (v497_data + (v489_data * v495_data));
              float v500_data = s0[42];
              float v502_data = ir2[2];
              ir2[2] = (v502_data + (v489_data * v500_data));
              float v505_data = s0[58];
              float v507_data = ir2[3];
              ir2[3] = (v507_data + (v489_data * v505_data));
              float v510_data = s0[74];
              float v512_data = ir2[4];
              ir2[4] = (v512_data + (v489_data * v510_data));
              float v515_data = s0[90];
              float v517_data = ir2[5];
              ir2[5] = (v517_data + (v489_data * v515_data));
              float v520_data = s0[106];
              float v522_data = ir2[6];
              ir2[6] = (v522_data + (v489_data * v520_data));
              float v525_data = s0[122];
              float v527_data = ir2[7];
              ir2[7] = (v527_data + (v489_data * v525_data));
            }
            if (v3_lead < 12) {
              float v533_data = r0[11];
              float v534_data = s0[11];
              float v536_data = ir2[0];
              ir2[0] = (v536_data + (v533_data * v534_data));
              float v539_data = s0[27];
              float v541_data = ir2[1];
              ir2[1] = (v541_data + (v533_data * v539_data));
              float v544_data = s0[43];
              float v546_data = ir2[2];
              ir2[2] = (v546_data + (v533_data * v544_data));
              float v549_data = s0[59];
              float v551_data = ir2[3];
              ir2[3] = (v551_data + (v533_data * v549_data));
              float v554_data = s0[75];
              float v556_data = ir2[4];
              ir2[4] = (v556_data + (v533_data * v554_data));
              float v559_data = s0[91];
              float v561_data = ir2[5];
              ir2[5] = (v561_data + (v533_data * v559_data));
              float v564_data = s0[107];
              float v566_data = ir2[6];
              ir2[6] = (v566_data + (v533_data * v564_data));
              float v569_data = s0[123];
              float v571_data = ir2[7];
              ir2[7] = (v571_data + (v533_data * v569_data));
            }
            if (v3_lead < 12) {
              float v577_data = r0[12];
              float v578_data = s0[12];
              float v580_data = ir2[0];
              ir2[0] = (v580_data + (v577_data * v578_data));
              float v583_data = s0[28];
              float v585_data = ir2[1];
              ir2[1] = (v585_data + (v577_data * v583_data));
              float v588_data = s0[44];
              float v590_data = ir2[2];
              ir2[2] = (v590_data + (v577_data * v588_data));
              float v593_data = s0[60];
              float v595_data = ir2[3];
              ir2[3] = (v595_data + (v577_data * v593_data));
              float v598_data = s0[76];
              float v600_data = ir2[4];
              ir2[4] = (v600_data + (v577_data * v598_data));
              float v603_data = s0[92];
              float v605_data = ir2[5];
              ir2[5] = (v605_data + (v577_data * v603_data));
              float v608_data = s0[108];
              float v610_data = ir2[6];
              ir2[6] = (v610_data + (v577_data * v608_data));
              float v613_data = s0[124];
              float v615_data = ir2[7];
              ir2[7] = (v615_data + (v577_data * v613_data));
            }
            if (v3_lead < 12) {
              float v621_data = r0[13];
              float v622_data = s0[13];
              float v624_data = ir2[0];
              ir2[0] = (v624_data + (v621_data * v622_data));
              float v627_data = s0[29];
              float v629_data = ir2[1];
              ir2[1] = (v629_data + (v621_data * v627_data));
              float v632_data = s0[45];
              float v634_data = ir2[2];
              ir2[2] = (v634_data + (v621_data * v632_data));
              float v637_data = s0[61];
              float v639_data = ir2[3];
              ir2[3] = (v639_data + (v621_data * v637_data));
              float v642_data = s0[77];
              float v644_data = ir2[4];
              ir2[4] = (v644_data + (v621_data * v642_data));
              float v647_data = s0[93];
              float v649_data = ir2[5];
              ir2[5] = (v649_data + (v621_data * v647_data));
              float v652_data = s0[109];
              float v654_data = ir2[6];
              ir2[6] = (v654_data + (v621_data * v652_data));
              float v657_data = s0[125];
              float v659_data = ir2[7];
              ir2[7] = (v659_data + (v621_data * v657_data));
            }
            if (v3_lead < 12) {
              float v665_data = r0[14];
              float v666_data = s0[14];
              float v668_data = ir2[0];
              ir2[0] = (v668_data + (v665_data * v666_data));
              float v671_data = s0[30];
              float v673_data = ir2[1];
              ir2[1] = (v673_data + (v665_data * v671_data));
              float v676_data = s0[46];
              float v678_data = ir2[2];
              ir2[2] = (v678_data + (v665_data * v676_data));
              float v681_data = s0[62];
              float v683_data = ir2[3];
              ir2[3] = (v683_data + (v665_data * v681_data));
              float v686_data = s0[78];
              float v688_data = ir2[4];
              ir2[4] = (v688_data + (v665_data * v686_data));
              float v691_data = s0[94];
              float v693_data = ir2[5];
              ir2[5] = (v693_data + (v665_data * v691_data));
              float v696_data = s0[110];
              float v698_data = ir2[6];
              ir2[6] = (v698_data + (v665_data * v696_data));
              float v701_data = s0[126];
              float v703_data = ir2[7];
              ir2[7] = (v703_data + (v665_data * v701_data));
            }
            if (v3_lead < 12) {
              float v709_data = r0[15];
              float v710_data = s0[15];
              float v712_data = ir2[0];
              ir2[0] = (v712_data + (v709_data * v710_data));
              float v715_data = s0[31];
              float v717_data = ir2[1];
              ir2[1] = (v717_data + (v709_data * v715_data));
              float v720_data = s0[47];
              float v722_data = ir2[2];
              ir2[2] = (v722_data + (v709_data * v720_data));
              float v725_data = s0[63];
              float v727_data = ir2[3];
              ir2[3] = (v727_data + (v709_data * v725_data));
              float v730_data = s0[79];
              float v732_data = ir2[4];
              ir2[4] = (v732_data + (v709_data * v730_data));
              float v735_data = s0[95];
              float v737_data = ir2[5];
              ir2[5] = (v737_data + (v709_data * v735_data));
              float v740_data = s0[111];
              float v742_data = ir2[6];
              ir2[6] = (v742_data + (v709_data * v740_data));
              float v745_data = s0[127];
              float v747_data = ir2[7];
              ir2[7] = (v747_data + (v709_data * v745_data));
            }
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v753_n1 = 0; v753_n1 < 8; ++v753_n1) {
                int32_t v754_a = 0 + v753_n1;
                float v756_data = ir2[v753_n1];
                int32_t v757_a = 0 + v753_n1;
                float v759_data = r1[v753_n1];
                int32_t v761_a = 0 + v753_n1;
                r2[v753_n1] = (v759_data + v756_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v767_i1 = 0; v767_i1 < 8; ++v767_i1) {
              int32_t v768_a = 0 + v767_i1;
              float v770_data = r2[v767_i1];
              int32_t v777_a = v3_lead + (v767_i1 * 12);
              glb_m0[v777_a] = v770_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

