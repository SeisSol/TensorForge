// === base name ===
kernel_f61651fe59

// === header ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f61651fe59, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_f61651fe59, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f61651fe59<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 32×16(12×16) {4..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(12×16) {4..16}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            int32_t v11_a = (v2_lead + 4) - 4;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v13_a = v11_a + (v4_i1 * 12);
              float v14_data;
              {
                v14_data = __ldcg(&glb_m1[v13_a]);
              }
              int32_t v15_a = 0 + v4_i1;
              r0[v15_a] = v14_data;
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
            int32_t v18_lead = threadIdx.x % 16;
            if (v18_lead < 12) {
              float v20_data = r0[0];
              float v21_data = s0[0];
              float v23_data = ir1[0];
              ir1[0] = (v23_data + (v20_data * v21_data));
              float v26_data = s0[16];
              float v28_data = ir1[1];
              ir1[1] = (v28_data + (v20_data * v26_data));
              float v31_data = s0[32];
              float v33_data = ir1[2];
              ir1[2] = (v33_data + (v20_data * v31_data));
              float v36_data = s0[48];
              float v38_data = ir1[3];
              ir1[3] = (v38_data + (v20_data * v36_data));
              float v41_data = s0[64];
              float v43_data = ir1[4];
              ir1[4] = (v43_data + (v20_data * v41_data));
              float v46_data = s0[80];
              float v48_data = ir1[5];
              ir1[5] = (v48_data + (v20_data * v46_data));
              float v51_data = s0[96];
              float v53_data = ir1[6];
              ir1[6] = (v53_data + (v20_data * v51_data));
              float v56_data = s0[112];
              float v58_data = ir1[7];
              ir1[7] = (v58_data + (v20_data * v56_data));
            }
            if (v18_lead < 12) {
              float v64_data = r0[1];
              float v65_data = s0[1];
              float v67_data = ir1[0];
              ir1[0] = (v67_data + (v64_data * v65_data));
              float v70_data = s0[17];
              float v72_data = ir1[1];
              ir1[1] = (v72_data + (v64_data * v70_data));
              float v75_data = s0[33];
              float v77_data = ir1[2];
              ir1[2] = (v77_data + (v64_data * v75_data));
              float v80_data = s0[49];
              float v82_data = ir1[3];
              ir1[3] = (v82_data + (v64_data * v80_data));
              float v85_data = s0[65];
              float v87_data = ir1[4];
              ir1[4] = (v87_data + (v64_data * v85_data));
              float v90_data = s0[81];
              float v92_data = ir1[5];
              ir1[5] = (v92_data + (v64_data * v90_data));
              float v95_data = s0[97];
              float v97_data = ir1[6];
              ir1[6] = (v97_data + (v64_data * v95_data));
              float v100_data = s0[113];
              float v102_data = ir1[7];
              ir1[7] = (v102_data + (v64_data * v100_data));
            }
            if (v18_lead < 12) {
              float v108_data = r0[2];
              float v109_data = s0[2];
              float v111_data = ir1[0];
              ir1[0] = (v111_data + (v108_data * v109_data));
              float v114_data = s0[18];
              float v116_data = ir1[1];
              ir1[1] = (v116_data + (v108_data * v114_data));
              float v119_data = s0[34];
              float v121_data = ir1[2];
              ir1[2] = (v121_data + (v108_data * v119_data));
              float v124_data = s0[50];
              float v126_data = ir1[3];
              ir1[3] = (v126_data + (v108_data * v124_data));
              float v129_data = s0[66];
              float v131_data = ir1[4];
              ir1[4] = (v131_data + (v108_data * v129_data));
              float v134_data = s0[82];
              float v136_data = ir1[5];
              ir1[5] = (v136_data + (v108_data * v134_data));
              float v139_data = s0[98];
              float v141_data = ir1[6];
              ir1[6] = (v141_data + (v108_data * v139_data));
              float v144_data = s0[114];
              float v146_data = ir1[7];
              ir1[7] = (v146_data + (v108_data * v144_data));
            }
            if (v18_lead < 12) {
              float v152_data = r0[3];
              float v153_data = s0[3];
              float v155_data = ir1[0];
              ir1[0] = (v155_data + (v152_data * v153_data));
              float v158_data = s0[19];
              float v160_data = ir1[1];
              ir1[1] = (v160_data + (v152_data * v158_data));
              float v163_data = s0[35];
              float v165_data = ir1[2];
              ir1[2] = (v165_data + (v152_data * v163_data));
              float v168_data = s0[51];
              float v170_data = ir1[3];
              ir1[3] = (v170_data + (v152_data * v168_data));
              float v173_data = s0[67];
              float v175_data = ir1[4];
              ir1[4] = (v175_data + (v152_data * v173_data));
              float v178_data = s0[83];
              float v180_data = ir1[5];
              ir1[5] = (v180_data + (v152_data * v178_data));
              float v183_data = s0[99];
              float v185_data = ir1[6];
              ir1[6] = (v185_data + (v152_data * v183_data));
              float v188_data = s0[115];
              float v190_data = ir1[7];
              ir1[7] = (v190_data + (v152_data * v188_data));
            }
            if (v18_lead < 12) {
              float v196_data = r0[4];
              float v197_data = s0[4];
              float v199_data = ir1[0];
              ir1[0] = (v199_data + (v196_data * v197_data));
              float v202_data = s0[20];
              float v204_data = ir1[1];
              ir1[1] = (v204_data + (v196_data * v202_data));
              float v207_data = s0[36];
              float v209_data = ir1[2];
              ir1[2] = (v209_data + (v196_data * v207_data));
              float v212_data = s0[52];
              float v214_data = ir1[3];
              ir1[3] = (v214_data + (v196_data * v212_data));
              float v217_data = s0[68];
              float v219_data = ir1[4];
              ir1[4] = (v219_data + (v196_data * v217_data));
              float v222_data = s0[84];
              float v224_data = ir1[5];
              ir1[5] = (v224_data + (v196_data * v222_data));
              float v227_data = s0[100];
              float v229_data = ir1[6];
              ir1[6] = (v229_data + (v196_data * v227_data));
              float v232_data = s0[116];
              float v234_data = ir1[7];
              ir1[7] = (v234_data + (v196_data * v232_data));
            }
            if (v18_lead < 12) {
              float v240_data = r0[5];
              float v241_data = s0[5];
              float v243_data = ir1[0];
              ir1[0] = (v243_data + (v240_data * v241_data));
              float v246_data = s0[21];
              float v248_data = ir1[1];
              ir1[1] = (v248_data + (v240_data * v246_data));
              float v251_data = s0[37];
              float v253_data = ir1[2];
              ir1[2] = (v253_data + (v240_data * v251_data));
              float v256_data = s0[53];
              float v258_data = ir1[3];
              ir1[3] = (v258_data + (v240_data * v256_data));
              float v261_data = s0[69];
              float v263_data = ir1[4];
              ir1[4] = (v263_data + (v240_data * v261_data));
              float v266_data = s0[85];
              float v268_data = ir1[5];
              ir1[5] = (v268_data + (v240_data * v266_data));
              float v271_data = s0[101];
              float v273_data = ir1[6];
              ir1[6] = (v273_data + (v240_data * v271_data));
              float v276_data = s0[117];
              float v278_data = ir1[7];
              ir1[7] = (v278_data + (v240_data * v276_data));
            }
            if (v18_lead < 12) {
              float v284_data = r0[6];
              float v285_data = s0[6];
              float v287_data = ir1[0];
              ir1[0] = (v287_data + (v284_data * v285_data));
              float v290_data = s0[22];
              float v292_data = ir1[1];
              ir1[1] = (v292_data + (v284_data * v290_data));
              float v295_data = s0[38];
              float v297_data = ir1[2];
              ir1[2] = (v297_data + (v284_data * v295_data));
              float v300_data = s0[54];
              float v302_data = ir1[3];
              ir1[3] = (v302_data + (v284_data * v300_data));
              float v305_data = s0[70];
              float v307_data = ir1[4];
              ir1[4] = (v307_data + (v284_data * v305_data));
              float v310_data = s0[86];
              float v312_data = ir1[5];
              ir1[5] = (v312_data + (v284_data * v310_data));
              float v315_data = s0[102];
              float v317_data = ir1[6];
              ir1[6] = (v317_data + (v284_data * v315_data));
              float v320_data = s0[118];
              float v322_data = ir1[7];
              ir1[7] = (v322_data + (v284_data * v320_data));
            }
            if (v18_lead < 12) {
              float v328_data = r0[7];
              float v329_data = s0[7];
              float v331_data = ir1[0];
              ir1[0] = (v331_data + (v328_data * v329_data));
              float v334_data = s0[23];
              float v336_data = ir1[1];
              ir1[1] = (v336_data + (v328_data * v334_data));
              float v339_data = s0[39];
              float v341_data = ir1[2];
              ir1[2] = (v341_data + (v328_data * v339_data));
              float v344_data = s0[55];
              float v346_data = ir1[3];
              ir1[3] = (v346_data + (v328_data * v344_data));
              float v349_data = s0[71];
              float v351_data = ir1[4];
              ir1[4] = (v351_data + (v328_data * v349_data));
              float v354_data = s0[87];
              float v356_data = ir1[5];
              ir1[5] = (v356_data + (v328_data * v354_data));
              float v359_data = s0[103];
              float v361_data = ir1[6];
              ir1[6] = (v361_data + (v328_data * v359_data));
              float v364_data = s0[119];
              float v366_data = ir1[7];
              ir1[7] = (v366_data + (v328_data * v364_data));
            }
            if (v18_lead < 12) {
              float v372_data = r0[8];
              float v373_data = s0[8];
              float v375_data = ir1[0];
              ir1[0] = (v375_data + (v372_data * v373_data));
              float v378_data = s0[24];
              float v380_data = ir1[1];
              ir1[1] = (v380_data + (v372_data * v378_data));
              float v383_data = s0[40];
              float v385_data = ir1[2];
              ir1[2] = (v385_data + (v372_data * v383_data));
              float v388_data = s0[56];
              float v390_data = ir1[3];
              ir1[3] = (v390_data + (v372_data * v388_data));
              float v393_data = s0[72];
              float v395_data = ir1[4];
              ir1[4] = (v395_data + (v372_data * v393_data));
              float v398_data = s0[88];
              float v400_data = ir1[5];
              ir1[5] = (v400_data + (v372_data * v398_data));
              float v403_data = s0[104];
              float v405_data = ir1[6];
              ir1[6] = (v405_data + (v372_data * v403_data));
              float v408_data = s0[120];
              float v410_data = ir1[7];
              ir1[7] = (v410_data + (v372_data * v408_data));
            }
            if (v18_lead < 12) {
              float v416_data = r0[9];
              float v417_data = s0[9];
              float v419_data = ir1[0];
              ir1[0] = (v419_data + (v416_data * v417_data));
              float v422_data = s0[25];
              float v424_data = ir1[1];
              ir1[1] = (v424_data + (v416_data * v422_data));
              float v427_data = s0[41];
              float v429_data = ir1[2];
              ir1[2] = (v429_data + (v416_data * v427_data));
              float v432_data = s0[57];
              float v434_data = ir1[3];
              ir1[3] = (v434_data + (v416_data * v432_data));
              float v437_data = s0[73];
              float v439_data = ir1[4];
              ir1[4] = (v439_data + (v416_data * v437_data));
              float v442_data = s0[89];
              float v444_data = ir1[5];
              ir1[5] = (v444_data + (v416_data * v442_data));
              float v447_data = s0[105];
              float v449_data = ir1[6];
              ir1[6] = (v449_data + (v416_data * v447_data));
              float v452_data = s0[121];
              float v454_data = ir1[7];
              ir1[7] = (v454_data + (v416_data * v452_data));
            }
            if (v18_lead < 12) {
              float v460_data = r0[10];
              float v461_data = s0[10];
              float v463_data = ir1[0];
              ir1[0] = (v463_data + (v460_data * v461_data));
              float v466_data = s0[26];
              float v468_data = ir1[1];
              ir1[1] = (v468_data + (v460_data * v466_data));
              float v471_data = s0[42];
              float v473_data = ir1[2];
              ir1[2] = (v473_data + (v460_data * v471_data));
              float v476_data = s0[58];
              float v478_data = ir1[3];
              ir1[3] = (v478_data + (v460_data * v476_data));
              float v481_data = s0[74];
              float v483_data = ir1[4];
              ir1[4] = (v483_data + (v460_data * v481_data));
              float v486_data = s0[90];
              float v488_data = ir1[5];
              ir1[5] = (v488_data + (v460_data * v486_data));
              float v491_data = s0[106];
              float v493_data = ir1[6];
              ir1[6] = (v493_data + (v460_data * v491_data));
              float v496_data = s0[122];
              float v498_data = ir1[7];
              ir1[7] = (v498_data + (v460_data * v496_data));
            }
            if (v18_lead < 12) {
              float v504_data = r0[11];
              float v505_data = s0[11];
              float v507_data = ir1[0];
              ir1[0] = (v507_data + (v504_data * v505_data));
              float v510_data = s0[27];
              float v512_data = ir1[1];
              ir1[1] = (v512_data + (v504_data * v510_data));
              float v515_data = s0[43];
              float v517_data = ir1[2];
              ir1[2] = (v517_data + (v504_data * v515_data));
              float v520_data = s0[59];
              float v522_data = ir1[3];
              ir1[3] = (v522_data + (v504_data * v520_data));
              float v525_data = s0[75];
              float v527_data = ir1[4];
              ir1[4] = (v527_data + (v504_data * v525_data));
              float v530_data = s0[91];
              float v532_data = ir1[5];
              ir1[5] = (v532_data + (v504_data * v530_data));
              float v535_data = s0[107];
              float v537_data = ir1[6];
              ir1[6] = (v537_data + (v504_data * v535_data));
              float v540_data = s0[123];
              float v542_data = ir1[7];
              ir1[7] = (v542_data + (v504_data * v540_data));
            }
            if (v18_lead < 12) {
              float v548_data = r0[12];
              float v549_data = s0[12];
              float v551_data = ir1[0];
              ir1[0] = (v551_data + (v548_data * v549_data));
              float v554_data = s0[28];
              float v556_data = ir1[1];
              ir1[1] = (v556_data + (v548_data * v554_data));
              float v559_data = s0[44];
              float v561_data = ir1[2];
              ir1[2] = (v561_data + (v548_data * v559_data));
              float v564_data = s0[60];
              float v566_data = ir1[3];
              ir1[3] = (v566_data + (v548_data * v564_data));
              float v569_data = s0[76];
              float v571_data = ir1[4];
              ir1[4] = (v571_data + (v548_data * v569_data));
              float v574_data = s0[92];
              float v576_data = ir1[5];
              ir1[5] = (v576_data + (v548_data * v574_data));
              float v579_data = s0[108];
              float v581_data = ir1[6];
              ir1[6] = (v581_data + (v548_data * v579_data));
              float v584_data = s0[124];
              float v586_data = ir1[7];
              ir1[7] = (v586_data + (v548_data * v584_data));
            }
            if (v18_lead < 12) {
              float v592_data = r0[13];
              float v593_data = s0[13];
              float v595_data = ir1[0];
              ir1[0] = (v595_data + (v592_data * v593_data));
              float v598_data = s0[29];
              float v600_data = ir1[1];
              ir1[1] = (v600_data + (v592_data * v598_data));
              float v603_data = s0[45];
              float v605_data = ir1[2];
              ir1[2] = (v605_data + (v592_data * v603_data));
              float v608_data = s0[61];
              float v610_data = ir1[3];
              ir1[3] = (v610_data + (v592_data * v608_data));
              float v613_data = s0[77];
              float v615_data = ir1[4];
              ir1[4] = (v615_data + (v592_data * v613_data));
              float v618_data = s0[93];
              float v620_data = ir1[5];
              ir1[5] = (v620_data + (v592_data * v618_data));
              float v623_data = s0[109];
              float v625_data = ir1[6];
              ir1[6] = (v625_data + (v592_data * v623_data));
              float v628_data = s0[125];
              float v630_data = ir1[7];
              ir1[7] = (v630_data + (v592_data * v628_data));
            }
            if (v18_lead < 12) {
              float v636_data = r0[14];
              float v637_data = s0[14];
              float v639_data = ir1[0];
              ir1[0] = (v639_data + (v636_data * v637_data));
              float v642_data = s0[30];
              float v644_data = ir1[1];
              ir1[1] = (v644_data + (v636_data * v642_data));
              float v647_data = s0[46];
              float v649_data = ir1[2];
              ir1[2] = (v649_data + (v636_data * v647_data));
              float v652_data = s0[62];
              float v654_data = ir1[3];
              ir1[3] = (v654_data + (v636_data * v652_data));
              float v657_data = s0[78];
              float v659_data = ir1[4];
              ir1[4] = (v659_data + (v636_data * v657_data));
              float v662_data = s0[94];
              float v664_data = ir1[5];
              ir1[5] = (v664_data + (v636_data * v662_data));
              float v667_data = s0[110];
              float v669_data = ir1[6];
              ir1[6] = (v669_data + (v636_data * v667_data));
              float v672_data = s0[126];
              float v674_data = ir1[7];
              ir1[7] = (v674_data + (v636_data * v672_data));
            }
            if (v18_lead < 12) {
              float v680_data = r0[15];
              float v681_data = s0[15];
              float v683_data = ir1[0];
              ir1[0] = (v683_data + (v680_data * v681_data));
              float v686_data = s0[31];
              float v688_data = ir1[1];
              ir1[1] = (v688_data + (v680_data * v686_data));
              float v691_data = s0[47];
              float v693_data = ir1[2];
              ir1[2] = (v693_data + (v680_data * v691_data));
              float v696_data = s0[63];
              float v698_data = ir1[3];
              ir1[3] = (v698_data + (v680_data * v696_data));
              float v701_data = s0[79];
              float v703_data = ir1[4];
              ir1[4] = (v703_data + (v680_data * v701_data));
              float v706_data = s0[95];
              float v708_data = ir1[5];
              ir1[5] = (v708_data + (v680_data * v706_data));
              float v711_data = s0[111];
              float v713_data = ir1[6];
              ir1[6] = (v713_data + (v680_data * v711_data));
              float v716_data = s0[127];
              float v718_data = ir1[7];
              ir1[7] = (v718_data + (v680_data * v716_data));
            }
            if (v18_lead < 12) {
              #pragma unroll
              for (int32_t v724_n1 = 0; v724_n1 < 8; ++v724_n1) {
                int32_t v725_a = 0 + v724_n1;
                float v726_data = ir1[v725_a];
                int32_t v727_a = 0 + v724_n1;
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
              float v734_data = r1[v733_a];
              int32_t v741_a = v730_lead + (v732_i1 * 12);
              glb_m0[v741_a] = v734_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

