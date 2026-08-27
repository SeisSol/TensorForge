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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            int32_t v11_off = v3_lead + 4;
            int32_t v19_off = v3_lead + 4;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v12_a = v5_i1 * 32;
              int32_t v13_a = v11_off + v12_a;
              float v22_data = __ldcg(&glb_m1[(v19_off + v12_a)]);
              int32_t v23_a = 0 + v5_i1;
              r0[v23_a] = v22_data;
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
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir1[8]{};
          if (v3_lead < 12) {
            float v30_data = r0[0];
            float v31_data = s0[0];
            float v33_data = ir1[0];
            ir1[0] = (v33_data + (v30_data * v31_data));
            float v36_data = s0[16];
            float v38_data = ir1[1];
            ir1[1] = (v38_data + (v30_data * v36_data));
            float v41_data = s0[32];
            float v43_data = ir1[2];
            ir1[2] = (v43_data + (v30_data * v41_data));
            float v46_data = s0[48];
            float v48_data = ir1[3];
            ir1[3] = (v48_data + (v30_data * v46_data));
            float v51_data = s0[64];
            float v53_data = ir1[4];
            ir1[4] = (v53_data + (v30_data * v51_data));
            float v56_data = s0[80];
            float v58_data = ir1[5];
            ir1[5] = (v58_data + (v30_data * v56_data));
            float v61_data = s0[96];
            float v63_data = ir1[6];
            ir1[6] = (v63_data + (v30_data * v61_data));
            float v66_data = s0[112];
            float v68_data = ir1[7];
            ir1[7] = (v68_data + (v30_data * v66_data));
          }
          if (v3_lead < 12) {
            float v74_data = r0[1];
            float v75_data = s0[1];
            float v77_data = ir1[0];
            ir1[0] = (v77_data + (v74_data * v75_data));
            float v80_data = s0[17];
            float v82_data = ir1[1];
            ir1[1] = (v82_data + (v74_data * v80_data));
            float v85_data = s0[33];
            float v87_data = ir1[2];
            ir1[2] = (v87_data + (v74_data * v85_data));
            float v90_data = s0[49];
            float v92_data = ir1[3];
            ir1[3] = (v92_data + (v74_data * v90_data));
            float v95_data = s0[65];
            float v97_data = ir1[4];
            ir1[4] = (v97_data + (v74_data * v95_data));
            float v100_data = s0[81];
            float v102_data = ir1[5];
            ir1[5] = (v102_data + (v74_data * v100_data));
            float v105_data = s0[97];
            float v107_data = ir1[6];
            ir1[6] = (v107_data + (v74_data * v105_data));
            float v110_data = s0[113];
            float v112_data = ir1[7];
            ir1[7] = (v112_data + (v74_data * v110_data));
          }
          if (v3_lead < 12) {
            float v118_data = r0[2];
            float v119_data = s0[2];
            float v121_data = ir1[0];
            ir1[0] = (v121_data + (v118_data * v119_data));
            float v124_data = s0[18];
            float v126_data = ir1[1];
            ir1[1] = (v126_data + (v118_data * v124_data));
            float v129_data = s0[34];
            float v131_data = ir1[2];
            ir1[2] = (v131_data + (v118_data * v129_data));
            float v134_data = s0[50];
            float v136_data = ir1[3];
            ir1[3] = (v136_data + (v118_data * v134_data));
            float v139_data = s0[66];
            float v141_data = ir1[4];
            ir1[4] = (v141_data + (v118_data * v139_data));
            float v144_data = s0[82];
            float v146_data = ir1[5];
            ir1[5] = (v146_data + (v118_data * v144_data));
            float v149_data = s0[98];
            float v151_data = ir1[6];
            ir1[6] = (v151_data + (v118_data * v149_data));
            float v154_data = s0[114];
            float v156_data = ir1[7];
            ir1[7] = (v156_data + (v118_data * v154_data));
          }
          if (v3_lead < 12) {
            float v162_data = r0[3];
            float v163_data = s0[3];
            float v165_data = ir1[0];
            ir1[0] = (v165_data + (v162_data * v163_data));
            float v168_data = s0[19];
            float v170_data = ir1[1];
            ir1[1] = (v170_data + (v162_data * v168_data));
            float v173_data = s0[35];
            float v175_data = ir1[2];
            ir1[2] = (v175_data + (v162_data * v173_data));
            float v178_data = s0[51];
            float v180_data = ir1[3];
            ir1[3] = (v180_data + (v162_data * v178_data));
            float v183_data = s0[67];
            float v185_data = ir1[4];
            ir1[4] = (v185_data + (v162_data * v183_data));
            float v188_data = s0[83];
            float v190_data = ir1[5];
            ir1[5] = (v190_data + (v162_data * v188_data));
            float v193_data = s0[99];
            float v195_data = ir1[6];
            ir1[6] = (v195_data + (v162_data * v193_data));
            float v198_data = s0[115];
            float v200_data = ir1[7];
            ir1[7] = (v200_data + (v162_data * v198_data));
          }
          if (v3_lead < 12) {
            float v206_data = r0[4];
            float v207_data = s0[4];
            float v209_data = ir1[0];
            ir1[0] = (v209_data + (v206_data * v207_data));
            float v212_data = s0[20];
            float v214_data = ir1[1];
            ir1[1] = (v214_data + (v206_data * v212_data));
            float v217_data = s0[36];
            float v219_data = ir1[2];
            ir1[2] = (v219_data + (v206_data * v217_data));
            float v222_data = s0[52];
            float v224_data = ir1[3];
            ir1[3] = (v224_data + (v206_data * v222_data));
            float v227_data = s0[68];
            float v229_data = ir1[4];
            ir1[4] = (v229_data + (v206_data * v227_data));
            float v232_data = s0[84];
            float v234_data = ir1[5];
            ir1[5] = (v234_data + (v206_data * v232_data));
            float v237_data = s0[100];
            float v239_data = ir1[6];
            ir1[6] = (v239_data + (v206_data * v237_data));
            float v242_data = s0[116];
            float v244_data = ir1[7];
            ir1[7] = (v244_data + (v206_data * v242_data));
          }
          if (v3_lead < 12) {
            float v250_data = r0[5];
            float v251_data = s0[5];
            float v253_data = ir1[0];
            ir1[0] = (v253_data + (v250_data * v251_data));
            float v256_data = s0[21];
            float v258_data = ir1[1];
            ir1[1] = (v258_data + (v250_data * v256_data));
            float v261_data = s0[37];
            float v263_data = ir1[2];
            ir1[2] = (v263_data + (v250_data * v261_data));
            float v266_data = s0[53];
            float v268_data = ir1[3];
            ir1[3] = (v268_data + (v250_data * v266_data));
            float v271_data = s0[69];
            float v273_data = ir1[4];
            ir1[4] = (v273_data + (v250_data * v271_data));
            float v276_data = s0[85];
            float v278_data = ir1[5];
            ir1[5] = (v278_data + (v250_data * v276_data));
            float v281_data = s0[101];
            float v283_data = ir1[6];
            ir1[6] = (v283_data + (v250_data * v281_data));
            float v286_data = s0[117];
            float v288_data = ir1[7];
            ir1[7] = (v288_data + (v250_data * v286_data));
          }
          if (v3_lead < 12) {
            float v294_data = r0[6];
            float v295_data = s0[6];
            float v297_data = ir1[0];
            ir1[0] = (v297_data + (v294_data * v295_data));
            float v300_data = s0[22];
            float v302_data = ir1[1];
            ir1[1] = (v302_data + (v294_data * v300_data));
            float v305_data = s0[38];
            float v307_data = ir1[2];
            ir1[2] = (v307_data + (v294_data * v305_data));
            float v310_data = s0[54];
            float v312_data = ir1[3];
            ir1[3] = (v312_data + (v294_data * v310_data));
            float v315_data = s0[70];
            float v317_data = ir1[4];
            ir1[4] = (v317_data + (v294_data * v315_data));
            float v320_data = s0[86];
            float v322_data = ir1[5];
            ir1[5] = (v322_data + (v294_data * v320_data));
            float v325_data = s0[102];
            float v327_data = ir1[6];
            ir1[6] = (v327_data + (v294_data * v325_data));
            float v330_data = s0[118];
            float v332_data = ir1[7];
            ir1[7] = (v332_data + (v294_data * v330_data));
          }
          if (v3_lead < 12) {
            float v338_data = r0[7];
            float v339_data = s0[7];
            float v341_data = ir1[0];
            ir1[0] = (v341_data + (v338_data * v339_data));
            float v344_data = s0[23];
            float v346_data = ir1[1];
            ir1[1] = (v346_data + (v338_data * v344_data));
            float v349_data = s0[39];
            float v351_data = ir1[2];
            ir1[2] = (v351_data + (v338_data * v349_data));
            float v354_data = s0[55];
            float v356_data = ir1[3];
            ir1[3] = (v356_data + (v338_data * v354_data));
            float v359_data = s0[71];
            float v361_data = ir1[4];
            ir1[4] = (v361_data + (v338_data * v359_data));
            float v364_data = s0[87];
            float v366_data = ir1[5];
            ir1[5] = (v366_data + (v338_data * v364_data));
            float v369_data = s0[103];
            float v371_data = ir1[6];
            ir1[6] = (v371_data + (v338_data * v369_data));
            float v374_data = s0[119];
            float v376_data = ir1[7];
            ir1[7] = (v376_data + (v338_data * v374_data));
          }
          if (v3_lead < 12) {
            float v382_data = r0[8];
            float v383_data = s0[8];
            float v385_data = ir1[0];
            ir1[0] = (v385_data + (v382_data * v383_data));
            float v388_data = s0[24];
            float v390_data = ir1[1];
            ir1[1] = (v390_data + (v382_data * v388_data));
            float v393_data = s0[40];
            float v395_data = ir1[2];
            ir1[2] = (v395_data + (v382_data * v393_data));
            float v398_data = s0[56];
            float v400_data = ir1[3];
            ir1[3] = (v400_data + (v382_data * v398_data));
            float v403_data = s0[72];
            float v405_data = ir1[4];
            ir1[4] = (v405_data + (v382_data * v403_data));
            float v408_data = s0[88];
            float v410_data = ir1[5];
            ir1[5] = (v410_data + (v382_data * v408_data));
            float v413_data = s0[104];
            float v415_data = ir1[6];
            ir1[6] = (v415_data + (v382_data * v413_data));
            float v418_data = s0[120];
            float v420_data = ir1[7];
            ir1[7] = (v420_data + (v382_data * v418_data));
          }
          if (v3_lead < 12) {
            float v426_data = r0[9];
            float v427_data = s0[9];
            float v429_data = ir1[0];
            ir1[0] = (v429_data + (v426_data * v427_data));
            float v432_data = s0[25];
            float v434_data = ir1[1];
            ir1[1] = (v434_data + (v426_data * v432_data));
            float v437_data = s0[41];
            float v439_data = ir1[2];
            ir1[2] = (v439_data + (v426_data * v437_data));
            float v442_data = s0[57];
            float v444_data = ir1[3];
            ir1[3] = (v444_data + (v426_data * v442_data));
            float v447_data = s0[73];
            float v449_data = ir1[4];
            ir1[4] = (v449_data + (v426_data * v447_data));
            float v452_data = s0[89];
            float v454_data = ir1[5];
            ir1[5] = (v454_data + (v426_data * v452_data));
            float v457_data = s0[105];
            float v459_data = ir1[6];
            ir1[6] = (v459_data + (v426_data * v457_data));
            float v462_data = s0[121];
            float v464_data = ir1[7];
            ir1[7] = (v464_data + (v426_data * v462_data));
          }
          if (v3_lead < 12) {
            float v470_data = r0[10];
            float v471_data = s0[10];
            float v473_data = ir1[0];
            ir1[0] = (v473_data + (v470_data * v471_data));
            float v476_data = s0[26];
            float v478_data = ir1[1];
            ir1[1] = (v478_data + (v470_data * v476_data));
            float v481_data = s0[42];
            float v483_data = ir1[2];
            ir1[2] = (v483_data + (v470_data * v481_data));
            float v486_data = s0[58];
            float v488_data = ir1[3];
            ir1[3] = (v488_data + (v470_data * v486_data));
            float v491_data = s0[74];
            float v493_data = ir1[4];
            ir1[4] = (v493_data + (v470_data * v491_data));
            float v496_data = s0[90];
            float v498_data = ir1[5];
            ir1[5] = (v498_data + (v470_data * v496_data));
            float v501_data = s0[106];
            float v503_data = ir1[6];
            ir1[6] = (v503_data + (v470_data * v501_data));
            float v506_data = s0[122];
            float v508_data = ir1[7];
            ir1[7] = (v508_data + (v470_data * v506_data));
          }
          if (v3_lead < 12) {
            float v514_data = r0[11];
            float v515_data = s0[11];
            float v517_data = ir1[0];
            ir1[0] = (v517_data + (v514_data * v515_data));
            float v520_data = s0[27];
            float v522_data = ir1[1];
            ir1[1] = (v522_data + (v514_data * v520_data));
            float v525_data = s0[43];
            float v527_data = ir1[2];
            ir1[2] = (v527_data + (v514_data * v525_data));
            float v530_data = s0[59];
            float v532_data = ir1[3];
            ir1[3] = (v532_data + (v514_data * v530_data));
            float v535_data = s0[75];
            float v537_data = ir1[4];
            ir1[4] = (v537_data + (v514_data * v535_data));
            float v540_data = s0[91];
            float v542_data = ir1[5];
            ir1[5] = (v542_data + (v514_data * v540_data));
            float v545_data = s0[107];
            float v547_data = ir1[6];
            ir1[6] = (v547_data + (v514_data * v545_data));
            float v550_data = s0[123];
            float v552_data = ir1[7];
            ir1[7] = (v552_data + (v514_data * v550_data));
          }
          if (v3_lead < 12) {
            float v558_data = r0[12];
            float v559_data = s0[12];
            float v561_data = ir1[0];
            ir1[0] = (v561_data + (v558_data * v559_data));
            float v564_data = s0[28];
            float v566_data = ir1[1];
            ir1[1] = (v566_data + (v558_data * v564_data));
            float v569_data = s0[44];
            float v571_data = ir1[2];
            ir1[2] = (v571_data + (v558_data * v569_data));
            float v574_data = s0[60];
            float v576_data = ir1[3];
            ir1[3] = (v576_data + (v558_data * v574_data));
            float v579_data = s0[76];
            float v581_data = ir1[4];
            ir1[4] = (v581_data + (v558_data * v579_data));
            float v584_data = s0[92];
            float v586_data = ir1[5];
            ir1[5] = (v586_data + (v558_data * v584_data));
            float v589_data = s0[108];
            float v591_data = ir1[6];
            ir1[6] = (v591_data + (v558_data * v589_data));
            float v594_data = s0[124];
            float v596_data = ir1[7];
            ir1[7] = (v596_data + (v558_data * v594_data));
          }
          if (v3_lead < 12) {
            float v602_data = r0[13];
            float v603_data = s0[13];
            float v605_data = ir1[0];
            ir1[0] = (v605_data + (v602_data * v603_data));
            float v608_data = s0[29];
            float v610_data = ir1[1];
            ir1[1] = (v610_data + (v602_data * v608_data));
            float v613_data = s0[45];
            float v615_data = ir1[2];
            ir1[2] = (v615_data + (v602_data * v613_data));
            float v618_data = s0[61];
            float v620_data = ir1[3];
            ir1[3] = (v620_data + (v602_data * v618_data));
            float v623_data = s0[77];
            float v625_data = ir1[4];
            ir1[4] = (v625_data + (v602_data * v623_data));
            float v628_data = s0[93];
            float v630_data = ir1[5];
            ir1[5] = (v630_data + (v602_data * v628_data));
            float v633_data = s0[109];
            float v635_data = ir1[6];
            ir1[6] = (v635_data + (v602_data * v633_data));
            float v638_data = s0[125];
            float v640_data = ir1[7];
            ir1[7] = (v640_data + (v602_data * v638_data));
          }
          if (v3_lead < 12) {
            float v646_data = r0[14];
            float v647_data = s0[14];
            float v649_data = ir1[0];
            ir1[0] = (v649_data + (v646_data * v647_data));
            float v652_data = s0[30];
            float v654_data = ir1[1];
            ir1[1] = (v654_data + (v646_data * v652_data));
            float v657_data = s0[46];
            float v659_data = ir1[2];
            ir1[2] = (v659_data + (v646_data * v657_data));
            float v662_data = s0[62];
            float v664_data = ir1[3];
            ir1[3] = (v664_data + (v646_data * v662_data));
            float v667_data = s0[78];
            float v669_data = ir1[4];
            ir1[4] = (v669_data + (v646_data * v667_data));
            float v672_data = s0[94];
            float v674_data = ir1[5];
            ir1[5] = (v674_data + (v646_data * v672_data));
            float v677_data = s0[110];
            float v679_data = ir1[6];
            ir1[6] = (v679_data + (v646_data * v677_data));
            float v682_data = s0[126];
            float v684_data = ir1[7];
            ir1[7] = (v684_data + (v646_data * v682_data));
          }
          if (v3_lead < 12) {
            float v690_data = r0[15];
            float v691_data = s0[15];
            float v693_data = ir1[0];
            ir1[0] = (v693_data + (v690_data * v691_data));
            float v696_data = s0[31];
            float v698_data = ir1[1];
            ir1[1] = (v698_data + (v690_data * v696_data));
            float v701_data = s0[47];
            float v703_data = ir1[2];
            ir1[2] = (v703_data + (v690_data * v701_data));
            float v706_data = s0[63];
            float v708_data = ir1[3];
            ir1[3] = (v708_data + (v690_data * v706_data));
            float v711_data = s0[79];
            float v713_data = ir1[4];
            ir1[4] = (v713_data + (v690_data * v711_data));
            float v716_data = s0[95];
            float v718_data = ir1[5];
            ir1[5] = (v718_data + (v690_data * v716_data));
            float v721_data = s0[111];
            float v723_data = ir1[6];
            ir1[6] = (v723_data + (v690_data * v721_data));
            float v726_data = s0[127];
            float v728_data = ir1[7];
            ir1[7] = (v728_data + (v690_data * v726_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v734_n1 = 0; v734_n1 < 8; ++v734_n1) {
              int32_t v735_a = 0 + v734_n1;
              float v737_data = ir1[v734_n1];
              int32_t v738_a = 0 + v734_n1;
              r1[v734_n1] = v737_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v744_i1 = 0; v744_i1 < 8; ++v744_i1) {
              int32_t v745_a = 0 + v744_i1;
              float v747_data = r1[v744_i1];
              int32_t v754_a = v3_lead + (v744_i1 * 12);
              glb_m0[v754_a] = v747_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

