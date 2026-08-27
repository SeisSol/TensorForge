// === base name ===
kernel_4b59b6f027

// === header ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4b59b6f027, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_4b59b6f027, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_4b59b6f027<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(12×8) {4..16}×{0..8} strided
    // m1 16×16(12×16) {4..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(12×8) {4..16}×{0..8} strided({4..16}×{0..8})[0, 1] = m1 16×16(12×16) {4..16}×{0..16} strided({4..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
            int32_t v12_a = (v3_lead + 4) - 4;
            int32_t v21_a = (v3_lead + 4) - 4;
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v13_a = v5_i1 * 12;
              int32_t v14_a = v12_a + v13_a;
              float v24_data = __ldcg(&glb_m1[(v21_a + v13_a)]);
              int32_t v25_a = 0 + v5_i1;
              r0[v25_a] = v24_data;
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
          // [(16, 28), (0, 8)] [(0, 16)]
          float ir1[8]{};
          if (v3_lead < 12) {
            float v33_data = r0[0];
            float v34_data = s0[0];
            float v36_data = ir1[0];
            ir1[0] = (v36_data + (v33_data * v34_data));
            float v39_data = s0[16];
            float v41_data = ir1[1];
            ir1[1] = (v41_data + (v33_data * v39_data));
            float v44_data = s0[32];
            float v46_data = ir1[2];
            ir1[2] = (v46_data + (v33_data * v44_data));
            float v49_data = s0[48];
            float v51_data = ir1[3];
            ir1[3] = (v51_data + (v33_data * v49_data));
            float v54_data = s0[64];
            float v56_data = ir1[4];
            ir1[4] = (v56_data + (v33_data * v54_data));
            float v59_data = s0[80];
            float v61_data = ir1[5];
            ir1[5] = (v61_data + (v33_data * v59_data));
            float v64_data = s0[96];
            float v66_data = ir1[6];
            ir1[6] = (v66_data + (v33_data * v64_data));
            float v69_data = s0[112];
            float v71_data = ir1[7];
            ir1[7] = (v71_data + (v33_data * v69_data));
          }
          if (v3_lead < 12) {
            float v77_data = r0[1];
            float v78_data = s0[1];
            float v80_data = ir1[0];
            ir1[0] = (v80_data + (v77_data * v78_data));
            float v83_data = s0[17];
            float v85_data = ir1[1];
            ir1[1] = (v85_data + (v77_data * v83_data));
            float v88_data = s0[33];
            float v90_data = ir1[2];
            ir1[2] = (v90_data + (v77_data * v88_data));
            float v93_data = s0[49];
            float v95_data = ir1[3];
            ir1[3] = (v95_data + (v77_data * v93_data));
            float v98_data = s0[65];
            float v100_data = ir1[4];
            ir1[4] = (v100_data + (v77_data * v98_data));
            float v103_data = s0[81];
            float v105_data = ir1[5];
            ir1[5] = (v105_data + (v77_data * v103_data));
            float v108_data = s0[97];
            float v110_data = ir1[6];
            ir1[6] = (v110_data + (v77_data * v108_data));
            float v113_data = s0[113];
            float v115_data = ir1[7];
            ir1[7] = (v115_data + (v77_data * v113_data));
          }
          if (v3_lead < 12) {
            float v121_data = r0[2];
            float v122_data = s0[2];
            float v124_data = ir1[0];
            ir1[0] = (v124_data + (v121_data * v122_data));
            float v127_data = s0[18];
            float v129_data = ir1[1];
            ir1[1] = (v129_data + (v121_data * v127_data));
            float v132_data = s0[34];
            float v134_data = ir1[2];
            ir1[2] = (v134_data + (v121_data * v132_data));
            float v137_data = s0[50];
            float v139_data = ir1[3];
            ir1[3] = (v139_data + (v121_data * v137_data));
            float v142_data = s0[66];
            float v144_data = ir1[4];
            ir1[4] = (v144_data + (v121_data * v142_data));
            float v147_data = s0[82];
            float v149_data = ir1[5];
            ir1[5] = (v149_data + (v121_data * v147_data));
            float v152_data = s0[98];
            float v154_data = ir1[6];
            ir1[6] = (v154_data + (v121_data * v152_data));
            float v157_data = s0[114];
            float v159_data = ir1[7];
            ir1[7] = (v159_data + (v121_data * v157_data));
          }
          if (v3_lead < 12) {
            float v165_data = r0[3];
            float v166_data = s0[3];
            float v168_data = ir1[0];
            ir1[0] = (v168_data + (v165_data * v166_data));
            float v171_data = s0[19];
            float v173_data = ir1[1];
            ir1[1] = (v173_data + (v165_data * v171_data));
            float v176_data = s0[35];
            float v178_data = ir1[2];
            ir1[2] = (v178_data + (v165_data * v176_data));
            float v181_data = s0[51];
            float v183_data = ir1[3];
            ir1[3] = (v183_data + (v165_data * v181_data));
            float v186_data = s0[67];
            float v188_data = ir1[4];
            ir1[4] = (v188_data + (v165_data * v186_data));
            float v191_data = s0[83];
            float v193_data = ir1[5];
            ir1[5] = (v193_data + (v165_data * v191_data));
            float v196_data = s0[99];
            float v198_data = ir1[6];
            ir1[6] = (v198_data + (v165_data * v196_data));
            float v201_data = s0[115];
            float v203_data = ir1[7];
            ir1[7] = (v203_data + (v165_data * v201_data));
          }
          if (v3_lead < 12) {
            float v209_data = r0[4];
            float v210_data = s0[4];
            float v212_data = ir1[0];
            ir1[0] = (v212_data + (v209_data * v210_data));
            float v215_data = s0[20];
            float v217_data = ir1[1];
            ir1[1] = (v217_data + (v209_data * v215_data));
            float v220_data = s0[36];
            float v222_data = ir1[2];
            ir1[2] = (v222_data + (v209_data * v220_data));
            float v225_data = s0[52];
            float v227_data = ir1[3];
            ir1[3] = (v227_data + (v209_data * v225_data));
            float v230_data = s0[68];
            float v232_data = ir1[4];
            ir1[4] = (v232_data + (v209_data * v230_data));
            float v235_data = s0[84];
            float v237_data = ir1[5];
            ir1[5] = (v237_data + (v209_data * v235_data));
            float v240_data = s0[100];
            float v242_data = ir1[6];
            ir1[6] = (v242_data + (v209_data * v240_data));
            float v245_data = s0[116];
            float v247_data = ir1[7];
            ir1[7] = (v247_data + (v209_data * v245_data));
          }
          if (v3_lead < 12) {
            float v253_data = r0[5];
            float v254_data = s0[5];
            float v256_data = ir1[0];
            ir1[0] = (v256_data + (v253_data * v254_data));
            float v259_data = s0[21];
            float v261_data = ir1[1];
            ir1[1] = (v261_data + (v253_data * v259_data));
            float v264_data = s0[37];
            float v266_data = ir1[2];
            ir1[2] = (v266_data + (v253_data * v264_data));
            float v269_data = s0[53];
            float v271_data = ir1[3];
            ir1[3] = (v271_data + (v253_data * v269_data));
            float v274_data = s0[69];
            float v276_data = ir1[4];
            ir1[4] = (v276_data + (v253_data * v274_data));
            float v279_data = s0[85];
            float v281_data = ir1[5];
            ir1[5] = (v281_data + (v253_data * v279_data));
            float v284_data = s0[101];
            float v286_data = ir1[6];
            ir1[6] = (v286_data + (v253_data * v284_data));
            float v289_data = s0[117];
            float v291_data = ir1[7];
            ir1[7] = (v291_data + (v253_data * v289_data));
          }
          if (v3_lead < 12) {
            float v297_data = r0[6];
            float v298_data = s0[6];
            float v300_data = ir1[0];
            ir1[0] = (v300_data + (v297_data * v298_data));
            float v303_data = s0[22];
            float v305_data = ir1[1];
            ir1[1] = (v305_data + (v297_data * v303_data));
            float v308_data = s0[38];
            float v310_data = ir1[2];
            ir1[2] = (v310_data + (v297_data * v308_data));
            float v313_data = s0[54];
            float v315_data = ir1[3];
            ir1[3] = (v315_data + (v297_data * v313_data));
            float v318_data = s0[70];
            float v320_data = ir1[4];
            ir1[4] = (v320_data + (v297_data * v318_data));
            float v323_data = s0[86];
            float v325_data = ir1[5];
            ir1[5] = (v325_data + (v297_data * v323_data));
            float v328_data = s0[102];
            float v330_data = ir1[6];
            ir1[6] = (v330_data + (v297_data * v328_data));
            float v333_data = s0[118];
            float v335_data = ir1[7];
            ir1[7] = (v335_data + (v297_data * v333_data));
          }
          if (v3_lead < 12) {
            float v341_data = r0[7];
            float v342_data = s0[7];
            float v344_data = ir1[0];
            ir1[0] = (v344_data + (v341_data * v342_data));
            float v347_data = s0[23];
            float v349_data = ir1[1];
            ir1[1] = (v349_data + (v341_data * v347_data));
            float v352_data = s0[39];
            float v354_data = ir1[2];
            ir1[2] = (v354_data + (v341_data * v352_data));
            float v357_data = s0[55];
            float v359_data = ir1[3];
            ir1[3] = (v359_data + (v341_data * v357_data));
            float v362_data = s0[71];
            float v364_data = ir1[4];
            ir1[4] = (v364_data + (v341_data * v362_data));
            float v367_data = s0[87];
            float v369_data = ir1[5];
            ir1[5] = (v369_data + (v341_data * v367_data));
            float v372_data = s0[103];
            float v374_data = ir1[6];
            ir1[6] = (v374_data + (v341_data * v372_data));
            float v377_data = s0[119];
            float v379_data = ir1[7];
            ir1[7] = (v379_data + (v341_data * v377_data));
          }
          if (v3_lead < 12) {
            float v385_data = r0[8];
            float v386_data = s0[8];
            float v388_data = ir1[0];
            ir1[0] = (v388_data + (v385_data * v386_data));
            float v391_data = s0[24];
            float v393_data = ir1[1];
            ir1[1] = (v393_data + (v385_data * v391_data));
            float v396_data = s0[40];
            float v398_data = ir1[2];
            ir1[2] = (v398_data + (v385_data * v396_data));
            float v401_data = s0[56];
            float v403_data = ir1[3];
            ir1[3] = (v403_data + (v385_data * v401_data));
            float v406_data = s0[72];
            float v408_data = ir1[4];
            ir1[4] = (v408_data + (v385_data * v406_data));
            float v411_data = s0[88];
            float v413_data = ir1[5];
            ir1[5] = (v413_data + (v385_data * v411_data));
            float v416_data = s0[104];
            float v418_data = ir1[6];
            ir1[6] = (v418_data + (v385_data * v416_data));
            float v421_data = s0[120];
            float v423_data = ir1[7];
            ir1[7] = (v423_data + (v385_data * v421_data));
          }
          if (v3_lead < 12) {
            float v429_data = r0[9];
            float v430_data = s0[9];
            float v432_data = ir1[0];
            ir1[0] = (v432_data + (v429_data * v430_data));
            float v435_data = s0[25];
            float v437_data = ir1[1];
            ir1[1] = (v437_data + (v429_data * v435_data));
            float v440_data = s0[41];
            float v442_data = ir1[2];
            ir1[2] = (v442_data + (v429_data * v440_data));
            float v445_data = s0[57];
            float v447_data = ir1[3];
            ir1[3] = (v447_data + (v429_data * v445_data));
            float v450_data = s0[73];
            float v452_data = ir1[4];
            ir1[4] = (v452_data + (v429_data * v450_data));
            float v455_data = s0[89];
            float v457_data = ir1[5];
            ir1[5] = (v457_data + (v429_data * v455_data));
            float v460_data = s0[105];
            float v462_data = ir1[6];
            ir1[6] = (v462_data + (v429_data * v460_data));
            float v465_data = s0[121];
            float v467_data = ir1[7];
            ir1[7] = (v467_data + (v429_data * v465_data));
          }
          if (v3_lead < 12) {
            float v473_data = r0[10];
            float v474_data = s0[10];
            float v476_data = ir1[0];
            ir1[0] = (v476_data + (v473_data * v474_data));
            float v479_data = s0[26];
            float v481_data = ir1[1];
            ir1[1] = (v481_data + (v473_data * v479_data));
            float v484_data = s0[42];
            float v486_data = ir1[2];
            ir1[2] = (v486_data + (v473_data * v484_data));
            float v489_data = s0[58];
            float v491_data = ir1[3];
            ir1[3] = (v491_data + (v473_data * v489_data));
            float v494_data = s0[74];
            float v496_data = ir1[4];
            ir1[4] = (v496_data + (v473_data * v494_data));
            float v499_data = s0[90];
            float v501_data = ir1[5];
            ir1[5] = (v501_data + (v473_data * v499_data));
            float v504_data = s0[106];
            float v506_data = ir1[6];
            ir1[6] = (v506_data + (v473_data * v504_data));
            float v509_data = s0[122];
            float v511_data = ir1[7];
            ir1[7] = (v511_data + (v473_data * v509_data));
          }
          if (v3_lead < 12) {
            float v517_data = r0[11];
            float v518_data = s0[11];
            float v520_data = ir1[0];
            ir1[0] = (v520_data + (v517_data * v518_data));
            float v523_data = s0[27];
            float v525_data = ir1[1];
            ir1[1] = (v525_data + (v517_data * v523_data));
            float v528_data = s0[43];
            float v530_data = ir1[2];
            ir1[2] = (v530_data + (v517_data * v528_data));
            float v533_data = s0[59];
            float v535_data = ir1[3];
            ir1[3] = (v535_data + (v517_data * v533_data));
            float v538_data = s0[75];
            float v540_data = ir1[4];
            ir1[4] = (v540_data + (v517_data * v538_data));
            float v543_data = s0[91];
            float v545_data = ir1[5];
            ir1[5] = (v545_data + (v517_data * v543_data));
            float v548_data = s0[107];
            float v550_data = ir1[6];
            ir1[6] = (v550_data + (v517_data * v548_data));
            float v553_data = s0[123];
            float v555_data = ir1[7];
            ir1[7] = (v555_data + (v517_data * v553_data));
          }
          if (v3_lead < 12) {
            float v561_data = r0[12];
            float v562_data = s0[12];
            float v564_data = ir1[0];
            ir1[0] = (v564_data + (v561_data * v562_data));
            float v567_data = s0[28];
            float v569_data = ir1[1];
            ir1[1] = (v569_data + (v561_data * v567_data));
            float v572_data = s0[44];
            float v574_data = ir1[2];
            ir1[2] = (v574_data + (v561_data * v572_data));
            float v577_data = s0[60];
            float v579_data = ir1[3];
            ir1[3] = (v579_data + (v561_data * v577_data));
            float v582_data = s0[76];
            float v584_data = ir1[4];
            ir1[4] = (v584_data + (v561_data * v582_data));
            float v587_data = s0[92];
            float v589_data = ir1[5];
            ir1[5] = (v589_data + (v561_data * v587_data));
            float v592_data = s0[108];
            float v594_data = ir1[6];
            ir1[6] = (v594_data + (v561_data * v592_data));
            float v597_data = s0[124];
            float v599_data = ir1[7];
            ir1[7] = (v599_data + (v561_data * v597_data));
          }
          if (v3_lead < 12) {
            float v605_data = r0[13];
            float v606_data = s0[13];
            float v608_data = ir1[0];
            ir1[0] = (v608_data + (v605_data * v606_data));
            float v611_data = s0[29];
            float v613_data = ir1[1];
            ir1[1] = (v613_data + (v605_data * v611_data));
            float v616_data = s0[45];
            float v618_data = ir1[2];
            ir1[2] = (v618_data + (v605_data * v616_data));
            float v621_data = s0[61];
            float v623_data = ir1[3];
            ir1[3] = (v623_data + (v605_data * v621_data));
            float v626_data = s0[77];
            float v628_data = ir1[4];
            ir1[4] = (v628_data + (v605_data * v626_data));
            float v631_data = s0[93];
            float v633_data = ir1[5];
            ir1[5] = (v633_data + (v605_data * v631_data));
            float v636_data = s0[109];
            float v638_data = ir1[6];
            ir1[6] = (v638_data + (v605_data * v636_data));
            float v641_data = s0[125];
            float v643_data = ir1[7];
            ir1[7] = (v643_data + (v605_data * v641_data));
          }
          if (v3_lead < 12) {
            float v649_data = r0[14];
            float v650_data = s0[14];
            float v652_data = ir1[0];
            ir1[0] = (v652_data + (v649_data * v650_data));
            float v655_data = s0[30];
            float v657_data = ir1[1];
            ir1[1] = (v657_data + (v649_data * v655_data));
            float v660_data = s0[46];
            float v662_data = ir1[2];
            ir1[2] = (v662_data + (v649_data * v660_data));
            float v665_data = s0[62];
            float v667_data = ir1[3];
            ir1[3] = (v667_data + (v649_data * v665_data));
            float v670_data = s0[78];
            float v672_data = ir1[4];
            ir1[4] = (v672_data + (v649_data * v670_data));
            float v675_data = s0[94];
            float v677_data = ir1[5];
            ir1[5] = (v677_data + (v649_data * v675_data));
            float v680_data = s0[110];
            float v682_data = ir1[6];
            ir1[6] = (v682_data + (v649_data * v680_data));
            float v685_data = s0[126];
            float v687_data = ir1[7];
            ir1[7] = (v687_data + (v649_data * v685_data));
          }
          if (v3_lead < 12) {
            float v693_data = r0[15];
            float v694_data = s0[15];
            float v696_data = ir1[0];
            ir1[0] = (v696_data + (v693_data * v694_data));
            float v699_data = s0[31];
            float v701_data = ir1[1];
            ir1[1] = (v701_data + (v693_data * v699_data));
            float v704_data = s0[47];
            float v706_data = ir1[2];
            ir1[2] = (v706_data + (v693_data * v704_data));
            float v709_data = s0[63];
            float v711_data = ir1[3];
            ir1[3] = (v711_data + (v693_data * v709_data));
            float v714_data = s0[79];
            float v716_data = ir1[4];
            ir1[4] = (v716_data + (v693_data * v714_data));
            float v719_data = s0[95];
            float v721_data = ir1[5];
            ir1[5] = (v721_data + (v693_data * v719_data));
            float v724_data = s0[111];
            float v726_data = ir1[6];
            ir1[6] = (v726_data + (v693_data * v724_data));
            float v729_data = s0[127];
            float v731_data = ir1[7];
            ir1[7] = (v731_data + (v693_data * v729_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v737_n1 = 0; v737_n1 < 8; ++v737_n1) {
              int32_t v738_a = 0 + v737_n1;
              float v740_data = ir1[v737_n1];
              int32_t v741_a = 0 + v737_n1;
              r1[v737_n1] = v740_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v3_lead < 12) {
            int32_t v757_a = ((v3_lead + 16_i32) + -12) - 4;
            #pragma unroll
            for (int32_t v747_i1 = 0; v747_i1 < 8; ++v747_i1) {
              int32_t v748_a = 0 + v747_i1;
              float v750_data = r1[v747_i1];
              int32_t v759_a = v757_a + (v747_i1 * 12);
              glb_m0[v759_a] = v750_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

