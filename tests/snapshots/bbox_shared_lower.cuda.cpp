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
          int32_t v7_lead = threadIdx.x % 16;
          if (v7_lead < 12) {
            int32_t v16_a = (v7_lead + 4) - 4;
            int32_t v25_a = (v7_lead + 4) - 4;
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
              int32_t v17_a = v9_i1 * 12;
              int32_t v18_a = v16_a + v17_a;
              float v28_data = __ldcg(&glb_m1[(v25_a + v17_a)]);
              r0[v9_i1] = v28_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(16, 28), (0, 8)] [(0, 16)]
          float ir1[8]{};
          if (v7_lead < 12) {
            float v38_data = r0[0];
            float v39_data = s0[0];
            float v41_data = ir1[0];
            ir1[0] = (v41_data + (v38_data * v39_data));
            float v44_data = s0[16];
            float v46_data = ir1[1];
            ir1[1] = (v46_data + (v38_data * v44_data));
            float v49_data = s0[32];
            float v51_data = ir1[2];
            ir1[2] = (v51_data + (v38_data * v49_data));
            float v54_data = s0[48];
            float v56_data = ir1[3];
            ir1[3] = (v56_data + (v38_data * v54_data));
            float v59_data = s0[64];
            float v61_data = ir1[4];
            ir1[4] = (v61_data + (v38_data * v59_data));
            float v64_data = s0[80];
            float v66_data = ir1[5];
            ir1[5] = (v66_data + (v38_data * v64_data));
            float v69_data = s0[96];
            float v71_data = ir1[6];
            ir1[6] = (v71_data + (v38_data * v69_data));
            float v74_data = s0[112];
            float v76_data = ir1[7];
            ir1[7] = (v76_data + (v38_data * v74_data));
          }
          if (v7_lead < 12) {
            float v82_data = r0[1];
            float v83_data = s0[1];
            float v85_data = ir1[0];
            ir1[0] = (v85_data + (v82_data * v83_data));
            float v88_data = s0[17];
            float v90_data = ir1[1];
            ir1[1] = (v90_data + (v82_data * v88_data));
            float v93_data = s0[33];
            float v95_data = ir1[2];
            ir1[2] = (v95_data + (v82_data * v93_data));
            float v98_data = s0[49];
            float v100_data = ir1[3];
            ir1[3] = (v100_data + (v82_data * v98_data));
            float v103_data = s0[65];
            float v105_data = ir1[4];
            ir1[4] = (v105_data + (v82_data * v103_data));
            float v108_data = s0[81];
            float v110_data = ir1[5];
            ir1[5] = (v110_data + (v82_data * v108_data));
            float v113_data = s0[97];
            float v115_data = ir1[6];
            ir1[6] = (v115_data + (v82_data * v113_data));
            float v118_data = s0[113];
            float v120_data = ir1[7];
            ir1[7] = (v120_data + (v82_data * v118_data));
          }
          if (v7_lead < 12) {
            float v126_data = r0[2];
            float v127_data = s0[2];
            float v129_data = ir1[0];
            ir1[0] = (v129_data + (v126_data * v127_data));
            float v132_data = s0[18];
            float v134_data = ir1[1];
            ir1[1] = (v134_data + (v126_data * v132_data));
            float v137_data = s0[34];
            float v139_data = ir1[2];
            ir1[2] = (v139_data + (v126_data * v137_data));
            float v142_data = s0[50];
            float v144_data = ir1[3];
            ir1[3] = (v144_data + (v126_data * v142_data));
            float v147_data = s0[66];
            float v149_data = ir1[4];
            ir1[4] = (v149_data + (v126_data * v147_data));
            float v152_data = s0[82];
            float v154_data = ir1[5];
            ir1[5] = (v154_data + (v126_data * v152_data));
            float v157_data = s0[98];
            float v159_data = ir1[6];
            ir1[6] = (v159_data + (v126_data * v157_data));
            float v162_data = s0[114];
            float v164_data = ir1[7];
            ir1[7] = (v164_data + (v126_data * v162_data));
          }
          if (v7_lead < 12) {
            float v170_data = r0[3];
            float v171_data = s0[3];
            float v173_data = ir1[0];
            ir1[0] = (v173_data + (v170_data * v171_data));
            float v176_data = s0[19];
            float v178_data = ir1[1];
            ir1[1] = (v178_data + (v170_data * v176_data));
            float v181_data = s0[35];
            float v183_data = ir1[2];
            ir1[2] = (v183_data + (v170_data * v181_data));
            float v186_data = s0[51];
            float v188_data = ir1[3];
            ir1[3] = (v188_data + (v170_data * v186_data));
            float v191_data = s0[67];
            float v193_data = ir1[4];
            ir1[4] = (v193_data + (v170_data * v191_data));
            float v196_data = s0[83];
            float v198_data = ir1[5];
            ir1[5] = (v198_data + (v170_data * v196_data));
            float v201_data = s0[99];
            float v203_data = ir1[6];
            ir1[6] = (v203_data + (v170_data * v201_data));
            float v206_data = s0[115];
            float v208_data = ir1[7];
            ir1[7] = (v208_data + (v170_data * v206_data));
          }
          if (v7_lead < 12) {
            float v214_data = r0[4];
            float v215_data = s0[4];
            float v217_data = ir1[0];
            ir1[0] = (v217_data + (v214_data * v215_data));
            float v220_data = s0[20];
            float v222_data = ir1[1];
            ir1[1] = (v222_data + (v214_data * v220_data));
            float v225_data = s0[36];
            float v227_data = ir1[2];
            ir1[2] = (v227_data + (v214_data * v225_data));
            float v230_data = s0[52];
            float v232_data = ir1[3];
            ir1[3] = (v232_data + (v214_data * v230_data));
            float v235_data = s0[68];
            float v237_data = ir1[4];
            ir1[4] = (v237_data + (v214_data * v235_data));
            float v240_data = s0[84];
            float v242_data = ir1[5];
            ir1[5] = (v242_data + (v214_data * v240_data));
            float v245_data = s0[100];
            float v247_data = ir1[6];
            ir1[6] = (v247_data + (v214_data * v245_data));
            float v250_data = s0[116];
            float v252_data = ir1[7];
            ir1[7] = (v252_data + (v214_data * v250_data));
          }
          if (v7_lead < 12) {
            float v258_data = r0[5];
            float v259_data = s0[5];
            float v261_data = ir1[0];
            ir1[0] = (v261_data + (v258_data * v259_data));
            float v264_data = s0[21];
            float v266_data = ir1[1];
            ir1[1] = (v266_data + (v258_data * v264_data));
            float v269_data = s0[37];
            float v271_data = ir1[2];
            ir1[2] = (v271_data + (v258_data * v269_data));
            float v274_data = s0[53];
            float v276_data = ir1[3];
            ir1[3] = (v276_data + (v258_data * v274_data));
            float v279_data = s0[69];
            float v281_data = ir1[4];
            ir1[4] = (v281_data + (v258_data * v279_data));
            float v284_data = s0[85];
            float v286_data = ir1[5];
            ir1[5] = (v286_data + (v258_data * v284_data));
            float v289_data = s0[101];
            float v291_data = ir1[6];
            ir1[6] = (v291_data + (v258_data * v289_data));
            float v294_data = s0[117];
            float v296_data = ir1[7];
            ir1[7] = (v296_data + (v258_data * v294_data));
          }
          if (v7_lead < 12) {
            float v302_data = r0[6];
            float v303_data = s0[6];
            float v305_data = ir1[0];
            ir1[0] = (v305_data + (v302_data * v303_data));
            float v308_data = s0[22];
            float v310_data = ir1[1];
            ir1[1] = (v310_data + (v302_data * v308_data));
            float v313_data = s0[38];
            float v315_data = ir1[2];
            ir1[2] = (v315_data + (v302_data * v313_data));
            float v318_data = s0[54];
            float v320_data = ir1[3];
            ir1[3] = (v320_data + (v302_data * v318_data));
            float v323_data = s0[70];
            float v325_data = ir1[4];
            ir1[4] = (v325_data + (v302_data * v323_data));
            float v328_data = s0[86];
            float v330_data = ir1[5];
            ir1[5] = (v330_data + (v302_data * v328_data));
            float v333_data = s0[102];
            float v335_data = ir1[6];
            ir1[6] = (v335_data + (v302_data * v333_data));
            float v338_data = s0[118];
            float v340_data = ir1[7];
            ir1[7] = (v340_data + (v302_data * v338_data));
          }
          if (v7_lead < 12) {
            float v346_data = r0[7];
            float v347_data = s0[7];
            float v349_data = ir1[0];
            ir1[0] = (v349_data + (v346_data * v347_data));
            float v352_data = s0[23];
            float v354_data = ir1[1];
            ir1[1] = (v354_data + (v346_data * v352_data));
            float v357_data = s0[39];
            float v359_data = ir1[2];
            ir1[2] = (v359_data + (v346_data * v357_data));
            float v362_data = s0[55];
            float v364_data = ir1[3];
            ir1[3] = (v364_data + (v346_data * v362_data));
            float v367_data = s0[71];
            float v369_data = ir1[4];
            ir1[4] = (v369_data + (v346_data * v367_data));
            float v372_data = s0[87];
            float v374_data = ir1[5];
            ir1[5] = (v374_data + (v346_data * v372_data));
            float v377_data = s0[103];
            float v379_data = ir1[6];
            ir1[6] = (v379_data + (v346_data * v377_data));
            float v382_data = s0[119];
            float v384_data = ir1[7];
            ir1[7] = (v384_data + (v346_data * v382_data));
          }
          if (v7_lead < 12) {
            float v390_data = r0[8];
            float v391_data = s0[8];
            float v393_data = ir1[0];
            ir1[0] = (v393_data + (v390_data * v391_data));
            float v396_data = s0[24];
            float v398_data = ir1[1];
            ir1[1] = (v398_data + (v390_data * v396_data));
            float v401_data = s0[40];
            float v403_data = ir1[2];
            ir1[2] = (v403_data + (v390_data * v401_data));
            float v406_data = s0[56];
            float v408_data = ir1[3];
            ir1[3] = (v408_data + (v390_data * v406_data));
            float v411_data = s0[72];
            float v413_data = ir1[4];
            ir1[4] = (v413_data + (v390_data * v411_data));
            float v416_data = s0[88];
            float v418_data = ir1[5];
            ir1[5] = (v418_data + (v390_data * v416_data));
            float v421_data = s0[104];
            float v423_data = ir1[6];
            ir1[6] = (v423_data + (v390_data * v421_data));
            float v426_data = s0[120];
            float v428_data = ir1[7];
            ir1[7] = (v428_data + (v390_data * v426_data));
          }
          if (v7_lead < 12) {
            float v434_data = r0[9];
            float v435_data = s0[9];
            float v437_data = ir1[0];
            ir1[0] = (v437_data + (v434_data * v435_data));
            float v440_data = s0[25];
            float v442_data = ir1[1];
            ir1[1] = (v442_data + (v434_data * v440_data));
            float v445_data = s0[41];
            float v447_data = ir1[2];
            ir1[2] = (v447_data + (v434_data * v445_data));
            float v450_data = s0[57];
            float v452_data = ir1[3];
            ir1[3] = (v452_data + (v434_data * v450_data));
            float v455_data = s0[73];
            float v457_data = ir1[4];
            ir1[4] = (v457_data + (v434_data * v455_data));
            float v460_data = s0[89];
            float v462_data = ir1[5];
            ir1[5] = (v462_data + (v434_data * v460_data));
            float v465_data = s0[105];
            float v467_data = ir1[6];
            ir1[6] = (v467_data + (v434_data * v465_data));
            float v470_data = s0[121];
            float v472_data = ir1[7];
            ir1[7] = (v472_data + (v434_data * v470_data));
          }
          if (v7_lead < 12) {
            float v478_data = r0[10];
            float v479_data = s0[10];
            float v481_data = ir1[0];
            ir1[0] = (v481_data + (v478_data * v479_data));
            float v484_data = s0[26];
            float v486_data = ir1[1];
            ir1[1] = (v486_data + (v478_data * v484_data));
            float v489_data = s0[42];
            float v491_data = ir1[2];
            ir1[2] = (v491_data + (v478_data * v489_data));
            float v494_data = s0[58];
            float v496_data = ir1[3];
            ir1[3] = (v496_data + (v478_data * v494_data));
            float v499_data = s0[74];
            float v501_data = ir1[4];
            ir1[4] = (v501_data + (v478_data * v499_data));
            float v504_data = s0[90];
            float v506_data = ir1[5];
            ir1[5] = (v506_data + (v478_data * v504_data));
            float v509_data = s0[106];
            float v511_data = ir1[6];
            ir1[6] = (v511_data + (v478_data * v509_data));
            float v514_data = s0[122];
            float v516_data = ir1[7];
            ir1[7] = (v516_data + (v478_data * v514_data));
          }
          if (v7_lead < 12) {
            float v522_data = r0[11];
            float v523_data = s0[11];
            float v525_data = ir1[0];
            ir1[0] = (v525_data + (v522_data * v523_data));
            float v528_data = s0[27];
            float v530_data = ir1[1];
            ir1[1] = (v530_data + (v522_data * v528_data));
            float v533_data = s0[43];
            float v535_data = ir1[2];
            ir1[2] = (v535_data + (v522_data * v533_data));
            float v538_data = s0[59];
            float v540_data = ir1[3];
            ir1[3] = (v540_data + (v522_data * v538_data));
            float v543_data = s0[75];
            float v545_data = ir1[4];
            ir1[4] = (v545_data + (v522_data * v543_data));
            float v548_data = s0[91];
            float v550_data = ir1[5];
            ir1[5] = (v550_data + (v522_data * v548_data));
            float v553_data = s0[107];
            float v555_data = ir1[6];
            ir1[6] = (v555_data + (v522_data * v553_data));
            float v558_data = s0[123];
            float v560_data = ir1[7];
            ir1[7] = (v560_data + (v522_data * v558_data));
          }
          if (v7_lead < 12) {
            float v566_data = r0[12];
            float v567_data = s0[12];
            float v569_data = ir1[0];
            ir1[0] = (v569_data + (v566_data * v567_data));
            float v572_data = s0[28];
            float v574_data = ir1[1];
            ir1[1] = (v574_data + (v566_data * v572_data));
            float v577_data = s0[44];
            float v579_data = ir1[2];
            ir1[2] = (v579_data + (v566_data * v577_data));
            float v582_data = s0[60];
            float v584_data = ir1[3];
            ir1[3] = (v584_data + (v566_data * v582_data));
            float v587_data = s0[76];
            float v589_data = ir1[4];
            ir1[4] = (v589_data + (v566_data * v587_data));
            float v592_data = s0[92];
            float v594_data = ir1[5];
            ir1[5] = (v594_data + (v566_data * v592_data));
            float v597_data = s0[108];
            float v599_data = ir1[6];
            ir1[6] = (v599_data + (v566_data * v597_data));
            float v602_data = s0[124];
            float v604_data = ir1[7];
            ir1[7] = (v604_data + (v566_data * v602_data));
          }
          if (v7_lead < 12) {
            float v610_data = r0[13];
            float v611_data = s0[13];
            float v613_data = ir1[0];
            ir1[0] = (v613_data + (v610_data * v611_data));
            float v616_data = s0[29];
            float v618_data = ir1[1];
            ir1[1] = (v618_data + (v610_data * v616_data));
            float v621_data = s0[45];
            float v623_data = ir1[2];
            ir1[2] = (v623_data + (v610_data * v621_data));
            float v626_data = s0[61];
            float v628_data = ir1[3];
            ir1[3] = (v628_data + (v610_data * v626_data));
            float v631_data = s0[77];
            float v633_data = ir1[4];
            ir1[4] = (v633_data + (v610_data * v631_data));
            float v636_data = s0[93];
            float v638_data = ir1[5];
            ir1[5] = (v638_data + (v610_data * v636_data));
            float v641_data = s0[109];
            float v643_data = ir1[6];
            ir1[6] = (v643_data + (v610_data * v641_data));
            float v646_data = s0[125];
            float v648_data = ir1[7];
            ir1[7] = (v648_data + (v610_data * v646_data));
          }
          if (v7_lead < 12) {
            float v654_data = r0[14];
            float v655_data = s0[14];
            float v657_data = ir1[0];
            ir1[0] = (v657_data + (v654_data * v655_data));
            float v660_data = s0[30];
            float v662_data = ir1[1];
            ir1[1] = (v662_data + (v654_data * v660_data));
            float v665_data = s0[46];
            float v667_data = ir1[2];
            ir1[2] = (v667_data + (v654_data * v665_data));
            float v670_data = s0[62];
            float v672_data = ir1[3];
            ir1[3] = (v672_data + (v654_data * v670_data));
            float v675_data = s0[78];
            float v677_data = ir1[4];
            ir1[4] = (v677_data + (v654_data * v675_data));
            float v680_data = s0[94];
            float v682_data = ir1[5];
            ir1[5] = (v682_data + (v654_data * v680_data));
            float v685_data = s0[110];
            float v687_data = ir1[6];
            ir1[6] = (v687_data + (v654_data * v685_data));
            float v690_data = s0[126];
            float v692_data = ir1[7];
            ir1[7] = (v692_data + (v654_data * v690_data));
          }
          if (v7_lead < 12) {
            float v698_data = r0[15];
            float v699_data = s0[15];
            float v701_data = ir1[0];
            ir1[0] = (v701_data + (v698_data * v699_data));
            float v704_data = s0[31];
            float v706_data = ir1[1];
            ir1[1] = (v706_data + (v698_data * v704_data));
            float v709_data = s0[47];
            float v711_data = ir1[2];
            ir1[2] = (v711_data + (v698_data * v709_data));
            float v714_data = s0[63];
            float v716_data = ir1[3];
            ir1[3] = (v716_data + (v698_data * v714_data));
            float v719_data = s0[79];
            float v721_data = ir1[4];
            ir1[4] = (v721_data + (v698_data * v719_data));
            float v724_data = s0[95];
            float v726_data = ir1[5];
            ir1[5] = (v726_data + (v698_data * v724_data));
            float v729_data = s0[111];
            float v731_data = ir1[6];
            ir1[6] = (v731_data + (v698_data * v729_data));
            float v734_data = s0[127];
            float v736_data = ir1[7];
            ir1[7] = (v736_data + (v698_data * v734_data));
          }
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v742_n1 = 0; v742_n1 < 8; ++v742_n1) {
              int32_t v743_a = 0 + v742_n1;
              float v745_data = ir1[v742_n1];
              r1[v742_n1] = v745_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v7_lead < 12) {
            int32_t v761_a = ((v7_lead + 16_i32) + -12) - 4;
            #pragma unroll
            for (int32_t v751_i1 = 0; v751_i1 < 8; ++v751_i1) {
              int32_t v752_a = 0 + v751_i1;
              float v754_data = r1[v751_i1];
              glb_m0[(v761_a + (v751_i1 * 12))] = v754_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

