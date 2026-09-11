// === base name ===
kernel_745f24ba73e36ed7

// === header ===
void launcher_kernel_745f24ba73e36ed7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_745f24ba73e36ed7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_745f24ba73e36ed7, block.x * block.y * block.z, 1152 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_745f24ba73e36ed7, cudaFuncAttributeMaxDynamicSharedMemorySize, 1152 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_745f24ba73e36ed7<<<grid,block,1152 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_745f24ba73e36ed7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[128];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 1024 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v26_off = (v18_lead + (v19_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v20_i1 = 8; v20_i1 < 24; ++v20_i1) {
              float v29_data = __ldcg(&glb_m1[(v26_off + (v20_i1 * 32))]);
              r0[(v19_i0 + (v20_i1 - 8))] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          float ir1[8]{};
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
          float v81_data = r0[1];
          float v82_data = s0[1];
          float v84_data = ir1[0];
          ir1[0] = (v84_data + (v81_data * v82_data));
          float v87_data = s0[17];
          float v89_data = ir1[1];
          ir1[1] = (v89_data + (v81_data * v87_data));
          float v92_data = s0[33];
          float v94_data = ir1[2];
          ir1[2] = (v94_data + (v81_data * v92_data));
          float v97_data = s0[49];
          float v99_data = ir1[3];
          ir1[3] = (v99_data + (v81_data * v97_data));
          float v102_data = s0[65];
          float v104_data = ir1[4];
          ir1[4] = (v104_data + (v81_data * v102_data));
          float v107_data = s0[81];
          float v109_data = ir1[5];
          ir1[5] = (v109_data + (v81_data * v107_data));
          float v112_data = s0[97];
          float v114_data = ir1[6];
          ir1[6] = (v114_data + (v81_data * v112_data));
          float v117_data = s0[113];
          float v119_data = ir1[7];
          ir1[7] = (v119_data + (v81_data * v117_data));
          float v124_data = r0[2];
          float v125_data = s0[2];
          float v127_data = ir1[0];
          ir1[0] = (v127_data + (v124_data * v125_data));
          float v130_data = s0[18];
          float v132_data = ir1[1];
          ir1[1] = (v132_data + (v124_data * v130_data));
          float v135_data = s0[34];
          float v137_data = ir1[2];
          ir1[2] = (v137_data + (v124_data * v135_data));
          float v140_data = s0[50];
          float v142_data = ir1[3];
          ir1[3] = (v142_data + (v124_data * v140_data));
          float v145_data = s0[66];
          float v147_data = ir1[4];
          ir1[4] = (v147_data + (v124_data * v145_data));
          float v150_data = s0[82];
          float v152_data = ir1[5];
          ir1[5] = (v152_data + (v124_data * v150_data));
          float v155_data = s0[98];
          float v157_data = ir1[6];
          ir1[6] = (v157_data + (v124_data * v155_data));
          float v160_data = s0[114];
          float v162_data = ir1[7];
          ir1[7] = (v162_data + (v124_data * v160_data));
          float v167_data = r0[3];
          float v168_data = s0[3];
          float v170_data = ir1[0];
          ir1[0] = (v170_data + (v167_data * v168_data));
          float v173_data = s0[19];
          float v175_data = ir1[1];
          ir1[1] = (v175_data + (v167_data * v173_data));
          float v178_data = s0[35];
          float v180_data = ir1[2];
          ir1[2] = (v180_data + (v167_data * v178_data));
          float v183_data = s0[51];
          float v185_data = ir1[3];
          ir1[3] = (v185_data + (v167_data * v183_data));
          float v188_data = s0[67];
          float v190_data = ir1[4];
          ir1[4] = (v190_data + (v167_data * v188_data));
          float v193_data = s0[83];
          float v195_data = ir1[5];
          ir1[5] = (v195_data + (v167_data * v193_data));
          float v198_data = s0[99];
          float v200_data = ir1[6];
          ir1[6] = (v200_data + (v167_data * v198_data));
          float v203_data = s0[115];
          float v205_data = ir1[7];
          ir1[7] = (v205_data + (v167_data * v203_data));
          float v210_data = r0[4];
          float v211_data = s0[4];
          float v213_data = ir1[0];
          ir1[0] = (v213_data + (v210_data * v211_data));
          float v216_data = s0[20];
          float v218_data = ir1[1];
          ir1[1] = (v218_data + (v210_data * v216_data));
          float v221_data = s0[36];
          float v223_data = ir1[2];
          ir1[2] = (v223_data + (v210_data * v221_data));
          float v226_data = s0[52];
          float v228_data = ir1[3];
          ir1[3] = (v228_data + (v210_data * v226_data));
          float v231_data = s0[68];
          float v233_data = ir1[4];
          ir1[4] = (v233_data + (v210_data * v231_data));
          float v236_data = s0[84];
          float v238_data = ir1[5];
          ir1[5] = (v238_data + (v210_data * v236_data));
          float v241_data = s0[100];
          float v243_data = ir1[6];
          ir1[6] = (v243_data + (v210_data * v241_data));
          float v246_data = s0[116];
          float v248_data = ir1[7];
          ir1[7] = (v248_data + (v210_data * v246_data));
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
          float v296_data = r0[6];
          float v297_data = s0[6];
          float v299_data = ir1[0];
          ir1[0] = (v299_data + (v296_data * v297_data));
          float v302_data = s0[22];
          float v304_data = ir1[1];
          ir1[1] = (v304_data + (v296_data * v302_data));
          float v307_data = s0[38];
          float v309_data = ir1[2];
          ir1[2] = (v309_data + (v296_data * v307_data));
          float v312_data = s0[54];
          float v314_data = ir1[3];
          ir1[3] = (v314_data + (v296_data * v312_data));
          float v317_data = s0[70];
          float v319_data = ir1[4];
          ir1[4] = (v319_data + (v296_data * v317_data));
          float v322_data = s0[86];
          float v324_data = ir1[5];
          ir1[5] = (v324_data + (v296_data * v322_data));
          float v327_data = s0[102];
          float v329_data = ir1[6];
          ir1[6] = (v329_data + (v296_data * v327_data));
          float v332_data = s0[118];
          float v334_data = ir1[7];
          ir1[7] = (v334_data + (v296_data * v332_data));
          float v339_data = r0[7];
          float v340_data = s0[7];
          float v342_data = ir1[0];
          ir1[0] = (v342_data + (v339_data * v340_data));
          float v345_data = s0[23];
          float v347_data = ir1[1];
          ir1[1] = (v347_data + (v339_data * v345_data));
          float v350_data = s0[39];
          float v352_data = ir1[2];
          ir1[2] = (v352_data + (v339_data * v350_data));
          float v355_data = s0[55];
          float v357_data = ir1[3];
          ir1[3] = (v357_data + (v339_data * v355_data));
          float v360_data = s0[71];
          float v362_data = ir1[4];
          ir1[4] = (v362_data + (v339_data * v360_data));
          float v365_data = s0[87];
          float v367_data = ir1[5];
          ir1[5] = (v367_data + (v339_data * v365_data));
          float v370_data = s0[103];
          float v372_data = ir1[6];
          ir1[6] = (v372_data + (v339_data * v370_data));
          float v375_data = s0[119];
          float v377_data = ir1[7];
          ir1[7] = (v377_data + (v339_data * v375_data));
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
          float v425_data = r0[9];
          float v426_data = s0[9];
          float v428_data = ir1[0];
          ir1[0] = (v428_data + (v425_data * v426_data));
          float v431_data = s0[25];
          float v433_data = ir1[1];
          ir1[1] = (v433_data + (v425_data * v431_data));
          float v436_data = s0[41];
          float v438_data = ir1[2];
          ir1[2] = (v438_data + (v425_data * v436_data));
          float v441_data = s0[57];
          float v443_data = ir1[3];
          ir1[3] = (v443_data + (v425_data * v441_data));
          float v446_data = s0[73];
          float v448_data = ir1[4];
          ir1[4] = (v448_data + (v425_data * v446_data));
          float v451_data = s0[89];
          float v453_data = ir1[5];
          ir1[5] = (v453_data + (v425_data * v451_data));
          float v456_data = s0[105];
          float v458_data = ir1[6];
          ir1[6] = (v458_data + (v425_data * v456_data));
          float v461_data = s0[121];
          float v463_data = ir1[7];
          ir1[7] = (v463_data + (v425_data * v461_data));
          float v468_data = r0[10];
          float v469_data = s0[10];
          float v471_data = ir1[0];
          ir1[0] = (v471_data + (v468_data * v469_data));
          float v474_data = s0[26];
          float v476_data = ir1[1];
          ir1[1] = (v476_data + (v468_data * v474_data));
          float v479_data = s0[42];
          float v481_data = ir1[2];
          ir1[2] = (v481_data + (v468_data * v479_data));
          float v484_data = s0[58];
          float v486_data = ir1[3];
          ir1[3] = (v486_data + (v468_data * v484_data));
          float v489_data = s0[74];
          float v491_data = ir1[4];
          ir1[4] = (v491_data + (v468_data * v489_data));
          float v494_data = s0[90];
          float v496_data = ir1[5];
          ir1[5] = (v496_data + (v468_data * v494_data));
          float v499_data = s0[106];
          float v501_data = ir1[6];
          ir1[6] = (v501_data + (v468_data * v499_data));
          float v504_data = s0[122];
          float v506_data = ir1[7];
          ir1[7] = (v506_data + (v468_data * v504_data));
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
          float v554_data = r0[12];
          float v555_data = s0[12];
          float v557_data = ir1[0];
          ir1[0] = (v557_data + (v554_data * v555_data));
          float v560_data = s0[28];
          float v562_data = ir1[1];
          ir1[1] = (v562_data + (v554_data * v560_data));
          float v565_data = s0[44];
          float v567_data = ir1[2];
          ir1[2] = (v567_data + (v554_data * v565_data));
          float v570_data = s0[60];
          float v572_data = ir1[3];
          ir1[3] = (v572_data + (v554_data * v570_data));
          float v575_data = s0[76];
          float v577_data = ir1[4];
          ir1[4] = (v577_data + (v554_data * v575_data));
          float v580_data = s0[92];
          float v582_data = ir1[5];
          ir1[5] = (v582_data + (v554_data * v580_data));
          float v585_data = s0[108];
          float v587_data = ir1[6];
          ir1[6] = (v587_data + (v554_data * v585_data));
          float v590_data = s0[124];
          float v592_data = ir1[7];
          ir1[7] = (v592_data + (v554_data * v590_data));
          float v597_data = r0[13];
          float v598_data = s0[13];
          float v600_data = ir1[0];
          ir1[0] = (v600_data + (v597_data * v598_data));
          float v603_data = s0[29];
          float v605_data = ir1[1];
          ir1[1] = (v605_data + (v597_data * v603_data));
          float v608_data = s0[45];
          float v610_data = ir1[2];
          ir1[2] = (v610_data + (v597_data * v608_data));
          float v613_data = s0[61];
          float v615_data = ir1[3];
          ir1[3] = (v615_data + (v597_data * v613_data));
          float v618_data = s0[77];
          float v620_data = ir1[4];
          ir1[4] = (v620_data + (v597_data * v618_data));
          float v623_data = s0[93];
          float v625_data = ir1[5];
          ir1[5] = (v625_data + (v597_data * v623_data));
          float v628_data = s0[109];
          float v630_data = ir1[6];
          ir1[6] = (v630_data + (v597_data * v628_data));
          float v633_data = s0[125];
          float v635_data = ir1[7];
          ir1[7] = (v635_data + (v597_data * v633_data));
          float v640_data = r0[14];
          float v641_data = s0[14];
          float v643_data = ir1[0];
          ir1[0] = (v643_data + (v640_data * v641_data));
          float v646_data = s0[30];
          float v648_data = ir1[1];
          ir1[1] = (v648_data + (v640_data * v646_data));
          float v651_data = s0[46];
          float v653_data = ir1[2];
          ir1[2] = (v653_data + (v640_data * v651_data));
          float v656_data = s0[62];
          float v658_data = ir1[3];
          ir1[3] = (v658_data + (v640_data * v656_data));
          float v661_data = s0[78];
          float v663_data = ir1[4];
          ir1[4] = (v663_data + (v640_data * v661_data));
          float v666_data = s0[94];
          float v668_data = ir1[5];
          ir1[5] = (v668_data + (v640_data * v666_data));
          float v671_data = s0[110];
          float v673_data = ir1[6];
          ir1[6] = (v673_data + (v640_data * v671_data));
          float v676_data = s0[126];
          float v678_data = ir1[7];
          ir1[7] = (v678_data + (v640_data * v676_data));
          float v683_data = r0[15];
          float v684_data = s0[15];
          float v686_data = ir1[0];
          ir1[0] = (v686_data + (v683_data * v684_data));
          float v689_data = s0[31];
          float v691_data = ir1[1];
          ir1[1] = (v691_data + (v683_data * v689_data));
          float v694_data = s0[47];
          float v696_data = ir1[2];
          ir1[2] = (v696_data + (v683_data * v694_data));
          float v699_data = s0[63];
          float v701_data = ir1[3];
          ir1[3] = (v701_data + (v683_data * v699_data));
          float v704_data = s0[79];
          float v706_data = ir1[4];
          ir1[4] = (v706_data + (v683_data * v704_data));
          float v709_data = s0[95];
          float v711_data = ir1[5];
          ir1[5] = (v711_data + (v683_data * v709_data));
          float v714_data = s0[111];
          float v716_data = ir1[6];
          ir1[6] = (v716_data + (v683_data * v714_data));
          float v719_data = s0[127];
          float v721_data = ir1[7];
          ir1[7] = (v721_data + (v683_data * v719_data));
          #pragma unroll
          for (int32_t v726_n0 = 0; v726_n0 < 1; ++v726_n0) {
            #pragma unroll
            for (int32_t v727_n1 = 0; v727_n1 < 8; ++v727_n1) {
              int32_t v728_a = v726_n0 + v727_n1;
              float v729_data = ir1[v728_a];
              r1[v728_a] = v729_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v734_i0 = 0; v734_i0 < 1; ++v734_i0) {
            int32_t v742_lead = v18_lead + (v734_i0 * 16);
            #pragma unroll
            for (int32_t v735_i1 = 0; v735_i1 < 8; ++v735_i1) {
              float v737_data = r1[(v734_i0 + v735_i1)];
              glb_m0[(v742_lead + (v735_i1 * 16))] = v737_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

