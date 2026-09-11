// === base name ===
kernel_243d4315660bd964

// === header ===
void launcher_kernel_243d4315660bd964(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_243d4315660bd964(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_243d4315660bd964, block.x * block.y * block.z, 1152 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_243d4315660bd964, cudaFuncAttributeMaxDynamicSharedMemorySize, 1152 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_243d4315660bd964<<<grid,block,1152 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_243d4315660bd964(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v4_batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v4_batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v4_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v26_off = (v18_lead + (v19_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v20_i1 = 8; v20_i1 < 24; ++v20_i1) {
              double v29_data = __ldcg(&glb_m1[(v26_off + (v20_i1 * 32))]);
              r0[(v19_i0 + (v20_i1 - 8))] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          double ir1[8]{};
          double v38_data = r0[0];
          double v39_data = s0[0];
          double v41_data = ir1[0];
          ir1[0] = (v41_data + (v38_data * v39_data));
          double v44_data = s0[16];
          double v46_data = ir1[1];
          ir1[1] = (v46_data + (v38_data * v44_data));
          double v49_data = s0[32];
          double v51_data = ir1[2];
          ir1[2] = (v51_data + (v38_data * v49_data));
          double v54_data = s0[48];
          double v56_data = ir1[3];
          ir1[3] = (v56_data + (v38_data * v54_data));
          double v59_data = s0[64];
          double v61_data = ir1[4];
          ir1[4] = (v61_data + (v38_data * v59_data));
          double v64_data = s0[80];
          double v66_data = ir1[5];
          ir1[5] = (v66_data + (v38_data * v64_data));
          double v69_data = s0[96];
          double v71_data = ir1[6];
          ir1[6] = (v71_data + (v38_data * v69_data));
          double v74_data = s0[112];
          double v76_data = ir1[7];
          ir1[7] = (v76_data + (v38_data * v74_data));
          double v81_data = r0[1];
          double v82_data = s0[1];
          double v84_data = ir1[0];
          ir1[0] = (v84_data + (v81_data * v82_data));
          double v87_data = s0[17];
          double v89_data = ir1[1];
          ir1[1] = (v89_data + (v81_data * v87_data));
          double v92_data = s0[33];
          double v94_data = ir1[2];
          ir1[2] = (v94_data + (v81_data * v92_data));
          double v97_data = s0[49];
          double v99_data = ir1[3];
          ir1[3] = (v99_data + (v81_data * v97_data));
          double v102_data = s0[65];
          double v104_data = ir1[4];
          ir1[4] = (v104_data + (v81_data * v102_data));
          double v107_data = s0[81];
          double v109_data = ir1[5];
          ir1[5] = (v109_data + (v81_data * v107_data));
          double v112_data = s0[97];
          double v114_data = ir1[6];
          ir1[6] = (v114_data + (v81_data * v112_data));
          double v117_data = s0[113];
          double v119_data = ir1[7];
          ir1[7] = (v119_data + (v81_data * v117_data));
          double v124_data = r0[2];
          double v125_data = s0[2];
          double v127_data = ir1[0];
          ir1[0] = (v127_data + (v124_data * v125_data));
          double v130_data = s0[18];
          double v132_data = ir1[1];
          ir1[1] = (v132_data + (v124_data * v130_data));
          double v135_data = s0[34];
          double v137_data = ir1[2];
          ir1[2] = (v137_data + (v124_data * v135_data));
          double v140_data = s0[50];
          double v142_data = ir1[3];
          ir1[3] = (v142_data + (v124_data * v140_data));
          double v145_data = s0[66];
          double v147_data = ir1[4];
          ir1[4] = (v147_data + (v124_data * v145_data));
          double v150_data = s0[82];
          double v152_data = ir1[5];
          ir1[5] = (v152_data + (v124_data * v150_data));
          double v155_data = s0[98];
          double v157_data = ir1[6];
          ir1[6] = (v157_data + (v124_data * v155_data));
          double v160_data = s0[114];
          double v162_data = ir1[7];
          ir1[7] = (v162_data + (v124_data * v160_data));
          double v167_data = r0[3];
          double v168_data = s0[3];
          double v170_data = ir1[0];
          ir1[0] = (v170_data + (v167_data * v168_data));
          double v173_data = s0[19];
          double v175_data = ir1[1];
          ir1[1] = (v175_data + (v167_data * v173_data));
          double v178_data = s0[35];
          double v180_data = ir1[2];
          ir1[2] = (v180_data + (v167_data * v178_data));
          double v183_data = s0[51];
          double v185_data = ir1[3];
          ir1[3] = (v185_data + (v167_data * v183_data));
          double v188_data = s0[67];
          double v190_data = ir1[4];
          ir1[4] = (v190_data + (v167_data * v188_data));
          double v193_data = s0[83];
          double v195_data = ir1[5];
          ir1[5] = (v195_data + (v167_data * v193_data));
          double v198_data = s0[99];
          double v200_data = ir1[6];
          ir1[6] = (v200_data + (v167_data * v198_data));
          double v203_data = s0[115];
          double v205_data = ir1[7];
          ir1[7] = (v205_data + (v167_data * v203_data));
          double v210_data = r0[4];
          double v211_data = s0[4];
          double v213_data = ir1[0];
          ir1[0] = (v213_data + (v210_data * v211_data));
          double v216_data = s0[20];
          double v218_data = ir1[1];
          ir1[1] = (v218_data + (v210_data * v216_data));
          double v221_data = s0[36];
          double v223_data = ir1[2];
          ir1[2] = (v223_data + (v210_data * v221_data));
          double v226_data = s0[52];
          double v228_data = ir1[3];
          ir1[3] = (v228_data + (v210_data * v226_data));
          double v231_data = s0[68];
          double v233_data = ir1[4];
          ir1[4] = (v233_data + (v210_data * v231_data));
          double v236_data = s0[84];
          double v238_data = ir1[5];
          ir1[5] = (v238_data + (v210_data * v236_data));
          double v241_data = s0[100];
          double v243_data = ir1[6];
          ir1[6] = (v243_data + (v210_data * v241_data));
          double v246_data = s0[116];
          double v248_data = ir1[7];
          ir1[7] = (v248_data + (v210_data * v246_data));
          double v253_data = r0[5];
          double v254_data = s0[5];
          double v256_data = ir1[0];
          ir1[0] = (v256_data + (v253_data * v254_data));
          double v259_data = s0[21];
          double v261_data = ir1[1];
          ir1[1] = (v261_data + (v253_data * v259_data));
          double v264_data = s0[37];
          double v266_data = ir1[2];
          ir1[2] = (v266_data + (v253_data * v264_data));
          double v269_data = s0[53];
          double v271_data = ir1[3];
          ir1[3] = (v271_data + (v253_data * v269_data));
          double v274_data = s0[69];
          double v276_data = ir1[4];
          ir1[4] = (v276_data + (v253_data * v274_data));
          double v279_data = s0[85];
          double v281_data = ir1[5];
          ir1[5] = (v281_data + (v253_data * v279_data));
          double v284_data = s0[101];
          double v286_data = ir1[6];
          ir1[6] = (v286_data + (v253_data * v284_data));
          double v289_data = s0[117];
          double v291_data = ir1[7];
          ir1[7] = (v291_data + (v253_data * v289_data));
          double v296_data = r0[6];
          double v297_data = s0[6];
          double v299_data = ir1[0];
          ir1[0] = (v299_data + (v296_data * v297_data));
          double v302_data = s0[22];
          double v304_data = ir1[1];
          ir1[1] = (v304_data + (v296_data * v302_data));
          double v307_data = s0[38];
          double v309_data = ir1[2];
          ir1[2] = (v309_data + (v296_data * v307_data));
          double v312_data = s0[54];
          double v314_data = ir1[3];
          ir1[3] = (v314_data + (v296_data * v312_data));
          double v317_data = s0[70];
          double v319_data = ir1[4];
          ir1[4] = (v319_data + (v296_data * v317_data));
          double v322_data = s0[86];
          double v324_data = ir1[5];
          ir1[5] = (v324_data + (v296_data * v322_data));
          double v327_data = s0[102];
          double v329_data = ir1[6];
          ir1[6] = (v329_data + (v296_data * v327_data));
          double v332_data = s0[118];
          double v334_data = ir1[7];
          ir1[7] = (v334_data + (v296_data * v332_data));
          double v339_data = r0[7];
          double v340_data = s0[7];
          double v342_data = ir1[0];
          ir1[0] = (v342_data + (v339_data * v340_data));
          double v345_data = s0[23];
          double v347_data = ir1[1];
          ir1[1] = (v347_data + (v339_data * v345_data));
          double v350_data = s0[39];
          double v352_data = ir1[2];
          ir1[2] = (v352_data + (v339_data * v350_data));
          double v355_data = s0[55];
          double v357_data = ir1[3];
          ir1[3] = (v357_data + (v339_data * v355_data));
          double v360_data = s0[71];
          double v362_data = ir1[4];
          ir1[4] = (v362_data + (v339_data * v360_data));
          double v365_data = s0[87];
          double v367_data = ir1[5];
          ir1[5] = (v367_data + (v339_data * v365_data));
          double v370_data = s0[103];
          double v372_data = ir1[6];
          ir1[6] = (v372_data + (v339_data * v370_data));
          double v375_data = s0[119];
          double v377_data = ir1[7];
          ir1[7] = (v377_data + (v339_data * v375_data));
          double v382_data = r0[8];
          double v383_data = s0[8];
          double v385_data = ir1[0];
          ir1[0] = (v385_data + (v382_data * v383_data));
          double v388_data = s0[24];
          double v390_data = ir1[1];
          ir1[1] = (v390_data + (v382_data * v388_data));
          double v393_data = s0[40];
          double v395_data = ir1[2];
          ir1[2] = (v395_data + (v382_data * v393_data));
          double v398_data = s0[56];
          double v400_data = ir1[3];
          ir1[3] = (v400_data + (v382_data * v398_data));
          double v403_data = s0[72];
          double v405_data = ir1[4];
          ir1[4] = (v405_data + (v382_data * v403_data));
          double v408_data = s0[88];
          double v410_data = ir1[5];
          ir1[5] = (v410_data + (v382_data * v408_data));
          double v413_data = s0[104];
          double v415_data = ir1[6];
          ir1[6] = (v415_data + (v382_data * v413_data));
          double v418_data = s0[120];
          double v420_data = ir1[7];
          ir1[7] = (v420_data + (v382_data * v418_data));
          double v425_data = r0[9];
          double v426_data = s0[9];
          double v428_data = ir1[0];
          ir1[0] = (v428_data + (v425_data * v426_data));
          double v431_data = s0[25];
          double v433_data = ir1[1];
          ir1[1] = (v433_data + (v425_data * v431_data));
          double v436_data = s0[41];
          double v438_data = ir1[2];
          ir1[2] = (v438_data + (v425_data * v436_data));
          double v441_data = s0[57];
          double v443_data = ir1[3];
          ir1[3] = (v443_data + (v425_data * v441_data));
          double v446_data = s0[73];
          double v448_data = ir1[4];
          ir1[4] = (v448_data + (v425_data * v446_data));
          double v451_data = s0[89];
          double v453_data = ir1[5];
          ir1[5] = (v453_data + (v425_data * v451_data));
          double v456_data = s0[105];
          double v458_data = ir1[6];
          ir1[6] = (v458_data + (v425_data * v456_data));
          double v461_data = s0[121];
          double v463_data = ir1[7];
          ir1[7] = (v463_data + (v425_data * v461_data));
          double v468_data = r0[10];
          double v469_data = s0[10];
          double v471_data = ir1[0];
          ir1[0] = (v471_data + (v468_data * v469_data));
          double v474_data = s0[26];
          double v476_data = ir1[1];
          ir1[1] = (v476_data + (v468_data * v474_data));
          double v479_data = s0[42];
          double v481_data = ir1[2];
          ir1[2] = (v481_data + (v468_data * v479_data));
          double v484_data = s0[58];
          double v486_data = ir1[3];
          ir1[3] = (v486_data + (v468_data * v484_data));
          double v489_data = s0[74];
          double v491_data = ir1[4];
          ir1[4] = (v491_data + (v468_data * v489_data));
          double v494_data = s0[90];
          double v496_data = ir1[5];
          ir1[5] = (v496_data + (v468_data * v494_data));
          double v499_data = s0[106];
          double v501_data = ir1[6];
          ir1[6] = (v501_data + (v468_data * v499_data));
          double v504_data = s0[122];
          double v506_data = ir1[7];
          ir1[7] = (v506_data + (v468_data * v504_data));
          double v511_data = r0[11];
          double v512_data = s0[11];
          double v514_data = ir1[0];
          ir1[0] = (v514_data + (v511_data * v512_data));
          double v517_data = s0[27];
          double v519_data = ir1[1];
          ir1[1] = (v519_data + (v511_data * v517_data));
          double v522_data = s0[43];
          double v524_data = ir1[2];
          ir1[2] = (v524_data + (v511_data * v522_data));
          double v527_data = s0[59];
          double v529_data = ir1[3];
          ir1[3] = (v529_data + (v511_data * v527_data));
          double v532_data = s0[75];
          double v534_data = ir1[4];
          ir1[4] = (v534_data + (v511_data * v532_data));
          double v537_data = s0[91];
          double v539_data = ir1[5];
          ir1[5] = (v539_data + (v511_data * v537_data));
          double v542_data = s0[107];
          double v544_data = ir1[6];
          ir1[6] = (v544_data + (v511_data * v542_data));
          double v547_data = s0[123];
          double v549_data = ir1[7];
          ir1[7] = (v549_data + (v511_data * v547_data));
          double v554_data = r0[12];
          double v555_data = s0[12];
          double v557_data = ir1[0];
          ir1[0] = (v557_data + (v554_data * v555_data));
          double v560_data = s0[28];
          double v562_data = ir1[1];
          ir1[1] = (v562_data + (v554_data * v560_data));
          double v565_data = s0[44];
          double v567_data = ir1[2];
          ir1[2] = (v567_data + (v554_data * v565_data));
          double v570_data = s0[60];
          double v572_data = ir1[3];
          ir1[3] = (v572_data + (v554_data * v570_data));
          double v575_data = s0[76];
          double v577_data = ir1[4];
          ir1[4] = (v577_data + (v554_data * v575_data));
          double v580_data = s0[92];
          double v582_data = ir1[5];
          ir1[5] = (v582_data + (v554_data * v580_data));
          double v585_data = s0[108];
          double v587_data = ir1[6];
          ir1[6] = (v587_data + (v554_data * v585_data));
          double v590_data = s0[124];
          double v592_data = ir1[7];
          ir1[7] = (v592_data + (v554_data * v590_data));
          double v597_data = r0[13];
          double v598_data = s0[13];
          double v600_data = ir1[0];
          ir1[0] = (v600_data + (v597_data * v598_data));
          double v603_data = s0[29];
          double v605_data = ir1[1];
          ir1[1] = (v605_data + (v597_data * v603_data));
          double v608_data = s0[45];
          double v610_data = ir1[2];
          ir1[2] = (v610_data + (v597_data * v608_data));
          double v613_data = s0[61];
          double v615_data = ir1[3];
          ir1[3] = (v615_data + (v597_data * v613_data));
          double v618_data = s0[77];
          double v620_data = ir1[4];
          ir1[4] = (v620_data + (v597_data * v618_data));
          double v623_data = s0[93];
          double v625_data = ir1[5];
          ir1[5] = (v625_data + (v597_data * v623_data));
          double v628_data = s0[109];
          double v630_data = ir1[6];
          ir1[6] = (v630_data + (v597_data * v628_data));
          double v633_data = s0[125];
          double v635_data = ir1[7];
          ir1[7] = (v635_data + (v597_data * v633_data));
          double v640_data = r0[14];
          double v641_data = s0[14];
          double v643_data = ir1[0];
          ir1[0] = (v643_data + (v640_data * v641_data));
          double v646_data = s0[30];
          double v648_data = ir1[1];
          ir1[1] = (v648_data + (v640_data * v646_data));
          double v651_data = s0[46];
          double v653_data = ir1[2];
          ir1[2] = (v653_data + (v640_data * v651_data));
          double v656_data = s0[62];
          double v658_data = ir1[3];
          ir1[3] = (v658_data + (v640_data * v656_data));
          double v661_data = s0[78];
          double v663_data = ir1[4];
          ir1[4] = (v663_data + (v640_data * v661_data));
          double v666_data = s0[94];
          double v668_data = ir1[5];
          ir1[5] = (v668_data + (v640_data * v666_data));
          double v671_data = s0[110];
          double v673_data = ir1[6];
          ir1[6] = (v673_data + (v640_data * v671_data));
          double v676_data = s0[126];
          double v678_data = ir1[7];
          ir1[7] = (v678_data + (v640_data * v676_data));
          double v683_data = r0[15];
          double v684_data = s0[15];
          double v686_data = ir1[0];
          ir1[0] = (v686_data + (v683_data * v684_data));
          double v689_data = s0[31];
          double v691_data = ir1[1];
          ir1[1] = (v691_data + (v683_data * v689_data));
          double v694_data = s0[47];
          double v696_data = ir1[2];
          ir1[2] = (v696_data + (v683_data * v694_data));
          double v699_data = s0[63];
          double v701_data = ir1[3];
          ir1[3] = (v701_data + (v683_data * v699_data));
          double v704_data = s0[79];
          double v706_data = ir1[4];
          ir1[4] = (v706_data + (v683_data * v704_data));
          double v709_data = s0[95];
          double v711_data = ir1[5];
          ir1[5] = (v711_data + (v683_data * v709_data));
          double v714_data = s0[111];
          double v716_data = ir1[6];
          ir1[6] = (v716_data + (v683_data * v714_data));
          double v719_data = s0[127];
          double v721_data = ir1[7];
          ir1[7] = (v721_data + (v683_data * v719_data));
          #pragma unroll
          for (int32_t v726_n0 = 0; v726_n0 < 1; ++v726_n0) {
            #pragma unroll
            for (int32_t v727_n1 = 0; v727_n1 < 8; ++v727_n1) {
              int32_t v728_a = v726_n0 + v727_n1;
              double v729_data = ir1[v728_a];
              r1[v728_a] = v729_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v734_i0 = 0; v734_i0 < 1; ++v734_i0) {
            int32_t v742_lead = v18_lead + (v734_i0 * 16);
            #pragma unroll
            for (int32_t v735_i1 = 0; v735_i1 < 8; ++v735_i1) {
              double v737_data = r1[(v734_i0 + v735_i1)];
              glb_m0[(v742_lead + (v735_i1 * 16))] = v737_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

