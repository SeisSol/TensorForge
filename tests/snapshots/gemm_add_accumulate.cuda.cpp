// === base name ===
kernel_2bfa83deb8242816

// === header ===
void launcher_kernel_2bfa83deb8242816(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_2bfa83deb8242816(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_2bfa83deb8242816, block.x * block.y * block.z, 1152 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_2bfa83deb8242816, cudaFuncAttributeMaxDynamicSharedMemorySize, 1152 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_2bfa83deb8242816<<<grid,block,1152 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_2bfa83deb8242816(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×16(12×16) {0..12}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 16;
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
              float v28_data = __ldcg(&glb_m1[(v18_lead + (v20_i1 * 12))]);
              r0[v20_i1] = v28_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
              float v44_data = glb_m0[(v18_lead + (v36_i1 * 12))];
              r1[v36_i1] = v44_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          float r2[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir2[8]{};
          if (v18_lead < 12) {
            float v52_data = r0[0];
            float v53_data = s0[0];
            float v55_data = ir2[0];
            ir2[0] = (v55_data + (v52_data * v53_data));
            float v58_data = s0[16];
            float v60_data = ir2[1];
            ir2[1] = (v60_data + (v52_data * v58_data));
            float v63_data = s0[32];
            float v65_data = ir2[2];
            ir2[2] = (v65_data + (v52_data * v63_data));
            float v68_data = s0[48];
            float v70_data = ir2[3];
            ir2[3] = (v70_data + (v52_data * v68_data));
            float v73_data = s0[64];
            float v75_data = ir2[4];
            ir2[4] = (v75_data + (v52_data * v73_data));
            float v78_data = s0[80];
            float v80_data = ir2[5];
            ir2[5] = (v80_data + (v52_data * v78_data));
            float v83_data = s0[96];
            float v85_data = ir2[6];
            ir2[6] = (v85_data + (v52_data * v83_data));
            float v88_data = s0[112];
            float v90_data = ir2[7];
            ir2[7] = (v90_data + (v52_data * v88_data));
          }
          if (v18_lead < 12) {
            float v96_data = r0[1];
            float v97_data = s0[1];
            float v99_data = ir2[0];
            ir2[0] = (v99_data + (v96_data * v97_data));
            float v102_data = s0[17];
            float v104_data = ir2[1];
            ir2[1] = (v104_data + (v96_data * v102_data));
            float v107_data = s0[33];
            float v109_data = ir2[2];
            ir2[2] = (v109_data + (v96_data * v107_data));
            float v112_data = s0[49];
            float v114_data = ir2[3];
            ir2[3] = (v114_data + (v96_data * v112_data));
            float v117_data = s0[65];
            float v119_data = ir2[4];
            ir2[4] = (v119_data + (v96_data * v117_data));
            float v122_data = s0[81];
            float v124_data = ir2[5];
            ir2[5] = (v124_data + (v96_data * v122_data));
            float v127_data = s0[97];
            float v129_data = ir2[6];
            ir2[6] = (v129_data + (v96_data * v127_data));
            float v132_data = s0[113];
            float v134_data = ir2[7];
            ir2[7] = (v134_data + (v96_data * v132_data));
          }
          if (v18_lead < 12) {
            float v140_data = r0[2];
            float v141_data = s0[2];
            float v143_data = ir2[0];
            ir2[0] = (v143_data + (v140_data * v141_data));
            float v146_data = s0[18];
            float v148_data = ir2[1];
            ir2[1] = (v148_data + (v140_data * v146_data));
            float v151_data = s0[34];
            float v153_data = ir2[2];
            ir2[2] = (v153_data + (v140_data * v151_data));
            float v156_data = s0[50];
            float v158_data = ir2[3];
            ir2[3] = (v158_data + (v140_data * v156_data));
            float v161_data = s0[66];
            float v163_data = ir2[4];
            ir2[4] = (v163_data + (v140_data * v161_data));
            float v166_data = s0[82];
            float v168_data = ir2[5];
            ir2[5] = (v168_data + (v140_data * v166_data));
            float v171_data = s0[98];
            float v173_data = ir2[6];
            ir2[6] = (v173_data + (v140_data * v171_data));
            float v176_data = s0[114];
            float v178_data = ir2[7];
            ir2[7] = (v178_data + (v140_data * v176_data));
          }
          if (v18_lead < 12) {
            float v184_data = r0[3];
            float v185_data = s0[3];
            float v187_data = ir2[0];
            ir2[0] = (v187_data + (v184_data * v185_data));
            float v190_data = s0[19];
            float v192_data = ir2[1];
            ir2[1] = (v192_data + (v184_data * v190_data));
            float v195_data = s0[35];
            float v197_data = ir2[2];
            ir2[2] = (v197_data + (v184_data * v195_data));
            float v200_data = s0[51];
            float v202_data = ir2[3];
            ir2[3] = (v202_data + (v184_data * v200_data));
            float v205_data = s0[67];
            float v207_data = ir2[4];
            ir2[4] = (v207_data + (v184_data * v205_data));
            float v210_data = s0[83];
            float v212_data = ir2[5];
            ir2[5] = (v212_data + (v184_data * v210_data));
            float v215_data = s0[99];
            float v217_data = ir2[6];
            ir2[6] = (v217_data + (v184_data * v215_data));
            float v220_data = s0[115];
            float v222_data = ir2[7];
            ir2[7] = (v222_data + (v184_data * v220_data));
          }
          if (v18_lead < 12) {
            float v228_data = r0[4];
            float v229_data = s0[4];
            float v231_data = ir2[0];
            ir2[0] = (v231_data + (v228_data * v229_data));
            float v234_data = s0[20];
            float v236_data = ir2[1];
            ir2[1] = (v236_data + (v228_data * v234_data));
            float v239_data = s0[36];
            float v241_data = ir2[2];
            ir2[2] = (v241_data + (v228_data * v239_data));
            float v244_data = s0[52];
            float v246_data = ir2[3];
            ir2[3] = (v246_data + (v228_data * v244_data));
            float v249_data = s0[68];
            float v251_data = ir2[4];
            ir2[4] = (v251_data + (v228_data * v249_data));
            float v254_data = s0[84];
            float v256_data = ir2[5];
            ir2[5] = (v256_data + (v228_data * v254_data));
            float v259_data = s0[100];
            float v261_data = ir2[6];
            ir2[6] = (v261_data + (v228_data * v259_data));
            float v264_data = s0[116];
            float v266_data = ir2[7];
            ir2[7] = (v266_data + (v228_data * v264_data));
          }
          if (v18_lead < 12) {
            float v272_data = r0[5];
            float v273_data = s0[5];
            float v275_data = ir2[0];
            ir2[0] = (v275_data + (v272_data * v273_data));
            float v278_data = s0[21];
            float v280_data = ir2[1];
            ir2[1] = (v280_data + (v272_data * v278_data));
            float v283_data = s0[37];
            float v285_data = ir2[2];
            ir2[2] = (v285_data + (v272_data * v283_data));
            float v288_data = s0[53];
            float v290_data = ir2[3];
            ir2[3] = (v290_data + (v272_data * v288_data));
            float v293_data = s0[69];
            float v295_data = ir2[4];
            ir2[4] = (v295_data + (v272_data * v293_data));
            float v298_data = s0[85];
            float v300_data = ir2[5];
            ir2[5] = (v300_data + (v272_data * v298_data));
            float v303_data = s0[101];
            float v305_data = ir2[6];
            ir2[6] = (v305_data + (v272_data * v303_data));
            float v308_data = s0[117];
            float v310_data = ir2[7];
            ir2[7] = (v310_data + (v272_data * v308_data));
          }
          if (v18_lead < 12) {
            float v316_data = r0[6];
            float v317_data = s0[6];
            float v319_data = ir2[0];
            ir2[0] = (v319_data + (v316_data * v317_data));
            float v322_data = s0[22];
            float v324_data = ir2[1];
            ir2[1] = (v324_data + (v316_data * v322_data));
            float v327_data = s0[38];
            float v329_data = ir2[2];
            ir2[2] = (v329_data + (v316_data * v327_data));
            float v332_data = s0[54];
            float v334_data = ir2[3];
            ir2[3] = (v334_data + (v316_data * v332_data));
            float v337_data = s0[70];
            float v339_data = ir2[4];
            ir2[4] = (v339_data + (v316_data * v337_data));
            float v342_data = s0[86];
            float v344_data = ir2[5];
            ir2[5] = (v344_data + (v316_data * v342_data));
            float v347_data = s0[102];
            float v349_data = ir2[6];
            ir2[6] = (v349_data + (v316_data * v347_data));
            float v352_data = s0[118];
            float v354_data = ir2[7];
            ir2[7] = (v354_data + (v316_data * v352_data));
          }
          if (v18_lead < 12) {
            float v360_data = r0[7];
            float v361_data = s0[7];
            float v363_data = ir2[0];
            ir2[0] = (v363_data + (v360_data * v361_data));
            float v366_data = s0[23];
            float v368_data = ir2[1];
            ir2[1] = (v368_data + (v360_data * v366_data));
            float v371_data = s0[39];
            float v373_data = ir2[2];
            ir2[2] = (v373_data + (v360_data * v371_data));
            float v376_data = s0[55];
            float v378_data = ir2[3];
            ir2[3] = (v378_data + (v360_data * v376_data));
            float v381_data = s0[71];
            float v383_data = ir2[4];
            ir2[4] = (v383_data + (v360_data * v381_data));
            float v386_data = s0[87];
            float v388_data = ir2[5];
            ir2[5] = (v388_data + (v360_data * v386_data));
            float v391_data = s0[103];
            float v393_data = ir2[6];
            ir2[6] = (v393_data + (v360_data * v391_data));
            float v396_data = s0[119];
            float v398_data = ir2[7];
            ir2[7] = (v398_data + (v360_data * v396_data));
          }
          if (v18_lead < 12) {
            float v404_data = r0[8];
            float v405_data = s0[8];
            float v407_data = ir2[0];
            ir2[0] = (v407_data + (v404_data * v405_data));
            float v410_data = s0[24];
            float v412_data = ir2[1];
            ir2[1] = (v412_data + (v404_data * v410_data));
            float v415_data = s0[40];
            float v417_data = ir2[2];
            ir2[2] = (v417_data + (v404_data * v415_data));
            float v420_data = s0[56];
            float v422_data = ir2[3];
            ir2[3] = (v422_data + (v404_data * v420_data));
            float v425_data = s0[72];
            float v427_data = ir2[4];
            ir2[4] = (v427_data + (v404_data * v425_data));
            float v430_data = s0[88];
            float v432_data = ir2[5];
            ir2[5] = (v432_data + (v404_data * v430_data));
            float v435_data = s0[104];
            float v437_data = ir2[6];
            ir2[6] = (v437_data + (v404_data * v435_data));
            float v440_data = s0[120];
            float v442_data = ir2[7];
            ir2[7] = (v442_data + (v404_data * v440_data));
          }
          if (v18_lead < 12) {
            float v448_data = r0[9];
            float v449_data = s0[9];
            float v451_data = ir2[0];
            ir2[0] = (v451_data + (v448_data * v449_data));
            float v454_data = s0[25];
            float v456_data = ir2[1];
            ir2[1] = (v456_data + (v448_data * v454_data));
            float v459_data = s0[41];
            float v461_data = ir2[2];
            ir2[2] = (v461_data + (v448_data * v459_data));
            float v464_data = s0[57];
            float v466_data = ir2[3];
            ir2[3] = (v466_data + (v448_data * v464_data));
            float v469_data = s0[73];
            float v471_data = ir2[4];
            ir2[4] = (v471_data + (v448_data * v469_data));
            float v474_data = s0[89];
            float v476_data = ir2[5];
            ir2[5] = (v476_data + (v448_data * v474_data));
            float v479_data = s0[105];
            float v481_data = ir2[6];
            ir2[6] = (v481_data + (v448_data * v479_data));
            float v484_data = s0[121];
            float v486_data = ir2[7];
            ir2[7] = (v486_data + (v448_data * v484_data));
          }
          if (v18_lead < 12) {
            float v492_data = r0[10];
            float v493_data = s0[10];
            float v495_data = ir2[0];
            ir2[0] = (v495_data + (v492_data * v493_data));
            float v498_data = s0[26];
            float v500_data = ir2[1];
            ir2[1] = (v500_data + (v492_data * v498_data));
            float v503_data = s0[42];
            float v505_data = ir2[2];
            ir2[2] = (v505_data + (v492_data * v503_data));
            float v508_data = s0[58];
            float v510_data = ir2[3];
            ir2[3] = (v510_data + (v492_data * v508_data));
            float v513_data = s0[74];
            float v515_data = ir2[4];
            ir2[4] = (v515_data + (v492_data * v513_data));
            float v518_data = s0[90];
            float v520_data = ir2[5];
            ir2[5] = (v520_data + (v492_data * v518_data));
            float v523_data = s0[106];
            float v525_data = ir2[6];
            ir2[6] = (v525_data + (v492_data * v523_data));
            float v528_data = s0[122];
            float v530_data = ir2[7];
            ir2[7] = (v530_data + (v492_data * v528_data));
          }
          if (v18_lead < 12) {
            float v536_data = r0[11];
            float v537_data = s0[11];
            float v539_data = ir2[0];
            ir2[0] = (v539_data + (v536_data * v537_data));
            float v542_data = s0[27];
            float v544_data = ir2[1];
            ir2[1] = (v544_data + (v536_data * v542_data));
            float v547_data = s0[43];
            float v549_data = ir2[2];
            ir2[2] = (v549_data + (v536_data * v547_data));
            float v552_data = s0[59];
            float v554_data = ir2[3];
            ir2[3] = (v554_data + (v536_data * v552_data));
            float v557_data = s0[75];
            float v559_data = ir2[4];
            ir2[4] = (v559_data + (v536_data * v557_data));
            float v562_data = s0[91];
            float v564_data = ir2[5];
            ir2[5] = (v564_data + (v536_data * v562_data));
            float v567_data = s0[107];
            float v569_data = ir2[6];
            ir2[6] = (v569_data + (v536_data * v567_data));
            float v572_data = s0[123];
            float v574_data = ir2[7];
            ir2[7] = (v574_data + (v536_data * v572_data));
          }
          if (v18_lead < 12) {
            float v580_data = r0[12];
            float v581_data = s0[12];
            float v583_data = ir2[0];
            ir2[0] = (v583_data + (v580_data * v581_data));
            float v586_data = s0[28];
            float v588_data = ir2[1];
            ir2[1] = (v588_data + (v580_data * v586_data));
            float v591_data = s0[44];
            float v593_data = ir2[2];
            ir2[2] = (v593_data + (v580_data * v591_data));
            float v596_data = s0[60];
            float v598_data = ir2[3];
            ir2[3] = (v598_data + (v580_data * v596_data));
            float v601_data = s0[76];
            float v603_data = ir2[4];
            ir2[4] = (v603_data + (v580_data * v601_data));
            float v606_data = s0[92];
            float v608_data = ir2[5];
            ir2[5] = (v608_data + (v580_data * v606_data));
            float v611_data = s0[108];
            float v613_data = ir2[6];
            ir2[6] = (v613_data + (v580_data * v611_data));
            float v616_data = s0[124];
            float v618_data = ir2[7];
            ir2[7] = (v618_data + (v580_data * v616_data));
          }
          if (v18_lead < 12) {
            float v624_data = r0[13];
            float v625_data = s0[13];
            float v627_data = ir2[0];
            ir2[0] = (v627_data + (v624_data * v625_data));
            float v630_data = s0[29];
            float v632_data = ir2[1];
            ir2[1] = (v632_data + (v624_data * v630_data));
            float v635_data = s0[45];
            float v637_data = ir2[2];
            ir2[2] = (v637_data + (v624_data * v635_data));
            float v640_data = s0[61];
            float v642_data = ir2[3];
            ir2[3] = (v642_data + (v624_data * v640_data));
            float v645_data = s0[77];
            float v647_data = ir2[4];
            ir2[4] = (v647_data + (v624_data * v645_data));
            float v650_data = s0[93];
            float v652_data = ir2[5];
            ir2[5] = (v652_data + (v624_data * v650_data));
            float v655_data = s0[109];
            float v657_data = ir2[6];
            ir2[6] = (v657_data + (v624_data * v655_data));
            float v660_data = s0[125];
            float v662_data = ir2[7];
            ir2[7] = (v662_data + (v624_data * v660_data));
          }
          if (v18_lead < 12) {
            float v668_data = r0[14];
            float v669_data = s0[14];
            float v671_data = ir2[0];
            ir2[0] = (v671_data + (v668_data * v669_data));
            float v674_data = s0[30];
            float v676_data = ir2[1];
            ir2[1] = (v676_data + (v668_data * v674_data));
            float v679_data = s0[46];
            float v681_data = ir2[2];
            ir2[2] = (v681_data + (v668_data * v679_data));
            float v684_data = s0[62];
            float v686_data = ir2[3];
            ir2[3] = (v686_data + (v668_data * v684_data));
            float v689_data = s0[78];
            float v691_data = ir2[4];
            ir2[4] = (v691_data + (v668_data * v689_data));
            float v694_data = s0[94];
            float v696_data = ir2[5];
            ir2[5] = (v696_data + (v668_data * v694_data));
            float v699_data = s0[110];
            float v701_data = ir2[6];
            ir2[6] = (v701_data + (v668_data * v699_data));
            float v704_data = s0[126];
            float v706_data = ir2[7];
            ir2[7] = (v706_data + (v668_data * v704_data));
          }
          if (v18_lead < 12) {
            float v712_data = r0[15];
            float v713_data = s0[15];
            float v715_data = ir2[0];
            ir2[0] = (v715_data + (v712_data * v713_data));
            float v718_data = s0[31];
            float v720_data = ir2[1];
            ir2[1] = (v720_data + (v712_data * v718_data));
            float v723_data = s0[47];
            float v725_data = ir2[2];
            ir2[2] = (v725_data + (v712_data * v723_data));
            float v728_data = s0[63];
            float v730_data = ir2[3];
            ir2[3] = (v730_data + (v712_data * v728_data));
            float v733_data = s0[79];
            float v735_data = ir2[4];
            ir2[4] = (v735_data + (v712_data * v733_data));
            float v738_data = s0[95];
            float v740_data = ir2[5];
            ir2[5] = (v740_data + (v712_data * v738_data));
            float v743_data = s0[111];
            float v745_data = ir2[6];
            ir2[6] = (v745_data + (v712_data * v743_data));
            float v748_data = s0[127];
            float v750_data = ir2[7];
            ir2[7] = (v750_data + (v712_data * v748_data));
          }
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v756_n1 = 0; v756_n1 < 8; ++v756_n1) {
              float v758_data = ir2[v756_n1];
              float v760_data = r1[v756_n1];
              r2[v756_n1] = (v760_data + v758_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v767_i1 = 0; v767_i1 < 8; ++v767_i1) {
              float v769_data = r2[v767_i1];
              glb_m0[(v18_lead + (v767_i1 * 12))] = v769_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

