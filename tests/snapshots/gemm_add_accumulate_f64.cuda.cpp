// === base name ===
kernel_250fd6dadbd96789

// === header ===
void launcher_kernel_250fd6dadbd96789(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_250fd6dadbd96789(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_250fd6dadbd96789, block.x * block.y * block.z, 1152 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_250fd6dadbd96789, cudaFuncAttributeMaxDynamicSharedMemorySize, 1152 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_250fd6dadbd96789<<<grid,block,1152 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_250fd6dadbd96789(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v4_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v4_batchId0 < numElements0; v4_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v5_ahead1 = v4_batchId0 + (gridDim.x * blockDim.y);
        size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v4_batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v4_batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v4_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v18_lead = threadIdx.x % 16;
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
              double v28_data = __ldcg(&glb_m1[(v18_lead + (v20_i1 * 12))]);
              r0[v20_i1] = v28_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
              double v44_data = glb_m0[(v18_lead + (v36_i1 * 12))];
              r1[v36_i1] = v44_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          if (v18_lead < 12) {
            double v52_data = r0[0];
            double v53_data = s0[0];
            double v55_data = ir2[0];
            ir2[0] = (v55_data + (v52_data * v53_data));
            double v58_data = s0[16];
            double v60_data = ir2[1];
            ir2[1] = (v60_data + (v52_data * v58_data));
            double v63_data = s0[32];
            double v65_data = ir2[2];
            ir2[2] = (v65_data + (v52_data * v63_data));
            double v68_data = s0[48];
            double v70_data = ir2[3];
            ir2[3] = (v70_data + (v52_data * v68_data));
            double v73_data = s0[64];
            double v75_data = ir2[4];
            ir2[4] = (v75_data + (v52_data * v73_data));
            double v78_data = s0[80];
            double v80_data = ir2[5];
            ir2[5] = (v80_data + (v52_data * v78_data));
            double v83_data = s0[96];
            double v85_data = ir2[6];
            ir2[6] = (v85_data + (v52_data * v83_data));
            double v88_data = s0[112];
            double v90_data = ir2[7];
            ir2[7] = (v90_data + (v52_data * v88_data));
          }
          if (v18_lead < 12) {
            double v96_data = r0[1];
            double v97_data = s0[1];
            double v99_data = ir2[0];
            ir2[0] = (v99_data + (v96_data * v97_data));
            double v102_data = s0[17];
            double v104_data = ir2[1];
            ir2[1] = (v104_data + (v96_data * v102_data));
            double v107_data = s0[33];
            double v109_data = ir2[2];
            ir2[2] = (v109_data + (v96_data * v107_data));
            double v112_data = s0[49];
            double v114_data = ir2[3];
            ir2[3] = (v114_data + (v96_data * v112_data));
            double v117_data = s0[65];
            double v119_data = ir2[4];
            ir2[4] = (v119_data + (v96_data * v117_data));
            double v122_data = s0[81];
            double v124_data = ir2[5];
            ir2[5] = (v124_data + (v96_data * v122_data));
            double v127_data = s0[97];
            double v129_data = ir2[6];
            ir2[6] = (v129_data + (v96_data * v127_data));
            double v132_data = s0[113];
            double v134_data = ir2[7];
            ir2[7] = (v134_data + (v96_data * v132_data));
          }
          if (v18_lead < 12) {
            double v140_data = r0[2];
            double v141_data = s0[2];
            double v143_data = ir2[0];
            ir2[0] = (v143_data + (v140_data * v141_data));
            double v146_data = s0[18];
            double v148_data = ir2[1];
            ir2[1] = (v148_data + (v140_data * v146_data));
            double v151_data = s0[34];
            double v153_data = ir2[2];
            ir2[2] = (v153_data + (v140_data * v151_data));
            double v156_data = s0[50];
            double v158_data = ir2[3];
            ir2[3] = (v158_data + (v140_data * v156_data));
            double v161_data = s0[66];
            double v163_data = ir2[4];
            ir2[4] = (v163_data + (v140_data * v161_data));
            double v166_data = s0[82];
            double v168_data = ir2[5];
            ir2[5] = (v168_data + (v140_data * v166_data));
            double v171_data = s0[98];
            double v173_data = ir2[6];
            ir2[6] = (v173_data + (v140_data * v171_data));
            double v176_data = s0[114];
            double v178_data = ir2[7];
            ir2[7] = (v178_data + (v140_data * v176_data));
          }
          if (v18_lead < 12) {
            double v184_data = r0[3];
            double v185_data = s0[3];
            double v187_data = ir2[0];
            ir2[0] = (v187_data + (v184_data * v185_data));
            double v190_data = s0[19];
            double v192_data = ir2[1];
            ir2[1] = (v192_data + (v184_data * v190_data));
            double v195_data = s0[35];
            double v197_data = ir2[2];
            ir2[2] = (v197_data + (v184_data * v195_data));
            double v200_data = s0[51];
            double v202_data = ir2[3];
            ir2[3] = (v202_data + (v184_data * v200_data));
            double v205_data = s0[67];
            double v207_data = ir2[4];
            ir2[4] = (v207_data + (v184_data * v205_data));
            double v210_data = s0[83];
            double v212_data = ir2[5];
            ir2[5] = (v212_data + (v184_data * v210_data));
            double v215_data = s0[99];
            double v217_data = ir2[6];
            ir2[6] = (v217_data + (v184_data * v215_data));
            double v220_data = s0[115];
            double v222_data = ir2[7];
            ir2[7] = (v222_data + (v184_data * v220_data));
          }
          if (v18_lead < 12) {
            double v228_data = r0[4];
            double v229_data = s0[4];
            double v231_data = ir2[0];
            ir2[0] = (v231_data + (v228_data * v229_data));
            double v234_data = s0[20];
            double v236_data = ir2[1];
            ir2[1] = (v236_data + (v228_data * v234_data));
            double v239_data = s0[36];
            double v241_data = ir2[2];
            ir2[2] = (v241_data + (v228_data * v239_data));
            double v244_data = s0[52];
            double v246_data = ir2[3];
            ir2[3] = (v246_data + (v228_data * v244_data));
            double v249_data = s0[68];
            double v251_data = ir2[4];
            ir2[4] = (v251_data + (v228_data * v249_data));
            double v254_data = s0[84];
            double v256_data = ir2[5];
            ir2[5] = (v256_data + (v228_data * v254_data));
            double v259_data = s0[100];
            double v261_data = ir2[6];
            ir2[6] = (v261_data + (v228_data * v259_data));
            double v264_data = s0[116];
            double v266_data = ir2[7];
            ir2[7] = (v266_data + (v228_data * v264_data));
          }
          if (v18_lead < 12) {
            double v272_data = r0[5];
            double v273_data = s0[5];
            double v275_data = ir2[0];
            ir2[0] = (v275_data + (v272_data * v273_data));
            double v278_data = s0[21];
            double v280_data = ir2[1];
            ir2[1] = (v280_data + (v272_data * v278_data));
            double v283_data = s0[37];
            double v285_data = ir2[2];
            ir2[2] = (v285_data + (v272_data * v283_data));
            double v288_data = s0[53];
            double v290_data = ir2[3];
            ir2[3] = (v290_data + (v272_data * v288_data));
            double v293_data = s0[69];
            double v295_data = ir2[4];
            ir2[4] = (v295_data + (v272_data * v293_data));
            double v298_data = s0[85];
            double v300_data = ir2[5];
            ir2[5] = (v300_data + (v272_data * v298_data));
            double v303_data = s0[101];
            double v305_data = ir2[6];
            ir2[6] = (v305_data + (v272_data * v303_data));
            double v308_data = s0[117];
            double v310_data = ir2[7];
            ir2[7] = (v310_data + (v272_data * v308_data));
          }
          if (v18_lead < 12) {
            double v316_data = r0[6];
            double v317_data = s0[6];
            double v319_data = ir2[0];
            ir2[0] = (v319_data + (v316_data * v317_data));
            double v322_data = s0[22];
            double v324_data = ir2[1];
            ir2[1] = (v324_data + (v316_data * v322_data));
            double v327_data = s0[38];
            double v329_data = ir2[2];
            ir2[2] = (v329_data + (v316_data * v327_data));
            double v332_data = s0[54];
            double v334_data = ir2[3];
            ir2[3] = (v334_data + (v316_data * v332_data));
            double v337_data = s0[70];
            double v339_data = ir2[4];
            ir2[4] = (v339_data + (v316_data * v337_data));
            double v342_data = s0[86];
            double v344_data = ir2[5];
            ir2[5] = (v344_data + (v316_data * v342_data));
            double v347_data = s0[102];
            double v349_data = ir2[6];
            ir2[6] = (v349_data + (v316_data * v347_data));
            double v352_data = s0[118];
            double v354_data = ir2[7];
            ir2[7] = (v354_data + (v316_data * v352_data));
          }
          if (v18_lead < 12) {
            double v360_data = r0[7];
            double v361_data = s0[7];
            double v363_data = ir2[0];
            ir2[0] = (v363_data + (v360_data * v361_data));
            double v366_data = s0[23];
            double v368_data = ir2[1];
            ir2[1] = (v368_data + (v360_data * v366_data));
            double v371_data = s0[39];
            double v373_data = ir2[2];
            ir2[2] = (v373_data + (v360_data * v371_data));
            double v376_data = s0[55];
            double v378_data = ir2[3];
            ir2[3] = (v378_data + (v360_data * v376_data));
            double v381_data = s0[71];
            double v383_data = ir2[4];
            ir2[4] = (v383_data + (v360_data * v381_data));
            double v386_data = s0[87];
            double v388_data = ir2[5];
            ir2[5] = (v388_data + (v360_data * v386_data));
            double v391_data = s0[103];
            double v393_data = ir2[6];
            ir2[6] = (v393_data + (v360_data * v391_data));
            double v396_data = s0[119];
            double v398_data = ir2[7];
            ir2[7] = (v398_data + (v360_data * v396_data));
          }
          if (v18_lead < 12) {
            double v404_data = r0[8];
            double v405_data = s0[8];
            double v407_data = ir2[0];
            ir2[0] = (v407_data + (v404_data * v405_data));
            double v410_data = s0[24];
            double v412_data = ir2[1];
            ir2[1] = (v412_data + (v404_data * v410_data));
            double v415_data = s0[40];
            double v417_data = ir2[2];
            ir2[2] = (v417_data + (v404_data * v415_data));
            double v420_data = s0[56];
            double v422_data = ir2[3];
            ir2[3] = (v422_data + (v404_data * v420_data));
            double v425_data = s0[72];
            double v427_data = ir2[4];
            ir2[4] = (v427_data + (v404_data * v425_data));
            double v430_data = s0[88];
            double v432_data = ir2[5];
            ir2[5] = (v432_data + (v404_data * v430_data));
            double v435_data = s0[104];
            double v437_data = ir2[6];
            ir2[6] = (v437_data + (v404_data * v435_data));
            double v440_data = s0[120];
            double v442_data = ir2[7];
            ir2[7] = (v442_data + (v404_data * v440_data));
          }
          if (v18_lead < 12) {
            double v448_data = r0[9];
            double v449_data = s0[9];
            double v451_data = ir2[0];
            ir2[0] = (v451_data + (v448_data * v449_data));
            double v454_data = s0[25];
            double v456_data = ir2[1];
            ir2[1] = (v456_data + (v448_data * v454_data));
            double v459_data = s0[41];
            double v461_data = ir2[2];
            ir2[2] = (v461_data + (v448_data * v459_data));
            double v464_data = s0[57];
            double v466_data = ir2[3];
            ir2[3] = (v466_data + (v448_data * v464_data));
            double v469_data = s0[73];
            double v471_data = ir2[4];
            ir2[4] = (v471_data + (v448_data * v469_data));
            double v474_data = s0[89];
            double v476_data = ir2[5];
            ir2[5] = (v476_data + (v448_data * v474_data));
            double v479_data = s0[105];
            double v481_data = ir2[6];
            ir2[6] = (v481_data + (v448_data * v479_data));
            double v484_data = s0[121];
            double v486_data = ir2[7];
            ir2[7] = (v486_data + (v448_data * v484_data));
          }
          if (v18_lead < 12) {
            double v492_data = r0[10];
            double v493_data = s0[10];
            double v495_data = ir2[0];
            ir2[0] = (v495_data + (v492_data * v493_data));
            double v498_data = s0[26];
            double v500_data = ir2[1];
            ir2[1] = (v500_data + (v492_data * v498_data));
            double v503_data = s0[42];
            double v505_data = ir2[2];
            ir2[2] = (v505_data + (v492_data * v503_data));
            double v508_data = s0[58];
            double v510_data = ir2[3];
            ir2[3] = (v510_data + (v492_data * v508_data));
            double v513_data = s0[74];
            double v515_data = ir2[4];
            ir2[4] = (v515_data + (v492_data * v513_data));
            double v518_data = s0[90];
            double v520_data = ir2[5];
            ir2[5] = (v520_data + (v492_data * v518_data));
            double v523_data = s0[106];
            double v525_data = ir2[6];
            ir2[6] = (v525_data + (v492_data * v523_data));
            double v528_data = s0[122];
            double v530_data = ir2[7];
            ir2[7] = (v530_data + (v492_data * v528_data));
          }
          if (v18_lead < 12) {
            double v536_data = r0[11];
            double v537_data = s0[11];
            double v539_data = ir2[0];
            ir2[0] = (v539_data + (v536_data * v537_data));
            double v542_data = s0[27];
            double v544_data = ir2[1];
            ir2[1] = (v544_data + (v536_data * v542_data));
            double v547_data = s0[43];
            double v549_data = ir2[2];
            ir2[2] = (v549_data + (v536_data * v547_data));
            double v552_data = s0[59];
            double v554_data = ir2[3];
            ir2[3] = (v554_data + (v536_data * v552_data));
            double v557_data = s0[75];
            double v559_data = ir2[4];
            ir2[4] = (v559_data + (v536_data * v557_data));
            double v562_data = s0[91];
            double v564_data = ir2[5];
            ir2[5] = (v564_data + (v536_data * v562_data));
            double v567_data = s0[107];
            double v569_data = ir2[6];
            ir2[6] = (v569_data + (v536_data * v567_data));
            double v572_data = s0[123];
            double v574_data = ir2[7];
            ir2[7] = (v574_data + (v536_data * v572_data));
          }
          if (v18_lead < 12) {
            double v580_data = r0[12];
            double v581_data = s0[12];
            double v583_data = ir2[0];
            ir2[0] = (v583_data + (v580_data * v581_data));
            double v586_data = s0[28];
            double v588_data = ir2[1];
            ir2[1] = (v588_data + (v580_data * v586_data));
            double v591_data = s0[44];
            double v593_data = ir2[2];
            ir2[2] = (v593_data + (v580_data * v591_data));
            double v596_data = s0[60];
            double v598_data = ir2[3];
            ir2[3] = (v598_data + (v580_data * v596_data));
            double v601_data = s0[76];
            double v603_data = ir2[4];
            ir2[4] = (v603_data + (v580_data * v601_data));
            double v606_data = s0[92];
            double v608_data = ir2[5];
            ir2[5] = (v608_data + (v580_data * v606_data));
            double v611_data = s0[108];
            double v613_data = ir2[6];
            ir2[6] = (v613_data + (v580_data * v611_data));
            double v616_data = s0[124];
            double v618_data = ir2[7];
            ir2[7] = (v618_data + (v580_data * v616_data));
          }
          if (v18_lead < 12) {
            double v624_data = r0[13];
            double v625_data = s0[13];
            double v627_data = ir2[0];
            ir2[0] = (v627_data + (v624_data * v625_data));
            double v630_data = s0[29];
            double v632_data = ir2[1];
            ir2[1] = (v632_data + (v624_data * v630_data));
            double v635_data = s0[45];
            double v637_data = ir2[2];
            ir2[2] = (v637_data + (v624_data * v635_data));
            double v640_data = s0[61];
            double v642_data = ir2[3];
            ir2[3] = (v642_data + (v624_data * v640_data));
            double v645_data = s0[77];
            double v647_data = ir2[4];
            ir2[4] = (v647_data + (v624_data * v645_data));
            double v650_data = s0[93];
            double v652_data = ir2[5];
            ir2[5] = (v652_data + (v624_data * v650_data));
            double v655_data = s0[109];
            double v657_data = ir2[6];
            ir2[6] = (v657_data + (v624_data * v655_data));
            double v660_data = s0[125];
            double v662_data = ir2[7];
            ir2[7] = (v662_data + (v624_data * v660_data));
          }
          if (v18_lead < 12) {
            double v668_data = r0[14];
            double v669_data = s0[14];
            double v671_data = ir2[0];
            ir2[0] = (v671_data + (v668_data * v669_data));
            double v674_data = s0[30];
            double v676_data = ir2[1];
            ir2[1] = (v676_data + (v668_data * v674_data));
            double v679_data = s0[46];
            double v681_data = ir2[2];
            ir2[2] = (v681_data + (v668_data * v679_data));
            double v684_data = s0[62];
            double v686_data = ir2[3];
            ir2[3] = (v686_data + (v668_data * v684_data));
            double v689_data = s0[78];
            double v691_data = ir2[4];
            ir2[4] = (v691_data + (v668_data * v689_data));
            double v694_data = s0[94];
            double v696_data = ir2[5];
            ir2[5] = (v696_data + (v668_data * v694_data));
            double v699_data = s0[110];
            double v701_data = ir2[6];
            ir2[6] = (v701_data + (v668_data * v699_data));
            double v704_data = s0[126];
            double v706_data = ir2[7];
            ir2[7] = (v706_data + (v668_data * v704_data));
          }
          if (v18_lead < 12) {
            double v712_data = r0[15];
            double v713_data = s0[15];
            double v715_data = ir2[0];
            ir2[0] = (v715_data + (v712_data * v713_data));
            double v718_data = s0[31];
            double v720_data = ir2[1];
            ir2[1] = (v720_data + (v712_data * v718_data));
            double v723_data = s0[47];
            double v725_data = ir2[2];
            ir2[2] = (v725_data + (v712_data * v723_data));
            double v728_data = s0[63];
            double v730_data = ir2[3];
            ir2[3] = (v730_data + (v712_data * v728_data));
            double v733_data = s0[79];
            double v735_data = ir2[4];
            ir2[4] = (v735_data + (v712_data * v733_data));
            double v738_data = s0[95];
            double v740_data = ir2[5];
            ir2[5] = (v740_data + (v712_data * v738_data));
            double v743_data = s0[111];
            double v745_data = ir2[6];
            ir2[6] = (v745_data + (v712_data * v743_data));
            double v748_data = s0[127];
            double v750_data = ir2[7];
            ir2[7] = (v750_data + (v712_data * v748_data));
          }
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v756_n1 = 0; v756_n1 < 8; ++v756_n1) {
              double v758_data = ir2[v756_n1];
              double v760_data = r1[v756_n1];
              r2[v756_n1] = (v760_data + v758_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v18_lead < 12) {
            #pragma unroll
            for (int32_t v767_i1 = 0; v767_i1 < 8; ++v767_i1) {
              double v769_data = r2[v767_i1];
              glb_m0[(v18_lead + (v767_i1 * 12))] = v769_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

