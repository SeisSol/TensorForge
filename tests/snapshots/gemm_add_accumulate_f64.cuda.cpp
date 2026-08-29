// === base name ===
kernel_16c847f49d

// === header ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_16c847f49d, block.x * block.y * block.z, 2304 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_16c847f49d, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_16c847f49d<<<grid,block,2304 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              double v20_data = __ldcg(&glb_m1[(v10_lead + (v12_i1 * 12))]);
              r0[v12_i1] = v20_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
              double v37_data = glb_m0[(v10_lead + (v29_i1 * 12))];
              r1[v29_i1] = v37_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          if (v10_lead < 12) {
            double v45_data = r0[0];
            double v46_data = s0[0];
            double v48_data = ir2[0];
            ir2[0] = (v48_data + (v45_data * v46_data));
            double v51_data = s0[17];
            double v53_data = ir2[1];
            ir2[1] = (v53_data + (v45_data * v51_data));
            double v56_data = s0[34];
            double v58_data = ir2[2];
            ir2[2] = (v58_data + (v45_data * v56_data));
            double v61_data = s0[51];
            double v63_data = ir2[3];
            ir2[3] = (v63_data + (v45_data * v61_data));
            double v66_data = s0[68];
            double v68_data = ir2[4];
            ir2[4] = (v68_data + (v45_data * v66_data));
            double v71_data = s0[85];
            double v73_data = ir2[5];
            ir2[5] = (v73_data + (v45_data * v71_data));
            double v76_data = s0[102];
            double v78_data = ir2[6];
            ir2[6] = (v78_data + (v45_data * v76_data));
            double v81_data = s0[119];
            double v83_data = ir2[7];
            ir2[7] = (v83_data + (v45_data * v81_data));
          }
          if (v10_lead < 12) {
            double v89_data = r0[1];
            double v90_data = s0[1];
            double v92_data = ir2[0];
            ir2[0] = (v92_data + (v89_data * v90_data));
            double v95_data = s0[16];
            double v97_data = ir2[1];
            ir2[1] = (v97_data + (v89_data * v95_data));
            double v100_data = s0[35];
            double v102_data = ir2[2];
            ir2[2] = (v102_data + (v89_data * v100_data));
            double v105_data = s0[50];
            double v107_data = ir2[3];
            ir2[3] = (v107_data + (v89_data * v105_data));
            double v110_data = s0[69];
            double v112_data = ir2[4];
            ir2[4] = (v112_data + (v89_data * v110_data));
            double v115_data = s0[84];
            double v117_data = ir2[5];
            ir2[5] = (v117_data + (v89_data * v115_data));
            double v120_data = s0[103];
            double v122_data = ir2[6];
            ir2[6] = (v122_data + (v89_data * v120_data));
            double v125_data = s0[118];
            double v127_data = ir2[7];
            ir2[7] = (v127_data + (v89_data * v125_data));
          }
          if (v10_lead < 12) {
            double v133_data = r0[2];
            double v134_data = s0[2];
            double v136_data = ir2[0];
            ir2[0] = (v136_data + (v133_data * v134_data));
            double v139_data = s0[19];
            double v141_data = ir2[1];
            ir2[1] = (v141_data + (v133_data * v139_data));
            double v144_data = s0[32];
            double v146_data = ir2[2];
            ir2[2] = (v146_data + (v133_data * v144_data));
            double v149_data = s0[49];
            double v151_data = ir2[3];
            ir2[3] = (v151_data + (v133_data * v149_data));
            double v154_data = s0[70];
            double v156_data = ir2[4];
            ir2[4] = (v156_data + (v133_data * v154_data));
            double v159_data = s0[87];
            double v161_data = ir2[5];
            ir2[5] = (v161_data + (v133_data * v159_data));
            double v164_data = s0[100];
            double v166_data = ir2[6];
            ir2[6] = (v166_data + (v133_data * v164_data));
            double v169_data = s0[117];
            double v171_data = ir2[7];
            ir2[7] = (v171_data + (v133_data * v169_data));
          }
          if (v10_lead < 12) {
            double v177_data = r0[3];
            double v178_data = s0[3];
            double v180_data = ir2[0];
            ir2[0] = (v180_data + (v177_data * v178_data));
            double v183_data = s0[18];
            double v185_data = ir2[1];
            ir2[1] = (v185_data + (v177_data * v183_data));
            double v188_data = s0[33];
            double v190_data = ir2[2];
            ir2[2] = (v190_data + (v177_data * v188_data));
            double v193_data = s0[48];
            double v195_data = ir2[3];
            ir2[3] = (v195_data + (v177_data * v193_data));
            double v198_data = s0[71];
            double v200_data = ir2[4];
            ir2[4] = (v200_data + (v177_data * v198_data));
            double v203_data = s0[86];
            double v205_data = ir2[5];
            ir2[5] = (v205_data + (v177_data * v203_data));
            double v208_data = s0[101];
            double v210_data = ir2[6];
            ir2[6] = (v210_data + (v177_data * v208_data));
            double v213_data = s0[116];
            double v215_data = ir2[7];
            ir2[7] = (v215_data + (v177_data * v213_data));
          }
          if (v10_lead < 12) {
            double v221_data = r0[4];
            double v222_data = s0[4];
            double v224_data = ir2[0];
            ir2[0] = (v224_data + (v221_data * v222_data));
            double v227_data = s0[21];
            double v229_data = ir2[1];
            ir2[1] = (v229_data + (v221_data * v227_data));
            double v232_data = s0[38];
            double v234_data = ir2[2];
            ir2[2] = (v234_data + (v221_data * v232_data));
            double v237_data = s0[55];
            double v239_data = ir2[3];
            ir2[3] = (v239_data + (v221_data * v237_data));
            double v242_data = s0[64];
            double v244_data = ir2[4];
            ir2[4] = (v244_data + (v221_data * v242_data));
            double v247_data = s0[81];
            double v249_data = ir2[5];
            ir2[5] = (v249_data + (v221_data * v247_data));
            double v252_data = s0[98];
            double v254_data = ir2[6];
            ir2[6] = (v254_data + (v221_data * v252_data));
            double v257_data = s0[115];
            double v259_data = ir2[7];
            ir2[7] = (v259_data + (v221_data * v257_data));
          }
          if (v10_lead < 12) {
            double v265_data = r0[5];
            double v266_data = s0[5];
            double v268_data = ir2[0];
            ir2[0] = (v268_data + (v265_data * v266_data));
            double v271_data = s0[20];
            double v273_data = ir2[1];
            ir2[1] = (v273_data + (v265_data * v271_data));
            double v276_data = s0[39];
            double v278_data = ir2[2];
            ir2[2] = (v278_data + (v265_data * v276_data));
            double v281_data = s0[54];
            double v283_data = ir2[3];
            ir2[3] = (v283_data + (v265_data * v281_data));
            double v286_data = s0[65];
            double v288_data = ir2[4];
            ir2[4] = (v288_data + (v265_data * v286_data));
            double v291_data = s0[80];
            double v293_data = ir2[5];
            ir2[5] = (v293_data + (v265_data * v291_data));
            double v296_data = s0[99];
            double v298_data = ir2[6];
            ir2[6] = (v298_data + (v265_data * v296_data));
            double v301_data = s0[114];
            double v303_data = ir2[7];
            ir2[7] = (v303_data + (v265_data * v301_data));
          }
          if (v10_lead < 12) {
            double v309_data = r0[6];
            double v310_data = s0[6];
            double v312_data = ir2[0];
            ir2[0] = (v312_data + (v309_data * v310_data));
            double v315_data = s0[23];
            double v317_data = ir2[1];
            ir2[1] = (v317_data + (v309_data * v315_data));
            double v320_data = s0[36];
            double v322_data = ir2[2];
            ir2[2] = (v322_data + (v309_data * v320_data));
            double v325_data = s0[53];
            double v327_data = ir2[3];
            ir2[3] = (v327_data + (v309_data * v325_data));
            double v330_data = s0[66];
            double v332_data = ir2[4];
            ir2[4] = (v332_data + (v309_data * v330_data));
            double v335_data = s0[83];
            double v337_data = ir2[5];
            ir2[5] = (v337_data + (v309_data * v335_data));
            double v340_data = s0[96];
            double v342_data = ir2[6];
            ir2[6] = (v342_data + (v309_data * v340_data));
            double v345_data = s0[113];
            double v347_data = ir2[7];
            ir2[7] = (v347_data + (v309_data * v345_data));
          }
          if (v10_lead < 12) {
            double v353_data = r0[7];
            double v354_data = s0[7];
            double v356_data = ir2[0];
            ir2[0] = (v356_data + (v353_data * v354_data));
            double v359_data = s0[22];
            double v361_data = ir2[1];
            ir2[1] = (v361_data + (v353_data * v359_data));
            double v364_data = s0[37];
            double v366_data = ir2[2];
            ir2[2] = (v366_data + (v353_data * v364_data));
            double v369_data = s0[52];
            double v371_data = ir2[3];
            ir2[3] = (v371_data + (v353_data * v369_data));
            double v374_data = s0[67];
            double v376_data = ir2[4];
            ir2[4] = (v376_data + (v353_data * v374_data));
            double v379_data = s0[82];
            double v381_data = ir2[5];
            ir2[5] = (v381_data + (v353_data * v379_data));
            double v384_data = s0[97];
            double v386_data = ir2[6];
            ir2[6] = (v386_data + (v353_data * v384_data));
            double v389_data = s0[112];
            double v391_data = ir2[7];
            ir2[7] = (v391_data + (v353_data * v389_data));
          }
          if (v10_lead < 12) {
            double v397_data = r0[8];
            double v398_data = s0[8];
            double v400_data = ir2[0];
            ir2[0] = (v400_data + (v397_data * v398_data));
            double v403_data = s0[25];
            double v405_data = ir2[1];
            ir2[1] = (v405_data + (v397_data * v403_data));
            double v408_data = s0[42];
            double v410_data = ir2[2];
            ir2[2] = (v410_data + (v397_data * v408_data));
            double v413_data = s0[59];
            double v415_data = ir2[3];
            ir2[3] = (v415_data + (v397_data * v413_data));
            double v418_data = s0[76];
            double v420_data = ir2[4];
            ir2[4] = (v420_data + (v397_data * v418_data));
            double v423_data = s0[93];
            double v425_data = ir2[5];
            ir2[5] = (v425_data + (v397_data * v423_data));
            double v428_data = s0[110];
            double v430_data = ir2[6];
            ir2[6] = (v430_data + (v397_data * v428_data));
            double v433_data = s0[127];
            double v435_data = ir2[7];
            ir2[7] = (v435_data + (v397_data * v433_data));
          }
          if (v10_lead < 12) {
            double v441_data = r0[9];
            double v442_data = s0[9];
            double v444_data = ir2[0];
            ir2[0] = (v444_data + (v441_data * v442_data));
            double v447_data = s0[24];
            double v449_data = ir2[1];
            ir2[1] = (v449_data + (v441_data * v447_data));
            double v452_data = s0[43];
            double v454_data = ir2[2];
            ir2[2] = (v454_data + (v441_data * v452_data));
            double v457_data = s0[58];
            double v459_data = ir2[3];
            ir2[3] = (v459_data + (v441_data * v457_data));
            double v462_data = s0[77];
            double v464_data = ir2[4];
            ir2[4] = (v464_data + (v441_data * v462_data));
            double v467_data = s0[92];
            double v469_data = ir2[5];
            ir2[5] = (v469_data + (v441_data * v467_data));
            double v472_data = s0[111];
            double v474_data = ir2[6];
            ir2[6] = (v474_data + (v441_data * v472_data));
            double v477_data = s0[126];
            double v479_data = ir2[7];
            ir2[7] = (v479_data + (v441_data * v477_data));
          }
          if (v10_lead < 12) {
            double v485_data = r0[10];
            double v486_data = s0[10];
            double v488_data = ir2[0];
            ir2[0] = (v488_data + (v485_data * v486_data));
            double v491_data = s0[27];
            double v493_data = ir2[1];
            ir2[1] = (v493_data + (v485_data * v491_data));
            double v496_data = s0[40];
            double v498_data = ir2[2];
            ir2[2] = (v498_data + (v485_data * v496_data));
            double v501_data = s0[57];
            double v503_data = ir2[3];
            ir2[3] = (v503_data + (v485_data * v501_data));
            double v506_data = s0[78];
            double v508_data = ir2[4];
            ir2[4] = (v508_data + (v485_data * v506_data));
            double v511_data = s0[95];
            double v513_data = ir2[5];
            ir2[5] = (v513_data + (v485_data * v511_data));
            double v516_data = s0[108];
            double v518_data = ir2[6];
            ir2[6] = (v518_data + (v485_data * v516_data));
            double v521_data = s0[125];
            double v523_data = ir2[7];
            ir2[7] = (v523_data + (v485_data * v521_data));
          }
          if (v10_lead < 12) {
            double v529_data = r0[11];
            double v530_data = s0[11];
            double v532_data = ir2[0];
            ir2[0] = (v532_data + (v529_data * v530_data));
            double v535_data = s0[26];
            double v537_data = ir2[1];
            ir2[1] = (v537_data + (v529_data * v535_data));
            double v540_data = s0[41];
            double v542_data = ir2[2];
            ir2[2] = (v542_data + (v529_data * v540_data));
            double v545_data = s0[56];
            double v547_data = ir2[3];
            ir2[3] = (v547_data + (v529_data * v545_data));
            double v550_data = s0[79];
            double v552_data = ir2[4];
            ir2[4] = (v552_data + (v529_data * v550_data));
            double v555_data = s0[94];
            double v557_data = ir2[5];
            ir2[5] = (v557_data + (v529_data * v555_data));
            double v560_data = s0[109];
            double v562_data = ir2[6];
            ir2[6] = (v562_data + (v529_data * v560_data));
            double v565_data = s0[124];
            double v567_data = ir2[7];
            ir2[7] = (v567_data + (v529_data * v565_data));
          }
          if (v10_lead < 12) {
            double v573_data = r0[12];
            double v574_data = s0[12];
            double v576_data = ir2[0];
            ir2[0] = (v576_data + (v573_data * v574_data));
            double v579_data = s0[29];
            double v581_data = ir2[1];
            ir2[1] = (v581_data + (v573_data * v579_data));
            double v584_data = s0[46];
            double v586_data = ir2[2];
            ir2[2] = (v586_data + (v573_data * v584_data));
            double v589_data = s0[63];
            double v591_data = ir2[3];
            ir2[3] = (v591_data + (v573_data * v589_data));
            double v594_data = s0[72];
            double v596_data = ir2[4];
            ir2[4] = (v596_data + (v573_data * v594_data));
            double v599_data = s0[89];
            double v601_data = ir2[5];
            ir2[5] = (v601_data + (v573_data * v599_data));
            double v604_data = s0[106];
            double v606_data = ir2[6];
            ir2[6] = (v606_data + (v573_data * v604_data));
            double v609_data = s0[123];
            double v611_data = ir2[7];
            ir2[7] = (v611_data + (v573_data * v609_data));
          }
          if (v10_lead < 12) {
            double v617_data = r0[13];
            double v618_data = s0[13];
            double v620_data = ir2[0];
            ir2[0] = (v620_data + (v617_data * v618_data));
            double v623_data = s0[28];
            double v625_data = ir2[1];
            ir2[1] = (v625_data + (v617_data * v623_data));
            double v628_data = s0[47];
            double v630_data = ir2[2];
            ir2[2] = (v630_data + (v617_data * v628_data));
            double v633_data = s0[62];
            double v635_data = ir2[3];
            ir2[3] = (v635_data + (v617_data * v633_data));
            double v638_data = s0[73];
            double v640_data = ir2[4];
            ir2[4] = (v640_data + (v617_data * v638_data));
            double v643_data = s0[88];
            double v645_data = ir2[5];
            ir2[5] = (v645_data + (v617_data * v643_data));
            double v648_data = s0[107];
            double v650_data = ir2[6];
            ir2[6] = (v650_data + (v617_data * v648_data));
            double v653_data = s0[122];
            double v655_data = ir2[7];
            ir2[7] = (v655_data + (v617_data * v653_data));
          }
          if (v10_lead < 12) {
            double v661_data = r0[14];
            double v662_data = s0[14];
            double v664_data = ir2[0];
            ir2[0] = (v664_data + (v661_data * v662_data));
            double v667_data = s0[31];
            double v669_data = ir2[1];
            ir2[1] = (v669_data + (v661_data * v667_data));
            double v672_data = s0[44];
            double v674_data = ir2[2];
            ir2[2] = (v674_data + (v661_data * v672_data));
            double v677_data = s0[61];
            double v679_data = ir2[3];
            ir2[3] = (v679_data + (v661_data * v677_data));
            double v682_data = s0[74];
            double v684_data = ir2[4];
            ir2[4] = (v684_data + (v661_data * v682_data));
            double v687_data = s0[91];
            double v689_data = ir2[5];
            ir2[5] = (v689_data + (v661_data * v687_data));
            double v692_data = s0[104];
            double v694_data = ir2[6];
            ir2[6] = (v694_data + (v661_data * v692_data));
            double v697_data = s0[121];
            double v699_data = ir2[7];
            ir2[7] = (v699_data + (v661_data * v697_data));
          }
          if (v10_lead < 12) {
            double v705_data = r0[15];
            double v706_data = s0[15];
            double v708_data = ir2[0];
            ir2[0] = (v708_data + (v705_data * v706_data));
            double v711_data = s0[30];
            double v713_data = ir2[1];
            ir2[1] = (v713_data + (v705_data * v711_data));
            double v716_data = s0[45];
            double v718_data = ir2[2];
            ir2[2] = (v718_data + (v705_data * v716_data));
            double v721_data = s0[60];
            double v723_data = ir2[3];
            ir2[3] = (v723_data + (v705_data * v721_data));
            double v726_data = s0[75];
            double v728_data = ir2[4];
            ir2[4] = (v728_data + (v705_data * v726_data));
            double v731_data = s0[90];
            double v733_data = ir2[5];
            ir2[5] = (v733_data + (v705_data * v731_data));
            double v736_data = s0[105];
            double v738_data = ir2[6];
            ir2[6] = (v738_data + (v705_data * v736_data));
            double v741_data = s0[120];
            double v743_data = ir2[7];
            ir2[7] = (v743_data + (v705_data * v741_data));
          }
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v749_n1 = 0; v749_n1 < 8; ++v749_n1) {
              double v751_data = ir2[v749_n1];
              double v753_data = r1[v749_n1];
              r2[v749_n1] = (v753_data + v751_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v760_i1 = 0; v760_i1 < 8; ++v760_i1) {
              double v762_data = r2[v760_i1];
              glb_m0[(v10_lead + (v760_i1 * 12))] = v762_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

