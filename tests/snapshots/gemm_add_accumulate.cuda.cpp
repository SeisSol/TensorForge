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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              float v20_data = __ldcg(&glb_m1[(v10_lead + (v12_i1 * 12))]);
              r0[v12_i1] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
              float v37_data = glb_m0[(v10_lead + (v29_i1 * 12))];
              r1[v29_i1] = v37_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          float r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir2[8]{};
          if (v10_lead < 12) {
            float v45_data = r0[0];
            float v46_data = s0[0];
            float v48_data = ir2[0];
            ir2[0] = (v48_data + (v45_data * v46_data));
            float v51_data = s0[16];
            float v53_data = ir2[1];
            ir2[1] = (v53_data + (v45_data * v51_data));
            float v56_data = s0[33];
            float v58_data = ir2[2];
            ir2[2] = (v58_data + (v45_data * v56_data));
            float v61_data = s0[49];
            float v63_data = ir2[3];
            ir2[3] = (v63_data + (v45_data * v61_data));
            float v66_data = s0[66];
            float v68_data = ir2[4];
            ir2[4] = (v68_data + (v45_data * v66_data));
            float v71_data = s0[82];
            float v73_data = ir2[5];
            ir2[5] = (v73_data + (v45_data * v71_data));
            float v76_data = s0[99];
            float v78_data = ir2[6];
            ir2[6] = (v78_data + (v45_data * v76_data));
            float v81_data = s0[115];
            float v83_data = ir2[7];
            ir2[7] = (v83_data + (v45_data * v81_data));
          }
          if (v10_lead < 12) {
            float v89_data = r0[1];
            float v90_data = s0[1];
            float v92_data = ir2[0];
            ir2[0] = (v92_data + (v89_data * v90_data));
            float v95_data = s0[17];
            float v97_data = ir2[1];
            ir2[1] = (v97_data + (v89_data * v95_data));
            float v100_data = s0[32];
            float v102_data = ir2[2];
            ir2[2] = (v102_data + (v89_data * v100_data));
            float v105_data = s0[48];
            float v107_data = ir2[3];
            ir2[3] = (v107_data + (v89_data * v105_data));
            float v110_data = s0[67];
            float v112_data = ir2[4];
            ir2[4] = (v112_data + (v89_data * v110_data));
            float v115_data = s0[83];
            float v117_data = ir2[5];
            ir2[5] = (v117_data + (v89_data * v115_data));
            float v120_data = s0[98];
            float v122_data = ir2[6];
            ir2[6] = (v122_data + (v89_data * v120_data));
            float v125_data = s0[114];
            float v127_data = ir2[7];
            ir2[7] = (v127_data + (v89_data * v125_data));
          }
          if (v10_lead < 12) {
            float v133_data = r0[2];
            float v134_data = s0[2];
            float v136_data = ir2[0];
            ir2[0] = (v136_data + (v133_data * v134_data));
            float v139_data = s0[18];
            float v141_data = ir2[1];
            ir2[1] = (v141_data + (v133_data * v139_data));
            float v144_data = s0[35];
            float v146_data = ir2[2];
            ir2[2] = (v146_data + (v133_data * v144_data));
            float v149_data = s0[51];
            float v151_data = ir2[3];
            ir2[3] = (v151_data + (v133_data * v149_data));
            float v154_data = s0[64];
            float v156_data = ir2[4];
            ir2[4] = (v156_data + (v133_data * v154_data));
            float v159_data = s0[80];
            float v161_data = ir2[5];
            ir2[5] = (v161_data + (v133_data * v159_data));
            float v164_data = s0[97];
            float v166_data = ir2[6];
            ir2[6] = (v166_data + (v133_data * v164_data));
            float v169_data = s0[113];
            float v171_data = ir2[7];
            ir2[7] = (v171_data + (v133_data * v169_data));
          }
          if (v10_lead < 12) {
            float v177_data = r0[3];
            float v178_data = s0[3];
            float v180_data = ir2[0];
            ir2[0] = (v180_data + (v177_data * v178_data));
            float v183_data = s0[19];
            float v185_data = ir2[1];
            ir2[1] = (v185_data + (v177_data * v183_data));
            float v188_data = s0[34];
            float v190_data = ir2[2];
            ir2[2] = (v190_data + (v177_data * v188_data));
            float v193_data = s0[50];
            float v195_data = ir2[3];
            ir2[3] = (v195_data + (v177_data * v193_data));
            float v198_data = s0[65];
            float v200_data = ir2[4];
            ir2[4] = (v200_data + (v177_data * v198_data));
            float v203_data = s0[81];
            float v205_data = ir2[5];
            ir2[5] = (v205_data + (v177_data * v203_data));
            float v208_data = s0[96];
            float v210_data = ir2[6];
            ir2[6] = (v210_data + (v177_data * v208_data));
            float v213_data = s0[112];
            float v215_data = ir2[7];
            ir2[7] = (v215_data + (v177_data * v213_data));
          }
          if (v10_lead < 12) {
            float v221_data = r0[4];
            float v222_data = s0[4];
            float v224_data = ir2[0];
            ir2[0] = (v224_data + (v221_data * v222_data));
            float v227_data = s0[20];
            float v229_data = ir2[1];
            ir2[1] = (v229_data + (v221_data * v227_data));
            float v232_data = s0[37];
            float v234_data = ir2[2];
            ir2[2] = (v234_data + (v221_data * v232_data));
            float v237_data = s0[53];
            float v239_data = ir2[3];
            ir2[3] = (v239_data + (v221_data * v237_data));
            float v242_data = s0[70];
            float v244_data = ir2[4];
            ir2[4] = (v244_data + (v221_data * v242_data));
            float v247_data = s0[86];
            float v249_data = ir2[5];
            ir2[5] = (v249_data + (v221_data * v247_data));
            float v252_data = s0[103];
            float v254_data = ir2[6];
            ir2[6] = (v254_data + (v221_data * v252_data));
            float v257_data = s0[119];
            float v259_data = ir2[7];
            ir2[7] = (v259_data + (v221_data * v257_data));
          }
          if (v10_lead < 12) {
            float v265_data = r0[5];
            float v266_data = s0[5];
            float v268_data = ir2[0];
            ir2[0] = (v268_data + (v265_data * v266_data));
            float v271_data = s0[21];
            float v273_data = ir2[1];
            ir2[1] = (v273_data + (v265_data * v271_data));
            float v276_data = s0[36];
            float v278_data = ir2[2];
            ir2[2] = (v278_data + (v265_data * v276_data));
            float v281_data = s0[52];
            float v283_data = ir2[3];
            ir2[3] = (v283_data + (v265_data * v281_data));
            float v286_data = s0[71];
            float v288_data = ir2[4];
            ir2[4] = (v288_data + (v265_data * v286_data));
            float v291_data = s0[87];
            float v293_data = ir2[5];
            ir2[5] = (v293_data + (v265_data * v291_data));
            float v296_data = s0[102];
            float v298_data = ir2[6];
            ir2[6] = (v298_data + (v265_data * v296_data));
            float v301_data = s0[118];
            float v303_data = ir2[7];
            ir2[7] = (v303_data + (v265_data * v301_data));
          }
          if (v10_lead < 12) {
            float v309_data = r0[6];
            float v310_data = s0[6];
            float v312_data = ir2[0];
            ir2[0] = (v312_data + (v309_data * v310_data));
            float v315_data = s0[22];
            float v317_data = ir2[1];
            ir2[1] = (v317_data + (v309_data * v315_data));
            float v320_data = s0[39];
            float v322_data = ir2[2];
            ir2[2] = (v322_data + (v309_data * v320_data));
            float v325_data = s0[55];
            float v327_data = ir2[3];
            ir2[3] = (v327_data + (v309_data * v325_data));
            float v330_data = s0[68];
            float v332_data = ir2[4];
            ir2[4] = (v332_data + (v309_data * v330_data));
            float v335_data = s0[84];
            float v337_data = ir2[5];
            ir2[5] = (v337_data + (v309_data * v335_data));
            float v340_data = s0[101];
            float v342_data = ir2[6];
            ir2[6] = (v342_data + (v309_data * v340_data));
            float v345_data = s0[117];
            float v347_data = ir2[7];
            ir2[7] = (v347_data + (v309_data * v345_data));
          }
          if (v10_lead < 12) {
            float v353_data = r0[7];
            float v354_data = s0[7];
            float v356_data = ir2[0];
            ir2[0] = (v356_data + (v353_data * v354_data));
            float v359_data = s0[23];
            float v361_data = ir2[1];
            ir2[1] = (v361_data + (v353_data * v359_data));
            float v364_data = s0[38];
            float v366_data = ir2[2];
            ir2[2] = (v366_data + (v353_data * v364_data));
            float v369_data = s0[54];
            float v371_data = ir2[3];
            ir2[3] = (v371_data + (v353_data * v369_data));
            float v374_data = s0[69];
            float v376_data = ir2[4];
            ir2[4] = (v376_data + (v353_data * v374_data));
            float v379_data = s0[85];
            float v381_data = ir2[5];
            ir2[5] = (v381_data + (v353_data * v379_data));
            float v384_data = s0[100];
            float v386_data = ir2[6];
            ir2[6] = (v386_data + (v353_data * v384_data));
            float v389_data = s0[116];
            float v391_data = ir2[7];
            ir2[7] = (v391_data + (v353_data * v389_data));
          }
          if (v10_lead < 12) {
            float v397_data = r0[8];
            float v398_data = s0[8];
            float v400_data = ir2[0];
            ir2[0] = (v400_data + (v397_data * v398_data));
            float v403_data = s0[24];
            float v405_data = ir2[1];
            ir2[1] = (v405_data + (v397_data * v403_data));
            float v408_data = s0[41];
            float v410_data = ir2[2];
            ir2[2] = (v410_data + (v397_data * v408_data));
            float v413_data = s0[57];
            float v415_data = ir2[3];
            ir2[3] = (v415_data + (v397_data * v413_data));
            float v418_data = s0[74];
            float v420_data = ir2[4];
            ir2[4] = (v420_data + (v397_data * v418_data));
            float v423_data = s0[90];
            float v425_data = ir2[5];
            ir2[5] = (v425_data + (v397_data * v423_data));
            float v428_data = s0[107];
            float v430_data = ir2[6];
            ir2[6] = (v430_data + (v397_data * v428_data));
            float v433_data = s0[123];
            float v435_data = ir2[7];
            ir2[7] = (v435_data + (v397_data * v433_data));
          }
          if (v10_lead < 12) {
            float v441_data = r0[9];
            float v442_data = s0[9];
            float v444_data = ir2[0];
            ir2[0] = (v444_data + (v441_data * v442_data));
            float v447_data = s0[25];
            float v449_data = ir2[1];
            ir2[1] = (v449_data + (v441_data * v447_data));
            float v452_data = s0[40];
            float v454_data = ir2[2];
            ir2[2] = (v454_data + (v441_data * v452_data));
            float v457_data = s0[56];
            float v459_data = ir2[3];
            ir2[3] = (v459_data + (v441_data * v457_data));
            float v462_data = s0[75];
            float v464_data = ir2[4];
            ir2[4] = (v464_data + (v441_data * v462_data));
            float v467_data = s0[91];
            float v469_data = ir2[5];
            ir2[5] = (v469_data + (v441_data * v467_data));
            float v472_data = s0[106];
            float v474_data = ir2[6];
            ir2[6] = (v474_data + (v441_data * v472_data));
            float v477_data = s0[122];
            float v479_data = ir2[7];
            ir2[7] = (v479_data + (v441_data * v477_data));
          }
          if (v10_lead < 12) {
            float v485_data = r0[10];
            float v486_data = s0[10];
            float v488_data = ir2[0];
            ir2[0] = (v488_data + (v485_data * v486_data));
            float v491_data = s0[26];
            float v493_data = ir2[1];
            ir2[1] = (v493_data + (v485_data * v491_data));
            float v496_data = s0[43];
            float v498_data = ir2[2];
            ir2[2] = (v498_data + (v485_data * v496_data));
            float v501_data = s0[59];
            float v503_data = ir2[3];
            ir2[3] = (v503_data + (v485_data * v501_data));
            float v506_data = s0[72];
            float v508_data = ir2[4];
            ir2[4] = (v508_data + (v485_data * v506_data));
            float v511_data = s0[88];
            float v513_data = ir2[5];
            ir2[5] = (v513_data + (v485_data * v511_data));
            float v516_data = s0[105];
            float v518_data = ir2[6];
            ir2[6] = (v518_data + (v485_data * v516_data));
            float v521_data = s0[121];
            float v523_data = ir2[7];
            ir2[7] = (v523_data + (v485_data * v521_data));
          }
          if (v10_lead < 12) {
            float v529_data = r0[11];
            float v530_data = s0[11];
            float v532_data = ir2[0];
            ir2[0] = (v532_data + (v529_data * v530_data));
            float v535_data = s0[27];
            float v537_data = ir2[1];
            ir2[1] = (v537_data + (v529_data * v535_data));
            float v540_data = s0[42];
            float v542_data = ir2[2];
            ir2[2] = (v542_data + (v529_data * v540_data));
            float v545_data = s0[58];
            float v547_data = ir2[3];
            ir2[3] = (v547_data + (v529_data * v545_data));
            float v550_data = s0[73];
            float v552_data = ir2[4];
            ir2[4] = (v552_data + (v529_data * v550_data));
            float v555_data = s0[89];
            float v557_data = ir2[5];
            ir2[5] = (v557_data + (v529_data * v555_data));
            float v560_data = s0[104];
            float v562_data = ir2[6];
            ir2[6] = (v562_data + (v529_data * v560_data));
            float v565_data = s0[120];
            float v567_data = ir2[7];
            ir2[7] = (v567_data + (v529_data * v565_data));
          }
          if (v10_lead < 12) {
            float v573_data = r0[12];
            float v574_data = s0[12];
            float v576_data = ir2[0];
            ir2[0] = (v576_data + (v573_data * v574_data));
            float v579_data = s0[28];
            float v581_data = ir2[1];
            ir2[1] = (v581_data + (v573_data * v579_data));
            float v584_data = s0[45];
            float v586_data = ir2[2];
            ir2[2] = (v586_data + (v573_data * v584_data));
            float v589_data = s0[61];
            float v591_data = ir2[3];
            ir2[3] = (v591_data + (v573_data * v589_data));
            float v594_data = s0[78];
            float v596_data = ir2[4];
            ir2[4] = (v596_data + (v573_data * v594_data));
            float v599_data = s0[94];
            float v601_data = ir2[5];
            ir2[5] = (v601_data + (v573_data * v599_data));
            float v604_data = s0[111];
            float v606_data = ir2[6];
            ir2[6] = (v606_data + (v573_data * v604_data));
            float v609_data = s0[127];
            float v611_data = ir2[7];
            ir2[7] = (v611_data + (v573_data * v609_data));
          }
          if (v10_lead < 12) {
            float v617_data = r0[13];
            float v618_data = s0[13];
            float v620_data = ir2[0];
            ir2[0] = (v620_data + (v617_data * v618_data));
            float v623_data = s0[29];
            float v625_data = ir2[1];
            ir2[1] = (v625_data + (v617_data * v623_data));
            float v628_data = s0[44];
            float v630_data = ir2[2];
            ir2[2] = (v630_data + (v617_data * v628_data));
            float v633_data = s0[60];
            float v635_data = ir2[3];
            ir2[3] = (v635_data + (v617_data * v633_data));
            float v638_data = s0[79];
            float v640_data = ir2[4];
            ir2[4] = (v640_data + (v617_data * v638_data));
            float v643_data = s0[95];
            float v645_data = ir2[5];
            ir2[5] = (v645_data + (v617_data * v643_data));
            float v648_data = s0[110];
            float v650_data = ir2[6];
            ir2[6] = (v650_data + (v617_data * v648_data));
            float v653_data = s0[126];
            float v655_data = ir2[7];
            ir2[7] = (v655_data + (v617_data * v653_data));
          }
          if (v10_lead < 12) {
            float v661_data = r0[14];
            float v662_data = s0[14];
            float v664_data = ir2[0];
            ir2[0] = (v664_data + (v661_data * v662_data));
            float v667_data = s0[30];
            float v669_data = ir2[1];
            ir2[1] = (v669_data + (v661_data * v667_data));
            float v672_data = s0[47];
            float v674_data = ir2[2];
            ir2[2] = (v674_data + (v661_data * v672_data));
            float v677_data = s0[63];
            float v679_data = ir2[3];
            ir2[3] = (v679_data + (v661_data * v677_data));
            float v682_data = s0[76];
            float v684_data = ir2[4];
            ir2[4] = (v684_data + (v661_data * v682_data));
            float v687_data = s0[92];
            float v689_data = ir2[5];
            ir2[5] = (v689_data + (v661_data * v687_data));
            float v692_data = s0[109];
            float v694_data = ir2[6];
            ir2[6] = (v694_data + (v661_data * v692_data));
            float v697_data = s0[125];
            float v699_data = ir2[7];
            ir2[7] = (v699_data + (v661_data * v697_data));
          }
          if (v10_lead < 12) {
            float v705_data = r0[15];
            float v706_data = s0[15];
            float v708_data = ir2[0];
            ir2[0] = (v708_data + (v705_data * v706_data));
            float v711_data = s0[31];
            float v713_data = ir2[1];
            ir2[1] = (v713_data + (v705_data * v711_data));
            float v716_data = s0[46];
            float v718_data = ir2[2];
            ir2[2] = (v718_data + (v705_data * v716_data));
            float v721_data = s0[62];
            float v723_data = ir2[3];
            ir2[3] = (v723_data + (v705_data * v721_data));
            float v726_data = s0[77];
            float v728_data = ir2[4];
            ir2[4] = (v728_data + (v705_data * v726_data));
            float v731_data = s0[93];
            float v733_data = ir2[5];
            ir2[5] = (v733_data + (v705_data * v731_data));
            float v736_data = s0[108];
            float v738_data = ir2[6];
            ir2[6] = (v738_data + (v705_data * v736_data));
            float v741_data = s0[124];
            float v743_data = ir2[7];
            ir2[7] = (v743_data + (v705_data * v741_data));
          }
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v749_n1 = 0; v749_n1 < 8; ++v749_n1) {
              float v751_data = ir2[v749_n1];
              float v753_data = r1[v749_n1];
              r2[v749_n1] = (v753_data + v751_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v760_i1 = 0; v760_i1 < 8; ++v760_i1) {
              float v762_data = r2[v760_i1];
              glb_m0[(v10_lead + (v760_i1 * 12))] = v762_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

