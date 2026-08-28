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
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          alignas(16) double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v6_lead = threadIdx.x % 16;
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 16; ++v8_i1) {
              int32_t v14_a = v8_i1 * 12;
              int32_t v15_a = v6_lead + v14_a;
              double v23_data = __ldcg(&glb_m1[(v6_lead + v14_a)]);
              int32_t v24_a = 0 + v8_i1;
              r0[v24_a] = v23_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          alignas(16) double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
              int32_t v38_a = v32_i1 * 12;
              int32_t v39_a = v6_lead + v38_a;
              double v47_data = glb_m0[(v6_lead + v38_a)];
              int32_t v48_a = 0 + v32_i1;
              r1[v48_a] = v47_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          alignas(16) double r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          if (v6_lead < 12) {
            double v55_data = r0[0];
            double v56_data = s0[0];
            double v58_data = ir2[0];
            ir2[0] = (v58_data + (v55_data * v56_data));
            double v61_data = s0[16];
            double v63_data = ir2[1];
            ir2[1] = (v63_data + (v55_data * v61_data));
            double v66_data = s0[32];
            double v68_data = ir2[2];
            ir2[2] = (v68_data + (v55_data * v66_data));
            double v71_data = s0[48];
            double v73_data = ir2[3];
            ir2[3] = (v73_data + (v55_data * v71_data));
            double v76_data = s0[64];
            double v78_data = ir2[4];
            ir2[4] = (v78_data + (v55_data * v76_data));
            double v81_data = s0[80];
            double v83_data = ir2[5];
            ir2[5] = (v83_data + (v55_data * v81_data));
            double v86_data = s0[96];
            double v88_data = ir2[6];
            ir2[6] = (v88_data + (v55_data * v86_data));
            double v91_data = s0[112];
            double v93_data = ir2[7];
            ir2[7] = (v93_data + (v55_data * v91_data));
          }
          if (v6_lead < 12) {
            double v99_data = r0[1];
            double v100_data = s0[1];
            double v102_data = ir2[0];
            ir2[0] = (v102_data + (v99_data * v100_data));
            double v105_data = s0[17];
            double v107_data = ir2[1];
            ir2[1] = (v107_data + (v99_data * v105_data));
            double v110_data = s0[33];
            double v112_data = ir2[2];
            ir2[2] = (v112_data + (v99_data * v110_data));
            double v115_data = s0[49];
            double v117_data = ir2[3];
            ir2[3] = (v117_data + (v99_data * v115_data));
            double v120_data = s0[65];
            double v122_data = ir2[4];
            ir2[4] = (v122_data + (v99_data * v120_data));
            double v125_data = s0[81];
            double v127_data = ir2[5];
            ir2[5] = (v127_data + (v99_data * v125_data));
            double v130_data = s0[97];
            double v132_data = ir2[6];
            ir2[6] = (v132_data + (v99_data * v130_data));
            double v135_data = s0[113];
            double v137_data = ir2[7];
            ir2[7] = (v137_data + (v99_data * v135_data));
          }
          if (v6_lead < 12) {
            double v143_data = r0[2];
            double v144_data = s0[2];
            double v146_data = ir2[0];
            ir2[0] = (v146_data + (v143_data * v144_data));
            double v149_data = s0[18];
            double v151_data = ir2[1];
            ir2[1] = (v151_data + (v143_data * v149_data));
            double v154_data = s0[34];
            double v156_data = ir2[2];
            ir2[2] = (v156_data + (v143_data * v154_data));
            double v159_data = s0[50];
            double v161_data = ir2[3];
            ir2[3] = (v161_data + (v143_data * v159_data));
            double v164_data = s0[66];
            double v166_data = ir2[4];
            ir2[4] = (v166_data + (v143_data * v164_data));
            double v169_data = s0[82];
            double v171_data = ir2[5];
            ir2[5] = (v171_data + (v143_data * v169_data));
            double v174_data = s0[98];
            double v176_data = ir2[6];
            ir2[6] = (v176_data + (v143_data * v174_data));
            double v179_data = s0[114];
            double v181_data = ir2[7];
            ir2[7] = (v181_data + (v143_data * v179_data));
          }
          if (v6_lead < 12) {
            double v187_data = r0[3];
            double v188_data = s0[3];
            double v190_data = ir2[0];
            ir2[0] = (v190_data + (v187_data * v188_data));
            double v193_data = s0[19];
            double v195_data = ir2[1];
            ir2[1] = (v195_data + (v187_data * v193_data));
            double v198_data = s0[35];
            double v200_data = ir2[2];
            ir2[2] = (v200_data + (v187_data * v198_data));
            double v203_data = s0[51];
            double v205_data = ir2[3];
            ir2[3] = (v205_data + (v187_data * v203_data));
            double v208_data = s0[67];
            double v210_data = ir2[4];
            ir2[4] = (v210_data + (v187_data * v208_data));
            double v213_data = s0[83];
            double v215_data = ir2[5];
            ir2[5] = (v215_data + (v187_data * v213_data));
            double v218_data = s0[99];
            double v220_data = ir2[6];
            ir2[6] = (v220_data + (v187_data * v218_data));
            double v223_data = s0[115];
            double v225_data = ir2[7];
            ir2[7] = (v225_data + (v187_data * v223_data));
          }
          if (v6_lead < 12) {
            double v231_data = r0[4];
            double v232_data = s0[4];
            double v234_data = ir2[0];
            ir2[0] = (v234_data + (v231_data * v232_data));
            double v237_data = s0[20];
            double v239_data = ir2[1];
            ir2[1] = (v239_data + (v231_data * v237_data));
            double v242_data = s0[36];
            double v244_data = ir2[2];
            ir2[2] = (v244_data + (v231_data * v242_data));
            double v247_data = s0[52];
            double v249_data = ir2[3];
            ir2[3] = (v249_data + (v231_data * v247_data));
            double v252_data = s0[68];
            double v254_data = ir2[4];
            ir2[4] = (v254_data + (v231_data * v252_data));
            double v257_data = s0[84];
            double v259_data = ir2[5];
            ir2[5] = (v259_data + (v231_data * v257_data));
            double v262_data = s0[100];
            double v264_data = ir2[6];
            ir2[6] = (v264_data + (v231_data * v262_data));
            double v267_data = s0[116];
            double v269_data = ir2[7];
            ir2[7] = (v269_data + (v231_data * v267_data));
          }
          if (v6_lead < 12) {
            double v275_data = r0[5];
            double v276_data = s0[5];
            double v278_data = ir2[0];
            ir2[0] = (v278_data + (v275_data * v276_data));
            double v281_data = s0[21];
            double v283_data = ir2[1];
            ir2[1] = (v283_data + (v275_data * v281_data));
            double v286_data = s0[37];
            double v288_data = ir2[2];
            ir2[2] = (v288_data + (v275_data * v286_data));
            double v291_data = s0[53];
            double v293_data = ir2[3];
            ir2[3] = (v293_data + (v275_data * v291_data));
            double v296_data = s0[69];
            double v298_data = ir2[4];
            ir2[4] = (v298_data + (v275_data * v296_data));
            double v301_data = s0[85];
            double v303_data = ir2[5];
            ir2[5] = (v303_data + (v275_data * v301_data));
            double v306_data = s0[101];
            double v308_data = ir2[6];
            ir2[6] = (v308_data + (v275_data * v306_data));
            double v311_data = s0[117];
            double v313_data = ir2[7];
            ir2[7] = (v313_data + (v275_data * v311_data));
          }
          if (v6_lead < 12) {
            double v319_data = r0[6];
            double v320_data = s0[6];
            double v322_data = ir2[0];
            ir2[0] = (v322_data + (v319_data * v320_data));
            double v325_data = s0[22];
            double v327_data = ir2[1];
            ir2[1] = (v327_data + (v319_data * v325_data));
            double v330_data = s0[38];
            double v332_data = ir2[2];
            ir2[2] = (v332_data + (v319_data * v330_data));
            double v335_data = s0[54];
            double v337_data = ir2[3];
            ir2[3] = (v337_data + (v319_data * v335_data));
            double v340_data = s0[70];
            double v342_data = ir2[4];
            ir2[4] = (v342_data + (v319_data * v340_data));
            double v345_data = s0[86];
            double v347_data = ir2[5];
            ir2[5] = (v347_data + (v319_data * v345_data));
            double v350_data = s0[102];
            double v352_data = ir2[6];
            ir2[6] = (v352_data + (v319_data * v350_data));
            double v355_data = s0[118];
            double v357_data = ir2[7];
            ir2[7] = (v357_data + (v319_data * v355_data));
          }
          if (v6_lead < 12) {
            double v363_data = r0[7];
            double v364_data = s0[7];
            double v366_data = ir2[0];
            ir2[0] = (v366_data + (v363_data * v364_data));
            double v369_data = s0[23];
            double v371_data = ir2[1];
            ir2[1] = (v371_data + (v363_data * v369_data));
            double v374_data = s0[39];
            double v376_data = ir2[2];
            ir2[2] = (v376_data + (v363_data * v374_data));
            double v379_data = s0[55];
            double v381_data = ir2[3];
            ir2[3] = (v381_data + (v363_data * v379_data));
            double v384_data = s0[71];
            double v386_data = ir2[4];
            ir2[4] = (v386_data + (v363_data * v384_data));
            double v389_data = s0[87];
            double v391_data = ir2[5];
            ir2[5] = (v391_data + (v363_data * v389_data));
            double v394_data = s0[103];
            double v396_data = ir2[6];
            ir2[6] = (v396_data + (v363_data * v394_data));
            double v399_data = s0[119];
            double v401_data = ir2[7];
            ir2[7] = (v401_data + (v363_data * v399_data));
          }
          if (v6_lead < 12) {
            double v407_data = r0[8];
            double v408_data = s0[8];
            double v410_data = ir2[0];
            ir2[0] = (v410_data + (v407_data * v408_data));
            double v413_data = s0[24];
            double v415_data = ir2[1];
            ir2[1] = (v415_data + (v407_data * v413_data));
            double v418_data = s0[40];
            double v420_data = ir2[2];
            ir2[2] = (v420_data + (v407_data * v418_data));
            double v423_data = s0[56];
            double v425_data = ir2[3];
            ir2[3] = (v425_data + (v407_data * v423_data));
            double v428_data = s0[72];
            double v430_data = ir2[4];
            ir2[4] = (v430_data + (v407_data * v428_data));
            double v433_data = s0[88];
            double v435_data = ir2[5];
            ir2[5] = (v435_data + (v407_data * v433_data));
            double v438_data = s0[104];
            double v440_data = ir2[6];
            ir2[6] = (v440_data + (v407_data * v438_data));
            double v443_data = s0[120];
            double v445_data = ir2[7];
            ir2[7] = (v445_data + (v407_data * v443_data));
          }
          if (v6_lead < 12) {
            double v451_data = r0[9];
            double v452_data = s0[9];
            double v454_data = ir2[0];
            ir2[0] = (v454_data + (v451_data * v452_data));
            double v457_data = s0[25];
            double v459_data = ir2[1];
            ir2[1] = (v459_data + (v451_data * v457_data));
            double v462_data = s0[41];
            double v464_data = ir2[2];
            ir2[2] = (v464_data + (v451_data * v462_data));
            double v467_data = s0[57];
            double v469_data = ir2[3];
            ir2[3] = (v469_data + (v451_data * v467_data));
            double v472_data = s0[73];
            double v474_data = ir2[4];
            ir2[4] = (v474_data + (v451_data * v472_data));
            double v477_data = s0[89];
            double v479_data = ir2[5];
            ir2[5] = (v479_data + (v451_data * v477_data));
            double v482_data = s0[105];
            double v484_data = ir2[6];
            ir2[6] = (v484_data + (v451_data * v482_data));
            double v487_data = s0[121];
            double v489_data = ir2[7];
            ir2[7] = (v489_data + (v451_data * v487_data));
          }
          if (v6_lead < 12) {
            double v495_data = r0[10];
            double v496_data = s0[10];
            double v498_data = ir2[0];
            ir2[0] = (v498_data + (v495_data * v496_data));
            double v501_data = s0[26];
            double v503_data = ir2[1];
            ir2[1] = (v503_data + (v495_data * v501_data));
            double v506_data = s0[42];
            double v508_data = ir2[2];
            ir2[2] = (v508_data + (v495_data * v506_data));
            double v511_data = s0[58];
            double v513_data = ir2[3];
            ir2[3] = (v513_data + (v495_data * v511_data));
            double v516_data = s0[74];
            double v518_data = ir2[4];
            ir2[4] = (v518_data + (v495_data * v516_data));
            double v521_data = s0[90];
            double v523_data = ir2[5];
            ir2[5] = (v523_data + (v495_data * v521_data));
            double v526_data = s0[106];
            double v528_data = ir2[6];
            ir2[6] = (v528_data + (v495_data * v526_data));
            double v531_data = s0[122];
            double v533_data = ir2[7];
            ir2[7] = (v533_data + (v495_data * v531_data));
          }
          if (v6_lead < 12) {
            double v539_data = r0[11];
            double v540_data = s0[11];
            double v542_data = ir2[0];
            ir2[0] = (v542_data + (v539_data * v540_data));
            double v545_data = s0[27];
            double v547_data = ir2[1];
            ir2[1] = (v547_data + (v539_data * v545_data));
            double v550_data = s0[43];
            double v552_data = ir2[2];
            ir2[2] = (v552_data + (v539_data * v550_data));
            double v555_data = s0[59];
            double v557_data = ir2[3];
            ir2[3] = (v557_data + (v539_data * v555_data));
            double v560_data = s0[75];
            double v562_data = ir2[4];
            ir2[4] = (v562_data + (v539_data * v560_data));
            double v565_data = s0[91];
            double v567_data = ir2[5];
            ir2[5] = (v567_data + (v539_data * v565_data));
            double v570_data = s0[107];
            double v572_data = ir2[6];
            ir2[6] = (v572_data + (v539_data * v570_data));
            double v575_data = s0[123];
            double v577_data = ir2[7];
            ir2[7] = (v577_data + (v539_data * v575_data));
          }
          if (v6_lead < 12) {
            double v583_data = r0[12];
            double v584_data = s0[12];
            double v586_data = ir2[0];
            ir2[0] = (v586_data + (v583_data * v584_data));
            double v589_data = s0[28];
            double v591_data = ir2[1];
            ir2[1] = (v591_data + (v583_data * v589_data));
            double v594_data = s0[44];
            double v596_data = ir2[2];
            ir2[2] = (v596_data + (v583_data * v594_data));
            double v599_data = s0[60];
            double v601_data = ir2[3];
            ir2[3] = (v601_data + (v583_data * v599_data));
            double v604_data = s0[76];
            double v606_data = ir2[4];
            ir2[4] = (v606_data + (v583_data * v604_data));
            double v609_data = s0[92];
            double v611_data = ir2[5];
            ir2[5] = (v611_data + (v583_data * v609_data));
            double v614_data = s0[108];
            double v616_data = ir2[6];
            ir2[6] = (v616_data + (v583_data * v614_data));
            double v619_data = s0[124];
            double v621_data = ir2[7];
            ir2[7] = (v621_data + (v583_data * v619_data));
          }
          if (v6_lead < 12) {
            double v627_data = r0[13];
            double v628_data = s0[13];
            double v630_data = ir2[0];
            ir2[0] = (v630_data + (v627_data * v628_data));
            double v633_data = s0[29];
            double v635_data = ir2[1];
            ir2[1] = (v635_data + (v627_data * v633_data));
            double v638_data = s0[45];
            double v640_data = ir2[2];
            ir2[2] = (v640_data + (v627_data * v638_data));
            double v643_data = s0[61];
            double v645_data = ir2[3];
            ir2[3] = (v645_data + (v627_data * v643_data));
            double v648_data = s0[77];
            double v650_data = ir2[4];
            ir2[4] = (v650_data + (v627_data * v648_data));
            double v653_data = s0[93];
            double v655_data = ir2[5];
            ir2[5] = (v655_data + (v627_data * v653_data));
            double v658_data = s0[109];
            double v660_data = ir2[6];
            ir2[6] = (v660_data + (v627_data * v658_data));
            double v663_data = s0[125];
            double v665_data = ir2[7];
            ir2[7] = (v665_data + (v627_data * v663_data));
          }
          if (v6_lead < 12) {
            double v671_data = r0[14];
            double v672_data = s0[14];
            double v674_data = ir2[0];
            ir2[0] = (v674_data + (v671_data * v672_data));
            double v677_data = s0[30];
            double v679_data = ir2[1];
            ir2[1] = (v679_data + (v671_data * v677_data));
            double v682_data = s0[46];
            double v684_data = ir2[2];
            ir2[2] = (v684_data + (v671_data * v682_data));
            double v687_data = s0[62];
            double v689_data = ir2[3];
            ir2[3] = (v689_data + (v671_data * v687_data));
            double v692_data = s0[78];
            double v694_data = ir2[4];
            ir2[4] = (v694_data + (v671_data * v692_data));
            double v697_data = s0[94];
            double v699_data = ir2[5];
            ir2[5] = (v699_data + (v671_data * v697_data));
            double v702_data = s0[110];
            double v704_data = ir2[6];
            ir2[6] = (v704_data + (v671_data * v702_data));
            double v707_data = s0[126];
            double v709_data = ir2[7];
            ir2[7] = (v709_data + (v671_data * v707_data));
          }
          if (v6_lead < 12) {
            double v715_data = r0[15];
            double v716_data = s0[15];
            double v718_data = ir2[0];
            ir2[0] = (v718_data + (v715_data * v716_data));
            double v721_data = s0[31];
            double v723_data = ir2[1];
            ir2[1] = (v723_data + (v715_data * v721_data));
            double v726_data = s0[47];
            double v728_data = ir2[2];
            ir2[2] = (v728_data + (v715_data * v726_data));
            double v731_data = s0[63];
            double v733_data = ir2[3];
            ir2[3] = (v733_data + (v715_data * v731_data));
            double v736_data = s0[79];
            double v738_data = ir2[4];
            ir2[4] = (v738_data + (v715_data * v736_data));
            double v741_data = s0[95];
            double v743_data = ir2[5];
            ir2[5] = (v743_data + (v715_data * v741_data));
            double v746_data = s0[111];
            double v748_data = ir2[6];
            ir2[6] = (v748_data + (v715_data * v746_data));
            double v751_data = s0[127];
            double v753_data = ir2[7];
            ir2[7] = (v753_data + (v715_data * v751_data));
          }
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v759_n1 = 0; v759_n1 < 8; ++v759_n1) {
              int32_t v760_a = 0 + v759_n1;
              double v762_data = ir2[v759_n1];
              int32_t v763_a = 0 + v759_n1;
              double v765_data = r1[v759_n1];
              r2[v759_n1] = (v765_data + v762_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v772_i1 = 0; v772_i1 < 8; ++v772_i1) {
              int32_t v773_a = 0 + v772_i1;
              double v775_data = r2[v772_i1];
              glb_m0[(v6_lead + (v772_i1 * 12))] = v775_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

