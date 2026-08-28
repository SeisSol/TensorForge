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
          int32_t v6_lead = threadIdx.x % 16;
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v8_i1 = 0; v8_i1 < 16; ++v8_i1) {
              int32_t v14_a = v8_i1 * 12;
              int32_t v15_a = v6_lead + v14_a;
              float v23_data = __ldcg(&glb_m1[(v6_lead + v14_a)]);
              int32_t v24_a = 0 + v8_i1;
              r0[v24_a] = v23_data;
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
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
              int32_t v38_a = v32_i1 * 12;
              int32_t v39_a = v6_lead + v38_a;
              float v47_data = glb_m0[(v6_lead + v38_a)];
              int32_t v48_a = 0 + v32_i1;
              r1[v48_a] = v47_data;
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
          if (v6_lead < 12) {
            float v55_data = r0[0];
            float v56_data = s0[0];
            float v58_data = ir2[0];
            ir2[0] = (v58_data + (v55_data * v56_data));
            float v61_data = s0[16];
            float v63_data = ir2[1];
            ir2[1] = (v63_data + (v55_data * v61_data));
            float v66_data = s0[32];
            float v68_data = ir2[2];
            ir2[2] = (v68_data + (v55_data * v66_data));
            float v71_data = s0[48];
            float v73_data = ir2[3];
            ir2[3] = (v73_data + (v55_data * v71_data));
            float v76_data = s0[64];
            float v78_data = ir2[4];
            ir2[4] = (v78_data + (v55_data * v76_data));
            float v81_data = s0[80];
            float v83_data = ir2[5];
            ir2[5] = (v83_data + (v55_data * v81_data));
            float v86_data = s0[96];
            float v88_data = ir2[6];
            ir2[6] = (v88_data + (v55_data * v86_data));
            float v91_data = s0[112];
            float v93_data = ir2[7];
            ir2[7] = (v93_data + (v55_data * v91_data));
          }
          if (v6_lead < 12) {
            float v99_data = r0[1];
            float v100_data = s0[1];
            float v102_data = ir2[0];
            ir2[0] = (v102_data + (v99_data * v100_data));
            float v105_data = s0[17];
            float v107_data = ir2[1];
            ir2[1] = (v107_data + (v99_data * v105_data));
            float v110_data = s0[33];
            float v112_data = ir2[2];
            ir2[2] = (v112_data + (v99_data * v110_data));
            float v115_data = s0[49];
            float v117_data = ir2[3];
            ir2[3] = (v117_data + (v99_data * v115_data));
            float v120_data = s0[65];
            float v122_data = ir2[4];
            ir2[4] = (v122_data + (v99_data * v120_data));
            float v125_data = s0[81];
            float v127_data = ir2[5];
            ir2[5] = (v127_data + (v99_data * v125_data));
            float v130_data = s0[97];
            float v132_data = ir2[6];
            ir2[6] = (v132_data + (v99_data * v130_data));
            float v135_data = s0[113];
            float v137_data = ir2[7];
            ir2[7] = (v137_data + (v99_data * v135_data));
          }
          if (v6_lead < 12) {
            float v143_data = r0[2];
            float v144_data = s0[2];
            float v146_data = ir2[0];
            ir2[0] = (v146_data + (v143_data * v144_data));
            float v149_data = s0[18];
            float v151_data = ir2[1];
            ir2[1] = (v151_data + (v143_data * v149_data));
            float v154_data = s0[34];
            float v156_data = ir2[2];
            ir2[2] = (v156_data + (v143_data * v154_data));
            float v159_data = s0[50];
            float v161_data = ir2[3];
            ir2[3] = (v161_data + (v143_data * v159_data));
            float v164_data = s0[66];
            float v166_data = ir2[4];
            ir2[4] = (v166_data + (v143_data * v164_data));
            float v169_data = s0[82];
            float v171_data = ir2[5];
            ir2[5] = (v171_data + (v143_data * v169_data));
            float v174_data = s0[98];
            float v176_data = ir2[6];
            ir2[6] = (v176_data + (v143_data * v174_data));
            float v179_data = s0[114];
            float v181_data = ir2[7];
            ir2[7] = (v181_data + (v143_data * v179_data));
          }
          if (v6_lead < 12) {
            float v187_data = r0[3];
            float v188_data = s0[3];
            float v190_data = ir2[0];
            ir2[0] = (v190_data + (v187_data * v188_data));
            float v193_data = s0[19];
            float v195_data = ir2[1];
            ir2[1] = (v195_data + (v187_data * v193_data));
            float v198_data = s0[35];
            float v200_data = ir2[2];
            ir2[2] = (v200_data + (v187_data * v198_data));
            float v203_data = s0[51];
            float v205_data = ir2[3];
            ir2[3] = (v205_data + (v187_data * v203_data));
            float v208_data = s0[67];
            float v210_data = ir2[4];
            ir2[4] = (v210_data + (v187_data * v208_data));
            float v213_data = s0[83];
            float v215_data = ir2[5];
            ir2[5] = (v215_data + (v187_data * v213_data));
            float v218_data = s0[99];
            float v220_data = ir2[6];
            ir2[6] = (v220_data + (v187_data * v218_data));
            float v223_data = s0[115];
            float v225_data = ir2[7];
            ir2[7] = (v225_data + (v187_data * v223_data));
          }
          if (v6_lead < 12) {
            float v231_data = r0[4];
            float v232_data = s0[4];
            float v234_data = ir2[0];
            ir2[0] = (v234_data + (v231_data * v232_data));
            float v237_data = s0[20];
            float v239_data = ir2[1];
            ir2[1] = (v239_data + (v231_data * v237_data));
            float v242_data = s0[36];
            float v244_data = ir2[2];
            ir2[2] = (v244_data + (v231_data * v242_data));
            float v247_data = s0[52];
            float v249_data = ir2[3];
            ir2[3] = (v249_data + (v231_data * v247_data));
            float v252_data = s0[68];
            float v254_data = ir2[4];
            ir2[4] = (v254_data + (v231_data * v252_data));
            float v257_data = s0[84];
            float v259_data = ir2[5];
            ir2[5] = (v259_data + (v231_data * v257_data));
            float v262_data = s0[100];
            float v264_data = ir2[6];
            ir2[6] = (v264_data + (v231_data * v262_data));
            float v267_data = s0[116];
            float v269_data = ir2[7];
            ir2[7] = (v269_data + (v231_data * v267_data));
          }
          if (v6_lead < 12) {
            float v275_data = r0[5];
            float v276_data = s0[5];
            float v278_data = ir2[0];
            ir2[0] = (v278_data + (v275_data * v276_data));
            float v281_data = s0[21];
            float v283_data = ir2[1];
            ir2[1] = (v283_data + (v275_data * v281_data));
            float v286_data = s0[37];
            float v288_data = ir2[2];
            ir2[2] = (v288_data + (v275_data * v286_data));
            float v291_data = s0[53];
            float v293_data = ir2[3];
            ir2[3] = (v293_data + (v275_data * v291_data));
            float v296_data = s0[69];
            float v298_data = ir2[4];
            ir2[4] = (v298_data + (v275_data * v296_data));
            float v301_data = s0[85];
            float v303_data = ir2[5];
            ir2[5] = (v303_data + (v275_data * v301_data));
            float v306_data = s0[101];
            float v308_data = ir2[6];
            ir2[6] = (v308_data + (v275_data * v306_data));
            float v311_data = s0[117];
            float v313_data = ir2[7];
            ir2[7] = (v313_data + (v275_data * v311_data));
          }
          if (v6_lead < 12) {
            float v319_data = r0[6];
            float v320_data = s0[6];
            float v322_data = ir2[0];
            ir2[0] = (v322_data + (v319_data * v320_data));
            float v325_data = s0[22];
            float v327_data = ir2[1];
            ir2[1] = (v327_data + (v319_data * v325_data));
            float v330_data = s0[38];
            float v332_data = ir2[2];
            ir2[2] = (v332_data + (v319_data * v330_data));
            float v335_data = s0[54];
            float v337_data = ir2[3];
            ir2[3] = (v337_data + (v319_data * v335_data));
            float v340_data = s0[70];
            float v342_data = ir2[4];
            ir2[4] = (v342_data + (v319_data * v340_data));
            float v345_data = s0[86];
            float v347_data = ir2[5];
            ir2[5] = (v347_data + (v319_data * v345_data));
            float v350_data = s0[102];
            float v352_data = ir2[6];
            ir2[6] = (v352_data + (v319_data * v350_data));
            float v355_data = s0[118];
            float v357_data = ir2[7];
            ir2[7] = (v357_data + (v319_data * v355_data));
          }
          if (v6_lead < 12) {
            float v363_data = r0[7];
            float v364_data = s0[7];
            float v366_data = ir2[0];
            ir2[0] = (v366_data + (v363_data * v364_data));
            float v369_data = s0[23];
            float v371_data = ir2[1];
            ir2[1] = (v371_data + (v363_data * v369_data));
            float v374_data = s0[39];
            float v376_data = ir2[2];
            ir2[2] = (v376_data + (v363_data * v374_data));
            float v379_data = s0[55];
            float v381_data = ir2[3];
            ir2[3] = (v381_data + (v363_data * v379_data));
            float v384_data = s0[71];
            float v386_data = ir2[4];
            ir2[4] = (v386_data + (v363_data * v384_data));
            float v389_data = s0[87];
            float v391_data = ir2[5];
            ir2[5] = (v391_data + (v363_data * v389_data));
            float v394_data = s0[103];
            float v396_data = ir2[6];
            ir2[6] = (v396_data + (v363_data * v394_data));
            float v399_data = s0[119];
            float v401_data = ir2[7];
            ir2[7] = (v401_data + (v363_data * v399_data));
          }
          if (v6_lead < 12) {
            float v407_data = r0[8];
            float v408_data = s0[8];
            float v410_data = ir2[0];
            ir2[0] = (v410_data + (v407_data * v408_data));
            float v413_data = s0[24];
            float v415_data = ir2[1];
            ir2[1] = (v415_data + (v407_data * v413_data));
            float v418_data = s0[40];
            float v420_data = ir2[2];
            ir2[2] = (v420_data + (v407_data * v418_data));
            float v423_data = s0[56];
            float v425_data = ir2[3];
            ir2[3] = (v425_data + (v407_data * v423_data));
            float v428_data = s0[72];
            float v430_data = ir2[4];
            ir2[4] = (v430_data + (v407_data * v428_data));
            float v433_data = s0[88];
            float v435_data = ir2[5];
            ir2[5] = (v435_data + (v407_data * v433_data));
            float v438_data = s0[104];
            float v440_data = ir2[6];
            ir2[6] = (v440_data + (v407_data * v438_data));
            float v443_data = s0[120];
            float v445_data = ir2[7];
            ir2[7] = (v445_data + (v407_data * v443_data));
          }
          if (v6_lead < 12) {
            float v451_data = r0[9];
            float v452_data = s0[9];
            float v454_data = ir2[0];
            ir2[0] = (v454_data + (v451_data * v452_data));
            float v457_data = s0[25];
            float v459_data = ir2[1];
            ir2[1] = (v459_data + (v451_data * v457_data));
            float v462_data = s0[41];
            float v464_data = ir2[2];
            ir2[2] = (v464_data + (v451_data * v462_data));
            float v467_data = s0[57];
            float v469_data = ir2[3];
            ir2[3] = (v469_data + (v451_data * v467_data));
            float v472_data = s0[73];
            float v474_data = ir2[4];
            ir2[4] = (v474_data + (v451_data * v472_data));
            float v477_data = s0[89];
            float v479_data = ir2[5];
            ir2[5] = (v479_data + (v451_data * v477_data));
            float v482_data = s0[105];
            float v484_data = ir2[6];
            ir2[6] = (v484_data + (v451_data * v482_data));
            float v487_data = s0[121];
            float v489_data = ir2[7];
            ir2[7] = (v489_data + (v451_data * v487_data));
          }
          if (v6_lead < 12) {
            float v495_data = r0[10];
            float v496_data = s0[10];
            float v498_data = ir2[0];
            ir2[0] = (v498_data + (v495_data * v496_data));
            float v501_data = s0[26];
            float v503_data = ir2[1];
            ir2[1] = (v503_data + (v495_data * v501_data));
            float v506_data = s0[42];
            float v508_data = ir2[2];
            ir2[2] = (v508_data + (v495_data * v506_data));
            float v511_data = s0[58];
            float v513_data = ir2[3];
            ir2[3] = (v513_data + (v495_data * v511_data));
            float v516_data = s0[74];
            float v518_data = ir2[4];
            ir2[4] = (v518_data + (v495_data * v516_data));
            float v521_data = s0[90];
            float v523_data = ir2[5];
            ir2[5] = (v523_data + (v495_data * v521_data));
            float v526_data = s0[106];
            float v528_data = ir2[6];
            ir2[6] = (v528_data + (v495_data * v526_data));
            float v531_data = s0[122];
            float v533_data = ir2[7];
            ir2[7] = (v533_data + (v495_data * v531_data));
          }
          if (v6_lead < 12) {
            float v539_data = r0[11];
            float v540_data = s0[11];
            float v542_data = ir2[0];
            ir2[0] = (v542_data + (v539_data * v540_data));
            float v545_data = s0[27];
            float v547_data = ir2[1];
            ir2[1] = (v547_data + (v539_data * v545_data));
            float v550_data = s0[43];
            float v552_data = ir2[2];
            ir2[2] = (v552_data + (v539_data * v550_data));
            float v555_data = s0[59];
            float v557_data = ir2[3];
            ir2[3] = (v557_data + (v539_data * v555_data));
            float v560_data = s0[75];
            float v562_data = ir2[4];
            ir2[4] = (v562_data + (v539_data * v560_data));
            float v565_data = s0[91];
            float v567_data = ir2[5];
            ir2[5] = (v567_data + (v539_data * v565_data));
            float v570_data = s0[107];
            float v572_data = ir2[6];
            ir2[6] = (v572_data + (v539_data * v570_data));
            float v575_data = s0[123];
            float v577_data = ir2[7];
            ir2[7] = (v577_data + (v539_data * v575_data));
          }
          if (v6_lead < 12) {
            float v583_data = r0[12];
            float v584_data = s0[12];
            float v586_data = ir2[0];
            ir2[0] = (v586_data + (v583_data * v584_data));
            float v589_data = s0[28];
            float v591_data = ir2[1];
            ir2[1] = (v591_data + (v583_data * v589_data));
            float v594_data = s0[44];
            float v596_data = ir2[2];
            ir2[2] = (v596_data + (v583_data * v594_data));
            float v599_data = s0[60];
            float v601_data = ir2[3];
            ir2[3] = (v601_data + (v583_data * v599_data));
            float v604_data = s0[76];
            float v606_data = ir2[4];
            ir2[4] = (v606_data + (v583_data * v604_data));
            float v609_data = s0[92];
            float v611_data = ir2[5];
            ir2[5] = (v611_data + (v583_data * v609_data));
            float v614_data = s0[108];
            float v616_data = ir2[6];
            ir2[6] = (v616_data + (v583_data * v614_data));
            float v619_data = s0[124];
            float v621_data = ir2[7];
            ir2[7] = (v621_data + (v583_data * v619_data));
          }
          if (v6_lead < 12) {
            float v627_data = r0[13];
            float v628_data = s0[13];
            float v630_data = ir2[0];
            ir2[0] = (v630_data + (v627_data * v628_data));
            float v633_data = s0[29];
            float v635_data = ir2[1];
            ir2[1] = (v635_data + (v627_data * v633_data));
            float v638_data = s0[45];
            float v640_data = ir2[2];
            ir2[2] = (v640_data + (v627_data * v638_data));
            float v643_data = s0[61];
            float v645_data = ir2[3];
            ir2[3] = (v645_data + (v627_data * v643_data));
            float v648_data = s0[77];
            float v650_data = ir2[4];
            ir2[4] = (v650_data + (v627_data * v648_data));
            float v653_data = s0[93];
            float v655_data = ir2[5];
            ir2[5] = (v655_data + (v627_data * v653_data));
            float v658_data = s0[109];
            float v660_data = ir2[6];
            ir2[6] = (v660_data + (v627_data * v658_data));
            float v663_data = s0[125];
            float v665_data = ir2[7];
            ir2[7] = (v665_data + (v627_data * v663_data));
          }
          if (v6_lead < 12) {
            float v671_data = r0[14];
            float v672_data = s0[14];
            float v674_data = ir2[0];
            ir2[0] = (v674_data + (v671_data * v672_data));
            float v677_data = s0[30];
            float v679_data = ir2[1];
            ir2[1] = (v679_data + (v671_data * v677_data));
            float v682_data = s0[46];
            float v684_data = ir2[2];
            ir2[2] = (v684_data + (v671_data * v682_data));
            float v687_data = s0[62];
            float v689_data = ir2[3];
            ir2[3] = (v689_data + (v671_data * v687_data));
            float v692_data = s0[78];
            float v694_data = ir2[4];
            ir2[4] = (v694_data + (v671_data * v692_data));
            float v697_data = s0[94];
            float v699_data = ir2[5];
            ir2[5] = (v699_data + (v671_data * v697_data));
            float v702_data = s0[110];
            float v704_data = ir2[6];
            ir2[6] = (v704_data + (v671_data * v702_data));
            float v707_data = s0[126];
            float v709_data = ir2[7];
            ir2[7] = (v709_data + (v671_data * v707_data));
          }
          if (v6_lead < 12) {
            float v715_data = r0[15];
            float v716_data = s0[15];
            float v718_data = ir2[0];
            ir2[0] = (v718_data + (v715_data * v716_data));
            float v721_data = s0[31];
            float v723_data = ir2[1];
            ir2[1] = (v723_data + (v715_data * v721_data));
            float v726_data = s0[47];
            float v728_data = ir2[2];
            ir2[2] = (v728_data + (v715_data * v726_data));
            float v731_data = s0[63];
            float v733_data = ir2[3];
            ir2[3] = (v733_data + (v715_data * v731_data));
            float v736_data = s0[79];
            float v738_data = ir2[4];
            ir2[4] = (v738_data + (v715_data * v736_data));
            float v741_data = s0[95];
            float v743_data = ir2[5];
            ir2[5] = (v743_data + (v715_data * v741_data));
            float v746_data = s0[111];
            float v748_data = ir2[6];
            ir2[6] = (v748_data + (v715_data * v746_data));
            float v751_data = s0[127];
            float v753_data = ir2[7];
            ir2[7] = (v753_data + (v715_data * v751_data));
          }
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v759_n1 = 0; v759_n1 < 8; ++v759_n1) {
              int32_t v760_a = 0 + v759_n1;
              float v762_data = ir2[v759_n1];
              int32_t v763_a = 0 + v759_n1;
              float v765_data = r1[v759_n1];
              r2[v759_n1] = (v765_data + v762_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v6_lead < 12) {
            #pragma unroll
            for (int32_t v772_i1 = 0; v772_i1 < 8; ++v772_i1) {
              int32_t v773_a = 0 + v772_i1;
              float v775_data = r2[v772_i1];
              glb_m0[(v6_lead + (v772_i1 * 12))] = v775_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

