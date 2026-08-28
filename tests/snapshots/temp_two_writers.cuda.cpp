// === base name ===
kernel_3e24e7feaf

// === header ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3e24e7feaf, block.x * block.y * block.z, 2816 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_3e24e7feaf, cudaFuncAttributeMaxDynamicSharedMemorySize, 2816 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3e24e7feaf<<<grid,block,2816 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×32(6×12) {0..6}×{0..12} strided
    // m1 32×32(12×12) {0..12}×{0..12} strided
    // m2 32×32(6×12) {0..6}×{0..12} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // m4 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
    // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[160];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v9_lead = threadIdx.x % 16;
          if (v9_lead < 6) {
            #pragma unroll
            for (int32_t v11_i1 = 0; v11_i1 < 12; ++v11_i1) {
              int32_t v17_a = v11_i1 * 6;
              int32_t v18_a = v9_lead + v17_a;
              float v26_data = __ldcg(&glb_m0[(v9_lead + v17_a)]);
              r0[v11_i1] = v26_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m1[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 9; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v9_lead < 6) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
              int32_t v41_a = v35_i1 * 6;
              int32_t v42_a = v9_lead + v41_a;
              float v50_data = __ldcg(&glb_m2[(v9_lead + v41_a)]);
              r2[v35_i1] = v50_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          if (v9_lead < 6) {
            float v57_data = r0[0];
            float v58_data = s0[0];
            float v60_data = r1[0];
            r1[0] = (v60_data + (v57_data * v58_data));
            float v63_data = s0[12];
            float v65_data = r1[1];
            r1[1] = (v65_data + (v57_data * v63_data));
            float v68_data = s0[24];
            float v70_data = r1[2];
            r1[2] = (v70_data + (v57_data * v68_data));
            float v73_data = s0[36];
            float v75_data = r1[3];
            r1[3] = (v75_data + (v57_data * v73_data));
            float v78_data = s0[48];
            float v80_data = r1[4];
            r1[4] = (v80_data + (v57_data * v78_data));
            float v83_data = s0[60];
            float v85_data = r1[5];
            r1[5] = (v85_data + (v57_data * v83_data));
            float v88_data = s0[72];
            float v90_data = r1[6];
            r1[6] = (v90_data + (v57_data * v88_data));
            float v93_data = s0[84];
            float v95_data = r1[7];
            r1[7] = (v95_data + (v57_data * v93_data));
            float v98_data = s0[96];
            float v100_data = r1[8];
            r1[8] = (v100_data + (v57_data * v98_data));
            float v103_data = s0[108];
            float v105_data = r1[9];
            r1[9] = (v105_data + (v57_data * v103_data));
            float v108_data = s0[120];
            float v110_data = r1[10];
            r1[10] = (v110_data + (v57_data * v108_data));
            float v113_data = s0[132];
            float v115_data = r1[11];
            r1[11] = (v115_data + (v57_data * v113_data));
          }
          if (v9_lead < 6) {
            float v121_data = r0[1];
            float v122_data = s0[1];
            float v124_data = r1[0];
            r1[0] = (v124_data + (v121_data * v122_data));
            float v127_data = s0[13];
            float v129_data = r1[1];
            r1[1] = (v129_data + (v121_data * v127_data));
            float v132_data = s0[25];
            float v134_data = r1[2];
            r1[2] = (v134_data + (v121_data * v132_data));
            float v137_data = s0[37];
            float v139_data = r1[3];
            r1[3] = (v139_data + (v121_data * v137_data));
            float v142_data = s0[49];
            float v144_data = r1[4];
            r1[4] = (v144_data + (v121_data * v142_data));
            float v147_data = s0[61];
            float v149_data = r1[5];
            r1[5] = (v149_data + (v121_data * v147_data));
            float v152_data = s0[73];
            float v154_data = r1[6];
            r1[6] = (v154_data + (v121_data * v152_data));
            float v157_data = s0[85];
            float v159_data = r1[7];
            r1[7] = (v159_data + (v121_data * v157_data));
            float v162_data = s0[97];
            float v164_data = r1[8];
            r1[8] = (v164_data + (v121_data * v162_data));
            float v167_data = s0[109];
            float v169_data = r1[9];
            r1[9] = (v169_data + (v121_data * v167_data));
            float v172_data = s0[121];
            float v174_data = r1[10];
            r1[10] = (v174_data + (v121_data * v172_data));
            float v177_data = s0[133];
            float v179_data = r1[11];
            r1[11] = (v179_data + (v121_data * v177_data));
          }
          if (v9_lead < 6) {
            float v185_data = r0[2];
            float v186_data = s0[2];
            float v188_data = r1[0];
            r1[0] = (v188_data + (v185_data * v186_data));
            float v191_data = s0[14];
            float v193_data = r1[1];
            r1[1] = (v193_data + (v185_data * v191_data));
            float v196_data = s0[26];
            float v198_data = r1[2];
            r1[2] = (v198_data + (v185_data * v196_data));
            float v201_data = s0[38];
            float v203_data = r1[3];
            r1[3] = (v203_data + (v185_data * v201_data));
            float v206_data = s0[50];
            float v208_data = r1[4];
            r1[4] = (v208_data + (v185_data * v206_data));
            float v211_data = s0[62];
            float v213_data = r1[5];
            r1[5] = (v213_data + (v185_data * v211_data));
            float v216_data = s0[74];
            float v218_data = r1[6];
            r1[6] = (v218_data + (v185_data * v216_data));
            float v221_data = s0[86];
            float v223_data = r1[7];
            r1[7] = (v223_data + (v185_data * v221_data));
            float v226_data = s0[98];
            float v228_data = r1[8];
            r1[8] = (v228_data + (v185_data * v226_data));
            float v231_data = s0[110];
            float v233_data = r1[9];
            r1[9] = (v233_data + (v185_data * v231_data));
            float v236_data = s0[122];
            float v238_data = r1[10];
            r1[10] = (v238_data + (v185_data * v236_data));
            float v241_data = s0[134];
            float v243_data = r1[11];
            r1[11] = (v243_data + (v185_data * v241_data));
          }
          if (v9_lead < 6) {
            float v249_data = r0[3];
            float v250_data = s0[3];
            float v252_data = r1[0];
            r1[0] = (v252_data + (v249_data * v250_data));
            float v255_data = s0[15];
            float v257_data = r1[1];
            r1[1] = (v257_data + (v249_data * v255_data));
            float v260_data = s0[27];
            float v262_data = r1[2];
            r1[2] = (v262_data + (v249_data * v260_data));
            float v265_data = s0[39];
            float v267_data = r1[3];
            r1[3] = (v267_data + (v249_data * v265_data));
            float v270_data = s0[51];
            float v272_data = r1[4];
            r1[4] = (v272_data + (v249_data * v270_data));
            float v275_data = s0[63];
            float v277_data = r1[5];
            r1[5] = (v277_data + (v249_data * v275_data));
            float v280_data = s0[75];
            float v282_data = r1[6];
            r1[6] = (v282_data + (v249_data * v280_data));
            float v285_data = s0[87];
            float v287_data = r1[7];
            r1[7] = (v287_data + (v249_data * v285_data));
            float v290_data = s0[99];
            float v292_data = r1[8];
            r1[8] = (v292_data + (v249_data * v290_data));
            float v295_data = s0[111];
            float v297_data = r1[9];
            r1[9] = (v297_data + (v249_data * v295_data));
            float v300_data = s0[123];
            float v302_data = r1[10];
            r1[10] = (v302_data + (v249_data * v300_data));
            float v305_data = s0[135];
            float v307_data = r1[11];
            r1[11] = (v307_data + (v249_data * v305_data));
          }
          if (v9_lead < 6) {
            float v313_data = r0[4];
            float v314_data = s0[4];
            float v316_data = r1[0];
            r1[0] = (v316_data + (v313_data * v314_data));
            float v319_data = s0[16];
            float v321_data = r1[1];
            r1[1] = (v321_data + (v313_data * v319_data));
            float v324_data = s0[28];
            float v326_data = r1[2];
            r1[2] = (v326_data + (v313_data * v324_data));
            float v329_data = s0[40];
            float v331_data = r1[3];
            r1[3] = (v331_data + (v313_data * v329_data));
            float v334_data = s0[52];
            float v336_data = r1[4];
            r1[4] = (v336_data + (v313_data * v334_data));
            float v339_data = s0[64];
            float v341_data = r1[5];
            r1[5] = (v341_data + (v313_data * v339_data));
            float v344_data = s0[76];
            float v346_data = r1[6];
            r1[6] = (v346_data + (v313_data * v344_data));
            float v349_data = s0[88];
            float v351_data = r1[7];
            r1[7] = (v351_data + (v313_data * v349_data));
            float v354_data = s0[100];
            float v356_data = r1[8];
            r1[8] = (v356_data + (v313_data * v354_data));
            float v359_data = s0[112];
            float v361_data = r1[9];
            r1[9] = (v361_data + (v313_data * v359_data));
            float v364_data = s0[124];
            float v366_data = r1[10];
            r1[10] = (v366_data + (v313_data * v364_data));
            float v369_data = s0[136];
            float v371_data = r1[11];
            r1[11] = (v371_data + (v313_data * v369_data));
          }
          if (v9_lead < 6) {
            float v377_data = r0[5];
            float v378_data = s0[5];
            float v380_data = r1[0];
            r1[0] = (v380_data + (v377_data * v378_data));
            float v383_data = s0[17];
            float v385_data = r1[1];
            r1[1] = (v385_data + (v377_data * v383_data));
            float v388_data = s0[29];
            float v390_data = r1[2];
            r1[2] = (v390_data + (v377_data * v388_data));
            float v393_data = s0[41];
            float v395_data = r1[3];
            r1[3] = (v395_data + (v377_data * v393_data));
            float v398_data = s0[53];
            float v400_data = r1[4];
            r1[4] = (v400_data + (v377_data * v398_data));
            float v403_data = s0[65];
            float v405_data = r1[5];
            r1[5] = (v405_data + (v377_data * v403_data));
            float v408_data = s0[77];
            float v410_data = r1[6];
            r1[6] = (v410_data + (v377_data * v408_data));
            float v413_data = s0[89];
            float v415_data = r1[7];
            r1[7] = (v415_data + (v377_data * v413_data));
            float v418_data = s0[101];
            float v420_data = r1[8];
            r1[8] = (v420_data + (v377_data * v418_data));
            float v423_data = s0[113];
            float v425_data = r1[9];
            r1[9] = (v425_data + (v377_data * v423_data));
            float v428_data = s0[125];
            float v430_data = r1[10];
            r1[10] = (v430_data + (v377_data * v428_data));
            float v433_data = s0[137];
            float v435_data = r1[11];
            r1[11] = (v435_data + (v377_data * v433_data));
          }
          if (v9_lead < 6) {
            float v441_data = r0[6];
            float v442_data = s0[6];
            float v444_data = r1[0];
            r1[0] = (v444_data + (v441_data * v442_data));
            float v447_data = s0[18];
            float v449_data = r1[1];
            r1[1] = (v449_data + (v441_data * v447_data));
            float v452_data = s0[30];
            float v454_data = r1[2];
            r1[2] = (v454_data + (v441_data * v452_data));
            float v457_data = s0[42];
            float v459_data = r1[3];
            r1[3] = (v459_data + (v441_data * v457_data));
            float v462_data = s0[54];
            float v464_data = r1[4];
            r1[4] = (v464_data + (v441_data * v462_data));
            float v467_data = s0[66];
            float v469_data = r1[5];
            r1[5] = (v469_data + (v441_data * v467_data));
            float v472_data = s0[78];
            float v474_data = r1[6];
            r1[6] = (v474_data + (v441_data * v472_data));
            float v477_data = s0[90];
            float v479_data = r1[7];
            r1[7] = (v479_data + (v441_data * v477_data));
            float v482_data = s0[102];
            float v484_data = r1[8];
            r1[8] = (v484_data + (v441_data * v482_data));
            float v487_data = s0[114];
            float v489_data = r1[9];
            r1[9] = (v489_data + (v441_data * v487_data));
            float v492_data = s0[126];
            float v494_data = r1[10];
            r1[10] = (v494_data + (v441_data * v492_data));
            float v497_data = s0[138];
            float v499_data = r1[11];
            r1[11] = (v499_data + (v441_data * v497_data));
          }
          if (v9_lead < 6) {
            float v505_data = r0[7];
            float v506_data = s0[7];
            float v508_data = r1[0];
            r1[0] = (v508_data + (v505_data * v506_data));
            float v511_data = s0[19];
            float v513_data = r1[1];
            r1[1] = (v513_data + (v505_data * v511_data));
            float v516_data = s0[31];
            float v518_data = r1[2];
            r1[2] = (v518_data + (v505_data * v516_data));
            float v521_data = s0[43];
            float v523_data = r1[3];
            r1[3] = (v523_data + (v505_data * v521_data));
            float v526_data = s0[55];
            float v528_data = r1[4];
            r1[4] = (v528_data + (v505_data * v526_data));
            float v531_data = s0[67];
            float v533_data = r1[5];
            r1[5] = (v533_data + (v505_data * v531_data));
            float v536_data = s0[79];
            float v538_data = r1[6];
            r1[6] = (v538_data + (v505_data * v536_data));
            float v541_data = s0[91];
            float v543_data = r1[7];
            r1[7] = (v543_data + (v505_data * v541_data));
            float v546_data = s0[103];
            float v548_data = r1[8];
            r1[8] = (v548_data + (v505_data * v546_data));
            float v551_data = s0[115];
            float v553_data = r1[9];
            r1[9] = (v553_data + (v505_data * v551_data));
            float v556_data = s0[127];
            float v558_data = r1[10];
            r1[10] = (v558_data + (v505_data * v556_data));
            float v561_data = s0[139];
            float v563_data = r1[11];
            r1[11] = (v563_data + (v505_data * v561_data));
          }
          if (v9_lead < 6) {
            float v569_data = r0[8];
            float v570_data = s0[8];
            float v572_data = r1[0];
            r1[0] = (v572_data + (v569_data * v570_data));
            float v575_data = s0[20];
            float v577_data = r1[1];
            r1[1] = (v577_data + (v569_data * v575_data));
            float v580_data = s0[32];
            float v582_data = r1[2];
            r1[2] = (v582_data + (v569_data * v580_data));
            float v585_data = s0[44];
            float v587_data = r1[3];
            r1[3] = (v587_data + (v569_data * v585_data));
            float v590_data = s0[56];
            float v592_data = r1[4];
            r1[4] = (v592_data + (v569_data * v590_data));
            float v595_data = s0[68];
            float v597_data = r1[5];
            r1[5] = (v597_data + (v569_data * v595_data));
            float v600_data = s0[80];
            float v602_data = r1[6];
            r1[6] = (v602_data + (v569_data * v600_data));
            float v605_data = s0[92];
            float v607_data = r1[7];
            r1[7] = (v607_data + (v569_data * v605_data));
            float v610_data = s0[104];
            float v612_data = r1[8];
            r1[8] = (v612_data + (v569_data * v610_data));
            float v615_data = s0[116];
            float v617_data = r1[9];
            r1[9] = (v617_data + (v569_data * v615_data));
            float v620_data = s0[128];
            float v622_data = r1[10];
            r1[10] = (v622_data + (v569_data * v620_data));
            float v625_data = s0[140];
            float v627_data = r1[11];
            r1[11] = (v627_data + (v569_data * v625_data));
          }
          if (v9_lead < 6) {
            float v633_data = r0[9];
            float v634_data = s0[9];
            float v636_data = r1[0];
            r1[0] = (v636_data + (v633_data * v634_data));
            float v639_data = s0[21];
            float v641_data = r1[1];
            r1[1] = (v641_data + (v633_data * v639_data));
            float v644_data = s0[33];
            float v646_data = r1[2];
            r1[2] = (v646_data + (v633_data * v644_data));
            float v649_data = s0[45];
            float v651_data = r1[3];
            r1[3] = (v651_data + (v633_data * v649_data));
            float v654_data = s0[57];
            float v656_data = r1[4];
            r1[4] = (v656_data + (v633_data * v654_data));
            float v659_data = s0[69];
            float v661_data = r1[5];
            r1[5] = (v661_data + (v633_data * v659_data));
            float v664_data = s0[81];
            float v666_data = r1[6];
            r1[6] = (v666_data + (v633_data * v664_data));
            float v669_data = s0[93];
            float v671_data = r1[7];
            r1[7] = (v671_data + (v633_data * v669_data));
            float v674_data = s0[105];
            float v676_data = r1[8];
            r1[8] = (v676_data + (v633_data * v674_data));
            float v679_data = s0[117];
            float v681_data = r1[9];
            r1[9] = (v681_data + (v633_data * v679_data));
            float v684_data = s0[129];
            float v686_data = r1[10];
            r1[10] = (v686_data + (v633_data * v684_data));
            float v689_data = s0[141];
            float v691_data = r1[11];
            r1[11] = (v691_data + (v633_data * v689_data));
          }
          if (v9_lead < 6) {
            float v697_data = r0[10];
            float v698_data = s0[10];
            float v700_data = r1[0];
            r1[0] = (v700_data + (v697_data * v698_data));
            float v703_data = s0[22];
            float v705_data = r1[1];
            r1[1] = (v705_data + (v697_data * v703_data));
            float v708_data = s0[34];
            float v710_data = r1[2];
            r1[2] = (v710_data + (v697_data * v708_data));
            float v713_data = s0[46];
            float v715_data = r1[3];
            r1[3] = (v715_data + (v697_data * v713_data));
            float v718_data = s0[58];
            float v720_data = r1[4];
            r1[4] = (v720_data + (v697_data * v718_data));
            float v723_data = s0[70];
            float v725_data = r1[5];
            r1[5] = (v725_data + (v697_data * v723_data));
            float v728_data = s0[82];
            float v730_data = r1[6];
            r1[6] = (v730_data + (v697_data * v728_data));
            float v733_data = s0[94];
            float v735_data = r1[7];
            r1[7] = (v735_data + (v697_data * v733_data));
            float v738_data = s0[106];
            float v740_data = r1[8];
            r1[8] = (v740_data + (v697_data * v738_data));
            float v743_data = s0[118];
            float v745_data = r1[9];
            r1[9] = (v745_data + (v697_data * v743_data));
            float v748_data = s0[130];
            float v750_data = r1[10];
            r1[10] = (v750_data + (v697_data * v748_data));
            float v753_data = s0[142];
            float v755_data = r1[11];
            r1[11] = (v755_data + (v697_data * v753_data));
          }
          if (v9_lead < 6) {
            float v761_data = r0[11];
            float v762_data = s0[11];
            float v764_data = r1[0];
            r1[0] = (v764_data + (v761_data * v762_data));
            float v767_data = s0[23];
            float v769_data = r1[1];
            r1[1] = (v769_data + (v761_data * v767_data));
            float v772_data = s0[35];
            float v774_data = r1[2];
            r1[2] = (v774_data + (v761_data * v772_data));
            float v777_data = s0[47];
            float v779_data = r1[3];
            r1[3] = (v779_data + (v761_data * v777_data));
            float v782_data = s0[59];
            float v784_data = r1[4];
            r1[4] = (v784_data + (v761_data * v782_data));
            float v787_data = s0[71];
            float v789_data = r1[5];
            r1[5] = (v789_data + (v761_data * v787_data));
            float v792_data = s0[83];
            float v794_data = r1[6];
            r1[6] = (v794_data + (v761_data * v792_data));
            float v797_data = s0[95];
            float v799_data = r1[7];
            r1[7] = (v799_data + (v761_data * v797_data));
            float v802_data = s0[107];
            float v804_data = r1[8];
            r1[8] = (v804_data + (v761_data * v802_data));
            float v807_data = s0[119];
            float v809_data = r1[9];
            r1[9] = (v809_data + (v761_data * v807_data));
            float v812_data = s0[131];
            float v814_data = r1[10];
            r1[10] = (v814_data + (v761_data * v812_data));
            float v817_data = s0[143];
            float v819_data = r1[11];
            r1[11] = (v819_data + (v761_data * v817_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v9_lead < 6) {
            #pragma unroll
            for (int32_t v826_i1 = 0; v826_i1 < 12; ++v826_i1) {
              int32_t v827_a = 0 + v826_i1;
              float v829_data = r1[v826_i1];
              s1[(v9_lead + (v826_i1 * 12))] = v829_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v9_lead < 12) {
            #pragma unroll
            for (int32_t v842_i1 = 0; v842_i1 < 12; ++v842_i1) {
              int32_t v848_a = v842_i1 * 12;
              int32_t v849_a = v9_lead + v848_a;
              float v857_data = __ldcg(&glb_m4[(v9_lead + v848_a)]);
              r4[v842_i1] = v857_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v9_lead < 6) {
            float v865_data = r2[0];
            float v866_data = s0[0];
            float v868_data = ir3[0];
            ir3[0] = (v868_data + (v865_data * v866_data));
            float v871_data = s0[12];
            float v873_data = ir3[1];
            ir3[1] = (v873_data + (v865_data * v871_data));
            float v876_data = s0[24];
            float v878_data = ir3[2];
            ir3[2] = (v878_data + (v865_data * v876_data));
            float v881_data = s0[36];
            float v883_data = ir3[3];
            ir3[3] = (v883_data + (v865_data * v881_data));
            float v886_data = s0[48];
            float v888_data = ir3[4];
            ir3[4] = (v888_data + (v865_data * v886_data));
            float v891_data = s0[60];
            float v893_data = ir3[5];
            ir3[5] = (v893_data + (v865_data * v891_data));
            float v896_data = s0[72];
            float v898_data = ir3[6];
            ir3[6] = (v898_data + (v865_data * v896_data));
            float v901_data = s0[84];
            float v903_data = ir3[7];
            ir3[7] = (v903_data + (v865_data * v901_data));
            float v906_data = s0[96];
            float v908_data = ir3[8];
            ir3[8] = (v908_data + (v865_data * v906_data));
            float v911_data = s0[108];
            float v913_data = ir3[9];
            ir3[9] = (v913_data + (v865_data * v911_data));
            float v916_data = s0[120];
            float v918_data = ir3[10];
            ir3[10] = (v918_data + (v865_data * v916_data));
            float v921_data = s0[132];
            float v923_data = ir3[11];
            ir3[11] = (v923_data + (v865_data * v921_data));
          }
          if (v9_lead < 6) {
            float v929_data = r2[1];
            float v930_data = s0[1];
            float v932_data = ir3[0];
            ir3[0] = (v932_data + (v929_data * v930_data));
            float v935_data = s0[13];
            float v937_data = ir3[1];
            ir3[1] = (v937_data + (v929_data * v935_data));
            float v940_data = s0[25];
            float v942_data = ir3[2];
            ir3[2] = (v942_data + (v929_data * v940_data));
            float v945_data = s0[37];
            float v947_data = ir3[3];
            ir3[3] = (v947_data + (v929_data * v945_data));
            float v950_data = s0[49];
            float v952_data = ir3[4];
            ir3[4] = (v952_data + (v929_data * v950_data));
            float v955_data = s0[61];
            float v957_data = ir3[5];
            ir3[5] = (v957_data + (v929_data * v955_data));
            float v960_data = s0[73];
            float v962_data = ir3[6];
            ir3[6] = (v962_data + (v929_data * v960_data));
            float v965_data = s0[85];
            float v967_data = ir3[7];
            ir3[7] = (v967_data + (v929_data * v965_data));
            float v970_data = s0[97];
            float v972_data = ir3[8];
            ir3[8] = (v972_data + (v929_data * v970_data));
            float v975_data = s0[109];
            float v977_data = ir3[9];
            ir3[9] = (v977_data + (v929_data * v975_data));
            float v980_data = s0[121];
            float v982_data = ir3[10];
            ir3[10] = (v982_data + (v929_data * v980_data));
            float v985_data = s0[133];
            float v987_data = ir3[11];
            ir3[11] = (v987_data + (v929_data * v985_data));
          }
          if (v9_lead < 6) {
            float v993_data = r2[2];
            float v994_data = s0[2];
            float v996_data = ir3[0];
            ir3[0] = (v996_data + (v993_data * v994_data));
            float v999_data = s0[14];
            float v1001_data = ir3[1];
            ir3[1] = (v1001_data + (v993_data * v999_data));
            float v1004_data = s0[26];
            float v1006_data = ir3[2];
            ir3[2] = (v1006_data + (v993_data * v1004_data));
            float v1009_data = s0[38];
            float v1011_data = ir3[3];
            ir3[3] = (v1011_data + (v993_data * v1009_data));
            float v1014_data = s0[50];
            float v1016_data = ir3[4];
            ir3[4] = (v1016_data + (v993_data * v1014_data));
            float v1019_data = s0[62];
            float v1021_data = ir3[5];
            ir3[5] = (v1021_data + (v993_data * v1019_data));
            float v1024_data = s0[74];
            float v1026_data = ir3[6];
            ir3[6] = (v1026_data + (v993_data * v1024_data));
            float v1029_data = s0[86];
            float v1031_data = ir3[7];
            ir3[7] = (v1031_data + (v993_data * v1029_data));
            float v1034_data = s0[98];
            float v1036_data = ir3[8];
            ir3[8] = (v1036_data + (v993_data * v1034_data));
            float v1039_data = s0[110];
            float v1041_data = ir3[9];
            ir3[9] = (v1041_data + (v993_data * v1039_data));
            float v1044_data = s0[122];
            float v1046_data = ir3[10];
            ir3[10] = (v1046_data + (v993_data * v1044_data));
            float v1049_data = s0[134];
            float v1051_data = ir3[11];
            ir3[11] = (v1051_data + (v993_data * v1049_data));
          }
          if (v9_lead < 6) {
            float v1057_data = r2[3];
            float v1058_data = s0[3];
            float v1060_data = ir3[0];
            ir3[0] = (v1060_data + (v1057_data * v1058_data));
            float v1063_data = s0[15];
            float v1065_data = ir3[1];
            ir3[1] = (v1065_data + (v1057_data * v1063_data));
            float v1068_data = s0[27];
            float v1070_data = ir3[2];
            ir3[2] = (v1070_data + (v1057_data * v1068_data));
            float v1073_data = s0[39];
            float v1075_data = ir3[3];
            ir3[3] = (v1075_data + (v1057_data * v1073_data));
            float v1078_data = s0[51];
            float v1080_data = ir3[4];
            ir3[4] = (v1080_data + (v1057_data * v1078_data));
            float v1083_data = s0[63];
            float v1085_data = ir3[5];
            ir3[5] = (v1085_data + (v1057_data * v1083_data));
            float v1088_data = s0[75];
            float v1090_data = ir3[6];
            ir3[6] = (v1090_data + (v1057_data * v1088_data));
            float v1093_data = s0[87];
            float v1095_data = ir3[7];
            ir3[7] = (v1095_data + (v1057_data * v1093_data));
            float v1098_data = s0[99];
            float v1100_data = ir3[8];
            ir3[8] = (v1100_data + (v1057_data * v1098_data));
            float v1103_data = s0[111];
            float v1105_data = ir3[9];
            ir3[9] = (v1105_data + (v1057_data * v1103_data));
            float v1108_data = s0[123];
            float v1110_data = ir3[10];
            ir3[10] = (v1110_data + (v1057_data * v1108_data));
            float v1113_data = s0[135];
            float v1115_data = ir3[11];
            ir3[11] = (v1115_data + (v1057_data * v1113_data));
          }
          if (v9_lead < 6) {
            float v1121_data = r2[4];
            float v1122_data = s0[4];
            float v1124_data = ir3[0];
            ir3[0] = (v1124_data + (v1121_data * v1122_data));
            float v1127_data = s0[16];
            float v1129_data = ir3[1];
            ir3[1] = (v1129_data + (v1121_data * v1127_data));
            float v1132_data = s0[28];
            float v1134_data = ir3[2];
            ir3[2] = (v1134_data + (v1121_data * v1132_data));
            float v1137_data = s0[40];
            float v1139_data = ir3[3];
            ir3[3] = (v1139_data + (v1121_data * v1137_data));
            float v1142_data = s0[52];
            float v1144_data = ir3[4];
            ir3[4] = (v1144_data + (v1121_data * v1142_data));
            float v1147_data = s0[64];
            float v1149_data = ir3[5];
            ir3[5] = (v1149_data + (v1121_data * v1147_data));
            float v1152_data = s0[76];
            float v1154_data = ir3[6];
            ir3[6] = (v1154_data + (v1121_data * v1152_data));
            float v1157_data = s0[88];
            float v1159_data = ir3[7];
            ir3[7] = (v1159_data + (v1121_data * v1157_data));
            float v1162_data = s0[100];
            float v1164_data = ir3[8];
            ir3[8] = (v1164_data + (v1121_data * v1162_data));
            float v1167_data = s0[112];
            float v1169_data = ir3[9];
            ir3[9] = (v1169_data + (v1121_data * v1167_data));
            float v1172_data = s0[124];
            float v1174_data = ir3[10];
            ir3[10] = (v1174_data + (v1121_data * v1172_data));
            float v1177_data = s0[136];
            float v1179_data = ir3[11];
            ir3[11] = (v1179_data + (v1121_data * v1177_data));
          }
          if (v9_lead < 6) {
            float v1185_data = r2[5];
            float v1186_data = s0[5];
            float v1188_data = ir3[0];
            ir3[0] = (v1188_data + (v1185_data * v1186_data));
            float v1191_data = s0[17];
            float v1193_data = ir3[1];
            ir3[1] = (v1193_data + (v1185_data * v1191_data));
            float v1196_data = s0[29];
            float v1198_data = ir3[2];
            ir3[2] = (v1198_data + (v1185_data * v1196_data));
            float v1201_data = s0[41];
            float v1203_data = ir3[3];
            ir3[3] = (v1203_data + (v1185_data * v1201_data));
            float v1206_data = s0[53];
            float v1208_data = ir3[4];
            ir3[4] = (v1208_data + (v1185_data * v1206_data));
            float v1211_data = s0[65];
            float v1213_data = ir3[5];
            ir3[5] = (v1213_data + (v1185_data * v1211_data));
            float v1216_data = s0[77];
            float v1218_data = ir3[6];
            ir3[6] = (v1218_data + (v1185_data * v1216_data));
            float v1221_data = s0[89];
            float v1223_data = ir3[7];
            ir3[7] = (v1223_data + (v1185_data * v1221_data));
            float v1226_data = s0[101];
            float v1228_data = ir3[8];
            ir3[8] = (v1228_data + (v1185_data * v1226_data));
            float v1231_data = s0[113];
            float v1233_data = ir3[9];
            ir3[9] = (v1233_data + (v1185_data * v1231_data));
            float v1236_data = s0[125];
            float v1238_data = ir3[10];
            ir3[10] = (v1238_data + (v1185_data * v1236_data));
            float v1241_data = s0[137];
            float v1243_data = ir3[11];
            ir3[11] = (v1243_data + (v1185_data * v1241_data));
          }
          if (v9_lead < 6) {
            float v1249_data = r2[6];
            float v1250_data = s0[6];
            float v1252_data = ir3[0];
            ir3[0] = (v1252_data + (v1249_data * v1250_data));
            float v1255_data = s0[18];
            float v1257_data = ir3[1];
            ir3[1] = (v1257_data + (v1249_data * v1255_data));
            float v1260_data = s0[30];
            float v1262_data = ir3[2];
            ir3[2] = (v1262_data + (v1249_data * v1260_data));
            float v1265_data = s0[42];
            float v1267_data = ir3[3];
            ir3[3] = (v1267_data + (v1249_data * v1265_data));
            float v1270_data = s0[54];
            float v1272_data = ir3[4];
            ir3[4] = (v1272_data + (v1249_data * v1270_data));
            float v1275_data = s0[66];
            float v1277_data = ir3[5];
            ir3[5] = (v1277_data + (v1249_data * v1275_data));
            float v1280_data = s0[78];
            float v1282_data = ir3[6];
            ir3[6] = (v1282_data + (v1249_data * v1280_data));
            float v1285_data = s0[90];
            float v1287_data = ir3[7];
            ir3[7] = (v1287_data + (v1249_data * v1285_data));
            float v1290_data = s0[102];
            float v1292_data = ir3[8];
            ir3[8] = (v1292_data + (v1249_data * v1290_data));
            float v1295_data = s0[114];
            float v1297_data = ir3[9];
            ir3[9] = (v1297_data + (v1249_data * v1295_data));
            float v1300_data = s0[126];
            float v1302_data = ir3[10];
            ir3[10] = (v1302_data + (v1249_data * v1300_data));
            float v1305_data = s0[138];
            float v1307_data = ir3[11];
            ir3[11] = (v1307_data + (v1249_data * v1305_data));
          }
          if (v9_lead < 6) {
            float v1313_data = r2[7];
            float v1314_data = s0[7];
            float v1316_data = ir3[0];
            ir3[0] = (v1316_data + (v1313_data * v1314_data));
            float v1319_data = s0[19];
            float v1321_data = ir3[1];
            ir3[1] = (v1321_data + (v1313_data * v1319_data));
            float v1324_data = s0[31];
            float v1326_data = ir3[2];
            ir3[2] = (v1326_data + (v1313_data * v1324_data));
            float v1329_data = s0[43];
            float v1331_data = ir3[3];
            ir3[3] = (v1331_data + (v1313_data * v1329_data));
            float v1334_data = s0[55];
            float v1336_data = ir3[4];
            ir3[4] = (v1336_data + (v1313_data * v1334_data));
            float v1339_data = s0[67];
            float v1341_data = ir3[5];
            ir3[5] = (v1341_data + (v1313_data * v1339_data));
            float v1344_data = s0[79];
            float v1346_data = ir3[6];
            ir3[6] = (v1346_data + (v1313_data * v1344_data));
            float v1349_data = s0[91];
            float v1351_data = ir3[7];
            ir3[7] = (v1351_data + (v1313_data * v1349_data));
            float v1354_data = s0[103];
            float v1356_data = ir3[8];
            ir3[8] = (v1356_data + (v1313_data * v1354_data));
            float v1359_data = s0[115];
            float v1361_data = ir3[9];
            ir3[9] = (v1361_data + (v1313_data * v1359_data));
            float v1364_data = s0[127];
            float v1366_data = ir3[10];
            ir3[10] = (v1366_data + (v1313_data * v1364_data));
            float v1369_data = s0[139];
            float v1371_data = ir3[11];
            ir3[11] = (v1371_data + (v1313_data * v1369_data));
          }
          if (v9_lead < 6) {
            float v1377_data = r2[8];
            float v1378_data = s0[8];
            float v1380_data = ir3[0];
            ir3[0] = (v1380_data + (v1377_data * v1378_data));
            float v1383_data = s0[20];
            float v1385_data = ir3[1];
            ir3[1] = (v1385_data + (v1377_data * v1383_data));
            float v1388_data = s0[32];
            float v1390_data = ir3[2];
            ir3[2] = (v1390_data + (v1377_data * v1388_data));
            float v1393_data = s0[44];
            float v1395_data = ir3[3];
            ir3[3] = (v1395_data + (v1377_data * v1393_data));
            float v1398_data = s0[56];
            float v1400_data = ir3[4];
            ir3[4] = (v1400_data + (v1377_data * v1398_data));
            float v1403_data = s0[68];
            float v1405_data = ir3[5];
            ir3[5] = (v1405_data + (v1377_data * v1403_data));
            float v1408_data = s0[80];
            float v1410_data = ir3[6];
            ir3[6] = (v1410_data + (v1377_data * v1408_data));
            float v1413_data = s0[92];
            float v1415_data = ir3[7];
            ir3[7] = (v1415_data + (v1377_data * v1413_data));
            float v1418_data = s0[104];
            float v1420_data = ir3[8];
            ir3[8] = (v1420_data + (v1377_data * v1418_data));
            float v1423_data = s0[116];
            float v1425_data = ir3[9];
            ir3[9] = (v1425_data + (v1377_data * v1423_data));
            float v1428_data = s0[128];
            float v1430_data = ir3[10];
            ir3[10] = (v1430_data + (v1377_data * v1428_data));
            float v1433_data = s0[140];
            float v1435_data = ir3[11];
            ir3[11] = (v1435_data + (v1377_data * v1433_data));
          }
          if (v9_lead < 6) {
            float v1441_data = r2[9];
            float v1442_data = s0[9];
            float v1444_data = ir3[0];
            ir3[0] = (v1444_data + (v1441_data * v1442_data));
            float v1447_data = s0[21];
            float v1449_data = ir3[1];
            ir3[1] = (v1449_data + (v1441_data * v1447_data));
            float v1452_data = s0[33];
            float v1454_data = ir3[2];
            ir3[2] = (v1454_data + (v1441_data * v1452_data));
            float v1457_data = s0[45];
            float v1459_data = ir3[3];
            ir3[3] = (v1459_data + (v1441_data * v1457_data));
            float v1462_data = s0[57];
            float v1464_data = ir3[4];
            ir3[4] = (v1464_data + (v1441_data * v1462_data));
            float v1467_data = s0[69];
            float v1469_data = ir3[5];
            ir3[5] = (v1469_data + (v1441_data * v1467_data));
            float v1472_data = s0[81];
            float v1474_data = ir3[6];
            ir3[6] = (v1474_data + (v1441_data * v1472_data));
            float v1477_data = s0[93];
            float v1479_data = ir3[7];
            ir3[7] = (v1479_data + (v1441_data * v1477_data));
            float v1482_data = s0[105];
            float v1484_data = ir3[8];
            ir3[8] = (v1484_data + (v1441_data * v1482_data));
            float v1487_data = s0[117];
            float v1489_data = ir3[9];
            ir3[9] = (v1489_data + (v1441_data * v1487_data));
            float v1492_data = s0[129];
            float v1494_data = ir3[10];
            ir3[10] = (v1494_data + (v1441_data * v1492_data));
            float v1497_data = s0[141];
            float v1499_data = ir3[11];
            ir3[11] = (v1499_data + (v1441_data * v1497_data));
          }
          if (v9_lead < 6) {
            float v1505_data = r2[10];
            float v1506_data = s0[10];
            float v1508_data = ir3[0];
            ir3[0] = (v1508_data + (v1505_data * v1506_data));
            float v1511_data = s0[22];
            float v1513_data = ir3[1];
            ir3[1] = (v1513_data + (v1505_data * v1511_data));
            float v1516_data = s0[34];
            float v1518_data = ir3[2];
            ir3[2] = (v1518_data + (v1505_data * v1516_data));
            float v1521_data = s0[46];
            float v1523_data = ir3[3];
            ir3[3] = (v1523_data + (v1505_data * v1521_data));
            float v1526_data = s0[58];
            float v1528_data = ir3[4];
            ir3[4] = (v1528_data + (v1505_data * v1526_data));
            float v1531_data = s0[70];
            float v1533_data = ir3[5];
            ir3[5] = (v1533_data + (v1505_data * v1531_data));
            float v1536_data = s0[82];
            float v1538_data = ir3[6];
            ir3[6] = (v1538_data + (v1505_data * v1536_data));
            float v1541_data = s0[94];
            float v1543_data = ir3[7];
            ir3[7] = (v1543_data + (v1505_data * v1541_data));
            float v1546_data = s0[106];
            float v1548_data = ir3[8];
            ir3[8] = (v1548_data + (v1505_data * v1546_data));
            float v1551_data = s0[118];
            float v1553_data = ir3[9];
            ir3[9] = (v1553_data + (v1505_data * v1551_data));
            float v1556_data = s0[130];
            float v1558_data = ir3[10];
            ir3[10] = (v1558_data + (v1505_data * v1556_data));
            float v1561_data = s0[142];
            float v1563_data = ir3[11];
            ir3[11] = (v1563_data + (v1505_data * v1561_data));
          }
          if (v9_lead < 6) {
            float v1569_data = r2[11];
            float v1570_data = s0[11];
            float v1572_data = ir3[0];
            ir3[0] = (v1572_data + (v1569_data * v1570_data));
            float v1575_data = s0[23];
            float v1577_data = ir3[1];
            ir3[1] = (v1577_data + (v1569_data * v1575_data));
            float v1580_data = s0[35];
            float v1582_data = ir3[2];
            ir3[2] = (v1582_data + (v1569_data * v1580_data));
            float v1585_data = s0[47];
            float v1587_data = ir3[3];
            ir3[3] = (v1587_data + (v1569_data * v1585_data));
            float v1590_data = s0[59];
            float v1592_data = ir3[4];
            ir3[4] = (v1592_data + (v1569_data * v1590_data));
            float v1595_data = s0[71];
            float v1597_data = ir3[5];
            ir3[5] = (v1597_data + (v1569_data * v1595_data));
            float v1600_data = s0[83];
            float v1602_data = ir3[6];
            ir3[6] = (v1602_data + (v1569_data * v1600_data));
            float v1605_data = s0[95];
            float v1607_data = ir3[7];
            ir3[7] = (v1607_data + (v1569_data * v1605_data));
            float v1610_data = s0[107];
            float v1612_data = ir3[8];
            ir3[8] = (v1612_data + (v1569_data * v1610_data));
            float v1615_data = s0[119];
            float v1617_data = ir3[9];
            ir3[9] = (v1617_data + (v1569_data * v1615_data));
            float v1620_data = s0[131];
            float v1622_data = ir3[10];
            ir3[10] = (v1622_data + (v1569_data * v1620_data));
            float v1625_data = s0[143];
            float v1627_data = ir3[11];
            ir3[11] = (v1627_data + (v1569_data * v1625_data));
          }
          if (v9_lead < 6) {
            #pragma unroll
            for (int32_t v1633_n1 = 0; v1633_n1 < 12; ++v1633_n1) {
              int32_t v1634_a = 0 + v1633_n1;
              float v1636_data = ir3[v1633_n1];
              r3[v1633_n1] = v1636_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v9_lead < 6) {
            int32_t v1651_off = v9_lead + 6;
            #pragma unroll
            for (int32_t v1642_i1 = 0; v1642_i1 < 12; ++v1642_i1) {
              int32_t v1643_a = 0 + v1642_i1;
              float v1645_data = r3[v1642_i1];
              s1[(v1651_off + (v1642_i1 * 12))] = v1645_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v9_lead < 12) {
            float v1660_data = r4[0];
            float v1661_data = s1[0];
            float v1663_data = ir5[0];
            ir5[0] = (v1663_data + (v1660_data * v1661_data));
            float v1666_data = s1[12];
            float v1668_data = ir5[1];
            ir5[1] = (v1668_data + (v1660_data * v1666_data));
            float v1671_data = s1[24];
            float v1673_data = ir5[2];
            ir5[2] = (v1673_data + (v1660_data * v1671_data));
            float v1676_data = s1[36];
            float v1678_data = ir5[3];
            ir5[3] = (v1678_data + (v1660_data * v1676_data));
            float v1681_data = s1[48];
            float v1683_data = ir5[4];
            ir5[4] = (v1683_data + (v1660_data * v1681_data));
            float v1686_data = s1[60];
            float v1688_data = ir5[5];
            ir5[5] = (v1688_data + (v1660_data * v1686_data));
            float v1691_data = s1[72];
            float v1693_data = ir5[6];
            ir5[6] = (v1693_data + (v1660_data * v1691_data));
            float v1696_data = s1[84];
            float v1698_data = ir5[7];
            ir5[7] = (v1698_data + (v1660_data * v1696_data));
            float v1701_data = s1[96];
            float v1703_data = ir5[8];
            ir5[8] = (v1703_data + (v1660_data * v1701_data));
            float v1706_data = s1[108];
            float v1708_data = ir5[9];
            ir5[9] = (v1708_data + (v1660_data * v1706_data));
            float v1711_data = s1[120];
            float v1713_data = ir5[10];
            ir5[10] = (v1713_data + (v1660_data * v1711_data));
            float v1716_data = s1[132];
            float v1718_data = ir5[11];
            ir5[11] = (v1718_data + (v1660_data * v1716_data));
          }
          if (v9_lead < 12) {
            float v1724_data = r4[1];
            float v1725_data = s1[1];
            float v1727_data = ir5[0];
            ir5[0] = (v1727_data + (v1724_data * v1725_data));
            float v1730_data = s1[13];
            float v1732_data = ir5[1];
            ir5[1] = (v1732_data + (v1724_data * v1730_data));
            float v1735_data = s1[25];
            float v1737_data = ir5[2];
            ir5[2] = (v1737_data + (v1724_data * v1735_data));
            float v1740_data = s1[37];
            float v1742_data = ir5[3];
            ir5[3] = (v1742_data + (v1724_data * v1740_data));
            float v1745_data = s1[49];
            float v1747_data = ir5[4];
            ir5[4] = (v1747_data + (v1724_data * v1745_data));
            float v1750_data = s1[61];
            float v1752_data = ir5[5];
            ir5[5] = (v1752_data + (v1724_data * v1750_data));
            float v1755_data = s1[73];
            float v1757_data = ir5[6];
            ir5[6] = (v1757_data + (v1724_data * v1755_data));
            float v1760_data = s1[85];
            float v1762_data = ir5[7];
            ir5[7] = (v1762_data + (v1724_data * v1760_data));
            float v1765_data = s1[97];
            float v1767_data = ir5[8];
            ir5[8] = (v1767_data + (v1724_data * v1765_data));
            float v1770_data = s1[109];
            float v1772_data = ir5[9];
            ir5[9] = (v1772_data + (v1724_data * v1770_data));
            float v1775_data = s1[121];
            float v1777_data = ir5[10];
            ir5[10] = (v1777_data + (v1724_data * v1775_data));
            float v1780_data = s1[133];
            float v1782_data = ir5[11];
            ir5[11] = (v1782_data + (v1724_data * v1780_data));
          }
          if (v9_lead < 12) {
            float v1788_data = r4[2];
            float v1789_data = s1[2];
            float v1791_data = ir5[0];
            ir5[0] = (v1791_data + (v1788_data * v1789_data));
            float v1794_data = s1[14];
            float v1796_data = ir5[1];
            ir5[1] = (v1796_data + (v1788_data * v1794_data));
            float v1799_data = s1[26];
            float v1801_data = ir5[2];
            ir5[2] = (v1801_data + (v1788_data * v1799_data));
            float v1804_data = s1[38];
            float v1806_data = ir5[3];
            ir5[3] = (v1806_data + (v1788_data * v1804_data));
            float v1809_data = s1[50];
            float v1811_data = ir5[4];
            ir5[4] = (v1811_data + (v1788_data * v1809_data));
            float v1814_data = s1[62];
            float v1816_data = ir5[5];
            ir5[5] = (v1816_data + (v1788_data * v1814_data));
            float v1819_data = s1[74];
            float v1821_data = ir5[6];
            ir5[6] = (v1821_data + (v1788_data * v1819_data));
            float v1824_data = s1[86];
            float v1826_data = ir5[7];
            ir5[7] = (v1826_data + (v1788_data * v1824_data));
            float v1829_data = s1[98];
            float v1831_data = ir5[8];
            ir5[8] = (v1831_data + (v1788_data * v1829_data));
            float v1834_data = s1[110];
            float v1836_data = ir5[9];
            ir5[9] = (v1836_data + (v1788_data * v1834_data));
            float v1839_data = s1[122];
            float v1841_data = ir5[10];
            ir5[10] = (v1841_data + (v1788_data * v1839_data));
            float v1844_data = s1[134];
            float v1846_data = ir5[11];
            ir5[11] = (v1846_data + (v1788_data * v1844_data));
          }
          if (v9_lead < 12) {
            float v1852_data = r4[3];
            float v1853_data = s1[3];
            float v1855_data = ir5[0];
            ir5[0] = (v1855_data + (v1852_data * v1853_data));
            float v1858_data = s1[15];
            float v1860_data = ir5[1];
            ir5[1] = (v1860_data + (v1852_data * v1858_data));
            float v1863_data = s1[27];
            float v1865_data = ir5[2];
            ir5[2] = (v1865_data + (v1852_data * v1863_data));
            float v1868_data = s1[39];
            float v1870_data = ir5[3];
            ir5[3] = (v1870_data + (v1852_data * v1868_data));
            float v1873_data = s1[51];
            float v1875_data = ir5[4];
            ir5[4] = (v1875_data + (v1852_data * v1873_data));
            float v1878_data = s1[63];
            float v1880_data = ir5[5];
            ir5[5] = (v1880_data + (v1852_data * v1878_data));
            float v1883_data = s1[75];
            float v1885_data = ir5[6];
            ir5[6] = (v1885_data + (v1852_data * v1883_data));
            float v1888_data = s1[87];
            float v1890_data = ir5[7];
            ir5[7] = (v1890_data + (v1852_data * v1888_data));
            float v1893_data = s1[99];
            float v1895_data = ir5[8];
            ir5[8] = (v1895_data + (v1852_data * v1893_data));
            float v1898_data = s1[111];
            float v1900_data = ir5[9];
            ir5[9] = (v1900_data + (v1852_data * v1898_data));
            float v1903_data = s1[123];
            float v1905_data = ir5[10];
            ir5[10] = (v1905_data + (v1852_data * v1903_data));
            float v1908_data = s1[135];
            float v1910_data = ir5[11];
            ir5[11] = (v1910_data + (v1852_data * v1908_data));
          }
          if (v9_lead < 12) {
            float v1916_data = r4[4];
            float v1917_data = s1[4];
            float v1919_data = ir5[0];
            ir5[0] = (v1919_data + (v1916_data * v1917_data));
            float v1922_data = s1[16];
            float v1924_data = ir5[1];
            ir5[1] = (v1924_data + (v1916_data * v1922_data));
            float v1927_data = s1[28];
            float v1929_data = ir5[2];
            ir5[2] = (v1929_data + (v1916_data * v1927_data));
            float v1932_data = s1[40];
            float v1934_data = ir5[3];
            ir5[3] = (v1934_data + (v1916_data * v1932_data));
            float v1937_data = s1[52];
            float v1939_data = ir5[4];
            ir5[4] = (v1939_data + (v1916_data * v1937_data));
            float v1942_data = s1[64];
            float v1944_data = ir5[5];
            ir5[5] = (v1944_data + (v1916_data * v1942_data));
            float v1947_data = s1[76];
            float v1949_data = ir5[6];
            ir5[6] = (v1949_data + (v1916_data * v1947_data));
            float v1952_data = s1[88];
            float v1954_data = ir5[7];
            ir5[7] = (v1954_data + (v1916_data * v1952_data));
            float v1957_data = s1[100];
            float v1959_data = ir5[8];
            ir5[8] = (v1959_data + (v1916_data * v1957_data));
            float v1962_data = s1[112];
            float v1964_data = ir5[9];
            ir5[9] = (v1964_data + (v1916_data * v1962_data));
            float v1967_data = s1[124];
            float v1969_data = ir5[10];
            ir5[10] = (v1969_data + (v1916_data * v1967_data));
            float v1972_data = s1[136];
            float v1974_data = ir5[11];
            ir5[11] = (v1974_data + (v1916_data * v1972_data));
          }
          if (v9_lead < 12) {
            float v1980_data = r4[5];
            float v1981_data = s1[5];
            float v1983_data = ir5[0];
            ir5[0] = (v1983_data + (v1980_data * v1981_data));
            float v1986_data = s1[17];
            float v1988_data = ir5[1];
            ir5[1] = (v1988_data + (v1980_data * v1986_data));
            float v1991_data = s1[29];
            float v1993_data = ir5[2];
            ir5[2] = (v1993_data + (v1980_data * v1991_data));
            float v1996_data = s1[41];
            float v1998_data = ir5[3];
            ir5[3] = (v1998_data + (v1980_data * v1996_data));
            float v2001_data = s1[53];
            float v2003_data = ir5[4];
            ir5[4] = (v2003_data + (v1980_data * v2001_data));
            float v2006_data = s1[65];
            float v2008_data = ir5[5];
            ir5[5] = (v2008_data + (v1980_data * v2006_data));
            float v2011_data = s1[77];
            float v2013_data = ir5[6];
            ir5[6] = (v2013_data + (v1980_data * v2011_data));
            float v2016_data = s1[89];
            float v2018_data = ir5[7];
            ir5[7] = (v2018_data + (v1980_data * v2016_data));
            float v2021_data = s1[101];
            float v2023_data = ir5[8];
            ir5[8] = (v2023_data + (v1980_data * v2021_data));
            float v2026_data = s1[113];
            float v2028_data = ir5[9];
            ir5[9] = (v2028_data + (v1980_data * v2026_data));
            float v2031_data = s1[125];
            float v2033_data = ir5[10];
            ir5[10] = (v2033_data + (v1980_data * v2031_data));
            float v2036_data = s1[137];
            float v2038_data = ir5[11];
            ir5[11] = (v2038_data + (v1980_data * v2036_data));
          }
          if (v9_lead < 12) {
            float v2044_data = r4[6];
            float v2045_data = s1[6];
            float v2047_data = ir5[0];
            ir5[0] = (v2047_data + (v2044_data * v2045_data));
            float v2050_data = s1[18];
            float v2052_data = ir5[1];
            ir5[1] = (v2052_data + (v2044_data * v2050_data));
            float v2055_data = s1[30];
            float v2057_data = ir5[2];
            ir5[2] = (v2057_data + (v2044_data * v2055_data));
            float v2060_data = s1[42];
            float v2062_data = ir5[3];
            ir5[3] = (v2062_data + (v2044_data * v2060_data));
            float v2065_data = s1[54];
            float v2067_data = ir5[4];
            ir5[4] = (v2067_data + (v2044_data * v2065_data));
            float v2070_data = s1[66];
            float v2072_data = ir5[5];
            ir5[5] = (v2072_data + (v2044_data * v2070_data));
            float v2075_data = s1[78];
            float v2077_data = ir5[6];
            ir5[6] = (v2077_data + (v2044_data * v2075_data));
            float v2080_data = s1[90];
            float v2082_data = ir5[7];
            ir5[7] = (v2082_data + (v2044_data * v2080_data));
            float v2085_data = s1[102];
            float v2087_data = ir5[8];
            ir5[8] = (v2087_data + (v2044_data * v2085_data));
            float v2090_data = s1[114];
            float v2092_data = ir5[9];
            ir5[9] = (v2092_data + (v2044_data * v2090_data));
            float v2095_data = s1[126];
            float v2097_data = ir5[10];
            ir5[10] = (v2097_data + (v2044_data * v2095_data));
            float v2100_data = s1[138];
            float v2102_data = ir5[11];
            ir5[11] = (v2102_data + (v2044_data * v2100_data));
          }
          if (v9_lead < 12) {
            float v2108_data = r4[7];
            float v2109_data = s1[7];
            float v2111_data = ir5[0];
            ir5[0] = (v2111_data + (v2108_data * v2109_data));
            float v2114_data = s1[19];
            float v2116_data = ir5[1];
            ir5[1] = (v2116_data + (v2108_data * v2114_data));
            float v2119_data = s1[31];
            float v2121_data = ir5[2];
            ir5[2] = (v2121_data + (v2108_data * v2119_data));
            float v2124_data = s1[43];
            float v2126_data = ir5[3];
            ir5[3] = (v2126_data + (v2108_data * v2124_data));
            float v2129_data = s1[55];
            float v2131_data = ir5[4];
            ir5[4] = (v2131_data + (v2108_data * v2129_data));
            float v2134_data = s1[67];
            float v2136_data = ir5[5];
            ir5[5] = (v2136_data + (v2108_data * v2134_data));
            float v2139_data = s1[79];
            float v2141_data = ir5[6];
            ir5[6] = (v2141_data + (v2108_data * v2139_data));
            float v2144_data = s1[91];
            float v2146_data = ir5[7];
            ir5[7] = (v2146_data + (v2108_data * v2144_data));
            float v2149_data = s1[103];
            float v2151_data = ir5[8];
            ir5[8] = (v2151_data + (v2108_data * v2149_data));
            float v2154_data = s1[115];
            float v2156_data = ir5[9];
            ir5[9] = (v2156_data + (v2108_data * v2154_data));
            float v2159_data = s1[127];
            float v2161_data = ir5[10];
            ir5[10] = (v2161_data + (v2108_data * v2159_data));
            float v2164_data = s1[139];
            float v2166_data = ir5[11];
            ir5[11] = (v2166_data + (v2108_data * v2164_data));
          }
          if (v9_lead < 12) {
            float v2172_data = r4[8];
            float v2173_data = s1[8];
            float v2175_data = ir5[0];
            ir5[0] = (v2175_data + (v2172_data * v2173_data));
            float v2178_data = s1[20];
            float v2180_data = ir5[1];
            ir5[1] = (v2180_data + (v2172_data * v2178_data));
            float v2183_data = s1[32];
            float v2185_data = ir5[2];
            ir5[2] = (v2185_data + (v2172_data * v2183_data));
            float v2188_data = s1[44];
            float v2190_data = ir5[3];
            ir5[3] = (v2190_data + (v2172_data * v2188_data));
            float v2193_data = s1[56];
            float v2195_data = ir5[4];
            ir5[4] = (v2195_data + (v2172_data * v2193_data));
            float v2198_data = s1[68];
            float v2200_data = ir5[5];
            ir5[5] = (v2200_data + (v2172_data * v2198_data));
            float v2203_data = s1[80];
            float v2205_data = ir5[6];
            ir5[6] = (v2205_data + (v2172_data * v2203_data));
            float v2208_data = s1[92];
            float v2210_data = ir5[7];
            ir5[7] = (v2210_data + (v2172_data * v2208_data));
            float v2213_data = s1[104];
            float v2215_data = ir5[8];
            ir5[8] = (v2215_data + (v2172_data * v2213_data));
            float v2218_data = s1[116];
            float v2220_data = ir5[9];
            ir5[9] = (v2220_data + (v2172_data * v2218_data));
            float v2223_data = s1[128];
            float v2225_data = ir5[10];
            ir5[10] = (v2225_data + (v2172_data * v2223_data));
            float v2228_data = s1[140];
            float v2230_data = ir5[11];
            ir5[11] = (v2230_data + (v2172_data * v2228_data));
          }
          if (v9_lead < 12) {
            float v2236_data = r4[9];
            float v2237_data = s1[9];
            float v2239_data = ir5[0];
            ir5[0] = (v2239_data + (v2236_data * v2237_data));
            float v2242_data = s1[21];
            float v2244_data = ir5[1];
            ir5[1] = (v2244_data + (v2236_data * v2242_data));
            float v2247_data = s1[33];
            float v2249_data = ir5[2];
            ir5[2] = (v2249_data + (v2236_data * v2247_data));
            float v2252_data = s1[45];
            float v2254_data = ir5[3];
            ir5[3] = (v2254_data + (v2236_data * v2252_data));
            float v2257_data = s1[57];
            float v2259_data = ir5[4];
            ir5[4] = (v2259_data + (v2236_data * v2257_data));
            float v2262_data = s1[69];
            float v2264_data = ir5[5];
            ir5[5] = (v2264_data + (v2236_data * v2262_data));
            float v2267_data = s1[81];
            float v2269_data = ir5[6];
            ir5[6] = (v2269_data + (v2236_data * v2267_data));
            float v2272_data = s1[93];
            float v2274_data = ir5[7];
            ir5[7] = (v2274_data + (v2236_data * v2272_data));
            float v2277_data = s1[105];
            float v2279_data = ir5[8];
            ir5[8] = (v2279_data + (v2236_data * v2277_data));
            float v2282_data = s1[117];
            float v2284_data = ir5[9];
            ir5[9] = (v2284_data + (v2236_data * v2282_data));
            float v2287_data = s1[129];
            float v2289_data = ir5[10];
            ir5[10] = (v2289_data + (v2236_data * v2287_data));
            float v2292_data = s1[141];
            float v2294_data = ir5[11];
            ir5[11] = (v2294_data + (v2236_data * v2292_data));
          }
          if (v9_lead < 12) {
            float v2300_data = r4[10];
            float v2301_data = s1[10];
            float v2303_data = ir5[0];
            ir5[0] = (v2303_data + (v2300_data * v2301_data));
            float v2306_data = s1[22];
            float v2308_data = ir5[1];
            ir5[1] = (v2308_data + (v2300_data * v2306_data));
            float v2311_data = s1[34];
            float v2313_data = ir5[2];
            ir5[2] = (v2313_data + (v2300_data * v2311_data));
            float v2316_data = s1[46];
            float v2318_data = ir5[3];
            ir5[3] = (v2318_data + (v2300_data * v2316_data));
            float v2321_data = s1[58];
            float v2323_data = ir5[4];
            ir5[4] = (v2323_data + (v2300_data * v2321_data));
            float v2326_data = s1[70];
            float v2328_data = ir5[5];
            ir5[5] = (v2328_data + (v2300_data * v2326_data));
            float v2331_data = s1[82];
            float v2333_data = ir5[6];
            ir5[6] = (v2333_data + (v2300_data * v2331_data));
            float v2336_data = s1[94];
            float v2338_data = ir5[7];
            ir5[7] = (v2338_data + (v2300_data * v2336_data));
            float v2341_data = s1[106];
            float v2343_data = ir5[8];
            ir5[8] = (v2343_data + (v2300_data * v2341_data));
            float v2346_data = s1[118];
            float v2348_data = ir5[9];
            ir5[9] = (v2348_data + (v2300_data * v2346_data));
            float v2351_data = s1[130];
            float v2353_data = ir5[10];
            ir5[10] = (v2353_data + (v2300_data * v2351_data));
            float v2356_data = s1[142];
            float v2358_data = ir5[11];
            ir5[11] = (v2358_data + (v2300_data * v2356_data));
          }
          if (v9_lead < 12) {
            float v2364_data = r4[11];
            float v2365_data = s1[11];
            float v2367_data = ir5[0];
            ir5[0] = (v2367_data + (v2364_data * v2365_data));
            float v2370_data = s1[23];
            float v2372_data = ir5[1];
            ir5[1] = (v2372_data + (v2364_data * v2370_data));
            float v2375_data = s1[35];
            float v2377_data = ir5[2];
            ir5[2] = (v2377_data + (v2364_data * v2375_data));
            float v2380_data = s1[47];
            float v2382_data = ir5[3];
            ir5[3] = (v2382_data + (v2364_data * v2380_data));
            float v2385_data = s1[59];
            float v2387_data = ir5[4];
            ir5[4] = (v2387_data + (v2364_data * v2385_data));
            float v2390_data = s1[71];
            float v2392_data = ir5[5];
            ir5[5] = (v2392_data + (v2364_data * v2390_data));
            float v2395_data = s1[83];
            float v2397_data = ir5[6];
            ir5[6] = (v2397_data + (v2364_data * v2395_data));
            float v2400_data = s1[95];
            float v2402_data = ir5[7];
            ir5[7] = (v2402_data + (v2364_data * v2400_data));
            float v2405_data = s1[107];
            float v2407_data = ir5[8];
            ir5[8] = (v2407_data + (v2364_data * v2405_data));
            float v2410_data = s1[119];
            float v2412_data = ir5[9];
            ir5[9] = (v2412_data + (v2364_data * v2410_data));
            float v2415_data = s1[131];
            float v2417_data = ir5[10];
            ir5[10] = (v2417_data + (v2364_data * v2415_data));
            float v2420_data = s1[143];
            float v2422_data = ir5[11];
            ir5[11] = (v2422_data + (v2364_data * v2420_data));
          }
          if (v9_lead < 12) {
            #pragma unroll
            for (int32_t v2428_n1 = 0; v2428_n1 < 12; ++v2428_n1) {
              int32_t v2429_a = 0 + v2428_n1;
              float v2431_data = ir5[v2428_n1];
              r5[v2428_n1] = v2431_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v9_lead < 12) {
            #pragma unroll
            for (int32_t v2437_i1 = 0; v2437_i1 < 12; ++v2437_i1) {
              int32_t v2438_a = 0 + v2437_i1;
              float v2440_data = r5[v2437_i1];
              glb_m3[(v9_lead + (v2437_i1 * 12))] = v2440_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

