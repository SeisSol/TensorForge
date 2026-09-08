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
      float* __restrict__ s0 = &localShrMem0[0];
      float* __restrict__ s1 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v17_lead = threadIdx.x % 16;
          if (v17_lead < 6) {
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 12; ++v19_i1) {
              float v27_data = __ldcg(&glb_m0[(v17_lead + (v19_i1 * 6))]);
              r0[v19_i1] = v27_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v17_lead < 6) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
              float v43_data = __ldcg(&glb_m2[(v17_lead + (v35_i1 * 6))]);
              r2[v35_i1] = v43_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          if (v17_lead < 6) {
            float v50_data = r0[0];
            float v51_data = s0[0];
            float v53_data = r1[0];
            r1[0] = (v53_data + (v50_data * v51_data));
            float v56_data = s0[12];
            float v58_data = r1[1];
            r1[1] = (v58_data + (v50_data * v56_data));
            float v61_data = s0[24];
            float v63_data = r1[2];
            r1[2] = (v63_data + (v50_data * v61_data));
            float v66_data = s0[36];
            float v68_data = r1[3];
            r1[3] = (v68_data + (v50_data * v66_data));
            float v71_data = s0[48];
            float v73_data = r1[4];
            r1[4] = (v73_data + (v50_data * v71_data));
            float v76_data = s0[60];
            float v78_data = r1[5];
            r1[5] = (v78_data + (v50_data * v76_data));
            float v81_data = s0[72];
            float v83_data = r1[6];
            r1[6] = (v83_data + (v50_data * v81_data));
            float v86_data = s0[84];
            float v88_data = r1[7];
            r1[7] = (v88_data + (v50_data * v86_data));
            float v91_data = s0[96];
            float v93_data = r1[8];
            r1[8] = (v93_data + (v50_data * v91_data));
            float v96_data = s0[108];
            float v98_data = r1[9];
            r1[9] = (v98_data + (v50_data * v96_data));
            float v101_data = s0[120];
            float v103_data = r1[10];
            r1[10] = (v103_data + (v50_data * v101_data));
            float v106_data = s0[132];
            float v108_data = r1[11];
            r1[11] = (v108_data + (v50_data * v106_data));
          }
          if (v17_lead < 6) {
            float v114_data = r0[1];
            float v115_data = s0[1];
            float v117_data = r1[0];
            r1[0] = (v117_data + (v114_data * v115_data));
            float v120_data = s0[13];
            float v122_data = r1[1];
            r1[1] = (v122_data + (v114_data * v120_data));
            float v125_data = s0[25];
            float v127_data = r1[2];
            r1[2] = (v127_data + (v114_data * v125_data));
            float v130_data = s0[37];
            float v132_data = r1[3];
            r1[3] = (v132_data + (v114_data * v130_data));
            float v135_data = s0[49];
            float v137_data = r1[4];
            r1[4] = (v137_data + (v114_data * v135_data));
            float v140_data = s0[61];
            float v142_data = r1[5];
            r1[5] = (v142_data + (v114_data * v140_data));
            float v145_data = s0[73];
            float v147_data = r1[6];
            r1[6] = (v147_data + (v114_data * v145_data));
            float v150_data = s0[85];
            float v152_data = r1[7];
            r1[7] = (v152_data + (v114_data * v150_data));
            float v155_data = s0[97];
            float v157_data = r1[8];
            r1[8] = (v157_data + (v114_data * v155_data));
            float v160_data = s0[109];
            float v162_data = r1[9];
            r1[9] = (v162_data + (v114_data * v160_data));
            float v165_data = s0[121];
            float v167_data = r1[10];
            r1[10] = (v167_data + (v114_data * v165_data));
            float v170_data = s0[133];
            float v172_data = r1[11];
            r1[11] = (v172_data + (v114_data * v170_data));
          }
          if (v17_lead < 6) {
            float v178_data = r0[2];
            float v179_data = s0[2];
            float v181_data = r1[0];
            r1[0] = (v181_data + (v178_data * v179_data));
            float v184_data = s0[14];
            float v186_data = r1[1];
            r1[1] = (v186_data + (v178_data * v184_data));
            float v189_data = s0[26];
            float v191_data = r1[2];
            r1[2] = (v191_data + (v178_data * v189_data));
            float v194_data = s0[38];
            float v196_data = r1[3];
            r1[3] = (v196_data + (v178_data * v194_data));
            float v199_data = s0[50];
            float v201_data = r1[4];
            r1[4] = (v201_data + (v178_data * v199_data));
            float v204_data = s0[62];
            float v206_data = r1[5];
            r1[5] = (v206_data + (v178_data * v204_data));
            float v209_data = s0[74];
            float v211_data = r1[6];
            r1[6] = (v211_data + (v178_data * v209_data));
            float v214_data = s0[86];
            float v216_data = r1[7];
            r1[7] = (v216_data + (v178_data * v214_data));
            float v219_data = s0[98];
            float v221_data = r1[8];
            r1[8] = (v221_data + (v178_data * v219_data));
            float v224_data = s0[110];
            float v226_data = r1[9];
            r1[9] = (v226_data + (v178_data * v224_data));
            float v229_data = s0[122];
            float v231_data = r1[10];
            r1[10] = (v231_data + (v178_data * v229_data));
            float v234_data = s0[134];
            float v236_data = r1[11];
            r1[11] = (v236_data + (v178_data * v234_data));
          }
          if (v17_lead < 6) {
            float v242_data = r0[3];
            float v243_data = s0[3];
            float v245_data = r1[0];
            r1[0] = (v245_data + (v242_data * v243_data));
            float v248_data = s0[15];
            float v250_data = r1[1];
            r1[1] = (v250_data + (v242_data * v248_data));
            float v253_data = s0[27];
            float v255_data = r1[2];
            r1[2] = (v255_data + (v242_data * v253_data));
            float v258_data = s0[39];
            float v260_data = r1[3];
            r1[3] = (v260_data + (v242_data * v258_data));
            float v263_data = s0[51];
            float v265_data = r1[4];
            r1[4] = (v265_data + (v242_data * v263_data));
            float v268_data = s0[63];
            float v270_data = r1[5];
            r1[5] = (v270_data + (v242_data * v268_data));
            float v273_data = s0[75];
            float v275_data = r1[6];
            r1[6] = (v275_data + (v242_data * v273_data));
            float v278_data = s0[87];
            float v280_data = r1[7];
            r1[7] = (v280_data + (v242_data * v278_data));
            float v283_data = s0[99];
            float v285_data = r1[8];
            r1[8] = (v285_data + (v242_data * v283_data));
            float v288_data = s0[111];
            float v290_data = r1[9];
            r1[9] = (v290_data + (v242_data * v288_data));
            float v293_data = s0[123];
            float v295_data = r1[10];
            r1[10] = (v295_data + (v242_data * v293_data));
            float v298_data = s0[135];
            float v300_data = r1[11];
            r1[11] = (v300_data + (v242_data * v298_data));
          }
          if (v17_lead < 6) {
            float v306_data = r0[4];
            float v307_data = s0[4];
            float v309_data = r1[0];
            r1[0] = (v309_data + (v306_data * v307_data));
            float v312_data = s0[16];
            float v314_data = r1[1];
            r1[1] = (v314_data + (v306_data * v312_data));
            float v317_data = s0[28];
            float v319_data = r1[2];
            r1[2] = (v319_data + (v306_data * v317_data));
            float v322_data = s0[40];
            float v324_data = r1[3];
            r1[3] = (v324_data + (v306_data * v322_data));
            float v327_data = s0[52];
            float v329_data = r1[4];
            r1[4] = (v329_data + (v306_data * v327_data));
            float v332_data = s0[64];
            float v334_data = r1[5];
            r1[5] = (v334_data + (v306_data * v332_data));
            float v337_data = s0[76];
            float v339_data = r1[6];
            r1[6] = (v339_data + (v306_data * v337_data));
            float v342_data = s0[88];
            float v344_data = r1[7];
            r1[7] = (v344_data + (v306_data * v342_data));
            float v347_data = s0[100];
            float v349_data = r1[8];
            r1[8] = (v349_data + (v306_data * v347_data));
            float v352_data = s0[112];
            float v354_data = r1[9];
            r1[9] = (v354_data + (v306_data * v352_data));
            float v357_data = s0[124];
            float v359_data = r1[10];
            r1[10] = (v359_data + (v306_data * v357_data));
            float v362_data = s0[136];
            float v364_data = r1[11];
            r1[11] = (v364_data + (v306_data * v362_data));
          }
          if (v17_lead < 6) {
            float v370_data = r0[5];
            float v371_data = s0[5];
            float v373_data = r1[0];
            r1[0] = (v373_data + (v370_data * v371_data));
            float v376_data = s0[17];
            float v378_data = r1[1];
            r1[1] = (v378_data + (v370_data * v376_data));
            float v381_data = s0[29];
            float v383_data = r1[2];
            r1[2] = (v383_data + (v370_data * v381_data));
            float v386_data = s0[41];
            float v388_data = r1[3];
            r1[3] = (v388_data + (v370_data * v386_data));
            float v391_data = s0[53];
            float v393_data = r1[4];
            r1[4] = (v393_data + (v370_data * v391_data));
            float v396_data = s0[65];
            float v398_data = r1[5];
            r1[5] = (v398_data + (v370_data * v396_data));
            float v401_data = s0[77];
            float v403_data = r1[6];
            r1[6] = (v403_data + (v370_data * v401_data));
            float v406_data = s0[89];
            float v408_data = r1[7];
            r1[7] = (v408_data + (v370_data * v406_data));
            float v411_data = s0[101];
            float v413_data = r1[8];
            r1[8] = (v413_data + (v370_data * v411_data));
            float v416_data = s0[113];
            float v418_data = r1[9];
            r1[9] = (v418_data + (v370_data * v416_data));
            float v421_data = s0[125];
            float v423_data = r1[10];
            r1[10] = (v423_data + (v370_data * v421_data));
            float v426_data = s0[137];
            float v428_data = r1[11];
            r1[11] = (v428_data + (v370_data * v426_data));
          }
          if (v17_lead < 6) {
            float v434_data = r0[6];
            float v435_data = s0[6];
            float v437_data = r1[0];
            r1[0] = (v437_data + (v434_data * v435_data));
            float v440_data = s0[18];
            float v442_data = r1[1];
            r1[1] = (v442_data + (v434_data * v440_data));
            float v445_data = s0[30];
            float v447_data = r1[2];
            r1[2] = (v447_data + (v434_data * v445_data));
            float v450_data = s0[42];
            float v452_data = r1[3];
            r1[3] = (v452_data + (v434_data * v450_data));
            float v455_data = s0[54];
            float v457_data = r1[4];
            r1[4] = (v457_data + (v434_data * v455_data));
            float v460_data = s0[66];
            float v462_data = r1[5];
            r1[5] = (v462_data + (v434_data * v460_data));
            float v465_data = s0[78];
            float v467_data = r1[6];
            r1[6] = (v467_data + (v434_data * v465_data));
            float v470_data = s0[90];
            float v472_data = r1[7];
            r1[7] = (v472_data + (v434_data * v470_data));
            float v475_data = s0[102];
            float v477_data = r1[8];
            r1[8] = (v477_data + (v434_data * v475_data));
            float v480_data = s0[114];
            float v482_data = r1[9];
            r1[9] = (v482_data + (v434_data * v480_data));
            float v485_data = s0[126];
            float v487_data = r1[10];
            r1[10] = (v487_data + (v434_data * v485_data));
            float v490_data = s0[138];
            float v492_data = r1[11];
            r1[11] = (v492_data + (v434_data * v490_data));
          }
          if (v17_lead < 6) {
            float v498_data = r0[7];
            float v499_data = s0[7];
            float v501_data = r1[0];
            r1[0] = (v501_data + (v498_data * v499_data));
            float v504_data = s0[19];
            float v506_data = r1[1];
            r1[1] = (v506_data + (v498_data * v504_data));
            float v509_data = s0[31];
            float v511_data = r1[2];
            r1[2] = (v511_data + (v498_data * v509_data));
            float v514_data = s0[43];
            float v516_data = r1[3];
            r1[3] = (v516_data + (v498_data * v514_data));
            float v519_data = s0[55];
            float v521_data = r1[4];
            r1[4] = (v521_data + (v498_data * v519_data));
            float v524_data = s0[67];
            float v526_data = r1[5];
            r1[5] = (v526_data + (v498_data * v524_data));
            float v529_data = s0[79];
            float v531_data = r1[6];
            r1[6] = (v531_data + (v498_data * v529_data));
            float v534_data = s0[91];
            float v536_data = r1[7];
            r1[7] = (v536_data + (v498_data * v534_data));
            float v539_data = s0[103];
            float v541_data = r1[8];
            r1[8] = (v541_data + (v498_data * v539_data));
            float v544_data = s0[115];
            float v546_data = r1[9];
            r1[9] = (v546_data + (v498_data * v544_data));
            float v549_data = s0[127];
            float v551_data = r1[10];
            r1[10] = (v551_data + (v498_data * v549_data));
            float v554_data = s0[139];
            float v556_data = r1[11];
            r1[11] = (v556_data + (v498_data * v554_data));
          }
          if (v17_lead < 6) {
            float v562_data = r0[8];
            float v563_data = s0[8];
            float v565_data = r1[0];
            r1[0] = (v565_data + (v562_data * v563_data));
            float v568_data = s0[20];
            float v570_data = r1[1];
            r1[1] = (v570_data + (v562_data * v568_data));
            float v573_data = s0[32];
            float v575_data = r1[2];
            r1[2] = (v575_data + (v562_data * v573_data));
            float v578_data = s0[44];
            float v580_data = r1[3];
            r1[3] = (v580_data + (v562_data * v578_data));
            float v583_data = s0[56];
            float v585_data = r1[4];
            r1[4] = (v585_data + (v562_data * v583_data));
            float v588_data = s0[68];
            float v590_data = r1[5];
            r1[5] = (v590_data + (v562_data * v588_data));
            float v593_data = s0[80];
            float v595_data = r1[6];
            r1[6] = (v595_data + (v562_data * v593_data));
            float v598_data = s0[92];
            float v600_data = r1[7];
            r1[7] = (v600_data + (v562_data * v598_data));
            float v603_data = s0[104];
            float v605_data = r1[8];
            r1[8] = (v605_data + (v562_data * v603_data));
            float v608_data = s0[116];
            float v610_data = r1[9];
            r1[9] = (v610_data + (v562_data * v608_data));
            float v613_data = s0[128];
            float v615_data = r1[10];
            r1[10] = (v615_data + (v562_data * v613_data));
            float v618_data = s0[140];
            float v620_data = r1[11];
            r1[11] = (v620_data + (v562_data * v618_data));
          }
          if (v17_lead < 6) {
            float v626_data = r0[9];
            float v627_data = s0[9];
            float v629_data = r1[0];
            r1[0] = (v629_data + (v626_data * v627_data));
            float v632_data = s0[21];
            float v634_data = r1[1];
            r1[1] = (v634_data + (v626_data * v632_data));
            float v637_data = s0[33];
            float v639_data = r1[2];
            r1[2] = (v639_data + (v626_data * v637_data));
            float v642_data = s0[45];
            float v644_data = r1[3];
            r1[3] = (v644_data + (v626_data * v642_data));
            float v647_data = s0[57];
            float v649_data = r1[4];
            r1[4] = (v649_data + (v626_data * v647_data));
            float v652_data = s0[69];
            float v654_data = r1[5];
            r1[5] = (v654_data + (v626_data * v652_data));
            float v657_data = s0[81];
            float v659_data = r1[6];
            r1[6] = (v659_data + (v626_data * v657_data));
            float v662_data = s0[93];
            float v664_data = r1[7];
            r1[7] = (v664_data + (v626_data * v662_data));
            float v667_data = s0[105];
            float v669_data = r1[8];
            r1[8] = (v669_data + (v626_data * v667_data));
            float v672_data = s0[117];
            float v674_data = r1[9];
            r1[9] = (v674_data + (v626_data * v672_data));
            float v677_data = s0[129];
            float v679_data = r1[10];
            r1[10] = (v679_data + (v626_data * v677_data));
            float v682_data = s0[141];
            float v684_data = r1[11];
            r1[11] = (v684_data + (v626_data * v682_data));
          }
          if (v17_lead < 6) {
            float v690_data = r0[10];
            float v691_data = s0[10];
            float v693_data = r1[0];
            r1[0] = (v693_data + (v690_data * v691_data));
            float v696_data = s0[22];
            float v698_data = r1[1];
            r1[1] = (v698_data + (v690_data * v696_data));
            float v701_data = s0[34];
            float v703_data = r1[2];
            r1[2] = (v703_data + (v690_data * v701_data));
            float v706_data = s0[46];
            float v708_data = r1[3];
            r1[3] = (v708_data + (v690_data * v706_data));
            float v711_data = s0[58];
            float v713_data = r1[4];
            r1[4] = (v713_data + (v690_data * v711_data));
            float v716_data = s0[70];
            float v718_data = r1[5];
            r1[5] = (v718_data + (v690_data * v716_data));
            float v721_data = s0[82];
            float v723_data = r1[6];
            r1[6] = (v723_data + (v690_data * v721_data));
            float v726_data = s0[94];
            float v728_data = r1[7];
            r1[7] = (v728_data + (v690_data * v726_data));
            float v731_data = s0[106];
            float v733_data = r1[8];
            r1[8] = (v733_data + (v690_data * v731_data));
            float v736_data = s0[118];
            float v738_data = r1[9];
            r1[9] = (v738_data + (v690_data * v736_data));
            float v741_data = s0[130];
            float v743_data = r1[10];
            r1[10] = (v743_data + (v690_data * v741_data));
            float v746_data = s0[142];
            float v748_data = r1[11];
            r1[11] = (v748_data + (v690_data * v746_data));
          }
          if (v17_lead < 6) {
            float v754_data = r0[11];
            float v755_data = s0[11];
            float v757_data = r1[0];
            r1[0] = (v757_data + (v754_data * v755_data));
            float v760_data = s0[23];
            float v762_data = r1[1];
            r1[1] = (v762_data + (v754_data * v760_data));
            float v765_data = s0[35];
            float v767_data = r1[2];
            r1[2] = (v767_data + (v754_data * v765_data));
            float v770_data = s0[47];
            float v772_data = r1[3];
            r1[3] = (v772_data + (v754_data * v770_data));
            float v775_data = s0[59];
            float v777_data = r1[4];
            r1[4] = (v777_data + (v754_data * v775_data));
            float v780_data = s0[71];
            float v782_data = r1[5];
            r1[5] = (v782_data + (v754_data * v780_data));
            float v785_data = s0[83];
            float v787_data = r1[6];
            r1[6] = (v787_data + (v754_data * v785_data));
            float v790_data = s0[95];
            float v792_data = r1[7];
            r1[7] = (v792_data + (v754_data * v790_data));
            float v795_data = s0[107];
            float v797_data = r1[8];
            r1[8] = (v797_data + (v754_data * v795_data));
            float v800_data = s0[119];
            float v802_data = r1[9];
            r1[9] = (v802_data + (v754_data * v800_data));
            float v805_data = s0[131];
            float v807_data = r1[10];
            r1[10] = (v807_data + (v754_data * v805_data));
            float v810_data = s0[143];
            float v812_data = r1[11];
            r1[11] = (v812_data + (v754_data * v810_data));
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v17_lead < 6) {
            #pragma unroll
            for (int32_t v818_i1 = 0; v818_i1 < 12; ++v818_i1) {
              float v820_data = r1[v818_i1];
              int32_t v827_a = v17_lead + (v818_i1 * 12);
              s1[(v827_a ^ ((v827_a >> 4) & 15))] = v820_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v836_i1 = 0; v836_i1 < 12; ++v836_i1) {
              float v844_data = __ldcg(&glb_m4[(v17_lead + (v836_i1 * 12))]);
              r4[v836_i1] = v844_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v17_lead < 6) {
            float v852_data = r2[0];
            float v853_data = s0[0];
            float v855_data = ir3[0];
            ir3[0] = (v855_data + (v852_data * v853_data));
            float v858_data = s0[12];
            float v860_data = ir3[1];
            ir3[1] = (v860_data + (v852_data * v858_data));
            float v863_data = s0[24];
            float v865_data = ir3[2];
            ir3[2] = (v865_data + (v852_data * v863_data));
            float v868_data = s0[36];
            float v870_data = ir3[3];
            ir3[3] = (v870_data + (v852_data * v868_data));
            float v873_data = s0[48];
            float v875_data = ir3[4];
            ir3[4] = (v875_data + (v852_data * v873_data));
            float v878_data = s0[60];
            float v880_data = ir3[5];
            ir3[5] = (v880_data + (v852_data * v878_data));
            float v883_data = s0[72];
            float v885_data = ir3[6];
            ir3[6] = (v885_data + (v852_data * v883_data));
            float v888_data = s0[84];
            float v890_data = ir3[7];
            ir3[7] = (v890_data + (v852_data * v888_data));
            float v893_data = s0[96];
            float v895_data = ir3[8];
            ir3[8] = (v895_data + (v852_data * v893_data));
            float v898_data = s0[108];
            float v900_data = ir3[9];
            ir3[9] = (v900_data + (v852_data * v898_data));
            float v903_data = s0[120];
            float v905_data = ir3[10];
            ir3[10] = (v905_data + (v852_data * v903_data));
            float v908_data = s0[132];
            float v910_data = ir3[11];
            ir3[11] = (v910_data + (v852_data * v908_data));
          }
          if (v17_lead < 6) {
            float v916_data = r2[1];
            float v917_data = s0[1];
            float v919_data = ir3[0];
            ir3[0] = (v919_data + (v916_data * v917_data));
            float v922_data = s0[13];
            float v924_data = ir3[1];
            ir3[1] = (v924_data + (v916_data * v922_data));
            float v927_data = s0[25];
            float v929_data = ir3[2];
            ir3[2] = (v929_data + (v916_data * v927_data));
            float v932_data = s0[37];
            float v934_data = ir3[3];
            ir3[3] = (v934_data + (v916_data * v932_data));
            float v937_data = s0[49];
            float v939_data = ir3[4];
            ir3[4] = (v939_data + (v916_data * v937_data));
            float v942_data = s0[61];
            float v944_data = ir3[5];
            ir3[5] = (v944_data + (v916_data * v942_data));
            float v947_data = s0[73];
            float v949_data = ir3[6];
            ir3[6] = (v949_data + (v916_data * v947_data));
            float v952_data = s0[85];
            float v954_data = ir3[7];
            ir3[7] = (v954_data + (v916_data * v952_data));
            float v957_data = s0[97];
            float v959_data = ir3[8];
            ir3[8] = (v959_data + (v916_data * v957_data));
            float v962_data = s0[109];
            float v964_data = ir3[9];
            ir3[9] = (v964_data + (v916_data * v962_data));
            float v967_data = s0[121];
            float v969_data = ir3[10];
            ir3[10] = (v969_data + (v916_data * v967_data));
            float v972_data = s0[133];
            float v974_data = ir3[11];
            ir3[11] = (v974_data + (v916_data * v972_data));
          }
          if (v17_lead < 6) {
            float v980_data = r2[2];
            float v981_data = s0[2];
            float v983_data = ir3[0];
            ir3[0] = (v983_data + (v980_data * v981_data));
            float v986_data = s0[14];
            float v988_data = ir3[1];
            ir3[1] = (v988_data + (v980_data * v986_data));
            float v991_data = s0[26];
            float v993_data = ir3[2];
            ir3[2] = (v993_data + (v980_data * v991_data));
            float v996_data = s0[38];
            float v998_data = ir3[3];
            ir3[3] = (v998_data + (v980_data * v996_data));
            float v1001_data = s0[50];
            float v1003_data = ir3[4];
            ir3[4] = (v1003_data + (v980_data * v1001_data));
            float v1006_data = s0[62];
            float v1008_data = ir3[5];
            ir3[5] = (v1008_data + (v980_data * v1006_data));
            float v1011_data = s0[74];
            float v1013_data = ir3[6];
            ir3[6] = (v1013_data + (v980_data * v1011_data));
            float v1016_data = s0[86];
            float v1018_data = ir3[7];
            ir3[7] = (v1018_data + (v980_data * v1016_data));
            float v1021_data = s0[98];
            float v1023_data = ir3[8];
            ir3[8] = (v1023_data + (v980_data * v1021_data));
            float v1026_data = s0[110];
            float v1028_data = ir3[9];
            ir3[9] = (v1028_data + (v980_data * v1026_data));
            float v1031_data = s0[122];
            float v1033_data = ir3[10];
            ir3[10] = (v1033_data + (v980_data * v1031_data));
            float v1036_data = s0[134];
            float v1038_data = ir3[11];
            ir3[11] = (v1038_data + (v980_data * v1036_data));
          }
          if (v17_lead < 6) {
            float v1044_data = r2[3];
            float v1045_data = s0[3];
            float v1047_data = ir3[0];
            ir3[0] = (v1047_data + (v1044_data * v1045_data));
            float v1050_data = s0[15];
            float v1052_data = ir3[1];
            ir3[1] = (v1052_data + (v1044_data * v1050_data));
            float v1055_data = s0[27];
            float v1057_data = ir3[2];
            ir3[2] = (v1057_data + (v1044_data * v1055_data));
            float v1060_data = s0[39];
            float v1062_data = ir3[3];
            ir3[3] = (v1062_data + (v1044_data * v1060_data));
            float v1065_data = s0[51];
            float v1067_data = ir3[4];
            ir3[4] = (v1067_data + (v1044_data * v1065_data));
            float v1070_data = s0[63];
            float v1072_data = ir3[5];
            ir3[5] = (v1072_data + (v1044_data * v1070_data));
            float v1075_data = s0[75];
            float v1077_data = ir3[6];
            ir3[6] = (v1077_data + (v1044_data * v1075_data));
            float v1080_data = s0[87];
            float v1082_data = ir3[7];
            ir3[7] = (v1082_data + (v1044_data * v1080_data));
            float v1085_data = s0[99];
            float v1087_data = ir3[8];
            ir3[8] = (v1087_data + (v1044_data * v1085_data));
            float v1090_data = s0[111];
            float v1092_data = ir3[9];
            ir3[9] = (v1092_data + (v1044_data * v1090_data));
            float v1095_data = s0[123];
            float v1097_data = ir3[10];
            ir3[10] = (v1097_data + (v1044_data * v1095_data));
            float v1100_data = s0[135];
            float v1102_data = ir3[11];
            ir3[11] = (v1102_data + (v1044_data * v1100_data));
          }
          if (v17_lead < 6) {
            float v1108_data = r2[4];
            float v1109_data = s0[4];
            float v1111_data = ir3[0];
            ir3[0] = (v1111_data + (v1108_data * v1109_data));
            float v1114_data = s0[16];
            float v1116_data = ir3[1];
            ir3[1] = (v1116_data + (v1108_data * v1114_data));
            float v1119_data = s0[28];
            float v1121_data = ir3[2];
            ir3[2] = (v1121_data + (v1108_data * v1119_data));
            float v1124_data = s0[40];
            float v1126_data = ir3[3];
            ir3[3] = (v1126_data + (v1108_data * v1124_data));
            float v1129_data = s0[52];
            float v1131_data = ir3[4];
            ir3[4] = (v1131_data + (v1108_data * v1129_data));
            float v1134_data = s0[64];
            float v1136_data = ir3[5];
            ir3[5] = (v1136_data + (v1108_data * v1134_data));
            float v1139_data = s0[76];
            float v1141_data = ir3[6];
            ir3[6] = (v1141_data + (v1108_data * v1139_data));
            float v1144_data = s0[88];
            float v1146_data = ir3[7];
            ir3[7] = (v1146_data + (v1108_data * v1144_data));
            float v1149_data = s0[100];
            float v1151_data = ir3[8];
            ir3[8] = (v1151_data + (v1108_data * v1149_data));
            float v1154_data = s0[112];
            float v1156_data = ir3[9];
            ir3[9] = (v1156_data + (v1108_data * v1154_data));
            float v1159_data = s0[124];
            float v1161_data = ir3[10];
            ir3[10] = (v1161_data + (v1108_data * v1159_data));
            float v1164_data = s0[136];
            float v1166_data = ir3[11];
            ir3[11] = (v1166_data + (v1108_data * v1164_data));
          }
          if (v17_lead < 6) {
            float v1172_data = r2[5];
            float v1173_data = s0[5];
            float v1175_data = ir3[0];
            ir3[0] = (v1175_data + (v1172_data * v1173_data));
            float v1178_data = s0[17];
            float v1180_data = ir3[1];
            ir3[1] = (v1180_data + (v1172_data * v1178_data));
            float v1183_data = s0[29];
            float v1185_data = ir3[2];
            ir3[2] = (v1185_data + (v1172_data * v1183_data));
            float v1188_data = s0[41];
            float v1190_data = ir3[3];
            ir3[3] = (v1190_data + (v1172_data * v1188_data));
            float v1193_data = s0[53];
            float v1195_data = ir3[4];
            ir3[4] = (v1195_data + (v1172_data * v1193_data));
            float v1198_data = s0[65];
            float v1200_data = ir3[5];
            ir3[5] = (v1200_data + (v1172_data * v1198_data));
            float v1203_data = s0[77];
            float v1205_data = ir3[6];
            ir3[6] = (v1205_data + (v1172_data * v1203_data));
            float v1208_data = s0[89];
            float v1210_data = ir3[7];
            ir3[7] = (v1210_data + (v1172_data * v1208_data));
            float v1213_data = s0[101];
            float v1215_data = ir3[8];
            ir3[8] = (v1215_data + (v1172_data * v1213_data));
            float v1218_data = s0[113];
            float v1220_data = ir3[9];
            ir3[9] = (v1220_data + (v1172_data * v1218_data));
            float v1223_data = s0[125];
            float v1225_data = ir3[10];
            ir3[10] = (v1225_data + (v1172_data * v1223_data));
            float v1228_data = s0[137];
            float v1230_data = ir3[11];
            ir3[11] = (v1230_data + (v1172_data * v1228_data));
          }
          if (v17_lead < 6) {
            float v1236_data = r2[6];
            float v1237_data = s0[6];
            float v1239_data = ir3[0];
            ir3[0] = (v1239_data + (v1236_data * v1237_data));
            float v1242_data = s0[18];
            float v1244_data = ir3[1];
            ir3[1] = (v1244_data + (v1236_data * v1242_data));
            float v1247_data = s0[30];
            float v1249_data = ir3[2];
            ir3[2] = (v1249_data + (v1236_data * v1247_data));
            float v1252_data = s0[42];
            float v1254_data = ir3[3];
            ir3[3] = (v1254_data + (v1236_data * v1252_data));
            float v1257_data = s0[54];
            float v1259_data = ir3[4];
            ir3[4] = (v1259_data + (v1236_data * v1257_data));
            float v1262_data = s0[66];
            float v1264_data = ir3[5];
            ir3[5] = (v1264_data + (v1236_data * v1262_data));
            float v1267_data = s0[78];
            float v1269_data = ir3[6];
            ir3[6] = (v1269_data + (v1236_data * v1267_data));
            float v1272_data = s0[90];
            float v1274_data = ir3[7];
            ir3[7] = (v1274_data + (v1236_data * v1272_data));
            float v1277_data = s0[102];
            float v1279_data = ir3[8];
            ir3[8] = (v1279_data + (v1236_data * v1277_data));
            float v1282_data = s0[114];
            float v1284_data = ir3[9];
            ir3[9] = (v1284_data + (v1236_data * v1282_data));
            float v1287_data = s0[126];
            float v1289_data = ir3[10];
            ir3[10] = (v1289_data + (v1236_data * v1287_data));
            float v1292_data = s0[138];
            float v1294_data = ir3[11];
            ir3[11] = (v1294_data + (v1236_data * v1292_data));
          }
          if (v17_lead < 6) {
            float v1300_data = r2[7];
            float v1301_data = s0[7];
            float v1303_data = ir3[0];
            ir3[0] = (v1303_data + (v1300_data * v1301_data));
            float v1306_data = s0[19];
            float v1308_data = ir3[1];
            ir3[1] = (v1308_data + (v1300_data * v1306_data));
            float v1311_data = s0[31];
            float v1313_data = ir3[2];
            ir3[2] = (v1313_data + (v1300_data * v1311_data));
            float v1316_data = s0[43];
            float v1318_data = ir3[3];
            ir3[3] = (v1318_data + (v1300_data * v1316_data));
            float v1321_data = s0[55];
            float v1323_data = ir3[4];
            ir3[4] = (v1323_data + (v1300_data * v1321_data));
            float v1326_data = s0[67];
            float v1328_data = ir3[5];
            ir3[5] = (v1328_data + (v1300_data * v1326_data));
            float v1331_data = s0[79];
            float v1333_data = ir3[6];
            ir3[6] = (v1333_data + (v1300_data * v1331_data));
            float v1336_data = s0[91];
            float v1338_data = ir3[7];
            ir3[7] = (v1338_data + (v1300_data * v1336_data));
            float v1341_data = s0[103];
            float v1343_data = ir3[8];
            ir3[8] = (v1343_data + (v1300_data * v1341_data));
            float v1346_data = s0[115];
            float v1348_data = ir3[9];
            ir3[9] = (v1348_data + (v1300_data * v1346_data));
            float v1351_data = s0[127];
            float v1353_data = ir3[10];
            ir3[10] = (v1353_data + (v1300_data * v1351_data));
            float v1356_data = s0[139];
            float v1358_data = ir3[11];
            ir3[11] = (v1358_data + (v1300_data * v1356_data));
          }
          if (v17_lead < 6) {
            float v1364_data = r2[8];
            float v1365_data = s0[8];
            float v1367_data = ir3[0];
            ir3[0] = (v1367_data + (v1364_data * v1365_data));
            float v1370_data = s0[20];
            float v1372_data = ir3[1];
            ir3[1] = (v1372_data + (v1364_data * v1370_data));
            float v1375_data = s0[32];
            float v1377_data = ir3[2];
            ir3[2] = (v1377_data + (v1364_data * v1375_data));
            float v1380_data = s0[44];
            float v1382_data = ir3[3];
            ir3[3] = (v1382_data + (v1364_data * v1380_data));
            float v1385_data = s0[56];
            float v1387_data = ir3[4];
            ir3[4] = (v1387_data + (v1364_data * v1385_data));
            float v1390_data = s0[68];
            float v1392_data = ir3[5];
            ir3[5] = (v1392_data + (v1364_data * v1390_data));
            float v1395_data = s0[80];
            float v1397_data = ir3[6];
            ir3[6] = (v1397_data + (v1364_data * v1395_data));
            float v1400_data = s0[92];
            float v1402_data = ir3[7];
            ir3[7] = (v1402_data + (v1364_data * v1400_data));
            float v1405_data = s0[104];
            float v1407_data = ir3[8];
            ir3[8] = (v1407_data + (v1364_data * v1405_data));
            float v1410_data = s0[116];
            float v1412_data = ir3[9];
            ir3[9] = (v1412_data + (v1364_data * v1410_data));
            float v1415_data = s0[128];
            float v1417_data = ir3[10];
            ir3[10] = (v1417_data + (v1364_data * v1415_data));
            float v1420_data = s0[140];
            float v1422_data = ir3[11];
            ir3[11] = (v1422_data + (v1364_data * v1420_data));
          }
          if (v17_lead < 6) {
            float v1428_data = r2[9];
            float v1429_data = s0[9];
            float v1431_data = ir3[0];
            ir3[0] = (v1431_data + (v1428_data * v1429_data));
            float v1434_data = s0[21];
            float v1436_data = ir3[1];
            ir3[1] = (v1436_data + (v1428_data * v1434_data));
            float v1439_data = s0[33];
            float v1441_data = ir3[2];
            ir3[2] = (v1441_data + (v1428_data * v1439_data));
            float v1444_data = s0[45];
            float v1446_data = ir3[3];
            ir3[3] = (v1446_data + (v1428_data * v1444_data));
            float v1449_data = s0[57];
            float v1451_data = ir3[4];
            ir3[4] = (v1451_data + (v1428_data * v1449_data));
            float v1454_data = s0[69];
            float v1456_data = ir3[5];
            ir3[5] = (v1456_data + (v1428_data * v1454_data));
            float v1459_data = s0[81];
            float v1461_data = ir3[6];
            ir3[6] = (v1461_data + (v1428_data * v1459_data));
            float v1464_data = s0[93];
            float v1466_data = ir3[7];
            ir3[7] = (v1466_data + (v1428_data * v1464_data));
            float v1469_data = s0[105];
            float v1471_data = ir3[8];
            ir3[8] = (v1471_data + (v1428_data * v1469_data));
            float v1474_data = s0[117];
            float v1476_data = ir3[9];
            ir3[9] = (v1476_data + (v1428_data * v1474_data));
            float v1479_data = s0[129];
            float v1481_data = ir3[10];
            ir3[10] = (v1481_data + (v1428_data * v1479_data));
            float v1484_data = s0[141];
            float v1486_data = ir3[11];
            ir3[11] = (v1486_data + (v1428_data * v1484_data));
          }
          if (v17_lead < 6) {
            float v1492_data = r2[10];
            float v1493_data = s0[10];
            float v1495_data = ir3[0];
            ir3[0] = (v1495_data + (v1492_data * v1493_data));
            float v1498_data = s0[22];
            float v1500_data = ir3[1];
            ir3[1] = (v1500_data + (v1492_data * v1498_data));
            float v1503_data = s0[34];
            float v1505_data = ir3[2];
            ir3[2] = (v1505_data + (v1492_data * v1503_data));
            float v1508_data = s0[46];
            float v1510_data = ir3[3];
            ir3[3] = (v1510_data + (v1492_data * v1508_data));
            float v1513_data = s0[58];
            float v1515_data = ir3[4];
            ir3[4] = (v1515_data + (v1492_data * v1513_data));
            float v1518_data = s0[70];
            float v1520_data = ir3[5];
            ir3[5] = (v1520_data + (v1492_data * v1518_data));
            float v1523_data = s0[82];
            float v1525_data = ir3[6];
            ir3[6] = (v1525_data + (v1492_data * v1523_data));
            float v1528_data = s0[94];
            float v1530_data = ir3[7];
            ir3[7] = (v1530_data + (v1492_data * v1528_data));
            float v1533_data = s0[106];
            float v1535_data = ir3[8];
            ir3[8] = (v1535_data + (v1492_data * v1533_data));
            float v1538_data = s0[118];
            float v1540_data = ir3[9];
            ir3[9] = (v1540_data + (v1492_data * v1538_data));
            float v1543_data = s0[130];
            float v1545_data = ir3[10];
            ir3[10] = (v1545_data + (v1492_data * v1543_data));
            float v1548_data = s0[142];
            float v1550_data = ir3[11];
            ir3[11] = (v1550_data + (v1492_data * v1548_data));
          }
          if (v17_lead < 6) {
            float v1556_data = r2[11];
            float v1557_data = s0[11];
            float v1559_data = ir3[0];
            ir3[0] = (v1559_data + (v1556_data * v1557_data));
            float v1562_data = s0[23];
            float v1564_data = ir3[1];
            ir3[1] = (v1564_data + (v1556_data * v1562_data));
            float v1567_data = s0[35];
            float v1569_data = ir3[2];
            ir3[2] = (v1569_data + (v1556_data * v1567_data));
            float v1572_data = s0[47];
            float v1574_data = ir3[3];
            ir3[3] = (v1574_data + (v1556_data * v1572_data));
            float v1577_data = s0[59];
            float v1579_data = ir3[4];
            ir3[4] = (v1579_data + (v1556_data * v1577_data));
            float v1582_data = s0[71];
            float v1584_data = ir3[5];
            ir3[5] = (v1584_data + (v1556_data * v1582_data));
            float v1587_data = s0[83];
            float v1589_data = ir3[6];
            ir3[6] = (v1589_data + (v1556_data * v1587_data));
            float v1592_data = s0[95];
            float v1594_data = ir3[7];
            ir3[7] = (v1594_data + (v1556_data * v1592_data));
            float v1597_data = s0[107];
            float v1599_data = ir3[8];
            ir3[8] = (v1599_data + (v1556_data * v1597_data));
            float v1602_data = s0[119];
            float v1604_data = ir3[9];
            ir3[9] = (v1604_data + (v1556_data * v1602_data));
            float v1607_data = s0[131];
            float v1609_data = ir3[10];
            ir3[10] = (v1609_data + (v1556_data * v1607_data));
            float v1612_data = s0[143];
            float v1614_data = ir3[11];
            ir3[11] = (v1614_data + (v1556_data * v1612_data));
          }
          if (v17_lead < 6) {
            #pragma unroll
            for (int32_t v1620_n1 = 0; v1620_n1 < 12; ++v1620_n1) {
              float v1622_data = ir3[v1620_n1];
              r3[v1620_n1] = v1622_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v17_lead < 6) {
            int32_t v1636_off = v17_lead + 6;
            #pragma unroll
            for (int32_t v1628_i1 = 0; v1628_i1 < 12; ++v1628_i1) {
              float v1630_data = r3[v1628_i1];
              int32_t v1638_a = v1636_off + (v1628_i1 * 12);
              s1[(v1638_a ^ ((v1638_a >> 4) & 15))] = v1630_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v17_lead < 12) {
            float v1648_data = r4[0];
            float v1649_data = s1[0];
            float v1651_data = ir5[0];
            ir5[0] = (v1651_data + (v1648_data * v1649_data));
            float v1654_data = s1[12];
            float v1656_data = ir5[1];
            ir5[1] = (v1656_data + (v1648_data * v1654_data));
            float v1659_data = s1[25];
            float v1661_data = ir5[2];
            ir5[2] = (v1661_data + (v1648_data * v1659_data));
            float v1664_data = s1[38];
            float v1666_data = ir5[3];
            ir5[3] = (v1666_data + (v1648_data * v1664_data));
            float v1669_data = s1[51];
            float v1671_data = ir5[4];
            ir5[4] = (v1671_data + (v1648_data * v1669_data));
            float v1674_data = s1[63];
            float v1676_data = ir5[5];
            ir5[5] = (v1676_data + (v1648_data * v1674_data));
            float v1679_data = s1[76];
            float v1681_data = ir5[6];
            ir5[6] = (v1681_data + (v1648_data * v1679_data));
            float v1684_data = s1[81];
            float v1686_data = ir5[7];
            ir5[7] = (v1686_data + (v1648_data * v1684_data));
            float v1689_data = s1[102];
            float v1691_data = ir5[8];
            ir5[8] = (v1691_data + (v1648_data * v1689_data));
            float v1694_data = s1[106];
            float v1696_data = ir5[9];
            ir5[9] = (v1696_data + (v1648_data * v1694_data));
            float v1699_data = s1[127];
            float v1701_data = ir5[10];
            ir5[10] = (v1701_data + (v1648_data * v1699_data));
            float v1704_data = s1[140];
            float v1706_data = ir5[11];
            ir5[11] = (v1706_data + (v1648_data * v1704_data));
          }
          if (v17_lead < 12) {
            float v1712_data = r4[1];
            float v1713_data = s1[1];
            float v1715_data = ir5[0];
            ir5[0] = (v1715_data + (v1712_data * v1713_data));
            float v1718_data = s1[13];
            float v1720_data = ir5[1];
            ir5[1] = (v1720_data + (v1712_data * v1718_data));
            float v1723_data = s1[24];
            float v1725_data = ir5[2];
            ir5[2] = (v1725_data + (v1712_data * v1723_data));
            float v1728_data = s1[39];
            float v1730_data = ir5[3];
            ir5[3] = (v1730_data + (v1712_data * v1728_data));
            float v1733_data = s1[50];
            float v1735_data = ir5[4];
            ir5[4] = (v1735_data + (v1712_data * v1733_data));
            float v1738_data = s1[62];
            float v1740_data = ir5[5];
            ir5[5] = (v1740_data + (v1712_data * v1738_data));
            float v1743_data = s1[77];
            float v1745_data = ir5[6];
            ir5[6] = (v1745_data + (v1712_data * v1743_data));
            float v1748_data = s1[80];
            float v1750_data = ir5[7];
            ir5[7] = (v1750_data + (v1712_data * v1748_data));
            float v1753_data = s1[103];
            float v1755_data = ir5[8];
            ir5[8] = (v1755_data + (v1712_data * v1753_data));
            float v1758_data = s1[107];
            float v1760_data = ir5[9];
            ir5[9] = (v1760_data + (v1712_data * v1758_data));
            float v1763_data = s1[126];
            float v1765_data = ir5[10];
            ir5[10] = (v1765_data + (v1712_data * v1763_data));
            float v1768_data = s1[141];
            float v1770_data = ir5[11];
            ir5[11] = (v1770_data + (v1712_data * v1768_data));
          }
          if (v17_lead < 12) {
            float v1776_data = r4[2];
            float v1777_data = s1[2];
            float v1779_data = ir5[0];
            ir5[0] = (v1779_data + (v1776_data * v1777_data));
            float v1782_data = s1[14];
            float v1784_data = ir5[1];
            ir5[1] = (v1784_data + (v1776_data * v1782_data));
            float v1787_data = s1[27];
            float v1789_data = ir5[2];
            ir5[2] = (v1789_data + (v1776_data * v1787_data));
            float v1792_data = s1[36];
            float v1794_data = ir5[3];
            ir5[3] = (v1794_data + (v1776_data * v1792_data));
            float v1797_data = s1[49];
            float v1799_data = ir5[4];
            ir5[4] = (v1799_data + (v1776_data * v1797_data));
            float v1802_data = s1[61];
            float v1804_data = ir5[5];
            ir5[5] = (v1804_data + (v1776_data * v1802_data));
            float v1807_data = s1[78];
            float v1809_data = ir5[6];
            ir5[6] = (v1809_data + (v1776_data * v1807_data));
            float v1812_data = s1[83];
            float v1814_data = ir5[7];
            ir5[7] = (v1814_data + (v1776_data * v1812_data));
            float v1817_data = s1[100];
            float v1819_data = ir5[8];
            ir5[8] = (v1819_data + (v1776_data * v1817_data));
            float v1822_data = s1[104];
            float v1824_data = ir5[9];
            ir5[9] = (v1824_data + (v1776_data * v1822_data));
            float v1827_data = s1[125];
            float v1829_data = ir5[10];
            ir5[10] = (v1829_data + (v1776_data * v1827_data));
            float v1832_data = s1[142];
            float v1834_data = ir5[11];
            ir5[11] = (v1834_data + (v1776_data * v1832_data));
          }
          if (v17_lead < 12) {
            float v1840_data = r4[3];
            float v1841_data = s1[3];
            float v1843_data = ir5[0];
            ir5[0] = (v1843_data + (v1840_data * v1841_data));
            float v1846_data = s1[15];
            float v1848_data = ir5[1];
            ir5[1] = (v1848_data + (v1840_data * v1846_data));
            float v1851_data = s1[26];
            float v1853_data = ir5[2];
            ir5[2] = (v1853_data + (v1840_data * v1851_data));
            float v1856_data = s1[37];
            float v1858_data = ir5[3];
            ir5[3] = (v1858_data + (v1840_data * v1856_data));
            float v1861_data = s1[48];
            float v1863_data = ir5[4];
            ir5[4] = (v1863_data + (v1840_data * v1861_data));
            float v1866_data = s1[60];
            float v1868_data = ir5[5];
            ir5[5] = (v1868_data + (v1840_data * v1866_data));
            float v1871_data = s1[79];
            float v1873_data = ir5[6];
            ir5[6] = (v1873_data + (v1840_data * v1871_data));
            float v1876_data = s1[82];
            float v1878_data = ir5[7];
            ir5[7] = (v1878_data + (v1840_data * v1876_data));
            float v1881_data = s1[101];
            float v1883_data = ir5[8];
            ir5[8] = (v1883_data + (v1840_data * v1881_data));
            float v1886_data = s1[105];
            float v1888_data = ir5[9];
            ir5[9] = (v1888_data + (v1840_data * v1886_data));
            float v1891_data = s1[124];
            float v1893_data = ir5[10];
            ir5[10] = (v1893_data + (v1840_data * v1891_data));
            float v1896_data = s1[143];
            float v1898_data = ir5[11];
            ir5[11] = (v1898_data + (v1840_data * v1896_data));
          }
          if (v17_lead < 12) {
            float v1904_data = r4[4];
            float v1905_data = s1[4];
            float v1907_data = ir5[0];
            ir5[0] = (v1907_data + (v1904_data * v1905_data));
            float v1910_data = s1[17];
            float v1912_data = ir5[1];
            ir5[1] = (v1912_data + (v1904_data * v1910_data));
            float v1915_data = s1[29];
            float v1917_data = ir5[2];
            ir5[2] = (v1917_data + (v1904_data * v1915_data));
            float v1920_data = s1[42];
            float v1922_data = ir5[3];
            ir5[3] = (v1922_data + (v1904_data * v1920_data));
            float v1925_data = s1[55];
            float v1927_data = ir5[4];
            ir5[4] = (v1927_data + (v1904_data * v1925_data));
            float v1930_data = s1[68];
            float v1932_data = ir5[5];
            ir5[5] = (v1932_data + (v1904_data * v1930_data));
            float v1935_data = s1[72];
            float v1937_data = ir5[6];
            ir5[6] = (v1937_data + (v1904_data * v1935_data));
            float v1940_data = s1[93];
            float v1942_data = ir5[7];
            ir5[7] = (v1942_data + (v1904_data * v1940_data));
            float v1945_data = s1[98];
            float v1947_data = ir5[8];
            ir5[8] = (v1947_data + (v1904_data * v1945_data));
            float v1950_data = s1[119];
            float v1952_data = ir5[9];
            ir5[9] = (v1952_data + (v1904_data * v1950_data));
            float v1955_data = s1[123];
            float v1957_data = ir5[10];
            ir5[10] = (v1957_data + (v1904_data * v1955_data));
            float v1960_data = s1[128];
            float v1962_data = ir5[11];
            ir5[11] = (v1962_data + (v1904_data * v1960_data));
          }
          if (v17_lead < 12) {
            float v1968_data = r4[5];
            float v1969_data = s1[5];
            float v1971_data = ir5[0];
            ir5[0] = (v1971_data + (v1968_data * v1969_data));
            float v1974_data = s1[16];
            float v1976_data = ir5[1];
            ir5[1] = (v1976_data + (v1968_data * v1974_data));
            float v1979_data = s1[28];
            float v1981_data = ir5[2];
            ir5[2] = (v1981_data + (v1968_data * v1979_data));
            float v1984_data = s1[43];
            float v1986_data = ir5[3];
            ir5[3] = (v1986_data + (v1968_data * v1984_data));
            float v1989_data = s1[54];
            float v1991_data = ir5[4];
            ir5[4] = (v1991_data + (v1968_data * v1989_data));
            float v1994_data = s1[69];
            float v1996_data = ir5[5];
            ir5[5] = (v1996_data + (v1968_data * v1994_data));
            float v1999_data = s1[73];
            float v2001_data = ir5[6];
            ir5[6] = (v2001_data + (v1968_data * v1999_data));
            float v2004_data = s1[92];
            float v2006_data = ir5[7];
            ir5[7] = (v2006_data + (v1968_data * v2004_data));
            float v2009_data = s1[99];
            float v2011_data = ir5[8];
            ir5[8] = (v2011_data + (v1968_data * v2009_data));
            float v2014_data = s1[118];
            float v2016_data = ir5[9];
            ir5[9] = (v2016_data + (v1968_data * v2014_data));
            float v2019_data = s1[122];
            float v2021_data = ir5[10];
            ir5[10] = (v2021_data + (v1968_data * v2019_data));
            float v2024_data = s1[129];
            float v2026_data = ir5[11];
            ir5[11] = (v2026_data + (v1968_data * v2024_data));
          }
          if (v17_lead < 12) {
            float v2032_data = r4[6];
            float v2033_data = s1[6];
            float v2035_data = ir5[0];
            ir5[0] = (v2035_data + (v2032_data * v2033_data));
            float v2038_data = s1[19];
            float v2040_data = ir5[1];
            ir5[1] = (v2040_data + (v2032_data * v2038_data));
            float v2043_data = s1[31];
            float v2045_data = ir5[2];
            ir5[2] = (v2045_data + (v2032_data * v2043_data));
            float v2048_data = s1[40];
            float v2050_data = ir5[3];
            ir5[3] = (v2050_data + (v2032_data * v2048_data));
            float v2053_data = s1[53];
            float v2055_data = ir5[4];
            ir5[4] = (v2055_data + (v2032_data * v2053_data));
            float v2058_data = s1[70];
            float v2060_data = ir5[5];
            ir5[5] = (v2060_data + (v2032_data * v2058_data));
            float v2063_data = s1[74];
            float v2065_data = ir5[6];
            ir5[6] = (v2065_data + (v2032_data * v2063_data));
            float v2068_data = s1[95];
            float v2070_data = ir5[7];
            ir5[7] = (v2070_data + (v2032_data * v2068_data));
            float v2073_data = s1[96];
            float v2075_data = ir5[8];
            ir5[8] = (v2075_data + (v2032_data * v2073_data));
            float v2078_data = s1[117];
            float v2080_data = ir5[9];
            ir5[9] = (v2080_data + (v2032_data * v2078_data));
            float v2083_data = s1[121];
            float v2085_data = ir5[10];
            ir5[10] = (v2085_data + (v2032_data * v2083_data));
            float v2088_data = s1[130];
            float v2090_data = ir5[11];
            ir5[11] = (v2090_data + (v2032_data * v2088_data));
          }
          if (v17_lead < 12) {
            float v2096_data = r4[7];
            float v2097_data = s1[7];
            float v2099_data = ir5[0];
            ir5[0] = (v2099_data + (v2096_data * v2097_data));
            float v2102_data = s1[18];
            float v2104_data = ir5[1];
            ir5[1] = (v2104_data + (v2096_data * v2102_data));
            float v2107_data = s1[30];
            float v2109_data = ir5[2];
            ir5[2] = (v2109_data + (v2096_data * v2107_data));
            float v2112_data = s1[41];
            float v2114_data = ir5[3];
            ir5[3] = (v2114_data + (v2096_data * v2112_data));
            float v2117_data = s1[52];
            float v2119_data = ir5[4];
            ir5[4] = (v2119_data + (v2096_data * v2117_data));
            float v2122_data = s1[71];
            float v2124_data = ir5[5];
            ir5[5] = (v2124_data + (v2096_data * v2122_data));
            float v2127_data = s1[75];
            float v2129_data = ir5[6];
            ir5[6] = (v2129_data + (v2096_data * v2127_data));
            float v2132_data = s1[94];
            float v2134_data = ir5[7];
            ir5[7] = (v2134_data + (v2096_data * v2132_data));
            float v2137_data = s1[97];
            float v2139_data = ir5[8];
            ir5[8] = (v2139_data + (v2096_data * v2137_data));
            float v2142_data = s1[116];
            float v2144_data = ir5[9];
            ir5[9] = (v2144_data + (v2096_data * v2142_data));
            float v2147_data = s1[120];
            float v2149_data = ir5[10];
            ir5[10] = (v2149_data + (v2096_data * v2147_data));
            float v2152_data = s1[131];
            float v2154_data = ir5[11];
            ir5[11] = (v2154_data + (v2096_data * v2152_data));
          }
          if (v17_lead < 12) {
            float v2160_data = r4[8];
            float v2161_data = s1[8];
            float v2163_data = ir5[0];
            ir5[0] = (v2163_data + (v2160_data * v2161_data));
            float v2166_data = s1[21];
            float v2168_data = ir5[1];
            ir5[1] = (v2168_data + (v2160_data * v2166_data));
            float v2171_data = s1[34];
            float v2173_data = ir5[2];
            ir5[2] = (v2173_data + (v2160_data * v2171_data));
            float v2176_data = s1[46];
            float v2178_data = ir5[3];
            ir5[3] = (v2178_data + (v2160_data * v2176_data));
            float v2181_data = s1[59];
            float v2183_data = ir5[4];
            ir5[4] = (v2183_data + (v2160_data * v2181_data));
            float v2186_data = s1[64];
            float v2188_data = ir5[5];
            ir5[5] = (v2188_data + (v2160_data * v2186_data));
            float v2191_data = s1[85];
            float v2193_data = ir5[6];
            ir5[6] = (v2193_data + (v2160_data * v2191_data));
            float v2196_data = s1[89];
            float v2198_data = ir5[7];
            ir5[7] = (v2198_data + (v2160_data * v2196_data));
            float v2201_data = s1[110];
            float v2203_data = ir5[8];
            ir5[8] = (v2203_data + (v2160_data * v2201_data));
            float v2206_data = s1[115];
            float v2208_data = ir5[9];
            ir5[9] = (v2208_data + (v2160_data * v2206_data));
            float v2211_data = s1[136];
            float v2213_data = ir5[10];
            ir5[10] = (v2213_data + (v2160_data * v2211_data));
            float v2216_data = s1[132];
            float v2218_data = ir5[11];
            ir5[11] = (v2218_data + (v2160_data * v2216_data));
          }
          if (v17_lead < 12) {
            float v2224_data = r4[9];
            float v2225_data = s1[9];
            float v2227_data = ir5[0];
            ir5[0] = (v2227_data + (v2224_data * v2225_data));
            float v2230_data = s1[20];
            float v2232_data = ir5[1];
            ir5[1] = (v2232_data + (v2224_data * v2230_data));
            float v2235_data = s1[35];
            float v2237_data = ir5[2];
            ir5[2] = (v2237_data + (v2224_data * v2235_data));
            float v2240_data = s1[47];
            float v2242_data = ir5[3];
            ir5[3] = (v2242_data + (v2224_data * v2240_data));
            float v2245_data = s1[58];
            float v2247_data = ir5[4];
            ir5[4] = (v2247_data + (v2224_data * v2245_data));
            float v2250_data = s1[65];
            float v2252_data = ir5[5];
            ir5[5] = (v2252_data + (v2224_data * v2250_data));
            float v2255_data = s1[84];
            float v2257_data = ir5[6];
            ir5[6] = (v2257_data + (v2224_data * v2255_data));
            float v2260_data = s1[88];
            float v2262_data = ir5[7];
            ir5[7] = (v2262_data + (v2224_data * v2260_data));
            float v2265_data = s1[111];
            float v2267_data = ir5[8];
            ir5[8] = (v2267_data + (v2224_data * v2265_data));
            float v2270_data = s1[114];
            float v2272_data = ir5[9];
            ir5[9] = (v2272_data + (v2224_data * v2270_data));
            float v2275_data = s1[137];
            float v2277_data = ir5[10];
            ir5[10] = (v2277_data + (v2224_data * v2275_data));
            float v2280_data = s1[133];
            float v2282_data = ir5[11];
            ir5[11] = (v2282_data + (v2224_data * v2280_data));
          }
          if (v17_lead < 12) {
            float v2288_data = r4[10];
            float v2289_data = s1[10];
            float v2291_data = ir5[0];
            ir5[0] = (v2291_data + (v2288_data * v2289_data));
            float v2294_data = s1[23];
            float v2296_data = ir5[1];
            ir5[1] = (v2296_data + (v2288_data * v2294_data));
            float v2299_data = s1[32];
            float v2301_data = ir5[2];
            ir5[2] = (v2301_data + (v2288_data * v2299_data));
            float v2304_data = s1[44];
            float v2306_data = ir5[3];
            ir5[3] = (v2306_data + (v2288_data * v2304_data));
            float v2309_data = s1[57];
            float v2311_data = ir5[4];
            ir5[4] = (v2311_data + (v2288_data * v2309_data));
            float v2314_data = s1[66];
            float v2316_data = ir5[5];
            ir5[5] = (v2316_data + (v2288_data * v2314_data));
            float v2319_data = s1[87];
            float v2321_data = ir5[6];
            ir5[6] = (v2321_data + (v2288_data * v2319_data));
            float v2324_data = s1[91];
            float v2326_data = ir5[7];
            ir5[7] = (v2326_data + (v2288_data * v2324_data));
            float v2329_data = s1[108];
            float v2331_data = ir5[8];
            ir5[8] = (v2331_data + (v2288_data * v2329_data));
            float v2334_data = s1[113];
            float v2336_data = ir5[9];
            ir5[9] = (v2336_data + (v2288_data * v2334_data));
            float v2339_data = s1[138];
            float v2341_data = ir5[10];
            ir5[10] = (v2341_data + (v2288_data * v2339_data));
            float v2344_data = s1[134];
            float v2346_data = ir5[11];
            ir5[11] = (v2346_data + (v2288_data * v2344_data));
          }
          if (v17_lead < 12) {
            float v2352_data = r4[11];
            float v2353_data = s1[11];
            float v2355_data = ir5[0];
            ir5[0] = (v2355_data + (v2352_data * v2353_data));
            float v2358_data = s1[22];
            float v2360_data = ir5[1];
            ir5[1] = (v2360_data + (v2352_data * v2358_data));
            float v2363_data = s1[33];
            float v2365_data = ir5[2];
            ir5[2] = (v2365_data + (v2352_data * v2363_data));
            float v2368_data = s1[45];
            float v2370_data = ir5[3];
            ir5[3] = (v2370_data + (v2352_data * v2368_data));
            float v2373_data = s1[56];
            float v2375_data = ir5[4];
            ir5[4] = (v2375_data + (v2352_data * v2373_data));
            float v2378_data = s1[67];
            float v2380_data = ir5[5];
            ir5[5] = (v2380_data + (v2352_data * v2378_data));
            float v2383_data = s1[86];
            float v2385_data = ir5[6];
            ir5[6] = (v2385_data + (v2352_data * v2383_data));
            float v2388_data = s1[90];
            float v2390_data = ir5[7];
            ir5[7] = (v2390_data + (v2352_data * v2388_data));
            float v2393_data = s1[109];
            float v2395_data = ir5[8];
            ir5[8] = (v2395_data + (v2352_data * v2393_data));
            float v2398_data = s1[112];
            float v2400_data = ir5[9];
            ir5[9] = (v2400_data + (v2352_data * v2398_data));
            float v2403_data = s1[139];
            float v2405_data = ir5[10];
            ir5[10] = (v2405_data + (v2352_data * v2403_data));
            float v2408_data = s1[135];
            float v2410_data = ir5[11];
            ir5[11] = (v2410_data + (v2352_data * v2408_data));
          }
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v2416_n1 = 0; v2416_n1 < 12; ++v2416_n1) {
              float v2418_data = ir5[v2416_n1];
              r5[v2416_n1] = v2418_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v2424_i1 = 0; v2424_i1 < 12; ++v2424_i1) {
              float v2426_data = r5[v2424_i1];
              glb_m3[(v17_lead + (v2424_i1 * 12))] = v2426_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

