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
            __pipeline_commit();
          }
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
              s1[(v17_lead + (v818_i1 * 12))] = v820_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v833_i1 = 0; v833_i1 < 12; ++v833_i1) {
              float v841_data = __ldcg(&glb_m4[(v17_lead + (v833_i1 * 12))]);
              r4[v833_i1] = v841_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v17_lead < 6) {
            float v849_data = r2[0];
            float v850_data = s0[0];
            float v852_data = ir3[0];
            ir3[0] = (v852_data + (v849_data * v850_data));
            float v855_data = s0[12];
            float v857_data = ir3[1];
            ir3[1] = (v857_data + (v849_data * v855_data));
            float v860_data = s0[24];
            float v862_data = ir3[2];
            ir3[2] = (v862_data + (v849_data * v860_data));
            float v865_data = s0[36];
            float v867_data = ir3[3];
            ir3[3] = (v867_data + (v849_data * v865_data));
            float v870_data = s0[48];
            float v872_data = ir3[4];
            ir3[4] = (v872_data + (v849_data * v870_data));
            float v875_data = s0[60];
            float v877_data = ir3[5];
            ir3[5] = (v877_data + (v849_data * v875_data));
            float v880_data = s0[72];
            float v882_data = ir3[6];
            ir3[6] = (v882_data + (v849_data * v880_data));
            float v885_data = s0[84];
            float v887_data = ir3[7];
            ir3[7] = (v887_data + (v849_data * v885_data));
            float v890_data = s0[96];
            float v892_data = ir3[8];
            ir3[8] = (v892_data + (v849_data * v890_data));
            float v895_data = s0[108];
            float v897_data = ir3[9];
            ir3[9] = (v897_data + (v849_data * v895_data));
            float v900_data = s0[120];
            float v902_data = ir3[10];
            ir3[10] = (v902_data + (v849_data * v900_data));
            float v905_data = s0[132];
            float v907_data = ir3[11];
            ir3[11] = (v907_data + (v849_data * v905_data));
          }
          if (v17_lead < 6) {
            float v913_data = r2[1];
            float v914_data = s0[1];
            float v916_data = ir3[0];
            ir3[0] = (v916_data + (v913_data * v914_data));
            float v919_data = s0[13];
            float v921_data = ir3[1];
            ir3[1] = (v921_data + (v913_data * v919_data));
            float v924_data = s0[25];
            float v926_data = ir3[2];
            ir3[2] = (v926_data + (v913_data * v924_data));
            float v929_data = s0[37];
            float v931_data = ir3[3];
            ir3[3] = (v931_data + (v913_data * v929_data));
            float v934_data = s0[49];
            float v936_data = ir3[4];
            ir3[4] = (v936_data + (v913_data * v934_data));
            float v939_data = s0[61];
            float v941_data = ir3[5];
            ir3[5] = (v941_data + (v913_data * v939_data));
            float v944_data = s0[73];
            float v946_data = ir3[6];
            ir3[6] = (v946_data + (v913_data * v944_data));
            float v949_data = s0[85];
            float v951_data = ir3[7];
            ir3[7] = (v951_data + (v913_data * v949_data));
            float v954_data = s0[97];
            float v956_data = ir3[8];
            ir3[8] = (v956_data + (v913_data * v954_data));
            float v959_data = s0[109];
            float v961_data = ir3[9];
            ir3[9] = (v961_data + (v913_data * v959_data));
            float v964_data = s0[121];
            float v966_data = ir3[10];
            ir3[10] = (v966_data + (v913_data * v964_data));
            float v969_data = s0[133];
            float v971_data = ir3[11];
            ir3[11] = (v971_data + (v913_data * v969_data));
          }
          if (v17_lead < 6) {
            float v977_data = r2[2];
            float v978_data = s0[2];
            float v980_data = ir3[0];
            ir3[0] = (v980_data + (v977_data * v978_data));
            float v983_data = s0[14];
            float v985_data = ir3[1];
            ir3[1] = (v985_data + (v977_data * v983_data));
            float v988_data = s0[26];
            float v990_data = ir3[2];
            ir3[2] = (v990_data + (v977_data * v988_data));
            float v993_data = s0[38];
            float v995_data = ir3[3];
            ir3[3] = (v995_data + (v977_data * v993_data));
            float v998_data = s0[50];
            float v1000_data = ir3[4];
            ir3[4] = (v1000_data + (v977_data * v998_data));
            float v1003_data = s0[62];
            float v1005_data = ir3[5];
            ir3[5] = (v1005_data + (v977_data * v1003_data));
            float v1008_data = s0[74];
            float v1010_data = ir3[6];
            ir3[6] = (v1010_data + (v977_data * v1008_data));
            float v1013_data = s0[86];
            float v1015_data = ir3[7];
            ir3[7] = (v1015_data + (v977_data * v1013_data));
            float v1018_data = s0[98];
            float v1020_data = ir3[8];
            ir3[8] = (v1020_data + (v977_data * v1018_data));
            float v1023_data = s0[110];
            float v1025_data = ir3[9];
            ir3[9] = (v1025_data + (v977_data * v1023_data));
            float v1028_data = s0[122];
            float v1030_data = ir3[10];
            ir3[10] = (v1030_data + (v977_data * v1028_data));
            float v1033_data = s0[134];
            float v1035_data = ir3[11];
            ir3[11] = (v1035_data + (v977_data * v1033_data));
          }
          if (v17_lead < 6) {
            float v1041_data = r2[3];
            float v1042_data = s0[3];
            float v1044_data = ir3[0];
            ir3[0] = (v1044_data + (v1041_data * v1042_data));
            float v1047_data = s0[15];
            float v1049_data = ir3[1];
            ir3[1] = (v1049_data + (v1041_data * v1047_data));
            float v1052_data = s0[27];
            float v1054_data = ir3[2];
            ir3[2] = (v1054_data + (v1041_data * v1052_data));
            float v1057_data = s0[39];
            float v1059_data = ir3[3];
            ir3[3] = (v1059_data + (v1041_data * v1057_data));
            float v1062_data = s0[51];
            float v1064_data = ir3[4];
            ir3[4] = (v1064_data + (v1041_data * v1062_data));
            float v1067_data = s0[63];
            float v1069_data = ir3[5];
            ir3[5] = (v1069_data + (v1041_data * v1067_data));
            float v1072_data = s0[75];
            float v1074_data = ir3[6];
            ir3[6] = (v1074_data + (v1041_data * v1072_data));
            float v1077_data = s0[87];
            float v1079_data = ir3[7];
            ir3[7] = (v1079_data + (v1041_data * v1077_data));
            float v1082_data = s0[99];
            float v1084_data = ir3[8];
            ir3[8] = (v1084_data + (v1041_data * v1082_data));
            float v1087_data = s0[111];
            float v1089_data = ir3[9];
            ir3[9] = (v1089_data + (v1041_data * v1087_data));
            float v1092_data = s0[123];
            float v1094_data = ir3[10];
            ir3[10] = (v1094_data + (v1041_data * v1092_data));
            float v1097_data = s0[135];
            float v1099_data = ir3[11];
            ir3[11] = (v1099_data + (v1041_data * v1097_data));
          }
          if (v17_lead < 6) {
            float v1105_data = r2[4];
            float v1106_data = s0[4];
            float v1108_data = ir3[0];
            ir3[0] = (v1108_data + (v1105_data * v1106_data));
            float v1111_data = s0[16];
            float v1113_data = ir3[1];
            ir3[1] = (v1113_data + (v1105_data * v1111_data));
            float v1116_data = s0[28];
            float v1118_data = ir3[2];
            ir3[2] = (v1118_data + (v1105_data * v1116_data));
            float v1121_data = s0[40];
            float v1123_data = ir3[3];
            ir3[3] = (v1123_data + (v1105_data * v1121_data));
            float v1126_data = s0[52];
            float v1128_data = ir3[4];
            ir3[4] = (v1128_data + (v1105_data * v1126_data));
            float v1131_data = s0[64];
            float v1133_data = ir3[5];
            ir3[5] = (v1133_data + (v1105_data * v1131_data));
            float v1136_data = s0[76];
            float v1138_data = ir3[6];
            ir3[6] = (v1138_data + (v1105_data * v1136_data));
            float v1141_data = s0[88];
            float v1143_data = ir3[7];
            ir3[7] = (v1143_data + (v1105_data * v1141_data));
            float v1146_data = s0[100];
            float v1148_data = ir3[8];
            ir3[8] = (v1148_data + (v1105_data * v1146_data));
            float v1151_data = s0[112];
            float v1153_data = ir3[9];
            ir3[9] = (v1153_data + (v1105_data * v1151_data));
            float v1156_data = s0[124];
            float v1158_data = ir3[10];
            ir3[10] = (v1158_data + (v1105_data * v1156_data));
            float v1161_data = s0[136];
            float v1163_data = ir3[11];
            ir3[11] = (v1163_data + (v1105_data * v1161_data));
          }
          if (v17_lead < 6) {
            float v1169_data = r2[5];
            float v1170_data = s0[5];
            float v1172_data = ir3[0];
            ir3[0] = (v1172_data + (v1169_data * v1170_data));
            float v1175_data = s0[17];
            float v1177_data = ir3[1];
            ir3[1] = (v1177_data + (v1169_data * v1175_data));
            float v1180_data = s0[29];
            float v1182_data = ir3[2];
            ir3[2] = (v1182_data + (v1169_data * v1180_data));
            float v1185_data = s0[41];
            float v1187_data = ir3[3];
            ir3[3] = (v1187_data + (v1169_data * v1185_data));
            float v1190_data = s0[53];
            float v1192_data = ir3[4];
            ir3[4] = (v1192_data + (v1169_data * v1190_data));
            float v1195_data = s0[65];
            float v1197_data = ir3[5];
            ir3[5] = (v1197_data + (v1169_data * v1195_data));
            float v1200_data = s0[77];
            float v1202_data = ir3[6];
            ir3[6] = (v1202_data + (v1169_data * v1200_data));
            float v1205_data = s0[89];
            float v1207_data = ir3[7];
            ir3[7] = (v1207_data + (v1169_data * v1205_data));
            float v1210_data = s0[101];
            float v1212_data = ir3[8];
            ir3[8] = (v1212_data + (v1169_data * v1210_data));
            float v1215_data = s0[113];
            float v1217_data = ir3[9];
            ir3[9] = (v1217_data + (v1169_data * v1215_data));
            float v1220_data = s0[125];
            float v1222_data = ir3[10];
            ir3[10] = (v1222_data + (v1169_data * v1220_data));
            float v1225_data = s0[137];
            float v1227_data = ir3[11];
            ir3[11] = (v1227_data + (v1169_data * v1225_data));
          }
          if (v17_lead < 6) {
            float v1233_data = r2[6];
            float v1234_data = s0[6];
            float v1236_data = ir3[0];
            ir3[0] = (v1236_data + (v1233_data * v1234_data));
            float v1239_data = s0[18];
            float v1241_data = ir3[1];
            ir3[1] = (v1241_data + (v1233_data * v1239_data));
            float v1244_data = s0[30];
            float v1246_data = ir3[2];
            ir3[2] = (v1246_data + (v1233_data * v1244_data));
            float v1249_data = s0[42];
            float v1251_data = ir3[3];
            ir3[3] = (v1251_data + (v1233_data * v1249_data));
            float v1254_data = s0[54];
            float v1256_data = ir3[4];
            ir3[4] = (v1256_data + (v1233_data * v1254_data));
            float v1259_data = s0[66];
            float v1261_data = ir3[5];
            ir3[5] = (v1261_data + (v1233_data * v1259_data));
            float v1264_data = s0[78];
            float v1266_data = ir3[6];
            ir3[6] = (v1266_data + (v1233_data * v1264_data));
            float v1269_data = s0[90];
            float v1271_data = ir3[7];
            ir3[7] = (v1271_data + (v1233_data * v1269_data));
            float v1274_data = s0[102];
            float v1276_data = ir3[8];
            ir3[8] = (v1276_data + (v1233_data * v1274_data));
            float v1279_data = s0[114];
            float v1281_data = ir3[9];
            ir3[9] = (v1281_data + (v1233_data * v1279_data));
            float v1284_data = s0[126];
            float v1286_data = ir3[10];
            ir3[10] = (v1286_data + (v1233_data * v1284_data));
            float v1289_data = s0[138];
            float v1291_data = ir3[11];
            ir3[11] = (v1291_data + (v1233_data * v1289_data));
          }
          if (v17_lead < 6) {
            float v1297_data = r2[7];
            float v1298_data = s0[7];
            float v1300_data = ir3[0];
            ir3[0] = (v1300_data + (v1297_data * v1298_data));
            float v1303_data = s0[19];
            float v1305_data = ir3[1];
            ir3[1] = (v1305_data + (v1297_data * v1303_data));
            float v1308_data = s0[31];
            float v1310_data = ir3[2];
            ir3[2] = (v1310_data + (v1297_data * v1308_data));
            float v1313_data = s0[43];
            float v1315_data = ir3[3];
            ir3[3] = (v1315_data + (v1297_data * v1313_data));
            float v1318_data = s0[55];
            float v1320_data = ir3[4];
            ir3[4] = (v1320_data + (v1297_data * v1318_data));
            float v1323_data = s0[67];
            float v1325_data = ir3[5];
            ir3[5] = (v1325_data + (v1297_data * v1323_data));
            float v1328_data = s0[79];
            float v1330_data = ir3[6];
            ir3[6] = (v1330_data + (v1297_data * v1328_data));
            float v1333_data = s0[91];
            float v1335_data = ir3[7];
            ir3[7] = (v1335_data + (v1297_data * v1333_data));
            float v1338_data = s0[103];
            float v1340_data = ir3[8];
            ir3[8] = (v1340_data + (v1297_data * v1338_data));
            float v1343_data = s0[115];
            float v1345_data = ir3[9];
            ir3[9] = (v1345_data + (v1297_data * v1343_data));
            float v1348_data = s0[127];
            float v1350_data = ir3[10];
            ir3[10] = (v1350_data + (v1297_data * v1348_data));
            float v1353_data = s0[139];
            float v1355_data = ir3[11];
            ir3[11] = (v1355_data + (v1297_data * v1353_data));
          }
          if (v17_lead < 6) {
            float v1361_data = r2[8];
            float v1362_data = s0[8];
            float v1364_data = ir3[0];
            ir3[0] = (v1364_data + (v1361_data * v1362_data));
            float v1367_data = s0[20];
            float v1369_data = ir3[1];
            ir3[1] = (v1369_data + (v1361_data * v1367_data));
            float v1372_data = s0[32];
            float v1374_data = ir3[2];
            ir3[2] = (v1374_data + (v1361_data * v1372_data));
            float v1377_data = s0[44];
            float v1379_data = ir3[3];
            ir3[3] = (v1379_data + (v1361_data * v1377_data));
            float v1382_data = s0[56];
            float v1384_data = ir3[4];
            ir3[4] = (v1384_data + (v1361_data * v1382_data));
            float v1387_data = s0[68];
            float v1389_data = ir3[5];
            ir3[5] = (v1389_data + (v1361_data * v1387_data));
            float v1392_data = s0[80];
            float v1394_data = ir3[6];
            ir3[6] = (v1394_data + (v1361_data * v1392_data));
            float v1397_data = s0[92];
            float v1399_data = ir3[7];
            ir3[7] = (v1399_data + (v1361_data * v1397_data));
            float v1402_data = s0[104];
            float v1404_data = ir3[8];
            ir3[8] = (v1404_data + (v1361_data * v1402_data));
            float v1407_data = s0[116];
            float v1409_data = ir3[9];
            ir3[9] = (v1409_data + (v1361_data * v1407_data));
            float v1412_data = s0[128];
            float v1414_data = ir3[10];
            ir3[10] = (v1414_data + (v1361_data * v1412_data));
            float v1417_data = s0[140];
            float v1419_data = ir3[11];
            ir3[11] = (v1419_data + (v1361_data * v1417_data));
          }
          if (v17_lead < 6) {
            float v1425_data = r2[9];
            float v1426_data = s0[9];
            float v1428_data = ir3[0];
            ir3[0] = (v1428_data + (v1425_data * v1426_data));
            float v1431_data = s0[21];
            float v1433_data = ir3[1];
            ir3[1] = (v1433_data + (v1425_data * v1431_data));
            float v1436_data = s0[33];
            float v1438_data = ir3[2];
            ir3[2] = (v1438_data + (v1425_data * v1436_data));
            float v1441_data = s0[45];
            float v1443_data = ir3[3];
            ir3[3] = (v1443_data + (v1425_data * v1441_data));
            float v1446_data = s0[57];
            float v1448_data = ir3[4];
            ir3[4] = (v1448_data + (v1425_data * v1446_data));
            float v1451_data = s0[69];
            float v1453_data = ir3[5];
            ir3[5] = (v1453_data + (v1425_data * v1451_data));
            float v1456_data = s0[81];
            float v1458_data = ir3[6];
            ir3[6] = (v1458_data + (v1425_data * v1456_data));
            float v1461_data = s0[93];
            float v1463_data = ir3[7];
            ir3[7] = (v1463_data + (v1425_data * v1461_data));
            float v1466_data = s0[105];
            float v1468_data = ir3[8];
            ir3[8] = (v1468_data + (v1425_data * v1466_data));
            float v1471_data = s0[117];
            float v1473_data = ir3[9];
            ir3[9] = (v1473_data + (v1425_data * v1471_data));
            float v1476_data = s0[129];
            float v1478_data = ir3[10];
            ir3[10] = (v1478_data + (v1425_data * v1476_data));
            float v1481_data = s0[141];
            float v1483_data = ir3[11];
            ir3[11] = (v1483_data + (v1425_data * v1481_data));
          }
          if (v17_lead < 6) {
            float v1489_data = r2[10];
            float v1490_data = s0[10];
            float v1492_data = ir3[0];
            ir3[0] = (v1492_data + (v1489_data * v1490_data));
            float v1495_data = s0[22];
            float v1497_data = ir3[1];
            ir3[1] = (v1497_data + (v1489_data * v1495_data));
            float v1500_data = s0[34];
            float v1502_data = ir3[2];
            ir3[2] = (v1502_data + (v1489_data * v1500_data));
            float v1505_data = s0[46];
            float v1507_data = ir3[3];
            ir3[3] = (v1507_data + (v1489_data * v1505_data));
            float v1510_data = s0[58];
            float v1512_data = ir3[4];
            ir3[4] = (v1512_data + (v1489_data * v1510_data));
            float v1515_data = s0[70];
            float v1517_data = ir3[5];
            ir3[5] = (v1517_data + (v1489_data * v1515_data));
            float v1520_data = s0[82];
            float v1522_data = ir3[6];
            ir3[6] = (v1522_data + (v1489_data * v1520_data));
            float v1525_data = s0[94];
            float v1527_data = ir3[7];
            ir3[7] = (v1527_data + (v1489_data * v1525_data));
            float v1530_data = s0[106];
            float v1532_data = ir3[8];
            ir3[8] = (v1532_data + (v1489_data * v1530_data));
            float v1535_data = s0[118];
            float v1537_data = ir3[9];
            ir3[9] = (v1537_data + (v1489_data * v1535_data));
            float v1540_data = s0[130];
            float v1542_data = ir3[10];
            ir3[10] = (v1542_data + (v1489_data * v1540_data));
            float v1545_data = s0[142];
            float v1547_data = ir3[11];
            ir3[11] = (v1547_data + (v1489_data * v1545_data));
          }
          if (v17_lead < 6) {
            float v1553_data = r2[11];
            float v1554_data = s0[11];
            float v1556_data = ir3[0];
            ir3[0] = (v1556_data + (v1553_data * v1554_data));
            float v1559_data = s0[23];
            float v1561_data = ir3[1];
            ir3[1] = (v1561_data + (v1553_data * v1559_data));
            float v1564_data = s0[35];
            float v1566_data = ir3[2];
            ir3[2] = (v1566_data + (v1553_data * v1564_data));
            float v1569_data = s0[47];
            float v1571_data = ir3[3];
            ir3[3] = (v1571_data + (v1553_data * v1569_data));
            float v1574_data = s0[59];
            float v1576_data = ir3[4];
            ir3[4] = (v1576_data + (v1553_data * v1574_data));
            float v1579_data = s0[71];
            float v1581_data = ir3[5];
            ir3[5] = (v1581_data + (v1553_data * v1579_data));
            float v1584_data = s0[83];
            float v1586_data = ir3[6];
            ir3[6] = (v1586_data + (v1553_data * v1584_data));
            float v1589_data = s0[95];
            float v1591_data = ir3[7];
            ir3[7] = (v1591_data + (v1553_data * v1589_data));
            float v1594_data = s0[107];
            float v1596_data = ir3[8];
            ir3[8] = (v1596_data + (v1553_data * v1594_data));
            float v1599_data = s0[119];
            float v1601_data = ir3[9];
            ir3[9] = (v1601_data + (v1553_data * v1599_data));
            float v1604_data = s0[131];
            float v1606_data = ir3[10];
            ir3[10] = (v1606_data + (v1553_data * v1604_data));
            float v1609_data = s0[143];
            float v1611_data = ir3[11];
            ir3[11] = (v1611_data + (v1553_data * v1609_data));
          }
          if (v17_lead < 6) {
            #pragma unroll
            for (int32_t v1617_n1 = 0; v1617_n1 < 12; ++v1617_n1) {
              float v1619_data = ir3[v1617_n1];
              r3[v1617_n1] = v1619_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v17_lead < 6) {
            int32_t v1633_off = v17_lead + 6;
            #pragma unroll
            for (int32_t v1625_i1 = 0; v1625_i1 < 12; ++v1625_i1) {
              float v1627_data = r3[v1625_i1];
              s1[(v1633_off + (v1625_i1 * 12))] = v1627_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v17_lead < 12) {
            float v1642_data = r4[0];
            float v1643_data = s1[0];
            float v1645_data = ir5[0];
            ir5[0] = (v1645_data + (v1642_data * v1643_data));
            float v1648_data = s1[12];
            float v1650_data = ir5[1];
            ir5[1] = (v1650_data + (v1642_data * v1648_data));
            float v1653_data = s1[24];
            float v1655_data = ir5[2];
            ir5[2] = (v1655_data + (v1642_data * v1653_data));
            float v1658_data = s1[36];
            float v1660_data = ir5[3];
            ir5[3] = (v1660_data + (v1642_data * v1658_data));
            float v1663_data = s1[48];
            float v1665_data = ir5[4];
            ir5[4] = (v1665_data + (v1642_data * v1663_data));
            float v1668_data = s1[60];
            float v1670_data = ir5[5];
            ir5[5] = (v1670_data + (v1642_data * v1668_data));
            float v1673_data = s1[72];
            float v1675_data = ir5[6];
            ir5[6] = (v1675_data + (v1642_data * v1673_data));
            float v1678_data = s1[84];
            float v1680_data = ir5[7];
            ir5[7] = (v1680_data + (v1642_data * v1678_data));
            float v1683_data = s1[96];
            float v1685_data = ir5[8];
            ir5[8] = (v1685_data + (v1642_data * v1683_data));
            float v1688_data = s1[108];
            float v1690_data = ir5[9];
            ir5[9] = (v1690_data + (v1642_data * v1688_data));
            float v1693_data = s1[120];
            float v1695_data = ir5[10];
            ir5[10] = (v1695_data + (v1642_data * v1693_data));
            float v1698_data = s1[132];
            float v1700_data = ir5[11];
            ir5[11] = (v1700_data + (v1642_data * v1698_data));
          }
          if (v17_lead < 12) {
            float v1706_data = r4[1];
            float v1707_data = s1[1];
            float v1709_data = ir5[0];
            ir5[0] = (v1709_data + (v1706_data * v1707_data));
            float v1712_data = s1[13];
            float v1714_data = ir5[1];
            ir5[1] = (v1714_data + (v1706_data * v1712_data));
            float v1717_data = s1[25];
            float v1719_data = ir5[2];
            ir5[2] = (v1719_data + (v1706_data * v1717_data));
            float v1722_data = s1[37];
            float v1724_data = ir5[3];
            ir5[3] = (v1724_data + (v1706_data * v1722_data));
            float v1727_data = s1[49];
            float v1729_data = ir5[4];
            ir5[4] = (v1729_data + (v1706_data * v1727_data));
            float v1732_data = s1[61];
            float v1734_data = ir5[5];
            ir5[5] = (v1734_data + (v1706_data * v1732_data));
            float v1737_data = s1[73];
            float v1739_data = ir5[6];
            ir5[6] = (v1739_data + (v1706_data * v1737_data));
            float v1742_data = s1[85];
            float v1744_data = ir5[7];
            ir5[7] = (v1744_data + (v1706_data * v1742_data));
            float v1747_data = s1[97];
            float v1749_data = ir5[8];
            ir5[8] = (v1749_data + (v1706_data * v1747_data));
            float v1752_data = s1[109];
            float v1754_data = ir5[9];
            ir5[9] = (v1754_data + (v1706_data * v1752_data));
            float v1757_data = s1[121];
            float v1759_data = ir5[10];
            ir5[10] = (v1759_data + (v1706_data * v1757_data));
            float v1762_data = s1[133];
            float v1764_data = ir5[11];
            ir5[11] = (v1764_data + (v1706_data * v1762_data));
          }
          if (v17_lead < 12) {
            float v1770_data = r4[2];
            float v1771_data = s1[2];
            float v1773_data = ir5[0];
            ir5[0] = (v1773_data + (v1770_data * v1771_data));
            float v1776_data = s1[14];
            float v1778_data = ir5[1];
            ir5[1] = (v1778_data + (v1770_data * v1776_data));
            float v1781_data = s1[26];
            float v1783_data = ir5[2];
            ir5[2] = (v1783_data + (v1770_data * v1781_data));
            float v1786_data = s1[38];
            float v1788_data = ir5[3];
            ir5[3] = (v1788_data + (v1770_data * v1786_data));
            float v1791_data = s1[50];
            float v1793_data = ir5[4];
            ir5[4] = (v1793_data + (v1770_data * v1791_data));
            float v1796_data = s1[62];
            float v1798_data = ir5[5];
            ir5[5] = (v1798_data + (v1770_data * v1796_data));
            float v1801_data = s1[74];
            float v1803_data = ir5[6];
            ir5[6] = (v1803_data + (v1770_data * v1801_data));
            float v1806_data = s1[86];
            float v1808_data = ir5[7];
            ir5[7] = (v1808_data + (v1770_data * v1806_data));
            float v1811_data = s1[98];
            float v1813_data = ir5[8];
            ir5[8] = (v1813_data + (v1770_data * v1811_data));
            float v1816_data = s1[110];
            float v1818_data = ir5[9];
            ir5[9] = (v1818_data + (v1770_data * v1816_data));
            float v1821_data = s1[122];
            float v1823_data = ir5[10];
            ir5[10] = (v1823_data + (v1770_data * v1821_data));
            float v1826_data = s1[134];
            float v1828_data = ir5[11];
            ir5[11] = (v1828_data + (v1770_data * v1826_data));
          }
          if (v17_lead < 12) {
            float v1834_data = r4[3];
            float v1835_data = s1[3];
            float v1837_data = ir5[0];
            ir5[0] = (v1837_data + (v1834_data * v1835_data));
            float v1840_data = s1[15];
            float v1842_data = ir5[1];
            ir5[1] = (v1842_data + (v1834_data * v1840_data));
            float v1845_data = s1[27];
            float v1847_data = ir5[2];
            ir5[2] = (v1847_data + (v1834_data * v1845_data));
            float v1850_data = s1[39];
            float v1852_data = ir5[3];
            ir5[3] = (v1852_data + (v1834_data * v1850_data));
            float v1855_data = s1[51];
            float v1857_data = ir5[4];
            ir5[4] = (v1857_data + (v1834_data * v1855_data));
            float v1860_data = s1[63];
            float v1862_data = ir5[5];
            ir5[5] = (v1862_data + (v1834_data * v1860_data));
            float v1865_data = s1[75];
            float v1867_data = ir5[6];
            ir5[6] = (v1867_data + (v1834_data * v1865_data));
            float v1870_data = s1[87];
            float v1872_data = ir5[7];
            ir5[7] = (v1872_data + (v1834_data * v1870_data));
            float v1875_data = s1[99];
            float v1877_data = ir5[8];
            ir5[8] = (v1877_data + (v1834_data * v1875_data));
            float v1880_data = s1[111];
            float v1882_data = ir5[9];
            ir5[9] = (v1882_data + (v1834_data * v1880_data));
            float v1885_data = s1[123];
            float v1887_data = ir5[10];
            ir5[10] = (v1887_data + (v1834_data * v1885_data));
            float v1890_data = s1[135];
            float v1892_data = ir5[11];
            ir5[11] = (v1892_data + (v1834_data * v1890_data));
          }
          if (v17_lead < 12) {
            float v1898_data = r4[4];
            float v1899_data = s1[4];
            float v1901_data = ir5[0];
            ir5[0] = (v1901_data + (v1898_data * v1899_data));
            float v1904_data = s1[16];
            float v1906_data = ir5[1];
            ir5[1] = (v1906_data + (v1898_data * v1904_data));
            float v1909_data = s1[28];
            float v1911_data = ir5[2];
            ir5[2] = (v1911_data + (v1898_data * v1909_data));
            float v1914_data = s1[40];
            float v1916_data = ir5[3];
            ir5[3] = (v1916_data + (v1898_data * v1914_data));
            float v1919_data = s1[52];
            float v1921_data = ir5[4];
            ir5[4] = (v1921_data + (v1898_data * v1919_data));
            float v1924_data = s1[64];
            float v1926_data = ir5[5];
            ir5[5] = (v1926_data + (v1898_data * v1924_data));
            float v1929_data = s1[76];
            float v1931_data = ir5[6];
            ir5[6] = (v1931_data + (v1898_data * v1929_data));
            float v1934_data = s1[88];
            float v1936_data = ir5[7];
            ir5[7] = (v1936_data + (v1898_data * v1934_data));
            float v1939_data = s1[100];
            float v1941_data = ir5[8];
            ir5[8] = (v1941_data + (v1898_data * v1939_data));
            float v1944_data = s1[112];
            float v1946_data = ir5[9];
            ir5[9] = (v1946_data + (v1898_data * v1944_data));
            float v1949_data = s1[124];
            float v1951_data = ir5[10];
            ir5[10] = (v1951_data + (v1898_data * v1949_data));
            float v1954_data = s1[136];
            float v1956_data = ir5[11];
            ir5[11] = (v1956_data + (v1898_data * v1954_data));
          }
          if (v17_lead < 12) {
            float v1962_data = r4[5];
            float v1963_data = s1[5];
            float v1965_data = ir5[0];
            ir5[0] = (v1965_data + (v1962_data * v1963_data));
            float v1968_data = s1[17];
            float v1970_data = ir5[1];
            ir5[1] = (v1970_data + (v1962_data * v1968_data));
            float v1973_data = s1[29];
            float v1975_data = ir5[2];
            ir5[2] = (v1975_data + (v1962_data * v1973_data));
            float v1978_data = s1[41];
            float v1980_data = ir5[3];
            ir5[3] = (v1980_data + (v1962_data * v1978_data));
            float v1983_data = s1[53];
            float v1985_data = ir5[4];
            ir5[4] = (v1985_data + (v1962_data * v1983_data));
            float v1988_data = s1[65];
            float v1990_data = ir5[5];
            ir5[5] = (v1990_data + (v1962_data * v1988_data));
            float v1993_data = s1[77];
            float v1995_data = ir5[6];
            ir5[6] = (v1995_data + (v1962_data * v1993_data));
            float v1998_data = s1[89];
            float v2000_data = ir5[7];
            ir5[7] = (v2000_data + (v1962_data * v1998_data));
            float v2003_data = s1[101];
            float v2005_data = ir5[8];
            ir5[8] = (v2005_data + (v1962_data * v2003_data));
            float v2008_data = s1[113];
            float v2010_data = ir5[9];
            ir5[9] = (v2010_data + (v1962_data * v2008_data));
            float v2013_data = s1[125];
            float v2015_data = ir5[10];
            ir5[10] = (v2015_data + (v1962_data * v2013_data));
            float v2018_data = s1[137];
            float v2020_data = ir5[11];
            ir5[11] = (v2020_data + (v1962_data * v2018_data));
          }
          if (v17_lead < 12) {
            float v2026_data = r4[6];
            float v2027_data = s1[6];
            float v2029_data = ir5[0];
            ir5[0] = (v2029_data + (v2026_data * v2027_data));
            float v2032_data = s1[18];
            float v2034_data = ir5[1];
            ir5[1] = (v2034_data + (v2026_data * v2032_data));
            float v2037_data = s1[30];
            float v2039_data = ir5[2];
            ir5[2] = (v2039_data + (v2026_data * v2037_data));
            float v2042_data = s1[42];
            float v2044_data = ir5[3];
            ir5[3] = (v2044_data + (v2026_data * v2042_data));
            float v2047_data = s1[54];
            float v2049_data = ir5[4];
            ir5[4] = (v2049_data + (v2026_data * v2047_data));
            float v2052_data = s1[66];
            float v2054_data = ir5[5];
            ir5[5] = (v2054_data + (v2026_data * v2052_data));
            float v2057_data = s1[78];
            float v2059_data = ir5[6];
            ir5[6] = (v2059_data + (v2026_data * v2057_data));
            float v2062_data = s1[90];
            float v2064_data = ir5[7];
            ir5[7] = (v2064_data + (v2026_data * v2062_data));
            float v2067_data = s1[102];
            float v2069_data = ir5[8];
            ir5[8] = (v2069_data + (v2026_data * v2067_data));
            float v2072_data = s1[114];
            float v2074_data = ir5[9];
            ir5[9] = (v2074_data + (v2026_data * v2072_data));
            float v2077_data = s1[126];
            float v2079_data = ir5[10];
            ir5[10] = (v2079_data + (v2026_data * v2077_data));
            float v2082_data = s1[138];
            float v2084_data = ir5[11];
            ir5[11] = (v2084_data + (v2026_data * v2082_data));
          }
          if (v17_lead < 12) {
            float v2090_data = r4[7];
            float v2091_data = s1[7];
            float v2093_data = ir5[0];
            ir5[0] = (v2093_data + (v2090_data * v2091_data));
            float v2096_data = s1[19];
            float v2098_data = ir5[1];
            ir5[1] = (v2098_data + (v2090_data * v2096_data));
            float v2101_data = s1[31];
            float v2103_data = ir5[2];
            ir5[2] = (v2103_data + (v2090_data * v2101_data));
            float v2106_data = s1[43];
            float v2108_data = ir5[3];
            ir5[3] = (v2108_data + (v2090_data * v2106_data));
            float v2111_data = s1[55];
            float v2113_data = ir5[4];
            ir5[4] = (v2113_data + (v2090_data * v2111_data));
            float v2116_data = s1[67];
            float v2118_data = ir5[5];
            ir5[5] = (v2118_data + (v2090_data * v2116_data));
            float v2121_data = s1[79];
            float v2123_data = ir5[6];
            ir5[6] = (v2123_data + (v2090_data * v2121_data));
            float v2126_data = s1[91];
            float v2128_data = ir5[7];
            ir5[7] = (v2128_data + (v2090_data * v2126_data));
            float v2131_data = s1[103];
            float v2133_data = ir5[8];
            ir5[8] = (v2133_data + (v2090_data * v2131_data));
            float v2136_data = s1[115];
            float v2138_data = ir5[9];
            ir5[9] = (v2138_data + (v2090_data * v2136_data));
            float v2141_data = s1[127];
            float v2143_data = ir5[10];
            ir5[10] = (v2143_data + (v2090_data * v2141_data));
            float v2146_data = s1[139];
            float v2148_data = ir5[11];
            ir5[11] = (v2148_data + (v2090_data * v2146_data));
          }
          if (v17_lead < 12) {
            float v2154_data = r4[8];
            float v2155_data = s1[8];
            float v2157_data = ir5[0];
            ir5[0] = (v2157_data + (v2154_data * v2155_data));
            float v2160_data = s1[20];
            float v2162_data = ir5[1];
            ir5[1] = (v2162_data + (v2154_data * v2160_data));
            float v2165_data = s1[32];
            float v2167_data = ir5[2];
            ir5[2] = (v2167_data + (v2154_data * v2165_data));
            float v2170_data = s1[44];
            float v2172_data = ir5[3];
            ir5[3] = (v2172_data + (v2154_data * v2170_data));
            float v2175_data = s1[56];
            float v2177_data = ir5[4];
            ir5[4] = (v2177_data + (v2154_data * v2175_data));
            float v2180_data = s1[68];
            float v2182_data = ir5[5];
            ir5[5] = (v2182_data + (v2154_data * v2180_data));
            float v2185_data = s1[80];
            float v2187_data = ir5[6];
            ir5[6] = (v2187_data + (v2154_data * v2185_data));
            float v2190_data = s1[92];
            float v2192_data = ir5[7];
            ir5[7] = (v2192_data + (v2154_data * v2190_data));
            float v2195_data = s1[104];
            float v2197_data = ir5[8];
            ir5[8] = (v2197_data + (v2154_data * v2195_data));
            float v2200_data = s1[116];
            float v2202_data = ir5[9];
            ir5[9] = (v2202_data + (v2154_data * v2200_data));
            float v2205_data = s1[128];
            float v2207_data = ir5[10];
            ir5[10] = (v2207_data + (v2154_data * v2205_data));
            float v2210_data = s1[140];
            float v2212_data = ir5[11];
            ir5[11] = (v2212_data + (v2154_data * v2210_data));
          }
          if (v17_lead < 12) {
            float v2218_data = r4[9];
            float v2219_data = s1[9];
            float v2221_data = ir5[0];
            ir5[0] = (v2221_data + (v2218_data * v2219_data));
            float v2224_data = s1[21];
            float v2226_data = ir5[1];
            ir5[1] = (v2226_data + (v2218_data * v2224_data));
            float v2229_data = s1[33];
            float v2231_data = ir5[2];
            ir5[2] = (v2231_data + (v2218_data * v2229_data));
            float v2234_data = s1[45];
            float v2236_data = ir5[3];
            ir5[3] = (v2236_data + (v2218_data * v2234_data));
            float v2239_data = s1[57];
            float v2241_data = ir5[4];
            ir5[4] = (v2241_data + (v2218_data * v2239_data));
            float v2244_data = s1[69];
            float v2246_data = ir5[5];
            ir5[5] = (v2246_data + (v2218_data * v2244_data));
            float v2249_data = s1[81];
            float v2251_data = ir5[6];
            ir5[6] = (v2251_data + (v2218_data * v2249_data));
            float v2254_data = s1[93];
            float v2256_data = ir5[7];
            ir5[7] = (v2256_data + (v2218_data * v2254_data));
            float v2259_data = s1[105];
            float v2261_data = ir5[8];
            ir5[8] = (v2261_data + (v2218_data * v2259_data));
            float v2264_data = s1[117];
            float v2266_data = ir5[9];
            ir5[9] = (v2266_data + (v2218_data * v2264_data));
            float v2269_data = s1[129];
            float v2271_data = ir5[10];
            ir5[10] = (v2271_data + (v2218_data * v2269_data));
            float v2274_data = s1[141];
            float v2276_data = ir5[11];
            ir5[11] = (v2276_data + (v2218_data * v2274_data));
          }
          if (v17_lead < 12) {
            float v2282_data = r4[10];
            float v2283_data = s1[10];
            float v2285_data = ir5[0];
            ir5[0] = (v2285_data + (v2282_data * v2283_data));
            float v2288_data = s1[22];
            float v2290_data = ir5[1];
            ir5[1] = (v2290_data + (v2282_data * v2288_data));
            float v2293_data = s1[34];
            float v2295_data = ir5[2];
            ir5[2] = (v2295_data + (v2282_data * v2293_data));
            float v2298_data = s1[46];
            float v2300_data = ir5[3];
            ir5[3] = (v2300_data + (v2282_data * v2298_data));
            float v2303_data = s1[58];
            float v2305_data = ir5[4];
            ir5[4] = (v2305_data + (v2282_data * v2303_data));
            float v2308_data = s1[70];
            float v2310_data = ir5[5];
            ir5[5] = (v2310_data + (v2282_data * v2308_data));
            float v2313_data = s1[82];
            float v2315_data = ir5[6];
            ir5[6] = (v2315_data + (v2282_data * v2313_data));
            float v2318_data = s1[94];
            float v2320_data = ir5[7];
            ir5[7] = (v2320_data + (v2282_data * v2318_data));
            float v2323_data = s1[106];
            float v2325_data = ir5[8];
            ir5[8] = (v2325_data + (v2282_data * v2323_data));
            float v2328_data = s1[118];
            float v2330_data = ir5[9];
            ir5[9] = (v2330_data + (v2282_data * v2328_data));
            float v2333_data = s1[130];
            float v2335_data = ir5[10];
            ir5[10] = (v2335_data + (v2282_data * v2333_data));
            float v2338_data = s1[142];
            float v2340_data = ir5[11];
            ir5[11] = (v2340_data + (v2282_data * v2338_data));
          }
          if (v17_lead < 12) {
            float v2346_data = r4[11];
            float v2347_data = s1[11];
            float v2349_data = ir5[0];
            ir5[0] = (v2349_data + (v2346_data * v2347_data));
            float v2352_data = s1[23];
            float v2354_data = ir5[1];
            ir5[1] = (v2354_data + (v2346_data * v2352_data));
            float v2357_data = s1[35];
            float v2359_data = ir5[2];
            ir5[2] = (v2359_data + (v2346_data * v2357_data));
            float v2362_data = s1[47];
            float v2364_data = ir5[3];
            ir5[3] = (v2364_data + (v2346_data * v2362_data));
            float v2367_data = s1[59];
            float v2369_data = ir5[4];
            ir5[4] = (v2369_data + (v2346_data * v2367_data));
            float v2372_data = s1[71];
            float v2374_data = ir5[5];
            ir5[5] = (v2374_data + (v2346_data * v2372_data));
            float v2377_data = s1[83];
            float v2379_data = ir5[6];
            ir5[6] = (v2379_data + (v2346_data * v2377_data));
            float v2382_data = s1[95];
            float v2384_data = ir5[7];
            ir5[7] = (v2384_data + (v2346_data * v2382_data));
            float v2387_data = s1[107];
            float v2389_data = ir5[8];
            ir5[8] = (v2389_data + (v2346_data * v2387_data));
            float v2392_data = s1[119];
            float v2394_data = ir5[9];
            ir5[9] = (v2394_data + (v2346_data * v2392_data));
            float v2397_data = s1[131];
            float v2399_data = ir5[10];
            ir5[10] = (v2399_data + (v2346_data * v2397_data));
            float v2402_data = s1[143];
            float v2404_data = ir5[11];
            ir5[11] = (v2404_data + (v2346_data * v2402_data));
          }
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v2410_n1 = 0; v2410_n1 < 12; ++v2410_n1) {
              float v2412_data = ir5[v2410_n1];
              r5[v2410_n1] = v2412_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v17_lead < 12) {
            #pragma unroll
            for (int32_t v2418_i1 = 0; v2418_i1 < 12; ++v2418_i1) {
              float v2420_data = r5[v2418_i1];
              glb_m3[(v17_lead + (v2418_i1 * 12))] = v2420_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

