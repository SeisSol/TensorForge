// === base name ===
kernel_3ff25cfed1

// === header ===
void launcher_kernel_3ff25cfed1(double* m0, unsigned m0_extraOffset, const double* m1, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3ff25cfed1(double* m0, unsigned m0_extraOffset, const double* m1, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3ff25cfed1, block.x * block.y * block.z, 4352 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_3ff25cfed1, cudaFuncAttributeMaxDynamicSharedMemorySize, 4352 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3ff25cfed1<<<grid,block,4352 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_3ff25cfed1(double* m0, unsigned m0_extraOffset, const double* m1, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} none
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} none({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[272 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[256];
      const double *const __restrict__ glb_m1 = &m1[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 16; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<8>(8), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          double r0[16]{};
          __syncwarp();
          {
            // r0 = +(glb_m1 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            double ir0[16]{};
            int32_t v2_lead = threadIdx.x % 16;
            int32_t v8_a = v2_lead + 0;
            double v9_data;
            {
              v9_data = glb_m1[v8_a];
            }
            double v10_data = s0[0];
            double v12_data = ir0[0];
            ir0[0] = (v12_data + (v9_data * v10_data));
            int32_t v19_a = v2_lead + 0;
            double v20_data;
            {
              v20_data = glb_m1[v19_a];
            }
            double v21_data = s0[16];
            double v23_data = ir0[1];
            ir0[1] = (v23_data + (v20_data * v21_data));
            int32_t v30_a = v2_lead + 0;
            double v31_data;
            {
              v31_data = glb_m1[v30_a];
            }
            double v32_data = s0[32];
            double v34_data = ir0[2];
            ir0[2] = (v34_data + (v31_data * v32_data));
            int32_t v41_a = v2_lead + 0;
            double v42_data;
            {
              v42_data = glb_m1[v41_a];
            }
            double v43_data = s0[48];
            double v45_data = ir0[3];
            ir0[3] = (v45_data + (v42_data * v43_data));
            int32_t v52_a = v2_lead + 0;
            double v53_data;
            {
              v53_data = glb_m1[v52_a];
            }
            double v54_data = s0[64];
            double v56_data = ir0[4];
            ir0[4] = (v56_data + (v53_data * v54_data));
            int32_t v63_a = v2_lead + 0;
            double v64_data;
            {
              v64_data = glb_m1[v63_a];
            }
            double v65_data = s0[80];
            double v67_data = ir0[5];
            ir0[5] = (v67_data + (v64_data * v65_data));
            int32_t v74_a = v2_lead + 0;
            double v75_data;
            {
              v75_data = glb_m1[v74_a];
            }
            double v76_data = s0[96];
            double v78_data = ir0[6];
            ir0[6] = (v78_data + (v75_data * v76_data));
            int32_t v85_a = v2_lead + 0;
            double v86_data;
            {
              v86_data = glb_m1[v85_a];
            }
            double v87_data = s0[112];
            double v89_data = ir0[7];
            ir0[7] = (v89_data + (v86_data * v87_data));
            int32_t v96_a = v2_lead + 0;
            double v97_data;
            {
              v97_data = glb_m1[v96_a];
            }
            double v98_data = s0[128];
            double v100_data = ir0[8];
            ir0[8] = (v100_data + (v97_data * v98_data));
            int32_t v107_a = v2_lead + 0;
            double v108_data;
            {
              v108_data = glb_m1[v107_a];
            }
            double v109_data = s0[144];
            double v111_data = ir0[9];
            ir0[9] = (v111_data + (v108_data * v109_data));
            int32_t v118_a = v2_lead + 0;
            double v119_data;
            {
              v119_data = glb_m1[v118_a];
            }
            double v120_data = s0[160];
            double v122_data = ir0[10];
            ir0[10] = (v122_data + (v119_data * v120_data));
            int32_t v129_a = v2_lead + 0;
            double v130_data;
            {
              v130_data = glb_m1[v129_a];
            }
            double v131_data = s0[176];
            double v133_data = ir0[11];
            ir0[11] = (v133_data + (v130_data * v131_data));
            int32_t v140_a = v2_lead + 0;
            double v141_data;
            {
              v141_data = glb_m1[v140_a];
            }
            double v142_data = s0[192];
            double v144_data = ir0[12];
            ir0[12] = (v144_data + (v141_data * v142_data));
            int32_t v151_a = v2_lead + 0;
            double v152_data;
            {
              v152_data = glb_m1[v151_a];
            }
            double v153_data = s0[208];
            double v155_data = ir0[13];
            ir0[13] = (v155_data + (v152_data * v153_data));
            int32_t v162_a = v2_lead + 0;
            double v163_data;
            {
              v163_data = glb_m1[v162_a];
            }
            double v164_data = s0[224];
            double v166_data = ir0[14];
            ir0[14] = (v166_data + (v163_data * v164_data));
            int32_t v173_a = v2_lead + 0;
            double v174_data;
            {
              v174_data = glb_m1[v173_a];
            }
            double v175_data = s0[240];
            double v177_data = ir0[15];
            ir0[15] = (v177_data + (v174_data * v175_data));
            int32_t v187_a = v2_lead + 16;
            double v188_data;
            {
              v188_data = glb_m1[v187_a];
            }
            double v189_data = s0[1];
            double v191_data = ir0[0];
            ir0[0] = (v191_data + (v188_data * v189_data));
            int32_t v198_a = v2_lead + 16;
            double v199_data;
            {
              v199_data = glb_m1[v198_a];
            }
            double v200_data = s0[17];
            double v202_data = ir0[1];
            ir0[1] = (v202_data + (v199_data * v200_data));
            int32_t v209_a = v2_lead + 16;
            double v210_data;
            {
              v210_data = glb_m1[v209_a];
            }
            double v211_data = s0[33];
            double v213_data = ir0[2];
            ir0[2] = (v213_data + (v210_data * v211_data));
            int32_t v220_a = v2_lead + 16;
            double v221_data;
            {
              v221_data = glb_m1[v220_a];
            }
            double v222_data = s0[49];
            double v224_data = ir0[3];
            ir0[3] = (v224_data + (v221_data * v222_data));
            int32_t v231_a = v2_lead + 16;
            double v232_data;
            {
              v232_data = glb_m1[v231_a];
            }
            double v233_data = s0[65];
            double v235_data = ir0[4];
            ir0[4] = (v235_data + (v232_data * v233_data));
            int32_t v242_a = v2_lead + 16;
            double v243_data;
            {
              v243_data = glb_m1[v242_a];
            }
            double v244_data = s0[81];
            double v246_data = ir0[5];
            ir0[5] = (v246_data + (v243_data * v244_data));
            int32_t v253_a = v2_lead + 16;
            double v254_data;
            {
              v254_data = glb_m1[v253_a];
            }
            double v255_data = s0[97];
            double v257_data = ir0[6];
            ir0[6] = (v257_data + (v254_data * v255_data));
            int32_t v264_a = v2_lead + 16;
            double v265_data;
            {
              v265_data = glb_m1[v264_a];
            }
            double v266_data = s0[113];
            double v268_data = ir0[7];
            ir0[7] = (v268_data + (v265_data * v266_data));
            int32_t v275_a = v2_lead + 16;
            double v276_data;
            {
              v276_data = glb_m1[v275_a];
            }
            double v277_data = s0[129];
            double v279_data = ir0[8];
            ir0[8] = (v279_data + (v276_data * v277_data));
            int32_t v286_a = v2_lead + 16;
            double v287_data;
            {
              v287_data = glb_m1[v286_a];
            }
            double v288_data = s0[145];
            double v290_data = ir0[9];
            ir0[9] = (v290_data + (v287_data * v288_data));
            int32_t v297_a = v2_lead + 16;
            double v298_data;
            {
              v298_data = glb_m1[v297_a];
            }
            double v299_data = s0[161];
            double v301_data = ir0[10];
            ir0[10] = (v301_data + (v298_data * v299_data));
            int32_t v308_a = v2_lead + 16;
            double v309_data;
            {
              v309_data = glb_m1[v308_a];
            }
            double v310_data = s0[177];
            double v312_data = ir0[11];
            ir0[11] = (v312_data + (v309_data * v310_data));
            int32_t v319_a = v2_lead + 16;
            double v320_data;
            {
              v320_data = glb_m1[v319_a];
            }
            double v321_data = s0[193];
            double v323_data = ir0[12];
            ir0[12] = (v323_data + (v320_data * v321_data));
            int32_t v330_a = v2_lead + 16;
            double v331_data;
            {
              v331_data = glb_m1[v330_a];
            }
            double v332_data = s0[209];
            double v334_data = ir0[13];
            ir0[13] = (v334_data + (v331_data * v332_data));
            int32_t v341_a = v2_lead + 16;
            double v342_data;
            {
              v342_data = glb_m1[v341_a];
            }
            double v343_data = s0[225];
            double v345_data = ir0[14];
            ir0[14] = (v345_data + (v342_data * v343_data));
            int32_t v352_a = v2_lead + 16;
            double v353_data;
            {
              v353_data = glb_m1[v352_a];
            }
            double v354_data = s0[241];
            double v356_data = ir0[15];
            ir0[15] = (v356_data + (v353_data * v354_data));
            int32_t v366_a = v2_lead + 32;
            double v367_data;
            {
              v367_data = glb_m1[v366_a];
            }
            double v368_data = s0[2];
            double v370_data = ir0[0];
            ir0[0] = (v370_data + (v367_data * v368_data));
            int32_t v377_a = v2_lead + 32;
            double v378_data;
            {
              v378_data = glb_m1[v377_a];
            }
            double v379_data = s0[18];
            double v381_data = ir0[1];
            ir0[1] = (v381_data + (v378_data * v379_data));
            int32_t v388_a = v2_lead + 32;
            double v389_data;
            {
              v389_data = glb_m1[v388_a];
            }
            double v390_data = s0[34];
            double v392_data = ir0[2];
            ir0[2] = (v392_data + (v389_data * v390_data));
            int32_t v399_a = v2_lead + 32;
            double v400_data;
            {
              v400_data = glb_m1[v399_a];
            }
            double v401_data = s0[50];
            double v403_data = ir0[3];
            ir0[3] = (v403_data + (v400_data * v401_data));
            int32_t v410_a = v2_lead + 32;
            double v411_data;
            {
              v411_data = glb_m1[v410_a];
            }
            double v412_data = s0[66];
            double v414_data = ir0[4];
            ir0[4] = (v414_data + (v411_data * v412_data));
            int32_t v421_a = v2_lead + 32;
            double v422_data;
            {
              v422_data = glb_m1[v421_a];
            }
            double v423_data = s0[82];
            double v425_data = ir0[5];
            ir0[5] = (v425_data + (v422_data * v423_data));
            int32_t v432_a = v2_lead + 32;
            double v433_data;
            {
              v433_data = glb_m1[v432_a];
            }
            double v434_data = s0[98];
            double v436_data = ir0[6];
            ir0[6] = (v436_data + (v433_data * v434_data));
            int32_t v443_a = v2_lead + 32;
            double v444_data;
            {
              v444_data = glb_m1[v443_a];
            }
            double v445_data = s0[114];
            double v447_data = ir0[7];
            ir0[7] = (v447_data + (v444_data * v445_data));
            int32_t v454_a = v2_lead + 32;
            double v455_data;
            {
              v455_data = glb_m1[v454_a];
            }
            double v456_data = s0[130];
            double v458_data = ir0[8];
            ir0[8] = (v458_data + (v455_data * v456_data));
            int32_t v465_a = v2_lead + 32;
            double v466_data;
            {
              v466_data = glb_m1[v465_a];
            }
            double v467_data = s0[146];
            double v469_data = ir0[9];
            ir0[9] = (v469_data + (v466_data * v467_data));
            int32_t v476_a = v2_lead + 32;
            double v477_data;
            {
              v477_data = glb_m1[v476_a];
            }
            double v478_data = s0[162];
            double v480_data = ir0[10];
            ir0[10] = (v480_data + (v477_data * v478_data));
            int32_t v487_a = v2_lead + 32;
            double v488_data;
            {
              v488_data = glb_m1[v487_a];
            }
            double v489_data = s0[178];
            double v491_data = ir0[11];
            ir0[11] = (v491_data + (v488_data * v489_data));
            int32_t v498_a = v2_lead + 32;
            double v499_data;
            {
              v499_data = glb_m1[v498_a];
            }
            double v500_data = s0[194];
            double v502_data = ir0[12];
            ir0[12] = (v502_data + (v499_data * v500_data));
            int32_t v509_a = v2_lead + 32;
            double v510_data;
            {
              v510_data = glb_m1[v509_a];
            }
            double v511_data = s0[210];
            double v513_data = ir0[13];
            ir0[13] = (v513_data + (v510_data * v511_data));
            int32_t v520_a = v2_lead + 32;
            double v521_data;
            {
              v521_data = glb_m1[v520_a];
            }
            double v522_data = s0[226];
            double v524_data = ir0[14];
            ir0[14] = (v524_data + (v521_data * v522_data));
            int32_t v531_a = v2_lead + 32;
            double v532_data;
            {
              v532_data = glb_m1[v531_a];
            }
            double v533_data = s0[242];
            double v535_data = ir0[15];
            ir0[15] = (v535_data + (v532_data * v533_data));
            int32_t v545_a = v2_lead + 48;
            double v546_data;
            {
              v546_data = glb_m1[v545_a];
            }
            double v547_data = s0[3];
            double v549_data = ir0[0];
            ir0[0] = (v549_data + (v546_data * v547_data));
            int32_t v556_a = v2_lead + 48;
            double v557_data;
            {
              v557_data = glb_m1[v556_a];
            }
            double v558_data = s0[19];
            double v560_data = ir0[1];
            ir0[1] = (v560_data + (v557_data * v558_data));
            int32_t v567_a = v2_lead + 48;
            double v568_data;
            {
              v568_data = glb_m1[v567_a];
            }
            double v569_data = s0[35];
            double v571_data = ir0[2];
            ir0[2] = (v571_data + (v568_data * v569_data));
            int32_t v578_a = v2_lead + 48;
            double v579_data;
            {
              v579_data = glb_m1[v578_a];
            }
            double v580_data = s0[51];
            double v582_data = ir0[3];
            ir0[3] = (v582_data + (v579_data * v580_data));
            int32_t v589_a = v2_lead + 48;
            double v590_data;
            {
              v590_data = glb_m1[v589_a];
            }
            double v591_data = s0[67];
            double v593_data = ir0[4];
            ir0[4] = (v593_data + (v590_data * v591_data));
            int32_t v600_a = v2_lead + 48;
            double v601_data;
            {
              v601_data = glb_m1[v600_a];
            }
            double v602_data = s0[83];
            double v604_data = ir0[5];
            ir0[5] = (v604_data + (v601_data * v602_data));
            int32_t v611_a = v2_lead + 48;
            double v612_data;
            {
              v612_data = glb_m1[v611_a];
            }
            double v613_data = s0[99];
            double v615_data = ir0[6];
            ir0[6] = (v615_data + (v612_data * v613_data));
            int32_t v622_a = v2_lead + 48;
            double v623_data;
            {
              v623_data = glb_m1[v622_a];
            }
            double v624_data = s0[115];
            double v626_data = ir0[7];
            ir0[7] = (v626_data + (v623_data * v624_data));
            int32_t v633_a = v2_lead + 48;
            double v634_data;
            {
              v634_data = glb_m1[v633_a];
            }
            double v635_data = s0[131];
            double v637_data = ir0[8];
            ir0[8] = (v637_data + (v634_data * v635_data));
            int32_t v644_a = v2_lead + 48;
            double v645_data;
            {
              v645_data = glb_m1[v644_a];
            }
            double v646_data = s0[147];
            double v648_data = ir0[9];
            ir0[9] = (v648_data + (v645_data * v646_data));
            int32_t v655_a = v2_lead + 48;
            double v656_data;
            {
              v656_data = glb_m1[v655_a];
            }
            double v657_data = s0[163];
            double v659_data = ir0[10];
            ir0[10] = (v659_data + (v656_data * v657_data));
            int32_t v666_a = v2_lead + 48;
            double v667_data;
            {
              v667_data = glb_m1[v666_a];
            }
            double v668_data = s0[179];
            double v670_data = ir0[11];
            ir0[11] = (v670_data + (v667_data * v668_data));
            int32_t v677_a = v2_lead + 48;
            double v678_data;
            {
              v678_data = glb_m1[v677_a];
            }
            double v679_data = s0[195];
            double v681_data = ir0[12];
            ir0[12] = (v681_data + (v678_data * v679_data));
            int32_t v688_a = v2_lead + 48;
            double v689_data;
            {
              v689_data = glb_m1[v688_a];
            }
            double v690_data = s0[211];
            double v692_data = ir0[13];
            ir0[13] = (v692_data + (v689_data * v690_data));
            int32_t v699_a = v2_lead + 48;
            double v700_data;
            {
              v700_data = glb_m1[v699_a];
            }
            double v701_data = s0[227];
            double v703_data = ir0[14];
            ir0[14] = (v703_data + (v700_data * v701_data));
            int32_t v710_a = v2_lead + 48;
            double v711_data;
            {
              v711_data = glb_m1[v710_a];
            }
            double v712_data = s0[243];
            double v714_data = ir0[15];
            ir0[15] = (v714_data + (v711_data * v712_data));
            int32_t v724_a = v2_lead + 64;
            double v725_data;
            {
              v725_data = glb_m1[v724_a];
            }
            double v726_data = s0[4];
            double v728_data = ir0[0];
            ir0[0] = (v728_data + (v725_data * v726_data));
            int32_t v735_a = v2_lead + 64;
            double v736_data;
            {
              v736_data = glb_m1[v735_a];
            }
            double v737_data = s0[20];
            double v739_data = ir0[1];
            ir0[1] = (v739_data + (v736_data * v737_data));
            int32_t v746_a = v2_lead + 64;
            double v747_data;
            {
              v747_data = glb_m1[v746_a];
            }
            double v748_data = s0[36];
            double v750_data = ir0[2];
            ir0[2] = (v750_data + (v747_data * v748_data));
            int32_t v757_a = v2_lead + 64;
            double v758_data;
            {
              v758_data = glb_m1[v757_a];
            }
            double v759_data = s0[52];
            double v761_data = ir0[3];
            ir0[3] = (v761_data + (v758_data * v759_data));
            int32_t v768_a = v2_lead + 64;
            double v769_data;
            {
              v769_data = glb_m1[v768_a];
            }
            double v770_data = s0[68];
            double v772_data = ir0[4];
            ir0[4] = (v772_data + (v769_data * v770_data));
            int32_t v779_a = v2_lead + 64;
            double v780_data;
            {
              v780_data = glb_m1[v779_a];
            }
            double v781_data = s0[84];
            double v783_data = ir0[5];
            ir0[5] = (v783_data + (v780_data * v781_data));
            int32_t v790_a = v2_lead + 64;
            double v791_data;
            {
              v791_data = glb_m1[v790_a];
            }
            double v792_data = s0[100];
            double v794_data = ir0[6];
            ir0[6] = (v794_data + (v791_data * v792_data));
            int32_t v801_a = v2_lead + 64;
            double v802_data;
            {
              v802_data = glb_m1[v801_a];
            }
            double v803_data = s0[116];
            double v805_data = ir0[7];
            ir0[7] = (v805_data + (v802_data * v803_data));
            int32_t v812_a = v2_lead + 64;
            double v813_data;
            {
              v813_data = glb_m1[v812_a];
            }
            double v814_data = s0[132];
            double v816_data = ir0[8];
            ir0[8] = (v816_data + (v813_data * v814_data));
            int32_t v823_a = v2_lead + 64;
            double v824_data;
            {
              v824_data = glb_m1[v823_a];
            }
            double v825_data = s0[148];
            double v827_data = ir0[9];
            ir0[9] = (v827_data + (v824_data * v825_data));
            int32_t v834_a = v2_lead + 64;
            double v835_data;
            {
              v835_data = glb_m1[v834_a];
            }
            double v836_data = s0[164];
            double v838_data = ir0[10];
            ir0[10] = (v838_data + (v835_data * v836_data));
            int32_t v845_a = v2_lead + 64;
            double v846_data;
            {
              v846_data = glb_m1[v845_a];
            }
            double v847_data = s0[180];
            double v849_data = ir0[11];
            ir0[11] = (v849_data + (v846_data * v847_data));
            int32_t v856_a = v2_lead + 64;
            double v857_data;
            {
              v857_data = glb_m1[v856_a];
            }
            double v858_data = s0[196];
            double v860_data = ir0[12];
            ir0[12] = (v860_data + (v857_data * v858_data));
            int32_t v867_a = v2_lead + 64;
            double v868_data;
            {
              v868_data = glb_m1[v867_a];
            }
            double v869_data = s0[212];
            double v871_data = ir0[13];
            ir0[13] = (v871_data + (v868_data * v869_data));
            int32_t v878_a = v2_lead + 64;
            double v879_data;
            {
              v879_data = glb_m1[v878_a];
            }
            double v880_data = s0[228];
            double v882_data = ir0[14];
            ir0[14] = (v882_data + (v879_data * v880_data));
            int32_t v889_a = v2_lead + 64;
            double v890_data;
            {
              v890_data = glb_m1[v889_a];
            }
            double v891_data = s0[244];
            double v893_data = ir0[15];
            ir0[15] = (v893_data + (v890_data * v891_data));
            int32_t v903_a = v2_lead + 80;
            double v904_data;
            {
              v904_data = glb_m1[v903_a];
            }
            double v905_data = s0[5];
            double v907_data = ir0[0];
            ir0[0] = (v907_data + (v904_data * v905_data));
            int32_t v914_a = v2_lead + 80;
            double v915_data;
            {
              v915_data = glb_m1[v914_a];
            }
            double v916_data = s0[21];
            double v918_data = ir0[1];
            ir0[1] = (v918_data + (v915_data * v916_data));
            int32_t v925_a = v2_lead + 80;
            double v926_data;
            {
              v926_data = glb_m1[v925_a];
            }
            double v927_data = s0[37];
            double v929_data = ir0[2];
            ir0[2] = (v929_data + (v926_data * v927_data));
            int32_t v936_a = v2_lead + 80;
            double v937_data;
            {
              v937_data = glb_m1[v936_a];
            }
            double v938_data = s0[53];
            double v940_data = ir0[3];
            ir0[3] = (v940_data + (v937_data * v938_data));
            int32_t v947_a = v2_lead + 80;
            double v948_data;
            {
              v948_data = glb_m1[v947_a];
            }
            double v949_data = s0[69];
            double v951_data = ir0[4];
            ir0[4] = (v951_data + (v948_data * v949_data));
            int32_t v958_a = v2_lead + 80;
            double v959_data;
            {
              v959_data = glb_m1[v958_a];
            }
            double v960_data = s0[85];
            double v962_data = ir0[5];
            ir0[5] = (v962_data + (v959_data * v960_data));
            int32_t v969_a = v2_lead + 80;
            double v970_data;
            {
              v970_data = glb_m1[v969_a];
            }
            double v971_data = s0[101];
            double v973_data = ir0[6];
            ir0[6] = (v973_data + (v970_data * v971_data));
            int32_t v980_a = v2_lead + 80;
            double v981_data;
            {
              v981_data = glb_m1[v980_a];
            }
            double v982_data = s0[117];
            double v984_data = ir0[7];
            ir0[7] = (v984_data + (v981_data * v982_data));
            int32_t v991_a = v2_lead + 80;
            double v992_data;
            {
              v992_data = glb_m1[v991_a];
            }
            double v993_data = s0[133];
            double v995_data = ir0[8];
            ir0[8] = (v995_data + (v992_data * v993_data));
            int32_t v1002_a = v2_lead + 80;
            double v1003_data;
            {
              v1003_data = glb_m1[v1002_a];
            }
            double v1004_data = s0[149];
            double v1006_data = ir0[9];
            ir0[9] = (v1006_data + (v1003_data * v1004_data));
            int32_t v1013_a = v2_lead + 80;
            double v1014_data;
            {
              v1014_data = glb_m1[v1013_a];
            }
            double v1015_data = s0[165];
            double v1017_data = ir0[10];
            ir0[10] = (v1017_data + (v1014_data * v1015_data));
            int32_t v1024_a = v2_lead + 80;
            double v1025_data;
            {
              v1025_data = glb_m1[v1024_a];
            }
            double v1026_data = s0[181];
            double v1028_data = ir0[11];
            ir0[11] = (v1028_data + (v1025_data * v1026_data));
            int32_t v1035_a = v2_lead + 80;
            double v1036_data;
            {
              v1036_data = glb_m1[v1035_a];
            }
            double v1037_data = s0[197];
            double v1039_data = ir0[12];
            ir0[12] = (v1039_data + (v1036_data * v1037_data));
            int32_t v1046_a = v2_lead + 80;
            double v1047_data;
            {
              v1047_data = glb_m1[v1046_a];
            }
            double v1048_data = s0[213];
            double v1050_data = ir0[13];
            ir0[13] = (v1050_data + (v1047_data * v1048_data));
            int32_t v1057_a = v2_lead + 80;
            double v1058_data;
            {
              v1058_data = glb_m1[v1057_a];
            }
            double v1059_data = s0[229];
            double v1061_data = ir0[14];
            ir0[14] = (v1061_data + (v1058_data * v1059_data));
            int32_t v1068_a = v2_lead + 80;
            double v1069_data;
            {
              v1069_data = glb_m1[v1068_a];
            }
            double v1070_data = s0[245];
            double v1072_data = ir0[15];
            ir0[15] = (v1072_data + (v1069_data * v1070_data));
            int32_t v1082_a = v2_lead + 96;
            double v1083_data;
            {
              v1083_data = glb_m1[v1082_a];
            }
            double v1084_data = s0[6];
            double v1086_data = ir0[0];
            ir0[0] = (v1086_data + (v1083_data * v1084_data));
            int32_t v1093_a = v2_lead + 96;
            double v1094_data;
            {
              v1094_data = glb_m1[v1093_a];
            }
            double v1095_data = s0[22];
            double v1097_data = ir0[1];
            ir0[1] = (v1097_data + (v1094_data * v1095_data));
            int32_t v1104_a = v2_lead + 96;
            double v1105_data;
            {
              v1105_data = glb_m1[v1104_a];
            }
            double v1106_data = s0[38];
            double v1108_data = ir0[2];
            ir0[2] = (v1108_data + (v1105_data * v1106_data));
            int32_t v1115_a = v2_lead + 96;
            double v1116_data;
            {
              v1116_data = glb_m1[v1115_a];
            }
            double v1117_data = s0[54];
            double v1119_data = ir0[3];
            ir0[3] = (v1119_data + (v1116_data * v1117_data));
            int32_t v1126_a = v2_lead + 96;
            double v1127_data;
            {
              v1127_data = glb_m1[v1126_a];
            }
            double v1128_data = s0[70];
            double v1130_data = ir0[4];
            ir0[4] = (v1130_data + (v1127_data * v1128_data));
            int32_t v1137_a = v2_lead + 96;
            double v1138_data;
            {
              v1138_data = glb_m1[v1137_a];
            }
            double v1139_data = s0[86];
            double v1141_data = ir0[5];
            ir0[5] = (v1141_data + (v1138_data * v1139_data));
            int32_t v1148_a = v2_lead + 96;
            double v1149_data;
            {
              v1149_data = glb_m1[v1148_a];
            }
            double v1150_data = s0[102];
            double v1152_data = ir0[6];
            ir0[6] = (v1152_data + (v1149_data * v1150_data));
            int32_t v1159_a = v2_lead + 96;
            double v1160_data;
            {
              v1160_data = glb_m1[v1159_a];
            }
            double v1161_data = s0[118];
            double v1163_data = ir0[7];
            ir0[7] = (v1163_data + (v1160_data * v1161_data));
            int32_t v1170_a = v2_lead + 96;
            double v1171_data;
            {
              v1171_data = glb_m1[v1170_a];
            }
            double v1172_data = s0[134];
            double v1174_data = ir0[8];
            ir0[8] = (v1174_data + (v1171_data * v1172_data));
            int32_t v1181_a = v2_lead + 96;
            double v1182_data;
            {
              v1182_data = glb_m1[v1181_a];
            }
            double v1183_data = s0[150];
            double v1185_data = ir0[9];
            ir0[9] = (v1185_data + (v1182_data * v1183_data));
            int32_t v1192_a = v2_lead + 96;
            double v1193_data;
            {
              v1193_data = glb_m1[v1192_a];
            }
            double v1194_data = s0[166];
            double v1196_data = ir0[10];
            ir0[10] = (v1196_data + (v1193_data * v1194_data));
            int32_t v1203_a = v2_lead + 96;
            double v1204_data;
            {
              v1204_data = glb_m1[v1203_a];
            }
            double v1205_data = s0[182];
            double v1207_data = ir0[11];
            ir0[11] = (v1207_data + (v1204_data * v1205_data));
            int32_t v1214_a = v2_lead + 96;
            double v1215_data;
            {
              v1215_data = glb_m1[v1214_a];
            }
            double v1216_data = s0[198];
            double v1218_data = ir0[12];
            ir0[12] = (v1218_data + (v1215_data * v1216_data));
            int32_t v1225_a = v2_lead + 96;
            double v1226_data;
            {
              v1226_data = glb_m1[v1225_a];
            }
            double v1227_data = s0[214];
            double v1229_data = ir0[13];
            ir0[13] = (v1229_data + (v1226_data * v1227_data));
            int32_t v1236_a = v2_lead + 96;
            double v1237_data;
            {
              v1237_data = glb_m1[v1236_a];
            }
            double v1238_data = s0[230];
            double v1240_data = ir0[14];
            ir0[14] = (v1240_data + (v1237_data * v1238_data));
            int32_t v1247_a = v2_lead + 96;
            double v1248_data;
            {
              v1248_data = glb_m1[v1247_a];
            }
            double v1249_data = s0[246];
            double v1251_data = ir0[15];
            ir0[15] = (v1251_data + (v1248_data * v1249_data));
            int32_t v1261_a = v2_lead + 112;
            double v1262_data;
            {
              v1262_data = glb_m1[v1261_a];
            }
            double v1263_data = s0[7];
            double v1265_data = ir0[0];
            ir0[0] = (v1265_data + (v1262_data * v1263_data));
            int32_t v1272_a = v2_lead + 112;
            double v1273_data;
            {
              v1273_data = glb_m1[v1272_a];
            }
            double v1274_data = s0[23];
            double v1276_data = ir0[1];
            ir0[1] = (v1276_data + (v1273_data * v1274_data));
            int32_t v1283_a = v2_lead + 112;
            double v1284_data;
            {
              v1284_data = glb_m1[v1283_a];
            }
            double v1285_data = s0[39];
            double v1287_data = ir0[2];
            ir0[2] = (v1287_data + (v1284_data * v1285_data));
            int32_t v1294_a = v2_lead + 112;
            double v1295_data;
            {
              v1295_data = glb_m1[v1294_a];
            }
            double v1296_data = s0[55];
            double v1298_data = ir0[3];
            ir0[3] = (v1298_data + (v1295_data * v1296_data));
            int32_t v1305_a = v2_lead + 112;
            double v1306_data;
            {
              v1306_data = glb_m1[v1305_a];
            }
            double v1307_data = s0[71];
            double v1309_data = ir0[4];
            ir0[4] = (v1309_data + (v1306_data * v1307_data));
            int32_t v1316_a = v2_lead + 112;
            double v1317_data;
            {
              v1317_data = glb_m1[v1316_a];
            }
            double v1318_data = s0[87];
            double v1320_data = ir0[5];
            ir0[5] = (v1320_data + (v1317_data * v1318_data));
            int32_t v1327_a = v2_lead + 112;
            double v1328_data;
            {
              v1328_data = glb_m1[v1327_a];
            }
            double v1329_data = s0[103];
            double v1331_data = ir0[6];
            ir0[6] = (v1331_data + (v1328_data * v1329_data));
            int32_t v1338_a = v2_lead + 112;
            double v1339_data;
            {
              v1339_data = glb_m1[v1338_a];
            }
            double v1340_data = s0[119];
            double v1342_data = ir0[7];
            ir0[7] = (v1342_data + (v1339_data * v1340_data));
            int32_t v1349_a = v2_lead + 112;
            double v1350_data;
            {
              v1350_data = glb_m1[v1349_a];
            }
            double v1351_data = s0[135];
            double v1353_data = ir0[8];
            ir0[8] = (v1353_data + (v1350_data * v1351_data));
            int32_t v1360_a = v2_lead + 112;
            double v1361_data;
            {
              v1361_data = glb_m1[v1360_a];
            }
            double v1362_data = s0[151];
            double v1364_data = ir0[9];
            ir0[9] = (v1364_data + (v1361_data * v1362_data));
            int32_t v1371_a = v2_lead + 112;
            double v1372_data;
            {
              v1372_data = glb_m1[v1371_a];
            }
            double v1373_data = s0[167];
            double v1375_data = ir0[10];
            ir0[10] = (v1375_data + (v1372_data * v1373_data));
            int32_t v1382_a = v2_lead + 112;
            double v1383_data;
            {
              v1383_data = glb_m1[v1382_a];
            }
            double v1384_data = s0[183];
            double v1386_data = ir0[11];
            ir0[11] = (v1386_data + (v1383_data * v1384_data));
            int32_t v1393_a = v2_lead + 112;
            double v1394_data;
            {
              v1394_data = glb_m1[v1393_a];
            }
            double v1395_data = s0[199];
            double v1397_data = ir0[12];
            ir0[12] = (v1397_data + (v1394_data * v1395_data));
            int32_t v1404_a = v2_lead + 112;
            double v1405_data;
            {
              v1405_data = glb_m1[v1404_a];
            }
            double v1406_data = s0[215];
            double v1408_data = ir0[13];
            ir0[13] = (v1408_data + (v1405_data * v1406_data));
            int32_t v1415_a = v2_lead + 112;
            double v1416_data;
            {
              v1416_data = glb_m1[v1415_a];
            }
            double v1417_data = s0[231];
            double v1419_data = ir0[14];
            ir0[14] = (v1419_data + (v1416_data * v1417_data));
            int32_t v1426_a = v2_lead + 112;
            double v1427_data;
            {
              v1427_data = glb_m1[v1426_a];
            }
            double v1428_data = s0[247];
            double v1430_data = ir0[15];
            ir0[15] = (v1430_data + (v1427_data * v1428_data));
            int32_t v1440_a = v2_lead + 128;
            double v1441_data;
            {
              v1441_data = glb_m1[v1440_a];
            }
            double v1442_data = s0[8];
            double v1444_data = ir0[0];
            ir0[0] = (v1444_data + (v1441_data * v1442_data));
            int32_t v1451_a = v2_lead + 128;
            double v1452_data;
            {
              v1452_data = glb_m1[v1451_a];
            }
            double v1453_data = s0[24];
            double v1455_data = ir0[1];
            ir0[1] = (v1455_data + (v1452_data * v1453_data));
            int32_t v1462_a = v2_lead + 128;
            double v1463_data;
            {
              v1463_data = glb_m1[v1462_a];
            }
            double v1464_data = s0[40];
            double v1466_data = ir0[2];
            ir0[2] = (v1466_data + (v1463_data * v1464_data));
            int32_t v1473_a = v2_lead + 128;
            double v1474_data;
            {
              v1474_data = glb_m1[v1473_a];
            }
            double v1475_data = s0[56];
            double v1477_data = ir0[3];
            ir0[3] = (v1477_data + (v1474_data * v1475_data));
            int32_t v1484_a = v2_lead + 128;
            double v1485_data;
            {
              v1485_data = glb_m1[v1484_a];
            }
            double v1486_data = s0[72];
            double v1488_data = ir0[4];
            ir0[4] = (v1488_data + (v1485_data * v1486_data));
            int32_t v1495_a = v2_lead + 128;
            double v1496_data;
            {
              v1496_data = glb_m1[v1495_a];
            }
            double v1497_data = s0[88];
            double v1499_data = ir0[5];
            ir0[5] = (v1499_data + (v1496_data * v1497_data));
            int32_t v1506_a = v2_lead + 128;
            double v1507_data;
            {
              v1507_data = glb_m1[v1506_a];
            }
            double v1508_data = s0[104];
            double v1510_data = ir0[6];
            ir0[6] = (v1510_data + (v1507_data * v1508_data));
            int32_t v1517_a = v2_lead + 128;
            double v1518_data;
            {
              v1518_data = glb_m1[v1517_a];
            }
            double v1519_data = s0[120];
            double v1521_data = ir0[7];
            ir0[7] = (v1521_data + (v1518_data * v1519_data));
            int32_t v1528_a = v2_lead + 128;
            double v1529_data;
            {
              v1529_data = glb_m1[v1528_a];
            }
            double v1530_data = s0[136];
            double v1532_data = ir0[8];
            ir0[8] = (v1532_data + (v1529_data * v1530_data));
            int32_t v1539_a = v2_lead + 128;
            double v1540_data;
            {
              v1540_data = glb_m1[v1539_a];
            }
            double v1541_data = s0[152];
            double v1543_data = ir0[9];
            ir0[9] = (v1543_data + (v1540_data * v1541_data));
            int32_t v1550_a = v2_lead + 128;
            double v1551_data;
            {
              v1551_data = glb_m1[v1550_a];
            }
            double v1552_data = s0[168];
            double v1554_data = ir0[10];
            ir0[10] = (v1554_data + (v1551_data * v1552_data));
            int32_t v1561_a = v2_lead + 128;
            double v1562_data;
            {
              v1562_data = glb_m1[v1561_a];
            }
            double v1563_data = s0[184];
            double v1565_data = ir0[11];
            ir0[11] = (v1565_data + (v1562_data * v1563_data));
            int32_t v1572_a = v2_lead + 128;
            double v1573_data;
            {
              v1573_data = glb_m1[v1572_a];
            }
            double v1574_data = s0[200];
            double v1576_data = ir0[12];
            ir0[12] = (v1576_data + (v1573_data * v1574_data));
            int32_t v1583_a = v2_lead + 128;
            double v1584_data;
            {
              v1584_data = glb_m1[v1583_a];
            }
            double v1585_data = s0[216];
            double v1587_data = ir0[13];
            ir0[13] = (v1587_data + (v1584_data * v1585_data));
            int32_t v1594_a = v2_lead + 128;
            double v1595_data;
            {
              v1595_data = glb_m1[v1594_a];
            }
            double v1596_data = s0[232];
            double v1598_data = ir0[14];
            ir0[14] = (v1598_data + (v1595_data * v1596_data));
            int32_t v1605_a = v2_lead + 128;
            double v1606_data;
            {
              v1606_data = glb_m1[v1605_a];
            }
            double v1607_data = s0[248];
            double v1609_data = ir0[15];
            ir0[15] = (v1609_data + (v1606_data * v1607_data));
            int32_t v1619_a = v2_lead + 144;
            double v1620_data;
            {
              v1620_data = glb_m1[v1619_a];
            }
            double v1621_data = s0[9];
            double v1623_data = ir0[0];
            ir0[0] = (v1623_data + (v1620_data * v1621_data));
            int32_t v1630_a = v2_lead + 144;
            double v1631_data;
            {
              v1631_data = glb_m1[v1630_a];
            }
            double v1632_data = s0[25];
            double v1634_data = ir0[1];
            ir0[1] = (v1634_data + (v1631_data * v1632_data));
            int32_t v1641_a = v2_lead + 144;
            double v1642_data;
            {
              v1642_data = glb_m1[v1641_a];
            }
            double v1643_data = s0[41];
            double v1645_data = ir0[2];
            ir0[2] = (v1645_data + (v1642_data * v1643_data));
            int32_t v1652_a = v2_lead + 144;
            double v1653_data;
            {
              v1653_data = glb_m1[v1652_a];
            }
            double v1654_data = s0[57];
            double v1656_data = ir0[3];
            ir0[3] = (v1656_data + (v1653_data * v1654_data));
            int32_t v1663_a = v2_lead + 144;
            double v1664_data;
            {
              v1664_data = glb_m1[v1663_a];
            }
            double v1665_data = s0[73];
            double v1667_data = ir0[4];
            ir0[4] = (v1667_data + (v1664_data * v1665_data));
            int32_t v1674_a = v2_lead + 144;
            double v1675_data;
            {
              v1675_data = glb_m1[v1674_a];
            }
            double v1676_data = s0[89];
            double v1678_data = ir0[5];
            ir0[5] = (v1678_data + (v1675_data * v1676_data));
            int32_t v1685_a = v2_lead + 144;
            double v1686_data;
            {
              v1686_data = glb_m1[v1685_a];
            }
            double v1687_data = s0[105];
            double v1689_data = ir0[6];
            ir0[6] = (v1689_data + (v1686_data * v1687_data));
            int32_t v1696_a = v2_lead + 144;
            double v1697_data;
            {
              v1697_data = glb_m1[v1696_a];
            }
            double v1698_data = s0[121];
            double v1700_data = ir0[7];
            ir0[7] = (v1700_data + (v1697_data * v1698_data));
            int32_t v1707_a = v2_lead + 144;
            double v1708_data;
            {
              v1708_data = glb_m1[v1707_a];
            }
            double v1709_data = s0[137];
            double v1711_data = ir0[8];
            ir0[8] = (v1711_data + (v1708_data * v1709_data));
            int32_t v1718_a = v2_lead + 144;
            double v1719_data;
            {
              v1719_data = glb_m1[v1718_a];
            }
            double v1720_data = s0[153];
            double v1722_data = ir0[9];
            ir0[9] = (v1722_data + (v1719_data * v1720_data));
            int32_t v1729_a = v2_lead + 144;
            double v1730_data;
            {
              v1730_data = glb_m1[v1729_a];
            }
            double v1731_data = s0[169];
            double v1733_data = ir0[10];
            ir0[10] = (v1733_data + (v1730_data * v1731_data));
            int32_t v1740_a = v2_lead + 144;
            double v1741_data;
            {
              v1741_data = glb_m1[v1740_a];
            }
            double v1742_data = s0[185];
            double v1744_data = ir0[11];
            ir0[11] = (v1744_data + (v1741_data * v1742_data));
            int32_t v1751_a = v2_lead + 144;
            double v1752_data;
            {
              v1752_data = glb_m1[v1751_a];
            }
            double v1753_data = s0[201];
            double v1755_data = ir0[12];
            ir0[12] = (v1755_data + (v1752_data * v1753_data));
            int32_t v1762_a = v2_lead + 144;
            double v1763_data;
            {
              v1763_data = glb_m1[v1762_a];
            }
            double v1764_data = s0[217];
            double v1766_data = ir0[13];
            ir0[13] = (v1766_data + (v1763_data * v1764_data));
            int32_t v1773_a = v2_lead + 144;
            double v1774_data;
            {
              v1774_data = glb_m1[v1773_a];
            }
            double v1775_data = s0[233];
            double v1777_data = ir0[14];
            ir0[14] = (v1777_data + (v1774_data * v1775_data));
            int32_t v1784_a = v2_lead + 144;
            double v1785_data;
            {
              v1785_data = glb_m1[v1784_a];
            }
            double v1786_data = s0[249];
            double v1788_data = ir0[15];
            ir0[15] = (v1788_data + (v1785_data * v1786_data));
            int32_t v1798_a = v2_lead + 160;
            double v1799_data;
            {
              v1799_data = glb_m1[v1798_a];
            }
            double v1800_data = s0[10];
            double v1802_data = ir0[0];
            ir0[0] = (v1802_data + (v1799_data * v1800_data));
            int32_t v1809_a = v2_lead + 160;
            double v1810_data;
            {
              v1810_data = glb_m1[v1809_a];
            }
            double v1811_data = s0[26];
            double v1813_data = ir0[1];
            ir0[1] = (v1813_data + (v1810_data * v1811_data));
            int32_t v1820_a = v2_lead + 160;
            double v1821_data;
            {
              v1821_data = glb_m1[v1820_a];
            }
            double v1822_data = s0[42];
            double v1824_data = ir0[2];
            ir0[2] = (v1824_data + (v1821_data * v1822_data));
            int32_t v1831_a = v2_lead + 160;
            double v1832_data;
            {
              v1832_data = glb_m1[v1831_a];
            }
            double v1833_data = s0[58];
            double v1835_data = ir0[3];
            ir0[3] = (v1835_data + (v1832_data * v1833_data));
            int32_t v1842_a = v2_lead + 160;
            double v1843_data;
            {
              v1843_data = glb_m1[v1842_a];
            }
            double v1844_data = s0[74];
            double v1846_data = ir0[4];
            ir0[4] = (v1846_data + (v1843_data * v1844_data));
            int32_t v1853_a = v2_lead + 160;
            double v1854_data;
            {
              v1854_data = glb_m1[v1853_a];
            }
            double v1855_data = s0[90];
            double v1857_data = ir0[5];
            ir0[5] = (v1857_data + (v1854_data * v1855_data));
            int32_t v1864_a = v2_lead + 160;
            double v1865_data;
            {
              v1865_data = glb_m1[v1864_a];
            }
            double v1866_data = s0[106];
            double v1868_data = ir0[6];
            ir0[6] = (v1868_data + (v1865_data * v1866_data));
            int32_t v1875_a = v2_lead + 160;
            double v1876_data;
            {
              v1876_data = glb_m1[v1875_a];
            }
            double v1877_data = s0[122];
            double v1879_data = ir0[7];
            ir0[7] = (v1879_data + (v1876_data * v1877_data));
            int32_t v1886_a = v2_lead + 160;
            double v1887_data;
            {
              v1887_data = glb_m1[v1886_a];
            }
            double v1888_data = s0[138];
            double v1890_data = ir0[8];
            ir0[8] = (v1890_data + (v1887_data * v1888_data));
            int32_t v1897_a = v2_lead + 160;
            double v1898_data;
            {
              v1898_data = glb_m1[v1897_a];
            }
            double v1899_data = s0[154];
            double v1901_data = ir0[9];
            ir0[9] = (v1901_data + (v1898_data * v1899_data));
            int32_t v1908_a = v2_lead + 160;
            double v1909_data;
            {
              v1909_data = glb_m1[v1908_a];
            }
            double v1910_data = s0[170];
            double v1912_data = ir0[10];
            ir0[10] = (v1912_data + (v1909_data * v1910_data));
            int32_t v1919_a = v2_lead + 160;
            double v1920_data;
            {
              v1920_data = glb_m1[v1919_a];
            }
            double v1921_data = s0[186];
            double v1923_data = ir0[11];
            ir0[11] = (v1923_data + (v1920_data * v1921_data));
            int32_t v1930_a = v2_lead + 160;
            double v1931_data;
            {
              v1931_data = glb_m1[v1930_a];
            }
            double v1932_data = s0[202];
            double v1934_data = ir0[12];
            ir0[12] = (v1934_data + (v1931_data * v1932_data));
            int32_t v1941_a = v2_lead + 160;
            double v1942_data;
            {
              v1942_data = glb_m1[v1941_a];
            }
            double v1943_data = s0[218];
            double v1945_data = ir0[13];
            ir0[13] = (v1945_data + (v1942_data * v1943_data));
            int32_t v1952_a = v2_lead + 160;
            double v1953_data;
            {
              v1953_data = glb_m1[v1952_a];
            }
            double v1954_data = s0[234];
            double v1956_data = ir0[14];
            ir0[14] = (v1956_data + (v1953_data * v1954_data));
            int32_t v1963_a = v2_lead + 160;
            double v1964_data;
            {
              v1964_data = glb_m1[v1963_a];
            }
            double v1965_data = s0[250];
            double v1967_data = ir0[15];
            ir0[15] = (v1967_data + (v1964_data * v1965_data));
            int32_t v1977_a = v2_lead + 176;
            double v1978_data;
            {
              v1978_data = glb_m1[v1977_a];
            }
            double v1979_data = s0[11];
            double v1981_data = ir0[0];
            ir0[0] = (v1981_data + (v1978_data * v1979_data));
            int32_t v1988_a = v2_lead + 176;
            double v1989_data;
            {
              v1989_data = glb_m1[v1988_a];
            }
            double v1990_data = s0[27];
            double v1992_data = ir0[1];
            ir0[1] = (v1992_data + (v1989_data * v1990_data));
            int32_t v1999_a = v2_lead + 176;
            double v2000_data;
            {
              v2000_data = glb_m1[v1999_a];
            }
            double v2001_data = s0[43];
            double v2003_data = ir0[2];
            ir0[2] = (v2003_data + (v2000_data * v2001_data));
            int32_t v2010_a = v2_lead + 176;
            double v2011_data;
            {
              v2011_data = glb_m1[v2010_a];
            }
            double v2012_data = s0[59];
            double v2014_data = ir0[3];
            ir0[3] = (v2014_data + (v2011_data * v2012_data));
            int32_t v2021_a = v2_lead + 176;
            double v2022_data;
            {
              v2022_data = glb_m1[v2021_a];
            }
            double v2023_data = s0[75];
            double v2025_data = ir0[4];
            ir0[4] = (v2025_data + (v2022_data * v2023_data));
            int32_t v2032_a = v2_lead + 176;
            double v2033_data;
            {
              v2033_data = glb_m1[v2032_a];
            }
            double v2034_data = s0[91];
            double v2036_data = ir0[5];
            ir0[5] = (v2036_data + (v2033_data * v2034_data));
            int32_t v2043_a = v2_lead + 176;
            double v2044_data;
            {
              v2044_data = glb_m1[v2043_a];
            }
            double v2045_data = s0[107];
            double v2047_data = ir0[6];
            ir0[6] = (v2047_data + (v2044_data * v2045_data));
            int32_t v2054_a = v2_lead + 176;
            double v2055_data;
            {
              v2055_data = glb_m1[v2054_a];
            }
            double v2056_data = s0[123];
            double v2058_data = ir0[7];
            ir0[7] = (v2058_data + (v2055_data * v2056_data));
            int32_t v2065_a = v2_lead + 176;
            double v2066_data;
            {
              v2066_data = glb_m1[v2065_a];
            }
            double v2067_data = s0[139];
            double v2069_data = ir0[8];
            ir0[8] = (v2069_data + (v2066_data * v2067_data));
            int32_t v2076_a = v2_lead + 176;
            double v2077_data;
            {
              v2077_data = glb_m1[v2076_a];
            }
            double v2078_data = s0[155];
            double v2080_data = ir0[9];
            ir0[9] = (v2080_data + (v2077_data * v2078_data));
            int32_t v2087_a = v2_lead + 176;
            double v2088_data;
            {
              v2088_data = glb_m1[v2087_a];
            }
            double v2089_data = s0[171];
            double v2091_data = ir0[10];
            ir0[10] = (v2091_data + (v2088_data * v2089_data));
            int32_t v2098_a = v2_lead + 176;
            double v2099_data;
            {
              v2099_data = glb_m1[v2098_a];
            }
            double v2100_data = s0[187];
            double v2102_data = ir0[11];
            ir0[11] = (v2102_data + (v2099_data * v2100_data));
            int32_t v2109_a = v2_lead + 176;
            double v2110_data;
            {
              v2110_data = glb_m1[v2109_a];
            }
            double v2111_data = s0[203];
            double v2113_data = ir0[12];
            ir0[12] = (v2113_data + (v2110_data * v2111_data));
            int32_t v2120_a = v2_lead + 176;
            double v2121_data;
            {
              v2121_data = glb_m1[v2120_a];
            }
            double v2122_data = s0[219];
            double v2124_data = ir0[13];
            ir0[13] = (v2124_data + (v2121_data * v2122_data));
            int32_t v2131_a = v2_lead + 176;
            double v2132_data;
            {
              v2132_data = glb_m1[v2131_a];
            }
            double v2133_data = s0[235];
            double v2135_data = ir0[14];
            ir0[14] = (v2135_data + (v2132_data * v2133_data));
            int32_t v2142_a = v2_lead + 176;
            double v2143_data;
            {
              v2143_data = glb_m1[v2142_a];
            }
            double v2144_data = s0[251];
            double v2146_data = ir0[15];
            ir0[15] = (v2146_data + (v2143_data * v2144_data));
            int32_t v2156_a = v2_lead + 192;
            double v2157_data;
            {
              v2157_data = glb_m1[v2156_a];
            }
            double v2158_data = s0[12];
            double v2160_data = ir0[0];
            ir0[0] = (v2160_data + (v2157_data * v2158_data));
            int32_t v2167_a = v2_lead + 192;
            double v2168_data;
            {
              v2168_data = glb_m1[v2167_a];
            }
            double v2169_data = s0[28];
            double v2171_data = ir0[1];
            ir0[1] = (v2171_data + (v2168_data * v2169_data));
            int32_t v2178_a = v2_lead + 192;
            double v2179_data;
            {
              v2179_data = glb_m1[v2178_a];
            }
            double v2180_data = s0[44];
            double v2182_data = ir0[2];
            ir0[2] = (v2182_data + (v2179_data * v2180_data));
            int32_t v2189_a = v2_lead + 192;
            double v2190_data;
            {
              v2190_data = glb_m1[v2189_a];
            }
            double v2191_data = s0[60];
            double v2193_data = ir0[3];
            ir0[3] = (v2193_data + (v2190_data * v2191_data));
            int32_t v2200_a = v2_lead + 192;
            double v2201_data;
            {
              v2201_data = glb_m1[v2200_a];
            }
            double v2202_data = s0[76];
            double v2204_data = ir0[4];
            ir0[4] = (v2204_data + (v2201_data * v2202_data));
            int32_t v2211_a = v2_lead + 192;
            double v2212_data;
            {
              v2212_data = glb_m1[v2211_a];
            }
            double v2213_data = s0[92];
            double v2215_data = ir0[5];
            ir0[5] = (v2215_data + (v2212_data * v2213_data));
            int32_t v2222_a = v2_lead + 192;
            double v2223_data;
            {
              v2223_data = glb_m1[v2222_a];
            }
            double v2224_data = s0[108];
            double v2226_data = ir0[6];
            ir0[6] = (v2226_data + (v2223_data * v2224_data));
            int32_t v2233_a = v2_lead + 192;
            double v2234_data;
            {
              v2234_data = glb_m1[v2233_a];
            }
            double v2235_data = s0[124];
            double v2237_data = ir0[7];
            ir0[7] = (v2237_data + (v2234_data * v2235_data));
            int32_t v2244_a = v2_lead + 192;
            double v2245_data;
            {
              v2245_data = glb_m1[v2244_a];
            }
            double v2246_data = s0[140];
            double v2248_data = ir0[8];
            ir0[8] = (v2248_data + (v2245_data * v2246_data));
            int32_t v2255_a = v2_lead + 192;
            double v2256_data;
            {
              v2256_data = glb_m1[v2255_a];
            }
            double v2257_data = s0[156];
            double v2259_data = ir0[9];
            ir0[9] = (v2259_data + (v2256_data * v2257_data));
            int32_t v2266_a = v2_lead + 192;
            double v2267_data;
            {
              v2267_data = glb_m1[v2266_a];
            }
            double v2268_data = s0[172];
            double v2270_data = ir0[10];
            ir0[10] = (v2270_data + (v2267_data * v2268_data));
            int32_t v2277_a = v2_lead + 192;
            double v2278_data;
            {
              v2278_data = glb_m1[v2277_a];
            }
            double v2279_data = s0[188];
            double v2281_data = ir0[11];
            ir0[11] = (v2281_data + (v2278_data * v2279_data));
            int32_t v2288_a = v2_lead + 192;
            double v2289_data;
            {
              v2289_data = glb_m1[v2288_a];
            }
            double v2290_data = s0[204];
            double v2292_data = ir0[12];
            ir0[12] = (v2292_data + (v2289_data * v2290_data));
            int32_t v2299_a = v2_lead + 192;
            double v2300_data;
            {
              v2300_data = glb_m1[v2299_a];
            }
            double v2301_data = s0[220];
            double v2303_data = ir0[13];
            ir0[13] = (v2303_data + (v2300_data * v2301_data));
            int32_t v2310_a = v2_lead + 192;
            double v2311_data;
            {
              v2311_data = glb_m1[v2310_a];
            }
            double v2312_data = s0[236];
            double v2314_data = ir0[14];
            ir0[14] = (v2314_data + (v2311_data * v2312_data));
            int32_t v2321_a = v2_lead + 192;
            double v2322_data;
            {
              v2322_data = glb_m1[v2321_a];
            }
            double v2323_data = s0[252];
            double v2325_data = ir0[15];
            ir0[15] = (v2325_data + (v2322_data * v2323_data));
            int32_t v2335_a = v2_lead + 208;
            double v2336_data;
            {
              v2336_data = glb_m1[v2335_a];
            }
            double v2337_data = s0[13];
            double v2339_data = ir0[0];
            ir0[0] = (v2339_data + (v2336_data * v2337_data));
            int32_t v2346_a = v2_lead + 208;
            double v2347_data;
            {
              v2347_data = glb_m1[v2346_a];
            }
            double v2348_data = s0[29];
            double v2350_data = ir0[1];
            ir0[1] = (v2350_data + (v2347_data * v2348_data));
            int32_t v2357_a = v2_lead + 208;
            double v2358_data;
            {
              v2358_data = glb_m1[v2357_a];
            }
            double v2359_data = s0[45];
            double v2361_data = ir0[2];
            ir0[2] = (v2361_data + (v2358_data * v2359_data));
            int32_t v2368_a = v2_lead + 208;
            double v2369_data;
            {
              v2369_data = glb_m1[v2368_a];
            }
            double v2370_data = s0[61];
            double v2372_data = ir0[3];
            ir0[3] = (v2372_data + (v2369_data * v2370_data));
            int32_t v2379_a = v2_lead + 208;
            double v2380_data;
            {
              v2380_data = glb_m1[v2379_a];
            }
            double v2381_data = s0[77];
            double v2383_data = ir0[4];
            ir0[4] = (v2383_data + (v2380_data * v2381_data));
            int32_t v2390_a = v2_lead + 208;
            double v2391_data;
            {
              v2391_data = glb_m1[v2390_a];
            }
            double v2392_data = s0[93];
            double v2394_data = ir0[5];
            ir0[5] = (v2394_data + (v2391_data * v2392_data));
            int32_t v2401_a = v2_lead + 208;
            double v2402_data;
            {
              v2402_data = glb_m1[v2401_a];
            }
            double v2403_data = s0[109];
            double v2405_data = ir0[6];
            ir0[6] = (v2405_data + (v2402_data * v2403_data));
            int32_t v2412_a = v2_lead + 208;
            double v2413_data;
            {
              v2413_data = glb_m1[v2412_a];
            }
            double v2414_data = s0[125];
            double v2416_data = ir0[7];
            ir0[7] = (v2416_data + (v2413_data * v2414_data));
            int32_t v2423_a = v2_lead + 208;
            double v2424_data;
            {
              v2424_data = glb_m1[v2423_a];
            }
            double v2425_data = s0[141];
            double v2427_data = ir0[8];
            ir0[8] = (v2427_data + (v2424_data * v2425_data));
            int32_t v2434_a = v2_lead + 208;
            double v2435_data;
            {
              v2435_data = glb_m1[v2434_a];
            }
            double v2436_data = s0[157];
            double v2438_data = ir0[9];
            ir0[9] = (v2438_data + (v2435_data * v2436_data));
            int32_t v2445_a = v2_lead + 208;
            double v2446_data;
            {
              v2446_data = glb_m1[v2445_a];
            }
            double v2447_data = s0[173];
            double v2449_data = ir0[10];
            ir0[10] = (v2449_data + (v2446_data * v2447_data));
            int32_t v2456_a = v2_lead + 208;
            double v2457_data;
            {
              v2457_data = glb_m1[v2456_a];
            }
            double v2458_data = s0[189];
            double v2460_data = ir0[11];
            ir0[11] = (v2460_data + (v2457_data * v2458_data));
            int32_t v2467_a = v2_lead + 208;
            double v2468_data;
            {
              v2468_data = glb_m1[v2467_a];
            }
            double v2469_data = s0[205];
            double v2471_data = ir0[12];
            ir0[12] = (v2471_data + (v2468_data * v2469_data));
            int32_t v2478_a = v2_lead + 208;
            double v2479_data;
            {
              v2479_data = glb_m1[v2478_a];
            }
            double v2480_data = s0[221];
            double v2482_data = ir0[13];
            ir0[13] = (v2482_data + (v2479_data * v2480_data));
            int32_t v2489_a = v2_lead + 208;
            double v2490_data;
            {
              v2490_data = glb_m1[v2489_a];
            }
            double v2491_data = s0[237];
            double v2493_data = ir0[14];
            ir0[14] = (v2493_data + (v2490_data * v2491_data));
            int32_t v2500_a = v2_lead + 208;
            double v2501_data;
            {
              v2501_data = glb_m1[v2500_a];
            }
            double v2502_data = s0[253];
            double v2504_data = ir0[15];
            ir0[15] = (v2504_data + (v2501_data * v2502_data));
            int32_t v2514_a = v2_lead + 224;
            double v2515_data;
            {
              v2515_data = glb_m1[v2514_a];
            }
            double v2516_data = s0[14];
            double v2518_data = ir0[0];
            ir0[0] = (v2518_data + (v2515_data * v2516_data));
            int32_t v2525_a = v2_lead + 224;
            double v2526_data;
            {
              v2526_data = glb_m1[v2525_a];
            }
            double v2527_data = s0[30];
            double v2529_data = ir0[1];
            ir0[1] = (v2529_data + (v2526_data * v2527_data));
            int32_t v2536_a = v2_lead + 224;
            double v2537_data;
            {
              v2537_data = glb_m1[v2536_a];
            }
            double v2538_data = s0[46];
            double v2540_data = ir0[2];
            ir0[2] = (v2540_data + (v2537_data * v2538_data));
            int32_t v2547_a = v2_lead + 224;
            double v2548_data;
            {
              v2548_data = glb_m1[v2547_a];
            }
            double v2549_data = s0[62];
            double v2551_data = ir0[3];
            ir0[3] = (v2551_data + (v2548_data * v2549_data));
            int32_t v2558_a = v2_lead + 224;
            double v2559_data;
            {
              v2559_data = glb_m1[v2558_a];
            }
            double v2560_data = s0[78];
            double v2562_data = ir0[4];
            ir0[4] = (v2562_data + (v2559_data * v2560_data));
            int32_t v2569_a = v2_lead + 224;
            double v2570_data;
            {
              v2570_data = glb_m1[v2569_a];
            }
            double v2571_data = s0[94];
            double v2573_data = ir0[5];
            ir0[5] = (v2573_data + (v2570_data * v2571_data));
            int32_t v2580_a = v2_lead + 224;
            double v2581_data;
            {
              v2581_data = glb_m1[v2580_a];
            }
            double v2582_data = s0[110];
            double v2584_data = ir0[6];
            ir0[6] = (v2584_data + (v2581_data * v2582_data));
            int32_t v2591_a = v2_lead + 224;
            double v2592_data;
            {
              v2592_data = glb_m1[v2591_a];
            }
            double v2593_data = s0[126];
            double v2595_data = ir0[7];
            ir0[7] = (v2595_data + (v2592_data * v2593_data));
            int32_t v2602_a = v2_lead + 224;
            double v2603_data;
            {
              v2603_data = glb_m1[v2602_a];
            }
            double v2604_data = s0[142];
            double v2606_data = ir0[8];
            ir0[8] = (v2606_data + (v2603_data * v2604_data));
            int32_t v2613_a = v2_lead + 224;
            double v2614_data;
            {
              v2614_data = glb_m1[v2613_a];
            }
            double v2615_data = s0[158];
            double v2617_data = ir0[9];
            ir0[9] = (v2617_data + (v2614_data * v2615_data));
            int32_t v2624_a = v2_lead + 224;
            double v2625_data;
            {
              v2625_data = glb_m1[v2624_a];
            }
            double v2626_data = s0[174];
            double v2628_data = ir0[10];
            ir0[10] = (v2628_data + (v2625_data * v2626_data));
            int32_t v2635_a = v2_lead + 224;
            double v2636_data;
            {
              v2636_data = glb_m1[v2635_a];
            }
            double v2637_data = s0[190];
            double v2639_data = ir0[11];
            ir0[11] = (v2639_data + (v2636_data * v2637_data));
            int32_t v2646_a = v2_lead + 224;
            double v2647_data;
            {
              v2647_data = glb_m1[v2646_a];
            }
            double v2648_data = s0[206];
            double v2650_data = ir0[12];
            ir0[12] = (v2650_data + (v2647_data * v2648_data));
            int32_t v2657_a = v2_lead + 224;
            double v2658_data;
            {
              v2658_data = glb_m1[v2657_a];
            }
            double v2659_data = s0[222];
            double v2661_data = ir0[13];
            ir0[13] = (v2661_data + (v2658_data * v2659_data));
            int32_t v2668_a = v2_lead + 224;
            double v2669_data;
            {
              v2669_data = glb_m1[v2668_a];
            }
            double v2670_data = s0[238];
            double v2672_data = ir0[14];
            ir0[14] = (v2672_data + (v2669_data * v2670_data));
            int32_t v2679_a = v2_lead + 224;
            double v2680_data;
            {
              v2680_data = glb_m1[v2679_a];
            }
            double v2681_data = s0[254];
            double v2683_data = ir0[15];
            ir0[15] = (v2683_data + (v2680_data * v2681_data));
            int32_t v2693_a = v2_lead + 240;
            double v2694_data;
            {
              v2694_data = glb_m1[v2693_a];
            }
            double v2695_data = s0[15];
            double v2697_data = ir0[0];
            ir0[0] = (v2697_data + (v2694_data * v2695_data));
            int32_t v2704_a = v2_lead + 240;
            double v2705_data;
            {
              v2705_data = glb_m1[v2704_a];
            }
            double v2706_data = s0[31];
            double v2708_data = ir0[1];
            ir0[1] = (v2708_data + (v2705_data * v2706_data));
            int32_t v2715_a = v2_lead + 240;
            double v2716_data;
            {
              v2716_data = glb_m1[v2715_a];
            }
            double v2717_data = s0[47];
            double v2719_data = ir0[2];
            ir0[2] = (v2719_data + (v2716_data * v2717_data));
            int32_t v2726_a = v2_lead + 240;
            double v2727_data;
            {
              v2727_data = glb_m1[v2726_a];
            }
            double v2728_data = s0[63];
            double v2730_data = ir0[3];
            ir0[3] = (v2730_data + (v2727_data * v2728_data));
            int32_t v2737_a = v2_lead + 240;
            double v2738_data;
            {
              v2738_data = glb_m1[v2737_a];
            }
            double v2739_data = s0[79];
            double v2741_data = ir0[4];
            ir0[4] = (v2741_data + (v2738_data * v2739_data));
            int32_t v2748_a = v2_lead + 240;
            double v2749_data;
            {
              v2749_data = glb_m1[v2748_a];
            }
            double v2750_data = s0[95];
            double v2752_data = ir0[5];
            ir0[5] = (v2752_data + (v2749_data * v2750_data));
            int32_t v2759_a = v2_lead + 240;
            double v2760_data;
            {
              v2760_data = glb_m1[v2759_a];
            }
            double v2761_data = s0[111];
            double v2763_data = ir0[6];
            ir0[6] = (v2763_data + (v2760_data * v2761_data));
            int32_t v2770_a = v2_lead + 240;
            double v2771_data;
            {
              v2771_data = glb_m1[v2770_a];
            }
            double v2772_data = s0[127];
            double v2774_data = ir0[7];
            ir0[7] = (v2774_data + (v2771_data * v2772_data));
            int32_t v2781_a = v2_lead + 240;
            double v2782_data;
            {
              v2782_data = glb_m1[v2781_a];
            }
            double v2783_data = s0[143];
            double v2785_data = ir0[8];
            ir0[8] = (v2785_data + (v2782_data * v2783_data));
            int32_t v2792_a = v2_lead + 240;
            double v2793_data;
            {
              v2793_data = glb_m1[v2792_a];
            }
            double v2794_data = s0[159];
            double v2796_data = ir0[9];
            ir0[9] = (v2796_data + (v2793_data * v2794_data));
            int32_t v2803_a = v2_lead + 240;
            double v2804_data;
            {
              v2804_data = glb_m1[v2803_a];
            }
            double v2805_data = s0[175];
            double v2807_data = ir0[10];
            ir0[10] = (v2807_data + (v2804_data * v2805_data));
            int32_t v2814_a = v2_lead + 240;
            double v2815_data;
            {
              v2815_data = glb_m1[v2814_a];
            }
            double v2816_data = s0[191];
            double v2818_data = ir0[11];
            ir0[11] = (v2818_data + (v2815_data * v2816_data));
            int32_t v2825_a = v2_lead + 240;
            double v2826_data;
            {
              v2826_data = glb_m1[v2825_a];
            }
            double v2827_data = s0[207];
            double v2829_data = ir0[12];
            ir0[12] = (v2829_data + (v2826_data * v2827_data));
            int32_t v2836_a = v2_lead + 240;
            double v2837_data;
            {
              v2837_data = glb_m1[v2836_a];
            }
            double v2838_data = s0[223];
            double v2840_data = ir0[13];
            ir0[13] = (v2840_data + (v2837_data * v2838_data));
            int32_t v2847_a = v2_lead + 240;
            double v2848_data;
            {
              v2848_data = glb_m1[v2847_a];
            }
            double v2849_data = s0[239];
            double v2851_data = ir0[14];
            ir0[14] = (v2851_data + (v2848_data * v2849_data));
            int32_t v2858_a = v2_lead + 240;
            double v2859_data;
            {
              v2859_data = glb_m1[v2858_a];
            }
            double v2860_data = s0[255];
            double v2862_data = ir0[15];
            ir0[15] = (v2862_data + (v2859_data * v2860_data));
            #pragma unroll
            for (int32_t v2867_n0 = 0; v2867_n0 < 1; ++v2867_n0) {
              #pragma unroll
              for (int32_t v2868_n1 = 0; v2868_n1 < 16; ++v2868_n1) {
                int32_t v2869_a = v2867_n0 + v2868_n1;
                double v2870_data = ir0[v2869_a];
                int32_t v2871_a = v2867_n0 + v2868_n1;
                r0[v2871_a] = v2870_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r0);
          int32_t v2874_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v2875_i0 = 0; v2875_i0 < 1; ++v2875_i0) {
            int32_t v2883_lead = v2874_lead + (v2875_i0 * 16);
            #pragma unroll
            for (int32_t v2876_i1 = 0; v2876_i1 < 16; ++v2876_i1) {
              int32_t v2877_a = v2875_i0 + v2876_i1;
              double v2878_data = r0[v2877_a];
              int32_t v2885_a = v2883_lead + (v2876_i1 * 16);
              glb_m0[v2885_a] = v2878_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

