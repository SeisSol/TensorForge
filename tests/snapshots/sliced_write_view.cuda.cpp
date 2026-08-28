// === base name ===
kernel_939857c66e

// === header ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_939857c66e, block.x * block.y * block.z, 1536 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_939857c66e, cudaFuncAttributeMaxDynamicSharedMemorySize, 1536 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_939857c66e<<<grid,block,1536 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×13(32×13) {0..32}×{0..13} strided
    // m1 32×13(32×13) {0..32}×{0..13} strided
    // m2 13×13(13×13) {0..13}×{0..13} strided
    // m3 32×13(32×13) {0..32}×{0..13} strided
    // m4 13×13(13×13) {0..13}×{0..13} strided
    // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
    // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v9_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v10_i0 = 0; v10_i0 < 1; ++v10_i0) {
            int32_t v15_lead = v10_i0 * 32;
            int32_t v16_lead = v9_lead + v15_lead;
            int32_t v23_lead = v9_lead + v15_lead;
            #pragma unroll
            for (int32_t v11_i1 = 10; v11_i1 < 13; ++v11_i1) {
              int32_t v17_a = v11_i1 * 32;
              int32_t v18_a = v16_lead + v17_a;
              float v26_data = __ldcg(&glb_m1[(v23_lead + v17_a)]);
              r0[(v10_i0 + (v11_i1 - 10))] = v26_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
              __pipeline_commit();
            }
            if (threadIdx.x < 9) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[1]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v37_data = r0[0];
          float v38_data = s0[114];
          float v40_data = ir1[0];
          ir1[0] = (v40_data + (v37_data * v38_data));
          float v45_data = r0[1];
          float v46_data = s0[115];
          float v48_data = ir1[0];
          ir1[0] = (v48_data + (v45_data * v46_data));
          float v53_data = r0[2];
          float v54_data = s0[116];
          float v56_data = ir1[0];
          ir1[0] = (v56_data + (v53_data * v54_data));
          #pragma unroll
          for (int32_t v61_n0 = 0; v61_n0 < 1; ++v61_n0) {
            #pragma unroll
            for (int32_t v62_n1 = 0; v62_n1 < 1; ++v62_n1) {
              int32_t v63_a = v61_n0 + v62_n1;
              int32_t v64_a = v61_n0 + v62_n1;
              float v65_data = ir1[v64_a];
              r1[v64_a] = v65_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v70_i0 = 0; v70_i0 < 1; ++v70_i0) {
            int32_t v79_lead = v9_lead + (v70_i0 * 32);
            #pragma unroll
            for (int32_t v71_i1 = 0; v71_i1 < 1; ++v71_i1) {
              int32_t v72_a = v70_i0 + v71_i1;
              float v74_data = r1[(v70_i0 + v71_i1)];
              glb_m0[(v79_lead + ((v71_i1 + 8) * 32))] = v74_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v87_i0 = 0; v87_i0 < 1; ++v87_i0) {
            int32_t v92_lead = v87_i0 * 32;
            int32_t v93_lead = v9_lead + v92_lead;
            int32_t v100_lead = v9_lead + v92_lead;
            #pragma unroll
            for (int32_t v88_i1 = 0; v88_i1 < 13; ++v88_i1) {
              int32_t v94_a = v88_i1 * 32;
              int32_t v95_a = v93_lead + v94_a;
              float v103_data = glb_m0[(v100_lead + v94_a)];
              r2[(v87_i0 + v88_i1)] = v103_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          {
            // s1 = load{g>s}(glb_m4[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 5; i += 1) {
              __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 32], 4);
              __pipeline_commit();
            }
            if (threadIdx.x < 9) {
              __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 160], &glb_m4[0 + 0 + 1 * threadIdx.x + 160], 4);
              __pipeline_commit();
            }
          }
          // wait(r2 = load{g>r}(glb_m0););
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[13]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v113_data = r2[0];
          float v114_data = s1[0];
          float v116_data = ir3[0];
          ir3[0] = (v116_data + (v113_data * v114_data));
          float v119_data = s1[13];
          float v121_data = ir3[1];
          ir3[1] = (v121_data + (v113_data * v119_data));
          float v124_data = s1[26];
          float v126_data = ir3[2];
          ir3[2] = (v126_data + (v113_data * v124_data));
          float v129_data = s1[39];
          float v131_data = ir3[3];
          ir3[3] = (v131_data + (v113_data * v129_data));
          float v134_data = s1[52];
          float v136_data = ir3[4];
          ir3[4] = (v136_data + (v113_data * v134_data));
          float v139_data = s1[65];
          float v141_data = ir3[5];
          ir3[5] = (v141_data + (v113_data * v139_data));
          float v144_data = s1[78];
          float v146_data = ir3[6];
          ir3[6] = (v146_data + (v113_data * v144_data));
          float v149_data = s1[91];
          float v151_data = ir3[7];
          ir3[7] = (v151_data + (v113_data * v149_data));
          float v154_data = s1[104];
          float v156_data = ir3[8];
          ir3[8] = (v156_data + (v113_data * v154_data));
          float v159_data = s1[117];
          float v161_data = ir3[9];
          ir3[9] = (v161_data + (v113_data * v159_data));
          float v164_data = s1[130];
          float v166_data = ir3[10];
          ir3[10] = (v166_data + (v113_data * v164_data));
          float v169_data = s1[143];
          float v171_data = ir3[11];
          ir3[11] = (v171_data + (v113_data * v169_data));
          float v174_data = s1[156];
          float v176_data = ir3[12];
          ir3[12] = (v176_data + (v113_data * v174_data));
          float v181_data = r2[1];
          float v182_data = s1[1];
          float v184_data = ir3[0];
          ir3[0] = (v184_data + (v181_data * v182_data));
          float v187_data = s1[14];
          float v189_data = ir3[1];
          ir3[1] = (v189_data + (v181_data * v187_data));
          float v192_data = s1[27];
          float v194_data = ir3[2];
          ir3[2] = (v194_data + (v181_data * v192_data));
          float v197_data = s1[40];
          float v199_data = ir3[3];
          ir3[3] = (v199_data + (v181_data * v197_data));
          float v202_data = s1[53];
          float v204_data = ir3[4];
          ir3[4] = (v204_data + (v181_data * v202_data));
          float v207_data = s1[66];
          float v209_data = ir3[5];
          ir3[5] = (v209_data + (v181_data * v207_data));
          float v212_data = s1[79];
          float v214_data = ir3[6];
          ir3[6] = (v214_data + (v181_data * v212_data));
          float v217_data = s1[92];
          float v219_data = ir3[7];
          ir3[7] = (v219_data + (v181_data * v217_data));
          float v222_data = s1[105];
          float v224_data = ir3[8];
          ir3[8] = (v224_data + (v181_data * v222_data));
          float v227_data = s1[118];
          float v229_data = ir3[9];
          ir3[9] = (v229_data + (v181_data * v227_data));
          float v232_data = s1[131];
          float v234_data = ir3[10];
          ir3[10] = (v234_data + (v181_data * v232_data));
          float v237_data = s1[144];
          float v239_data = ir3[11];
          ir3[11] = (v239_data + (v181_data * v237_data));
          float v242_data = s1[157];
          float v244_data = ir3[12];
          ir3[12] = (v244_data + (v181_data * v242_data));
          float v249_data = r2[2];
          float v250_data = s1[2];
          float v252_data = ir3[0];
          ir3[0] = (v252_data + (v249_data * v250_data));
          float v255_data = s1[15];
          float v257_data = ir3[1];
          ir3[1] = (v257_data + (v249_data * v255_data));
          float v260_data = s1[28];
          float v262_data = ir3[2];
          ir3[2] = (v262_data + (v249_data * v260_data));
          float v265_data = s1[41];
          float v267_data = ir3[3];
          ir3[3] = (v267_data + (v249_data * v265_data));
          float v270_data = s1[54];
          float v272_data = ir3[4];
          ir3[4] = (v272_data + (v249_data * v270_data));
          float v275_data = s1[67];
          float v277_data = ir3[5];
          ir3[5] = (v277_data + (v249_data * v275_data));
          float v280_data = s1[80];
          float v282_data = ir3[6];
          ir3[6] = (v282_data + (v249_data * v280_data));
          float v285_data = s1[93];
          float v287_data = ir3[7];
          ir3[7] = (v287_data + (v249_data * v285_data));
          float v290_data = s1[106];
          float v292_data = ir3[8];
          ir3[8] = (v292_data + (v249_data * v290_data));
          float v295_data = s1[119];
          float v297_data = ir3[9];
          ir3[9] = (v297_data + (v249_data * v295_data));
          float v300_data = s1[132];
          float v302_data = ir3[10];
          ir3[10] = (v302_data + (v249_data * v300_data));
          float v305_data = s1[145];
          float v307_data = ir3[11];
          ir3[11] = (v307_data + (v249_data * v305_data));
          float v310_data = s1[158];
          float v312_data = ir3[12];
          ir3[12] = (v312_data + (v249_data * v310_data));
          float v317_data = r2[3];
          float v318_data = s1[3];
          float v320_data = ir3[0];
          ir3[0] = (v320_data + (v317_data * v318_data));
          float v323_data = s1[16];
          float v325_data = ir3[1];
          ir3[1] = (v325_data + (v317_data * v323_data));
          float v328_data = s1[29];
          float v330_data = ir3[2];
          ir3[2] = (v330_data + (v317_data * v328_data));
          float v333_data = s1[42];
          float v335_data = ir3[3];
          ir3[3] = (v335_data + (v317_data * v333_data));
          float v338_data = s1[55];
          float v340_data = ir3[4];
          ir3[4] = (v340_data + (v317_data * v338_data));
          float v343_data = s1[68];
          float v345_data = ir3[5];
          ir3[5] = (v345_data + (v317_data * v343_data));
          float v348_data = s1[81];
          float v350_data = ir3[6];
          ir3[6] = (v350_data + (v317_data * v348_data));
          float v353_data = s1[94];
          float v355_data = ir3[7];
          ir3[7] = (v355_data + (v317_data * v353_data));
          float v358_data = s1[107];
          float v360_data = ir3[8];
          ir3[8] = (v360_data + (v317_data * v358_data));
          float v363_data = s1[120];
          float v365_data = ir3[9];
          ir3[9] = (v365_data + (v317_data * v363_data));
          float v368_data = s1[133];
          float v370_data = ir3[10];
          ir3[10] = (v370_data + (v317_data * v368_data));
          float v373_data = s1[146];
          float v375_data = ir3[11];
          ir3[11] = (v375_data + (v317_data * v373_data));
          float v378_data = s1[159];
          float v380_data = ir3[12];
          ir3[12] = (v380_data + (v317_data * v378_data));
          float v385_data = r2[4];
          float v386_data = s1[4];
          float v388_data = ir3[0];
          ir3[0] = (v388_data + (v385_data * v386_data));
          float v391_data = s1[17];
          float v393_data = ir3[1];
          ir3[1] = (v393_data + (v385_data * v391_data));
          float v396_data = s1[30];
          float v398_data = ir3[2];
          ir3[2] = (v398_data + (v385_data * v396_data));
          float v401_data = s1[43];
          float v403_data = ir3[3];
          ir3[3] = (v403_data + (v385_data * v401_data));
          float v406_data = s1[56];
          float v408_data = ir3[4];
          ir3[4] = (v408_data + (v385_data * v406_data));
          float v411_data = s1[69];
          float v413_data = ir3[5];
          ir3[5] = (v413_data + (v385_data * v411_data));
          float v416_data = s1[82];
          float v418_data = ir3[6];
          ir3[6] = (v418_data + (v385_data * v416_data));
          float v421_data = s1[95];
          float v423_data = ir3[7];
          ir3[7] = (v423_data + (v385_data * v421_data));
          float v426_data = s1[108];
          float v428_data = ir3[8];
          ir3[8] = (v428_data + (v385_data * v426_data));
          float v431_data = s1[121];
          float v433_data = ir3[9];
          ir3[9] = (v433_data + (v385_data * v431_data));
          float v436_data = s1[134];
          float v438_data = ir3[10];
          ir3[10] = (v438_data + (v385_data * v436_data));
          float v441_data = s1[147];
          float v443_data = ir3[11];
          ir3[11] = (v443_data + (v385_data * v441_data));
          float v446_data = s1[160];
          float v448_data = ir3[12];
          ir3[12] = (v448_data + (v385_data * v446_data));
          float v453_data = r2[5];
          float v454_data = s1[5];
          float v456_data = ir3[0];
          ir3[0] = (v456_data + (v453_data * v454_data));
          float v459_data = s1[18];
          float v461_data = ir3[1];
          ir3[1] = (v461_data + (v453_data * v459_data));
          float v464_data = s1[31];
          float v466_data = ir3[2];
          ir3[2] = (v466_data + (v453_data * v464_data));
          float v469_data = s1[44];
          float v471_data = ir3[3];
          ir3[3] = (v471_data + (v453_data * v469_data));
          float v474_data = s1[57];
          float v476_data = ir3[4];
          ir3[4] = (v476_data + (v453_data * v474_data));
          float v479_data = s1[70];
          float v481_data = ir3[5];
          ir3[5] = (v481_data + (v453_data * v479_data));
          float v484_data = s1[83];
          float v486_data = ir3[6];
          ir3[6] = (v486_data + (v453_data * v484_data));
          float v489_data = s1[96];
          float v491_data = ir3[7];
          ir3[7] = (v491_data + (v453_data * v489_data));
          float v494_data = s1[109];
          float v496_data = ir3[8];
          ir3[8] = (v496_data + (v453_data * v494_data));
          float v499_data = s1[122];
          float v501_data = ir3[9];
          ir3[9] = (v501_data + (v453_data * v499_data));
          float v504_data = s1[135];
          float v506_data = ir3[10];
          ir3[10] = (v506_data + (v453_data * v504_data));
          float v509_data = s1[148];
          float v511_data = ir3[11];
          ir3[11] = (v511_data + (v453_data * v509_data));
          float v514_data = s1[161];
          float v516_data = ir3[12];
          ir3[12] = (v516_data + (v453_data * v514_data));
          float v521_data = r2[6];
          float v522_data = s1[6];
          float v524_data = ir3[0];
          ir3[0] = (v524_data + (v521_data * v522_data));
          float v527_data = s1[19];
          float v529_data = ir3[1];
          ir3[1] = (v529_data + (v521_data * v527_data));
          float v532_data = s1[32];
          float v534_data = ir3[2];
          ir3[2] = (v534_data + (v521_data * v532_data));
          float v537_data = s1[45];
          float v539_data = ir3[3];
          ir3[3] = (v539_data + (v521_data * v537_data));
          float v542_data = s1[58];
          float v544_data = ir3[4];
          ir3[4] = (v544_data + (v521_data * v542_data));
          float v547_data = s1[71];
          float v549_data = ir3[5];
          ir3[5] = (v549_data + (v521_data * v547_data));
          float v552_data = s1[84];
          float v554_data = ir3[6];
          ir3[6] = (v554_data + (v521_data * v552_data));
          float v557_data = s1[97];
          float v559_data = ir3[7];
          ir3[7] = (v559_data + (v521_data * v557_data));
          float v562_data = s1[110];
          float v564_data = ir3[8];
          ir3[8] = (v564_data + (v521_data * v562_data));
          float v567_data = s1[123];
          float v569_data = ir3[9];
          ir3[9] = (v569_data + (v521_data * v567_data));
          float v572_data = s1[136];
          float v574_data = ir3[10];
          ir3[10] = (v574_data + (v521_data * v572_data));
          float v577_data = s1[149];
          float v579_data = ir3[11];
          ir3[11] = (v579_data + (v521_data * v577_data));
          float v582_data = s1[162];
          float v584_data = ir3[12];
          ir3[12] = (v584_data + (v521_data * v582_data));
          float v589_data = r2[7];
          float v590_data = s1[7];
          float v592_data = ir3[0];
          ir3[0] = (v592_data + (v589_data * v590_data));
          float v595_data = s1[20];
          float v597_data = ir3[1];
          ir3[1] = (v597_data + (v589_data * v595_data));
          float v600_data = s1[33];
          float v602_data = ir3[2];
          ir3[2] = (v602_data + (v589_data * v600_data));
          float v605_data = s1[46];
          float v607_data = ir3[3];
          ir3[3] = (v607_data + (v589_data * v605_data));
          float v610_data = s1[59];
          float v612_data = ir3[4];
          ir3[4] = (v612_data + (v589_data * v610_data));
          float v615_data = s1[72];
          float v617_data = ir3[5];
          ir3[5] = (v617_data + (v589_data * v615_data));
          float v620_data = s1[85];
          float v622_data = ir3[6];
          ir3[6] = (v622_data + (v589_data * v620_data));
          float v625_data = s1[98];
          float v627_data = ir3[7];
          ir3[7] = (v627_data + (v589_data * v625_data));
          float v630_data = s1[111];
          float v632_data = ir3[8];
          ir3[8] = (v632_data + (v589_data * v630_data));
          float v635_data = s1[124];
          float v637_data = ir3[9];
          ir3[9] = (v637_data + (v589_data * v635_data));
          float v640_data = s1[137];
          float v642_data = ir3[10];
          ir3[10] = (v642_data + (v589_data * v640_data));
          float v645_data = s1[150];
          float v647_data = ir3[11];
          ir3[11] = (v647_data + (v589_data * v645_data));
          float v650_data = s1[163];
          float v652_data = ir3[12];
          ir3[12] = (v652_data + (v589_data * v650_data));
          float v657_data = r2[8];
          float v658_data = s1[8];
          float v660_data = ir3[0];
          ir3[0] = (v660_data + (v657_data * v658_data));
          float v663_data = s1[21];
          float v665_data = ir3[1];
          ir3[1] = (v665_data + (v657_data * v663_data));
          float v668_data = s1[34];
          float v670_data = ir3[2];
          ir3[2] = (v670_data + (v657_data * v668_data));
          float v673_data = s1[47];
          float v675_data = ir3[3];
          ir3[3] = (v675_data + (v657_data * v673_data));
          float v678_data = s1[60];
          float v680_data = ir3[4];
          ir3[4] = (v680_data + (v657_data * v678_data));
          float v683_data = s1[73];
          float v685_data = ir3[5];
          ir3[5] = (v685_data + (v657_data * v683_data));
          float v688_data = s1[86];
          float v690_data = ir3[6];
          ir3[6] = (v690_data + (v657_data * v688_data));
          float v693_data = s1[99];
          float v695_data = ir3[7];
          ir3[7] = (v695_data + (v657_data * v693_data));
          float v698_data = s1[112];
          float v700_data = ir3[8];
          ir3[8] = (v700_data + (v657_data * v698_data));
          float v703_data = s1[125];
          float v705_data = ir3[9];
          ir3[9] = (v705_data + (v657_data * v703_data));
          float v708_data = s1[138];
          float v710_data = ir3[10];
          ir3[10] = (v710_data + (v657_data * v708_data));
          float v713_data = s1[151];
          float v715_data = ir3[11];
          ir3[11] = (v715_data + (v657_data * v713_data));
          float v718_data = s1[164];
          float v720_data = ir3[12];
          ir3[12] = (v720_data + (v657_data * v718_data));
          float v725_data = r2[9];
          float v726_data = s1[9];
          float v728_data = ir3[0];
          ir3[0] = (v728_data + (v725_data * v726_data));
          float v731_data = s1[22];
          float v733_data = ir3[1];
          ir3[1] = (v733_data + (v725_data * v731_data));
          float v736_data = s1[35];
          float v738_data = ir3[2];
          ir3[2] = (v738_data + (v725_data * v736_data));
          float v741_data = s1[48];
          float v743_data = ir3[3];
          ir3[3] = (v743_data + (v725_data * v741_data));
          float v746_data = s1[61];
          float v748_data = ir3[4];
          ir3[4] = (v748_data + (v725_data * v746_data));
          float v751_data = s1[74];
          float v753_data = ir3[5];
          ir3[5] = (v753_data + (v725_data * v751_data));
          float v756_data = s1[87];
          float v758_data = ir3[6];
          ir3[6] = (v758_data + (v725_data * v756_data));
          float v761_data = s1[100];
          float v763_data = ir3[7];
          ir3[7] = (v763_data + (v725_data * v761_data));
          float v766_data = s1[113];
          float v768_data = ir3[8];
          ir3[8] = (v768_data + (v725_data * v766_data));
          float v771_data = s1[126];
          float v773_data = ir3[9];
          ir3[9] = (v773_data + (v725_data * v771_data));
          float v776_data = s1[139];
          float v778_data = ir3[10];
          ir3[10] = (v778_data + (v725_data * v776_data));
          float v781_data = s1[152];
          float v783_data = ir3[11];
          ir3[11] = (v783_data + (v725_data * v781_data));
          float v786_data = s1[165];
          float v788_data = ir3[12];
          ir3[12] = (v788_data + (v725_data * v786_data));
          float v793_data = r2[10];
          float v794_data = s1[10];
          float v796_data = ir3[0];
          ir3[0] = (v796_data + (v793_data * v794_data));
          float v799_data = s1[23];
          float v801_data = ir3[1];
          ir3[1] = (v801_data + (v793_data * v799_data));
          float v804_data = s1[36];
          float v806_data = ir3[2];
          ir3[2] = (v806_data + (v793_data * v804_data));
          float v809_data = s1[49];
          float v811_data = ir3[3];
          ir3[3] = (v811_data + (v793_data * v809_data));
          float v814_data = s1[62];
          float v816_data = ir3[4];
          ir3[4] = (v816_data + (v793_data * v814_data));
          float v819_data = s1[75];
          float v821_data = ir3[5];
          ir3[5] = (v821_data + (v793_data * v819_data));
          float v824_data = s1[88];
          float v826_data = ir3[6];
          ir3[6] = (v826_data + (v793_data * v824_data));
          float v829_data = s1[101];
          float v831_data = ir3[7];
          ir3[7] = (v831_data + (v793_data * v829_data));
          float v834_data = s1[114];
          float v836_data = ir3[8];
          ir3[8] = (v836_data + (v793_data * v834_data));
          float v839_data = s1[127];
          float v841_data = ir3[9];
          ir3[9] = (v841_data + (v793_data * v839_data));
          float v844_data = s1[140];
          float v846_data = ir3[10];
          ir3[10] = (v846_data + (v793_data * v844_data));
          float v849_data = s1[153];
          float v851_data = ir3[11];
          ir3[11] = (v851_data + (v793_data * v849_data));
          float v854_data = s1[166];
          float v856_data = ir3[12];
          ir3[12] = (v856_data + (v793_data * v854_data));
          float v861_data = r2[11];
          float v862_data = s1[11];
          float v864_data = ir3[0];
          ir3[0] = (v864_data + (v861_data * v862_data));
          float v867_data = s1[24];
          float v869_data = ir3[1];
          ir3[1] = (v869_data + (v861_data * v867_data));
          float v872_data = s1[37];
          float v874_data = ir3[2];
          ir3[2] = (v874_data + (v861_data * v872_data));
          float v877_data = s1[50];
          float v879_data = ir3[3];
          ir3[3] = (v879_data + (v861_data * v877_data));
          float v882_data = s1[63];
          float v884_data = ir3[4];
          ir3[4] = (v884_data + (v861_data * v882_data));
          float v887_data = s1[76];
          float v889_data = ir3[5];
          ir3[5] = (v889_data + (v861_data * v887_data));
          float v892_data = s1[89];
          float v894_data = ir3[6];
          ir3[6] = (v894_data + (v861_data * v892_data));
          float v897_data = s1[102];
          float v899_data = ir3[7];
          ir3[7] = (v899_data + (v861_data * v897_data));
          float v902_data = s1[115];
          float v904_data = ir3[8];
          ir3[8] = (v904_data + (v861_data * v902_data));
          float v907_data = s1[128];
          float v909_data = ir3[9];
          ir3[9] = (v909_data + (v861_data * v907_data));
          float v912_data = s1[141];
          float v914_data = ir3[10];
          ir3[10] = (v914_data + (v861_data * v912_data));
          float v917_data = s1[154];
          float v919_data = ir3[11];
          ir3[11] = (v919_data + (v861_data * v917_data));
          float v922_data = s1[167];
          float v924_data = ir3[12];
          ir3[12] = (v924_data + (v861_data * v922_data));
          float v929_data = r2[12];
          float v930_data = s1[12];
          float v932_data = ir3[0];
          ir3[0] = (v932_data + (v929_data * v930_data));
          float v935_data = s1[25];
          float v937_data = ir3[1];
          ir3[1] = (v937_data + (v929_data * v935_data));
          float v940_data = s1[38];
          float v942_data = ir3[2];
          ir3[2] = (v942_data + (v929_data * v940_data));
          float v945_data = s1[51];
          float v947_data = ir3[3];
          ir3[3] = (v947_data + (v929_data * v945_data));
          float v950_data = s1[64];
          float v952_data = ir3[4];
          ir3[4] = (v952_data + (v929_data * v950_data));
          float v955_data = s1[77];
          float v957_data = ir3[5];
          ir3[5] = (v957_data + (v929_data * v955_data));
          float v960_data = s1[90];
          float v962_data = ir3[6];
          ir3[6] = (v962_data + (v929_data * v960_data));
          float v965_data = s1[103];
          float v967_data = ir3[7];
          ir3[7] = (v967_data + (v929_data * v965_data));
          float v970_data = s1[116];
          float v972_data = ir3[8];
          ir3[8] = (v972_data + (v929_data * v970_data));
          float v975_data = s1[129];
          float v977_data = ir3[9];
          ir3[9] = (v977_data + (v929_data * v975_data));
          float v980_data = s1[142];
          float v982_data = ir3[10];
          ir3[10] = (v982_data + (v929_data * v980_data));
          float v985_data = s1[155];
          float v987_data = ir3[11];
          ir3[11] = (v987_data + (v929_data * v985_data));
          float v990_data = s1[168];
          float v992_data = ir3[12];
          ir3[12] = (v992_data + (v929_data * v990_data));
          #pragma unroll
          for (int32_t v997_n0 = 0; v997_n0 < 1; ++v997_n0) {
            #pragma unroll
            for (int32_t v998_n1 = 0; v998_n1 < 13; ++v998_n1) {
              int32_t v999_a = v997_n0 + v998_n1;
              int32_t v1000_a = v997_n0 + v998_n1;
              float v1001_data = ir3[v1000_a];
              r3[v1000_a] = v1001_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v1006_i0 = 0; v1006_i0 < 1; ++v1006_i0) {
            int32_t v1015_lead = v9_lead + (v1006_i0 * 32);
            #pragma unroll
            for (int32_t v1007_i1 = 0; v1007_i1 < 13; ++v1007_i1) {
              int32_t v1008_a = v1006_i0 + v1007_i1;
              float v1010_data = r3[(v1006_i0 + v1007_i1)];
              glb_m3[(v1015_lead + (v1007_i1 * 32))] = v1010_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

