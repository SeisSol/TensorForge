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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v19_lead = v12_lead + (v13_i0 * 32);
            #pragma unroll
            for (int32_t v14_i1 = 10; v14_i1 < 13; ++v14_i1) {
              float v22_data = __ldcg(&glb_m1[(v19_lead + (v14_i1 * 32))]);
              r0[(v13_i0 + (v14_i1 - 10))] = v22_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
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
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[1]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v33_data = r0[0];
          float v34_data = s0[114];
          float v36_data = ir1[0];
          ir1[0] = (v36_data + (v33_data * v34_data));
          float v41_data = r0[1];
          float v42_data = s0[115];
          float v44_data = ir1[0];
          ir1[0] = (v44_data + (v41_data * v42_data));
          float v49_data = r0[2];
          float v50_data = s0[116];
          float v52_data = ir1[0];
          ir1[0] = (v52_data + (v49_data * v50_data));
          #pragma unroll
          for (int32_t v57_n0 = 0; v57_n0 < 1; ++v57_n0) {
            #pragma unroll
            for (int32_t v58_n1 = 0; v58_n1 < 1; ++v58_n1) {
              int32_t v59_a = v57_n0 + v58_n1;
              float v60_data = ir1[v59_a];
              r1[v59_a] = v60_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v65_i0 = 0; v65_i0 < 1; ++v65_i0) {
            int32_t v73_lead = v12_lead + (v65_i0 * 32);
            #pragma unroll
            for (int32_t v66_i1 = 0; v66_i1 < 1; ++v66_i1) {
              float v68_data = r1[(v65_i0 + v66_i1)];
              glb_m0[(v73_lead + ((v66_i1 + 8) * 32))] = v68_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v81_i0 = 0; v81_i0 < 1; ++v81_i0) {
            int32_t v87_lead = v12_lead + (v81_i0 * 32);
            #pragma unroll
            for (int32_t v82_i1 = 0; v82_i1 < 13; ++v82_i1) {
              float v90_data = glb_m0[(v87_lead + (v82_i1 * 32))];
              r2[(v81_i0 + v82_i1)] = v90_data;
            }
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
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
          // wait(r2 = load{g>r}(glb_m0););
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[13]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v100_data = r2[0];
          float v101_data = s1[0];
          float v103_data = ir3[0];
          ir3[0] = (v103_data + (v100_data * v101_data));
          float v106_data = s1[13];
          float v108_data = ir3[1];
          ir3[1] = (v108_data + (v100_data * v106_data));
          float v111_data = s1[26];
          float v113_data = ir3[2];
          ir3[2] = (v113_data + (v100_data * v111_data));
          float v116_data = s1[39];
          float v118_data = ir3[3];
          ir3[3] = (v118_data + (v100_data * v116_data));
          float v121_data = s1[52];
          float v123_data = ir3[4];
          ir3[4] = (v123_data + (v100_data * v121_data));
          float v126_data = s1[65];
          float v128_data = ir3[5];
          ir3[5] = (v128_data + (v100_data * v126_data));
          float v131_data = s1[78];
          float v133_data = ir3[6];
          ir3[6] = (v133_data + (v100_data * v131_data));
          float v136_data = s1[91];
          float v138_data = ir3[7];
          ir3[7] = (v138_data + (v100_data * v136_data));
          float v141_data = s1[104];
          float v143_data = ir3[8];
          ir3[8] = (v143_data + (v100_data * v141_data));
          float v146_data = s1[117];
          float v148_data = ir3[9];
          ir3[9] = (v148_data + (v100_data * v146_data));
          float v151_data = s1[130];
          float v153_data = ir3[10];
          ir3[10] = (v153_data + (v100_data * v151_data));
          float v156_data = s1[143];
          float v158_data = ir3[11];
          ir3[11] = (v158_data + (v100_data * v156_data));
          float v161_data = s1[156];
          float v163_data = ir3[12];
          ir3[12] = (v163_data + (v100_data * v161_data));
          float v168_data = r2[1];
          float v169_data = s1[1];
          float v171_data = ir3[0];
          ir3[0] = (v171_data + (v168_data * v169_data));
          float v174_data = s1[14];
          float v176_data = ir3[1];
          ir3[1] = (v176_data + (v168_data * v174_data));
          float v179_data = s1[27];
          float v181_data = ir3[2];
          ir3[2] = (v181_data + (v168_data * v179_data));
          float v184_data = s1[40];
          float v186_data = ir3[3];
          ir3[3] = (v186_data + (v168_data * v184_data));
          float v189_data = s1[53];
          float v191_data = ir3[4];
          ir3[4] = (v191_data + (v168_data * v189_data));
          float v194_data = s1[66];
          float v196_data = ir3[5];
          ir3[5] = (v196_data + (v168_data * v194_data));
          float v199_data = s1[79];
          float v201_data = ir3[6];
          ir3[6] = (v201_data + (v168_data * v199_data));
          float v204_data = s1[92];
          float v206_data = ir3[7];
          ir3[7] = (v206_data + (v168_data * v204_data));
          float v209_data = s1[105];
          float v211_data = ir3[8];
          ir3[8] = (v211_data + (v168_data * v209_data));
          float v214_data = s1[118];
          float v216_data = ir3[9];
          ir3[9] = (v216_data + (v168_data * v214_data));
          float v219_data = s1[131];
          float v221_data = ir3[10];
          ir3[10] = (v221_data + (v168_data * v219_data));
          float v224_data = s1[144];
          float v226_data = ir3[11];
          ir3[11] = (v226_data + (v168_data * v224_data));
          float v229_data = s1[157];
          float v231_data = ir3[12];
          ir3[12] = (v231_data + (v168_data * v229_data));
          float v236_data = r2[2];
          float v237_data = s1[2];
          float v239_data = ir3[0];
          ir3[0] = (v239_data + (v236_data * v237_data));
          float v242_data = s1[15];
          float v244_data = ir3[1];
          ir3[1] = (v244_data + (v236_data * v242_data));
          float v247_data = s1[28];
          float v249_data = ir3[2];
          ir3[2] = (v249_data + (v236_data * v247_data));
          float v252_data = s1[41];
          float v254_data = ir3[3];
          ir3[3] = (v254_data + (v236_data * v252_data));
          float v257_data = s1[54];
          float v259_data = ir3[4];
          ir3[4] = (v259_data + (v236_data * v257_data));
          float v262_data = s1[67];
          float v264_data = ir3[5];
          ir3[5] = (v264_data + (v236_data * v262_data));
          float v267_data = s1[80];
          float v269_data = ir3[6];
          ir3[6] = (v269_data + (v236_data * v267_data));
          float v272_data = s1[93];
          float v274_data = ir3[7];
          ir3[7] = (v274_data + (v236_data * v272_data));
          float v277_data = s1[106];
          float v279_data = ir3[8];
          ir3[8] = (v279_data + (v236_data * v277_data));
          float v282_data = s1[119];
          float v284_data = ir3[9];
          ir3[9] = (v284_data + (v236_data * v282_data));
          float v287_data = s1[132];
          float v289_data = ir3[10];
          ir3[10] = (v289_data + (v236_data * v287_data));
          float v292_data = s1[145];
          float v294_data = ir3[11];
          ir3[11] = (v294_data + (v236_data * v292_data));
          float v297_data = s1[158];
          float v299_data = ir3[12];
          ir3[12] = (v299_data + (v236_data * v297_data));
          float v304_data = r2[3];
          float v305_data = s1[3];
          float v307_data = ir3[0];
          ir3[0] = (v307_data + (v304_data * v305_data));
          float v310_data = s1[16];
          float v312_data = ir3[1];
          ir3[1] = (v312_data + (v304_data * v310_data));
          float v315_data = s1[29];
          float v317_data = ir3[2];
          ir3[2] = (v317_data + (v304_data * v315_data));
          float v320_data = s1[42];
          float v322_data = ir3[3];
          ir3[3] = (v322_data + (v304_data * v320_data));
          float v325_data = s1[55];
          float v327_data = ir3[4];
          ir3[4] = (v327_data + (v304_data * v325_data));
          float v330_data = s1[68];
          float v332_data = ir3[5];
          ir3[5] = (v332_data + (v304_data * v330_data));
          float v335_data = s1[81];
          float v337_data = ir3[6];
          ir3[6] = (v337_data + (v304_data * v335_data));
          float v340_data = s1[94];
          float v342_data = ir3[7];
          ir3[7] = (v342_data + (v304_data * v340_data));
          float v345_data = s1[107];
          float v347_data = ir3[8];
          ir3[8] = (v347_data + (v304_data * v345_data));
          float v350_data = s1[120];
          float v352_data = ir3[9];
          ir3[9] = (v352_data + (v304_data * v350_data));
          float v355_data = s1[133];
          float v357_data = ir3[10];
          ir3[10] = (v357_data + (v304_data * v355_data));
          float v360_data = s1[146];
          float v362_data = ir3[11];
          ir3[11] = (v362_data + (v304_data * v360_data));
          float v365_data = s1[159];
          float v367_data = ir3[12];
          ir3[12] = (v367_data + (v304_data * v365_data));
          float v372_data = r2[4];
          float v373_data = s1[4];
          float v375_data = ir3[0];
          ir3[0] = (v375_data + (v372_data * v373_data));
          float v378_data = s1[17];
          float v380_data = ir3[1];
          ir3[1] = (v380_data + (v372_data * v378_data));
          float v383_data = s1[30];
          float v385_data = ir3[2];
          ir3[2] = (v385_data + (v372_data * v383_data));
          float v388_data = s1[43];
          float v390_data = ir3[3];
          ir3[3] = (v390_data + (v372_data * v388_data));
          float v393_data = s1[56];
          float v395_data = ir3[4];
          ir3[4] = (v395_data + (v372_data * v393_data));
          float v398_data = s1[69];
          float v400_data = ir3[5];
          ir3[5] = (v400_data + (v372_data * v398_data));
          float v403_data = s1[82];
          float v405_data = ir3[6];
          ir3[6] = (v405_data + (v372_data * v403_data));
          float v408_data = s1[95];
          float v410_data = ir3[7];
          ir3[7] = (v410_data + (v372_data * v408_data));
          float v413_data = s1[108];
          float v415_data = ir3[8];
          ir3[8] = (v415_data + (v372_data * v413_data));
          float v418_data = s1[121];
          float v420_data = ir3[9];
          ir3[9] = (v420_data + (v372_data * v418_data));
          float v423_data = s1[134];
          float v425_data = ir3[10];
          ir3[10] = (v425_data + (v372_data * v423_data));
          float v428_data = s1[147];
          float v430_data = ir3[11];
          ir3[11] = (v430_data + (v372_data * v428_data));
          float v433_data = s1[160];
          float v435_data = ir3[12];
          ir3[12] = (v435_data + (v372_data * v433_data));
          float v440_data = r2[5];
          float v441_data = s1[5];
          float v443_data = ir3[0];
          ir3[0] = (v443_data + (v440_data * v441_data));
          float v446_data = s1[18];
          float v448_data = ir3[1];
          ir3[1] = (v448_data + (v440_data * v446_data));
          float v451_data = s1[31];
          float v453_data = ir3[2];
          ir3[2] = (v453_data + (v440_data * v451_data));
          float v456_data = s1[44];
          float v458_data = ir3[3];
          ir3[3] = (v458_data + (v440_data * v456_data));
          float v461_data = s1[57];
          float v463_data = ir3[4];
          ir3[4] = (v463_data + (v440_data * v461_data));
          float v466_data = s1[70];
          float v468_data = ir3[5];
          ir3[5] = (v468_data + (v440_data * v466_data));
          float v471_data = s1[83];
          float v473_data = ir3[6];
          ir3[6] = (v473_data + (v440_data * v471_data));
          float v476_data = s1[96];
          float v478_data = ir3[7];
          ir3[7] = (v478_data + (v440_data * v476_data));
          float v481_data = s1[109];
          float v483_data = ir3[8];
          ir3[8] = (v483_data + (v440_data * v481_data));
          float v486_data = s1[122];
          float v488_data = ir3[9];
          ir3[9] = (v488_data + (v440_data * v486_data));
          float v491_data = s1[135];
          float v493_data = ir3[10];
          ir3[10] = (v493_data + (v440_data * v491_data));
          float v496_data = s1[148];
          float v498_data = ir3[11];
          ir3[11] = (v498_data + (v440_data * v496_data));
          float v501_data = s1[161];
          float v503_data = ir3[12];
          ir3[12] = (v503_data + (v440_data * v501_data));
          float v508_data = r2[6];
          float v509_data = s1[6];
          float v511_data = ir3[0];
          ir3[0] = (v511_data + (v508_data * v509_data));
          float v514_data = s1[19];
          float v516_data = ir3[1];
          ir3[1] = (v516_data + (v508_data * v514_data));
          float v519_data = s1[32];
          float v521_data = ir3[2];
          ir3[2] = (v521_data + (v508_data * v519_data));
          float v524_data = s1[45];
          float v526_data = ir3[3];
          ir3[3] = (v526_data + (v508_data * v524_data));
          float v529_data = s1[58];
          float v531_data = ir3[4];
          ir3[4] = (v531_data + (v508_data * v529_data));
          float v534_data = s1[71];
          float v536_data = ir3[5];
          ir3[5] = (v536_data + (v508_data * v534_data));
          float v539_data = s1[84];
          float v541_data = ir3[6];
          ir3[6] = (v541_data + (v508_data * v539_data));
          float v544_data = s1[97];
          float v546_data = ir3[7];
          ir3[7] = (v546_data + (v508_data * v544_data));
          float v549_data = s1[110];
          float v551_data = ir3[8];
          ir3[8] = (v551_data + (v508_data * v549_data));
          float v554_data = s1[123];
          float v556_data = ir3[9];
          ir3[9] = (v556_data + (v508_data * v554_data));
          float v559_data = s1[136];
          float v561_data = ir3[10];
          ir3[10] = (v561_data + (v508_data * v559_data));
          float v564_data = s1[149];
          float v566_data = ir3[11];
          ir3[11] = (v566_data + (v508_data * v564_data));
          float v569_data = s1[162];
          float v571_data = ir3[12];
          ir3[12] = (v571_data + (v508_data * v569_data));
          float v576_data = r2[7];
          float v577_data = s1[7];
          float v579_data = ir3[0];
          ir3[0] = (v579_data + (v576_data * v577_data));
          float v582_data = s1[20];
          float v584_data = ir3[1];
          ir3[1] = (v584_data + (v576_data * v582_data));
          float v587_data = s1[33];
          float v589_data = ir3[2];
          ir3[2] = (v589_data + (v576_data * v587_data));
          float v592_data = s1[46];
          float v594_data = ir3[3];
          ir3[3] = (v594_data + (v576_data * v592_data));
          float v597_data = s1[59];
          float v599_data = ir3[4];
          ir3[4] = (v599_data + (v576_data * v597_data));
          float v602_data = s1[72];
          float v604_data = ir3[5];
          ir3[5] = (v604_data + (v576_data * v602_data));
          float v607_data = s1[85];
          float v609_data = ir3[6];
          ir3[6] = (v609_data + (v576_data * v607_data));
          float v612_data = s1[98];
          float v614_data = ir3[7];
          ir3[7] = (v614_data + (v576_data * v612_data));
          float v617_data = s1[111];
          float v619_data = ir3[8];
          ir3[8] = (v619_data + (v576_data * v617_data));
          float v622_data = s1[124];
          float v624_data = ir3[9];
          ir3[9] = (v624_data + (v576_data * v622_data));
          float v627_data = s1[137];
          float v629_data = ir3[10];
          ir3[10] = (v629_data + (v576_data * v627_data));
          float v632_data = s1[150];
          float v634_data = ir3[11];
          ir3[11] = (v634_data + (v576_data * v632_data));
          float v637_data = s1[163];
          float v639_data = ir3[12];
          ir3[12] = (v639_data + (v576_data * v637_data));
          float v644_data = r2[8];
          float v645_data = s1[8];
          float v647_data = ir3[0];
          ir3[0] = (v647_data + (v644_data * v645_data));
          float v650_data = s1[21];
          float v652_data = ir3[1];
          ir3[1] = (v652_data + (v644_data * v650_data));
          float v655_data = s1[34];
          float v657_data = ir3[2];
          ir3[2] = (v657_data + (v644_data * v655_data));
          float v660_data = s1[47];
          float v662_data = ir3[3];
          ir3[3] = (v662_data + (v644_data * v660_data));
          float v665_data = s1[60];
          float v667_data = ir3[4];
          ir3[4] = (v667_data + (v644_data * v665_data));
          float v670_data = s1[73];
          float v672_data = ir3[5];
          ir3[5] = (v672_data + (v644_data * v670_data));
          float v675_data = s1[86];
          float v677_data = ir3[6];
          ir3[6] = (v677_data + (v644_data * v675_data));
          float v680_data = s1[99];
          float v682_data = ir3[7];
          ir3[7] = (v682_data + (v644_data * v680_data));
          float v685_data = s1[112];
          float v687_data = ir3[8];
          ir3[8] = (v687_data + (v644_data * v685_data));
          float v690_data = s1[125];
          float v692_data = ir3[9];
          ir3[9] = (v692_data + (v644_data * v690_data));
          float v695_data = s1[138];
          float v697_data = ir3[10];
          ir3[10] = (v697_data + (v644_data * v695_data));
          float v700_data = s1[151];
          float v702_data = ir3[11];
          ir3[11] = (v702_data + (v644_data * v700_data));
          float v705_data = s1[164];
          float v707_data = ir3[12];
          ir3[12] = (v707_data + (v644_data * v705_data));
          float v712_data = r2[9];
          float v713_data = s1[9];
          float v715_data = ir3[0];
          ir3[0] = (v715_data + (v712_data * v713_data));
          float v718_data = s1[22];
          float v720_data = ir3[1];
          ir3[1] = (v720_data + (v712_data * v718_data));
          float v723_data = s1[35];
          float v725_data = ir3[2];
          ir3[2] = (v725_data + (v712_data * v723_data));
          float v728_data = s1[48];
          float v730_data = ir3[3];
          ir3[3] = (v730_data + (v712_data * v728_data));
          float v733_data = s1[61];
          float v735_data = ir3[4];
          ir3[4] = (v735_data + (v712_data * v733_data));
          float v738_data = s1[74];
          float v740_data = ir3[5];
          ir3[5] = (v740_data + (v712_data * v738_data));
          float v743_data = s1[87];
          float v745_data = ir3[6];
          ir3[6] = (v745_data + (v712_data * v743_data));
          float v748_data = s1[100];
          float v750_data = ir3[7];
          ir3[7] = (v750_data + (v712_data * v748_data));
          float v753_data = s1[113];
          float v755_data = ir3[8];
          ir3[8] = (v755_data + (v712_data * v753_data));
          float v758_data = s1[126];
          float v760_data = ir3[9];
          ir3[9] = (v760_data + (v712_data * v758_data));
          float v763_data = s1[139];
          float v765_data = ir3[10];
          ir3[10] = (v765_data + (v712_data * v763_data));
          float v768_data = s1[152];
          float v770_data = ir3[11];
          ir3[11] = (v770_data + (v712_data * v768_data));
          float v773_data = s1[165];
          float v775_data = ir3[12];
          ir3[12] = (v775_data + (v712_data * v773_data));
          float v780_data = r2[10];
          float v781_data = s1[10];
          float v783_data = ir3[0];
          ir3[0] = (v783_data + (v780_data * v781_data));
          float v786_data = s1[23];
          float v788_data = ir3[1];
          ir3[1] = (v788_data + (v780_data * v786_data));
          float v791_data = s1[36];
          float v793_data = ir3[2];
          ir3[2] = (v793_data + (v780_data * v791_data));
          float v796_data = s1[49];
          float v798_data = ir3[3];
          ir3[3] = (v798_data + (v780_data * v796_data));
          float v801_data = s1[62];
          float v803_data = ir3[4];
          ir3[4] = (v803_data + (v780_data * v801_data));
          float v806_data = s1[75];
          float v808_data = ir3[5];
          ir3[5] = (v808_data + (v780_data * v806_data));
          float v811_data = s1[88];
          float v813_data = ir3[6];
          ir3[6] = (v813_data + (v780_data * v811_data));
          float v816_data = s1[101];
          float v818_data = ir3[7];
          ir3[7] = (v818_data + (v780_data * v816_data));
          float v821_data = s1[114];
          float v823_data = ir3[8];
          ir3[8] = (v823_data + (v780_data * v821_data));
          float v826_data = s1[127];
          float v828_data = ir3[9];
          ir3[9] = (v828_data + (v780_data * v826_data));
          float v831_data = s1[140];
          float v833_data = ir3[10];
          ir3[10] = (v833_data + (v780_data * v831_data));
          float v836_data = s1[153];
          float v838_data = ir3[11];
          ir3[11] = (v838_data + (v780_data * v836_data));
          float v841_data = s1[166];
          float v843_data = ir3[12];
          ir3[12] = (v843_data + (v780_data * v841_data));
          float v848_data = r2[11];
          float v849_data = s1[11];
          float v851_data = ir3[0];
          ir3[0] = (v851_data + (v848_data * v849_data));
          float v854_data = s1[24];
          float v856_data = ir3[1];
          ir3[1] = (v856_data + (v848_data * v854_data));
          float v859_data = s1[37];
          float v861_data = ir3[2];
          ir3[2] = (v861_data + (v848_data * v859_data));
          float v864_data = s1[50];
          float v866_data = ir3[3];
          ir3[3] = (v866_data + (v848_data * v864_data));
          float v869_data = s1[63];
          float v871_data = ir3[4];
          ir3[4] = (v871_data + (v848_data * v869_data));
          float v874_data = s1[76];
          float v876_data = ir3[5];
          ir3[5] = (v876_data + (v848_data * v874_data));
          float v879_data = s1[89];
          float v881_data = ir3[6];
          ir3[6] = (v881_data + (v848_data * v879_data));
          float v884_data = s1[102];
          float v886_data = ir3[7];
          ir3[7] = (v886_data + (v848_data * v884_data));
          float v889_data = s1[115];
          float v891_data = ir3[8];
          ir3[8] = (v891_data + (v848_data * v889_data));
          float v894_data = s1[128];
          float v896_data = ir3[9];
          ir3[9] = (v896_data + (v848_data * v894_data));
          float v899_data = s1[141];
          float v901_data = ir3[10];
          ir3[10] = (v901_data + (v848_data * v899_data));
          float v904_data = s1[154];
          float v906_data = ir3[11];
          ir3[11] = (v906_data + (v848_data * v904_data));
          float v909_data = s1[167];
          float v911_data = ir3[12];
          ir3[12] = (v911_data + (v848_data * v909_data));
          float v916_data = r2[12];
          float v917_data = s1[12];
          float v919_data = ir3[0];
          ir3[0] = (v919_data + (v916_data * v917_data));
          float v922_data = s1[25];
          float v924_data = ir3[1];
          ir3[1] = (v924_data + (v916_data * v922_data));
          float v927_data = s1[38];
          float v929_data = ir3[2];
          ir3[2] = (v929_data + (v916_data * v927_data));
          float v932_data = s1[51];
          float v934_data = ir3[3];
          ir3[3] = (v934_data + (v916_data * v932_data));
          float v937_data = s1[64];
          float v939_data = ir3[4];
          ir3[4] = (v939_data + (v916_data * v937_data));
          float v942_data = s1[77];
          float v944_data = ir3[5];
          ir3[5] = (v944_data + (v916_data * v942_data));
          float v947_data = s1[90];
          float v949_data = ir3[6];
          ir3[6] = (v949_data + (v916_data * v947_data));
          float v952_data = s1[103];
          float v954_data = ir3[7];
          ir3[7] = (v954_data + (v916_data * v952_data));
          float v957_data = s1[116];
          float v959_data = ir3[8];
          ir3[8] = (v959_data + (v916_data * v957_data));
          float v962_data = s1[129];
          float v964_data = ir3[9];
          ir3[9] = (v964_data + (v916_data * v962_data));
          float v967_data = s1[142];
          float v969_data = ir3[10];
          ir3[10] = (v969_data + (v916_data * v967_data));
          float v972_data = s1[155];
          float v974_data = ir3[11];
          ir3[11] = (v974_data + (v916_data * v972_data));
          float v977_data = s1[168];
          float v979_data = ir3[12];
          ir3[12] = (v979_data + (v916_data * v977_data));
          #pragma unroll
          for (int32_t v984_n0 = 0; v984_n0 < 1; ++v984_n0) {
            #pragma unroll
            for (int32_t v985_n1 = 0; v985_n1 < 13; ++v985_n1) {
              int32_t v986_a = v984_n0 + v985_n1;
              float v987_data = ir3[v986_a];
              r3[v986_a] = v987_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v992_i0 = 0; v992_i0 < 1; ++v992_i0) {
            int32_t v1000_lead = v12_lead + (v992_i0 * 32);
            #pragma unroll
            for (int32_t v993_i1 = 0; v993_i1 < 13; ++v993_i1) {
              float v995_data = r3[(v992_i0 + v993_i1)];
              glb_m3[(v1000_lead + (v993_i1 * 32))] = v995_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

