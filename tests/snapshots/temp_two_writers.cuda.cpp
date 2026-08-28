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
          int32_t v8_lead = threadIdx.x % 16;
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 12; ++v10_i1) {
              int32_t v16_a = v10_i1 * 6;
              int32_t v17_a = v8_lead + v16_a;
              float v25_data = __ldcg(&glb_m0[(v8_lead + v16_a)]);
              int32_t v26_a = 0 + v10_i1;
              r0[v26_a] = v25_data;
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
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
              int32_t v40_a = v34_i1 * 6;
              int32_t v41_a = v8_lead + v40_a;
              float v49_data = __ldcg(&glb_m2[(v8_lead + v40_a)]);
              int32_t v50_a = 0 + v34_i1;
              r2[v50_a] = v49_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          if (v8_lead < 6) {
            float v56_data = r0[0];
            float v57_data = s0[0];
            float v59_data = r1[0];
            r1[0] = (v59_data + (v56_data * v57_data));
            float v62_data = s0[12];
            float v64_data = r1[1];
            r1[1] = (v64_data + (v56_data * v62_data));
            float v67_data = s0[24];
            float v69_data = r1[2];
            r1[2] = (v69_data + (v56_data * v67_data));
            float v72_data = s0[36];
            float v74_data = r1[3];
            r1[3] = (v74_data + (v56_data * v72_data));
            float v77_data = s0[48];
            float v79_data = r1[4];
            r1[4] = (v79_data + (v56_data * v77_data));
            float v82_data = s0[60];
            float v84_data = r1[5];
            r1[5] = (v84_data + (v56_data * v82_data));
            float v87_data = s0[72];
            float v89_data = r1[6];
            r1[6] = (v89_data + (v56_data * v87_data));
            float v92_data = s0[84];
            float v94_data = r1[7];
            r1[7] = (v94_data + (v56_data * v92_data));
            float v97_data = s0[96];
            float v99_data = r1[8];
            r1[8] = (v99_data + (v56_data * v97_data));
            float v102_data = s0[108];
            float v104_data = r1[9];
            r1[9] = (v104_data + (v56_data * v102_data));
            float v107_data = s0[120];
            float v109_data = r1[10];
            r1[10] = (v109_data + (v56_data * v107_data));
            float v112_data = s0[132];
            float v114_data = r1[11];
            r1[11] = (v114_data + (v56_data * v112_data));
          }
          if (v8_lead < 6) {
            float v120_data = r0[1];
            float v121_data = s0[1];
            float v123_data = r1[0];
            r1[0] = (v123_data + (v120_data * v121_data));
            float v126_data = s0[13];
            float v128_data = r1[1];
            r1[1] = (v128_data + (v120_data * v126_data));
            float v131_data = s0[25];
            float v133_data = r1[2];
            r1[2] = (v133_data + (v120_data * v131_data));
            float v136_data = s0[37];
            float v138_data = r1[3];
            r1[3] = (v138_data + (v120_data * v136_data));
            float v141_data = s0[49];
            float v143_data = r1[4];
            r1[4] = (v143_data + (v120_data * v141_data));
            float v146_data = s0[61];
            float v148_data = r1[5];
            r1[5] = (v148_data + (v120_data * v146_data));
            float v151_data = s0[73];
            float v153_data = r1[6];
            r1[6] = (v153_data + (v120_data * v151_data));
            float v156_data = s0[85];
            float v158_data = r1[7];
            r1[7] = (v158_data + (v120_data * v156_data));
            float v161_data = s0[97];
            float v163_data = r1[8];
            r1[8] = (v163_data + (v120_data * v161_data));
            float v166_data = s0[109];
            float v168_data = r1[9];
            r1[9] = (v168_data + (v120_data * v166_data));
            float v171_data = s0[121];
            float v173_data = r1[10];
            r1[10] = (v173_data + (v120_data * v171_data));
            float v176_data = s0[133];
            float v178_data = r1[11];
            r1[11] = (v178_data + (v120_data * v176_data));
          }
          if (v8_lead < 6) {
            float v184_data = r0[2];
            float v185_data = s0[2];
            float v187_data = r1[0];
            r1[0] = (v187_data + (v184_data * v185_data));
            float v190_data = s0[14];
            float v192_data = r1[1];
            r1[1] = (v192_data + (v184_data * v190_data));
            float v195_data = s0[26];
            float v197_data = r1[2];
            r1[2] = (v197_data + (v184_data * v195_data));
            float v200_data = s0[38];
            float v202_data = r1[3];
            r1[3] = (v202_data + (v184_data * v200_data));
            float v205_data = s0[50];
            float v207_data = r1[4];
            r1[4] = (v207_data + (v184_data * v205_data));
            float v210_data = s0[62];
            float v212_data = r1[5];
            r1[5] = (v212_data + (v184_data * v210_data));
            float v215_data = s0[74];
            float v217_data = r1[6];
            r1[6] = (v217_data + (v184_data * v215_data));
            float v220_data = s0[86];
            float v222_data = r1[7];
            r1[7] = (v222_data + (v184_data * v220_data));
            float v225_data = s0[98];
            float v227_data = r1[8];
            r1[8] = (v227_data + (v184_data * v225_data));
            float v230_data = s0[110];
            float v232_data = r1[9];
            r1[9] = (v232_data + (v184_data * v230_data));
            float v235_data = s0[122];
            float v237_data = r1[10];
            r1[10] = (v237_data + (v184_data * v235_data));
            float v240_data = s0[134];
            float v242_data = r1[11];
            r1[11] = (v242_data + (v184_data * v240_data));
          }
          if (v8_lead < 6) {
            float v248_data = r0[3];
            float v249_data = s0[3];
            float v251_data = r1[0];
            r1[0] = (v251_data + (v248_data * v249_data));
            float v254_data = s0[15];
            float v256_data = r1[1];
            r1[1] = (v256_data + (v248_data * v254_data));
            float v259_data = s0[27];
            float v261_data = r1[2];
            r1[2] = (v261_data + (v248_data * v259_data));
            float v264_data = s0[39];
            float v266_data = r1[3];
            r1[3] = (v266_data + (v248_data * v264_data));
            float v269_data = s0[51];
            float v271_data = r1[4];
            r1[4] = (v271_data + (v248_data * v269_data));
            float v274_data = s0[63];
            float v276_data = r1[5];
            r1[5] = (v276_data + (v248_data * v274_data));
            float v279_data = s0[75];
            float v281_data = r1[6];
            r1[6] = (v281_data + (v248_data * v279_data));
            float v284_data = s0[87];
            float v286_data = r1[7];
            r1[7] = (v286_data + (v248_data * v284_data));
            float v289_data = s0[99];
            float v291_data = r1[8];
            r1[8] = (v291_data + (v248_data * v289_data));
            float v294_data = s0[111];
            float v296_data = r1[9];
            r1[9] = (v296_data + (v248_data * v294_data));
            float v299_data = s0[123];
            float v301_data = r1[10];
            r1[10] = (v301_data + (v248_data * v299_data));
            float v304_data = s0[135];
            float v306_data = r1[11];
            r1[11] = (v306_data + (v248_data * v304_data));
          }
          if (v8_lead < 6) {
            float v312_data = r0[4];
            float v313_data = s0[4];
            float v315_data = r1[0];
            r1[0] = (v315_data + (v312_data * v313_data));
            float v318_data = s0[16];
            float v320_data = r1[1];
            r1[1] = (v320_data + (v312_data * v318_data));
            float v323_data = s0[28];
            float v325_data = r1[2];
            r1[2] = (v325_data + (v312_data * v323_data));
            float v328_data = s0[40];
            float v330_data = r1[3];
            r1[3] = (v330_data + (v312_data * v328_data));
            float v333_data = s0[52];
            float v335_data = r1[4];
            r1[4] = (v335_data + (v312_data * v333_data));
            float v338_data = s0[64];
            float v340_data = r1[5];
            r1[5] = (v340_data + (v312_data * v338_data));
            float v343_data = s0[76];
            float v345_data = r1[6];
            r1[6] = (v345_data + (v312_data * v343_data));
            float v348_data = s0[88];
            float v350_data = r1[7];
            r1[7] = (v350_data + (v312_data * v348_data));
            float v353_data = s0[100];
            float v355_data = r1[8];
            r1[8] = (v355_data + (v312_data * v353_data));
            float v358_data = s0[112];
            float v360_data = r1[9];
            r1[9] = (v360_data + (v312_data * v358_data));
            float v363_data = s0[124];
            float v365_data = r1[10];
            r1[10] = (v365_data + (v312_data * v363_data));
            float v368_data = s0[136];
            float v370_data = r1[11];
            r1[11] = (v370_data + (v312_data * v368_data));
          }
          if (v8_lead < 6) {
            float v376_data = r0[5];
            float v377_data = s0[5];
            float v379_data = r1[0];
            r1[0] = (v379_data + (v376_data * v377_data));
            float v382_data = s0[17];
            float v384_data = r1[1];
            r1[1] = (v384_data + (v376_data * v382_data));
            float v387_data = s0[29];
            float v389_data = r1[2];
            r1[2] = (v389_data + (v376_data * v387_data));
            float v392_data = s0[41];
            float v394_data = r1[3];
            r1[3] = (v394_data + (v376_data * v392_data));
            float v397_data = s0[53];
            float v399_data = r1[4];
            r1[4] = (v399_data + (v376_data * v397_data));
            float v402_data = s0[65];
            float v404_data = r1[5];
            r1[5] = (v404_data + (v376_data * v402_data));
            float v407_data = s0[77];
            float v409_data = r1[6];
            r1[6] = (v409_data + (v376_data * v407_data));
            float v412_data = s0[89];
            float v414_data = r1[7];
            r1[7] = (v414_data + (v376_data * v412_data));
            float v417_data = s0[101];
            float v419_data = r1[8];
            r1[8] = (v419_data + (v376_data * v417_data));
            float v422_data = s0[113];
            float v424_data = r1[9];
            r1[9] = (v424_data + (v376_data * v422_data));
            float v427_data = s0[125];
            float v429_data = r1[10];
            r1[10] = (v429_data + (v376_data * v427_data));
            float v432_data = s0[137];
            float v434_data = r1[11];
            r1[11] = (v434_data + (v376_data * v432_data));
          }
          if (v8_lead < 6) {
            float v440_data = r0[6];
            float v441_data = s0[6];
            float v443_data = r1[0];
            r1[0] = (v443_data + (v440_data * v441_data));
            float v446_data = s0[18];
            float v448_data = r1[1];
            r1[1] = (v448_data + (v440_data * v446_data));
            float v451_data = s0[30];
            float v453_data = r1[2];
            r1[2] = (v453_data + (v440_data * v451_data));
            float v456_data = s0[42];
            float v458_data = r1[3];
            r1[3] = (v458_data + (v440_data * v456_data));
            float v461_data = s0[54];
            float v463_data = r1[4];
            r1[4] = (v463_data + (v440_data * v461_data));
            float v466_data = s0[66];
            float v468_data = r1[5];
            r1[5] = (v468_data + (v440_data * v466_data));
            float v471_data = s0[78];
            float v473_data = r1[6];
            r1[6] = (v473_data + (v440_data * v471_data));
            float v476_data = s0[90];
            float v478_data = r1[7];
            r1[7] = (v478_data + (v440_data * v476_data));
            float v481_data = s0[102];
            float v483_data = r1[8];
            r1[8] = (v483_data + (v440_data * v481_data));
            float v486_data = s0[114];
            float v488_data = r1[9];
            r1[9] = (v488_data + (v440_data * v486_data));
            float v491_data = s0[126];
            float v493_data = r1[10];
            r1[10] = (v493_data + (v440_data * v491_data));
            float v496_data = s0[138];
            float v498_data = r1[11];
            r1[11] = (v498_data + (v440_data * v496_data));
          }
          if (v8_lead < 6) {
            float v504_data = r0[7];
            float v505_data = s0[7];
            float v507_data = r1[0];
            r1[0] = (v507_data + (v504_data * v505_data));
            float v510_data = s0[19];
            float v512_data = r1[1];
            r1[1] = (v512_data + (v504_data * v510_data));
            float v515_data = s0[31];
            float v517_data = r1[2];
            r1[2] = (v517_data + (v504_data * v515_data));
            float v520_data = s0[43];
            float v522_data = r1[3];
            r1[3] = (v522_data + (v504_data * v520_data));
            float v525_data = s0[55];
            float v527_data = r1[4];
            r1[4] = (v527_data + (v504_data * v525_data));
            float v530_data = s0[67];
            float v532_data = r1[5];
            r1[5] = (v532_data + (v504_data * v530_data));
            float v535_data = s0[79];
            float v537_data = r1[6];
            r1[6] = (v537_data + (v504_data * v535_data));
            float v540_data = s0[91];
            float v542_data = r1[7];
            r1[7] = (v542_data + (v504_data * v540_data));
            float v545_data = s0[103];
            float v547_data = r1[8];
            r1[8] = (v547_data + (v504_data * v545_data));
            float v550_data = s0[115];
            float v552_data = r1[9];
            r1[9] = (v552_data + (v504_data * v550_data));
            float v555_data = s0[127];
            float v557_data = r1[10];
            r1[10] = (v557_data + (v504_data * v555_data));
            float v560_data = s0[139];
            float v562_data = r1[11];
            r1[11] = (v562_data + (v504_data * v560_data));
          }
          if (v8_lead < 6) {
            float v568_data = r0[8];
            float v569_data = s0[8];
            float v571_data = r1[0];
            r1[0] = (v571_data + (v568_data * v569_data));
            float v574_data = s0[20];
            float v576_data = r1[1];
            r1[1] = (v576_data + (v568_data * v574_data));
            float v579_data = s0[32];
            float v581_data = r1[2];
            r1[2] = (v581_data + (v568_data * v579_data));
            float v584_data = s0[44];
            float v586_data = r1[3];
            r1[3] = (v586_data + (v568_data * v584_data));
            float v589_data = s0[56];
            float v591_data = r1[4];
            r1[4] = (v591_data + (v568_data * v589_data));
            float v594_data = s0[68];
            float v596_data = r1[5];
            r1[5] = (v596_data + (v568_data * v594_data));
            float v599_data = s0[80];
            float v601_data = r1[6];
            r1[6] = (v601_data + (v568_data * v599_data));
            float v604_data = s0[92];
            float v606_data = r1[7];
            r1[7] = (v606_data + (v568_data * v604_data));
            float v609_data = s0[104];
            float v611_data = r1[8];
            r1[8] = (v611_data + (v568_data * v609_data));
            float v614_data = s0[116];
            float v616_data = r1[9];
            r1[9] = (v616_data + (v568_data * v614_data));
            float v619_data = s0[128];
            float v621_data = r1[10];
            r1[10] = (v621_data + (v568_data * v619_data));
            float v624_data = s0[140];
            float v626_data = r1[11];
            r1[11] = (v626_data + (v568_data * v624_data));
          }
          if (v8_lead < 6) {
            float v632_data = r0[9];
            float v633_data = s0[9];
            float v635_data = r1[0];
            r1[0] = (v635_data + (v632_data * v633_data));
            float v638_data = s0[21];
            float v640_data = r1[1];
            r1[1] = (v640_data + (v632_data * v638_data));
            float v643_data = s0[33];
            float v645_data = r1[2];
            r1[2] = (v645_data + (v632_data * v643_data));
            float v648_data = s0[45];
            float v650_data = r1[3];
            r1[3] = (v650_data + (v632_data * v648_data));
            float v653_data = s0[57];
            float v655_data = r1[4];
            r1[4] = (v655_data + (v632_data * v653_data));
            float v658_data = s0[69];
            float v660_data = r1[5];
            r1[5] = (v660_data + (v632_data * v658_data));
            float v663_data = s0[81];
            float v665_data = r1[6];
            r1[6] = (v665_data + (v632_data * v663_data));
            float v668_data = s0[93];
            float v670_data = r1[7];
            r1[7] = (v670_data + (v632_data * v668_data));
            float v673_data = s0[105];
            float v675_data = r1[8];
            r1[8] = (v675_data + (v632_data * v673_data));
            float v678_data = s0[117];
            float v680_data = r1[9];
            r1[9] = (v680_data + (v632_data * v678_data));
            float v683_data = s0[129];
            float v685_data = r1[10];
            r1[10] = (v685_data + (v632_data * v683_data));
            float v688_data = s0[141];
            float v690_data = r1[11];
            r1[11] = (v690_data + (v632_data * v688_data));
          }
          if (v8_lead < 6) {
            float v696_data = r0[10];
            float v697_data = s0[10];
            float v699_data = r1[0];
            r1[0] = (v699_data + (v696_data * v697_data));
            float v702_data = s0[22];
            float v704_data = r1[1];
            r1[1] = (v704_data + (v696_data * v702_data));
            float v707_data = s0[34];
            float v709_data = r1[2];
            r1[2] = (v709_data + (v696_data * v707_data));
            float v712_data = s0[46];
            float v714_data = r1[3];
            r1[3] = (v714_data + (v696_data * v712_data));
            float v717_data = s0[58];
            float v719_data = r1[4];
            r1[4] = (v719_data + (v696_data * v717_data));
            float v722_data = s0[70];
            float v724_data = r1[5];
            r1[5] = (v724_data + (v696_data * v722_data));
            float v727_data = s0[82];
            float v729_data = r1[6];
            r1[6] = (v729_data + (v696_data * v727_data));
            float v732_data = s0[94];
            float v734_data = r1[7];
            r1[7] = (v734_data + (v696_data * v732_data));
            float v737_data = s0[106];
            float v739_data = r1[8];
            r1[8] = (v739_data + (v696_data * v737_data));
            float v742_data = s0[118];
            float v744_data = r1[9];
            r1[9] = (v744_data + (v696_data * v742_data));
            float v747_data = s0[130];
            float v749_data = r1[10];
            r1[10] = (v749_data + (v696_data * v747_data));
            float v752_data = s0[142];
            float v754_data = r1[11];
            r1[11] = (v754_data + (v696_data * v752_data));
          }
          if (v8_lead < 6) {
            float v760_data = r0[11];
            float v761_data = s0[11];
            float v763_data = r1[0];
            r1[0] = (v763_data + (v760_data * v761_data));
            float v766_data = s0[23];
            float v768_data = r1[1];
            r1[1] = (v768_data + (v760_data * v766_data));
            float v771_data = s0[35];
            float v773_data = r1[2];
            r1[2] = (v773_data + (v760_data * v771_data));
            float v776_data = s0[47];
            float v778_data = r1[3];
            r1[3] = (v778_data + (v760_data * v776_data));
            float v781_data = s0[59];
            float v783_data = r1[4];
            r1[4] = (v783_data + (v760_data * v781_data));
            float v786_data = s0[71];
            float v788_data = r1[5];
            r1[5] = (v788_data + (v760_data * v786_data));
            float v791_data = s0[83];
            float v793_data = r1[6];
            r1[6] = (v793_data + (v760_data * v791_data));
            float v796_data = s0[95];
            float v798_data = r1[7];
            r1[7] = (v798_data + (v760_data * v796_data));
            float v801_data = s0[107];
            float v803_data = r1[8];
            r1[8] = (v803_data + (v760_data * v801_data));
            float v806_data = s0[119];
            float v808_data = r1[9];
            r1[9] = (v808_data + (v760_data * v806_data));
            float v811_data = s0[131];
            float v813_data = r1[10];
            r1[10] = (v813_data + (v760_data * v811_data));
            float v816_data = s0[143];
            float v818_data = r1[11];
            r1[11] = (v818_data + (v760_data * v816_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v825_i1 = 0; v825_i1 < 12; ++v825_i1) {
              int32_t v826_a = 0 + v825_i1;
              float v828_data = r1[v825_i1];
              int32_t v835_a = v8_lead + (v825_i1 * 12);
              s1[v835_a] = v828_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v841_i1 = 0; v841_i1 < 12; ++v841_i1) {
              int32_t v847_a = v841_i1 * 12;
              int32_t v848_a = v8_lead + v847_a;
              float v856_data = __ldcg(&glb_m4[(v8_lead + v847_a)]);
              int32_t v857_a = 0 + v841_i1;
              r4[v857_a] = v856_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v8_lead < 6) {
            float v864_data = r2[0];
            float v865_data = s0[0];
            float v867_data = ir3[0];
            ir3[0] = (v867_data + (v864_data * v865_data));
            float v870_data = s0[12];
            float v872_data = ir3[1];
            ir3[1] = (v872_data + (v864_data * v870_data));
            float v875_data = s0[24];
            float v877_data = ir3[2];
            ir3[2] = (v877_data + (v864_data * v875_data));
            float v880_data = s0[36];
            float v882_data = ir3[3];
            ir3[3] = (v882_data + (v864_data * v880_data));
            float v885_data = s0[48];
            float v887_data = ir3[4];
            ir3[4] = (v887_data + (v864_data * v885_data));
            float v890_data = s0[60];
            float v892_data = ir3[5];
            ir3[5] = (v892_data + (v864_data * v890_data));
            float v895_data = s0[72];
            float v897_data = ir3[6];
            ir3[6] = (v897_data + (v864_data * v895_data));
            float v900_data = s0[84];
            float v902_data = ir3[7];
            ir3[7] = (v902_data + (v864_data * v900_data));
            float v905_data = s0[96];
            float v907_data = ir3[8];
            ir3[8] = (v907_data + (v864_data * v905_data));
            float v910_data = s0[108];
            float v912_data = ir3[9];
            ir3[9] = (v912_data + (v864_data * v910_data));
            float v915_data = s0[120];
            float v917_data = ir3[10];
            ir3[10] = (v917_data + (v864_data * v915_data));
            float v920_data = s0[132];
            float v922_data = ir3[11];
            ir3[11] = (v922_data + (v864_data * v920_data));
          }
          if (v8_lead < 6) {
            float v928_data = r2[1];
            float v929_data = s0[1];
            float v931_data = ir3[0];
            ir3[0] = (v931_data + (v928_data * v929_data));
            float v934_data = s0[13];
            float v936_data = ir3[1];
            ir3[1] = (v936_data + (v928_data * v934_data));
            float v939_data = s0[25];
            float v941_data = ir3[2];
            ir3[2] = (v941_data + (v928_data * v939_data));
            float v944_data = s0[37];
            float v946_data = ir3[3];
            ir3[3] = (v946_data + (v928_data * v944_data));
            float v949_data = s0[49];
            float v951_data = ir3[4];
            ir3[4] = (v951_data + (v928_data * v949_data));
            float v954_data = s0[61];
            float v956_data = ir3[5];
            ir3[5] = (v956_data + (v928_data * v954_data));
            float v959_data = s0[73];
            float v961_data = ir3[6];
            ir3[6] = (v961_data + (v928_data * v959_data));
            float v964_data = s0[85];
            float v966_data = ir3[7];
            ir3[7] = (v966_data + (v928_data * v964_data));
            float v969_data = s0[97];
            float v971_data = ir3[8];
            ir3[8] = (v971_data + (v928_data * v969_data));
            float v974_data = s0[109];
            float v976_data = ir3[9];
            ir3[9] = (v976_data + (v928_data * v974_data));
            float v979_data = s0[121];
            float v981_data = ir3[10];
            ir3[10] = (v981_data + (v928_data * v979_data));
            float v984_data = s0[133];
            float v986_data = ir3[11];
            ir3[11] = (v986_data + (v928_data * v984_data));
          }
          if (v8_lead < 6) {
            float v992_data = r2[2];
            float v993_data = s0[2];
            float v995_data = ir3[0];
            ir3[0] = (v995_data + (v992_data * v993_data));
            float v998_data = s0[14];
            float v1000_data = ir3[1];
            ir3[1] = (v1000_data + (v992_data * v998_data));
            float v1003_data = s0[26];
            float v1005_data = ir3[2];
            ir3[2] = (v1005_data + (v992_data * v1003_data));
            float v1008_data = s0[38];
            float v1010_data = ir3[3];
            ir3[3] = (v1010_data + (v992_data * v1008_data));
            float v1013_data = s0[50];
            float v1015_data = ir3[4];
            ir3[4] = (v1015_data + (v992_data * v1013_data));
            float v1018_data = s0[62];
            float v1020_data = ir3[5];
            ir3[5] = (v1020_data + (v992_data * v1018_data));
            float v1023_data = s0[74];
            float v1025_data = ir3[6];
            ir3[6] = (v1025_data + (v992_data * v1023_data));
            float v1028_data = s0[86];
            float v1030_data = ir3[7];
            ir3[7] = (v1030_data + (v992_data * v1028_data));
            float v1033_data = s0[98];
            float v1035_data = ir3[8];
            ir3[8] = (v1035_data + (v992_data * v1033_data));
            float v1038_data = s0[110];
            float v1040_data = ir3[9];
            ir3[9] = (v1040_data + (v992_data * v1038_data));
            float v1043_data = s0[122];
            float v1045_data = ir3[10];
            ir3[10] = (v1045_data + (v992_data * v1043_data));
            float v1048_data = s0[134];
            float v1050_data = ir3[11];
            ir3[11] = (v1050_data + (v992_data * v1048_data));
          }
          if (v8_lead < 6) {
            float v1056_data = r2[3];
            float v1057_data = s0[3];
            float v1059_data = ir3[0];
            ir3[0] = (v1059_data + (v1056_data * v1057_data));
            float v1062_data = s0[15];
            float v1064_data = ir3[1];
            ir3[1] = (v1064_data + (v1056_data * v1062_data));
            float v1067_data = s0[27];
            float v1069_data = ir3[2];
            ir3[2] = (v1069_data + (v1056_data * v1067_data));
            float v1072_data = s0[39];
            float v1074_data = ir3[3];
            ir3[3] = (v1074_data + (v1056_data * v1072_data));
            float v1077_data = s0[51];
            float v1079_data = ir3[4];
            ir3[4] = (v1079_data + (v1056_data * v1077_data));
            float v1082_data = s0[63];
            float v1084_data = ir3[5];
            ir3[5] = (v1084_data + (v1056_data * v1082_data));
            float v1087_data = s0[75];
            float v1089_data = ir3[6];
            ir3[6] = (v1089_data + (v1056_data * v1087_data));
            float v1092_data = s0[87];
            float v1094_data = ir3[7];
            ir3[7] = (v1094_data + (v1056_data * v1092_data));
            float v1097_data = s0[99];
            float v1099_data = ir3[8];
            ir3[8] = (v1099_data + (v1056_data * v1097_data));
            float v1102_data = s0[111];
            float v1104_data = ir3[9];
            ir3[9] = (v1104_data + (v1056_data * v1102_data));
            float v1107_data = s0[123];
            float v1109_data = ir3[10];
            ir3[10] = (v1109_data + (v1056_data * v1107_data));
            float v1112_data = s0[135];
            float v1114_data = ir3[11];
            ir3[11] = (v1114_data + (v1056_data * v1112_data));
          }
          if (v8_lead < 6) {
            float v1120_data = r2[4];
            float v1121_data = s0[4];
            float v1123_data = ir3[0];
            ir3[0] = (v1123_data + (v1120_data * v1121_data));
            float v1126_data = s0[16];
            float v1128_data = ir3[1];
            ir3[1] = (v1128_data + (v1120_data * v1126_data));
            float v1131_data = s0[28];
            float v1133_data = ir3[2];
            ir3[2] = (v1133_data + (v1120_data * v1131_data));
            float v1136_data = s0[40];
            float v1138_data = ir3[3];
            ir3[3] = (v1138_data + (v1120_data * v1136_data));
            float v1141_data = s0[52];
            float v1143_data = ir3[4];
            ir3[4] = (v1143_data + (v1120_data * v1141_data));
            float v1146_data = s0[64];
            float v1148_data = ir3[5];
            ir3[5] = (v1148_data + (v1120_data * v1146_data));
            float v1151_data = s0[76];
            float v1153_data = ir3[6];
            ir3[6] = (v1153_data + (v1120_data * v1151_data));
            float v1156_data = s0[88];
            float v1158_data = ir3[7];
            ir3[7] = (v1158_data + (v1120_data * v1156_data));
            float v1161_data = s0[100];
            float v1163_data = ir3[8];
            ir3[8] = (v1163_data + (v1120_data * v1161_data));
            float v1166_data = s0[112];
            float v1168_data = ir3[9];
            ir3[9] = (v1168_data + (v1120_data * v1166_data));
            float v1171_data = s0[124];
            float v1173_data = ir3[10];
            ir3[10] = (v1173_data + (v1120_data * v1171_data));
            float v1176_data = s0[136];
            float v1178_data = ir3[11];
            ir3[11] = (v1178_data + (v1120_data * v1176_data));
          }
          if (v8_lead < 6) {
            float v1184_data = r2[5];
            float v1185_data = s0[5];
            float v1187_data = ir3[0];
            ir3[0] = (v1187_data + (v1184_data * v1185_data));
            float v1190_data = s0[17];
            float v1192_data = ir3[1];
            ir3[1] = (v1192_data + (v1184_data * v1190_data));
            float v1195_data = s0[29];
            float v1197_data = ir3[2];
            ir3[2] = (v1197_data + (v1184_data * v1195_data));
            float v1200_data = s0[41];
            float v1202_data = ir3[3];
            ir3[3] = (v1202_data + (v1184_data * v1200_data));
            float v1205_data = s0[53];
            float v1207_data = ir3[4];
            ir3[4] = (v1207_data + (v1184_data * v1205_data));
            float v1210_data = s0[65];
            float v1212_data = ir3[5];
            ir3[5] = (v1212_data + (v1184_data * v1210_data));
            float v1215_data = s0[77];
            float v1217_data = ir3[6];
            ir3[6] = (v1217_data + (v1184_data * v1215_data));
            float v1220_data = s0[89];
            float v1222_data = ir3[7];
            ir3[7] = (v1222_data + (v1184_data * v1220_data));
            float v1225_data = s0[101];
            float v1227_data = ir3[8];
            ir3[8] = (v1227_data + (v1184_data * v1225_data));
            float v1230_data = s0[113];
            float v1232_data = ir3[9];
            ir3[9] = (v1232_data + (v1184_data * v1230_data));
            float v1235_data = s0[125];
            float v1237_data = ir3[10];
            ir3[10] = (v1237_data + (v1184_data * v1235_data));
            float v1240_data = s0[137];
            float v1242_data = ir3[11];
            ir3[11] = (v1242_data + (v1184_data * v1240_data));
          }
          if (v8_lead < 6) {
            float v1248_data = r2[6];
            float v1249_data = s0[6];
            float v1251_data = ir3[0];
            ir3[0] = (v1251_data + (v1248_data * v1249_data));
            float v1254_data = s0[18];
            float v1256_data = ir3[1];
            ir3[1] = (v1256_data + (v1248_data * v1254_data));
            float v1259_data = s0[30];
            float v1261_data = ir3[2];
            ir3[2] = (v1261_data + (v1248_data * v1259_data));
            float v1264_data = s0[42];
            float v1266_data = ir3[3];
            ir3[3] = (v1266_data + (v1248_data * v1264_data));
            float v1269_data = s0[54];
            float v1271_data = ir3[4];
            ir3[4] = (v1271_data + (v1248_data * v1269_data));
            float v1274_data = s0[66];
            float v1276_data = ir3[5];
            ir3[5] = (v1276_data + (v1248_data * v1274_data));
            float v1279_data = s0[78];
            float v1281_data = ir3[6];
            ir3[6] = (v1281_data + (v1248_data * v1279_data));
            float v1284_data = s0[90];
            float v1286_data = ir3[7];
            ir3[7] = (v1286_data + (v1248_data * v1284_data));
            float v1289_data = s0[102];
            float v1291_data = ir3[8];
            ir3[8] = (v1291_data + (v1248_data * v1289_data));
            float v1294_data = s0[114];
            float v1296_data = ir3[9];
            ir3[9] = (v1296_data + (v1248_data * v1294_data));
            float v1299_data = s0[126];
            float v1301_data = ir3[10];
            ir3[10] = (v1301_data + (v1248_data * v1299_data));
            float v1304_data = s0[138];
            float v1306_data = ir3[11];
            ir3[11] = (v1306_data + (v1248_data * v1304_data));
          }
          if (v8_lead < 6) {
            float v1312_data = r2[7];
            float v1313_data = s0[7];
            float v1315_data = ir3[0];
            ir3[0] = (v1315_data + (v1312_data * v1313_data));
            float v1318_data = s0[19];
            float v1320_data = ir3[1];
            ir3[1] = (v1320_data + (v1312_data * v1318_data));
            float v1323_data = s0[31];
            float v1325_data = ir3[2];
            ir3[2] = (v1325_data + (v1312_data * v1323_data));
            float v1328_data = s0[43];
            float v1330_data = ir3[3];
            ir3[3] = (v1330_data + (v1312_data * v1328_data));
            float v1333_data = s0[55];
            float v1335_data = ir3[4];
            ir3[4] = (v1335_data + (v1312_data * v1333_data));
            float v1338_data = s0[67];
            float v1340_data = ir3[5];
            ir3[5] = (v1340_data + (v1312_data * v1338_data));
            float v1343_data = s0[79];
            float v1345_data = ir3[6];
            ir3[6] = (v1345_data + (v1312_data * v1343_data));
            float v1348_data = s0[91];
            float v1350_data = ir3[7];
            ir3[7] = (v1350_data + (v1312_data * v1348_data));
            float v1353_data = s0[103];
            float v1355_data = ir3[8];
            ir3[8] = (v1355_data + (v1312_data * v1353_data));
            float v1358_data = s0[115];
            float v1360_data = ir3[9];
            ir3[9] = (v1360_data + (v1312_data * v1358_data));
            float v1363_data = s0[127];
            float v1365_data = ir3[10];
            ir3[10] = (v1365_data + (v1312_data * v1363_data));
            float v1368_data = s0[139];
            float v1370_data = ir3[11];
            ir3[11] = (v1370_data + (v1312_data * v1368_data));
          }
          if (v8_lead < 6) {
            float v1376_data = r2[8];
            float v1377_data = s0[8];
            float v1379_data = ir3[0];
            ir3[0] = (v1379_data + (v1376_data * v1377_data));
            float v1382_data = s0[20];
            float v1384_data = ir3[1];
            ir3[1] = (v1384_data + (v1376_data * v1382_data));
            float v1387_data = s0[32];
            float v1389_data = ir3[2];
            ir3[2] = (v1389_data + (v1376_data * v1387_data));
            float v1392_data = s0[44];
            float v1394_data = ir3[3];
            ir3[3] = (v1394_data + (v1376_data * v1392_data));
            float v1397_data = s0[56];
            float v1399_data = ir3[4];
            ir3[4] = (v1399_data + (v1376_data * v1397_data));
            float v1402_data = s0[68];
            float v1404_data = ir3[5];
            ir3[5] = (v1404_data + (v1376_data * v1402_data));
            float v1407_data = s0[80];
            float v1409_data = ir3[6];
            ir3[6] = (v1409_data + (v1376_data * v1407_data));
            float v1412_data = s0[92];
            float v1414_data = ir3[7];
            ir3[7] = (v1414_data + (v1376_data * v1412_data));
            float v1417_data = s0[104];
            float v1419_data = ir3[8];
            ir3[8] = (v1419_data + (v1376_data * v1417_data));
            float v1422_data = s0[116];
            float v1424_data = ir3[9];
            ir3[9] = (v1424_data + (v1376_data * v1422_data));
            float v1427_data = s0[128];
            float v1429_data = ir3[10];
            ir3[10] = (v1429_data + (v1376_data * v1427_data));
            float v1432_data = s0[140];
            float v1434_data = ir3[11];
            ir3[11] = (v1434_data + (v1376_data * v1432_data));
          }
          if (v8_lead < 6) {
            float v1440_data = r2[9];
            float v1441_data = s0[9];
            float v1443_data = ir3[0];
            ir3[0] = (v1443_data + (v1440_data * v1441_data));
            float v1446_data = s0[21];
            float v1448_data = ir3[1];
            ir3[1] = (v1448_data + (v1440_data * v1446_data));
            float v1451_data = s0[33];
            float v1453_data = ir3[2];
            ir3[2] = (v1453_data + (v1440_data * v1451_data));
            float v1456_data = s0[45];
            float v1458_data = ir3[3];
            ir3[3] = (v1458_data + (v1440_data * v1456_data));
            float v1461_data = s0[57];
            float v1463_data = ir3[4];
            ir3[4] = (v1463_data + (v1440_data * v1461_data));
            float v1466_data = s0[69];
            float v1468_data = ir3[5];
            ir3[5] = (v1468_data + (v1440_data * v1466_data));
            float v1471_data = s0[81];
            float v1473_data = ir3[6];
            ir3[6] = (v1473_data + (v1440_data * v1471_data));
            float v1476_data = s0[93];
            float v1478_data = ir3[7];
            ir3[7] = (v1478_data + (v1440_data * v1476_data));
            float v1481_data = s0[105];
            float v1483_data = ir3[8];
            ir3[8] = (v1483_data + (v1440_data * v1481_data));
            float v1486_data = s0[117];
            float v1488_data = ir3[9];
            ir3[9] = (v1488_data + (v1440_data * v1486_data));
            float v1491_data = s0[129];
            float v1493_data = ir3[10];
            ir3[10] = (v1493_data + (v1440_data * v1491_data));
            float v1496_data = s0[141];
            float v1498_data = ir3[11];
            ir3[11] = (v1498_data + (v1440_data * v1496_data));
          }
          if (v8_lead < 6) {
            float v1504_data = r2[10];
            float v1505_data = s0[10];
            float v1507_data = ir3[0];
            ir3[0] = (v1507_data + (v1504_data * v1505_data));
            float v1510_data = s0[22];
            float v1512_data = ir3[1];
            ir3[1] = (v1512_data + (v1504_data * v1510_data));
            float v1515_data = s0[34];
            float v1517_data = ir3[2];
            ir3[2] = (v1517_data + (v1504_data * v1515_data));
            float v1520_data = s0[46];
            float v1522_data = ir3[3];
            ir3[3] = (v1522_data + (v1504_data * v1520_data));
            float v1525_data = s0[58];
            float v1527_data = ir3[4];
            ir3[4] = (v1527_data + (v1504_data * v1525_data));
            float v1530_data = s0[70];
            float v1532_data = ir3[5];
            ir3[5] = (v1532_data + (v1504_data * v1530_data));
            float v1535_data = s0[82];
            float v1537_data = ir3[6];
            ir3[6] = (v1537_data + (v1504_data * v1535_data));
            float v1540_data = s0[94];
            float v1542_data = ir3[7];
            ir3[7] = (v1542_data + (v1504_data * v1540_data));
            float v1545_data = s0[106];
            float v1547_data = ir3[8];
            ir3[8] = (v1547_data + (v1504_data * v1545_data));
            float v1550_data = s0[118];
            float v1552_data = ir3[9];
            ir3[9] = (v1552_data + (v1504_data * v1550_data));
            float v1555_data = s0[130];
            float v1557_data = ir3[10];
            ir3[10] = (v1557_data + (v1504_data * v1555_data));
            float v1560_data = s0[142];
            float v1562_data = ir3[11];
            ir3[11] = (v1562_data + (v1504_data * v1560_data));
          }
          if (v8_lead < 6) {
            float v1568_data = r2[11];
            float v1569_data = s0[11];
            float v1571_data = ir3[0];
            ir3[0] = (v1571_data + (v1568_data * v1569_data));
            float v1574_data = s0[23];
            float v1576_data = ir3[1];
            ir3[1] = (v1576_data + (v1568_data * v1574_data));
            float v1579_data = s0[35];
            float v1581_data = ir3[2];
            ir3[2] = (v1581_data + (v1568_data * v1579_data));
            float v1584_data = s0[47];
            float v1586_data = ir3[3];
            ir3[3] = (v1586_data + (v1568_data * v1584_data));
            float v1589_data = s0[59];
            float v1591_data = ir3[4];
            ir3[4] = (v1591_data + (v1568_data * v1589_data));
            float v1594_data = s0[71];
            float v1596_data = ir3[5];
            ir3[5] = (v1596_data + (v1568_data * v1594_data));
            float v1599_data = s0[83];
            float v1601_data = ir3[6];
            ir3[6] = (v1601_data + (v1568_data * v1599_data));
            float v1604_data = s0[95];
            float v1606_data = ir3[7];
            ir3[7] = (v1606_data + (v1568_data * v1604_data));
            float v1609_data = s0[107];
            float v1611_data = ir3[8];
            ir3[8] = (v1611_data + (v1568_data * v1609_data));
            float v1614_data = s0[119];
            float v1616_data = ir3[9];
            ir3[9] = (v1616_data + (v1568_data * v1614_data));
            float v1619_data = s0[131];
            float v1621_data = ir3[10];
            ir3[10] = (v1621_data + (v1568_data * v1619_data));
            float v1624_data = s0[143];
            float v1626_data = ir3[11];
            ir3[11] = (v1626_data + (v1568_data * v1624_data));
          }
          if (v8_lead < 6) {
            #pragma unroll
            for (int32_t v1632_n1 = 0; v1632_n1 < 12; ++v1632_n1) {
              int32_t v1633_a = 0 + v1632_n1;
              float v1635_data = ir3[v1632_n1];
              r3[v1632_n1] = v1635_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v8_lead < 6) {
            int32_t v1650_off = v8_lead + 6;
            #pragma unroll
            for (int32_t v1641_i1 = 0; v1641_i1 < 12; ++v1641_i1) {
              int32_t v1642_a = 0 + v1641_i1;
              float v1644_data = r3[v1641_i1];
              int32_t v1652_a = v1650_off + (v1641_i1 * 12);
              s1[v1652_a] = v1644_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v8_lead < 12) {
            float v1659_data = r4[0];
            float v1660_data = s1[0];
            float v1662_data = ir5[0];
            ir5[0] = (v1662_data + (v1659_data * v1660_data));
            float v1665_data = s1[12];
            float v1667_data = ir5[1];
            ir5[1] = (v1667_data + (v1659_data * v1665_data));
            float v1670_data = s1[24];
            float v1672_data = ir5[2];
            ir5[2] = (v1672_data + (v1659_data * v1670_data));
            float v1675_data = s1[36];
            float v1677_data = ir5[3];
            ir5[3] = (v1677_data + (v1659_data * v1675_data));
            float v1680_data = s1[48];
            float v1682_data = ir5[4];
            ir5[4] = (v1682_data + (v1659_data * v1680_data));
            float v1685_data = s1[60];
            float v1687_data = ir5[5];
            ir5[5] = (v1687_data + (v1659_data * v1685_data));
            float v1690_data = s1[72];
            float v1692_data = ir5[6];
            ir5[6] = (v1692_data + (v1659_data * v1690_data));
            float v1695_data = s1[84];
            float v1697_data = ir5[7];
            ir5[7] = (v1697_data + (v1659_data * v1695_data));
            float v1700_data = s1[96];
            float v1702_data = ir5[8];
            ir5[8] = (v1702_data + (v1659_data * v1700_data));
            float v1705_data = s1[108];
            float v1707_data = ir5[9];
            ir5[9] = (v1707_data + (v1659_data * v1705_data));
            float v1710_data = s1[120];
            float v1712_data = ir5[10];
            ir5[10] = (v1712_data + (v1659_data * v1710_data));
            float v1715_data = s1[132];
            float v1717_data = ir5[11];
            ir5[11] = (v1717_data + (v1659_data * v1715_data));
          }
          if (v8_lead < 12) {
            float v1723_data = r4[1];
            float v1724_data = s1[1];
            float v1726_data = ir5[0];
            ir5[0] = (v1726_data + (v1723_data * v1724_data));
            float v1729_data = s1[13];
            float v1731_data = ir5[1];
            ir5[1] = (v1731_data + (v1723_data * v1729_data));
            float v1734_data = s1[25];
            float v1736_data = ir5[2];
            ir5[2] = (v1736_data + (v1723_data * v1734_data));
            float v1739_data = s1[37];
            float v1741_data = ir5[3];
            ir5[3] = (v1741_data + (v1723_data * v1739_data));
            float v1744_data = s1[49];
            float v1746_data = ir5[4];
            ir5[4] = (v1746_data + (v1723_data * v1744_data));
            float v1749_data = s1[61];
            float v1751_data = ir5[5];
            ir5[5] = (v1751_data + (v1723_data * v1749_data));
            float v1754_data = s1[73];
            float v1756_data = ir5[6];
            ir5[6] = (v1756_data + (v1723_data * v1754_data));
            float v1759_data = s1[85];
            float v1761_data = ir5[7];
            ir5[7] = (v1761_data + (v1723_data * v1759_data));
            float v1764_data = s1[97];
            float v1766_data = ir5[8];
            ir5[8] = (v1766_data + (v1723_data * v1764_data));
            float v1769_data = s1[109];
            float v1771_data = ir5[9];
            ir5[9] = (v1771_data + (v1723_data * v1769_data));
            float v1774_data = s1[121];
            float v1776_data = ir5[10];
            ir5[10] = (v1776_data + (v1723_data * v1774_data));
            float v1779_data = s1[133];
            float v1781_data = ir5[11];
            ir5[11] = (v1781_data + (v1723_data * v1779_data));
          }
          if (v8_lead < 12) {
            float v1787_data = r4[2];
            float v1788_data = s1[2];
            float v1790_data = ir5[0];
            ir5[0] = (v1790_data + (v1787_data * v1788_data));
            float v1793_data = s1[14];
            float v1795_data = ir5[1];
            ir5[1] = (v1795_data + (v1787_data * v1793_data));
            float v1798_data = s1[26];
            float v1800_data = ir5[2];
            ir5[2] = (v1800_data + (v1787_data * v1798_data));
            float v1803_data = s1[38];
            float v1805_data = ir5[3];
            ir5[3] = (v1805_data + (v1787_data * v1803_data));
            float v1808_data = s1[50];
            float v1810_data = ir5[4];
            ir5[4] = (v1810_data + (v1787_data * v1808_data));
            float v1813_data = s1[62];
            float v1815_data = ir5[5];
            ir5[5] = (v1815_data + (v1787_data * v1813_data));
            float v1818_data = s1[74];
            float v1820_data = ir5[6];
            ir5[6] = (v1820_data + (v1787_data * v1818_data));
            float v1823_data = s1[86];
            float v1825_data = ir5[7];
            ir5[7] = (v1825_data + (v1787_data * v1823_data));
            float v1828_data = s1[98];
            float v1830_data = ir5[8];
            ir5[8] = (v1830_data + (v1787_data * v1828_data));
            float v1833_data = s1[110];
            float v1835_data = ir5[9];
            ir5[9] = (v1835_data + (v1787_data * v1833_data));
            float v1838_data = s1[122];
            float v1840_data = ir5[10];
            ir5[10] = (v1840_data + (v1787_data * v1838_data));
            float v1843_data = s1[134];
            float v1845_data = ir5[11];
            ir5[11] = (v1845_data + (v1787_data * v1843_data));
          }
          if (v8_lead < 12) {
            float v1851_data = r4[3];
            float v1852_data = s1[3];
            float v1854_data = ir5[0];
            ir5[0] = (v1854_data + (v1851_data * v1852_data));
            float v1857_data = s1[15];
            float v1859_data = ir5[1];
            ir5[1] = (v1859_data + (v1851_data * v1857_data));
            float v1862_data = s1[27];
            float v1864_data = ir5[2];
            ir5[2] = (v1864_data + (v1851_data * v1862_data));
            float v1867_data = s1[39];
            float v1869_data = ir5[3];
            ir5[3] = (v1869_data + (v1851_data * v1867_data));
            float v1872_data = s1[51];
            float v1874_data = ir5[4];
            ir5[4] = (v1874_data + (v1851_data * v1872_data));
            float v1877_data = s1[63];
            float v1879_data = ir5[5];
            ir5[5] = (v1879_data + (v1851_data * v1877_data));
            float v1882_data = s1[75];
            float v1884_data = ir5[6];
            ir5[6] = (v1884_data + (v1851_data * v1882_data));
            float v1887_data = s1[87];
            float v1889_data = ir5[7];
            ir5[7] = (v1889_data + (v1851_data * v1887_data));
            float v1892_data = s1[99];
            float v1894_data = ir5[8];
            ir5[8] = (v1894_data + (v1851_data * v1892_data));
            float v1897_data = s1[111];
            float v1899_data = ir5[9];
            ir5[9] = (v1899_data + (v1851_data * v1897_data));
            float v1902_data = s1[123];
            float v1904_data = ir5[10];
            ir5[10] = (v1904_data + (v1851_data * v1902_data));
            float v1907_data = s1[135];
            float v1909_data = ir5[11];
            ir5[11] = (v1909_data + (v1851_data * v1907_data));
          }
          if (v8_lead < 12) {
            float v1915_data = r4[4];
            float v1916_data = s1[4];
            float v1918_data = ir5[0];
            ir5[0] = (v1918_data + (v1915_data * v1916_data));
            float v1921_data = s1[16];
            float v1923_data = ir5[1];
            ir5[1] = (v1923_data + (v1915_data * v1921_data));
            float v1926_data = s1[28];
            float v1928_data = ir5[2];
            ir5[2] = (v1928_data + (v1915_data * v1926_data));
            float v1931_data = s1[40];
            float v1933_data = ir5[3];
            ir5[3] = (v1933_data + (v1915_data * v1931_data));
            float v1936_data = s1[52];
            float v1938_data = ir5[4];
            ir5[4] = (v1938_data + (v1915_data * v1936_data));
            float v1941_data = s1[64];
            float v1943_data = ir5[5];
            ir5[5] = (v1943_data + (v1915_data * v1941_data));
            float v1946_data = s1[76];
            float v1948_data = ir5[6];
            ir5[6] = (v1948_data + (v1915_data * v1946_data));
            float v1951_data = s1[88];
            float v1953_data = ir5[7];
            ir5[7] = (v1953_data + (v1915_data * v1951_data));
            float v1956_data = s1[100];
            float v1958_data = ir5[8];
            ir5[8] = (v1958_data + (v1915_data * v1956_data));
            float v1961_data = s1[112];
            float v1963_data = ir5[9];
            ir5[9] = (v1963_data + (v1915_data * v1961_data));
            float v1966_data = s1[124];
            float v1968_data = ir5[10];
            ir5[10] = (v1968_data + (v1915_data * v1966_data));
            float v1971_data = s1[136];
            float v1973_data = ir5[11];
            ir5[11] = (v1973_data + (v1915_data * v1971_data));
          }
          if (v8_lead < 12) {
            float v1979_data = r4[5];
            float v1980_data = s1[5];
            float v1982_data = ir5[0];
            ir5[0] = (v1982_data + (v1979_data * v1980_data));
            float v1985_data = s1[17];
            float v1987_data = ir5[1];
            ir5[1] = (v1987_data + (v1979_data * v1985_data));
            float v1990_data = s1[29];
            float v1992_data = ir5[2];
            ir5[2] = (v1992_data + (v1979_data * v1990_data));
            float v1995_data = s1[41];
            float v1997_data = ir5[3];
            ir5[3] = (v1997_data + (v1979_data * v1995_data));
            float v2000_data = s1[53];
            float v2002_data = ir5[4];
            ir5[4] = (v2002_data + (v1979_data * v2000_data));
            float v2005_data = s1[65];
            float v2007_data = ir5[5];
            ir5[5] = (v2007_data + (v1979_data * v2005_data));
            float v2010_data = s1[77];
            float v2012_data = ir5[6];
            ir5[6] = (v2012_data + (v1979_data * v2010_data));
            float v2015_data = s1[89];
            float v2017_data = ir5[7];
            ir5[7] = (v2017_data + (v1979_data * v2015_data));
            float v2020_data = s1[101];
            float v2022_data = ir5[8];
            ir5[8] = (v2022_data + (v1979_data * v2020_data));
            float v2025_data = s1[113];
            float v2027_data = ir5[9];
            ir5[9] = (v2027_data + (v1979_data * v2025_data));
            float v2030_data = s1[125];
            float v2032_data = ir5[10];
            ir5[10] = (v2032_data + (v1979_data * v2030_data));
            float v2035_data = s1[137];
            float v2037_data = ir5[11];
            ir5[11] = (v2037_data + (v1979_data * v2035_data));
          }
          if (v8_lead < 12) {
            float v2043_data = r4[6];
            float v2044_data = s1[6];
            float v2046_data = ir5[0];
            ir5[0] = (v2046_data + (v2043_data * v2044_data));
            float v2049_data = s1[18];
            float v2051_data = ir5[1];
            ir5[1] = (v2051_data + (v2043_data * v2049_data));
            float v2054_data = s1[30];
            float v2056_data = ir5[2];
            ir5[2] = (v2056_data + (v2043_data * v2054_data));
            float v2059_data = s1[42];
            float v2061_data = ir5[3];
            ir5[3] = (v2061_data + (v2043_data * v2059_data));
            float v2064_data = s1[54];
            float v2066_data = ir5[4];
            ir5[4] = (v2066_data + (v2043_data * v2064_data));
            float v2069_data = s1[66];
            float v2071_data = ir5[5];
            ir5[5] = (v2071_data + (v2043_data * v2069_data));
            float v2074_data = s1[78];
            float v2076_data = ir5[6];
            ir5[6] = (v2076_data + (v2043_data * v2074_data));
            float v2079_data = s1[90];
            float v2081_data = ir5[7];
            ir5[7] = (v2081_data + (v2043_data * v2079_data));
            float v2084_data = s1[102];
            float v2086_data = ir5[8];
            ir5[8] = (v2086_data + (v2043_data * v2084_data));
            float v2089_data = s1[114];
            float v2091_data = ir5[9];
            ir5[9] = (v2091_data + (v2043_data * v2089_data));
            float v2094_data = s1[126];
            float v2096_data = ir5[10];
            ir5[10] = (v2096_data + (v2043_data * v2094_data));
            float v2099_data = s1[138];
            float v2101_data = ir5[11];
            ir5[11] = (v2101_data + (v2043_data * v2099_data));
          }
          if (v8_lead < 12) {
            float v2107_data = r4[7];
            float v2108_data = s1[7];
            float v2110_data = ir5[0];
            ir5[0] = (v2110_data + (v2107_data * v2108_data));
            float v2113_data = s1[19];
            float v2115_data = ir5[1];
            ir5[1] = (v2115_data + (v2107_data * v2113_data));
            float v2118_data = s1[31];
            float v2120_data = ir5[2];
            ir5[2] = (v2120_data + (v2107_data * v2118_data));
            float v2123_data = s1[43];
            float v2125_data = ir5[3];
            ir5[3] = (v2125_data + (v2107_data * v2123_data));
            float v2128_data = s1[55];
            float v2130_data = ir5[4];
            ir5[4] = (v2130_data + (v2107_data * v2128_data));
            float v2133_data = s1[67];
            float v2135_data = ir5[5];
            ir5[5] = (v2135_data + (v2107_data * v2133_data));
            float v2138_data = s1[79];
            float v2140_data = ir5[6];
            ir5[6] = (v2140_data + (v2107_data * v2138_data));
            float v2143_data = s1[91];
            float v2145_data = ir5[7];
            ir5[7] = (v2145_data + (v2107_data * v2143_data));
            float v2148_data = s1[103];
            float v2150_data = ir5[8];
            ir5[8] = (v2150_data + (v2107_data * v2148_data));
            float v2153_data = s1[115];
            float v2155_data = ir5[9];
            ir5[9] = (v2155_data + (v2107_data * v2153_data));
            float v2158_data = s1[127];
            float v2160_data = ir5[10];
            ir5[10] = (v2160_data + (v2107_data * v2158_data));
            float v2163_data = s1[139];
            float v2165_data = ir5[11];
            ir5[11] = (v2165_data + (v2107_data * v2163_data));
          }
          if (v8_lead < 12) {
            float v2171_data = r4[8];
            float v2172_data = s1[8];
            float v2174_data = ir5[0];
            ir5[0] = (v2174_data + (v2171_data * v2172_data));
            float v2177_data = s1[20];
            float v2179_data = ir5[1];
            ir5[1] = (v2179_data + (v2171_data * v2177_data));
            float v2182_data = s1[32];
            float v2184_data = ir5[2];
            ir5[2] = (v2184_data + (v2171_data * v2182_data));
            float v2187_data = s1[44];
            float v2189_data = ir5[3];
            ir5[3] = (v2189_data + (v2171_data * v2187_data));
            float v2192_data = s1[56];
            float v2194_data = ir5[4];
            ir5[4] = (v2194_data + (v2171_data * v2192_data));
            float v2197_data = s1[68];
            float v2199_data = ir5[5];
            ir5[5] = (v2199_data + (v2171_data * v2197_data));
            float v2202_data = s1[80];
            float v2204_data = ir5[6];
            ir5[6] = (v2204_data + (v2171_data * v2202_data));
            float v2207_data = s1[92];
            float v2209_data = ir5[7];
            ir5[7] = (v2209_data + (v2171_data * v2207_data));
            float v2212_data = s1[104];
            float v2214_data = ir5[8];
            ir5[8] = (v2214_data + (v2171_data * v2212_data));
            float v2217_data = s1[116];
            float v2219_data = ir5[9];
            ir5[9] = (v2219_data + (v2171_data * v2217_data));
            float v2222_data = s1[128];
            float v2224_data = ir5[10];
            ir5[10] = (v2224_data + (v2171_data * v2222_data));
            float v2227_data = s1[140];
            float v2229_data = ir5[11];
            ir5[11] = (v2229_data + (v2171_data * v2227_data));
          }
          if (v8_lead < 12) {
            float v2235_data = r4[9];
            float v2236_data = s1[9];
            float v2238_data = ir5[0];
            ir5[0] = (v2238_data + (v2235_data * v2236_data));
            float v2241_data = s1[21];
            float v2243_data = ir5[1];
            ir5[1] = (v2243_data + (v2235_data * v2241_data));
            float v2246_data = s1[33];
            float v2248_data = ir5[2];
            ir5[2] = (v2248_data + (v2235_data * v2246_data));
            float v2251_data = s1[45];
            float v2253_data = ir5[3];
            ir5[3] = (v2253_data + (v2235_data * v2251_data));
            float v2256_data = s1[57];
            float v2258_data = ir5[4];
            ir5[4] = (v2258_data + (v2235_data * v2256_data));
            float v2261_data = s1[69];
            float v2263_data = ir5[5];
            ir5[5] = (v2263_data + (v2235_data * v2261_data));
            float v2266_data = s1[81];
            float v2268_data = ir5[6];
            ir5[6] = (v2268_data + (v2235_data * v2266_data));
            float v2271_data = s1[93];
            float v2273_data = ir5[7];
            ir5[7] = (v2273_data + (v2235_data * v2271_data));
            float v2276_data = s1[105];
            float v2278_data = ir5[8];
            ir5[8] = (v2278_data + (v2235_data * v2276_data));
            float v2281_data = s1[117];
            float v2283_data = ir5[9];
            ir5[9] = (v2283_data + (v2235_data * v2281_data));
            float v2286_data = s1[129];
            float v2288_data = ir5[10];
            ir5[10] = (v2288_data + (v2235_data * v2286_data));
            float v2291_data = s1[141];
            float v2293_data = ir5[11];
            ir5[11] = (v2293_data + (v2235_data * v2291_data));
          }
          if (v8_lead < 12) {
            float v2299_data = r4[10];
            float v2300_data = s1[10];
            float v2302_data = ir5[0];
            ir5[0] = (v2302_data + (v2299_data * v2300_data));
            float v2305_data = s1[22];
            float v2307_data = ir5[1];
            ir5[1] = (v2307_data + (v2299_data * v2305_data));
            float v2310_data = s1[34];
            float v2312_data = ir5[2];
            ir5[2] = (v2312_data + (v2299_data * v2310_data));
            float v2315_data = s1[46];
            float v2317_data = ir5[3];
            ir5[3] = (v2317_data + (v2299_data * v2315_data));
            float v2320_data = s1[58];
            float v2322_data = ir5[4];
            ir5[4] = (v2322_data + (v2299_data * v2320_data));
            float v2325_data = s1[70];
            float v2327_data = ir5[5];
            ir5[5] = (v2327_data + (v2299_data * v2325_data));
            float v2330_data = s1[82];
            float v2332_data = ir5[6];
            ir5[6] = (v2332_data + (v2299_data * v2330_data));
            float v2335_data = s1[94];
            float v2337_data = ir5[7];
            ir5[7] = (v2337_data + (v2299_data * v2335_data));
            float v2340_data = s1[106];
            float v2342_data = ir5[8];
            ir5[8] = (v2342_data + (v2299_data * v2340_data));
            float v2345_data = s1[118];
            float v2347_data = ir5[9];
            ir5[9] = (v2347_data + (v2299_data * v2345_data));
            float v2350_data = s1[130];
            float v2352_data = ir5[10];
            ir5[10] = (v2352_data + (v2299_data * v2350_data));
            float v2355_data = s1[142];
            float v2357_data = ir5[11];
            ir5[11] = (v2357_data + (v2299_data * v2355_data));
          }
          if (v8_lead < 12) {
            float v2363_data = r4[11];
            float v2364_data = s1[11];
            float v2366_data = ir5[0];
            ir5[0] = (v2366_data + (v2363_data * v2364_data));
            float v2369_data = s1[23];
            float v2371_data = ir5[1];
            ir5[1] = (v2371_data + (v2363_data * v2369_data));
            float v2374_data = s1[35];
            float v2376_data = ir5[2];
            ir5[2] = (v2376_data + (v2363_data * v2374_data));
            float v2379_data = s1[47];
            float v2381_data = ir5[3];
            ir5[3] = (v2381_data + (v2363_data * v2379_data));
            float v2384_data = s1[59];
            float v2386_data = ir5[4];
            ir5[4] = (v2386_data + (v2363_data * v2384_data));
            float v2389_data = s1[71];
            float v2391_data = ir5[5];
            ir5[5] = (v2391_data + (v2363_data * v2389_data));
            float v2394_data = s1[83];
            float v2396_data = ir5[6];
            ir5[6] = (v2396_data + (v2363_data * v2394_data));
            float v2399_data = s1[95];
            float v2401_data = ir5[7];
            ir5[7] = (v2401_data + (v2363_data * v2399_data));
            float v2404_data = s1[107];
            float v2406_data = ir5[8];
            ir5[8] = (v2406_data + (v2363_data * v2404_data));
            float v2409_data = s1[119];
            float v2411_data = ir5[9];
            ir5[9] = (v2411_data + (v2363_data * v2409_data));
            float v2414_data = s1[131];
            float v2416_data = ir5[10];
            ir5[10] = (v2416_data + (v2363_data * v2414_data));
            float v2419_data = s1[143];
            float v2421_data = ir5[11];
            ir5[11] = (v2421_data + (v2363_data * v2419_data));
          }
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v2427_n1 = 0; v2427_n1 < 12; ++v2427_n1) {
              int32_t v2428_a = 0 + v2427_n1;
              float v2430_data = ir5[v2427_n1];
              r5[v2427_n1] = v2430_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v8_lead < 12) {
            #pragma unroll
            for (int32_t v2436_i1 = 0; v2436_i1 < 12; ++v2436_i1) {
              int32_t v2437_a = 0 + v2436_i1;
              float v2439_data = r5[v2436_i1];
              glb_m3[(v8_lead + (v2436_i1 * 12))] = v2439_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

