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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v12_lead = threadIdx.x % 16;
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 12; ++v14_i1) {
              float v22_data = __ldcg(&glb_m0[(v12_lead + (v14_i1 * 6))]);
              r0[v14_i1] = v22_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m1[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 12; ++v31_i1) {
              float v39_data = __ldcg(&glb_m2[(v12_lead + (v31_i1 * 6))]);
              r2[v31_i1] = v39_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          if (v12_lead < 6) {
            float v46_data = r0[0];
            float v47_data = s0[0];
            float v49_data = r1[0];
            r1[0] = (v49_data + (v46_data * v47_data));
            float v52_data = s0[12];
            float v54_data = r1[1];
            r1[1] = (v54_data + (v46_data * v52_data));
            float v57_data = s0[24];
            float v59_data = r1[2];
            r1[2] = (v59_data + (v46_data * v57_data));
            float v62_data = s0[36];
            float v64_data = r1[3];
            r1[3] = (v64_data + (v46_data * v62_data));
            float v67_data = s0[48];
            float v69_data = r1[4];
            r1[4] = (v69_data + (v46_data * v67_data));
            float v72_data = s0[60];
            float v74_data = r1[5];
            r1[5] = (v74_data + (v46_data * v72_data));
            float v77_data = s0[72];
            float v79_data = r1[6];
            r1[6] = (v79_data + (v46_data * v77_data));
            float v82_data = s0[84];
            float v84_data = r1[7];
            r1[7] = (v84_data + (v46_data * v82_data));
            float v87_data = s0[96];
            float v89_data = r1[8];
            r1[8] = (v89_data + (v46_data * v87_data));
            float v92_data = s0[108];
            float v94_data = r1[9];
            r1[9] = (v94_data + (v46_data * v92_data));
            float v97_data = s0[120];
            float v99_data = r1[10];
            r1[10] = (v99_data + (v46_data * v97_data));
            float v102_data = s0[132];
            float v104_data = r1[11];
            r1[11] = (v104_data + (v46_data * v102_data));
          }
          if (v12_lead < 6) {
            float v110_data = r0[1];
            float v111_data = s0[1];
            float v113_data = r1[0];
            r1[0] = (v113_data + (v110_data * v111_data));
            float v116_data = s0[13];
            float v118_data = r1[1];
            r1[1] = (v118_data + (v110_data * v116_data));
            float v121_data = s0[25];
            float v123_data = r1[2];
            r1[2] = (v123_data + (v110_data * v121_data));
            float v126_data = s0[37];
            float v128_data = r1[3];
            r1[3] = (v128_data + (v110_data * v126_data));
            float v131_data = s0[49];
            float v133_data = r1[4];
            r1[4] = (v133_data + (v110_data * v131_data));
            float v136_data = s0[61];
            float v138_data = r1[5];
            r1[5] = (v138_data + (v110_data * v136_data));
            float v141_data = s0[73];
            float v143_data = r1[6];
            r1[6] = (v143_data + (v110_data * v141_data));
            float v146_data = s0[85];
            float v148_data = r1[7];
            r1[7] = (v148_data + (v110_data * v146_data));
            float v151_data = s0[97];
            float v153_data = r1[8];
            r1[8] = (v153_data + (v110_data * v151_data));
            float v156_data = s0[109];
            float v158_data = r1[9];
            r1[9] = (v158_data + (v110_data * v156_data));
            float v161_data = s0[121];
            float v163_data = r1[10];
            r1[10] = (v163_data + (v110_data * v161_data));
            float v166_data = s0[133];
            float v168_data = r1[11];
            r1[11] = (v168_data + (v110_data * v166_data));
          }
          if (v12_lead < 6) {
            float v174_data = r0[2];
            float v175_data = s0[2];
            float v177_data = r1[0];
            r1[0] = (v177_data + (v174_data * v175_data));
            float v180_data = s0[14];
            float v182_data = r1[1];
            r1[1] = (v182_data + (v174_data * v180_data));
            float v185_data = s0[26];
            float v187_data = r1[2];
            r1[2] = (v187_data + (v174_data * v185_data));
            float v190_data = s0[38];
            float v192_data = r1[3];
            r1[3] = (v192_data + (v174_data * v190_data));
            float v195_data = s0[50];
            float v197_data = r1[4];
            r1[4] = (v197_data + (v174_data * v195_data));
            float v200_data = s0[62];
            float v202_data = r1[5];
            r1[5] = (v202_data + (v174_data * v200_data));
            float v205_data = s0[74];
            float v207_data = r1[6];
            r1[6] = (v207_data + (v174_data * v205_data));
            float v210_data = s0[86];
            float v212_data = r1[7];
            r1[7] = (v212_data + (v174_data * v210_data));
            float v215_data = s0[98];
            float v217_data = r1[8];
            r1[8] = (v217_data + (v174_data * v215_data));
            float v220_data = s0[110];
            float v222_data = r1[9];
            r1[9] = (v222_data + (v174_data * v220_data));
            float v225_data = s0[122];
            float v227_data = r1[10];
            r1[10] = (v227_data + (v174_data * v225_data));
            float v230_data = s0[134];
            float v232_data = r1[11];
            r1[11] = (v232_data + (v174_data * v230_data));
          }
          if (v12_lead < 6) {
            float v238_data = r0[3];
            float v239_data = s0[3];
            float v241_data = r1[0];
            r1[0] = (v241_data + (v238_data * v239_data));
            float v244_data = s0[15];
            float v246_data = r1[1];
            r1[1] = (v246_data + (v238_data * v244_data));
            float v249_data = s0[27];
            float v251_data = r1[2];
            r1[2] = (v251_data + (v238_data * v249_data));
            float v254_data = s0[39];
            float v256_data = r1[3];
            r1[3] = (v256_data + (v238_data * v254_data));
            float v259_data = s0[51];
            float v261_data = r1[4];
            r1[4] = (v261_data + (v238_data * v259_data));
            float v264_data = s0[63];
            float v266_data = r1[5];
            r1[5] = (v266_data + (v238_data * v264_data));
            float v269_data = s0[75];
            float v271_data = r1[6];
            r1[6] = (v271_data + (v238_data * v269_data));
            float v274_data = s0[87];
            float v276_data = r1[7];
            r1[7] = (v276_data + (v238_data * v274_data));
            float v279_data = s0[99];
            float v281_data = r1[8];
            r1[8] = (v281_data + (v238_data * v279_data));
            float v284_data = s0[111];
            float v286_data = r1[9];
            r1[9] = (v286_data + (v238_data * v284_data));
            float v289_data = s0[123];
            float v291_data = r1[10];
            r1[10] = (v291_data + (v238_data * v289_data));
            float v294_data = s0[135];
            float v296_data = r1[11];
            r1[11] = (v296_data + (v238_data * v294_data));
          }
          if (v12_lead < 6) {
            float v302_data = r0[4];
            float v303_data = s0[4];
            float v305_data = r1[0];
            r1[0] = (v305_data + (v302_data * v303_data));
            float v308_data = s0[16];
            float v310_data = r1[1];
            r1[1] = (v310_data + (v302_data * v308_data));
            float v313_data = s0[28];
            float v315_data = r1[2];
            r1[2] = (v315_data + (v302_data * v313_data));
            float v318_data = s0[40];
            float v320_data = r1[3];
            r1[3] = (v320_data + (v302_data * v318_data));
            float v323_data = s0[52];
            float v325_data = r1[4];
            r1[4] = (v325_data + (v302_data * v323_data));
            float v328_data = s0[64];
            float v330_data = r1[5];
            r1[5] = (v330_data + (v302_data * v328_data));
            float v333_data = s0[76];
            float v335_data = r1[6];
            r1[6] = (v335_data + (v302_data * v333_data));
            float v338_data = s0[88];
            float v340_data = r1[7];
            r1[7] = (v340_data + (v302_data * v338_data));
            float v343_data = s0[100];
            float v345_data = r1[8];
            r1[8] = (v345_data + (v302_data * v343_data));
            float v348_data = s0[112];
            float v350_data = r1[9];
            r1[9] = (v350_data + (v302_data * v348_data));
            float v353_data = s0[124];
            float v355_data = r1[10];
            r1[10] = (v355_data + (v302_data * v353_data));
            float v358_data = s0[136];
            float v360_data = r1[11];
            r1[11] = (v360_data + (v302_data * v358_data));
          }
          if (v12_lead < 6) {
            float v366_data = r0[5];
            float v367_data = s0[5];
            float v369_data = r1[0];
            r1[0] = (v369_data + (v366_data * v367_data));
            float v372_data = s0[17];
            float v374_data = r1[1];
            r1[1] = (v374_data + (v366_data * v372_data));
            float v377_data = s0[29];
            float v379_data = r1[2];
            r1[2] = (v379_data + (v366_data * v377_data));
            float v382_data = s0[41];
            float v384_data = r1[3];
            r1[3] = (v384_data + (v366_data * v382_data));
            float v387_data = s0[53];
            float v389_data = r1[4];
            r1[4] = (v389_data + (v366_data * v387_data));
            float v392_data = s0[65];
            float v394_data = r1[5];
            r1[5] = (v394_data + (v366_data * v392_data));
            float v397_data = s0[77];
            float v399_data = r1[6];
            r1[6] = (v399_data + (v366_data * v397_data));
            float v402_data = s0[89];
            float v404_data = r1[7];
            r1[7] = (v404_data + (v366_data * v402_data));
            float v407_data = s0[101];
            float v409_data = r1[8];
            r1[8] = (v409_data + (v366_data * v407_data));
            float v412_data = s0[113];
            float v414_data = r1[9];
            r1[9] = (v414_data + (v366_data * v412_data));
            float v417_data = s0[125];
            float v419_data = r1[10];
            r1[10] = (v419_data + (v366_data * v417_data));
            float v422_data = s0[137];
            float v424_data = r1[11];
            r1[11] = (v424_data + (v366_data * v422_data));
          }
          if (v12_lead < 6) {
            float v430_data = r0[6];
            float v431_data = s0[6];
            float v433_data = r1[0];
            r1[0] = (v433_data + (v430_data * v431_data));
            float v436_data = s0[18];
            float v438_data = r1[1];
            r1[1] = (v438_data + (v430_data * v436_data));
            float v441_data = s0[30];
            float v443_data = r1[2];
            r1[2] = (v443_data + (v430_data * v441_data));
            float v446_data = s0[42];
            float v448_data = r1[3];
            r1[3] = (v448_data + (v430_data * v446_data));
            float v451_data = s0[54];
            float v453_data = r1[4];
            r1[4] = (v453_data + (v430_data * v451_data));
            float v456_data = s0[66];
            float v458_data = r1[5];
            r1[5] = (v458_data + (v430_data * v456_data));
            float v461_data = s0[78];
            float v463_data = r1[6];
            r1[6] = (v463_data + (v430_data * v461_data));
            float v466_data = s0[90];
            float v468_data = r1[7];
            r1[7] = (v468_data + (v430_data * v466_data));
            float v471_data = s0[102];
            float v473_data = r1[8];
            r1[8] = (v473_data + (v430_data * v471_data));
            float v476_data = s0[114];
            float v478_data = r1[9];
            r1[9] = (v478_data + (v430_data * v476_data));
            float v481_data = s0[126];
            float v483_data = r1[10];
            r1[10] = (v483_data + (v430_data * v481_data));
            float v486_data = s0[138];
            float v488_data = r1[11];
            r1[11] = (v488_data + (v430_data * v486_data));
          }
          if (v12_lead < 6) {
            float v494_data = r0[7];
            float v495_data = s0[7];
            float v497_data = r1[0];
            r1[0] = (v497_data + (v494_data * v495_data));
            float v500_data = s0[19];
            float v502_data = r1[1];
            r1[1] = (v502_data + (v494_data * v500_data));
            float v505_data = s0[31];
            float v507_data = r1[2];
            r1[2] = (v507_data + (v494_data * v505_data));
            float v510_data = s0[43];
            float v512_data = r1[3];
            r1[3] = (v512_data + (v494_data * v510_data));
            float v515_data = s0[55];
            float v517_data = r1[4];
            r1[4] = (v517_data + (v494_data * v515_data));
            float v520_data = s0[67];
            float v522_data = r1[5];
            r1[5] = (v522_data + (v494_data * v520_data));
            float v525_data = s0[79];
            float v527_data = r1[6];
            r1[6] = (v527_data + (v494_data * v525_data));
            float v530_data = s0[91];
            float v532_data = r1[7];
            r1[7] = (v532_data + (v494_data * v530_data));
            float v535_data = s0[103];
            float v537_data = r1[8];
            r1[8] = (v537_data + (v494_data * v535_data));
            float v540_data = s0[115];
            float v542_data = r1[9];
            r1[9] = (v542_data + (v494_data * v540_data));
            float v545_data = s0[127];
            float v547_data = r1[10];
            r1[10] = (v547_data + (v494_data * v545_data));
            float v550_data = s0[139];
            float v552_data = r1[11];
            r1[11] = (v552_data + (v494_data * v550_data));
          }
          if (v12_lead < 6) {
            float v558_data = r0[8];
            float v559_data = s0[8];
            float v561_data = r1[0];
            r1[0] = (v561_data + (v558_data * v559_data));
            float v564_data = s0[20];
            float v566_data = r1[1];
            r1[1] = (v566_data + (v558_data * v564_data));
            float v569_data = s0[32];
            float v571_data = r1[2];
            r1[2] = (v571_data + (v558_data * v569_data));
            float v574_data = s0[44];
            float v576_data = r1[3];
            r1[3] = (v576_data + (v558_data * v574_data));
            float v579_data = s0[56];
            float v581_data = r1[4];
            r1[4] = (v581_data + (v558_data * v579_data));
            float v584_data = s0[68];
            float v586_data = r1[5];
            r1[5] = (v586_data + (v558_data * v584_data));
            float v589_data = s0[80];
            float v591_data = r1[6];
            r1[6] = (v591_data + (v558_data * v589_data));
            float v594_data = s0[92];
            float v596_data = r1[7];
            r1[7] = (v596_data + (v558_data * v594_data));
            float v599_data = s0[104];
            float v601_data = r1[8];
            r1[8] = (v601_data + (v558_data * v599_data));
            float v604_data = s0[116];
            float v606_data = r1[9];
            r1[9] = (v606_data + (v558_data * v604_data));
            float v609_data = s0[128];
            float v611_data = r1[10];
            r1[10] = (v611_data + (v558_data * v609_data));
            float v614_data = s0[140];
            float v616_data = r1[11];
            r1[11] = (v616_data + (v558_data * v614_data));
          }
          if (v12_lead < 6) {
            float v622_data = r0[9];
            float v623_data = s0[9];
            float v625_data = r1[0];
            r1[0] = (v625_data + (v622_data * v623_data));
            float v628_data = s0[21];
            float v630_data = r1[1];
            r1[1] = (v630_data + (v622_data * v628_data));
            float v633_data = s0[33];
            float v635_data = r1[2];
            r1[2] = (v635_data + (v622_data * v633_data));
            float v638_data = s0[45];
            float v640_data = r1[3];
            r1[3] = (v640_data + (v622_data * v638_data));
            float v643_data = s0[57];
            float v645_data = r1[4];
            r1[4] = (v645_data + (v622_data * v643_data));
            float v648_data = s0[69];
            float v650_data = r1[5];
            r1[5] = (v650_data + (v622_data * v648_data));
            float v653_data = s0[81];
            float v655_data = r1[6];
            r1[6] = (v655_data + (v622_data * v653_data));
            float v658_data = s0[93];
            float v660_data = r1[7];
            r1[7] = (v660_data + (v622_data * v658_data));
            float v663_data = s0[105];
            float v665_data = r1[8];
            r1[8] = (v665_data + (v622_data * v663_data));
            float v668_data = s0[117];
            float v670_data = r1[9];
            r1[9] = (v670_data + (v622_data * v668_data));
            float v673_data = s0[129];
            float v675_data = r1[10];
            r1[10] = (v675_data + (v622_data * v673_data));
            float v678_data = s0[141];
            float v680_data = r1[11];
            r1[11] = (v680_data + (v622_data * v678_data));
          }
          if (v12_lead < 6) {
            float v686_data = r0[10];
            float v687_data = s0[10];
            float v689_data = r1[0];
            r1[0] = (v689_data + (v686_data * v687_data));
            float v692_data = s0[22];
            float v694_data = r1[1];
            r1[1] = (v694_data + (v686_data * v692_data));
            float v697_data = s0[34];
            float v699_data = r1[2];
            r1[2] = (v699_data + (v686_data * v697_data));
            float v702_data = s0[46];
            float v704_data = r1[3];
            r1[3] = (v704_data + (v686_data * v702_data));
            float v707_data = s0[58];
            float v709_data = r1[4];
            r1[4] = (v709_data + (v686_data * v707_data));
            float v712_data = s0[70];
            float v714_data = r1[5];
            r1[5] = (v714_data + (v686_data * v712_data));
            float v717_data = s0[82];
            float v719_data = r1[6];
            r1[6] = (v719_data + (v686_data * v717_data));
            float v722_data = s0[94];
            float v724_data = r1[7];
            r1[7] = (v724_data + (v686_data * v722_data));
            float v727_data = s0[106];
            float v729_data = r1[8];
            r1[8] = (v729_data + (v686_data * v727_data));
            float v732_data = s0[118];
            float v734_data = r1[9];
            r1[9] = (v734_data + (v686_data * v732_data));
            float v737_data = s0[130];
            float v739_data = r1[10];
            r1[10] = (v739_data + (v686_data * v737_data));
            float v742_data = s0[142];
            float v744_data = r1[11];
            r1[11] = (v744_data + (v686_data * v742_data));
          }
          if (v12_lead < 6) {
            float v750_data = r0[11];
            float v751_data = s0[11];
            float v753_data = r1[0];
            r1[0] = (v753_data + (v750_data * v751_data));
            float v756_data = s0[23];
            float v758_data = r1[1];
            r1[1] = (v758_data + (v750_data * v756_data));
            float v761_data = s0[35];
            float v763_data = r1[2];
            r1[2] = (v763_data + (v750_data * v761_data));
            float v766_data = s0[47];
            float v768_data = r1[3];
            r1[3] = (v768_data + (v750_data * v766_data));
            float v771_data = s0[59];
            float v773_data = r1[4];
            r1[4] = (v773_data + (v750_data * v771_data));
            float v776_data = s0[71];
            float v778_data = r1[5];
            r1[5] = (v778_data + (v750_data * v776_data));
            float v781_data = s0[83];
            float v783_data = r1[6];
            r1[6] = (v783_data + (v750_data * v781_data));
            float v786_data = s0[95];
            float v788_data = r1[7];
            r1[7] = (v788_data + (v750_data * v786_data));
            float v791_data = s0[107];
            float v793_data = r1[8];
            r1[8] = (v793_data + (v750_data * v791_data));
            float v796_data = s0[119];
            float v798_data = r1[9];
            r1[9] = (v798_data + (v750_data * v796_data));
            float v801_data = s0[131];
            float v803_data = r1[10];
            r1[10] = (v803_data + (v750_data * v801_data));
            float v806_data = s0[143];
            float v808_data = r1[11];
            r1[11] = (v808_data + (v750_data * v806_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v815_i1 = 0; v815_i1 < 12; ++v815_i1) {
              float v817_data = r1[v815_i1];
              s1[(v12_lead + (v815_i1 * 12))] = v817_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v830_i1 = 0; v830_i1 < 12; ++v830_i1) {
              float v838_data = __ldcg(&glb_m4[(v12_lead + (v830_i1 * 12))]);
              r4[v830_i1] = v838_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v12_lead < 6) {
            float v846_data = r2[0];
            float v847_data = s0[0];
            float v849_data = ir3[0];
            ir3[0] = (v849_data + (v846_data * v847_data));
            float v852_data = s0[12];
            float v854_data = ir3[1];
            ir3[1] = (v854_data + (v846_data * v852_data));
            float v857_data = s0[24];
            float v859_data = ir3[2];
            ir3[2] = (v859_data + (v846_data * v857_data));
            float v862_data = s0[36];
            float v864_data = ir3[3];
            ir3[3] = (v864_data + (v846_data * v862_data));
            float v867_data = s0[48];
            float v869_data = ir3[4];
            ir3[4] = (v869_data + (v846_data * v867_data));
            float v872_data = s0[60];
            float v874_data = ir3[5];
            ir3[5] = (v874_data + (v846_data * v872_data));
            float v877_data = s0[72];
            float v879_data = ir3[6];
            ir3[6] = (v879_data + (v846_data * v877_data));
            float v882_data = s0[84];
            float v884_data = ir3[7];
            ir3[7] = (v884_data + (v846_data * v882_data));
            float v887_data = s0[96];
            float v889_data = ir3[8];
            ir3[8] = (v889_data + (v846_data * v887_data));
            float v892_data = s0[108];
            float v894_data = ir3[9];
            ir3[9] = (v894_data + (v846_data * v892_data));
            float v897_data = s0[120];
            float v899_data = ir3[10];
            ir3[10] = (v899_data + (v846_data * v897_data));
            float v902_data = s0[132];
            float v904_data = ir3[11];
            ir3[11] = (v904_data + (v846_data * v902_data));
          }
          if (v12_lead < 6) {
            float v910_data = r2[1];
            float v911_data = s0[1];
            float v913_data = ir3[0];
            ir3[0] = (v913_data + (v910_data * v911_data));
            float v916_data = s0[13];
            float v918_data = ir3[1];
            ir3[1] = (v918_data + (v910_data * v916_data));
            float v921_data = s0[25];
            float v923_data = ir3[2];
            ir3[2] = (v923_data + (v910_data * v921_data));
            float v926_data = s0[37];
            float v928_data = ir3[3];
            ir3[3] = (v928_data + (v910_data * v926_data));
            float v931_data = s0[49];
            float v933_data = ir3[4];
            ir3[4] = (v933_data + (v910_data * v931_data));
            float v936_data = s0[61];
            float v938_data = ir3[5];
            ir3[5] = (v938_data + (v910_data * v936_data));
            float v941_data = s0[73];
            float v943_data = ir3[6];
            ir3[6] = (v943_data + (v910_data * v941_data));
            float v946_data = s0[85];
            float v948_data = ir3[7];
            ir3[7] = (v948_data + (v910_data * v946_data));
            float v951_data = s0[97];
            float v953_data = ir3[8];
            ir3[8] = (v953_data + (v910_data * v951_data));
            float v956_data = s0[109];
            float v958_data = ir3[9];
            ir3[9] = (v958_data + (v910_data * v956_data));
            float v961_data = s0[121];
            float v963_data = ir3[10];
            ir3[10] = (v963_data + (v910_data * v961_data));
            float v966_data = s0[133];
            float v968_data = ir3[11];
            ir3[11] = (v968_data + (v910_data * v966_data));
          }
          if (v12_lead < 6) {
            float v974_data = r2[2];
            float v975_data = s0[2];
            float v977_data = ir3[0];
            ir3[0] = (v977_data + (v974_data * v975_data));
            float v980_data = s0[14];
            float v982_data = ir3[1];
            ir3[1] = (v982_data + (v974_data * v980_data));
            float v985_data = s0[26];
            float v987_data = ir3[2];
            ir3[2] = (v987_data + (v974_data * v985_data));
            float v990_data = s0[38];
            float v992_data = ir3[3];
            ir3[3] = (v992_data + (v974_data * v990_data));
            float v995_data = s0[50];
            float v997_data = ir3[4];
            ir3[4] = (v997_data + (v974_data * v995_data));
            float v1000_data = s0[62];
            float v1002_data = ir3[5];
            ir3[5] = (v1002_data + (v974_data * v1000_data));
            float v1005_data = s0[74];
            float v1007_data = ir3[6];
            ir3[6] = (v1007_data + (v974_data * v1005_data));
            float v1010_data = s0[86];
            float v1012_data = ir3[7];
            ir3[7] = (v1012_data + (v974_data * v1010_data));
            float v1015_data = s0[98];
            float v1017_data = ir3[8];
            ir3[8] = (v1017_data + (v974_data * v1015_data));
            float v1020_data = s0[110];
            float v1022_data = ir3[9];
            ir3[9] = (v1022_data + (v974_data * v1020_data));
            float v1025_data = s0[122];
            float v1027_data = ir3[10];
            ir3[10] = (v1027_data + (v974_data * v1025_data));
            float v1030_data = s0[134];
            float v1032_data = ir3[11];
            ir3[11] = (v1032_data + (v974_data * v1030_data));
          }
          if (v12_lead < 6) {
            float v1038_data = r2[3];
            float v1039_data = s0[3];
            float v1041_data = ir3[0];
            ir3[0] = (v1041_data + (v1038_data * v1039_data));
            float v1044_data = s0[15];
            float v1046_data = ir3[1];
            ir3[1] = (v1046_data + (v1038_data * v1044_data));
            float v1049_data = s0[27];
            float v1051_data = ir3[2];
            ir3[2] = (v1051_data + (v1038_data * v1049_data));
            float v1054_data = s0[39];
            float v1056_data = ir3[3];
            ir3[3] = (v1056_data + (v1038_data * v1054_data));
            float v1059_data = s0[51];
            float v1061_data = ir3[4];
            ir3[4] = (v1061_data + (v1038_data * v1059_data));
            float v1064_data = s0[63];
            float v1066_data = ir3[5];
            ir3[5] = (v1066_data + (v1038_data * v1064_data));
            float v1069_data = s0[75];
            float v1071_data = ir3[6];
            ir3[6] = (v1071_data + (v1038_data * v1069_data));
            float v1074_data = s0[87];
            float v1076_data = ir3[7];
            ir3[7] = (v1076_data + (v1038_data * v1074_data));
            float v1079_data = s0[99];
            float v1081_data = ir3[8];
            ir3[8] = (v1081_data + (v1038_data * v1079_data));
            float v1084_data = s0[111];
            float v1086_data = ir3[9];
            ir3[9] = (v1086_data + (v1038_data * v1084_data));
            float v1089_data = s0[123];
            float v1091_data = ir3[10];
            ir3[10] = (v1091_data + (v1038_data * v1089_data));
            float v1094_data = s0[135];
            float v1096_data = ir3[11];
            ir3[11] = (v1096_data + (v1038_data * v1094_data));
          }
          if (v12_lead < 6) {
            float v1102_data = r2[4];
            float v1103_data = s0[4];
            float v1105_data = ir3[0];
            ir3[0] = (v1105_data + (v1102_data * v1103_data));
            float v1108_data = s0[16];
            float v1110_data = ir3[1];
            ir3[1] = (v1110_data + (v1102_data * v1108_data));
            float v1113_data = s0[28];
            float v1115_data = ir3[2];
            ir3[2] = (v1115_data + (v1102_data * v1113_data));
            float v1118_data = s0[40];
            float v1120_data = ir3[3];
            ir3[3] = (v1120_data + (v1102_data * v1118_data));
            float v1123_data = s0[52];
            float v1125_data = ir3[4];
            ir3[4] = (v1125_data + (v1102_data * v1123_data));
            float v1128_data = s0[64];
            float v1130_data = ir3[5];
            ir3[5] = (v1130_data + (v1102_data * v1128_data));
            float v1133_data = s0[76];
            float v1135_data = ir3[6];
            ir3[6] = (v1135_data + (v1102_data * v1133_data));
            float v1138_data = s0[88];
            float v1140_data = ir3[7];
            ir3[7] = (v1140_data + (v1102_data * v1138_data));
            float v1143_data = s0[100];
            float v1145_data = ir3[8];
            ir3[8] = (v1145_data + (v1102_data * v1143_data));
            float v1148_data = s0[112];
            float v1150_data = ir3[9];
            ir3[9] = (v1150_data + (v1102_data * v1148_data));
            float v1153_data = s0[124];
            float v1155_data = ir3[10];
            ir3[10] = (v1155_data + (v1102_data * v1153_data));
            float v1158_data = s0[136];
            float v1160_data = ir3[11];
            ir3[11] = (v1160_data + (v1102_data * v1158_data));
          }
          if (v12_lead < 6) {
            float v1166_data = r2[5];
            float v1167_data = s0[5];
            float v1169_data = ir3[0];
            ir3[0] = (v1169_data + (v1166_data * v1167_data));
            float v1172_data = s0[17];
            float v1174_data = ir3[1];
            ir3[1] = (v1174_data + (v1166_data * v1172_data));
            float v1177_data = s0[29];
            float v1179_data = ir3[2];
            ir3[2] = (v1179_data + (v1166_data * v1177_data));
            float v1182_data = s0[41];
            float v1184_data = ir3[3];
            ir3[3] = (v1184_data + (v1166_data * v1182_data));
            float v1187_data = s0[53];
            float v1189_data = ir3[4];
            ir3[4] = (v1189_data + (v1166_data * v1187_data));
            float v1192_data = s0[65];
            float v1194_data = ir3[5];
            ir3[5] = (v1194_data + (v1166_data * v1192_data));
            float v1197_data = s0[77];
            float v1199_data = ir3[6];
            ir3[6] = (v1199_data + (v1166_data * v1197_data));
            float v1202_data = s0[89];
            float v1204_data = ir3[7];
            ir3[7] = (v1204_data + (v1166_data * v1202_data));
            float v1207_data = s0[101];
            float v1209_data = ir3[8];
            ir3[8] = (v1209_data + (v1166_data * v1207_data));
            float v1212_data = s0[113];
            float v1214_data = ir3[9];
            ir3[9] = (v1214_data + (v1166_data * v1212_data));
            float v1217_data = s0[125];
            float v1219_data = ir3[10];
            ir3[10] = (v1219_data + (v1166_data * v1217_data));
            float v1222_data = s0[137];
            float v1224_data = ir3[11];
            ir3[11] = (v1224_data + (v1166_data * v1222_data));
          }
          if (v12_lead < 6) {
            float v1230_data = r2[6];
            float v1231_data = s0[6];
            float v1233_data = ir3[0];
            ir3[0] = (v1233_data + (v1230_data * v1231_data));
            float v1236_data = s0[18];
            float v1238_data = ir3[1];
            ir3[1] = (v1238_data + (v1230_data * v1236_data));
            float v1241_data = s0[30];
            float v1243_data = ir3[2];
            ir3[2] = (v1243_data + (v1230_data * v1241_data));
            float v1246_data = s0[42];
            float v1248_data = ir3[3];
            ir3[3] = (v1248_data + (v1230_data * v1246_data));
            float v1251_data = s0[54];
            float v1253_data = ir3[4];
            ir3[4] = (v1253_data + (v1230_data * v1251_data));
            float v1256_data = s0[66];
            float v1258_data = ir3[5];
            ir3[5] = (v1258_data + (v1230_data * v1256_data));
            float v1261_data = s0[78];
            float v1263_data = ir3[6];
            ir3[6] = (v1263_data + (v1230_data * v1261_data));
            float v1266_data = s0[90];
            float v1268_data = ir3[7];
            ir3[7] = (v1268_data + (v1230_data * v1266_data));
            float v1271_data = s0[102];
            float v1273_data = ir3[8];
            ir3[8] = (v1273_data + (v1230_data * v1271_data));
            float v1276_data = s0[114];
            float v1278_data = ir3[9];
            ir3[9] = (v1278_data + (v1230_data * v1276_data));
            float v1281_data = s0[126];
            float v1283_data = ir3[10];
            ir3[10] = (v1283_data + (v1230_data * v1281_data));
            float v1286_data = s0[138];
            float v1288_data = ir3[11];
            ir3[11] = (v1288_data + (v1230_data * v1286_data));
          }
          if (v12_lead < 6) {
            float v1294_data = r2[7];
            float v1295_data = s0[7];
            float v1297_data = ir3[0];
            ir3[0] = (v1297_data + (v1294_data * v1295_data));
            float v1300_data = s0[19];
            float v1302_data = ir3[1];
            ir3[1] = (v1302_data + (v1294_data * v1300_data));
            float v1305_data = s0[31];
            float v1307_data = ir3[2];
            ir3[2] = (v1307_data + (v1294_data * v1305_data));
            float v1310_data = s0[43];
            float v1312_data = ir3[3];
            ir3[3] = (v1312_data + (v1294_data * v1310_data));
            float v1315_data = s0[55];
            float v1317_data = ir3[4];
            ir3[4] = (v1317_data + (v1294_data * v1315_data));
            float v1320_data = s0[67];
            float v1322_data = ir3[5];
            ir3[5] = (v1322_data + (v1294_data * v1320_data));
            float v1325_data = s0[79];
            float v1327_data = ir3[6];
            ir3[6] = (v1327_data + (v1294_data * v1325_data));
            float v1330_data = s0[91];
            float v1332_data = ir3[7];
            ir3[7] = (v1332_data + (v1294_data * v1330_data));
            float v1335_data = s0[103];
            float v1337_data = ir3[8];
            ir3[8] = (v1337_data + (v1294_data * v1335_data));
            float v1340_data = s0[115];
            float v1342_data = ir3[9];
            ir3[9] = (v1342_data + (v1294_data * v1340_data));
            float v1345_data = s0[127];
            float v1347_data = ir3[10];
            ir3[10] = (v1347_data + (v1294_data * v1345_data));
            float v1350_data = s0[139];
            float v1352_data = ir3[11];
            ir3[11] = (v1352_data + (v1294_data * v1350_data));
          }
          if (v12_lead < 6) {
            float v1358_data = r2[8];
            float v1359_data = s0[8];
            float v1361_data = ir3[0];
            ir3[0] = (v1361_data + (v1358_data * v1359_data));
            float v1364_data = s0[20];
            float v1366_data = ir3[1];
            ir3[1] = (v1366_data + (v1358_data * v1364_data));
            float v1369_data = s0[32];
            float v1371_data = ir3[2];
            ir3[2] = (v1371_data + (v1358_data * v1369_data));
            float v1374_data = s0[44];
            float v1376_data = ir3[3];
            ir3[3] = (v1376_data + (v1358_data * v1374_data));
            float v1379_data = s0[56];
            float v1381_data = ir3[4];
            ir3[4] = (v1381_data + (v1358_data * v1379_data));
            float v1384_data = s0[68];
            float v1386_data = ir3[5];
            ir3[5] = (v1386_data + (v1358_data * v1384_data));
            float v1389_data = s0[80];
            float v1391_data = ir3[6];
            ir3[6] = (v1391_data + (v1358_data * v1389_data));
            float v1394_data = s0[92];
            float v1396_data = ir3[7];
            ir3[7] = (v1396_data + (v1358_data * v1394_data));
            float v1399_data = s0[104];
            float v1401_data = ir3[8];
            ir3[8] = (v1401_data + (v1358_data * v1399_data));
            float v1404_data = s0[116];
            float v1406_data = ir3[9];
            ir3[9] = (v1406_data + (v1358_data * v1404_data));
            float v1409_data = s0[128];
            float v1411_data = ir3[10];
            ir3[10] = (v1411_data + (v1358_data * v1409_data));
            float v1414_data = s0[140];
            float v1416_data = ir3[11];
            ir3[11] = (v1416_data + (v1358_data * v1414_data));
          }
          if (v12_lead < 6) {
            float v1422_data = r2[9];
            float v1423_data = s0[9];
            float v1425_data = ir3[0];
            ir3[0] = (v1425_data + (v1422_data * v1423_data));
            float v1428_data = s0[21];
            float v1430_data = ir3[1];
            ir3[1] = (v1430_data + (v1422_data * v1428_data));
            float v1433_data = s0[33];
            float v1435_data = ir3[2];
            ir3[2] = (v1435_data + (v1422_data * v1433_data));
            float v1438_data = s0[45];
            float v1440_data = ir3[3];
            ir3[3] = (v1440_data + (v1422_data * v1438_data));
            float v1443_data = s0[57];
            float v1445_data = ir3[4];
            ir3[4] = (v1445_data + (v1422_data * v1443_data));
            float v1448_data = s0[69];
            float v1450_data = ir3[5];
            ir3[5] = (v1450_data + (v1422_data * v1448_data));
            float v1453_data = s0[81];
            float v1455_data = ir3[6];
            ir3[6] = (v1455_data + (v1422_data * v1453_data));
            float v1458_data = s0[93];
            float v1460_data = ir3[7];
            ir3[7] = (v1460_data + (v1422_data * v1458_data));
            float v1463_data = s0[105];
            float v1465_data = ir3[8];
            ir3[8] = (v1465_data + (v1422_data * v1463_data));
            float v1468_data = s0[117];
            float v1470_data = ir3[9];
            ir3[9] = (v1470_data + (v1422_data * v1468_data));
            float v1473_data = s0[129];
            float v1475_data = ir3[10];
            ir3[10] = (v1475_data + (v1422_data * v1473_data));
            float v1478_data = s0[141];
            float v1480_data = ir3[11];
            ir3[11] = (v1480_data + (v1422_data * v1478_data));
          }
          if (v12_lead < 6) {
            float v1486_data = r2[10];
            float v1487_data = s0[10];
            float v1489_data = ir3[0];
            ir3[0] = (v1489_data + (v1486_data * v1487_data));
            float v1492_data = s0[22];
            float v1494_data = ir3[1];
            ir3[1] = (v1494_data + (v1486_data * v1492_data));
            float v1497_data = s0[34];
            float v1499_data = ir3[2];
            ir3[2] = (v1499_data + (v1486_data * v1497_data));
            float v1502_data = s0[46];
            float v1504_data = ir3[3];
            ir3[3] = (v1504_data + (v1486_data * v1502_data));
            float v1507_data = s0[58];
            float v1509_data = ir3[4];
            ir3[4] = (v1509_data + (v1486_data * v1507_data));
            float v1512_data = s0[70];
            float v1514_data = ir3[5];
            ir3[5] = (v1514_data + (v1486_data * v1512_data));
            float v1517_data = s0[82];
            float v1519_data = ir3[6];
            ir3[6] = (v1519_data + (v1486_data * v1517_data));
            float v1522_data = s0[94];
            float v1524_data = ir3[7];
            ir3[7] = (v1524_data + (v1486_data * v1522_data));
            float v1527_data = s0[106];
            float v1529_data = ir3[8];
            ir3[8] = (v1529_data + (v1486_data * v1527_data));
            float v1532_data = s0[118];
            float v1534_data = ir3[9];
            ir3[9] = (v1534_data + (v1486_data * v1532_data));
            float v1537_data = s0[130];
            float v1539_data = ir3[10];
            ir3[10] = (v1539_data + (v1486_data * v1537_data));
            float v1542_data = s0[142];
            float v1544_data = ir3[11];
            ir3[11] = (v1544_data + (v1486_data * v1542_data));
          }
          if (v12_lead < 6) {
            float v1550_data = r2[11];
            float v1551_data = s0[11];
            float v1553_data = ir3[0];
            ir3[0] = (v1553_data + (v1550_data * v1551_data));
            float v1556_data = s0[23];
            float v1558_data = ir3[1];
            ir3[1] = (v1558_data + (v1550_data * v1556_data));
            float v1561_data = s0[35];
            float v1563_data = ir3[2];
            ir3[2] = (v1563_data + (v1550_data * v1561_data));
            float v1566_data = s0[47];
            float v1568_data = ir3[3];
            ir3[3] = (v1568_data + (v1550_data * v1566_data));
            float v1571_data = s0[59];
            float v1573_data = ir3[4];
            ir3[4] = (v1573_data + (v1550_data * v1571_data));
            float v1576_data = s0[71];
            float v1578_data = ir3[5];
            ir3[5] = (v1578_data + (v1550_data * v1576_data));
            float v1581_data = s0[83];
            float v1583_data = ir3[6];
            ir3[6] = (v1583_data + (v1550_data * v1581_data));
            float v1586_data = s0[95];
            float v1588_data = ir3[7];
            ir3[7] = (v1588_data + (v1550_data * v1586_data));
            float v1591_data = s0[107];
            float v1593_data = ir3[8];
            ir3[8] = (v1593_data + (v1550_data * v1591_data));
            float v1596_data = s0[119];
            float v1598_data = ir3[9];
            ir3[9] = (v1598_data + (v1550_data * v1596_data));
            float v1601_data = s0[131];
            float v1603_data = ir3[10];
            ir3[10] = (v1603_data + (v1550_data * v1601_data));
            float v1606_data = s0[143];
            float v1608_data = ir3[11];
            ir3[11] = (v1608_data + (v1550_data * v1606_data));
          }
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v1614_n1 = 0; v1614_n1 < 12; ++v1614_n1) {
              float v1616_data = ir3[v1614_n1];
              r3[v1614_n1] = v1616_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v12_lead < 6) {
            int32_t v1630_off = v12_lead + 6;
            #pragma unroll
            for (int32_t v1622_i1 = 0; v1622_i1 < 12; ++v1622_i1) {
              float v1624_data = r3[v1622_i1];
              s1[(v1630_off + (v1622_i1 * 12))] = v1624_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v12_lead < 12) {
            float v1639_data = r4[0];
            float v1640_data = s1[0];
            float v1642_data = ir5[0];
            ir5[0] = (v1642_data + (v1639_data * v1640_data));
            float v1645_data = s1[12];
            float v1647_data = ir5[1];
            ir5[1] = (v1647_data + (v1639_data * v1645_data));
            float v1650_data = s1[24];
            float v1652_data = ir5[2];
            ir5[2] = (v1652_data + (v1639_data * v1650_data));
            float v1655_data = s1[36];
            float v1657_data = ir5[3];
            ir5[3] = (v1657_data + (v1639_data * v1655_data));
            float v1660_data = s1[48];
            float v1662_data = ir5[4];
            ir5[4] = (v1662_data + (v1639_data * v1660_data));
            float v1665_data = s1[60];
            float v1667_data = ir5[5];
            ir5[5] = (v1667_data + (v1639_data * v1665_data));
            float v1670_data = s1[72];
            float v1672_data = ir5[6];
            ir5[6] = (v1672_data + (v1639_data * v1670_data));
            float v1675_data = s1[84];
            float v1677_data = ir5[7];
            ir5[7] = (v1677_data + (v1639_data * v1675_data));
            float v1680_data = s1[96];
            float v1682_data = ir5[8];
            ir5[8] = (v1682_data + (v1639_data * v1680_data));
            float v1685_data = s1[108];
            float v1687_data = ir5[9];
            ir5[9] = (v1687_data + (v1639_data * v1685_data));
            float v1690_data = s1[120];
            float v1692_data = ir5[10];
            ir5[10] = (v1692_data + (v1639_data * v1690_data));
            float v1695_data = s1[132];
            float v1697_data = ir5[11];
            ir5[11] = (v1697_data + (v1639_data * v1695_data));
          }
          if (v12_lead < 12) {
            float v1703_data = r4[1];
            float v1704_data = s1[1];
            float v1706_data = ir5[0];
            ir5[0] = (v1706_data + (v1703_data * v1704_data));
            float v1709_data = s1[13];
            float v1711_data = ir5[1];
            ir5[1] = (v1711_data + (v1703_data * v1709_data));
            float v1714_data = s1[25];
            float v1716_data = ir5[2];
            ir5[2] = (v1716_data + (v1703_data * v1714_data));
            float v1719_data = s1[37];
            float v1721_data = ir5[3];
            ir5[3] = (v1721_data + (v1703_data * v1719_data));
            float v1724_data = s1[49];
            float v1726_data = ir5[4];
            ir5[4] = (v1726_data + (v1703_data * v1724_data));
            float v1729_data = s1[61];
            float v1731_data = ir5[5];
            ir5[5] = (v1731_data + (v1703_data * v1729_data));
            float v1734_data = s1[73];
            float v1736_data = ir5[6];
            ir5[6] = (v1736_data + (v1703_data * v1734_data));
            float v1739_data = s1[85];
            float v1741_data = ir5[7];
            ir5[7] = (v1741_data + (v1703_data * v1739_data));
            float v1744_data = s1[97];
            float v1746_data = ir5[8];
            ir5[8] = (v1746_data + (v1703_data * v1744_data));
            float v1749_data = s1[109];
            float v1751_data = ir5[9];
            ir5[9] = (v1751_data + (v1703_data * v1749_data));
            float v1754_data = s1[121];
            float v1756_data = ir5[10];
            ir5[10] = (v1756_data + (v1703_data * v1754_data));
            float v1759_data = s1[133];
            float v1761_data = ir5[11];
            ir5[11] = (v1761_data + (v1703_data * v1759_data));
          }
          if (v12_lead < 12) {
            float v1767_data = r4[2];
            float v1768_data = s1[2];
            float v1770_data = ir5[0];
            ir5[0] = (v1770_data + (v1767_data * v1768_data));
            float v1773_data = s1[14];
            float v1775_data = ir5[1];
            ir5[1] = (v1775_data + (v1767_data * v1773_data));
            float v1778_data = s1[26];
            float v1780_data = ir5[2];
            ir5[2] = (v1780_data + (v1767_data * v1778_data));
            float v1783_data = s1[38];
            float v1785_data = ir5[3];
            ir5[3] = (v1785_data + (v1767_data * v1783_data));
            float v1788_data = s1[50];
            float v1790_data = ir5[4];
            ir5[4] = (v1790_data + (v1767_data * v1788_data));
            float v1793_data = s1[62];
            float v1795_data = ir5[5];
            ir5[5] = (v1795_data + (v1767_data * v1793_data));
            float v1798_data = s1[74];
            float v1800_data = ir5[6];
            ir5[6] = (v1800_data + (v1767_data * v1798_data));
            float v1803_data = s1[86];
            float v1805_data = ir5[7];
            ir5[7] = (v1805_data + (v1767_data * v1803_data));
            float v1808_data = s1[98];
            float v1810_data = ir5[8];
            ir5[8] = (v1810_data + (v1767_data * v1808_data));
            float v1813_data = s1[110];
            float v1815_data = ir5[9];
            ir5[9] = (v1815_data + (v1767_data * v1813_data));
            float v1818_data = s1[122];
            float v1820_data = ir5[10];
            ir5[10] = (v1820_data + (v1767_data * v1818_data));
            float v1823_data = s1[134];
            float v1825_data = ir5[11];
            ir5[11] = (v1825_data + (v1767_data * v1823_data));
          }
          if (v12_lead < 12) {
            float v1831_data = r4[3];
            float v1832_data = s1[3];
            float v1834_data = ir5[0];
            ir5[0] = (v1834_data + (v1831_data * v1832_data));
            float v1837_data = s1[15];
            float v1839_data = ir5[1];
            ir5[1] = (v1839_data + (v1831_data * v1837_data));
            float v1842_data = s1[27];
            float v1844_data = ir5[2];
            ir5[2] = (v1844_data + (v1831_data * v1842_data));
            float v1847_data = s1[39];
            float v1849_data = ir5[3];
            ir5[3] = (v1849_data + (v1831_data * v1847_data));
            float v1852_data = s1[51];
            float v1854_data = ir5[4];
            ir5[4] = (v1854_data + (v1831_data * v1852_data));
            float v1857_data = s1[63];
            float v1859_data = ir5[5];
            ir5[5] = (v1859_data + (v1831_data * v1857_data));
            float v1862_data = s1[75];
            float v1864_data = ir5[6];
            ir5[6] = (v1864_data + (v1831_data * v1862_data));
            float v1867_data = s1[87];
            float v1869_data = ir5[7];
            ir5[7] = (v1869_data + (v1831_data * v1867_data));
            float v1872_data = s1[99];
            float v1874_data = ir5[8];
            ir5[8] = (v1874_data + (v1831_data * v1872_data));
            float v1877_data = s1[111];
            float v1879_data = ir5[9];
            ir5[9] = (v1879_data + (v1831_data * v1877_data));
            float v1882_data = s1[123];
            float v1884_data = ir5[10];
            ir5[10] = (v1884_data + (v1831_data * v1882_data));
            float v1887_data = s1[135];
            float v1889_data = ir5[11];
            ir5[11] = (v1889_data + (v1831_data * v1887_data));
          }
          if (v12_lead < 12) {
            float v1895_data = r4[4];
            float v1896_data = s1[4];
            float v1898_data = ir5[0];
            ir5[0] = (v1898_data + (v1895_data * v1896_data));
            float v1901_data = s1[16];
            float v1903_data = ir5[1];
            ir5[1] = (v1903_data + (v1895_data * v1901_data));
            float v1906_data = s1[28];
            float v1908_data = ir5[2];
            ir5[2] = (v1908_data + (v1895_data * v1906_data));
            float v1911_data = s1[40];
            float v1913_data = ir5[3];
            ir5[3] = (v1913_data + (v1895_data * v1911_data));
            float v1916_data = s1[52];
            float v1918_data = ir5[4];
            ir5[4] = (v1918_data + (v1895_data * v1916_data));
            float v1921_data = s1[64];
            float v1923_data = ir5[5];
            ir5[5] = (v1923_data + (v1895_data * v1921_data));
            float v1926_data = s1[76];
            float v1928_data = ir5[6];
            ir5[6] = (v1928_data + (v1895_data * v1926_data));
            float v1931_data = s1[88];
            float v1933_data = ir5[7];
            ir5[7] = (v1933_data + (v1895_data * v1931_data));
            float v1936_data = s1[100];
            float v1938_data = ir5[8];
            ir5[8] = (v1938_data + (v1895_data * v1936_data));
            float v1941_data = s1[112];
            float v1943_data = ir5[9];
            ir5[9] = (v1943_data + (v1895_data * v1941_data));
            float v1946_data = s1[124];
            float v1948_data = ir5[10];
            ir5[10] = (v1948_data + (v1895_data * v1946_data));
            float v1951_data = s1[136];
            float v1953_data = ir5[11];
            ir5[11] = (v1953_data + (v1895_data * v1951_data));
          }
          if (v12_lead < 12) {
            float v1959_data = r4[5];
            float v1960_data = s1[5];
            float v1962_data = ir5[0];
            ir5[0] = (v1962_data + (v1959_data * v1960_data));
            float v1965_data = s1[17];
            float v1967_data = ir5[1];
            ir5[1] = (v1967_data + (v1959_data * v1965_data));
            float v1970_data = s1[29];
            float v1972_data = ir5[2];
            ir5[2] = (v1972_data + (v1959_data * v1970_data));
            float v1975_data = s1[41];
            float v1977_data = ir5[3];
            ir5[3] = (v1977_data + (v1959_data * v1975_data));
            float v1980_data = s1[53];
            float v1982_data = ir5[4];
            ir5[4] = (v1982_data + (v1959_data * v1980_data));
            float v1985_data = s1[65];
            float v1987_data = ir5[5];
            ir5[5] = (v1987_data + (v1959_data * v1985_data));
            float v1990_data = s1[77];
            float v1992_data = ir5[6];
            ir5[6] = (v1992_data + (v1959_data * v1990_data));
            float v1995_data = s1[89];
            float v1997_data = ir5[7];
            ir5[7] = (v1997_data + (v1959_data * v1995_data));
            float v2000_data = s1[101];
            float v2002_data = ir5[8];
            ir5[8] = (v2002_data + (v1959_data * v2000_data));
            float v2005_data = s1[113];
            float v2007_data = ir5[9];
            ir5[9] = (v2007_data + (v1959_data * v2005_data));
            float v2010_data = s1[125];
            float v2012_data = ir5[10];
            ir5[10] = (v2012_data + (v1959_data * v2010_data));
            float v2015_data = s1[137];
            float v2017_data = ir5[11];
            ir5[11] = (v2017_data + (v1959_data * v2015_data));
          }
          if (v12_lead < 12) {
            float v2023_data = r4[6];
            float v2024_data = s1[6];
            float v2026_data = ir5[0];
            ir5[0] = (v2026_data + (v2023_data * v2024_data));
            float v2029_data = s1[18];
            float v2031_data = ir5[1];
            ir5[1] = (v2031_data + (v2023_data * v2029_data));
            float v2034_data = s1[30];
            float v2036_data = ir5[2];
            ir5[2] = (v2036_data + (v2023_data * v2034_data));
            float v2039_data = s1[42];
            float v2041_data = ir5[3];
            ir5[3] = (v2041_data + (v2023_data * v2039_data));
            float v2044_data = s1[54];
            float v2046_data = ir5[4];
            ir5[4] = (v2046_data + (v2023_data * v2044_data));
            float v2049_data = s1[66];
            float v2051_data = ir5[5];
            ir5[5] = (v2051_data + (v2023_data * v2049_data));
            float v2054_data = s1[78];
            float v2056_data = ir5[6];
            ir5[6] = (v2056_data + (v2023_data * v2054_data));
            float v2059_data = s1[90];
            float v2061_data = ir5[7];
            ir5[7] = (v2061_data + (v2023_data * v2059_data));
            float v2064_data = s1[102];
            float v2066_data = ir5[8];
            ir5[8] = (v2066_data + (v2023_data * v2064_data));
            float v2069_data = s1[114];
            float v2071_data = ir5[9];
            ir5[9] = (v2071_data + (v2023_data * v2069_data));
            float v2074_data = s1[126];
            float v2076_data = ir5[10];
            ir5[10] = (v2076_data + (v2023_data * v2074_data));
            float v2079_data = s1[138];
            float v2081_data = ir5[11];
            ir5[11] = (v2081_data + (v2023_data * v2079_data));
          }
          if (v12_lead < 12) {
            float v2087_data = r4[7];
            float v2088_data = s1[7];
            float v2090_data = ir5[0];
            ir5[0] = (v2090_data + (v2087_data * v2088_data));
            float v2093_data = s1[19];
            float v2095_data = ir5[1];
            ir5[1] = (v2095_data + (v2087_data * v2093_data));
            float v2098_data = s1[31];
            float v2100_data = ir5[2];
            ir5[2] = (v2100_data + (v2087_data * v2098_data));
            float v2103_data = s1[43];
            float v2105_data = ir5[3];
            ir5[3] = (v2105_data + (v2087_data * v2103_data));
            float v2108_data = s1[55];
            float v2110_data = ir5[4];
            ir5[4] = (v2110_data + (v2087_data * v2108_data));
            float v2113_data = s1[67];
            float v2115_data = ir5[5];
            ir5[5] = (v2115_data + (v2087_data * v2113_data));
            float v2118_data = s1[79];
            float v2120_data = ir5[6];
            ir5[6] = (v2120_data + (v2087_data * v2118_data));
            float v2123_data = s1[91];
            float v2125_data = ir5[7];
            ir5[7] = (v2125_data + (v2087_data * v2123_data));
            float v2128_data = s1[103];
            float v2130_data = ir5[8];
            ir5[8] = (v2130_data + (v2087_data * v2128_data));
            float v2133_data = s1[115];
            float v2135_data = ir5[9];
            ir5[9] = (v2135_data + (v2087_data * v2133_data));
            float v2138_data = s1[127];
            float v2140_data = ir5[10];
            ir5[10] = (v2140_data + (v2087_data * v2138_data));
            float v2143_data = s1[139];
            float v2145_data = ir5[11];
            ir5[11] = (v2145_data + (v2087_data * v2143_data));
          }
          if (v12_lead < 12) {
            float v2151_data = r4[8];
            float v2152_data = s1[8];
            float v2154_data = ir5[0];
            ir5[0] = (v2154_data + (v2151_data * v2152_data));
            float v2157_data = s1[20];
            float v2159_data = ir5[1];
            ir5[1] = (v2159_data + (v2151_data * v2157_data));
            float v2162_data = s1[32];
            float v2164_data = ir5[2];
            ir5[2] = (v2164_data + (v2151_data * v2162_data));
            float v2167_data = s1[44];
            float v2169_data = ir5[3];
            ir5[3] = (v2169_data + (v2151_data * v2167_data));
            float v2172_data = s1[56];
            float v2174_data = ir5[4];
            ir5[4] = (v2174_data + (v2151_data * v2172_data));
            float v2177_data = s1[68];
            float v2179_data = ir5[5];
            ir5[5] = (v2179_data + (v2151_data * v2177_data));
            float v2182_data = s1[80];
            float v2184_data = ir5[6];
            ir5[6] = (v2184_data + (v2151_data * v2182_data));
            float v2187_data = s1[92];
            float v2189_data = ir5[7];
            ir5[7] = (v2189_data + (v2151_data * v2187_data));
            float v2192_data = s1[104];
            float v2194_data = ir5[8];
            ir5[8] = (v2194_data + (v2151_data * v2192_data));
            float v2197_data = s1[116];
            float v2199_data = ir5[9];
            ir5[9] = (v2199_data + (v2151_data * v2197_data));
            float v2202_data = s1[128];
            float v2204_data = ir5[10];
            ir5[10] = (v2204_data + (v2151_data * v2202_data));
            float v2207_data = s1[140];
            float v2209_data = ir5[11];
            ir5[11] = (v2209_data + (v2151_data * v2207_data));
          }
          if (v12_lead < 12) {
            float v2215_data = r4[9];
            float v2216_data = s1[9];
            float v2218_data = ir5[0];
            ir5[0] = (v2218_data + (v2215_data * v2216_data));
            float v2221_data = s1[21];
            float v2223_data = ir5[1];
            ir5[1] = (v2223_data + (v2215_data * v2221_data));
            float v2226_data = s1[33];
            float v2228_data = ir5[2];
            ir5[2] = (v2228_data + (v2215_data * v2226_data));
            float v2231_data = s1[45];
            float v2233_data = ir5[3];
            ir5[3] = (v2233_data + (v2215_data * v2231_data));
            float v2236_data = s1[57];
            float v2238_data = ir5[4];
            ir5[4] = (v2238_data + (v2215_data * v2236_data));
            float v2241_data = s1[69];
            float v2243_data = ir5[5];
            ir5[5] = (v2243_data + (v2215_data * v2241_data));
            float v2246_data = s1[81];
            float v2248_data = ir5[6];
            ir5[6] = (v2248_data + (v2215_data * v2246_data));
            float v2251_data = s1[93];
            float v2253_data = ir5[7];
            ir5[7] = (v2253_data + (v2215_data * v2251_data));
            float v2256_data = s1[105];
            float v2258_data = ir5[8];
            ir5[8] = (v2258_data + (v2215_data * v2256_data));
            float v2261_data = s1[117];
            float v2263_data = ir5[9];
            ir5[9] = (v2263_data + (v2215_data * v2261_data));
            float v2266_data = s1[129];
            float v2268_data = ir5[10];
            ir5[10] = (v2268_data + (v2215_data * v2266_data));
            float v2271_data = s1[141];
            float v2273_data = ir5[11];
            ir5[11] = (v2273_data + (v2215_data * v2271_data));
          }
          if (v12_lead < 12) {
            float v2279_data = r4[10];
            float v2280_data = s1[10];
            float v2282_data = ir5[0];
            ir5[0] = (v2282_data + (v2279_data * v2280_data));
            float v2285_data = s1[22];
            float v2287_data = ir5[1];
            ir5[1] = (v2287_data + (v2279_data * v2285_data));
            float v2290_data = s1[34];
            float v2292_data = ir5[2];
            ir5[2] = (v2292_data + (v2279_data * v2290_data));
            float v2295_data = s1[46];
            float v2297_data = ir5[3];
            ir5[3] = (v2297_data + (v2279_data * v2295_data));
            float v2300_data = s1[58];
            float v2302_data = ir5[4];
            ir5[4] = (v2302_data + (v2279_data * v2300_data));
            float v2305_data = s1[70];
            float v2307_data = ir5[5];
            ir5[5] = (v2307_data + (v2279_data * v2305_data));
            float v2310_data = s1[82];
            float v2312_data = ir5[6];
            ir5[6] = (v2312_data + (v2279_data * v2310_data));
            float v2315_data = s1[94];
            float v2317_data = ir5[7];
            ir5[7] = (v2317_data + (v2279_data * v2315_data));
            float v2320_data = s1[106];
            float v2322_data = ir5[8];
            ir5[8] = (v2322_data + (v2279_data * v2320_data));
            float v2325_data = s1[118];
            float v2327_data = ir5[9];
            ir5[9] = (v2327_data + (v2279_data * v2325_data));
            float v2330_data = s1[130];
            float v2332_data = ir5[10];
            ir5[10] = (v2332_data + (v2279_data * v2330_data));
            float v2335_data = s1[142];
            float v2337_data = ir5[11];
            ir5[11] = (v2337_data + (v2279_data * v2335_data));
          }
          if (v12_lead < 12) {
            float v2343_data = r4[11];
            float v2344_data = s1[11];
            float v2346_data = ir5[0];
            ir5[0] = (v2346_data + (v2343_data * v2344_data));
            float v2349_data = s1[23];
            float v2351_data = ir5[1];
            ir5[1] = (v2351_data + (v2343_data * v2349_data));
            float v2354_data = s1[35];
            float v2356_data = ir5[2];
            ir5[2] = (v2356_data + (v2343_data * v2354_data));
            float v2359_data = s1[47];
            float v2361_data = ir5[3];
            ir5[3] = (v2361_data + (v2343_data * v2359_data));
            float v2364_data = s1[59];
            float v2366_data = ir5[4];
            ir5[4] = (v2366_data + (v2343_data * v2364_data));
            float v2369_data = s1[71];
            float v2371_data = ir5[5];
            ir5[5] = (v2371_data + (v2343_data * v2369_data));
            float v2374_data = s1[83];
            float v2376_data = ir5[6];
            ir5[6] = (v2376_data + (v2343_data * v2374_data));
            float v2379_data = s1[95];
            float v2381_data = ir5[7];
            ir5[7] = (v2381_data + (v2343_data * v2379_data));
            float v2384_data = s1[107];
            float v2386_data = ir5[8];
            ir5[8] = (v2386_data + (v2343_data * v2384_data));
            float v2389_data = s1[119];
            float v2391_data = ir5[9];
            ir5[9] = (v2391_data + (v2343_data * v2389_data));
            float v2394_data = s1[131];
            float v2396_data = ir5[10];
            ir5[10] = (v2396_data + (v2343_data * v2394_data));
            float v2399_data = s1[143];
            float v2401_data = ir5[11];
            ir5[11] = (v2401_data + (v2343_data * v2399_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2407_n1 = 0; v2407_n1 < 12; ++v2407_n1) {
              float v2409_data = ir5[v2407_n1];
              r5[v2407_n1] = v2409_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2415_i1 = 0; v2415_i1 < 12; ++v2415_i1) {
              float v2417_data = r5[v2415_i1];
              glb_m3[(v12_lead + (v2415_i1 * 12))] = v2417_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

