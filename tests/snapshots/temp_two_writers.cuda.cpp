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
            float v57_data = s0[25];
            float v59_data = r1[2];
            r1[2] = (v59_data + (v46_data * v57_data));
            float v62_data = s0[38];
            float v64_data = r1[3];
            r1[3] = (v64_data + (v46_data * v62_data));
            float v67_data = s0[51];
            float v69_data = r1[4];
            r1[4] = (v69_data + (v46_data * v67_data));
            float v72_data = s0[63];
            float v74_data = r1[5];
            r1[5] = (v74_data + (v46_data * v72_data));
            float v77_data = s0[76];
            float v79_data = r1[6];
            r1[6] = (v79_data + (v46_data * v77_data));
            float v82_data = s0[81];
            float v84_data = r1[7];
            r1[7] = (v84_data + (v46_data * v82_data));
            float v87_data = s0[102];
            float v89_data = r1[8];
            r1[8] = (v89_data + (v46_data * v87_data));
            float v92_data = s0[106];
            float v94_data = r1[9];
            r1[9] = (v94_data + (v46_data * v92_data));
            float v97_data = s0[127];
            float v99_data = r1[10];
            r1[10] = (v99_data + (v46_data * v97_data));
            float v102_data = s0[140];
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
            float v121_data = s0[24];
            float v123_data = r1[2];
            r1[2] = (v123_data + (v110_data * v121_data));
            float v126_data = s0[39];
            float v128_data = r1[3];
            r1[3] = (v128_data + (v110_data * v126_data));
            float v131_data = s0[50];
            float v133_data = r1[4];
            r1[4] = (v133_data + (v110_data * v131_data));
            float v136_data = s0[62];
            float v138_data = r1[5];
            r1[5] = (v138_data + (v110_data * v136_data));
            float v141_data = s0[77];
            float v143_data = r1[6];
            r1[6] = (v143_data + (v110_data * v141_data));
            float v146_data = s0[80];
            float v148_data = r1[7];
            r1[7] = (v148_data + (v110_data * v146_data));
            float v151_data = s0[103];
            float v153_data = r1[8];
            r1[8] = (v153_data + (v110_data * v151_data));
            float v156_data = s0[107];
            float v158_data = r1[9];
            r1[9] = (v158_data + (v110_data * v156_data));
            float v161_data = s0[126];
            float v163_data = r1[10];
            r1[10] = (v163_data + (v110_data * v161_data));
            float v166_data = s0[141];
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
            float v185_data = s0[27];
            float v187_data = r1[2];
            r1[2] = (v187_data + (v174_data * v185_data));
            float v190_data = s0[36];
            float v192_data = r1[3];
            r1[3] = (v192_data + (v174_data * v190_data));
            float v195_data = s0[49];
            float v197_data = r1[4];
            r1[4] = (v197_data + (v174_data * v195_data));
            float v200_data = s0[61];
            float v202_data = r1[5];
            r1[5] = (v202_data + (v174_data * v200_data));
            float v205_data = s0[78];
            float v207_data = r1[6];
            r1[6] = (v207_data + (v174_data * v205_data));
            float v210_data = s0[83];
            float v212_data = r1[7];
            r1[7] = (v212_data + (v174_data * v210_data));
            float v215_data = s0[100];
            float v217_data = r1[8];
            r1[8] = (v217_data + (v174_data * v215_data));
            float v220_data = s0[104];
            float v222_data = r1[9];
            r1[9] = (v222_data + (v174_data * v220_data));
            float v225_data = s0[125];
            float v227_data = r1[10];
            r1[10] = (v227_data + (v174_data * v225_data));
            float v230_data = s0[142];
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
            float v249_data = s0[26];
            float v251_data = r1[2];
            r1[2] = (v251_data + (v238_data * v249_data));
            float v254_data = s0[37];
            float v256_data = r1[3];
            r1[3] = (v256_data + (v238_data * v254_data));
            float v259_data = s0[48];
            float v261_data = r1[4];
            r1[4] = (v261_data + (v238_data * v259_data));
            float v264_data = s0[60];
            float v266_data = r1[5];
            r1[5] = (v266_data + (v238_data * v264_data));
            float v269_data = s0[79];
            float v271_data = r1[6];
            r1[6] = (v271_data + (v238_data * v269_data));
            float v274_data = s0[82];
            float v276_data = r1[7];
            r1[7] = (v276_data + (v238_data * v274_data));
            float v279_data = s0[101];
            float v281_data = r1[8];
            r1[8] = (v281_data + (v238_data * v279_data));
            float v284_data = s0[105];
            float v286_data = r1[9];
            r1[9] = (v286_data + (v238_data * v284_data));
            float v289_data = s0[124];
            float v291_data = r1[10];
            r1[10] = (v291_data + (v238_data * v289_data));
            float v294_data = s0[143];
            float v296_data = r1[11];
            r1[11] = (v296_data + (v238_data * v294_data));
          }
          if (v12_lead < 6) {
            float v302_data = r0[4];
            float v303_data = s0[4];
            float v305_data = r1[0];
            r1[0] = (v305_data + (v302_data * v303_data));
            float v308_data = s0[17];
            float v310_data = r1[1];
            r1[1] = (v310_data + (v302_data * v308_data));
            float v313_data = s0[29];
            float v315_data = r1[2];
            r1[2] = (v315_data + (v302_data * v313_data));
            float v318_data = s0[42];
            float v320_data = r1[3];
            r1[3] = (v320_data + (v302_data * v318_data));
            float v323_data = s0[55];
            float v325_data = r1[4];
            r1[4] = (v325_data + (v302_data * v323_data));
            float v328_data = s0[68];
            float v330_data = r1[5];
            r1[5] = (v330_data + (v302_data * v328_data));
            float v333_data = s0[72];
            float v335_data = r1[6];
            r1[6] = (v335_data + (v302_data * v333_data));
            float v338_data = s0[93];
            float v340_data = r1[7];
            r1[7] = (v340_data + (v302_data * v338_data));
            float v343_data = s0[98];
            float v345_data = r1[8];
            r1[8] = (v345_data + (v302_data * v343_data));
            float v348_data = s0[119];
            float v350_data = r1[9];
            r1[9] = (v350_data + (v302_data * v348_data));
            float v353_data = s0[123];
            float v355_data = r1[10];
            r1[10] = (v355_data + (v302_data * v353_data));
            float v358_data = s0[128];
            float v360_data = r1[11];
            r1[11] = (v360_data + (v302_data * v358_data));
          }
          if (v12_lead < 6) {
            float v366_data = r0[5];
            float v367_data = s0[5];
            float v369_data = r1[0];
            r1[0] = (v369_data + (v366_data * v367_data));
            float v372_data = s0[16];
            float v374_data = r1[1];
            r1[1] = (v374_data + (v366_data * v372_data));
            float v377_data = s0[28];
            float v379_data = r1[2];
            r1[2] = (v379_data + (v366_data * v377_data));
            float v382_data = s0[43];
            float v384_data = r1[3];
            r1[3] = (v384_data + (v366_data * v382_data));
            float v387_data = s0[54];
            float v389_data = r1[4];
            r1[4] = (v389_data + (v366_data * v387_data));
            float v392_data = s0[69];
            float v394_data = r1[5];
            r1[5] = (v394_data + (v366_data * v392_data));
            float v397_data = s0[73];
            float v399_data = r1[6];
            r1[6] = (v399_data + (v366_data * v397_data));
            float v402_data = s0[92];
            float v404_data = r1[7];
            r1[7] = (v404_data + (v366_data * v402_data));
            float v407_data = s0[99];
            float v409_data = r1[8];
            r1[8] = (v409_data + (v366_data * v407_data));
            float v412_data = s0[118];
            float v414_data = r1[9];
            r1[9] = (v414_data + (v366_data * v412_data));
            float v417_data = s0[122];
            float v419_data = r1[10];
            r1[10] = (v419_data + (v366_data * v417_data));
            float v422_data = s0[129];
            float v424_data = r1[11];
            r1[11] = (v424_data + (v366_data * v422_data));
          }
          if (v12_lead < 6) {
            float v430_data = r0[6];
            float v431_data = s0[6];
            float v433_data = r1[0];
            r1[0] = (v433_data + (v430_data * v431_data));
            float v436_data = s0[19];
            float v438_data = r1[1];
            r1[1] = (v438_data + (v430_data * v436_data));
            float v441_data = s0[31];
            float v443_data = r1[2];
            r1[2] = (v443_data + (v430_data * v441_data));
            float v446_data = s0[40];
            float v448_data = r1[3];
            r1[3] = (v448_data + (v430_data * v446_data));
            float v451_data = s0[53];
            float v453_data = r1[4];
            r1[4] = (v453_data + (v430_data * v451_data));
            float v456_data = s0[70];
            float v458_data = r1[5];
            r1[5] = (v458_data + (v430_data * v456_data));
            float v461_data = s0[74];
            float v463_data = r1[6];
            r1[6] = (v463_data + (v430_data * v461_data));
            float v466_data = s0[95];
            float v468_data = r1[7];
            r1[7] = (v468_data + (v430_data * v466_data));
            float v471_data = s0[96];
            float v473_data = r1[8];
            r1[8] = (v473_data + (v430_data * v471_data));
            float v476_data = s0[117];
            float v478_data = r1[9];
            r1[9] = (v478_data + (v430_data * v476_data));
            float v481_data = s0[121];
            float v483_data = r1[10];
            r1[10] = (v483_data + (v430_data * v481_data));
            float v486_data = s0[130];
            float v488_data = r1[11];
            r1[11] = (v488_data + (v430_data * v486_data));
          }
          if (v12_lead < 6) {
            float v494_data = r0[7];
            float v495_data = s0[7];
            float v497_data = r1[0];
            r1[0] = (v497_data + (v494_data * v495_data));
            float v500_data = s0[18];
            float v502_data = r1[1];
            r1[1] = (v502_data + (v494_data * v500_data));
            float v505_data = s0[30];
            float v507_data = r1[2];
            r1[2] = (v507_data + (v494_data * v505_data));
            float v510_data = s0[41];
            float v512_data = r1[3];
            r1[3] = (v512_data + (v494_data * v510_data));
            float v515_data = s0[52];
            float v517_data = r1[4];
            r1[4] = (v517_data + (v494_data * v515_data));
            float v520_data = s0[71];
            float v522_data = r1[5];
            r1[5] = (v522_data + (v494_data * v520_data));
            float v525_data = s0[75];
            float v527_data = r1[6];
            r1[6] = (v527_data + (v494_data * v525_data));
            float v530_data = s0[94];
            float v532_data = r1[7];
            r1[7] = (v532_data + (v494_data * v530_data));
            float v535_data = s0[97];
            float v537_data = r1[8];
            r1[8] = (v537_data + (v494_data * v535_data));
            float v540_data = s0[116];
            float v542_data = r1[9];
            r1[9] = (v542_data + (v494_data * v540_data));
            float v545_data = s0[120];
            float v547_data = r1[10];
            r1[10] = (v547_data + (v494_data * v545_data));
            float v550_data = s0[131];
            float v552_data = r1[11];
            r1[11] = (v552_data + (v494_data * v550_data));
          }
          if (v12_lead < 6) {
            float v558_data = r0[8];
            float v559_data = s0[8];
            float v561_data = r1[0];
            r1[0] = (v561_data + (v558_data * v559_data));
            float v564_data = s0[21];
            float v566_data = r1[1];
            r1[1] = (v566_data + (v558_data * v564_data));
            float v569_data = s0[34];
            float v571_data = r1[2];
            r1[2] = (v571_data + (v558_data * v569_data));
            float v574_data = s0[46];
            float v576_data = r1[3];
            r1[3] = (v576_data + (v558_data * v574_data));
            float v579_data = s0[59];
            float v581_data = r1[4];
            r1[4] = (v581_data + (v558_data * v579_data));
            float v584_data = s0[64];
            float v586_data = r1[5];
            r1[5] = (v586_data + (v558_data * v584_data));
            float v589_data = s0[85];
            float v591_data = r1[6];
            r1[6] = (v591_data + (v558_data * v589_data));
            float v594_data = s0[89];
            float v596_data = r1[7];
            r1[7] = (v596_data + (v558_data * v594_data));
            float v599_data = s0[110];
            float v601_data = r1[8];
            r1[8] = (v601_data + (v558_data * v599_data));
            float v604_data = s0[115];
            float v606_data = r1[9];
            r1[9] = (v606_data + (v558_data * v604_data));
            float v609_data = s0[136];
            float v611_data = r1[10];
            r1[10] = (v611_data + (v558_data * v609_data));
            float v614_data = s0[132];
            float v616_data = r1[11];
            r1[11] = (v616_data + (v558_data * v614_data));
          }
          if (v12_lead < 6) {
            float v622_data = r0[9];
            float v623_data = s0[9];
            float v625_data = r1[0];
            r1[0] = (v625_data + (v622_data * v623_data));
            float v628_data = s0[20];
            float v630_data = r1[1];
            r1[1] = (v630_data + (v622_data * v628_data));
            float v633_data = s0[35];
            float v635_data = r1[2];
            r1[2] = (v635_data + (v622_data * v633_data));
            float v638_data = s0[47];
            float v640_data = r1[3];
            r1[3] = (v640_data + (v622_data * v638_data));
            float v643_data = s0[58];
            float v645_data = r1[4];
            r1[4] = (v645_data + (v622_data * v643_data));
            float v648_data = s0[65];
            float v650_data = r1[5];
            r1[5] = (v650_data + (v622_data * v648_data));
            float v653_data = s0[84];
            float v655_data = r1[6];
            r1[6] = (v655_data + (v622_data * v653_data));
            float v658_data = s0[88];
            float v660_data = r1[7];
            r1[7] = (v660_data + (v622_data * v658_data));
            float v663_data = s0[111];
            float v665_data = r1[8];
            r1[8] = (v665_data + (v622_data * v663_data));
            float v668_data = s0[114];
            float v670_data = r1[9];
            r1[9] = (v670_data + (v622_data * v668_data));
            float v673_data = s0[137];
            float v675_data = r1[10];
            r1[10] = (v675_data + (v622_data * v673_data));
            float v678_data = s0[133];
            float v680_data = r1[11];
            r1[11] = (v680_data + (v622_data * v678_data));
          }
          if (v12_lead < 6) {
            float v686_data = r0[10];
            float v687_data = s0[10];
            float v689_data = r1[0];
            r1[0] = (v689_data + (v686_data * v687_data));
            float v692_data = s0[23];
            float v694_data = r1[1];
            r1[1] = (v694_data + (v686_data * v692_data));
            float v697_data = s0[32];
            float v699_data = r1[2];
            r1[2] = (v699_data + (v686_data * v697_data));
            float v702_data = s0[44];
            float v704_data = r1[3];
            r1[3] = (v704_data + (v686_data * v702_data));
            float v707_data = s0[57];
            float v709_data = r1[4];
            r1[4] = (v709_data + (v686_data * v707_data));
            float v712_data = s0[66];
            float v714_data = r1[5];
            r1[5] = (v714_data + (v686_data * v712_data));
            float v717_data = s0[87];
            float v719_data = r1[6];
            r1[6] = (v719_data + (v686_data * v717_data));
            float v722_data = s0[91];
            float v724_data = r1[7];
            r1[7] = (v724_data + (v686_data * v722_data));
            float v727_data = s0[108];
            float v729_data = r1[8];
            r1[8] = (v729_data + (v686_data * v727_data));
            float v732_data = s0[113];
            float v734_data = r1[9];
            r1[9] = (v734_data + (v686_data * v732_data));
            float v737_data = s0[138];
            float v739_data = r1[10];
            r1[10] = (v739_data + (v686_data * v737_data));
            float v742_data = s0[134];
            float v744_data = r1[11];
            r1[11] = (v744_data + (v686_data * v742_data));
          }
          if (v12_lead < 6) {
            float v750_data = r0[11];
            float v751_data = s0[11];
            float v753_data = r1[0];
            r1[0] = (v753_data + (v750_data * v751_data));
            float v756_data = s0[22];
            float v758_data = r1[1];
            r1[1] = (v758_data + (v750_data * v756_data));
            float v761_data = s0[33];
            float v763_data = r1[2];
            r1[2] = (v763_data + (v750_data * v761_data));
            float v766_data = s0[45];
            float v768_data = r1[3];
            r1[3] = (v768_data + (v750_data * v766_data));
            float v771_data = s0[56];
            float v773_data = r1[4];
            r1[4] = (v773_data + (v750_data * v771_data));
            float v776_data = s0[67];
            float v778_data = r1[5];
            r1[5] = (v778_data + (v750_data * v776_data));
            float v781_data = s0[86];
            float v783_data = r1[6];
            r1[6] = (v783_data + (v750_data * v781_data));
            float v786_data = s0[90];
            float v788_data = r1[7];
            r1[7] = (v788_data + (v750_data * v786_data));
            float v791_data = s0[109];
            float v793_data = r1[8];
            r1[8] = (v793_data + (v750_data * v791_data));
            float v796_data = s0[112];
            float v798_data = r1[9];
            r1[9] = (v798_data + (v750_data * v796_data));
            float v801_data = s0[139];
            float v803_data = r1[10];
            r1[10] = (v803_data + (v750_data * v801_data));
            float v806_data = s0[135];
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
              int32_t v824_a = v12_lead + (v815_i1 * 12);
              s1[(v824_a ^ ((v824_a >> 4) & 15))] = v817_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v833_i1 = 0; v833_i1 < 12; ++v833_i1) {
              float v841_data = __ldcg(&glb_m4[(v12_lead + (v833_i1 * 12))]);
              r4[v833_i1] = v841_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v12_lead < 6) {
            float v849_data = r2[0];
            float v850_data = s0[0];
            float v852_data = ir3[0];
            ir3[0] = (v852_data + (v849_data * v850_data));
            float v855_data = s0[12];
            float v857_data = ir3[1];
            ir3[1] = (v857_data + (v849_data * v855_data));
            float v860_data = s0[25];
            float v862_data = ir3[2];
            ir3[2] = (v862_data + (v849_data * v860_data));
            float v865_data = s0[38];
            float v867_data = ir3[3];
            ir3[3] = (v867_data + (v849_data * v865_data));
            float v870_data = s0[51];
            float v872_data = ir3[4];
            ir3[4] = (v872_data + (v849_data * v870_data));
            float v875_data = s0[63];
            float v877_data = ir3[5];
            ir3[5] = (v877_data + (v849_data * v875_data));
            float v880_data = s0[76];
            float v882_data = ir3[6];
            ir3[6] = (v882_data + (v849_data * v880_data));
            float v885_data = s0[81];
            float v887_data = ir3[7];
            ir3[7] = (v887_data + (v849_data * v885_data));
            float v890_data = s0[102];
            float v892_data = ir3[8];
            ir3[8] = (v892_data + (v849_data * v890_data));
            float v895_data = s0[106];
            float v897_data = ir3[9];
            ir3[9] = (v897_data + (v849_data * v895_data));
            float v900_data = s0[127];
            float v902_data = ir3[10];
            ir3[10] = (v902_data + (v849_data * v900_data));
            float v905_data = s0[140];
            float v907_data = ir3[11];
            ir3[11] = (v907_data + (v849_data * v905_data));
          }
          if (v12_lead < 6) {
            float v913_data = r2[1];
            float v914_data = s0[1];
            float v916_data = ir3[0];
            ir3[0] = (v916_data + (v913_data * v914_data));
            float v919_data = s0[13];
            float v921_data = ir3[1];
            ir3[1] = (v921_data + (v913_data * v919_data));
            float v924_data = s0[24];
            float v926_data = ir3[2];
            ir3[2] = (v926_data + (v913_data * v924_data));
            float v929_data = s0[39];
            float v931_data = ir3[3];
            ir3[3] = (v931_data + (v913_data * v929_data));
            float v934_data = s0[50];
            float v936_data = ir3[4];
            ir3[4] = (v936_data + (v913_data * v934_data));
            float v939_data = s0[62];
            float v941_data = ir3[5];
            ir3[5] = (v941_data + (v913_data * v939_data));
            float v944_data = s0[77];
            float v946_data = ir3[6];
            ir3[6] = (v946_data + (v913_data * v944_data));
            float v949_data = s0[80];
            float v951_data = ir3[7];
            ir3[7] = (v951_data + (v913_data * v949_data));
            float v954_data = s0[103];
            float v956_data = ir3[8];
            ir3[8] = (v956_data + (v913_data * v954_data));
            float v959_data = s0[107];
            float v961_data = ir3[9];
            ir3[9] = (v961_data + (v913_data * v959_data));
            float v964_data = s0[126];
            float v966_data = ir3[10];
            ir3[10] = (v966_data + (v913_data * v964_data));
            float v969_data = s0[141];
            float v971_data = ir3[11];
            ir3[11] = (v971_data + (v913_data * v969_data));
          }
          if (v12_lead < 6) {
            float v977_data = r2[2];
            float v978_data = s0[2];
            float v980_data = ir3[0];
            ir3[0] = (v980_data + (v977_data * v978_data));
            float v983_data = s0[14];
            float v985_data = ir3[1];
            ir3[1] = (v985_data + (v977_data * v983_data));
            float v988_data = s0[27];
            float v990_data = ir3[2];
            ir3[2] = (v990_data + (v977_data * v988_data));
            float v993_data = s0[36];
            float v995_data = ir3[3];
            ir3[3] = (v995_data + (v977_data * v993_data));
            float v998_data = s0[49];
            float v1000_data = ir3[4];
            ir3[4] = (v1000_data + (v977_data * v998_data));
            float v1003_data = s0[61];
            float v1005_data = ir3[5];
            ir3[5] = (v1005_data + (v977_data * v1003_data));
            float v1008_data = s0[78];
            float v1010_data = ir3[6];
            ir3[6] = (v1010_data + (v977_data * v1008_data));
            float v1013_data = s0[83];
            float v1015_data = ir3[7];
            ir3[7] = (v1015_data + (v977_data * v1013_data));
            float v1018_data = s0[100];
            float v1020_data = ir3[8];
            ir3[8] = (v1020_data + (v977_data * v1018_data));
            float v1023_data = s0[104];
            float v1025_data = ir3[9];
            ir3[9] = (v1025_data + (v977_data * v1023_data));
            float v1028_data = s0[125];
            float v1030_data = ir3[10];
            ir3[10] = (v1030_data + (v977_data * v1028_data));
            float v1033_data = s0[142];
            float v1035_data = ir3[11];
            ir3[11] = (v1035_data + (v977_data * v1033_data));
          }
          if (v12_lead < 6) {
            float v1041_data = r2[3];
            float v1042_data = s0[3];
            float v1044_data = ir3[0];
            ir3[0] = (v1044_data + (v1041_data * v1042_data));
            float v1047_data = s0[15];
            float v1049_data = ir3[1];
            ir3[1] = (v1049_data + (v1041_data * v1047_data));
            float v1052_data = s0[26];
            float v1054_data = ir3[2];
            ir3[2] = (v1054_data + (v1041_data * v1052_data));
            float v1057_data = s0[37];
            float v1059_data = ir3[3];
            ir3[3] = (v1059_data + (v1041_data * v1057_data));
            float v1062_data = s0[48];
            float v1064_data = ir3[4];
            ir3[4] = (v1064_data + (v1041_data * v1062_data));
            float v1067_data = s0[60];
            float v1069_data = ir3[5];
            ir3[5] = (v1069_data + (v1041_data * v1067_data));
            float v1072_data = s0[79];
            float v1074_data = ir3[6];
            ir3[6] = (v1074_data + (v1041_data * v1072_data));
            float v1077_data = s0[82];
            float v1079_data = ir3[7];
            ir3[7] = (v1079_data + (v1041_data * v1077_data));
            float v1082_data = s0[101];
            float v1084_data = ir3[8];
            ir3[8] = (v1084_data + (v1041_data * v1082_data));
            float v1087_data = s0[105];
            float v1089_data = ir3[9];
            ir3[9] = (v1089_data + (v1041_data * v1087_data));
            float v1092_data = s0[124];
            float v1094_data = ir3[10];
            ir3[10] = (v1094_data + (v1041_data * v1092_data));
            float v1097_data = s0[143];
            float v1099_data = ir3[11];
            ir3[11] = (v1099_data + (v1041_data * v1097_data));
          }
          if (v12_lead < 6) {
            float v1105_data = r2[4];
            float v1106_data = s0[4];
            float v1108_data = ir3[0];
            ir3[0] = (v1108_data + (v1105_data * v1106_data));
            float v1111_data = s0[17];
            float v1113_data = ir3[1];
            ir3[1] = (v1113_data + (v1105_data * v1111_data));
            float v1116_data = s0[29];
            float v1118_data = ir3[2];
            ir3[2] = (v1118_data + (v1105_data * v1116_data));
            float v1121_data = s0[42];
            float v1123_data = ir3[3];
            ir3[3] = (v1123_data + (v1105_data * v1121_data));
            float v1126_data = s0[55];
            float v1128_data = ir3[4];
            ir3[4] = (v1128_data + (v1105_data * v1126_data));
            float v1131_data = s0[68];
            float v1133_data = ir3[5];
            ir3[5] = (v1133_data + (v1105_data * v1131_data));
            float v1136_data = s0[72];
            float v1138_data = ir3[6];
            ir3[6] = (v1138_data + (v1105_data * v1136_data));
            float v1141_data = s0[93];
            float v1143_data = ir3[7];
            ir3[7] = (v1143_data + (v1105_data * v1141_data));
            float v1146_data = s0[98];
            float v1148_data = ir3[8];
            ir3[8] = (v1148_data + (v1105_data * v1146_data));
            float v1151_data = s0[119];
            float v1153_data = ir3[9];
            ir3[9] = (v1153_data + (v1105_data * v1151_data));
            float v1156_data = s0[123];
            float v1158_data = ir3[10];
            ir3[10] = (v1158_data + (v1105_data * v1156_data));
            float v1161_data = s0[128];
            float v1163_data = ir3[11];
            ir3[11] = (v1163_data + (v1105_data * v1161_data));
          }
          if (v12_lead < 6) {
            float v1169_data = r2[5];
            float v1170_data = s0[5];
            float v1172_data = ir3[0];
            ir3[0] = (v1172_data + (v1169_data * v1170_data));
            float v1175_data = s0[16];
            float v1177_data = ir3[1];
            ir3[1] = (v1177_data + (v1169_data * v1175_data));
            float v1180_data = s0[28];
            float v1182_data = ir3[2];
            ir3[2] = (v1182_data + (v1169_data * v1180_data));
            float v1185_data = s0[43];
            float v1187_data = ir3[3];
            ir3[3] = (v1187_data + (v1169_data * v1185_data));
            float v1190_data = s0[54];
            float v1192_data = ir3[4];
            ir3[4] = (v1192_data + (v1169_data * v1190_data));
            float v1195_data = s0[69];
            float v1197_data = ir3[5];
            ir3[5] = (v1197_data + (v1169_data * v1195_data));
            float v1200_data = s0[73];
            float v1202_data = ir3[6];
            ir3[6] = (v1202_data + (v1169_data * v1200_data));
            float v1205_data = s0[92];
            float v1207_data = ir3[7];
            ir3[7] = (v1207_data + (v1169_data * v1205_data));
            float v1210_data = s0[99];
            float v1212_data = ir3[8];
            ir3[8] = (v1212_data + (v1169_data * v1210_data));
            float v1215_data = s0[118];
            float v1217_data = ir3[9];
            ir3[9] = (v1217_data + (v1169_data * v1215_data));
            float v1220_data = s0[122];
            float v1222_data = ir3[10];
            ir3[10] = (v1222_data + (v1169_data * v1220_data));
            float v1225_data = s0[129];
            float v1227_data = ir3[11];
            ir3[11] = (v1227_data + (v1169_data * v1225_data));
          }
          if (v12_lead < 6) {
            float v1233_data = r2[6];
            float v1234_data = s0[6];
            float v1236_data = ir3[0];
            ir3[0] = (v1236_data + (v1233_data * v1234_data));
            float v1239_data = s0[19];
            float v1241_data = ir3[1];
            ir3[1] = (v1241_data + (v1233_data * v1239_data));
            float v1244_data = s0[31];
            float v1246_data = ir3[2];
            ir3[2] = (v1246_data + (v1233_data * v1244_data));
            float v1249_data = s0[40];
            float v1251_data = ir3[3];
            ir3[3] = (v1251_data + (v1233_data * v1249_data));
            float v1254_data = s0[53];
            float v1256_data = ir3[4];
            ir3[4] = (v1256_data + (v1233_data * v1254_data));
            float v1259_data = s0[70];
            float v1261_data = ir3[5];
            ir3[5] = (v1261_data + (v1233_data * v1259_data));
            float v1264_data = s0[74];
            float v1266_data = ir3[6];
            ir3[6] = (v1266_data + (v1233_data * v1264_data));
            float v1269_data = s0[95];
            float v1271_data = ir3[7];
            ir3[7] = (v1271_data + (v1233_data * v1269_data));
            float v1274_data = s0[96];
            float v1276_data = ir3[8];
            ir3[8] = (v1276_data + (v1233_data * v1274_data));
            float v1279_data = s0[117];
            float v1281_data = ir3[9];
            ir3[9] = (v1281_data + (v1233_data * v1279_data));
            float v1284_data = s0[121];
            float v1286_data = ir3[10];
            ir3[10] = (v1286_data + (v1233_data * v1284_data));
            float v1289_data = s0[130];
            float v1291_data = ir3[11];
            ir3[11] = (v1291_data + (v1233_data * v1289_data));
          }
          if (v12_lead < 6) {
            float v1297_data = r2[7];
            float v1298_data = s0[7];
            float v1300_data = ir3[0];
            ir3[0] = (v1300_data + (v1297_data * v1298_data));
            float v1303_data = s0[18];
            float v1305_data = ir3[1];
            ir3[1] = (v1305_data + (v1297_data * v1303_data));
            float v1308_data = s0[30];
            float v1310_data = ir3[2];
            ir3[2] = (v1310_data + (v1297_data * v1308_data));
            float v1313_data = s0[41];
            float v1315_data = ir3[3];
            ir3[3] = (v1315_data + (v1297_data * v1313_data));
            float v1318_data = s0[52];
            float v1320_data = ir3[4];
            ir3[4] = (v1320_data + (v1297_data * v1318_data));
            float v1323_data = s0[71];
            float v1325_data = ir3[5];
            ir3[5] = (v1325_data + (v1297_data * v1323_data));
            float v1328_data = s0[75];
            float v1330_data = ir3[6];
            ir3[6] = (v1330_data + (v1297_data * v1328_data));
            float v1333_data = s0[94];
            float v1335_data = ir3[7];
            ir3[7] = (v1335_data + (v1297_data * v1333_data));
            float v1338_data = s0[97];
            float v1340_data = ir3[8];
            ir3[8] = (v1340_data + (v1297_data * v1338_data));
            float v1343_data = s0[116];
            float v1345_data = ir3[9];
            ir3[9] = (v1345_data + (v1297_data * v1343_data));
            float v1348_data = s0[120];
            float v1350_data = ir3[10];
            ir3[10] = (v1350_data + (v1297_data * v1348_data));
            float v1353_data = s0[131];
            float v1355_data = ir3[11];
            ir3[11] = (v1355_data + (v1297_data * v1353_data));
          }
          if (v12_lead < 6) {
            float v1361_data = r2[8];
            float v1362_data = s0[8];
            float v1364_data = ir3[0];
            ir3[0] = (v1364_data + (v1361_data * v1362_data));
            float v1367_data = s0[21];
            float v1369_data = ir3[1];
            ir3[1] = (v1369_data + (v1361_data * v1367_data));
            float v1372_data = s0[34];
            float v1374_data = ir3[2];
            ir3[2] = (v1374_data + (v1361_data * v1372_data));
            float v1377_data = s0[46];
            float v1379_data = ir3[3];
            ir3[3] = (v1379_data + (v1361_data * v1377_data));
            float v1382_data = s0[59];
            float v1384_data = ir3[4];
            ir3[4] = (v1384_data + (v1361_data * v1382_data));
            float v1387_data = s0[64];
            float v1389_data = ir3[5];
            ir3[5] = (v1389_data + (v1361_data * v1387_data));
            float v1392_data = s0[85];
            float v1394_data = ir3[6];
            ir3[6] = (v1394_data + (v1361_data * v1392_data));
            float v1397_data = s0[89];
            float v1399_data = ir3[7];
            ir3[7] = (v1399_data + (v1361_data * v1397_data));
            float v1402_data = s0[110];
            float v1404_data = ir3[8];
            ir3[8] = (v1404_data + (v1361_data * v1402_data));
            float v1407_data = s0[115];
            float v1409_data = ir3[9];
            ir3[9] = (v1409_data + (v1361_data * v1407_data));
            float v1412_data = s0[136];
            float v1414_data = ir3[10];
            ir3[10] = (v1414_data + (v1361_data * v1412_data));
            float v1417_data = s0[132];
            float v1419_data = ir3[11];
            ir3[11] = (v1419_data + (v1361_data * v1417_data));
          }
          if (v12_lead < 6) {
            float v1425_data = r2[9];
            float v1426_data = s0[9];
            float v1428_data = ir3[0];
            ir3[0] = (v1428_data + (v1425_data * v1426_data));
            float v1431_data = s0[20];
            float v1433_data = ir3[1];
            ir3[1] = (v1433_data + (v1425_data * v1431_data));
            float v1436_data = s0[35];
            float v1438_data = ir3[2];
            ir3[2] = (v1438_data + (v1425_data * v1436_data));
            float v1441_data = s0[47];
            float v1443_data = ir3[3];
            ir3[3] = (v1443_data + (v1425_data * v1441_data));
            float v1446_data = s0[58];
            float v1448_data = ir3[4];
            ir3[4] = (v1448_data + (v1425_data * v1446_data));
            float v1451_data = s0[65];
            float v1453_data = ir3[5];
            ir3[5] = (v1453_data + (v1425_data * v1451_data));
            float v1456_data = s0[84];
            float v1458_data = ir3[6];
            ir3[6] = (v1458_data + (v1425_data * v1456_data));
            float v1461_data = s0[88];
            float v1463_data = ir3[7];
            ir3[7] = (v1463_data + (v1425_data * v1461_data));
            float v1466_data = s0[111];
            float v1468_data = ir3[8];
            ir3[8] = (v1468_data + (v1425_data * v1466_data));
            float v1471_data = s0[114];
            float v1473_data = ir3[9];
            ir3[9] = (v1473_data + (v1425_data * v1471_data));
            float v1476_data = s0[137];
            float v1478_data = ir3[10];
            ir3[10] = (v1478_data + (v1425_data * v1476_data));
            float v1481_data = s0[133];
            float v1483_data = ir3[11];
            ir3[11] = (v1483_data + (v1425_data * v1481_data));
          }
          if (v12_lead < 6) {
            float v1489_data = r2[10];
            float v1490_data = s0[10];
            float v1492_data = ir3[0];
            ir3[0] = (v1492_data + (v1489_data * v1490_data));
            float v1495_data = s0[23];
            float v1497_data = ir3[1];
            ir3[1] = (v1497_data + (v1489_data * v1495_data));
            float v1500_data = s0[32];
            float v1502_data = ir3[2];
            ir3[2] = (v1502_data + (v1489_data * v1500_data));
            float v1505_data = s0[44];
            float v1507_data = ir3[3];
            ir3[3] = (v1507_data + (v1489_data * v1505_data));
            float v1510_data = s0[57];
            float v1512_data = ir3[4];
            ir3[4] = (v1512_data + (v1489_data * v1510_data));
            float v1515_data = s0[66];
            float v1517_data = ir3[5];
            ir3[5] = (v1517_data + (v1489_data * v1515_data));
            float v1520_data = s0[87];
            float v1522_data = ir3[6];
            ir3[6] = (v1522_data + (v1489_data * v1520_data));
            float v1525_data = s0[91];
            float v1527_data = ir3[7];
            ir3[7] = (v1527_data + (v1489_data * v1525_data));
            float v1530_data = s0[108];
            float v1532_data = ir3[8];
            ir3[8] = (v1532_data + (v1489_data * v1530_data));
            float v1535_data = s0[113];
            float v1537_data = ir3[9];
            ir3[9] = (v1537_data + (v1489_data * v1535_data));
            float v1540_data = s0[138];
            float v1542_data = ir3[10];
            ir3[10] = (v1542_data + (v1489_data * v1540_data));
            float v1545_data = s0[134];
            float v1547_data = ir3[11];
            ir3[11] = (v1547_data + (v1489_data * v1545_data));
          }
          if (v12_lead < 6) {
            float v1553_data = r2[11];
            float v1554_data = s0[11];
            float v1556_data = ir3[0];
            ir3[0] = (v1556_data + (v1553_data * v1554_data));
            float v1559_data = s0[22];
            float v1561_data = ir3[1];
            ir3[1] = (v1561_data + (v1553_data * v1559_data));
            float v1564_data = s0[33];
            float v1566_data = ir3[2];
            ir3[2] = (v1566_data + (v1553_data * v1564_data));
            float v1569_data = s0[45];
            float v1571_data = ir3[3];
            ir3[3] = (v1571_data + (v1553_data * v1569_data));
            float v1574_data = s0[56];
            float v1576_data = ir3[4];
            ir3[4] = (v1576_data + (v1553_data * v1574_data));
            float v1579_data = s0[67];
            float v1581_data = ir3[5];
            ir3[5] = (v1581_data + (v1553_data * v1579_data));
            float v1584_data = s0[86];
            float v1586_data = ir3[6];
            ir3[6] = (v1586_data + (v1553_data * v1584_data));
            float v1589_data = s0[90];
            float v1591_data = ir3[7];
            ir3[7] = (v1591_data + (v1553_data * v1589_data));
            float v1594_data = s0[109];
            float v1596_data = ir3[8];
            ir3[8] = (v1596_data + (v1553_data * v1594_data));
            float v1599_data = s0[112];
            float v1601_data = ir3[9];
            ir3[9] = (v1601_data + (v1553_data * v1599_data));
            float v1604_data = s0[139];
            float v1606_data = ir3[10];
            ir3[10] = (v1606_data + (v1553_data * v1604_data));
            float v1609_data = s0[135];
            float v1611_data = ir3[11];
            ir3[11] = (v1611_data + (v1553_data * v1609_data));
          }
          if (v12_lead < 6) {
            #pragma unroll
            for (int32_t v1617_n1 = 0; v1617_n1 < 12; ++v1617_n1) {
              float v1619_data = ir3[v1617_n1];
              r3[v1617_n1] = v1619_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v12_lead < 6) {
            int32_t v1633_off = v12_lead + 6;
            #pragma unroll
            for (int32_t v1625_i1 = 0; v1625_i1 < 12; ++v1625_i1) {
              float v1627_data = r3[v1625_i1];
              int32_t v1635_a = v1633_off + (v1625_i1 * 12);
              s1[(v1635_a ^ ((v1635_a >> 4) & 15))] = v1627_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v12_lead < 12) {
            float v1645_data = r4[0];
            float v1646_data = s1[0];
            float v1648_data = ir5[0];
            ir5[0] = (v1648_data + (v1645_data * v1646_data));
            float v1651_data = s1[12];
            float v1653_data = ir5[1];
            ir5[1] = (v1653_data + (v1645_data * v1651_data));
            float v1656_data = s1[25];
            float v1658_data = ir5[2];
            ir5[2] = (v1658_data + (v1645_data * v1656_data));
            float v1661_data = s1[38];
            float v1663_data = ir5[3];
            ir5[3] = (v1663_data + (v1645_data * v1661_data));
            float v1666_data = s1[51];
            float v1668_data = ir5[4];
            ir5[4] = (v1668_data + (v1645_data * v1666_data));
            float v1671_data = s1[63];
            float v1673_data = ir5[5];
            ir5[5] = (v1673_data + (v1645_data * v1671_data));
            float v1676_data = s1[76];
            float v1678_data = ir5[6];
            ir5[6] = (v1678_data + (v1645_data * v1676_data));
            float v1681_data = s1[81];
            float v1683_data = ir5[7];
            ir5[7] = (v1683_data + (v1645_data * v1681_data));
            float v1686_data = s1[102];
            float v1688_data = ir5[8];
            ir5[8] = (v1688_data + (v1645_data * v1686_data));
            float v1691_data = s1[106];
            float v1693_data = ir5[9];
            ir5[9] = (v1693_data + (v1645_data * v1691_data));
            float v1696_data = s1[127];
            float v1698_data = ir5[10];
            ir5[10] = (v1698_data + (v1645_data * v1696_data));
            float v1701_data = s1[140];
            float v1703_data = ir5[11];
            ir5[11] = (v1703_data + (v1645_data * v1701_data));
          }
          if (v12_lead < 12) {
            float v1709_data = r4[1];
            float v1710_data = s1[1];
            float v1712_data = ir5[0];
            ir5[0] = (v1712_data + (v1709_data * v1710_data));
            float v1715_data = s1[13];
            float v1717_data = ir5[1];
            ir5[1] = (v1717_data + (v1709_data * v1715_data));
            float v1720_data = s1[24];
            float v1722_data = ir5[2];
            ir5[2] = (v1722_data + (v1709_data * v1720_data));
            float v1725_data = s1[39];
            float v1727_data = ir5[3];
            ir5[3] = (v1727_data + (v1709_data * v1725_data));
            float v1730_data = s1[50];
            float v1732_data = ir5[4];
            ir5[4] = (v1732_data + (v1709_data * v1730_data));
            float v1735_data = s1[62];
            float v1737_data = ir5[5];
            ir5[5] = (v1737_data + (v1709_data * v1735_data));
            float v1740_data = s1[77];
            float v1742_data = ir5[6];
            ir5[6] = (v1742_data + (v1709_data * v1740_data));
            float v1745_data = s1[80];
            float v1747_data = ir5[7];
            ir5[7] = (v1747_data + (v1709_data * v1745_data));
            float v1750_data = s1[103];
            float v1752_data = ir5[8];
            ir5[8] = (v1752_data + (v1709_data * v1750_data));
            float v1755_data = s1[107];
            float v1757_data = ir5[9];
            ir5[9] = (v1757_data + (v1709_data * v1755_data));
            float v1760_data = s1[126];
            float v1762_data = ir5[10];
            ir5[10] = (v1762_data + (v1709_data * v1760_data));
            float v1765_data = s1[141];
            float v1767_data = ir5[11];
            ir5[11] = (v1767_data + (v1709_data * v1765_data));
          }
          if (v12_lead < 12) {
            float v1773_data = r4[2];
            float v1774_data = s1[2];
            float v1776_data = ir5[0];
            ir5[0] = (v1776_data + (v1773_data * v1774_data));
            float v1779_data = s1[14];
            float v1781_data = ir5[1];
            ir5[1] = (v1781_data + (v1773_data * v1779_data));
            float v1784_data = s1[27];
            float v1786_data = ir5[2];
            ir5[2] = (v1786_data + (v1773_data * v1784_data));
            float v1789_data = s1[36];
            float v1791_data = ir5[3];
            ir5[3] = (v1791_data + (v1773_data * v1789_data));
            float v1794_data = s1[49];
            float v1796_data = ir5[4];
            ir5[4] = (v1796_data + (v1773_data * v1794_data));
            float v1799_data = s1[61];
            float v1801_data = ir5[5];
            ir5[5] = (v1801_data + (v1773_data * v1799_data));
            float v1804_data = s1[78];
            float v1806_data = ir5[6];
            ir5[6] = (v1806_data + (v1773_data * v1804_data));
            float v1809_data = s1[83];
            float v1811_data = ir5[7];
            ir5[7] = (v1811_data + (v1773_data * v1809_data));
            float v1814_data = s1[100];
            float v1816_data = ir5[8];
            ir5[8] = (v1816_data + (v1773_data * v1814_data));
            float v1819_data = s1[104];
            float v1821_data = ir5[9];
            ir5[9] = (v1821_data + (v1773_data * v1819_data));
            float v1824_data = s1[125];
            float v1826_data = ir5[10];
            ir5[10] = (v1826_data + (v1773_data * v1824_data));
            float v1829_data = s1[142];
            float v1831_data = ir5[11];
            ir5[11] = (v1831_data + (v1773_data * v1829_data));
          }
          if (v12_lead < 12) {
            float v1837_data = r4[3];
            float v1838_data = s1[3];
            float v1840_data = ir5[0];
            ir5[0] = (v1840_data + (v1837_data * v1838_data));
            float v1843_data = s1[15];
            float v1845_data = ir5[1];
            ir5[1] = (v1845_data + (v1837_data * v1843_data));
            float v1848_data = s1[26];
            float v1850_data = ir5[2];
            ir5[2] = (v1850_data + (v1837_data * v1848_data));
            float v1853_data = s1[37];
            float v1855_data = ir5[3];
            ir5[3] = (v1855_data + (v1837_data * v1853_data));
            float v1858_data = s1[48];
            float v1860_data = ir5[4];
            ir5[4] = (v1860_data + (v1837_data * v1858_data));
            float v1863_data = s1[60];
            float v1865_data = ir5[5];
            ir5[5] = (v1865_data + (v1837_data * v1863_data));
            float v1868_data = s1[79];
            float v1870_data = ir5[6];
            ir5[6] = (v1870_data + (v1837_data * v1868_data));
            float v1873_data = s1[82];
            float v1875_data = ir5[7];
            ir5[7] = (v1875_data + (v1837_data * v1873_data));
            float v1878_data = s1[101];
            float v1880_data = ir5[8];
            ir5[8] = (v1880_data + (v1837_data * v1878_data));
            float v1883_data = s1[105];
            float v1885_data = ir5[9];
            ir5[9] = (v1885_data + (v1837_data * v1883_data));
            float v1888_data = s1[124];
            float v1890_data = ir5[10];
            ir5[10] = (v1890_data + (v1837_data * v1888_data));
            float v1893_data = s1[143];
            float v1895_data = ir5[11];
            ir5[11] = (v1895_data + (v1837_data * v1893_data));
          }
          if (v12_lead < 12) {
            float v1901_data = r4[4];
            float v1902_data = s1[4];
            float v1904_data = ir5[0];
            ir5[0] = (v1904_data + (v1901_data * v1902_data));
            float v1907_data = s1[17];
            float v1909_data = ir5[1];
            ir5[1] = (v1909_data + (v1901_data * v1907_data));
            float v1912_data = s1[29];
            float v1914_data = ir5[2];
            ir5[2] = (v1914_data + (v1901_data * v1912_data));
            float v1917_data = s1[42];
            float v1919_data = ir5[3];
            ir5[3] = (v1919_data + (v1901_data * v1917_data));
            float v1922_data = s1[55];
            float v1924_data = ir5[4];
            ir5[4] = (v1924_data + (v1901_data * v1922_data));
            float v1927_data = s1[68];
            float v1929_data = ir5[5];
            ir5[5] = (v1929_data + (v1901_data * v1927_data));
            float v1932_data = s1[72];
            float v1934_data = ir5[6];
            ir5[6] = (v1934_data + (v1901_data * v1932_data));
            float v1937_data = s1[93];
            float v1939_data = ir5[7];
            ir5[7] = (v1939_data + (v1901_data * v1937_data));
            float v1942_data = s1[98];
            float v1944_data = ir5[8];
            ir5[8] = (v1944_data + (v1901_data * v1942_data));
            float v1947_data = s1[119];
            float v1949_data = ir5[9];
            ir5[9] = (v1949_data + (v1901_data * v1947_data));
            float v1952_data = s1[123];
            float v1954_data = ir5[10];
            ir5[10] = (v1954_data + (v1901_data * v1952_data));
            float v1957_data = s1[128];
            float v1959_data = ir5[11];
            ir5[11] = (v1959_data + (v1901_data * v1957_data));
          }
          if (v12_lead < 12) {
            float v1965_data = r4[5];
            float v1966_data = s1[5];
            float v1968_data = ir5[0];
            ir5[0] = (v1968_data + (v1965_data * v1966_data));
            float v1971_data = s1[16];
            float v1973_data = ir5[1];
            ir5[1] = (v1973_data + (v1965_data * v1971_data));
            float v1976_data = s1[28];
            float v1978_data = ir5[2];
            ir5[2] = (v1978_data + (v1965_data * v1976_data));
            float v1981_data = s1[43];
            float v1983_data = ir5[3];
            ir5[3] = (v1983_data + (v1965_data * v1981_data));
            float v1986_data = s1[54];
            float v1988_data = ir5[4];
            ir5[4] = (v1988_data + (v1965_data * v1986_data));
            float v1991_data = s1[69];
            float v1993_data = ir5[5];
            ir5[5] = (v1993_data + (v1965_data * v1991_data));
            float v1996_data = s1[73];
            float v1998_data = ir5[6];
            ir5[6] = (v1998_data + (v1965_data * v1996_data));
            float v2001_data = s1[92];
            float v2003_data = ir5[7];
            ir5[7] = (v2003_data + (v1965_data * v2001_data));
            float v2006_data = s1[99];
            float v2008_data = ir5[8];
            ir5[8] = (v2008_data + (v1965_data * v2006_data));
            float v2011_data = s1[118];
            float v2013_data = ir5[9];
            ir5[9] = (v2013_data + (v1965_data * v2011_data));
            float v2016_data = s1[122];
            float v2018_data = ir5[10];
            ir5[10] = (v2018_data + (v1965_data * v2016_data));
            float v2021_data = s1[129];
            float v2023_data = ir5[11];
            ir5[11] = (v2023_data + (v1965_data * v2021_data));
          }
          if (v12_lead < 12) {
            float v2029_data = r4[6];
            float v2030_data = s1[6];
            float v2032_data = ir5[0];
            ir5[0] = (v2032_data + (v2029_data * v2030_data));
            float v2035_data = s1[19];
            float v2037_data = ir5[1];
            ir5[1] = (v2037_data + (v2029_data * v2035_data));
            float v2040_data = s1[31];
            float v2042_data = ir5[2];
            ir5[2] = (v2042_data + (v2029_data * v2040_data));
            float v2045_data = s1[40];
            float v2047_data = ir5[3];
            ir5[3] = (v2047_data + (v2029_data * v2045_data));
            float v2050_data = s1[53];
            float v2052_data = ir5[4];
            ir5[4] = (v2052_data + (v2029_data * v2050_data));
            float v2055_data = s1[70];
            float v2057_data = ir5[5];
            ir5[5] = (v2057_data + (v2029_data * v2055_data));
            float v2060_data = s1[74];
            float v2062_data = ir5[6];
            ir5[6] = (v2062_data + (v2029_data * v2060_data));
            float v2065_data = s1[95];
            float v2067_data = ir5[7];
            ir5[7] = (v2067_data + (v2029_data * v2065_data));
            float v2070_data = s1[96];
            float v2072_data = ir5[8];
            ir5[8] = (v2072_data + (v2029_data * v2070_data));
            float v2075_data = s1[117];
            float v2077_data = ir5[9];
            ir5[9] = (v2077_data + (v2029_data * v2075_data));
            float v2080_data = s1[121];
            float v2082_data = ir5[10];
            ir5[10] = (v2082_data + (v2029_data * v2080_data));
            float v2085_data = s1[130];
            float v2087_data = ir5[11];
            ir5[11] = (v2087_data + (v2029_data * v2085_data));
          }
          if (v12_lead < 12) {
            float v2093_data = r4[7];
            float v2094_data = s1[7];
            float v2096_data = ir5[0];
            ir5[0] = (v2096_data + (v2093_data * v2094_data));
            float v2099_data = s1[18];
            float v2101_data = ir5[1];
            ir5[1] = (v2101_data + (v2093_data * v2099_data));
            float v2104_data = s1[30];
            float v2106_data = ir5[2];
            ir5[2] = (v2106_data + (v2093_data * v2104_data));
            float v2109_data = s1[41];
            float v2111_data = ir5[3];
            ir5[3] = (v2111_data + (v2093_data * v2109_data));
            float v2114_data = s1[52];
            float v2116_data = ir5[4];
            ir5[4] = (v2116_data + (v2093_data * v2114_data));
            float v2119_data = s1[71];
            float v2121_data = ir5[5];
            ir5[5] = (v2121_data + (v2093_data * v2119_data));
            float v2124_data = s1[75];
            float v2126_data = ir5[6];
            ir5[6] = (v2126_data + (v2093_data * v2124_data));
            float v2129_data = s1[94];
            float v2131_data = ir5[7];
            ir5[7] = (v2131_data + (v2093_data * v2129_data));
            float v2134_data = s1[97];
            float v2136_data = ir5[8];
            ir5[8] = (v2136_data + (v2093_data * v2134_data));
            float v2139_data = s1[116];
            float v2141_data = ir5[9];
            ir5[9] = (v2141_data + (v2093_data * v2139_data));
            float v2144_data = s1[120];
            float v2146_data = ir5[10];
            ir5[10] = (v2146_data + (v2093_data * v2144_data));
            float v2149_data = s1[131];
            float v2151_data = ir5[11];
            ir5[11] = (v2151_data + (v2093_data * v2149_data));
          }
          if (v12_lead < 12) {
            float v2157_data = r4[8];
            float v2158_data = s1[8];
            float v2160_data = ir5[0];
            ir5[0] = (v2160_data + (v2157_data * v2158_data));
            float v2163_data = s1[21];
            float v2165_data = ir5[1];
            ir5[1] = (v2165_data + (v2157_data * v2163_data));
            float v2168_data = s1[34];
            float v2170_data = ir5[2];
            ir5[2] = (v2170_data + (v2157_data * v2168_data));
            float v2173_data = s1[46];
            float v2175_data = ir5[3];
            ir5[3] = (v2175_data + (v2157_data * v2173_data));
            float v2178_data = s1[59];
            float v2180_data = ir5[4];
            ir5[4] = (v2180_data + (v2157_data * v2178_data));
            float v2183_data = s1[64];
            float v2185_data = ir5[5];
            ir5[5] = (v2185_data + (v2157_data * v2183_data));
            float v2188_data = s1[85];
            float v2190_data = ir5[6];
            ir5[6] = (v2190_data + (v2157_data * v2188_data));
            float v2193_data = s1[89];
            float v2195_data = ir5[7];
            ir5[7] = (v2195_data + (v2157_data * v2193_data));
            float v2198_data = s1[110];
            float v2200_data = ir5[8];
            ir5[8] = (v2200_data + (v2157_data * v2198_data));
            float v2203_data = s1[115];
            float v2205_data = ir5[9];
            ir5[9] = (v2205_data + (v2157_data * v2203_data));
            float v2208_data = s1[136];
            float v2210_data = ir5[10];
            ir5[10] = (v2210_data + (v2157_data * v2208_data));
            float v2213_data = s1[132];
            float v2215_data = ir5[11];
            ir5[11] = (v2215_data + (v2157_data * v2213_data));
          }
          if (v12_lead < 12) {
            float v2221_data = r4[9];
            float v2222_data = s1[9];
            float v2224_data = ir5[0];
            ir5[0] = (v2224_data + (v2221_data * v2222_data));
            float v2227_data = s1[20];
            float v2229_data = ir5[1];
            ir5[1] = (v2229_data + (v2221_data * v2227_data));
            float v2232_data = s1[35];
            float v2234_data = ir5[2];
            ir5[2] = (v2234_data + (v2221_data * v2232_data));
            float v2237_data = s1[47];
            float v2239_data = ir5[3];
            ir5[3] = (v2239_data + (v2221_data * v2237_data));
            float v2242_data = s1[58];
            float v2244_data = ir5[4];
            ir5[4] = (v2244_data + (v2221_data * v2242_data));
            float v2247_data = s1[65];
            float v2249_data = ir5[5];
            ir5[5] = (v2249_data + (v2221_data * v2247_data));
            float v2252_data = s1[84];
            float v2254_data = ir5[6];
            ir5[6] = (v2254_data + (v2221_data * v2252_data));
            float v2257_data = s1[88];
            float v2259_data = ir5[7];
            ir5[7] = (v2259_data + (v2221_data * v2257_data));
            float v2262_data = s1[111];
            float v2264_data = ir5[8];
            ir5[8] = (v2264_data + (v2221_data * v2262_data));
            float v2267_data = s1[114];
            float v2269_data = ir5[9];
            ir5[9] = (v2269_data + (v2221_data * v2267_data));
            float v2272_data = s1[137];
            float v2274_data = ir5[10];
            ir5[10] = (v2274_data + (v2221_data * v2272_data));
            float v2277_data = s1[133];
            float v2279_data = ir5[11];
            ir5[11] = (v2279_data + (v2221_data * v2277_data));
          }
          if (v12_lead < 12) {
            float v2285_data = r4[10];
            float v2286_data = s1[10];
            float v2288_data = ir5[0];
            ir5[0] = (v2288_data + (v2285_data * v2286_data));
            float v2291_data = s1[23];
            float v2293_data = ir5[1];
            ir5[1] = (v2293_data + (v2285_data * v2291_data));
            float v2296_data = s1[32];
            float v2298_data = ir5[2];
            ir5[2] = (v2298_data + (v2285_data * v2296_data));
            float v2301_data = s1[44];
            float v2303_data = ir5[3];
            ir5[3] = (v2303_data + (v2285_data * v2301_data));
            float v2306_data = s1[57];
            float v2308_data = ir5[4];
            ir5[4] = (v2308_data + (v2285_data * v2306_data));
            float v2311_data = s1[66];
            float v2313_data = ir5[5];
            ir5[5] = (v2313_data + (v2285_data * v2311_data));
            float v2316_data = s1[87];
            float v2318_data = ir5[6];
            ir5[6] = (v2318_data + (v2285_data * v2316_data));
            float v2321_data = s1[91];
            float v2323_data = ir5[7];
            ir5[7] = (v2323_data + (v2285_data * v2321_data));
            float v2326_data = s1[108];
            float v2328_data = ir5[8];
            ir5[8] = (v2328_data + (v2285_data * v2326_data));
            float v2331_data = s1[113];
            float v2333_data = ir5[9];
            ir5[9] = (v2333_data + (v2285_data * v2331_data));
            float v2336_data = s1[138];
            float v2338_data = ir5[10];
            ir5[10] = (v2338_data + (v2285_data * v2336_data));
            float v2341_data = s1[134];
            float v2343_data = ir5[11];
            ir5[11] = (v2343_data + (v2285_data * v2341_data));
          }
          if (v12_lead < 12) {
            float v2349_data = r4[11];
            float v2350_data = s1[11];
            float v2352_data = ir5[0];
            ir5[0] = (v2352_data + (v2349_data * v2350_data));
            float v2355_data = s1[22];
            float v2357_data = ir5[1];
            ir5[1] = (v2357_data + (v2349_data * v2355_data));
            float v2360_data = s1[33];
            float v2362_data = ir5[2];
            ir5[2] = (v2362_data + (v2349_data * v2360_data));
            float v2365_data = s1[45];
            float v2367_data = ir5[3];
            ir5[3] = (v2367_data + (v2349_data * v2365_data));
            float v2370_data = s1[56];
            float v2372_data = ir5[4];
            ir5[4] = (v2372_data + (v2349_data * v2370_data));
            float v2375_data = s1[67];
            float v2377_data = ir5[5];
            ir5[5] = (v2377_data + (v2349_data * v2375_data));
            float v2380_data = s1[86];
            float v2382_data = ir5[6];
            ir5[6] = (v2382_data + (v2349_data * v2380_data));
            float v2385_data = s1[90];
            float v2387_data = ir5[7];
            ir5[7] = (v2387_data + (v2349_data * v2385_data));
            float v2390_data = s1[109];
            float v2392_data = ir5[8];
            ir5[8] = (v2392_data + (v2349_data * v2390_data));
            float v2395_data = s1[112];
            float v2397_data = ir5[9];
            ir5[9] = (v2397_data + (v2349_data * v2395_data));
            float v2400_data = s1[139];
            float v2402_data = ir5[10];
            ir5[10] = (v2402_data + (v2349_data * v2400_data));
            float v2405_data = s1[135];
            float v2407_data = ir5[11];
            ir5[11] = (v2407_data + (v2349_data * v2405_data));
          }
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2413_n1 = 0; v2413_n1 < 12; ++v2413_n1) {
              float v2415_data = ir5[v2413_n1];
              r5[v2413_n1] = v2415_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v12_lead < 12) {
            #pragma unroll
            for (int32_t v2421_i1 = 0; v2421_i1 < 12; ++v2421_i1) {
              float v2423_data = r5[v2421_i1];
              glb_m3[(v12_lead + (v2421_i1 * 12))] = v2423_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

