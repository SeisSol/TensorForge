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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 6) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 12; ++v4_i1) {
              int32_t v10_a = v4_i1 * 6;
              int32_t v11_a = v2_lead + v10_a;
              float v19_data = __ldcg(&glb_m0[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m1[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 9; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          int32_t v23_lead = threadIdx.x % 16;
          if (v23_lead < 6) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              int32_t v31_a = v25_i1 * 6;
              int32_t v32_a = v23_lead + v31_a;
              float v40_data = __ldcg(&glb_m2[(v23_lead + v31_a)]);
              int32_t v41_a = 0 + v25_i1;
              r2[v41_a] = v40_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[12]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          auto& ir1 = r1;
          int32_t v44_lead = threadIdx.x % 16;
          if (v44_lead < 6) {
            float v46_data = r0[0];
            float v47_data = s0[0];
            float v49_data = ir1[0];
            ir1[0] = (v49_data + (v46_data * v47_data));
            float v52_data = s0[12];
            float v54_data = ir1[1];
            ir1[1] = (v54_data + (v46_data * v52_data));
            float v57_data = s0[24];
            float v59_data = ir1[2];
            ir1[2] = (v59_data + (v46_data * v57_data));
            float v62_data = s0[36];
            float v64_data = ir1[3];
            ir1[3] = (v64_data + (v46_data * v62_data));
            float v67_data = s0[48];
            float v69_data = ir1[4];
            ir1[4] = (v69_data + (v46_data * v67_data));
            float v72_data = s0[60];
            float v74_data = ir1[5];
            ir1[5] = (v74_data + (v46_data * v72_data));
            float v77_data = s0[72];
            float v79_data = ir1[6];
            ir1[6] = (v79_data + (v46_data * v77_data));
            float v82_data = s0[84];
            float v84_data = ir1[7];
            ir1[7] = (v84_data + (v46_data * v82_data));
            float v87_data = s0[96];
            float v89_data = ir1[8];
            ir1[8] = (v89_data + (v46_data * v87_data));
            float v92_data = s0[108];
            float v94_data = ir1[9];
            ir1[9] = (v94_data + (v46_data * v92_data));
            float v97_data = s0[120];
            float v99_data = ir1[10];
            ir1[10] = (v99_data + (v46_data * v97_data));
            float v102_data = s0[132];
            float v104_data = ir1[11];
            ir1[11] = (v104_data + (v46_data * v102_data));
          }
          if (v44_lead < 6) {
            float v110_data = r0[1];
            float v111_data = s0[1];
            float v113_data = ir1[0];
            ir1[0] = (v113_data + (v110_data * v111_data));
            float v116_data = s0[13];
            float v118_data = ir1[1];
            ir1[1] = (v118_data + (v110_data * v116_data));
            float v121_data = s0[25];
            float v123_data = ir1[2];
            ir1[2] = (v123_data + (v110_data * v121_data));
            float v126_data = s0[37];
            float v128_data = ir1[3];
            ir1[3] = (v128_data + (v110_data * v126_data));
            float v131_data = s0[49];
            float v133_data = ir1[4];
            ir1[4] = (v133_data + (v110_data * v131_data));
            float v136_data = s0[61];
            float v138_data = ir1[5];
            ir1[5] = (v138_data + (v110_data * v136_data));
            float v141_data = s0[73];
            float v143_data = ir1[6];
            ir1[6] = (v143_data + (v110_data * v141_data));
            float v146_data = s0[85];
            float v148_data = ir1[7];
            ir1[7] = (v148_data + (v110_data * v146_data));
            float v151_data = s0[97];
            float v153_data = ir1[8];
            ir1[8] = (v153_data + (v110_data * v151_data));
            float v156_data = s0[109];
            float v158_data = ir1[9];
            ir1[9] = (v158_data + (v110_data * v156_data));
            float v161_data = s0[121];
            float v163_data = ir1[10];
            ir1[10] = (v163_data + (v110_data * v161_data));
            float v166_data = s0[133];
            float v168_data = ir1[11];
            ir1[11] = (v168_data + (v110_data * v166_data));
          }
          if (v44_lead < 6) {
            float v174_data = r0[2];
            float v175_data = s0[2];
            float v177_data = ir1[0];
            ir1[0] = (v177_data + (v174_data * v175_data));
            float v180_data = s0[14];
            float v182_data = ir1[1];
            ir1[1] = (v182_data + (v174_data * v180_data));
            float v185_data = s0[26];
            float v187_data = ir1[2];
            ir1[2] = (v187_data + (v174_data * v185_data));
            float v190_data = s0[38];
            float v192_data = ir1[3];
            ir1[3] = (v192_data + (v174_data * v190_data));
            float v195_data = s0[50];
            float v197_data = ir1[4];
            ir1[4] = (v197_data + (v174_data * v195_data));
            float v200_data = s0[62];
            float v202_data = ir1[5];
            ir1[5] = (v202_data + (v174_data * v200_data));
            float v205_data = s0[74];
            float v207_data = ir1[6];
            ir1[6] = (v207_data + (v174_data * v205_data));
            float v210_data = s0[86];
            float v212_data = ir1[7];
            ir1[7] = (v212_data + (v174_data * v210_data));
            float v215_data = s0[98];
            float v217_data = ir1[8];
            ir1[8] = (v217_data + (v174_data * v215_data));
            float v220_data = s0[110];
            float v222_data = ir1[9];
            ir1[9] = (v222_data + (v174_data * v220_data));
            float v225_data = s0[122];
            float v227_data = ir1[10];
            ir1[10] = (v227_data + (v174_data * v225_data));
            float v230_data = s0[134];
            float v232_data = ir1[11];
            ir1[11] = (v232_data + (v174_data * v230_data));
          }
          if (v44_lead < 6) {
            float v238_data = r0[3];
            float v239_data = s0[3];
            float v241_data = ir1[0];
            ir1[0] = (v241_data + (v238_data * v239_data));
            float v244_data = s0[15];
            float v246_data = ir1[1];
            ir1[1] = (v246_data + (v238_data * v244_data));
            float v249_data = s0[27];
            float v251_data = ir1[2];
            ir1[2] = (v251_data + (v238_data * v249_data));
            float v254_data = s0[39];
            float v256_data = ir1[3];
            ir1[3] = (v256_data + (v238_data * v254_data));
            float v259_data = s0[51];
            float v261_data = ir1[4];
            ir1[4] = (v261_data + (v238_data * v259_data));
            float v264_data = s0[63];
            float v266_data = ir1[5];
            ir1[5] = (v266_data + (v238_data * v264_data));
            float v269_data = s0[75];
            float v271_data = ir1[6];
            ir1[6] = (v271_data + (v238_data * v269_data));
            float v274_data = s0[87];
            float v276_data = ir1[7];
            ir1[7] = (v276_data + (v238_data * v274_data));
            float v279_data = s0[99];
            float v281_data = ir1[8];
            ir1[8] = (v281_data + (v238_data * v279_data));
            float v284_data = s0[111];
            float v286_data = ir1[9];
            ir1[9] = (v286_data + (v238_data * v284_data));
            float v289_data = s0[123];
            float v291_data = ir1[10];
            ir1[10] = (v291_data + (v238_data * v289_data));
            float v294_data = s0[135];
            float v296_data = ir1[11];
            ir1[11] = (v296_data + (v238_data * v294_data));
          }
          if (v44_lead < 6) {
            float v302_data = r0[4];
            float v303_data = s0[4];
            float v305_data = ir1[0];
            ir1[0] = (v305_data + (v302_data * v303_data));
            float v308_data = s0[16];
            float v310_data = ir1[1];
            ir1[1] = (v310_data + (v302_data * v308_data));
            float v313_data = s0[28];
            float v315_data = ir1[2];
            ir1[2] = (v315_data + (v302_data * v313_data));
            float v318_data = s0[40];
            float v320_data = ir1[3];
            ir1[3] = (v320_data + (v302_data * v318_data));
            float v323_data = s0[52];
            float v325_data = ir1[4];
            ir1[4] = (v325_data + (v302_data * v323_data));
            float v328_data = s0[64];
            float v330_data = ir1[5];
            ir1[5] = (v330_data + (v302_data * v328_data));
            float v333_data = s0[76];
            float v335_data = ir1[6];
            ir1[6] = (v335_data + (v302_data * v333_data));
            float v338_data = s0[88];
            float v340_data = ir1[7];
            ir1[7] = (v340_data + (v302_data * v338_data));
            float v343_data = s0[100];
            float v345_data = ir1[8];
            ir1[8] = (v345_data + (v302_data * v343_data));
            float v348_data = s0[112];
            float v350_data = ir1[9];
            ir1[9] = (v350_data + (v302_data * v348_data));
            float v353_data = s0[124];
            float v355_data = ir1[10];
            ir1[10] = (v355_data + (v302_data * v353_data));
            float v358_data = s0[136];
            float v360_data = ir1[11];
            ir1[11] = (v360_data + (v302_data * v358_data));
          }
          if (v44_lead < 6) {
            float v366_data = r0[5];
            float v367_data = s0[5];
            float v369_data = ir1[0];
            ir1[0] = (v369_data + (v366_data * v367_data));
            float v372_data = s0[17];
            float v374_data = ir1[1];
            ir1[1] = (v374_data + (v366_data * v372_data));
            float v377_data = s0[29];
            float v379_data = ir1[2];
            ir1[2] = (v379_data + (v366_data * v377_data));
            float v382_data = s0[41];
            float v384_data = ir1[3];
            ir1[3] = (v384_data + (v366_data * v382_data));
            float v387_data = s0[53];
            float v389_data = ir1[4];
            ir1[4] = (v389_data + (v366_data * v387_data));
            float v392_data = s0[65];
            float v394_data = ir1[5];
            ir1[5] = (v394_data + (v366_data * v392_data));
            float v397_data = s0[77];
            float v399_data = ir1[6];
            ir1[6] = (v399_data + (v366_data * v397_data));
            float v402_data = s0[89];
            float v404_data = ir1[7];
            ir1[7] = (v404_data + (v366_data * v402_data));
            float v407_data = s0[101];
            float v409_data = ir1[8];
            ir1[8] = (v409_data + (v366_data * v407_data));
            float v412_data = s0[113];
            float v414_data = ir1[9];
            ir1[9] = (v414_data + (v366_data * v412_data));
            float v417_data = s0[125];
            float v419_data = ir1[10];
            ir1[10] = (v419_data + (v366_data * v417_data));
            float v422_data = s0[137];
            float v424_data = ir1[11];
            ir1[11] = (v424_data + (v366_data * v422_data));
          }
          if (v44_lead < 6) {
            float v430_data = r0[6];
            float v431_data = s0[6];
            float v433_data = ir1[0];
            ir1[0] = (v433_data + (v430_data * v431_data));
            float v436_data = s0[18];
            float v438_data = ir1[1];
            ir1[1] = (v438_data + (v430_data * v436_data));
            float v441_data = s0[30];
            float v443_data = ir1[2];
            ir1[2] = (v443_data + (v430_data * v441_data));
            float v446_data = s0[42];
            float v448_data = ir1[3];
            ir1[3] = (v448_data + (v430_data * v446_data));
            float v451_data = s0[54];
            float v453_data = ir1[4];
            ir1[4] = (v453_data + (v430_data * v451_data));
            float v456_data = s0[66];
            float v458_data = ir1[5];
            ir1[5] = (v458_data + (v430_data * v456_data));
            float v461_data = s0[78];
            float v463_data = ir1[6];
            ir1[6] = (v463_data + (v430_data * v461_data));
            float v466_data = s0[90];
            float v468_data = ir1[7];
            ir1[7] = (v468_data + (v430_data * v466_data));
            float v471_data = s0[102];
            float v473_data = ir1[8];
            ir1[8] = (v473_data + (v430_data * v471_data));
            float v476_data = s0[114];
            float v478_data = ir1[9];
            ir1[9] = (v478_data + (v430_data * v476_data));
            float v481_data = s0[126];
            float v483_data = ir1[10];
            ir1[10] = (v483_data + (v430_data * v481_data));
            float v486_data = s0[138];
            float v488_data = ir1[11];
            ir1[11] = (v488_data + (v430_data * v486_data));
          }
          if (v44_lead < 6) {
            float v494_data = r0[7];
            float v495_data = s0[7];
            float v497_data = ir1[0];
            ir1[0] = (v497_data + (v494_data * v495_data));
            float v500_data = s0[19];
            float v502_data = ir1[1];
            ir1[1] = (v502_data + (v494_data * v500_data));
            float v505_data = s0[31];
            float v507_data = ir1[2];
            ir1[2] = (v507_data + (v494_data * v505_data));
            float v510_data = s0[43];
            float v512_data = ir1[3];
            ir1[3] = (v512_data + (v494_data * v510_data));
            float v515_data = s0[55];
            float v517_data = ir1[4];
            ir1[4] = (v517_data + (v494_data * v515_data));
            float v520_data = s0[67];
            float v522_data = ir1[5];
            ir1[5] = (v522_data + (v494_data * v520_data));
            float v525_data = s0[79];
            float v527_data = ir1[6];
            ir1[6] = (v527_data + (v494_data * v525_data));
            float v530_data = s0[91];
            float v532_data = ir1[7];
            ir1[7] = (v532_data + (v494_data * v530_data));
            float v535_data = s0[103];
            float v537_data = ir1[8];
            ir1[8] = (v537_data + (v494_data * v535_data));
            float v540_data = s0[115];
            float v542_data = ir1[9];
            ir1[9] = (v542_data + (v494_data * v540_data));
            float v545_data = s0[127];
            float v547_data = ir1[10];
            ir1[10] = (v547_data + (v494_data * v545_data));
            float v550_data = s0[139];
            float v552_data = ir1[11];
            ir1[11] = (v552_data + (v494_data * v550_data));
          }
          if (v44_lead < 6) {
            float v558_data = r0[8];
            float v559_data = s0[8];
            float v561_data = ir1[0];
            ir1[0] = (v561_data + (v558_data * v559_data));
            float v564_data = s0[20];
            float v566_data = ir1[1];
            ir1[1] = (v566_data + (v558_data * v564_data));
            float v569_data = s0[32];
            float v571_data = ir1[2];
            ir1[2] = (v571_data + (v558_data * v569_data));
            float v574_data = s0[44];
            float v576_data = ir1[3];
            ir1[3] = (v576_data + (v558_data * v574_data));
            float v579_data = s0[56];
            float v581_data = ir1[4];
            ir1[4] = (v581_data + (v558_data * v579_data));
            float v584_data = s0[68];
            float v586_data = ir1[5];
            ir1[5] = (v586_data + (v558_data * v584_data));
            float v589_data = s0[80];
            float v591_data = ir1[6];
            ir1[6] = (v591_data + (v558_data * v589_data));
            float v594_data = s0[92];
            float v596_data = ir1[7];
            ir1[7] = (v596_data + (v558_data * v594_data));
            float v599_data = s0[104];
            float v601_data = ir1[8];
            ir1[8] = (v601_data + (v558_data * v599_data));
            float v604_data = s0[116];
            float v606_data = ir1[9];
            ir1[9] = (v606_data + (v558_data * v604_data));
            float v609_data = s0[128];
            float v611_data = ir1[10];
            ir1[10] = (v611_data + (v558_data * v609_data));
            float v614_data = s0[140];
            float v616_data = ir1[11];
            ir1[11] = (v616_data + (v558_data * v614_data));
          }
          if (v44_lead < 6) {
            float v622_data = r0[9];
            float v623_data = s0[9];
            float v625_data = ir1[0];
            ir1[0] = (v625_data + (v622_data * v623_data));
            float v628_data = s0[21];
            float v630_data = ir1[1];
            ir1[1] = (v630_data + (v622_data * v628_data));
            float v633_data = s0[33];
            float v635_data = ir1[2];
            ir1[2] = (v635_data + (v622_data * v633_data));
            float v638_data = s0[45];
            float v640_data = ir1[3];
            ir1[3] = (v640_data + (v622_data * v638_data));
            float v643_data = s0[57];
            float v645_data = ir1[4];
            ir1[4] = (v645_data + (v622_data * v643_data));
            float v648_data = s0[69];
            float v650_data = ir1[5];
            ir1[5] = (v650_data + (v622_data * v648_data));
            float v653_data = s0[81];
            float v655_data = ir1[6];
            ir1[6] = (v655_data + (v622_data * v653_data));
            float v658_data = s0[93];
            float v660_data = ir1[7];
            ir1[7] = (v660_data + (v622_data * v658_data));
            float v663_data = s0[105];
            float v665_data = ir1[8];
            ir1[8] = (v665_data + (v622_data * v663_data));
            float v668_data = s0[117];
            float v670_data = ir1[9];
            ir1[9] = (v670_data + (v622_data * v668_data));
            float v673_data = s0[129];
            float v675_data = ir1[10];
            ir1[10] = (v675_data + (v622_data * v673_data));
            float v678_data = s0[141];
            float v680_data = ir1[11];
            ir1[11] = (v680_data + (v622_data * v678_data));
          }
          if (v44_lead < 6) {
            float v686_data = r0[10];
            float v687_data = s0[10];
            float v689_data = ir1[0];
            ir1[0] = (v689_data + (v686_data * v687_data));
            float v692_data = s0[22];
            float v694_data = ir1[1];
            ir1[1] = (v694_data + (v686_data * v692_data));
            float v697_data = s0[34];
            float v699_data = ir1[2];
            ir1[2] = (v699_data + (v686_data * v697_data));
            float v702_data = s0[46];
            float v704_data = ir1[3];
            ir1[3] = (v704_data + (v686_data * v702_data));
            float v707_data = s0[58];
            float v709_data = ir1[4];
            ir1[4] = (v709_data + (v686_data * v707_data));
            float v712_data = s0[70];
            float v714_data = ir1[5];
            ir1[5] = (v714_data + (v686_data * v712_data));
            float v717_data = s0[82];
            float v719_data = ir1[6];
            ir1[6] = (v719_data + (v686_data * v717_data));
            float v722_data = s0[94];
            float v724_data = ir1[7];
            ir1[7] = (v724_data + (v686_data * v722_data));
            float v727_data = s0[106];
            float v729_data = ir1[8];
            ir1[8] = (v729_data + (v686_data * v727_data));
            float v732_data = s0[118];
            float v734_data = ir1[9];
            ir1[9] = (v734_data + (v686_data * v732_data));
            float v737_data = s0[130];
            float v739_data = ir1[10];
            ir1[10] = (v739_data + (v686_data * v737_data));
            float v742_data = s0[142];
            float v744_data = ir1[11];
            ir1[11] = (v744_data + (v686_data * v742_data));
          }
          if (v44_lead < 6) {
            float v750_data = r0[11];
            float v751_data = s0[11];
            float v753_data = ir1[0];
            ir1[0] = (v753_data + (v750_data * v751_data));
            float v756_data = s0[23];
            float v758_data = ir1[1];
            ir1[1] = (v758_data + (v750_data * v756_data));
            float v761_data = s0[35];
            float v763_data = ir1[2];
            ir1[2] = (v763_data + (v750_data * v761_data));
            float v766_data = s0[47];
            float v768_data = ir1[3];
            ir1[3] = (v768_data + (v750_data * v766_data));
            float v771_data = s0[59];
            float v773_data = ir1[4];
            ir1[4] = (v773_data + (v750_data * v771_data));
            float v776_data = s0[71];
            float v778_data = ir1[5];
            ir1[5] = (v778_data + (v750_data * v776_data));
            float v781_data = s0[83];
            float v783_data = ir1[6];
            ir1[6] = (v783_data + (v750_data * v781_data));
            float v786_data = s0[95];
            float v788_data = ir1[7];
            ir1[7] = (v788_data + (v750_data * v786_data));
            float v791_data = s0[107];
            float v793_data = ir1[8];
            ir1[8] = (v793_data + (v750_data * v791_data));
            float v796_data = s0[119];
            float v798_data = ir1[9];
            ir1[9] = (v798_data + (v750_data * v796_data));
            float v801_data = s0[131];
            float v803_data = ir1[10];
            ir1[10] = (v803_data + (v750_data * v801_data));
            float v806_data = s0[143];
            float v808_data = ir1[11];
            ir1[11] = (v808_data + (v750_data * v806_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          int32_t v812_lead = threadIdx.x % 16;
          if (v812_lead < 6) {
            #pragma unroll
            for (int32_t v814_i1 = 0; v814_i1 < 12; ++v814_i1) {
              int32_t v815_a = 0 + v814_i1;
              float v817_data = r1[v814_i1];
              int32_t v824_a = v812_lead + (v814_i1 * 12);
              s1[v824_a] = v817_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          int32_t v827_lead = threadIdx.x % 16;
          if (v827_lead < 12) {
            #pragma unroll
            for (int32_t v829_i1 = 0; v829_i1 < 12; ++v829_i1) {
              int32_t v835_a = v829_i1 * 12;
              int32_t v836_a = v827_lead + v835_a;
              float v844_data = __ldcg(&glb_m4[(v827_lead + v835_a)]);
              int32_t v845_a = 0 + v829_i1;
              r4[v845_a] = v844_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          {
            // r3 = +(r2 * s0) + None
            // [(0, 6), (0, 12)] [(0, 12)]
            float ir3[12]{};
            int32_t v848_lead = threadIdx.x % 16;
            if (v848_lead < 6) {
              float v850_data = r2[0];
              float v851_data = s0[0];
              float v853_data = ir3[0];
              ir3[0] = (v853_data + (v850_data * v851_data));
              float v856_data = s0[12];
              float v858_data = ir3[1];
              ir3[1] = (v858_data + (v850_data * v856_data));
              float v861_data = s0[24];
              float v863_data = ir3[2];
              ir3[2] = (v863_data + (v850_data * v861_data));
              float v866_data = s0[36];
              float v868_data = ir3[3];
              ir3[3] = (v868_data + (v850_data * v866_data));
              float v871_data = s0[48];
              float v873_data = ir3[4];
              ir3[4] = (v873_data + (v850_data * v871_data));
              float v876_data = s0[60];
              float v878_data = ir3[5];
              ir3[5] = (v878_data + (v850_data * v876_data));
              float v881_data = s0[72];
              float v883_data = ir3[6];
              ir3[6] = (v883_data + (v850_data * v881_data));
              float v886_data = s0[84];
              float v888_data = ir3[7];
              ir3[7] = (v888_data + (v850_data * v886_data));
              float v891_data = s0[96];
              float v893_data = ir3[8];
              ir3[8] = (v893_data + (v850_data * v891_data));
              float v896_data = s0[108];
              float v898_data = ir3[9];
              ir3[9] = (v898_data + (v850_data * v896_data));
              float v901_data = s0[120];
              float v903_data = ir3[10];
              ir3[10] = (v903_data + (v850_data * v901_data));
              float v906_data = s0[132];
              float v908_data = ir3[11];
              ir3[11] = (v908_data + (v850_data * v906_data));
            }
            if (v848_lead < 6) {
              float v914_data = r2[1];
              float v915_data = s0[1];
              float v917_data = ir3[0];
              ir3[0] = (v917_data + (v914_data * v915_data));
              float v920_data = s0[13];
              float v922_data = ir3[1];
              ir3[1] = (v922_data + (v914_data * v920_data));
              float v925_data = s0[25];
              float v927_data = ir3[2];
              ir3[2] = (v927_data + (v914_data * v925_data));
              float v930_data = s0[37];
              float v932_data = ir3[3];
              ir3[3] = (v932_data + (v914_data * v930_data));
              float v935_data = s0[49];
              float v937_data = ir3[4];
              ir3[4] = (v937_data + (v914_data * v935_data));
              float v940_data = s0[61];
              float v942_data = ir3[5];
              ir3[5] = (v942_data + (v914_data * v940_data));
              float v945_data = s0[73];
              float v947_data = ir3[6];
              ir3[6] = (v947_data + (v914_data * v945_data));
              float v950_data = s0[85];
              float v952_data = ir3[7];
              ir3[7] = (v952_data + (v914_data * v950_data));
              float v955_data = s0[97];
              float v957_data = ir3[8];
              ir3[8] = (v957_data + (v914_data * v955_data));
              float v960_data = s0[109];
              float v962_data = ir3[9];
              ir3[9] = (v962_data + (v914_data * v960_data));
              float v965_data = s0[121];
              float v967_data = ir3[10];
              ir3[10] = (v967_data + (v914_data * v965_data));
              float v970_data = s0[133];
              float v972_data = ir3[11];
              ir3[11] = (v972_data + (v914_data * v970_data));
            }
            if (v848_lead < 6) {
              float v978_data = r2[2];
              float v979_data = s0[2];
              float v981_data = ir3[0];
              ir3[0] = (v981_data + (v978_data * v979_data));
              float v984_data = s0[14];
              float v986_data = ir3[1];
              ir3[1] = (v986_data + (v978_data * v984_data));
              float v989_data = s0[26];
              float v991_data = ir3[2];
              ir3[2] = (v991_data + (v978_data * v989_data));
              float v994_data = s0[38];
              float v996_data = ir3[3];
              ir3[3] = (v996_data + (v978_data * v994_data));
              float v999_data = s0[50];
              float v1001_data = ir3[4];
              ir3[4] = (v1001_data + (v978_data * v999_data));
              float v1004_data = s0[62];
              float v1006_data = ir3[5];
              ir3[5] = (v1006_data + (v978_data * v1004_data));
              float v1009_data = s0[74];
              float v1011_data = ir3[6];
              ir3[6] = (v1011_data + (v978_data * v1009_data));
              float v1014_data = s0[86];
              float v1016_data = ir3[7];
              ir3[7] = (v1016_data + (v978_data * v1014_data));
              float v1019_data = s0[98];
              float v1021_data = ir3[8];
              ir3[8] = (v1021_data + (v978_data * v1019_data));
              float v1024_data = s0[110];
              float v1026_data = ir3[9];
              ir3[9] = (v1026_data + (v978_data * v1024_data));
              float v1029_data = s0[122];
              float v1031_data = ir3[10];
              ir3[10] = (v1031_data + (v978_data * v1029_data));
              float v1034_data = s0[134];
              float v1036_data = ir3[11];
              ir3[11] = (v1036_data + (v978_data * v1034_data));
            }
            if (v848_lead < 6) {
              float v1042_data = r2[3];
              float v1043_data = s0[3];
              float v1045_data = ir3[0];
              ir3[0] = (v1045_data + (v1042_data * v1043_data));
              float v1048_data = s0[15];
              float v1050_data = ir3[1];
              ir3[1] = (v1050_data + (v1042_data * v1048_data));
              float v1053_data = s0[27];
              float v1055_data = ir3[2];
              ir3[2] = (v1055_data + (v1042_data * v1053_data));
              float v1058_data = s0[39];
              float v1060_data = ir3[3];
              ir3[3] = (v1060_data + (v1042_data * v1058_data));
              float v1063_data = s0[51];
              float v1065_data = ir3[4];
              ir3[4] = (v1065_data + (v1042_data * v1063_data));
              float v1068_data = s0[63];
              float v1070_data = ir3[5];
              ir3[5] = (v1070_data + (v1042_data * v1068_data));
              float v1073_data = s0[75];
              float v1075_data = ir3[6];
              ir3[6] = (v1075_data + (v1042_data * v1073_data));
              float v1078_data = s0[87];
              float v1080_data = ir3[7];
              ir3[7] = (v1080_data + (v1042_data * v1078_data));
              float v1083_data = s0[99];
              float v1085_data = ir3[8];
              ir3[8] = (v1085_data + (v1042_data * v1083_data));
              float v1088_data = s0[111];
              float v1090_data = ir3[9];
              ir3[9] = (v1090_data + (v1042_data * v1088_data));
              float v1093_data = s0[123];
              float v1095_data = ir3[10];
              ir3[10] = (v1095_data + (v1042_data * v1093_data));
              float v1098_data = s0[135];
              float v1100_data = ir3[11];
              ir3[11] = (v1100_data + (v1042_data * v1098_data));
            }
            if (v848_lead < 6) {
              float v1106_data = r2[4];
              float v1107_data = s0[4];
              float v1109_data = ir3[0];
              ir3[0] = (v1109_data + (v1106_data * v1107_data));
              float v1112_data = s0[16];
              float v1114_data = ir3[1];
              ir3[1] = (v1114_data + (v1106_data * v1112_data));
              float v1117_data = s0[28];
              float v1119_data = ir3[2];
              ir3[2] = (v1119_data + (v1106_data * v1117_data));
              float v1122_data = s0[40];
              float v1124_data = ir3[3];
              ir3[3] = (v1124_data + (v1106_data * v1122_data));
              float v1127_data = s0[52];
              float v1129_data = ir3[4];
              ir3[4] = (v1129_data + (v1106_data * v1127_data));
              float v1132_data = s0[64];
              float v1134_data = ir3[5];
              ir3[5] = (v1134_data + (v1106_data * v1132_data));
              float v1137_data = s0[76];
              float v1139_data = ir3[6];
              ir3[6] = (v1139_data + (v1106_data * v1137_data));
              float v1142_data = s0[88];
              float v1144_data = ir3[7];
              ir3[7] = (v1144_data + (v1106_data * v1142_data));
              float v1147_data = s0[100];
              float v1149_data = ir3[8];
              ir3[8] = (v1149_data + (v1106_data * v1147_data));
              float v1152_data = s0[112];
              float v1154_data = ir3[9];
              ir3[9] = (v1154_data + (v1106_data * v1152_data));
              float v1157_data = s0[124];
              float v1159_data = ir3[10];
              ir3[10] = (v1159_data + (v1106_data * v1157_data));
              float v1162_data = s0[136];
              float v1164_data = ir3[11];
              ir3[11] = (v1164_data + (v1106_data * v1162_data));
            }
            if (v848_lead < 6) {
              float v1170_data = r2[5];
              float v1171_data = s0[5];
              float v1173_data = ir3[0];
              ir3[0] = (v1173_data + (v1170_data * v1171_data));
              float v1176_data = s0[17];
              float v1178_data = ir3[1];
              ir3[1] = (v1178_data + (v1170_data * v1176_data));
              float v1181_data = s0[29];
              float v1183_data = ir3[2];
              ir3[2] = (v1183_data + (v1170_data * v1181_data));
              float v1186_data = s0[41];
              float v1188_data = ir3[3];
              ir3[3] = (v1188_data + (v1170_data * v1186_data));
              float v1191_data = s0[53];
              float v1193_data = ir3[4];
              ir3[4] = (v1193_data + (v1170_data * v1191_data));
              float v1196_data = s0[65];
              float v1198_data = ir3[5];
              ir3[5] = (v1198_data + (v1170_data * v1196_data));
              float v1201_data = s0[77];
              float v1203_data = ir3[6];
              ir3[6] = (v1203_data + (v1170_data * v1201_data));
              float v1206_data = s0[89];
              float v1208_data = ir3[7];
              ir3[7] = (v1208_data + (v1170_data * v1206_data));
              float v1211_data = s0[101];
              float v1213_data = ir3[8];
              ir3[8] = (v1213_data + (v1170_data * v1211_data));
              float v1216_data = s0[113];
              float v1218_data = ir3[9];
              ir3[9] = (v1218_data + (v1170_data * v1216_data));
              float v1221_data = s0[125];
              float v1223_data = ir3[10];
              ir3[10] = (v1223_data + (v1170_data * v1221_data));
              float v1226_data = s0[137];
              float v1228_data = ir3[11];
              ir3[11] = (v1228_data + (v1170_data * v1226_data));
            }
            if (v848_lead < 6) {
              float v1234_data = r2[6];
              float v1235_data = s0[6];
              float v1237_data = ir3[0];
              ir3[0] = (v1237_data + (v1234_data * v1235_data));
              float v1240_data = s0[18];
              float v1242_data = ir3[1];
              ir3[1] = (v1242_data + (v1234_data * v1240_data));
              float v1245_data = s0[30];
              float v1247_data = ir3[2];
              ir3[2] = (v1247_data + (v1234_data * v1245_data));
              float v1250_data = s0[42];
              float v1252_data = ir3[3];
              ir3[3] = (v1252_data + (v1234_data * v1250_data));
              float v1255_data = s0[54];
              float v1257_data = ir3[4];
              ir3[4] = (v1257_data + (v1234_data * v1255_data));
              float v1260_data = s0[66];
              float v1262_data = ir3[5];
              ir3[5] = (v1262_data + (v1234_data * v1260_data));
              float v1265_data = s0[78];
              float v1267_data = ir3[6];
              ir3[6] = (v1267_data + (v1234_data * v1265_data));
              float v1270_data = s0[90];
              float v1272_data = ir3[7];
              ir3[7] = (v1272_data + (v1234_data * v1270_data));
              float v1275_data = s0[102];
              float v1277_data = ir3[8];
              ir3[8] = (v1277_data + (v1234_data * v1275_data));
              float v1280_data = s0[114];
              float v1282_data = ir3[9];
              ir3[9] = (v1282_data + (v1234_data * v1280_data));
              float v1285_data = s0[126];
              float v1287_data = ir3[10];
              ir3[10] = (v1287_data + (v1234_data * v1285_data));
              float v1290_data = s0[138];
              float v1292_data = ir3[11];
              ir3[11] = (v1292_data + (v1234_data * v1290_data));
            }
            if (v848_lead < 6) {
              float v1298_data = r2[7];
              float v1299_data = s0[7];
              float v1301_data = ir3[0];
              ir3[0] = (v1301_data + (v1298_data * v1299_data));
              float v1304_data = s0[19];
              float v1306_data = ir3[1];
              ir3[1] = (v1306_data + (v1298_data * v1304_data));
              float v1309_data = s0[31];
              float v1311_data = ir3[2];
              ir3[2] = (v1311_data + (v1298_data * v1309_data));
              float v1314_data = s0[43];
              float v1316_data = ir3[3];
              ir3[3] = (v1316_data + (v1298_data * v1314_data));
              float v1319_data = s0[55];
              float v1321_data = ir3[4];
              ir3[4] = (v1321_data + (v1298_data * v1319_data));
              float v1324_data = s0[67];
              float v1326_data = ir3[5];
              ir3[5] = (v1326_data + (v1298_data * v1324_data));
              float v1329_data = s0[79];
              float v1331_data = ir3[6];
              ir3[6] = (v1331_data + (v1298_data * v1329_data));
              float v1334_data = s0[91];
              float v1336_data = ir3[7];
              ir3[7] = (v1336_data + (v1298_data * v1334_data));
              float v1339_data = s0[103];
              float v1341_data = ir3[8];
              ir3[8] = (v1341_data + (v1298_data * v1339_data));
              float v1344_data = s0[115];
              float v1346_data = ir3[9];
              ir3[9] = (v1346_data + (v1298_data * v1344_data));
              float v1349_data = s0[127];
              float v1351_data = ir3[10];
              ir3[10] = (v1351_data + (v1298_data * v1349_data));
              float v1354_data = s0[139];
              float v1356_data = ir3[11];
              ir3[11] = (v1356_data + (v1298_data * v1354_data));
            }
            if (v848_lead < 6) {
              float v1362_data = r2[8];
              float v1363_data = s0[8];
              float v1365_data = ir3[0];
              ir3[0] = (v1365_data + (v1362_data * v1363_data));
              float v1368_data = s0[20];
              float v1370_data = ir3[1];
              ir3[1] = (v1370_data + (v1362_data * v1368_data));
              float v1373_data = s0[32];
              float v1375_data = ir3[2];
              ir3[2] = (v1375_data + (v1362_data * v1373_data));
              float v1378_data = s0[44];
              float v1380_data = ir3[3];
              ir3[3] = (v1380_data + (v1362_data * v1378_data));
              float v1383_data = s0[56];
              float v1385_data = ir3[4];
              ir3[4] = (v1385_data + (v1362_data * v1383_data));
              float v1388_data = s0[68];
              float v1390_data = ir3[5];
              ir3[5] = (v1390_data + (v1362_data * v1388_data));
              float v1393_data = s0[80];
              float v1395_data = ir3[6];
              ir3[6] = (v1395_data + (v1362_data * v1393_data));
              float v1398_data = s0[92];
              float v1400_data = ir3[7];
              ir3[7] = (v1400_data + (v1362_data * v1398_data));
              float v1403_data = s0[104];
              float v1405_data = ir3[8];
              ir3[8] = (v1405_data + (v1362_data * v1403_data));
              float v1408_data = s0[116];
              float v1410_data = ir3[9];
              ir3[9] = (v1410_data + (v1362_data * v1408_data));
              float v1413_data = s0[128];
              float v1415_data = ir3[10];
              ir3[10] = (v1415_data + (v1362_data * v1413_data));
              float v1418_data = s0[140];
              float v1420_data = ir3[11];
              ir3[11] = (v1420_data + (v1362_data * v1418_data));
            }
            if (v848_lead < 6) {
              float v1426_data = r2[9];
              float v1427_data = s0[9];
              float v1429_data = ir3[0];
              ir3[0] = (v1429_data + (v1426_data * v1427_data));
              float v1432_data = s0[21];
              float v1434_data = ir3[1];
              ir3[1] = (v1434_data + (v1426_data * v1432_data));
              float v1437_data = s0[33];
              float v1439_data = ir3[2];
              ir3[2] = (v1439_data + (v1426_data * v1437_data));
              float v1442_data = s0[45];
              float v1444_data = ir3[3];
              ir3[3] = (v1444_data + (v1426_data * v1442_data));
              float v1447_data = s0[57];
              float v1449_data = ir3[4];
              ir3[4] = (v1449_data + (v1426_data * v1447_data));
              float v1452_data = s0[69];
              float v1454_data = ir3[5];
              ir3[5] = (v1454_data + (v1426_data * v1452_data));
              float v1457_data = s0[81];
              float v1459_data = ir3[6];
              ir3[6] = (v1459_data + (v1426_data * v1457_data));
              float v1462_data = s0[93];
              float v1464_data = ir3[7];
              ir3[7] = (v1464_data + (v1426_data * v1462_data));
              float v1467_data = s0[105];
              float v1469_data = ir3[8];
              ir3[8] = (v1469_data + (v1426_data * v1467_data));
              float v1472_data = s0[117];
              float v1474_data = ir3[9];
              ir3[9] = (v1474_data + (v1426_data * v1472_data));
              float v1477_data = s0[129];
              float v1479_data = ir3[10];
              ir3[10] = (v1479_data + (v1426_data * v1477_data));
              float v1482_data = s0[141];
              float v1484_data = ir3[11];
              ir3[11] = (v1484_data + (v1426_data * v1482_data));
            }
            if (v848_lead < 6) {
              float v1490_data = r2[10];
              float v1491_data = s0[10];
              float v1493_data = ir3[0];
              ir3[0] = (v1493_data + (v1490_data * v1491_data));
              float v1496_data = s0[22];
              float v1498_data = ir3[1];
              ir3[1] = (v1498_data + (v1490_data * v1496_data));
              float v1501_data = s0[34];
              float v1503_data = ir3[2];
              ir3[2] = (v1503_data + (v1490_data * v1501_data));
              float v1506_data = s0[46];
              float v1508_data = ir3[3];
              ir3[3] = (v1508_data + (v1490_data * v1506_data));
              float v1511_data = s0[58];
              float v1513_data = ir3[4];
              ir3[4] = (v1513_data + (v1490_data * v1511_data));
              float v1516_data = s0[70];
              float v1518_data = ir3[5];
              ir3[5] = (v1518_data + (v1490_data * v1516_data));
              float v1521_data = s0[82];
              float v1523_data = ir3[6];
              ir3[6] = (v1523_data + (v1490_data * v1521_data));
              float v1526_data = s0[94];
              float v1528_data = ir3[7];
              ir3[7] = (v1528_data + (v1490_data * v1526_data));
              float v1531_data = s0[106];
              float v1533_data = ir3[8];
              ir3[8] = (v1533_data + (v1490_data * v1531_data));
              float v1536_data = s0[118];
              float v1538_data = ir3[9];
              ir3[9] = (v1538_data + (v1490_data * v1536_data));
              float v1541_data = s0[130];
              float v1543_data = ir3[10];
              ir3[10] = (v1543_data + (v1490_data * v1541_data));
              float v1546_data = s0[142];
              float v1548_data = ir3[11];
              ir3[11] = (v1548_data + (v1490_data * v1546_data));
            }
            if (v848_lead < 6) {
              float v1554_data = r2[11];
              float v1555_data = s0[11];
              float v1557_data = ir3[0];
              ir3[0] = (v1557_data + (v1554_data * v1555_data));
              float v1560_data = s0[23];
              float v1562_data = ir3[1];
              ir3[1] = (v1562_data + (v1554_data * v1560_data));
              float v1565_data = s0[35];
              float v1567_data = ir3[2];
              ir3[2] = (v1567_data + (v1554_data * v1565_data));
              float v1570_data = s0[47];
              float v1572_data = ir3[3];
              ir3[3] = (v1572_data + (v1554_data * v1570_data));
              float v1575_data = s0[59];
              float v1577_data = ir3[4];
              ir3[4] = (v1577_data + (v1554_data * v1575_data));
              float v1580_data = s0[71];
              float v1582_data = ir3[5];
              ir3[5] = (v1582_data + (v1554_data * v1580_data));
              float v1585_data = s0[83];
              float v1587_data = ir3[6];
              ir3[6] = (v1587_data + (v1554_data * v1585_data));
              float v1590_data = s0[95];
              float v1592_data = ir3[7];
              ir3[7] = (v1592_data + (v1554_data * v1590_data));
              float v1595_data = s0[107];
              float v1597_data = ir3[8];
              ir3[8] = (v1597_data + (v1554_data * v1595_data));
              float v1600_data = s0[119];
              float v1602_data = ir3[9];
              ir3[9] = (v1602_data + (v1554_data * v1600_data));
              float v1605_data = s0[131];
              float v1607_data = ir3[10];
              ir3[10] = (v1607_data + (v1554_data * v1605_data));
              float v1610_data = s0[143];
              float v1612_data = ir3[11];
              ir3[11] = (v1612_data + (v1554_data * v1610_data));
            }
            if (v848_lead < 6) {
              #pragma unroll
              for (int32_t v1618_n1 = 0; v1618_n1 < 12; ++v1618_n1) {
                int32_t v1619_a = 0 + v1618_n1;
                float v1621_data = ir3[v1618_n1];
                int32_t v1622_a = 0 + v1618_n1;
                r3[v1618_n1] = v1621_data;
              }
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          int32_t v1626_lead = threadIdx.x % 16;
          if (v1626_lead < 6) {
            int32_t v1637_off = v1626_lead + 6;
            #pragma unroll
            for (int32_t v1628_i1 = 0; v1628_i1 < 12; ++v1628_i1) {
              int32_t v1629_a = 0 + v1628_i1;
              float v1631_data = r3[v1628_i1];
              int32_t v1639_a = v1637_off + (v1628_i1 * 12);
              s1[v1639_a] = v1631_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          {
            // r5 = +(r4 * s1) + None
            // [(0, 12), (0, 12)] [(0, 12)]
            float ir5[12]{};
            int32_t v1642_lead = threadIdx.x % 16;
            if (v1642_lead < 12) {
              float v1644_data = r4[0];
              float v1645_data = s1[0];
              float v1647_data = ir5[0];
              ir5[0] = (v1647_data + (v1644_data * v1645_data));
              float v1650_data = s1[12];
              float v1652_data = ir5[1];
              ir5[1] = (v1652_data + (v1644_data * v1650_data));
              float v1655_data = s1[24];
              float v1657_data = ir5[2];
              ir5[2] = (v1657_data + (v1644_data * v1655_data));
              float v1660_data = s1[36];
              float v1662_data = ir5[3];
              ir5[3] = (v1662_data + (v1644_data * v1660_data));
              float v1665_data = s1[48];
              float v1667_data = ir5[4];
              ir5[4] = (v1667_data + (v1644_data * v1665_data));
              float v1670_data = s1[60];
              float v1672_data = ir5[5];
              ir5[5] = (v1672_data + (v1644_data * v1670_data));
              float v1675_data = s1[72];
              float v1677_data = ir5[6];
              ir5[6] = (v1677_data + (v1644_data * v1675_data));
              float v1680_data = s1[84];
              float v1682_data = ir5[7];
              ir5[7] = (v1682_data + (v1644_data * v1680_data));
              float v1685_data = s1[96];
              float v1687_data = ir5[8];
              ir5[8] = (v1687_data + (v1644_data * v1685_data));
              float v1690_data = s1[108];
              float v1692_data = ir5[9];
              ir5[9] = (v1692_data + (v1644_data * v1690_data));
              float v1695_data = s1[120];
              float v1697_data = ir5[10];
              ir5[10] = (v1697_data + (v1644_data * v1695_data));
              float v1700_data = s1[132];
              float v1702_data = ir5[11];
              ir5[11] = (v1702_data + (v1644_data * v1700_data));
            }
            if (v1642_lead < 12) {
              float v1708_data = r4[1];
              float v1709_data = s1[1];
              float v1711_data = ir5[0];
              ir5[0] = (v1711_data + (v1708_data * v1709_data));
              float v1714_data = s1[13];
              float v1716_data = ir5[1];
              ir5[1] = (v1716_data + (v1708_data * v1714_data));
              float v1719_data = s1[25];
              float v1721_data = ir5[2];
              ir5[2] = (v1721_data + (v1708_data * v1719_data));
              float v1724_data = s1[37];
              float v1726_data = ir5[3];
              ir5[3] = (v1726_data + (v1708_data * v1724_data));
              float v1729_data = s1[49];
              float v1731_data = ir5[4];
              ir5[4] = (v1731_data + (v1708_data * v1729_data));
              float v1734_data = s1[61];
              float v1736_data = ir5[5];
              ir5[5] = (v1736_data + (v1708_data * v1734_data));
              float v1739_data = s1[73];
              float v1741_data = ir5[6];
              ir5[6] = (v1741_data + (v1708_data * v1739_data));
              float v1744_data = s1[85];
              float v1746_data = ir5[7];
              ir5[7] = (v1746_data + (v1708_data * v1744_data));
              float v1749_data = s1[97];
              float v1751_data = ir5[8];
              ir5[8] = (v1751_data + (v1708_data * v1749_data));
              float v1754_data = s1[109];
              float v1756_data = ir5[9];
              ir5[9] = (v1756_data + (v1708_data * v1754_data));
              float v1759_data = s1[121];
              float v1761_data = ir5[10];
              ir5[10] = (v1761_data + (v1708_data * v1759_data));
              float v1764_data = s1[133];
              float v1766_data = ir5[11];
              ir5[11] = (v1766_data + (v1708_data * v1764_data));
            }
            if (v1642_lead < 12) {
              float v1772_data = r4[2];
              float v1773_data = s1[2];
              float v1775_data = ir5[0];
              ir5[0] = (v1775_data + (v1772_data * v1773_data));
              float v1778_data = s1[14];
              float v1780_data = ir5[1];
              ir5[1] = (v1780_data + (v1772_data * v1778_data));
              float v1783_data = s1[26];
              float v1785_data = ir5[2];
              ir5[2] = (v1785_data + (v1772_data * v1783_data));
              float v1788_data = s1[38];
              float v1790_data = ir5[3];
              ir5[3] = (v1790_data + (v1772_data * v1788_data));
              float v1793_data = s1[50];
              float v1795_data = ir5[4];
              ir5[4] = (v1795_data + (v1772_data * v1793_data));
              float v1798_data = s1[62];
              float v1800_data = ir5[5];
              ir5[5] = (v1800_data + (v1772_data * v1798_data));
              float v1803_data = s1[74];
              float v1805_data = ir5[6];
              ir5[6] = (v1805_data + (v1772_data * v1803_data));
              float v1808_data = s1[86];
              float v1810_data = ir5[7];
              ir5[7] = (v1810_data + (v1772_data * v1808_data));
              float v1813_data = s1[98];
              float v1815_data = ir5[8];
              ir5[8] = (v1815_data + (v1772_data * v1813_data));
              float v1818_data = s1[110];
              float v1820_data = ir5[9];
              ir5[9] = (v1820_data + (v1772_data * v1818_data));
              float v1823_data = s1[122];
              float v1825_data = ir5[10];
              ir5[10] = (v1825_data + (v1772_data * v1823_data));
              float v1828_data = s1[134];
              float v1830_data = ir5[11];
              ir5[11] = (v1830_data + (v1772_data * v1828_data));
            }
            if (v1642_lead < 12) {
              float v1836_data = r4[3];
              float v1837_data = s1[3];
              float v1839_data = ir5[0];
              ir5[0] = (v1839_data + (v1836_data * v1837_data));
              float v1842_data = s1[15];
              float v1844_data = ir5[1];
              ir5[1] = (v1844_data + (v1836_data * v1842_data));
              float v1847_data = s1[27];
              float v1849_data = ir5[2];
              ir5[2] = (v1849_data + (v1836_data * v1847_data));
              float v1852_data = s1[39];
              float v1854_data = ir5[3];
              ir5[3] = (v1854_data + (v1836_data * v1852_data));
              float v1857_data = s1[51];
              float v1859_data = ir5[4];
              ir5[4] = (v1859_data + (v1836_data * v1857_data));
              float v1862_data = s1[63];
              float v1864_data = ir5[5];
              ir5[5] = (v1864_data + (v1836_data * v1862_data));
              float v1867_data = s1[75];
              float v1869_data = ir5[6];
              ir5[6] = (v1869_data + (v1836_data * v1867_data));
              float v1872_data = s1[87];
              float v1874_data = ir5[7];
              ir5[7] = (v1874_data + (v1836_data * v1872_data));
              float v1877_data = s1[99];
              float v1879_data = ir5[8];
              ir5[8] = (v1879_data + (v1836_data * v1877_data));
              float v1882_data = s1[111];
              float v1884_data = ir5[9];
              ir5[9] = (v1884_data + (v1836_data * v1882_data));
              float v1887_data = s1[123];
              float v1889_data = ir5[10];
              ir5[10] = (v1889_data + (v1836_data * v1887_data));
              float v1892_data = s1[135];
              float v1894_data = ir5[11];
              ir5[11] = (v1894_data + (v1836_data * v1892_data));
            }
            if (v1642_lead < 12) {
              float v1900_data = r4[4];
              float v1901_data = s1[4];
              float v1903_data = ir5[0];
              ir5[0] = (v1903_data + (v1900_data * v1901_data));
              float v1906_data = s1[16];
              float v1908_data = ir5[1];
              ir5[1] = (v1908_data + (v1900_data * v1906_data));
              float v1911_data = s1[28];
              float v1913_data = ir5[2];
              ir5[2] = (v1913_data + (v1900_data * v1911_data));
              float v1916_data = s1[40];
              float v1918_data = ir5[3];
              ir5[3] = (v1918_data + (v1900_data * v1916_data));
              float v1921_data = s1[52];
              float v1923_data = ir5[4];
              ir5[4] = (v1923_data + (v1900_data * v1921_data));
              float v1926_data = s1[64];
              float v1928_data = ir5[5];
              ir5[5] = (v1928_data + (v1900_data * v1926_data));
              float v1931_data = s1[76];
              float v1933_data = ir5[6];
              ir5[6] = (v1933_data + (v1900_data * v1931_data));
              float v1936_data = s1[88];
              float v1938_data = ir5[7];
              ir5[7] = (v1938_data + (v1900_data * v1936_data));
              float v1941_data = s1[100];
              float v1943_data = ir5[8];
              ir5[8] = (v1943_data + (v1900_data * v1941_data));
              float v1946_data = s1[112];
              float v1948_data = ir5[9];
              ir5[9] = (v1948_data + (v1900_data * v1946_data));
              float v1951_data = s1[124];
              float v1953_data = ir5[10];
              ir5[10] = (v1953_data + (v1900_data * v1951_data));
              float v1956_data = s1[136];
              float v1958_data = ir5[11];
              ir5[11] = (v1958_data + (v1900_data * v1956_data));
            }
            if (v1642_lead < 12) {
              float v1964_data = r4[5];
              float v1965_data = s1[5];
              float v1967_data = ir5[0];
              ir5[0] = (v1967_data + (v1964_data * v1965_data));
              float v1970_data = s1[17];
              float v1972_data = ir5[1];
              ir5[1] = (v1972_data + (v1964_data * v1970_data));
              float v1975_data = s1[29];
              float v1977_data = ir5[2];
              ir5[2] = (v1977_data + (v1964_data * v1975_data));
              float v1980_data = s1[41];
              float v1982_data = ir5[3];
              ir5[3] = (v1982_data + (v1964_data * v1980_data));
              float v1985_data = s1[53];
              float v1987_data = ir5[4];
              ir5[4] = (v1987_data + (v1964_data * v1985_data));
              float v1990_data = s1[65];
              float v1992_data = ir5[5];
              ir5[5] = (v1992_data + (v1964_data * v1990_data));
              float v1995_data = s1[77];
              float v1997_data = ir5[6];
              ir5[6] = (v1997_data + (v1964_data * v1995_data));
              float v2000_data = s1[89];
              float v2002_data = ir5[7];
              ir5[7] = (v2002_data + (v1964_data * v2000_data));
              float v2005_data = s1[101];
              float v2007_data = ir5[8];
              ir5[8] = (v2007_data + (v1964_data * v2005_data));
              float v2010_data = s1[113];
              float v2012_data = ir5[9];
              ir5[9] = (v2012_data + (v1964_data * v2010_data));
              float v2015_data = s1[125];
              float v2017_data = ir5[10];
              ir5[10] = (v2017_data + (v1964_data * v2015_data));
              float v2020_data = s1[137];
              float v2022_data = ir5[11];
              ir5[11] = (v2022_data + (v1964_data * v2020_data));
            }
            if (v1642_lead < 12) {
              float v2028_data = r4[6];
              float v2029_data = s1[6];
              float v2031_data = ir5[0];
              ir5[0] = (v2031_data + (v2028_data * v2029_data));
              float v2034_data = s1[18];
              float v2036_data = ir5[1];
              ir5[1] = (v2036_data + (v2028_data * v2034_data));
              float v2039_data = s1[30];
              float v2041_data = ir5[2];
              ir5[2] = (v2041_data + (v2028_data * v2039_data));
              float v2044_data = s1[42];
              float v2046_data = ir5[3];
              ir5[3] = (v2046_data + (v2028_data * v2044_data));
              float v2049_data = s1[54];
              float v2051_data = ir5[4];
              ir5[4] = (v2051_data + (v2028_data * v2049_data));
              float v2054_data = s1[66];
              float v2056_data = ir5[5];
              ir5[5] = (v2056_data + (v2028_data * v2054_data));
              float v2059_data = s1[78];
              float v2061_data = ir5[6];
              ir5[6] = (v2061_data + (v2028_data * v2059_data));
              float v2064_data = s1[90];
              float v2066_data = ir5[7];
              ir5[7] = (v2066_data + (v2028_data * v2064_data));
              float v2069_data = s1[102];
              float v2071_data = ir5[8];
              ir5[8] = (v2071_data + (v2028_data * v2069_data));
              float v2074_data = s1[114];
              float v2076_data = ir5[9];
              ir5[9] = (v2076_data + (v2028_data * v2074_data));
              float v2079_data = s1[126];
              float v2081_data = ir5[10];
              ir5[10] = (v2081_data + (v2028_data * v2079_data));
              float v2084_data = s1[138];
              float v2086_data = ir5[11];
              ir5[11] = (v2086_data + (v2028_data * v2084_data));
            }
            if (v1642_lead < 12) {
              float v2092_data = r4[7];
              float v2093_data = s1[7];
              float v2095_data = ir5[0];
              ir5[0] = (v2095_data + (v2092_data * v2093_data));
              float v2098_data = s1[19];
              float v2100_data = ir5[1];
              ir5[1] = (v2100_data + (v2092_data * v2098_data));
              float v2103_data = s1[31];
              float v2105_data = ir5[2];
              ir5[2] = (v2105_data + (v2092_data * v2103_data));
              float v2108_data = s1[43];
              float v2110_data = ir5[3];
              ir5[3] = (v2110_data + (v2092_data * v2108_data));
              float v2113_data = s1[55];
              float v2115_data = ir5[4];
              ir5[4] = (v2115_data + (v2092_data * v2113_data));
              float v2118_data = s1[67];
              float v2120_data = ir5[5];
              ir5[5] = (v2120_data + (v2092_data * v2118_data));
              float v2123_data = s1[79];
              float v2125_data = ir5[6];
              ir5[6] = (v2125_data + (v2092_data * v2123_data));
              float v2128_data = s1[91];
              float v2130_data = ir5[7];
              ir5[7] = (v2130_data + (v2092_data * v2128_data));
              float v2133_data = s1[103];
              float v2135_data = ir5[8];
              ir5[8] = (v2135_data + (v2092_data * v2133_data));
              float v2138_data = s1[115];
              float v2140_data = ir5[9];
              ir5[9] = (v2140_data + (v2092_data * v2138_data));
              float v2143_data = s1[127];
              float v2145_data = ir5[10];
              ir5[10] = (v2145_data + (v2092_data * v2143_data));
              float v2148_data = s1[139];
              float v2150_data = ir5[11];
              ir5[11] = (v2150_data + (v2092_data * v2148_data));
            }
            if (v1642_lead < 12) {
              float v2156_data = r4[8];
              float v2157_data = s1[8];
              float v2159_data = ir5[0];
              ir5[0] = (v2159_data + (v2156_data * v2157_data));
              float v2162_data = s1[20];
              float v2164_data = ir5[1];
              ir5[1] = (v2164_data + (v2156_data * v2162_data));
              float v2167_data = s1[32];
              float v2169_data = ir5[2];
              ir5[2] = (v2169_data + (v2156_data * v2167_data));
              float v2172_data = s1[44];
              float v2174_data = ir5[3];
              ir5[3] = (v2174_data + (v2156_data * v2172_data));
              float v2177_data = s1[56];
              float v2179_data = ir5[4];
              ir5[4] = (v2179_data + (v2156_data * v2177_data));
              float v2182_data = s1[68];
              float v2184_data = ir5[5];
              ir5[5] = (v2184_data + (v2156_data * v2182_data));
              float v2187_data = s1[80];
              float v2189_data = ir5[6];
              ir5[6] = (v2189_data + (v2156_data * v2187_data));
              float v2192_data = s1[92];
              float v2194_data = ir5[7];
              ir5[7] = (v2194_data + (v2156_data * v2192_data));
              float v2197_data = s1[104];
              float v2199_data = ir5[8];
              ir5[8] = (v2199_data + (v2156_data * v2197_data));
              float v2202_data = s1[116];
              float v2204_data = ir5[9];
              ir5[9] = (v2204_data + (v2156_data * v2202_data));
              float v2207_data = s1[128];
              float v2209_data = ir5[10];
              ir5[10] = (v2209_data + (v2156_data * v2207_data));
              float v2212_data = s1[140];
              float v2214_data = ir5[11];
              ir5[11] = (v2214_data + (v2156_data * v2212_data));
            }
            if (v1642_lead < 12) {
              float v2220_data = r4[9];
              float v2221_data = s1[9];
              float v2223_data = ir5[0];
              ir5[0] = (v2223_data + (v2220_data * v2221_data));
              float v2226_data = s1[21];
              float v2228_data = ir5[1];
              ir5[1] = (v2228_data + (v2220_data * v2226_data));
              float v2231_data = s1[33];
              float v2233_data = ir5[2];
              ir5[2] = (v2233_data + (v2220_data * v2231_data));
              float v2236_data = s1[45];
              float v2238_data = ir5[3];
              ir5[3] = (v2238_data + (v2220_data * v2236_data));
              float v2241_data = s1[57];
              float v2243_data = ir5[4];
              ir5[4] = (v2243_data + (v2220_data * v2241_data));
              float v2246_data = s1[69];
              float v2248_data = ir5[5];
              ir5[5] = (v2248_data + (v2220_data * v2246_data));
              float v2251_data = s1[81];
              float v2253_data = ir5[6];
              ir5[6] = (v2253_data + (v2220_data * v2251_data));
              float v2256_data = s1[93];
              float v2258_data = ir5[7];
              ir5[7] = (v2258_data + (v2220_data * v2256_data));
              float v2261_data = s1[105];
              float v2263_data = ir5[8];
              ir5[8] = (v2263_data + (v2220_data * v2261_data));
              float v2266_data = s1[117];
              float v2268_data = ir5[9];
              ir5[9] = (v2268_data + (v2220_data * v2266_data));
              float v2271_data = s1[129];
              float v2273_data = ir5[10];
              ir5[10] = (v2273_data + (v2220_data * v2271_data));
              float v2276_data = s1[141];
              float v2278_data = ir5[11];
              ir5[11] = (v2278_data + (v2220_data * v2276_data));
            }
            if (v1642_lead < 12) {
              float v2284_data = r4[10];
              float v2285_data = s1[10];
              float v2287_data = ir5[0];
              ir5[0] = (v2287_data + (v2284_data * v2285_data));
              float v2290_data = s1[22];
              float v2292_data = ir5[1];
              ir5[1] = (v2292_data + (v2284_data * v2290_data));
              float v2295_data = s1[34];
              float v2297_data = ir5[2];
              ir5[2] = (v2297_data + (v2284_data * v2295_data));
              float v2300_data = s1[46];
              float v2302_data = ir5[3];
              ir5[3] = (v2302_data + (v2284_data * v2300_data));
              float v2305_data = s1[58];
              float v2307_data = ir5[4];
              ir5[4] = (v2307_data + (v2284_data * v2305_data));
              float v2310_data = s1[70];
              float v2312_data = ir5[5];
              ir5[5] = (v2312_data + (v2284_data * v2310_data));
              float v2315_data = s1[82];
              float v2317_data = ir5[6];
              ir5[6] = (v2317_data + (v2284_data * v2315_data));
              float v2320_data = s1[94];
              float v2322_data = ir5[7];
              ir5[7] = (v2322_data + (v2284_data * v2320_data));
              float v2325_data = s1[106];
              float v2327_data = ir5[8];
              ir5[8] = (v2327_data + (v2284_data * v2325_data));
              float v2330_data = s1[118];
              float v2332_data = ir5[9];
              ir5[9] = (v2332_data + (v2284_data * v2330_data));
              float v2335_data = s1[130];
              float v2337_data = ir5[10];
              ir5[10] = (v2337_data + (v2284_data * v2335_data));
              float v2340_data = s1[142];
              float v2342_data = ir5[11];
              ir5[11] = (v2342_data + (v2284_data * v2340_data));
            }
            if (v1642_lead < 12) {
              float v2348_data = r4[11];
              float v2349_data = s1[11];
              float v2351_data = ir5[0];
              ir5[0] = (v2351_data + (v2348_data * v2349_data));
              float v2354_data = s1[23];
              float v2356_data = ir5[1];
              ir5[1] = (v2356_data + (v2348_data * v2354_data));
              float v2359_data = s1[35];
              float v2361_data = ir5[2];
              ir5[2] = (v2361_data + (v2348_data * v2359_data));
              float v2364_data = s1[47];
              float v2366_data = ir5[3];
              ir5[3] = (v2366_data + (v2348_data * v2364_data));
              float v2369_data = s1[59];
              float v2371_data = ir5[4];
              ir5[4] = (v2371_data + (v2348_data * v2369_data));
              float v2374_data = s1[71];
              float v2376_data = ir5[5];
              ir5[5] = (v2376_data + (v2348_data * v2374_data));
              float v2379_data = s1[83];
              float v2381_data = ir5[6];
              ir5[6] = (v2381_data + (v2348_data * v2379_data));
              float v2384_data = s1[95];
              float v2386_data = ir5[7];
              ir5[7] = (v2386_data + (v2348_data * v2384_data));
              float v2389_data = s1[107];
              float v2391_data = ir5[8];
              ir5[8] = (v2391_data + (v2348_data * v2389_data));
              float v2394_data = s1[119];
              float v2396_data = ir5[9];
              ir5[9] = (v2396_data + (v2348_data * v2394_data));
              float v2399_data = s1[131];
              float v2401_data = ir5[10];
              ir5[10] = (v2401_data + (v2348_data * v2399_data));
              float v2404_data = s1[143];
              float v2406_data = ir5[11];
              ir5[11] = (v2406_data + (v2348_data * v2404_data));
            }
            if (v1642_lead < 12) {
              #pragma unroll
              for (int32_t v2412_n1 = 0; v2412_n1 < 12; ++v2412_n1) {
                int32_t v2413_a = 0 + v2412_n1;
                float v2415_data = ir5[v2412_n1];
                int32_t v2416_a = 0 + v2412_n1;
                r5[v2412_n1] = v2415_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r5);
          int32_t v2420_lead = threadIdx.x % 16;
          if (v2420_lead < 12) {
            #pragma unroll
            for (int32_t v2422_i1 = 0; v2422_i1 < 12; ++v2422_i1) {
              int32_t v2423_a = 0 + v2422_i1;
              float v2425_data = r5[v2422_i1];
              int32_t v2432_a = v2420_lead + (v2422_i1 * 12);
              glb_m3[v2432_a] = v2425_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

