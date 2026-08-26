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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
              int32_t v11_a = v5_i1 * 6;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __ldcg(&glb_m0[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
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
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              int32_t v33_a = v27_i1 * 6;
              int32_t v34_a = v3_lead + v33_a;
              float v42_data = __ldcg(&glb_m2[(v3_lead + v33_a)]);
              int32_t v43_a = 0 + v27_i1;
              r2[v43_a] = v42_data;
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
          if (v3_lead < 6) {
            float v49_data = r0[0];
            float v50_data = s0[0];
            float v52_data = ir1[0];
            ir1[0] = (v52_data + (v49_data * v50_data));
            float v55_data = s0[12];
            float v57_data = ir1[1];
            ir1[1] = (v57_data + (v49_data * v55_data));
            float v60_data = s0[24];
            float v62_data = ir1[2];
            ir1[2] = (v62_data + (v49_data * v60_data));
            float v65_data = s0[36];
            float v67_data = ir1[3];
            ir1[3] = (v67_data + (v49_data * v65_data));
            float v70_data = s0[48];
            float v72_data = ir1[4];
            ir1[4] = (v72_data + (v49_data * v70_data));
            float v75_data = s0[60];
            float v77_data = ir1[5];
            ir1[5] = (v77_data + (v49_data * v75_data));
            float v80_data = s0[72];
            float v82_data = ir1[6];
            ir1[6] = (v82_data + (v49_data * v80_data));
            float v85_data = s0[84];
            float v87_data = ir1[7];
            ir1[7] = (v87_data + (v49_data * v85_data));
            float v90_data = s0[96];
            float v92_data = ir1[8];
            ir1[8] = (v92_data + (v49_data * v90_data));
            float v95_data = s0[108];
            float v97_data = ir1[9];
            ir1[9] = (v97_data + (v49_data * v95_data));
            float v100_data = s0[120];
            float v102_data = ir1[10];
            ir1[10] = (v102_data + (v49_data * v100_data));
            float v105_data = s0[132];
            float v107_data = ir1[11];
            ir1[11] = (v107_data + (v49_data * v105_data));
          }
          if (v3_lead < 6) {
            float v113_data = r0[1];
            float v114_data = s0[1];
            float v116_data = ir1[0];
            ir1[0] = (v116_data + (v113_data * v114_data));
            float v119_data = s0[13];
            float v121_data = ir1[1];
            ir1[1] = (v121_data + (v113_data * v119_data));
            float v124_data = s0[25];
            float v126_data = ir1[2];
            ir1[2] = (v126_data + (v113_data * v124_data));
            float v129_data = s0[37];
            float v131_data = ir1[3];
            ir1[3] = (v131_data + (v113_data * v129_data));
            float v134_data = s0[49];
            float v136_data = ir1[4];
            ir1[4] = (v136_data + (v113_data * v134_data));
            float v139_data = s0[61];
            float v141_data = ir1[5];
            ir1[5] = (v141_data + (v113_data * v139_data));
            float v144_data = s0[73];
            float v146_data = ir1[6];
            ir1[6] = (v146_data + (v113_data * v144_data));
            float v149_data = s0[85];
            float v151_data = ir1[7];
            ir1[7] = (v151_data + (v113_data * v149_data));
            float v154_data = s0[97];
            float v156_data = ir1[8];
            ir1[8] = (v156_data + (v113_data * v154_data));
            float v159_data = s0[109];
            float v161_data = ir1[9];
            ir1[9] = (v161_data + (v113_data * v159_data));
            float v164_data = s0[121];
            float v166_data = ir1[10];
            ir1[10] = (v166_data + (v113_data * v164_data));
            float v169_data = s0[133];
            float v171_data = ir1[11];
            ir1[11] = (v171_data + (v113_data * v169_data));
          }
          if (v3_lead < 6) {
            float v177_data = r0[2];
            float v178_data = s0[2];
            float v180_data = ir1[0];
            ir1[0] = (v180_data + (v177_data * v178_data));
            float v183_data = s0[14];
            float v185_data = ir1[1];
            ir1[1] = (v185_data + (v177_data * v183_data));
            float v188_data = s0[26];
            float v190_data = ir1[2];
            ir1[2] = (v190_data + (v177_data * v188_data));
            float v193_data = s0[38];
            float v195_data = ir1[3];
            ir1[3] = (v195_data + (v177_data * v193_data));
            float v198_data = s0[50];
            float v200_data = ir1[4];
            ir1[4] = (v200_data + (v177_data * v198_data));
            float v203_data = s0[62];
            float v205_data = ir1[5];
            ir1[5] = (v205_data + (v177_data * v203_data));
            float v208_data = s0[74];
            float v210_data = ir1[6];
            ir1[6] = (v210_data + (v177_data * v208_data));
            float v213_data = s0[86];
            float v215_data = ir1[7];
            ir1[7] = (v215_data + (v177_data * v213_data));
            float v218_data = s0[98];
            float v220_data = ir1[8];
            ir1[8] = (v220_data + (v177_data * v218_data));
            float v223_data = s0[110];
            float v225_data = ir1[9];
            ir1[9] = (v225_data + (v177_data * v223_data));
            float v228_data = s0[122];
            float v230_data = ir1[10];
            ir1[10] = (v230_data + (v177_data * v228_data));
            float v233_data = s0[134];
            float v235_data = ir1[11];
            ir1[11] = (v235_data + (v177_data * v233_data));
          }
          if (v3_lead < 6) {
            float v241_data = r0[3];
            float v242_data = s0[3];
            float v244_data = ir1[0];
            ir1[0] = (v244_data + (v241_data * v242_data));
            float v247_data = s0[15];
            float v249_data = ir1[1];
            ir1[1] = (v249_data + (v241_data * v247_data));
            float v252_data = s0[27];
            float v254_data = ir1[2];
            ir1[2] = (v254_data + (v241_data * v252_data));
            float v257_data = s0[39];
            float v259_data = ir1[3];
            ir1[3] = (v259_data + (v241_data * v257_data));
            float v262_data = s0[51];
            float v264_data = ir1[4];
            ir1[4] = (v264_data + (v241_data * v262_data));
            float v267_data = s0[63];
            float v269_data = ir1[5];
            ir1[5] = (v269_data + (v241_data * v267_data));
            float v272_data = s0[75];
            float v274_data = ir1[6];
            ir1[6] = (v274_data + (v241_data * v272_data));
            float v277_data = s0[87];
            float v279_data = ir1[7];
            ir1[7] = (v279_data + (v241_data * v277_data));
            float v282_data = s0[99];
            float v284_data = ir1[8];
            ir1[8] = (v284_data + (v241_data * v282_data));
            float v287_data = s0[111];
            float v289_data = ir1[9];
            ir1[9] = (v289_data + (v241_data * v287_data));
            float v292_data = s0[123];
            float v294_data = ir1[10];
            ir1[10] = (v294_data + (v241_data * v292_data));
            float v297_data = s0[135];
            float v299_data = ir1[11];
            ir1[11] = (v299_data + (v241_data * v297_data));
          }
          if (v3_lead < 6) {
            float v305_data = r0[4];
            float v306_data = s0[4];
            float v308_data = ir1[0];
            ir1[0] = (v308_data + (v305_data * v306_data));
            float v311_data = s0[16];
            float v313_data = ir1[1];
            ir1[1] = (v313_data + (v305_data * v311_data));
            float v316_data = s0[28];
            float v318_data = ir1[2];
            ir1[2] = (v318_data + (v305_data * v316_data));
            float v321_data = s0[40];
            float v323_data = ir1[3];
            ir1[3] = (v323_data + (v305_data * v321_data));
            float v326_data = s0[52];
            float v328_data = ir1[4];
            ir1[4] = (v328_data + (v305_data * v326_data));
            float v331_data = s0[64];
            float v333_data = ir1[5];
            ir1[5] = (v333_data + (v305_data * v331_data));
            float v336_data = s0[76];
            float v338_data = ir1[6];
            ir1[6] = (v338_data + (v305_data * v336_data));
            float v341_data = s0[88];
            float v343_data = ir1[7];
            ir1[7] = (v343_data + (v305_data * v341_data));
            float v346_data = s0[100];
            float v348_data = ir1[8];
            ir1[8] = (v348_data + (v305_data * v346_data));
            float v351_data = s0[112];
            float v353_data = ir1[9];
            ir1[9] = (v353_data + (v305_data * v351_data));
            float v356_data = s0[124];
            float v358_data = ir1[10];
            ir1[10] = (v358_data + (v305_data * v356_data));
            float v361_data = s0[136];
            float v363_data = ir1[11];
            ir1[11] = (v363_data + (v305_data * v361_data));
          }
          if (v3_lead < 6) {
            float v369_data = r0[5];
            float v370_data = s0[5];
            float v372_data = ir1[0];
            ir1[0] = (v372_data + (v369_data * v370_data));
            float v375_data = s0[17];
            float v377_data = ir1[1];
            ir1[1] = (v377_data + (v369_data * v375_data));
            float v380_data = s0[29];
            float v382_data = ir1[2];
            ir1[2] = (v382_data + (v369_data * v380_data));
            float v385_data = s0[41];
            float v387_data = ir1[3];
            ir1[3] = (v387_data + (v369_data * v385_data));
            float v390_data = s0[53];
            float v392_data = ir1[4];
            ir1[4] = (v392_data + (v369_data * v390_data));
            float v395_data = s0[65];
            float v397_data = ir1[5];
            ir1[5] = (v397_data + (v369_data * v395_data));
            float v400_data = s0[77];
            float v402_data = ir1[6];
            ir1[6] = (v402_data + (v369_data * v400_data));
            float v405_data = s0[89];
            float v407_data = ir1[7];
            ir1[7] = (v407_data + (v369_data * v405_data));
            float v410_data = s0[101];
            float v412_data = ir1[8];
            ir1[8] = (v412_data + (v369_data * v410_data));
            float v415_data = s0[113];
            float v417_data = ir1[9];
            ir1[9] = (v417_data + (v369_data * v415_data));
            float v420_data = s0[125];
            float v422_data = ir1[10];
            ir1[10] = (v422_data + (v369_data * v420_data));
            float v425_data = s0[137];
            float v427_data = ir1[11];
            ir1[11] = (v427_data + (v369_data * v425_data));
          }
          if (v3_lead < 6) {
            float v433_data = r0[6];
            float v434_data = s0[6];
            float v436_data = ir1[0];
            ir1[0] = (v436_data + (v433_data * v434_data));
            float v439_data = s0[18];
            float v441_data = ir1[1];
            ir1[1] = (v441_data + (v433_data * v439_data));
            float v444_data = s0[30];
            float v446_data = ir1[2];
            ir1[2] = (v446_data + (v433_data * v444_data));
            float v449_data = s0[42];
            float v451_data = ir1[3];
            ir1[3] = (v451_data + (v433_data * v449_data));
            float v454_data = s0[54];
            float v456_data = ir1[4];
            ir1[4] = (v456_data + (v433_data * v454_data));
            float v459_data = s0[66];
            float v461_data = ir1[5];
            ir1[5] = (v461_data + (v433_data * v459_data));
            float v464_data = s0[78];
            float v466_data = ir1[6];
            ir1[6] = (v466_data + (v433_data * v464_data));
            float v469_data = s0[90];
            float v471_data = ir1[7];
            ir1[7] = (v471_data + (v433_data * v469_data));
            float v474_data = s0[102];
            float v476_data = ir1[8];
            ir1[8] = (v476_data + (v433_data * v474_data));
            float v479_data = s0[114];
            float v481_data = ir1[9];
            ir1[9] = (v481_data + (v433_data * v479_data));
            float v484_data = s0[126];
            float v486_data = ir1[10];
            ir1[10] = (v486_data + (v433_data * v484_data));
            float v489_data = s0[138];
            float v491_data = ir1[11];
            ir1[11] = (v491_data + (v433_data * v489_data));
          }
          if (v3_lead < 6) {
            float v497_data = r0[7];
            float v498_data = s0[7];
            float v500_data = ir1[0];
            ir1[0] = (v500_data + (v497_data * v498_data));
            float v503_data = s0[19];
            float v505_data = ir1[1];
            ir1[1] = (v505_data + (v497_data * v503_data));
            float v508_data = s0[31];
            float v510_data = ir1[2];
            ir1[2] = (v510_data + (v497_data * v508_data));
            float v513_data = s0[43];
            float v515_data = ir1[3];
            ir1[3] = (v515_data + (v497_data * v513_data));
            float v518_data = s0[55];
            float v520_data = ir1[4];
            ir1[4] = (v520_data + (v497_data * v518_data));
            float v523_data = s0[67];
            float v525_data = ir1[5];
            ir1[5] = (v525_data + (v497_data * v523_data));
            float v528_data = s0[79];
            float v530_data = ir1[6];
            ir1[6] = (v530_data + (v497_data * v528_data));
            float v533_data = s0[91];
            float v535_data = ir1[7];
            ir1[7] = (v535_data + (v497_data * v533_data));
            float v538_data = s0[103];
            float v540_data = ir1[8];
            ir1[8] = (v540_data + (v497_data * v538_data));
            float v543_data = s0[115];
            float v545_data = ir1[9];
            ir1[9] = (v545_data + (v497_data * v543_data));
            float v548_data = s0[127];
            float v550_data = ir1[10];
            ir1[10] = (v550_data + (v497_data * v548_data));
            float v553_data = s0[139];
            float v555_data = ir1[11];
            ir1[11] = (v555_data + (v497_data * v553_data));
          }
          if (v3_lead < 6) {
            float v561_data = r0[8];
            float v562_data = s0[8];
            float v564_data = ir1[0];
            ir1[0] = (v564_data + (v561_data * v562_data));
            float v567_data = s0[20];
            float v569_data = ir1[1];
            ir1[1] = (v569_data + (v561_data * v567_data));
            float v572_data = s0[32];
            float v574_data = ir1[2];
            ir1[2] = (v574_data + (v561_data * v572_data));
            float v577_data = s0[44];
            float v579_data = ir1[3];
            ir1[3] = (v579_data + (v561_data * v577_data));
            float v582_data = s0[56];
            float v584_data = ir1[4];
            ir1[4] = (v584_data + (v561_data * v582_data));
            float v587_data = s0[68];
            float v589_data = ir1[5];
            ir1[5] = (v589_data + (v561_data * v587_data));
            float v592_data = s0[80];
            float v594_data = ir1[6];
            ir1[6] = (v594_data + (v561_data * v592_data));
            float v597_data = s0[92];
            float v599_data = ir1[7];
            ir1[7] = (v599_data + (v561_data * v597_data));
            float v602_data = s0[104];
            float v604_data = ir1[8];
            ir1[8] = (v604_data + (v561_data * v602_data));
            float v607_data = s0[116];
            float v609_data = ir1[9];
            ir1[9] = (v609_data + (v561_data * v607_data));
            float v612_data = s0[128];
            float v614_data = ir1[10];
            ir1[10] = (v614_data + (v561_data * v612_data));
            float v617_data = s0[140];
            float v619_data = ir1[11];
            ir1[11] = (v619_data + (v561_data * v617_data));
          }
          if (v3_lead < 6) {
            float v625_data = r0[9];
            float v626_data = s0[9];
            float v628_data = ir1[0];
            ir1[0] = (v628_data + (v625_data * v626_data));
            float v631_data = s0[21];
            float v633_data = ir1[1];
            ir1[1] = (v633_data + (v625_data * v631_data));
            float v636_data = s0[33];
            float v638_data = ir1[2];
            ir1[2] = (v638_data + (v625_data * v636_data));
            float v641_data = s0[45];
            float v643_data = ir1[3];
            ir1[3] = (v643_data + (v625_data * v641_data));
            float v646_data = s0[57];
            float v648_data = ir1[4];
            ir1[4] = (v648_data + (v625_data * v646_data));
            float v651_data = s0[69];
            float v653_data = ir1[5];
            ir1[5] = (v653_data + (v625_data * v651_data));
            float v656_data = s0[81];
            float v658_data = ir1[6];
            ir1[6] = (v658_data + (v625_data * v656_data));
            float v661_data = s0[93];
            float v663_data = ir1[7];
            ir1[7] = (v663_data + (v625_data * v661_data));
            float v666_data = s0[105];
            float v668_data = ir1[8];
            ir1[8] = (v668_data + (v625_data * v666_data));
            float v671_data = s0[117];
            float v673_data = ir1[9];
            ir1[9] = (v673_data + (v625_data * v671_data));
            float v676_data = s0[129];
            float v678_data = ir1[10];
            ir1[10] = (v678_data + (v625_data * v676_data));
            float v681_data = s0[141];
            float v683_data = ir1[11];
            ir1[11] = (v683_data + (v625_data * v681_data));
          }
          if (v3_lead < 6) {
            float v689_data = r0[10];
            float v690_data = s0[10];
            float v692_data = ir1[0];
            ir1[0] = (v692_data + (v689_data * v690_data));
            float v695_data = s0[22];
            float v697_data = ir1[1];
            ir1[1] = (v697_data + (v689_data * v695_data));
            float v700_data = s0[34];
            float v702_data = ir1[2];
            ir1[2] = (v702_data + (v689_data * v700_data));
            float v705_data = s0[46];
            float v707_data = ir1[3];
            ir1[3] = (v707_data + (v689_data * v705_data));
            float v710_data = s0[58];
            float v712_data = ir1[4];
            ir1[4] = (v712_data + (v689_data * v710_data));
            float v715_data = s0[70];
            float v717_data = ir1[5];
            ir1[5] = (v717_data + (v689_data * v715_data));
            float v720_data = s0[82];
            float v722_data = ir1[6];
            ir1[6] = (v722_data + (v689_data * v720_data));
            float v725_data = s0[94];
            float v727_data = ir1[7];
            ir1[7] = (v727_data + (v689_data * v725_data));
            float v730_data = s0[106];
            float v732_data = ir1[8];
            ir1[8] = (v732_data + (v689_data * v730_data));
            float v735_data = s0[118];
            float v737_data = ir1[9];
            ir1[9] = (v737_data + (v689_data * v735_data));
            float v740_data = s0[130];
            float v742_data = ir1[10];
            ir1[10] = (v742_data + (v689_data * v740_data));
            float v745_data = s0[142];
            float v747_data = ir1[11];
            ir1[11] = (v747_data + (v689_data * v745_data));
          }
          if (v3_lead < 6) {
            float v753_data = r0[11];
            float v754_data = s0[11];
            float v756_data = ir1[0];
            ir1[0] = (v756_data + (v753_data * v754_data));
            float v759_data = s0[23];
            float v761_data = ir1[1];
            ir1[1] = (v761_data + (v753_data * v759_data));
            float v764_data = s0[35];
            float v766_data = ir1[2];
            ir1[2] = (v766_data + (v753_data * v764_data));
            float v769_data = s0[47];
            float v771_data = ir1[3];
            ir1[3] = (v771_data + (v753_data * v769_data));
            float v774_data = s0[59];
            float v776_data = ir1[4];
            ir1[4] = (v776_data + (v753_data * v774_data));
            float v779_data = s0[71];
            float v781_data = ir1[5];
            ir1[5] = (v781_data + (v753_data * v779_data));
            float v784_data = s0[83];
            float v786_data = ir1[6];
            ir1[6] = (v786_data + (v753_data * v784_data));
            float v789_data = s0[95];
            float v791_data = ir1[7];
            ir1[7] = (v791_data + (v753_data * v789_data));
            float v794_data = s0[107];
            float v796_data = ir1[8];
            ir1[8] = (v796_data + (v753_data * v794_data));
            float v799_data = s0[119];
            float v801_data = ir1[9];
            ir1[9] = (v801_data + (v753_data * v799_data));
            float v804_data = s0[131];
            float v806_data = ir1[10];
            ir1[10] = (v806_data + (v753_data * v804_data));
            float v809_data = s0[143];
            float v811_data = ir1[11];
            ir1[11] = (v811_data + (v753_data * v809_data));
          }
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v817_i1 = 0; v817_i1 < 12; ++v817_i1) {
              int32_t v818_a = 0 + v817_i1;
              float v820_data = r1[v817_i1];
              int32_t v827_a = v3_lead + (v817_i1 * 12);
              s1[v827_a] = v820_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m4);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v833_i1 = 0; v833_i1 < 12; ++v833_i1) {
              int32_t v839_a = v833_i1 * 12;
              int32_t v840_a = v3_lead + v839_a;
              float v848_data = __ldcg(&glb_m4[(v3_lead + v839_a)]);
              int32_t v849_a = 0 + v833_i1;
              r4[v849_a] = v848_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          {
            // r3 = +(r2 * s0) + None
            // [(0, 6), (0, 12)] [(0, 12)]
            float ir3[12]{};
            if (v3_lead < 6) {
              float v855_data = r2[0];
              float v856_data = s0[0];
              float v858_data = ir3[0];
              ir3[0] = (v858_data + (v855_data * v856_data));
              float v861_data = s0[12];
              float v863_data = ir3[1];
              ir3[1] = (v863_data + (v855_data * v861_data));
              float v866_data = s0[24];
              float v868_data = ir3[2];
              ir3[2] = (v868_data + (v855_data * v866_data));
              float v871_data = s0[36];
              float v873_data = ir3[3];
              ir3[3] = (v873_data + (v855_data * v871_data));
              float v876_data = s0[48];
              float v878_data = ir3[4];
              ir3[4] = (v878_data + (v855_data * v876_data));
              float v881_data = s0[60];
              float v883_data = ir3[5];
              ir3[5] = (v883_data + (v855_data * v881_data));
              float v886_data = s0[72];
              float v888_data = ir3[6];
              ir3[6] = (v888_data + (v855_data * v886_data));
              float v891_data = s0[84];
              float v893_data = ir3[7];
              ir3[7] = (v893_data + (v855_data * v891_data));
              float v896_data = s0[96];
              float v898_data = ir3[8];
              ir3[8] = (v898_data + (v855_data * v896_data));
              float v901_data = s0[108];
              float v903_data = ir3[9];
              ir3[9] = (v903_data + (v855_data * v901_data));
              float v906_data = s0[120];
              float v908_data = ir3[10];
              ir3[10] = (v908_data + (v855_data * v906_data));
              float v911_data = s0[132];
              float v913_data = ir3[11];
              ir3[11] = (v913_data + (v855_data * v911_data));
            }
            if (v3_lead < 6) {
              float v919_data = r2[1];
              float v920_data = s0[1];
              float v922_data = ir3[0];
              ir3[0] = (v922_data + (v919_data * v920_data));
              float v925_data = s0[13];
              float v927_data = ir3[1];
              ir3[1] = (v927_data + (v919_data * v925_data));
              float v930_data = s0[25];
              float v932_data = ir3[2];
              ir3[2] = (v932_data + (v919_data * v930_data));
              float v935_data = s0[37];
              float v937_data = ir3[3];
              ir3[3] = (v937_data + (v919_data * v935_data));
              float v940_data = s0[49];
              float v942_data = ir3[4];
              ir3[4] = (v942_data + (v919_data * v940_data));
              float v945_data = s0[61];
              float v947_data = ir3[5];
              ir3[5] = (v947_data + (v919_data * v945_data));
              float v950_data = s0[73];
              float v952_data = ir3[6];
              ir3[6] = (v952_data + (v919_data * v950_data));
              float v955_data = s0[85];
              float v957_data = ir3[7];
              ir3[7] = (v957_data + (v919_data * v955_data));
              float v960_data = s0[97];
              float v962_data = ir3[8];
              ir3[8] = (v962_data + (v919_data * v960_data));
              float v965_data = s0[109];
              float v967_data = ir3[9];
              ir3[9] = (v967_data + (v919_data * v965_data));
              float v970_data = s0[121];
              float v972_data = ir3[10];
              ir3[10] = (v972_data + (v919_data * v970_data));
              float v975_data = s0[133];
              float v977_data = ir3[11];
              ir3[11] = (v977_data + (v919_data * v975_data));
            }
            if (v3_lead < 6) {
              float v983_data = r2[2];
              float v984_data = s0[2];
              float v986_data = ir3[0];
              ir3[0] = (v986_data + (v983_data * v984_data));
              float v989_data = s0[14];
              float v991_data = ir3[1];
              ir3[1] = (v991_data + (v983_data * v989_data));
              float v994_data = s0[26];
              float v996_data = ir3[2];
              ir3[2] = (v996_data + (v983_data * v994_data));
              float v999_data = s0[38];
              float v1001_data = ir3[3];
              ir3[3] = (v1001_data + (v983_data * v999_data));
              float v1004_data = s0[50];
              float v1006_data = ir3[4];
              ir3[4] = (v1006_data + (v983_data * v1004_data));
              float v1009_data = s0[62];
              float v1011_data = ir3[5];
              ir3[5] = (v1011_data + (v983_data * v1009_data));
              float v1014_data = s0[74];
              float v1016_data = ir3[6];
              ir3[6] = (v1016_data + (v983_data * v1014_data));
              float v1019_data = s0[86];
              float v1021_data = ir3[7];
              ir3[7] = (v1021_data + (v983_data * v1019_data));
              float v1024_data = s0[98];
              float v1026_data = ir3[8];
              ir3[8] = (v1026_data + (v983_data * v1024_data));
              float v1029_data = s0[110];
              float v1031_data = ir3[9];
              ir3[9] = (v1031_data + (v983_data * v1029_data));
              float v1034_data = s0[122];
              float v1036_data = ir3[10];
              ir3[10] = (v1036_data + (v983_data * v1034_data));
              float v1039_data = s0[134];
              float v1041_data = ir3[11];
              ir3[11] = (v1041_data + (v983_data * v1039_data));
            }
            if (v3_lead < 6) {
              float v1047_data = r2[3];
              float v1048_data = s0[3];
              float v1050_data = ir3[0];
              ir3[0] = (v1050_data + (v1047_data * v1048_data));
              float v1053_data = s0[15];
              float v1055_data = ir3[1];
              ir3[1] = (v1055_data + (v1047_data * v1053_data));
              float v1058_data = s0[27];
              float v1060_data = ir3[2];
              ir3[2] = (v1060_data + (v1047_data * v1058_data));
              float v1063_data = s0[39];
              float v1065_data = ir3[3];
              ir3[3] = (v1065_data + (v1047_data * v1063_data));
              float v1068_data = s0[51];
              float v1070_data = ir3[4];
              ir3[4] = (v1070_data + (v1047_data * v1068_data));
              float v1073_data = s0[63];
              float v1075_data = ir3[5];
              ir3[5] = (v1075_data + (v1047_data * v1073_data));
              float v1078_data = s0[75];
              float v1080_data = ir3[6];
              ir3[6] = (v1080_data + (v1047_data * v1078_data));
              float v1083_data = s0[87];
              float v1085_data = ir3[7];
              ir3[7] = (v1085_data + (v1047_data * v1083_data));
              float v1088_data = s0[99];
              float v1090_data = ir3[8];
              ir3[8] = (v1090_data + (v1047_data * v1088_data));
              float v1093_data = s0[111];
              float v1095_data = ir3[9];
              ir3[9] = (v1095_data + (v1047_data * v1093_data));
              float v1098_data = s0[123];
              float v1100_data = ir3[10];
              ir3[10] = (v1100_data + (v1047_data * v1098_data));
              float v1103_data = s0[135];
              float v1105_data = ir3[11];
              ir3[11] = (v1105_data + (v1047_data * v1103_data));
            }
            if (v3_lead < 6) {
              float v1111_data = r2[4];
              float v1112_data = s0[4];
              float v1114_data = ir3[0];
              ir3[0] = (v1114_data + (v1111_data * v1112_data));
              float v1117_data = s0[16];
              float v1119_data = ir3[1];
              ir3[1] = (v1119_data + (v1111_data * v1117_data));
              float v1122_data = s0[28];
              float v1124_data = ir3[2];
              ir3[2] = (v1124_data + (v1111_data * v1122_data));
              float v1127_data = s0[40];
              float v1129_data = ir3[3];
              ir3[3] = (v1129_data + (v1111_data * v1127_data));
              float v1132_data = s0[52];
              float v1134_data = ir3[4];
              ir3[4] = (v1134_data + (v1111_data * v1132_data));
              float v1137_data = s0[64];
              float v1139_data = ir3[5];
              ir3[5] = (v1139_data + (v1111_data * v1137_data));
              float v1142_data = s0[76];
              float v1144_data = ir3[6];
              ir3[6] = (v1144_data + (v1111_data * v1142_data));
              float v1147_data = s0[88];
              float v1149_data = ir3[7];
              ir3[7] = (v1149_data + (v1111_data * v1147_data));
              float v1152_data = s0[100];
              float v1154_data = ir3[8];
              ir3[8] = (v1154_data + (v1111_data * v1152_data));
              float v1157_data = s0[112];
              float v1159_data = ir3[9];
              ir3[9] = (v1159_data + (v1111_data * v1157_data));
              float v1162_data = s0[124];
              float v1164_data = ir3[10];
              ir3[10] = (v1164_data + (v1111_data * v1162_data));
              float v1167_data = s0[136];
              float v1169_data = ir3[11];
              ir3[11] = (v1169_data + (v1111_data * v1167_data));
            }
            if (v3_lead < 6) {
              float v1175_data = r2[5];
              float v1176_data = s0[5];
              float v1178_data = ir3[0];
              ir3[0] = (v1178_data + (v1175_data * v1176_data));
              float v1181_data = s0[17];
              float v1183_data = ir3[1];
              ir3[1] = (v1183_data + (v1175_data * v1181_data));
              float v1186_data = s0[29];
              float v1188_data = ir3[2];
              ir3[2] = (v1188_data + (v1175_data * v1186_data));
              float v1191_data = s0[41];
              float v1193_data = ir3[3];
              ir3[3] = (v1193_data + (v1175_data * v1191_data));
              float v1196_data = s0[53];
              float v1198_data = ir3[4];
              ir3[4] = (v1198_data + (v1175_data * v1196_data));
              float v1201_data = s0[65];
              float v1203_data = ir3[5];
              ir3[5] = (v1203_data + (v1175_data * v1201_data));
              float v1206_data = s0[77];
              float v1208_data = ir3[6];
              ir3[6] = (v1208_data + (v1175_data * v1206_data));
              float v1211_data = s0[89];
              float v1213_data = ir3[7];
              ir3[7] = (v1213_data + (v1175_data * v1211_data));
              float v1216_data = s0[101];
              float v1218_data = ir3[8];
              ir3[8] = (v1218_data + (v1175_data * v1216_data));
              float v1221_data = s0[113];
              float v1223_data = ir3[9];
              ir3[9] = (v1223_data + (v1175_data * v1221_data));
              float v1226_data = s0[125];
              float v1228_data = ir3[10];
              ir3[10] = (v1228_data + (v1175_data * v1226_data));
              float v1231_data = s0[137];
              float v1233_data = ir3[11];
              ir3[11] = (v1233_data + (v1175_data * v1231_data));
            }
            if (v3_lead < 6) {
              float v1239_data = r2[6];
              float v1240_data = s0[6];
              float v1242_data = ir3[0];
              ir3[0] = (v1242_data + (v1239_data * v1240_data));
              float v1245_data = s0[18];
              float v1247_data = ir3[1];
              ir3[1] = (v1247_data + (v1239_data * v1245_data));
              float v1250_data = s0[30];
              float v1252_data = ir3[2];
              ir3[2] = (v1252_data + (v1239_data * v1250_data));
              float v1255_data = s0[42];
              float v1257_data = ir3[3];
              ir3[3] = (v1257_data + (v1239_data * v1255_data));
              float v1260_data = s0[54];
              float v1262_data = ir3[4];
              ir3[4] = (v1262_data + (v1239_data * v1260_data));
              float v1265_data = s0[66];
              float v1267_data = ir3[5];
              ir3[5] = (v1267_data + (v1239_data * v1265_data));
              float v1270_data = s0[78];
              float v1272_data = ir3[6];
              ir3[6] = (v1272_data + (v1239_data * v1270_data));
              float v1275_data = s0[90];
              float v1277_data = ir3[7];
              ir3[7] = (v1277_data + (v1239_data * v1275_data));
              float v1280_data = s0[102];
              float v1282_data = ir3[8];
              ir3[8] = (v1282_data + (v1239_data * v1280_data));
              float v1285_data = s0[114];
              float v1287_data = ir3[9];
              ir3[9] = (v1287_data + (v1239_data * v1285_data));
              float v1290_data = s0[126];
              float v1292_data = ir3[10];
              ir3[10] = (v1292_data + (v1239_data * v1290_data));
              float v1295_data = s0[138];
              float v1297_data = ir3[11];
              ir3[11] = (v1297_data + (v1239_data * v1295_data));
            }
            if (v3_lead < 6) {
              float v1303_data = r2[7];
              float v1304_data = s0[7];
              float v1306_data = ir3[0];
              ir3[0] = (v1306_data + (v1303_data * v1304_data));
              float v1309_data = s0[19];
              float v1311_data = ir3[1];
              ir3[1] = (v1311_data + (v1303_data * v1309_data));
              float v1314_data = s0[31];
              float v1316_data = ir3[2];
              ir3[2] = (v1316_data + (v1303_data * v1314_data));
              float v1319_data = s0[43];
              float v1321_data = ir3[3];
              ir3[3] = (v1321_data + (v1303_data * v1319_data));
              float v1324_data = s0[55];
              float v1326_data = ir3[4];
              ir3[4] = (v1326_data + (v1303_data * v1324_data));
              float v1329_data = s0[67];
              float v1331_data = ir3[5];
              ir3[5] = (v1331_data + (v1303_data * v1329_data));
              float v1334_data = s0[79];
              float v1336_data = ir3[6];
              ir3[6] = (v1336_data + (v1303_data * v1334_data));
              float v1339_data = s0[91];
              float v1341_data = ir3[7];
              ir3[7] = (v1341_data + (v1303_data * v1339_data));
              float v1344_data = s0[103];
              float v1346_data = ir3[8];
              ir3[8] = (v1346_data + (v1303_data * v1344_data));
              float v1349_data = s0[115];
              float v1351_data = ir3[9];
              ir3[9] = (v1351_data + (v1303_data * v1349_data));
              float v1354_data = s0[127];
              float v1356_data = ir3[10];
              ir3[10] = (v1356_data + (v1303_data * v1354_data));
              float v1359_data = s0[139];
              float v1361_data = ir3[11];
              ir3[11] = (v1361_data + (v1303_data * v1359_data));
            }
            if (v3_lead < 6) {
              float v1367_data = r2[8];
              float v1368_data = s0[8];
              float v1370_data = ir3[0];
              ir3[0] = (v1370_data + (v1367_data * v1368_data));
              float v1373_data = s0[20];
              float v1375_data = ir3[1];
              ir3[1] = (v1375_data + (v1367_data * v1373_data));
              float v1378_data = s0[32];
              float v1380_data = ir3[2];
              ir3[2] = (v1380_data + (v1367_data * v1378_data));
              float v1383_data = s0[44];
              float v1385_data = ir3[3];
              ir3[3] = (v1385_data + (v1367_data * v1383_data));
              float v1388_data = s0[56];
              float v1390_data = ir3[4];
              ir3[4] = (v1390_data + (v1367_data * v1388_data));
              float v1393_data = s0[68];
              float v1395_data = ir3[5];
              ir3[5] = (v1395_data + (v1367_data * v1393_data));
              float v1398_data = s0[80];
              float v1400_data = ir3[6];
              ir3[6] = (v1400_data + (v1367_data * v1398_data));
              float v1403_data = s0[92];
              float v1405_data = ir3[7];
              ir3[7] = (v1405_data + (v1367_data * v1403_data));
              float v1408_data = s0[104];
              float v1410_data = ir3[8];
              ir3[8] = (v1410_data + (v1367_data * v1408_data));
              float v1413_data = s0[116];
              float v1415_data = ir3[9];
              ir3[9] = (v1415_data + (v1367_data * v1413_data));
              float v1418_data = s0[128];
              float v1420_data = ir3[10];
              ir3[10] = (v1420_data + (v1367_data * v1418_data));
              float v1423_data = s0[140];
              float v1425_data = ir3[11];
              ir3[11] = (v1425_data + (v1367_data * v1423_data));
            }
            if (v3_lead < 6) {
              float v1431_data = r2[9];
              float v1432_data = s0[9];
              float v1434_data = ir3[0];
              ir3[0] = (v1434_data + (v1431_data * v1432_data));
              float v1437_data = s0[21];
              float v1439_data = ir3[1];
              ir3[1] = (v1439_data + (v1431_data * v1437_data));
              float v1442_data = s0[33];
              float v1444_data = ir3[2];
              ir3[2] = (v1444_data + (v1431_data * v1442_data));
              float v1447_data = s0[45];
              float v1449_data = ir3[3];
              ir3[3] = (v1449_data + (v1431_data * v1447_data));
              float v1452_data = s0[57];
              float v1454_data = ir3[4];
              ir3[4] = (v1454_data + (v1431_data * v1452_data));
              float v1457_data = s0[69];
              float v1459_data = ir3[5];
              ir3[5] = (v1459_data + (v1431_data * v1457_data));
              float v1462_data = s0[81];
              float v1464_data = ir3[6];
              ir3[6] = (v1464_data + (v1431_data * v1462_data));
              float v1467_data = s0[93];
              float v1469_data = ir3[7];
              ir3[7] = (v1469_data + (v1431_data * v1467_data));
              float v1472_data = s0[105];
              float v1474_data = ir3[8];
              ir3[8] = (v1474_data + (v1431_data * v1472_data));
              float v1477_data = s0[117];
              float v1479_data = ir3[9];
              ir3[9] = (v1479_data + (v1431_data * v1477_data));
              float v1482_data = s0[129];
              float v1484_data = ir3[10];
              ir3[10] = (v1484_data + (v1431_data * v1482_data));
              float v1487_data = s0[141];
              float v1489_data = ir3[11];
              ir3[11] = (v1489_data + (v1431_data * v1487_data));
            }
            if (v3_lead < 6) {
              float v1495_data = r2[10];
              float v1496_data = s0[10];
              float v1498_data = ir3[0];
              ir3[0] = (v1498_data + (v1495_data * v1496_data));
              float v1501_data = s0[22];
              float v1503_data = ir3[1];
              ir3[1] = (v1503_data + (v1495_data * v1501_data));
              float v1506_data = s0[34];
              float v1508_data = ir3[2];
              ir3[2] = (v1508_data + (v1495_data * v1506_data));
              float v1511_data = s0[46];
              float v1513_data = ir3[3];
              ir3[3] = (v1513_data + (v1495_data * v1511_data));
              float v1516_data = s0[58];
              float v1518_data = ir3[4];
              ir3[4] = (v1518_data + (v1495_data * v1516_data));
              float v1521_data = s0[70];
              float v1523_data = ir3[5];
              ir3[5] = (v1523_data + (v1495_data * v1521_data));
              float v1526_data = s0[82];
              float v1528_data = ir3[6];
              ir3[6] = (v1528_data + (v1495_data * v1526_data));
              float v1531_data = s0[94];
              float v1533_data = ir3[7];
              ir3[7] = (v1533_data + (v1495_data * v1531_data));
              float v1536_data = s0[106];
              float v1538_data = ir3[8];
              ir3[8] = (v1538_data + (v1495_data * v1536_data));
              float v1541_data = s0[118];
              float v1543_data = ir3[9];
              ir3[9] = (v1543_data + (v1495_data * v1541_data));
              float v1546_data = s0[130];
              float v1548_data = ir3[10];
              ir3[10] = (v1548_data + (v1495_data * v1546_data));
              float v1551_data = s0[142];
              float v1553_data = ir3[11];
              ir3[11] = (v1553_data + (v1495_data * v1551_data));
            }
            if (v3_lead < 6) {
              float v1559_data = r2[11];
              float v1560_data = s0[11];
              float v1562_data = ir3[0];
              ir3[0] = (v1562_data + (v1559_data * v1560_data));
              float v1565_data = s0[23];
              float v1567_data = ir3[1];
              ir3[1] = (v1567_data + (v1559_data * v1565_data));
              float v1570_data = s0[35];
              float v1572_data = ir3[2];
              ir3[2] = (v1572_data + (v1559_data * v1570_data));
              float v1575_data = s0[47];
              float v1577_data = ir3[3];
              ir3[3] = (v1577_data + (v1559_data * v1575_data));
              float v1580_data = s0[59];
              float v1582_data = ir3[4];
              ir3[4] = (v1582_data + (v1559_data * v1580_data));
              float v1585_data = s0[71];
              float v1587_data = ir3[5];
              ir3[5] = (v1587_data + (v1559_data * v1585_data));
              float v1590_data = s0[83];
              float v1592_data = ir3[6];
              ir3[6] = (v1592_data + (v1559_data * v1590_data));
              float v1595_data = s0[95];
              float v1597_data = ir3[7];
              ir3[7] = (v1597_data + (v1559_data * v1595_data));
              float v1600_data = s0[107];
              float v1602_data = ir3[8];
              ir3[8] = (v1602_data + (v1559_data * v1600_data));
              float v1605_data = s0[119];
              float v1607_data = ir3[9];
              ir3[9] = (v1607_data + (v1559_data * v1605_data));
              float v1610_data = s0[131];
              float v1612_data = ir3[10];
              ir3[10] = (v1612_data + (v1559_data * v1610_data));
              float v1615_data = s0[143];
              float v1617_data = ir3[11];
              ir3[11] = (v1617_data + (v1559_data * v1615_data));
            }
            if (v3_lead < 6) {
              #pragma unroll
              for (int32_t v1623_n1 = 0; v1623_n1 < 12; ++v1623_n1) {
                int32_t v1624_a = 0 + v1623_n1;
                float v1626_data = ir3[v1623_n1];
                int32_t v1627_a = 0 + v1623_n1;
                r3[v1623_n1] = v1626_data;
              }
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v3_lead < 6) {
            int32_t v1642_off = v3_lead + 6;
            #pragma unroll
            for (int32_t v1633_i1 = 0; v1633_i1 < 12; ++v1633_i1) {
              int32_t v1634_a = 0 + v1633_i1;
              float v1636_data = r3[v1633_i1];
              int32_t v1644_a = v1642_off + (v1633_i1 * 12);
              s1[v1644_a] = v1636_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          {
            // r5 = +(r4 * s1) + None
            // [(0, 12), (0, 12)] [(0, 12)]
            float ir5[12]{};
            if (v3_lead < 12) {
              float v1650_data = r4[0];
              float v1651_data = s1[0];
              float v1653_data = ir5[0];
              ir5[0] = (v1653_data + (v1650_data * v1651_data));
              float v1656_data = s1[12];
              float v1658_data = ir5[1];
              ir5[1] = (v1658_data + (v1650_data * v1656_data));
              float v1661_data = s1[24];
              float v1663_data = ir5[2];
              ir5[2] = (v1663_data + (v1650_data * v1661_data));
              float v1666_data = s1[36];
              float v1668_data = ir5[3];
              ir5[3] = (v1668_data + (v1650_data * v1666_data));
              float v1671_data = s1[48];
              float v1673_data = ir5[4];
              ir5[4] = (v1673_data + (v1650_data * v1671_data));
              float v1676_data = s1[60];
              float v1678_data = ir5[5];
              ir5[5] = (v1678_data + (v1650_data * v1676_data));
              float v1681_data = s1[72];
              float v1683_data = ir5[6];
              ir5[6] = (v1683_data + (v1650_data * v1681_data));
              float v1686_data = s1[84];
              float v1688_data = ir5[7];
              ir5[7] = (v1688_data + (v1650_data * v1686_data));
              float v1691_data = s1[96];
              float v1693_data = ir5[8];
              ir5[8] = (v1693_data + (v1650_data * v1691_data));
              float v1696_data = s1[108];
              float v1698_data = ir5[9];
              ir5[9] = (v1698_data + (v1650_data * v1696_data));
              float v1701_data = s1[120];
              float v1703_data = ir5[10];
              ir5[10] = (v1703_data + (v1650_data * v1701_data));
              float v1706_data = s1[132];
              float v1708_data = ir5[11];
              ir5[11] = (v1708_data + (v1650_data * v1706_data));
            }
            if (v3_lead < 12) {
              float v1714_data = r4[1];
              float v1715_data = s1[1];
              float v1717_data = ir5[0];
              ir5[0] = (v1717_data + (v1714_data * v1715_data));
              float v1720_data = s1[13];
              float v1722_data = ir5[1];
              ir5[1] = (v1722_data + (v1714_data * v1720_data));
              float v1725_data = s1[25];
              float v1727_data = ir5[2];
              ir5[2] = (v1727_data + (v1714_data * v1725_data));
              float v1730_data = s1[37];
              float v1732_data = ir5[3];
              ir5[3] = (v1732_data + (v1714_data * v1730_data));
              float v1735_data = s1[49];
              float v1737_data = ir5[4];
              ir5[4] = (v1737_data + (v1714_data * v1735_data));
              float v1740_data = s1[61];
              float v1742_data = ir5[5];
              ir5[5] = (v1742_data + (v1714_data * v1740_data));
              float v1745_data = s1[73];
              float v1747_data = ir5[6];
              ir5[6] = (v1747_data + (v1714_data * v1745_data));
              float v1750_data = s1[85];
              float v1752_data = ir5[7];
              ir5[7] = (v1752_data + (v1714_data * v1750_data));
              float v1755_data = s1[97];
              float v1757_data = ir5[8];
              ir5[8] = (v1757_data + (v1714_data * v1755_data));
              float v1760_data = s1[109];
              float v1762_data = ir5[9];
              ir5[9] = (v1762_data + (v1714_data * v1760_data));
              float v1765_data = s1[121];
              float v1767_data = ir5[10];
              ir5[10] = (v1767_data + (v1714_data * v1765_data));
              float v1770_data = s1[133];
              float v1772_data = ir5[11];
              ir5[11] = (v1772_data + (v1714_data * v1770_data));
            }
            if (v3_lead < 12) {
              float v1778_data = r4[2];
              float v1779_data = s1[2];
              float v1781_data = ir5[0];
              ir5[0] = (v1781_data + (v1778_data * v1779_data));
              float v1784_data = s1[14];
              float v1786_data = ir5[1];
              ir5[1] = (v1786_data + (v1778_data * v1784_data));
              float v1789_data = s1[26];
              float v1791_data = ir5[2];
              ir5[2] = (v1791_data + (v1778_data * v1789_data));
              float v1794_data = s1[38];
              float v1796_data = ir5[3];
              ir5[3] = (v1796_data + (v1778_data * v1794_data));
              float v1799_data = s1[50];
              float v1801_data = ir5[4];
              ir5[4] = (v1801_data + (v1778_data * v1799_data));
              float v1804_data = s1[62];
              float v1806_data = ir5[5];
              ir5[5] = (v1806_data + (v1778_data * v1804_data));
              float v1809_data = s1[74];
              float v1811_data = ir5[6];
              ir5[6] = (v1811_data + (v1778_data * v1809_data));
              float v1814_data = s1[86];
              float v1816_data = ir5[7];
              ir5[7] = (v1816_data + (v1778_data * v1814_data));
              float v1819_data = s1[98];
              float v1821_data = ir5[8];
              ir5[8] = (v1821_data + (v1778_data * v1819_data));
              float v1824_data = s1[110];
              float v1826_data = ir5[9];
              ir5[9] = (v1826_data + (v1778_data * v1824_data));
              float v1829_data = s1[122];
              float v1831_data = ir5[10];
              ir5[10] = (v1831_data + (v1778_data * v1829_data));
              float v1834_data = s1[134];
              float v1836_data = ir5[11];
              ir5[11] = (v1836_data + (v1778_data * v1834_data));
            }
            if (v3_lead < 12) {
              float v1842_data = r4[3];
              float v1843_data = s1[3];
              float v1845_data = ir5[0];
              ir5[0] = (v1845_data + (v1842_data * v1843_data));
              float v1848_data = s1[15];
              float v1850_data = ir5[1];
              ir5[1] = (v1850_data + (v1842_data * v1848_data));
              float v1853_data = s1[27];
              float v1855_data = ir5[2];
              ir5[2] = (v1855_data + (v1842_data * v1853_data));
              float v1858_data = s1[39];
              float v1860_data = ir5[3];
              ir5[3] = (v1860_data + (v1842_data * v1858_data));
              float v1863_data = s1[51];
              float v1865_data = ir5[4];
              ir5[4] = (v1865_data + (v1842_data * v1863_data));
              float v1868_data = s1[63];
              float v1870_data = ir5[5];
              ir5[5] = (v1870_data + (v1842_data * v1868_data));
              float v1873_data = s1[75];
              float v1875_data = ir5[6];
              ir5[6] = (v1875_data + (v1842_data * v1873_data));
              float v1878_data = s1[87];
              float v1880_data = ir5[7];
              ir5[7] = (v1880_data + (v1842_data * v1878_data));
              float v1883_data = s1[99];
              float v1885_data = ir5[8];
              ir5[8] = (v1885_data + (v1842_data * v1883_data));
              float v1888_data = s1[111];
              float v1890_data = ir5[9];
              ir5[9] = (v1890_data + (v1842_data * v1888_data));
              float v1893_data = s1[123];
              float v1895_data = ir5[10];
              ir5[10] = (v1895_data + (v1842_data * v1893_data));
              float v1898_data = s1[135];
              float v1900_data = ir5[11];
              ir5[11] = (v1900_data + (v1842_data * v1898_data));
            }
            if (v3_lead < 12) {
              float v1906_data = r4[4];
              float v1907_data = s1[4];
              float v1909_data = ir5[0];
              ir5[0] = (v1909_data + (v1906_data * v1907_data));
              float v1912_data = s1[16];
              float v1914_data = ir5[1];
              ir5[1] = (v1914_data + (v1906_data * v1912_data));
              float v1917_data = s1[28];
              float v1919_data = ir5[2];
              ir5[2] = (v1919_data + (v1906_data * v1917_data));
              float v1922_data = s1[40];
              float v1924_data = ir5[3];
              ir5[3] = (v1924_data + (v1906_data * v1922_data));
              float v1927_data = s1[52];
              float v1929_data = ir5[4];
              ir5[4] = (v1929_data + (v1906_data * v1927_data));
              float v1932_data = s1[64];
              float v1934_data = ir5[5];
              ir5[5] = (v1934_data + (v1906_data * v1932_data));
              float v1937_data = s1[76];
              float v1939_data = ir5[6];
              ir5[6] = (v1939_data + (v1906_data * v1937_data));
              float v1942_data = s1[88];
              float v1944_data = ir5[7];
              ir5[7] = (v1944_data + (v1906_data * v1942_data));
              float v1947_data = s1[100];
              float v1949_data = ir5[8];
              ir5[8] = (v1949_data + (v1906_data * v1947_data));
              float v1952_data = s1[112];
              float v1954_data = ir5[9];
              ir5[9] = (v1954_data + (v1906_data * v1952_data));
              float v1957_data = s1[124];
              float v1959_data = ir5[10];
              ir5[10] = (v1959_data + (v1906_data * v1957_data));
              float v1962_data = s1[136];
              float v1964_data = ir5[11];
              ir5[11] = (v1964_data + (v1906_data * v1962_data));
            }
            if (v3_lead < 12) {
              float v1970_data = r4[5];
              float v1971_data = s1[5];
              float v1973_data = ir5[0];
              ir5[0] = (v1973_data + (v1970_data * v1971_data));
              float v1976_data = s1[17];
              float v1978_data = ir5[1];
              ir5[1] = (v1978_data + (v1970_data * v1976_data));
              float v1981_data = s1[29];
              float v1983_data = ir5[2];
              ir5[2] = (v1983_data + (v1970_data * v1981_data));
              float v1986_data = s1[41];
              float v1988_data = ir5[3];
              ir5[3] = (v1988_data + (v1970_data * v1986_data));
              float v1991_data = s1[53];
              float v1993_data = ir5[4];
              ir5[4] = (v1993_data + (v1970_data * v1991_data));
              float v1996_data = s1[65];
              float v1998_data = ir5[5];
              ir5[5] = (v1998_data + (v1970_data * v1996_data));
              float v2001_data = s1[77];
              float v2003_data = ir5[6];
              ir5[6] = (v2003_data + (v1970_data * v2001_data));
              float v2006_data = s1[89];
              float v2008_data = ir5[7];
              ir5[7] = (v2008_data + (v1970_data * v2006_data));
              float v2011_data = s1[101];
              float v2013_data = ir5[8];
              ir5[8] = (v2013_data + (v1970_data * v2011_data));
              float v2016_data = s1[113];
              float v2018_data = ir5[9];
              ir5[9] = (v2018_data + (v1970_data * v2016_data));
              float v2021_data = s1[125];
              float v2023_data = ir5[10];
              ir5[10] = (v2023_data + (v1970_data * v2021_data));
              float v2026_data = s1[137];
              float v2028_data = ir5[11];
              ir5[11] = (v2028_data + (v1970_data * v2026_data));
            }
            if (v3_lead < 12) {
              float v2034_data = r4[6];
              float v2035_data = s1[6];
              float v2037_data = ir5[0];
              ir5[0] = (v2037_data + (v2034_data * v2035_data));
              float v2040_data = s1[18];
              float v2042_data = ir5[1];
              ir5[1] = (v2042_data + (v2034_data * v2040_data));
              float v2045_data = s1[30];
              float v2047_data = ir5[2];
              ir5[2] = (v2047_data + (v2034_data * v2045_data));
              float v2050_data = s1[42];
              float v2052_data = ir5[3];
              ir5[3] = (v2052_data + (v2034_data * v2050_data));
              float v2055_data = s1[54];
              float v2057_data = ir5[4];
              ir5[4] = (v2057_data + (v2034_data * v2055_data));
              float v2060_data = s1[66];
              float v2062_data = ir5[5];
              ir5[5] = (v2062_data + (v2034_data * v2060_data));
              float v2065_data = s1[78];
              float v2067_data = ir5[6];
              ir5[6] = (v2067_data + (v2034_data * v2065_data));
              float v2070_data = s1[90];
              float v2072_data = ir5[7];
              ir5[7] = (v2072_data + (v2034_data * v2070_data));
              float v2075_data = s1[102];
              float v2077_data = ir5[8];
              ir5[8] = (v2077_data + (v2034_data * v2075_data));
              float v2080_data = s1[114];
              float v2082_data = ir5[9];
              ir5[9] = (v2082_data + (v2034_data * v2080_data));
              float v2085_data = s1[126];
              float v2087_data = ir5[10];
              ir5[10] = (v2087_data + (v2034_data * v2085_data));
              float v2090_data = s1[138];
              float v2092_data = ir5[11];
              ir5[11] = (v2092_data + (v2034_data * v2090_data));
            }
            if (v3_lead < 12) {
              float v2098_data = r4[7];
              float v2099_data = s1[7];
              float v2101_data = ir5[0];
              ir5[0] = (v2101_data + (v2098_data * v2099_data));
              float v2104_data = s1[19];
              float v2106_data = ir5[1];
              ir5[1] = (v2106_data + (v2098_data * v2104_data));
              float v2109_data = s1[31];
              float v2111_data = ir5[2];
              ir5[2] = (v2111_data + (v2098_data * v2109_data));
              float v2114_data = s1[43];
              float v2116_data = ir5[3];
              ir5[3] = (v2116_data + (v2098_data * v2114_data));
              float v2119_data = s1[55];
              float v2121_data = ir5[4];
              ir5[4] = (v2121_data + (v2098_data * v2119_data));
              float v2124_data = s1[67];
              float v2126_data = ir5[5];
              ir5[5] = (v2126_data + (v2098_data * v2124_data));
              float v2129_data = s1[79];
              float v2131_data = ir5[6];
              ir5[6] = (v2131_data + (v2098_data * v2129_data));
              float v2134_data = s1[91];
              float v2136_data = ir5[7];
              ir5[7] = (v2136_data + (v2098_data * v2134_data));
              float v2139_data = s1[103];
              float v2141_data = ir5[8];
              ir5[8] = (v2141_data + (v2098_data * v2139_data));
              float v2144_data = s1[115];
              float v2146_data = ir5[9];
              ir5[9] = (v2146_data + (v2098_data * v2144_data));
              float v2149_data = s1[127];
              float v2151_data = ir5[10];
              ir5[10] = (v2151_data + (v2098_data * v2149_data));
              float v2154_data = s1[139];
              float v2156_data = ir5[11];
              ir5[11] = (v2156_data + (v2098_data * v2154_data));
            }
            if (v3_lead < 12) {
              float v2162_data = r4[8];
              float v2163_data = s1[8];
              float v2165_data = ir5[0];
              ir5[0] = (v2165_data + (v2162_data * v2163_data));
              float v2168_data = s1[20];
              float v2170_data = ir5[1];
              ir5[1] = (v2170_data + (v2162_data * v2168_data));
              float v2173_data = s1[32];
              float v2175_data = ir5[2];
              ir5[2] = (v2175_data + (v2162_data * v2173_data));
              float v2178_data = s1[44];
              float v2180_data = ir5[3];
              ir5[3] = (v2180_data + (v2162_data * v2178_data));
              float v2183_data = s1[56];
              float v2185_data = ir5[4];
              ir5[4] = (v2185_data + (v2162_data * v2183_data));
              float v2188_data = s1[68];
              float v2190_data = ir5[5];
              ir5[5] = (v2190_data + (v2162_data * v2188_data));
              float v2193_data = s1[80];
              float v2195_data = ir5[6];
              ir5[6] = (v2195_data + (v2162_data * v2193_data));
              float v2198_data = s1[92];
              float v2200_data = ir5[7];
              ir5[7] = (v2200_data + (v2162_data * v2198_data));
              float v2203_data = s1[104];
              float v2205_data = ir5[8];
              ir5[8] = (v2205_data + (v2162_data * v2203_data));
              float v2208_data = s1[116];
              float v2210_data = ir5[9];
              ir5[9] = (v2210_data + (v2162_data * v2208_data));
              float v2213_data = s1[128];
              float v2215_data = ir5[10];
              ir5[10] = (v2215_data + (v2162_data * v2213_data));
              float v2218_data = s1[140];
              float v2220_data = ir5[11];
              ir5[11] = (v2220_data + (v2162_data * v2218_data));
            }
            if (v3_lead < 12) {
              float v2226_data = r4[9];
              float v2227_data = s1[9];
              float v2229_data = ir5[0];
              ir5[0] = (v2229_data + (v2226_data * v2227_data));
              float v2232_data = s1[21];
              float v2234_data = ir5[1];
              ir5[1] = (v2234_data + (v2226_data * v2232_data));
              float v2237_data = s1[33];
              float v2239_data = ir5[2];
              ir5[2] = (v2239_data + (v2226_data * v2237_data));
              float v2242_data = s1[45];
              float v2244_data = ir5[3];
              ir5[3] = (v2244_data + (v2226_data * v2242_data));
              float v2247_data = s1[57];
              float v2249_data = ir5[4];
              ir5[4] = (v2249_data + (v2226_data * v2247_data));
              float v2252_data = s1[69];
              float v2254_data = ir5[5];
              ir5[5] = (v2254_data + (v2226_data * v2252_data));
              float v2257_data = s1[81];
              float v2259_data = ir5[6];
              ir5[6] = (v2259_data + (v2226_data * v2257_data));
              float v2262_data = s1[93];
              float v2264_data = ir5[7];
              ir5[7] = (v2264_data + (v2226_data * v2262_data));
              float v2267_data = s1[105];
              float v2269_data = ir5[8];
              ir5[8] = (v2269_data + (v2226_data * v2267_data));
              float v2272_data = s1[117];
              float v2274_data = ir5[9];
              ir5[9] = (v2274_data + (v2226_data * v2272_data));
              float v2277_data = s1[129];
              float v2279_data = ir5[10];
              ir5[10] = (v2279_data + (v2226_data * v2277_data));
              float v2282_data = s1[141];
              float v2284_data = ir5[11];
              ir5[11] = (v2284_data + (v2226_data * v2282_data));
            }
            if (v3_lead < 12) {
              float v2290_data = r4[10];
              float v2291_data = s1[10];
              float v2293_data = ir5[0];
              ir5[0] = (v2293_data + (v2290_data * v2291_data));
              float v2296_data = s1[22];
              float v2298_data = ir5[1];
              ir5[1] = (v2298_data + (v2290_data * v2296_data));
              float v2301_data = s1[34];
              float v2303_data = ir5[2];
              ir5[2] = (v2303_data + (v2290_data * v2301_data));
              float v2306_data = s1[46];
              float v2308_data = ir5[3];
              ir5[3] = (v2308_data + (v2290_data * v2306_data));
              float v2311_data = s1[58];
              float v2313_data = ir5[4];
              ir5[4] = (v2313_data + (v2290_data * v2311_data));
              float v2316_data = s1[70];
              float v2318_data = ir5[5];
              ir5[5] = (v2318_data + (v2290_data * v2316_data));
              float v2321_data = s1[82];
              float v2323_data = ir5[6];
              ir5[6] = (v2323_data + (v2290_data * v2321_data));
              float v2326_data = s1[94];
              float v2328_data = ir5[7];
              ir5[7] = (v2328_data + (v2290_data * v2326_data));
              float v2331_data = s1[106];
              float v2333_data = ir5[8];
              ir5[8] = (v2333_data + (v2290_data * v2331_data));
              float v2336_data = s1[118];
              float v2338_data = ir5[9];
              ir5[9] = (v2338_data + (v2290_data * v2336_data));
              float v2341_data = s1[130];
              float v2343_data = ir5[10];
              ir5[10] = (v2343_data + (v2290_data * v2341_data));
              float v2346_data = s1[142];
              float v2348_data = ir5[11];
              ir5[11] = (v2348_data + (v2290_data * v2346_data));
            }
            if (v3_lead < 12) {
              float v2354_data = r4[11];
              float v2355_data = s1[11];
              float v2357_data = ir5[0];
              ir5[0] = (v2357_data + (v2354_data * v2355_data));
              float v2360_data = s1[23];
              float v2362_data = ir5[1];
              ir5[1] = (v2362_data + (v2354_data * v2360_data));
              float v2365_data = s1[35];
              float v2367_data = ir5[2];
              ir5[2] = (v2367_data + (v2354_data * v2365_data));
              float v2370_data = s1[47];
              float v2372_data = ir5[3];
              ir5[3] = (v2372_data + (v2354_data * v2370_data));
              float v2375_data = s1[59];
              float v2377_data = ir5[4];
              ir5[4] = (v2377_data + (v2354_data * v2375_data));
              float v2380_data = s1[71];
              float v2382_data = ir5[5];
              ir5[5] = (v2382_data + (v2354_data * v2380_data));
              float v2385_data = s1[83];
              float v2387_data = ir5[6];
              ir5[6] = (v2387_data + (v2354_data * v2385_data));
              float v2390_data = s1[95];
              float v2392_data = ir5[7];
              ir5[7] = (v2392_data + (v2354_data * v2390_data));
              float v2395_data = s1[107];
              float v2397_data = ir5[8];
              ir5[8] = (v2397_data + (v2354_data * v2395_data));
              float v2400_data = s1[119];
              float v2402_data = ir5[9];
              ir5[9] = (v2402_data + (v2354_data * v2400_data));
              float v2405_data = s1[131];
              float v2407_data = ir5[10];
              ir5[10] = (v2407_data + (v2354_data * v2405_data));
              float v2410_data = s1[143];
              float v2412_data = ir5[11];
              ir5[11] = (v2412_data + (v2354_data * v2410_data));
            }
            if (v3_lead < 12) {
              #pragma unroll
              for (int32_t v2418_n1 = 0; v2418_n1 < 12; ++v2418_n1) {
                int32_t v2419_a = 0 + v2418_n1;
                float v2421_data = ir5[v2418_n1];
                int32_t v2422_a = 0 + v2418_n1;
                r5[v2418_n1] = v2421_data;
              }
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v2428_i1 = 0; v2428_i1 < 12; ++v2428_i1) {
              int32_t v2429_a = 0 + v2428_i1;
              float v2431_data = r5[v2428_i1];
              int32_t v2438_a = v3_lead + (v2428_i1 * 12);
              glb_m3[v2438_a] = v2431_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

