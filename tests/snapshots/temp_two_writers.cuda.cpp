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
          // r3 = +(r2 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          if (v3_lead < 6) {
            float v856_data = r2[0];
            float v857_data = s0[0];
            float v859_data = ir3[0];
            ir3[0] = (v859_data + (v856_data * v857_data));
            float v862_data = s0[12];
            float v864_data = ir3[1];
            ir3[1] = (v864_data + (v856_data * v862_data));
            float v867_data = s0[24];
            float v869_data = ir3[2];
            ir3[2] = (v869_data + (v856_data * v867_data));
            float v872_data = s0[36];
            float v874_data = ir3[3];
            ir3[3] = (v874_data + (v856_data * v872_data));
            float v877_data = s0[48];
            float v879_data = ir3[4];
            ir3[4] = (v879_data + (v856_data * v877_data));
            float v882_data = s0[60];
            float v884_data = ir3[5];
            ir3[5] = (v884_data + (v856_data * v882_data));
            float v887_data = s0[72];
            float v889_data = ir3[6];
            ir3[6] = (v889_data + (v856_data * v887_data));
            float v892_data = s0[84];
            float v894_data = ir3[7];
            ir3[7] = (v894_data + (v856_data * v892_data));
            float v897_data = s0[96];
            float v899_data = ir3[8];
            ir3[8] = (v899_data + (v856_data * v897_data));
            float v902_data = s0[108];
            float v904_data = ir3[9];
            ir3[9] = (v904_data + (v856_data * v902_data));
            float v907_data = s0[120];
            float v909_data = ir3[10];
            ir3[10] = (v909_data + (v856_data * v907_data));
            float v912_data = s0[132];
            float v914_data = ir3[11];
            ir3[11] = (v914_data + (v856_data * v912_data));
          }
          if (v3_lead < 6) {
            float v920_data = r2[1];
            float v921_data = s0[1];
            float v923_data = ir3[0];
            ir3[0] = (v923_data + (v920_data * v921_data));
            float v926_data = s0[13];
            float v928_data = ir3[1];
            ir3[1] = (v928_data + (v920_data * v926_data));
            float v931_data = s0[25];
            float v933_data = ir3[2];
            ir3[2] = (v933_data + (v920_data * v931_data));
            float v936_data = s0[37];
            float v938_data = ir3[3];
            ir3[3] = (v938_data + (v920_data * v936_data));
            float v941_data = s0[49];
            float v943_data = ir3[4];
            ir3[4] = (v943_data + (v920_data * v941_data));
            float v946_data = s0[61];
            float v948_data = ir3[5];
            ir3[5] = (v948_data + (v920_data * v946_data));
            float v951_data = s0[73];
            float v953_data = ir3[6];
            ir3[6] = (v953_data + (v920_data * v951_data));
            float v956_data = s0[85];
            float v958_data = ir3[7];
            ir3[7] = (v958_data + (v920_data * v956_data));
            float v961_data = s0[97];
            float v963_data = ir3[8];
            ir3[8] = (v963_data + (v920_data * v961_data));
            float v966_data = s0[109];
            float v968_data = ir3[9];
            ir3[9] = (v968_data + (v920_data * v966_data));
            float v971_data = s0[121];
            float v973_data = ir3[10];
            ir3[10] = (v973_data + (v920_data * v971_data));
            float v976_data = s0[133];
            float v978_data = ir3[11];
            ir3[11] = (v978_data + (v920_data * v976_data));
          }
          if (v3_lead < 6) {
            float v984_data = r2[2];
            float v985_data = s0[2];
            float v987_data = ir3[0];
            ir3[0] = (v987_data + (v984_data * v985_data));
            float v990_data = s0[14];
            float v992_data = ir3[1];
            ir3[1] = (v992_data + (v984_data * v990_data));
            float v995_data = s0[26];
            float v997_data = ir3[2];
            ir3[2] = (v997_data + (v984_data * v995_data));
            float v1000_data = s0[38];
            float v1002_data = ir3[3];
            ir3[3] = (v1002_data + (v984_data * v1000_data));
            float v1005_data = s0[50];
            float v1007_data = ir3[4];
            ir3[4] = (v1007_data + (v984_data * v1005_data));
            float v1010_data = s0[62];
            float v1012_data = ir3[5];
            ir3[5] = (v1012_data + (v984_data * v1010_data));
            float v1015_data = s0[74];
            float v1017_data = ir3[6];
            ir3[6] = (v1017_data + (v984_data * v1015_data));
            float v1020_data = s0[86];
            float v1022_data = ir3[7];
            ir3[7] = (v1022_data + (v984_data * v1020_data));
            float v1025_data = s0[98];
            float v1027_data = ir3[8];
            ir3[8] = (v1027_data + (v984_data * v1025_data));
            float v1030_data = s0[110];
            float v1032_data = ir3[9];
            ir3[9] = (v1032_data + (v984_data * v1030_data));
            float v1035_data = s0[122];
            float v1037_data = ir3[10];
            ir3[10] = (v1037_data + (v984_data * v1035_data));
            float v1040_data = s0[134];
            float v1042_data = ir3[11];
            ir3[11] = (v1042_data + (v984_data * v1040_data));
          }
          if (v3_lead < 6) {
            float v1048_data = r2[3];
            float v1049_data = s0[3];
            float v1051_data = ir3[0];
            ir3[0] = (v1051_data + (v1048_data * v1049_data));
            float v1054_data = s0[15];
            float v1056_data = ir3[1];
            ir3[1] = (v1056_data + (v1048_data * v1054_data));
            float v1059_data = s0[27];
            float v1061_data = ir3[2];
            ir3[2] = (v1061_data + (v1048_data * v1059_data));
            float v1064_data = s0[39];
            float v1066_data = ir3[3];
            ir3[3] = (v1066_data + (v1048_data * v1064_data));
            float v1069_data = s0[51];
            float v1071_data = ir3[4];
            ir3[4] = (v1071_data + (v1048_data * v1069_data));
            float v1074_data = s0[63];
            float v1076_data = ir3[5];
            ir3[5] = (v1076_data + (v1048_data * v1074_data));
            float v1079_data = s0[75];
            float v1081_data = ir3[6];
            ir3[6] = (v1081_data + (v1048_data * v1079_data));
            float v1084_data = s0[87];
            float v1086_data = ir3[7];
            ir3[7] = (v1086_data + (v1048_data * v1084_data));
            float v1089_data = s0[99];
            float v1091_data = ir3[8];
            ir3[8] = (v1091_data + (v1048_data * v1089_data));
            float v1094_data = s0[111];
            float v1096_data = ir3[9];
            ir3[9] = (v1096_data + (v1048_data * v1094_data));
            float v1099_data = s0[123];
            float v1101_data = ir3[10];
            ir3[10] = (v1101_data + (v1048_data * v1099_data));
            float v1104_data = s0[135];
            float v1106_data = ir3[11];
            ir3[11] = (v1106_data + (v1048_data * v1104_data));
          }
          if (v3_lead < 6) {
            float v1112_data = r2[4];
            float v1113_data = s0[4];
            float v1115_data = ir3[0];
            ir3[0] = (v1115_data + (v1112_data * v1113_data));
            float v1118_data = s0[16];
            float v1120_data = ir3[1];
            ir3[1] = (v1120_data + (v1112_data * v1118_data));
            float v1123_data = s0[28];
            float v1125_data = ir3[2];
            ir3[2] = (v1125_data + (v1112_data * v1123_data));
            float v1128_data = s0[40];
            float v1130_data = ir3[3];
            ir3[3] = (v1130_data + (v1112_data * v1128_data));
            float v1133_data = s0[52];
            float v1135_data = ir3[4];
            ir3[4] = (v1135_data + (v1112_data * v1133_data));
            float v1138_data = s0[64];
            float v1140_data = ir3[5];
            ir3[5] = (v1140_data + (v1112_data * v1138_data));
            float v1143_data = s0[76];
            float v1145_data = ir3[6];
            ir3[6] = (v1145_data + (v1112_data * v1143_data));
            float v1148_data = s0[88];
            float v1150_data = ir3[7];
            ir3[7] = (v1150_data + (v1112_data * v1148_data));
            float v1153_data = s0[100];
            float v1155_data = ir3[8];
            ir3[8] = (v1155_data + (v1112_data * v1153_data));
            float v1158_data = s0[112];
            float v1160_data = ir3[9];
            ir3[9] = (v1160_data + (v1112_data * v1158_data));
            float v1163_data = s0[124];
            float v1165_data = ir3[10];
            ir3[10] = (v1165_data + (v1112_data * v1163_data));
            float v1168_data = s0[136];
            float v1170_data = ir3[11];
            ir3[11] = (v1170_data + (v1112_data * v1168_data));
          }
          if (v3_lead < 6) {
            float v1176_data = r2[5];
            float v1177_data = s0[5];
            float v1179_data = ir3[0];
            ir3[0] = (v1179_data + (v1176_data * v1177_data));
            float v1182_data = s0[17];
            float v1184_data = ir3[1];
            ir3[1] = (v1184_data + (v1176_data * v1182_data));
            float v1187_data = s0[29];
            float v1189_data = ir3[2];
            ir3[2] = (v1189_data + (v1176_data * v1187_data));
            float v1192_data = s0[41];
            float v1194_data = ir3[3];
            ir3[3] = (v1194_data + (v1176_data * v1192_data));
            float v1197_data = s0[53];
            float v1199_data = ir3[4];
            ir3[4] = (v1199_data + (v1176_data * v1197_data));
            float v1202_data = s0[65];
            float v1204_data = ir3[5];
            ir3[5] = (v1204_data + (v1176_data * v1202_data));
            float v1207_data = s0[77];
            float v1209_data = ir3[6];
            ir3[6] = (v1209_data + (v1176_data * v1207_data));
            float v1212_data = s0[89];
            float v1214_data = ir3[7];
            ir3[7] = (v1214_data + (v1176_data * v1212_data));
            float v1217_data = s0[101];
            float v1219_data = ir3[8];
            ir3[8] = (v1219_data + (v1176_data * v1217_data));
            float v1222_data = s0[113];
            float v1224_data = ir3[9];
            ir3[9] = (v1224_data + (v1176_data * v1222_data));
            float v1227_data = s0[125];
            float v1229_data = ir3[10];
            ir3[10] = (v1229_data + (v1176_data * v1227_data));
            float v1232_data = s0[137];
            float v1234_data = ir3[11];
            ir3[11] = (v1234_data + (v1176_data * v1232_data));
          }
          if (v3_lead < 6) {
            float v1240_data = r2[6];
            float v1241_data = s0[6];
            float v1243_data = ir3[0];
            ir3[0] = (v1243_data + (v1240_data * v1241_data));
            float v1246_data = s0[18];
            float v1248_data = ir3[1];
            ir3[1] = (v1248_data + (v1240_data * v1246_data));
            float v1251_data = s0[30];
            float v1253_data = ir3[2];
            ir3[2] = (v1253_data + (v1240_data * v1251_data));
            float v1256_data = s0[42];
            float v1258_data = ir3[3];
            ir3[3] = (v1258_data + (v1240_data * v1256_data));
            float v1261_data = s0[54];
            float v1263_data = ir3[4];
            ir3[4] = (v1263_data + (v1240_data * v1261_data));
            float v1266_data = s0[66];
            float v1268_data = ir3[5];
            ir3[5] = (v1268_data + (v1240_data * v1266_data));
            float v1271_data = s0[78];
            float v1273_data = ir3[6];
            ir3[6] = (v1273_data + (v1240_data * v1271_data));
            float v1276_data = s0[90];
            float v1278_data = ir3[7];
            ir3[7] = (v1278_data + (v1240_data * v1276_data));
            float v1281_data = s0[102];
            float v1283_data = ir3[8];
            ir3[8] = (v1283_data + (v1240_data * v1281_data));
            float v1286_data = s0[114];
            float v1288_data = ir3[9];
            ir3[9] = (v1288_data + (v1240_data * v1286_data));
            float v1291_data = s0[126];
            float v1293_data = ir3[10];
            ir3[10] = (v1293_data + (v1240_data * v1291_data));
            float v1296_data = s0[138];
            float v1298_data = ir3[11];
            ir3[11] = (v1298_data + (v1240_data * v1296_data));
          }
          if (v3_lead < 6) {
            float v1304_data = r2[7];
            float v1305_data = s0[7];
            float v1307_data = ir3[0];
            ir3[0] = (v1307_data + (v1304_data * v1305_data));
            float v1310_data = s0[19];
            float v1312_data = ir3[1];
            ir3[1] = (v1312_data + (v1304_data * v1310_data));
            float v1315_data = s0[31];
            float v1317_data = ir3[2];
            ir3[2] = (v1317_data + (v1304_data * v1315_data));
            float v1320_data = s0[43];
            float v1322_data = ir3[3];
            ir3[3] = (v1322_data + (v1304_data * v1320_data));
            float v1325_data = s0[55];
            float v1327_data = ir3[4];
            ir3[4] = (v1327_data + (v1304_data * v1325_data));
            float v1330_data = s0[67];
            float v1332_data = ir3[5];
            ir3[5] = (v1332_data + (v1304_data * v1330_data));
            float v1335_data = s0[79];
            float v1337_data = ir3[6];
            ir3[6] = (v1337_data + (v1304_data * v1335_data));
            float v1340_data = s0[91];
            float v1342_data = ir3[7];
            ir3[7] = (v1342_data + (v1304_data * v1340_data));
            float v1345_data = s0[103];
            float v1347_data = ir3[8];
            ir3[8] = (v1347_data + (v1304_data * v1345_data));
            float v1350_data = s0[115];
            float v1352_data = ir3[9];
            ir3[9] = (v1352_data + (v1304_data * v1350_data));
            float v1355_data = s0[127];
            float v1357_data = ir3[10];
            ir3[10] = (v1357_data + (v1304_data * v1355_data));
            float v1360_data = s0[139];
            float v1362_data = ir3[11];
            ir3[11] = (v1362_data + (v1304_data * v1360_data));
          }
          if (v3_lead < 6) {
            float v1368_data = r2[8];
            float v1369_data = s0[8];
            float v1371_data = ir3[0];
            ir3[0] = (v1371_data + (v1368_data * v1369_data));
            float v1374_data = s0[20];
            float v1376_data = ir3[1];
            ir3[1] = (v1376_data + (v1368_data * v1374_data));
            float v1379_data = s0[32];
            float v1381_data = ir3[2];
            ir3[2] = (v1381_data + (v1368_data * v1379_data));
            float v1384_data = s0[44];
            float v1386_data = ir3[3];
            ir3[3] = (v1386_data + (v1368_data * v1384_data));
            float v1389_data = s0[56];
            float v1391_data = ir3[4];
            ir3[4] = (v1391_data + (v1368_data * v1389_data));
            float v1394_data = s0[68];
            float v1396_data = ir3[5];
            ir3[5] = (v1396_data + (v1368_data * v1394_data));
            float v1399_data = s0[80];
            float v1401_data = ir3[6];
            ir3[6] = (v1401_data + (v1368_data * v1399_data));
            float v1404_data = s0[92];
            float v1406_data = ir3[7];
            ir3[7] = (v1406_data + (v1368_data * v1404_data));
            float v1409_data = s0[104];
            float v1411_data = ir3[8];
            ir3[8] = (v1411_data + (v1368_data * v1409_data));
            float v1414_data = s0[116];
            float v1416_data = ir3[9];
            ir3[9] = (v1416_data + (v1368_data * v1414_data));
            float v1419_data = s0[128];
            float v1421_data = ir3[10];
            ir3[10] = (v1421_data + (v1368_data * v1419_data));
            float v1424_data = s0[140];
            float v1426_data = ir3[11];
            ir3[11] = (v1426_data + (v1368_data * v1424_data));
          }
          if (v3_lead < 6) {
            float v1432_data = r2[9];
            float v1433_data = s0[9];
            float v1435_data = ir3[0];
            ir3[0] = (v1435_data + (v1432_data * v1433_data));
            float v1438_data = s0[21];
            float v1440_data = ir3[1];
            ir3[1] = (v1440_data + (v1432_data * v1438_data));
            float v1443_data = s0[33];
            float v1445_data = ir3[2];
            ir3[2] = (v1445_data + (v1432_data * v1443_data));
            float v1448_data = s0[45];
            float v1450_data = ir3[3];
            ir3[3] = (v1450_data + (v1432_data * v1448_data));
            float v1453_data = s0[57];
            float v1455_data = ir3[4];
            ir3[4] = (v1455_data + (v1432_data * v1453_data));
            float v1458_data = s0[69];
            float v1460_data = ir3[5];
            ir3[5] = (v1460_data + (v1432_data * v1458_data));
            float v1463_data = s0[81];
            float v1465_data = ir3[6];
            ir3[6] = (v1465_data + (v1432_data * v1463_data));
            float v1468_data = s0[93];
            float v1470_data = ir3[7];
            ir3[7] = (v1470_data + (v1432_data * v1468_data));
            float v1473_data = s0[105];
            float v1475_data = ir3[8];
            ir3[8] = (v1475_data + (v1432_data * v1473_data));
            float v1478_data = s0[117];
            float v1480_data = ir3[9];
            ir3[9] = (v1480_data + (v1432_data * v1478_data));
            float v1483_data = s0[129];
            float v1485_data = ir3[10];
            ir3[10] = (v1485_data + (v1432_data * v1483_data));
            float v1488_data = s0[141];
            float v1490_data = ir3[11];
            ir3[11] = (v1490_data + (v1432_data * v1488_data));
          }
          if (v3_lead < 6) {
            float v1496_data = r2[10];
            float v1497_data = s0[10];
            float v1499_data = ir3[0];
            ir3[0] = (v1499_data + (v1496_data * v1497_data));
            float v1502_data = s0[22];
            float v1504_data = ir3[1];
            ir3[1] = (v1504_data + (v1496_data * v1502_data));
            float v1507_data = s0[34];
            float v1509_data = ir3[2];
            ir3[2] = (v1509_data + (v1496_data * v1507_data));
            float v1512_data = s0[46];
            float v1514_data = ir3[3];
            ir3[3] = (v1514_data + (v1496_data * v1512_data));
            float v1517_data = s0[58];
            float v1519_data = ir3[4];
            ir3[4] = (v1519_data + (v1496_data * v1517_data));
            float v1522_data = s0[70];
            float v1524_data = ir3[5];
            ir3[5] = (v1524_data + (v1496_data * v1522_data));
            float v1527_data = s0[82];
            float v1529_data = ir3[6];
            ir3[6] = (v1529_data + (v1496_data * v1527_data));
            float v1532_data = s0[94];
            float v1534_data = ir3[7];
            ir3[7] = (v1534_data + (v1496_data * v1532_data));
            float v1537_data = s0[106];
            float v1539_data = ir3[8];
            ir3[8] = (v1539_data + (v1496_data * v1537_data));
            float v1542_data = s0[118];
            float v1544_data = ir3[9];
            ir3[9] = (v1544_data + (v1496_data * v1542_data));
            float v1547_data = s0[130];
            float v1549_data = ir3[10];
            ir3[10] = (v1549_data + (v1496_data * v1547_data));
            float v1552_data = s0[142];
            float v1554_data = ir3[11];
            ir3[11] = (v1554_data + (v1496_data * v1552_data));
          }
          if (v3_lead < 6) {
            float v1560_data = r2[11];
            float v1561_data = s0[11];
            float v1563_data = ir3[0];
            ir3[0] = (v1563_data + (v1560_data * v1561_data));
            float v1566_data = s0[23];
            float v1568_data = ir3[1];
            ir3[1] = (v1568_data + (v1560_data * v1566_data));
            float v1571_data = s0[35];
            float v1573_data = ir3[2];
            ir3[2] = (v1573_data + (v1560_data * v1571_data));
            float v1576_data = s0[47];
            float v1578_data = ir3[3];
            ir3[3] = (v1578_data + (v1560_data * v1576_data));
            float v1581_data = s0[59];
            float v1583_data = ir3[4];
            ir3[4] = (v1583_data + (v1560_data * v1581_data));
            float v1586_data = s0[71];
            float v1588_data = ir3[5];
            ir3[5] = (v1588_data + (v1560_data * v1586_data));
            float v1591_data = s0[83];
            float v1593_data = ir3[6];
            ir3[6] = (v1593_data + (v1560_data * v1591_data));
            float v1596_data = s0[95];
            float v1598_data = ir3[7];
            ir3[7] = (v1598_data + (v1560_data * v1596_data));
            float v1601_data = s0[107];
            float v1603_data = ir3[8];
            ir3[8] = (v1603_data + (v1560_data * v1601_data));
            float v1606_data = s0[119];
            float v1608_data = ir3[9];
            ir3[9] = (v1608_data + (v1560_data * v1606_data));
            float v1611_data = s0[131];
            float v1613_data = ir3[10];
            ir3[10] = (v1613_data + (v1560_data * v1611_data));
            float v1616_data = s0[143];
            float v1618_data = ir3[11];
            ir3[11] = (v1618_data + (v1560_data * v1616_data));
          }
          if (v3_lead < 6) {
            #pragma unroll
            for (int32_t v1624_n1 = 0; v1624_n1 < 12; ++v1624_n1) {
              int32_t v1625_a = 0 + v1624_n1;
              float v1627_data = ir3[v1624_n1];
              int32_t v1628_a = 0 + v1624_n1;
              r3[v1624_n1] = v1627_data;
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v3_lead < 6) {
            int32_t v1643_off = v3_lead + 6;
            #pragma unroll
            for (int32_t v1634_i1 = 0; v1634_i1 < 12; ++v1634_i1) {
              int32_t v1635_a = 0 + v1634_i1;
              float v1637_data = r3[v1634_i1];
              int32_t v1645_a = v1643_off + (v1634_i1 * 12);
              s1[v1645_a] = v1637_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m4););
          float r5[12]{};
          __syncwarp();
          // r5 = +(r4 * s1) + None
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir5[12]{};
          if (v3_lead < 12) {
            float v1652_data = r4[0];
            float v1653_data = s1[0];
            float v1655_data = ir5[0];
            ir5[0] = (v1655_data + (v1652_data * v1653_data));
            float v1658_data = s1[12];
            float v1660_data = ir5[1];
            ir5[1] = (v1660_data + (v1652_data * v1658_data));
            float v1663_data = s1[24];
            float v1665_data = ir5[2];
            ir5[2] = (v1665_data + (v1652_data * v1663_data));
            float v1668_data = s1[36];
            float v1670_data = ir5[3];
            ir5[3] = (v1670_data + (v1652_data * v1668_data));
            float v1673_data = s1[48];
            float v1675_data = ir5[4];
            ir5[4] = (v1675_data + (v1652_data * v1673_data));
            float v1678_data = s1[60];
            float v1680_data = ir5[5];
            ir5[5] = (v1680_data + (v1652_data * v1678_data));
            float v1683_data = s1[72];
            float v1685_data = ir5[6];
            ir5[6] = (v1685_data + (v1652_data * v1683_data));
            float v1688_data = s1[84];
            float v1690_data = ir5[7];
            ir5[7] = (v1690_data + (v1652_data * v1688_data));
            float v1693_data = s1[96];
            float v1695_data = ir5[8];
            ir5[8] = (v1695_data + (v1652_data * v1693_data));
            float v1698_data = s1[108];
            float v1700_data = ir5[9];
            ir5[9] = (v1700_data + (v1652_data * v1698_data));
            float v1703_data = s1[120];
            float v1705_data = ir5[10];
            ir5[10] = (v1705_data + (v1652_data * v1703_data));
            float v1708_data = s1[132];
            float v1710_data = ir5[11];
            ir5[11] = (v1710_data + (v1652_data * v1708_data));
          }
          if (v3_lead < 12) {
            float v1716_data = r4[1];
            float v1717_data = s1[1];
            float v1719_data = ir5[0];
            ir5[0] = (v1719_data + (v1716_data * v1717_data));
            float v1722_data = s1[13];
            float v1724_data = ir5[1];
            ir5[1] = (v1724_data + (v1716_data * v1722_data));
            float v1727_data = s1[25];
            float v1729_data = ir5[2];
            ir5[2] = (v1729_data + (v1716_data * v1727_data));
            float v1732_data = s1[37];
            float v1734_data = ir5[3];
            ir5[3] = (v1734_data + (v1716_data * v1732_data));
            float v1737_data = s1[49];
            float v1739_data = ir5[4];
            ir5[4] = (v1739_data + (v1716_data * v1737_data));
            float v1742_data = s1[61];
            float v1744_data = ir5[5];
            ir5[5] = (v1744_data + (v1716_data * v1742_data));
            float v1747_data = s1[73];
            float v1749_data = ir5[6];
            ir5[6] = (v1749_data + (v1716_data * v1747_data));
            float v1752_data = s1[85];
            float v1754_data = ir5[7];
            ir5[7] = (v1754_data + (v1716_data * v1752_data));
            float v1757_data = s1[97];
            float v1759_data = ir5[8];
            ir5[8] = (v1759_data + (v1716_data * v1757_data));
            float v1762_data = s1[109];
            float v1764_data = ir5[9];
            ir5[9] = (v1764_data + (v1716_data * v1762_data));
            float v1767_data = s1[121];
            float v1769_data = ir5[10];
            ir5[10] = (v1769_data + (v1716_data * v1767_data));
            float v1772_data = s1[133];
            float v1774_data = ir5[11];
            ir5[11] = (v1774_data + (v1716_data * v1772_data));
          }
          if (v3_lead < 12) {
            float v1780_data = r4[2];
            float v1781_data = s1[2];
            float v1783_data = ir5[0];
            ir5[0] = (v1783_data + (v1780_data * v1781_data));
            float v1786_data = s1[14];
            float v1788_data = ir5[1];
            ir5[1] = (v1788_data + (v1780_data * v1786_data));
            float v1791_data = s1[26];
            float v1793_data = ir5[2];
            ir5[2] = (v1793_data + (v1780_data * v1791_data));
            float v1796_data = s1[38];
            float v1798_data = ir5[3];
            ir5[3] = (v1798_data + (v1780_data * v1796_data));
            float v1801_data = s1[50];
            float v1803_data = ir5[4];
            ir5[4] = (v1803_data + (v1780_data * v1801_data));
            float v1806_data = s1[62];
            float v1808_data = ir5[5];
            ir5[5] = (v1808_data + (v1780_data * v1806_data));
            float v1811_data = s1[74];
            float v1813_data = ir5[6];
            ir5[6] = (v1813_data + (v1780_data * v1811_data));
            float v1816_data = s1[86];
            float v1818_data = ir5[7];
            ir5[7] = (v1818_data + (v1780_data * v1816_data));
            float v1821_data = s1[98];
            float v1823_data = ir5[8];
            ir5[8] = (v1823_data + (v1780_data * v1821_data));
            float v1826_data = s1[110];
            float v1828_data = ir5[9];
            ir5[9] = (v1828_data + (v1780_data * v1826_data));
            float v1831_data = s1[122];
            float v1833_data = ir5[10];
            ir5[10] = (v1833_data + (v1780_data * v1831_data));
            float v1836_data = s1[134];
            float v1838_data = ir5[11];
            ir5[11] = (v1838_data + (v1780_data * v1836_data));
          }
          if (v3_lead < 12) {
            float v1844_data = r4[3];
            float v1845_data = s1[3];
            float v1847_data = ir5[0];
            ir5[0] = (v1847_data + (v1844_data * v1845_data));
            float v1850_data = s1[15];
            float v1852_data = ir5[1];
            ir5[1] = (v1852_data + (v1844_data * v1850_data));
            float v1855_data = s1[27];
            float v1857_data = ir5[2];
            ir5[2] = (v1857_data + (v1844_data * v1855_data));
            float v1860_data = s1[39];
            float v1862_data = ir5[3];
            ir5[3] = (v1862_data + (v1844_data * v1860_data));
            float v1865_data = s1[51];
            float v1867_data = ir5[4];
            ir5[4] = (v1867_data + (v1844_data * v1865_data));
            float v1870_data = s1[63];
            float v1872_data = ir5[5];
            ir5[5] = (v1872_data + (v1844_data * v1870_data));
            float v1875_data = s1[75];
            float v1877_data = ir5[6];
            ir5[6] = (v1877_data + (v1844_data * v1875_data));
            float v1880_data = s1[87];
            float v1882_data = ir5[7];
            ir5[7] = (v1882_data + (v1844_data * v1880_data));
            float v1885_data = s1[99];
            float v1887_data = ir5[8];
            ir5[8] = (v1887_data + (v1844_data * v1885_data));
            float v1890_data = s1[111];
            float v1892_data = ir5[9];
            ir5[9] = (v1892_data + (v1844_data * v1890_data));
            float v1895_data = s1[123];
            float v1897_data = ir5[10];
            ir5[10] = (v1897_data + (v1844_data * v1895_data));
            float v1900_data = s1[135];
            float v1902_data = ir5[11];
            ir5[11] = (v1902_data + (v1844_data * v1900_data));
          }
          if (v3_lead < 12) {
            float v1908_data = r4[4];
            float v1909_data = s1[4];
            float v1911_data = ir5[0];
            ir5[0] = (v1911_data + (v1908_data * v1909_data));
            float v1914_data = s1[16];
            float v1916_data = ir5[1];
            ir5[1] = (v1916_data + (v1908_data * v1914_data));
            float v1919_data = s1[28];
            float v1921_data = ir5[2];
            ir5[2] = (v1921_data + (v1908_data * v1919_data));
            float v1924_data = s1[40];
            float v1926_data = ir5[3];
            ir5[3] = (v1926_data + (v1908_data * v1924_data));
            float v1929_data = s1[52];
            float v1931_data = ir5[4];
            ir5[4] = (v1931_data + (v1908_data * v1929_data));
            float v1934_data = s1[64];
            float v1936_data = ir5[5];
            ir5[5] = (v1936_data + (v1908_data * v1934_data));
            float v1939_data = s1[76];
            float v1941_data = ir5[6];
            ir5[6] = (v1941_data + (v1908_data * v1939_data));
            float v1944_data = s1[88];
            float v1946_data = ir5[7];
            ir5[7] = (v1946_data + (v1908_data * v1944_data));
            float v1949_data = s1[100];
            float v1951_data = ir5[8];
            ir5[8] = (v1951_data + (v1908_data * v1949_data));
            float v1954_data = s1[112];
            float v1956_data = ir5[9];
            ir5[9] = (v1956_data + (v1908_data * v1954_data));
            float v1959_data = s1[124];
            float v1961_data = ir5[10];
            ir5[10] = (v1961_data + (v1908_data * v1959_data));
            float v1964_data = s1[136];
            float v1966_data = ir5[11];
            ir5[11] = (v1966_data + (v1908_data * v1964_data));
          }
          if (v3_lead < 12) {
            float v1972_data = r4[5];
            float v1973_data = s1[5];
            float v1975_data = ir5[0];
            ir5[0] = (v1975_data + (v1972_data * v1973_data));
            float v1978_data = s1[17];
            float v1980_data = ir5[1];
            ir5[1] = (v1980_data + (v1972_data * v1978_data));
            float v1983_data = s1[29];
            float v1985_data = ir5[2];
            ir5[2] = (v1985_data + (v1972_data * v1983_data));
            float v1988_data = s1[41];
            float v1990_data = ir5[3];
            ir5[3] = (v1990_data + (v1972_data * v1988_data));
            float v1993_data = s1[53];
            float v1995_data = ir5[4];
            ir5[4] = (v1995_data + (v1972_data * v1993_data));
            float v1998_data = s1[65];
            float v2000_data = ir5[5];
            ir5[5] = (v2000_data + (v1972_data * v1998_data));
            float v2003_data = s1[77];
            float v2005_data = ir5[6];
            ir5[6] = (v2005_data + (v1972_data * v2003_data));
            float v2008_data = s1[89];
            float v2010_data = ir5[7];
            ir5[7] = (v2010_data + (v1972_data * v2008_data));
            float v2013_data = s1[101];
            float v2015_data = ir5[8];
            ir5[8] = (v2015_data + (v1972_data * v2013_data));
            float v2018_data = s1[113];
            float v2020_data = ir5[9];
            ir5[9] = (v2020_data + (v1972_data * v2018_data));
            float v2023_data = s1[125];
            float v2025_data = ir5[10];
            ir5[10] = (v2025_data + (v1972_data * v2023_data));
            float v2028_data = s1[137];
            float v2030_data = ir5[11];
            ir5[11] = (v2030_data + (v1972_data * v2028_data));
          }
          if (v3_lead < 12) {
            float v2036_data = r4[6];
            float v2037_data = s1[6];
            float v2039_data = ir5[0];
            ir5[0] = (v2039_data + (v2036_data * v2037_data));
            float v2042_data = s1[18];
            float v2044_data = ir5[1];
            ir5[1] = (v2044_data + (v2036_data * v2042_data));
            float v2047_data = s1[30];
            float v2049_data = ir5[2];
            ir5[2] = (v2049_data + (v2036_data * v2047_data));
            float v2052_data = s1[42];
            float v2054_data = ir5[3];
            ir5[3] = (v2054_data + (v2036_data * v2052_data));
            float v2057_data = s1[54];
            float v2059_data = ir5[4];
            ir5[4] = (v2059_data + (v2036_data * v2057_data));
            float v2062_data = s1[66];
            float v2064_data = ir5[5];
            ir5[5] = (v2064_data + (v2036_data * v2062_data));
            float v2067_data = s1[78];
            float v2069_data = ir5[6];
            ir5[6] = (v2069_data + (v2036_data * v2067_data));
            float v2072_data = s1[90];
            float v2074_data = ir5[7];
            ir5[7] = (v2074_data + (v2036_data * v2072_data));
            float v2077_data = s1[102];
            float v2079_data = ir5[8];
            ir5[8] = (v2079_data + (v2036_data * v2077_data));
            float v2082_data = s1[114];
            float v2084_data = ir5[9];
            ir5[9] = (v2084_data + (v2036_data * v2082_data));
            float v2087_data = s1[126];
            float v2089_data = ir5[10];
            ir5[10] = (v2089_data + (v2036_data * v2087_data));
            float v2092_data = s1[138];
            float v2094_data = ir5[11];
            ir5[11] = (v2094_data + (v2036_data * v2092_data));
          }
          if (v3_lead < 12) {
            float v2100_data = r4[7];
            float v2101_data = s1[7];
            float v2103_data = ir5[0];
            ir5[0] = (v2103_data + (v2100_data * v2101_data));
            float v2106_data = s1[19];
            float v2108_data = ir5[1];
            ir5[1] = (v2108_data + (v2100_data * v2106_data));
            float v2111_data = s1[31];
            float v2113_data = ir5[2];
            ir5[2] = (v2113_data + (v2100_data * v2111_data));
            float v2116_data = s1[43];
            float v2118_data = ir5[3];
            ir5[3] = (v2118_data + (v2100_data * v2116_data));
            float v2121_data = s1[55];
            float v2123_data = ir5[4];
            ir5[4] = (v2123_data + (v2100_data * v2121_data));
            float v2126_data = s1[67];
            float v2128_data = ir5[5];
            ir5[5] = (v2128_data + (v2100_data * v2126_data));
            float v2131_data = s1[79];
            float v2133_data = ir5[6];
            ir5[6] = (v2133_data + (v2100_data * v2131_data));
            float v2136_data = s1[91];
            float v2138_data = ir5[7];
            ir5[7] = (v2138_data + (v2100_data * v2136_data));
            float v2141_data = s1[103];
            float v2143_data = ir5[8];
            ir5[8] = (v2143_data + (v2100_data * v2141_data));
            float v2146_data = s1[115];
            float v2148_data = ir5[9];
            ir5[9] = (v2148_data + (v2100_data * v2146_data));
            float v2151_data = s1[127];
            float v2153_data = ir5[10];
            ir5[10] = (v2153_data + (v2100_data * v2151_data));
            float v2156_data = s1[139];
            float v2158_data = ir5[11];
            ir5[11] = (v2158_data + (v2100_data * v2156_data));
          }
          if (v3_lead < 12) {
            float v2164_data = r4[8];
            float v2165_data = s1[8];
            float v2167_data = ir5[0];
            ir5[0] = (v2167_data + (v2164_data * v2165_data));
            float v2170_data = s1[20];
            float v2172_data = ir5[1];
            ir5[1] = (v2172_data + (v2164_data * v2170_data));
            float v2175_data = s1[32];
            float v2177_data = ir5[2];
            ir5[2] = (v2177_data + (v2164_data * v2175_data));
            float v2180_data = s1[44];
            float v2182_data = ir5[3];
            ir5[3] = (v2182_data + (v2164_data * v2180_data));
            float v2185_data = s1[56];
            float v2187_data = ir5[4];
            ir5[4] = (v2187_data + (v2164_data * v2185_data));
            float v2190_data = s1[68];
            float v2192_data = ir5[5];
            ir5[5] = (v2192_data + (v2164_data * v2190_data));
            float v2195_data = s1[80];
            float v2197_data = ir5[6];
            ir5[6] = (v2197_data + (v2164_data * v2195_data));
            float v2200_data = s1[92];
            float v2202_data = ir5[7];
            ir5[7] = (v2202_data + (v2164_data * v2200_data));
            float v2205_data = s1[104];
            float v2207_data = ir5[8];
            ir5[8] = (v2207_data + (v2164_data * v2205_data));
            float v2210_data = s1[116];
            float v2212_data = ir5[9];
            ir5[9] = (v2212_data + (v2164_data * v2210_data));
            float v2215_data = s1[128];
            float v2217_data = ir5[10];
            ir5[10] = (v2217_data + (v2164_data * v2215_data));
            float v2220_data = s1[140];
            float v2222_data = ir5[11];
            ir5[11] = (v2222_data + (v2164_data * v2220_data));
          }
          if (v3_lead < 12) {
            float v2228_data = r4[9];
            float v2229_data = s1[9];
            float v2231_data = ir5[0];
            ir5[0] = (v2231_data + (v2228_data * v2229_data));
            float v2234_data = s1[21];
            float v2236_data = ir5[1];
            ir5[1] = (v2236_data + (v2228_data * v2234_data));
            float v2239_data = s1[33];
            float v2241_data = ir5[2];
            ir5[2] = (v2241_data + (v2228_data * v2239_data));
            float v2244_data = s1[45];
            float v2246_data = ir5[3];
            ir5[3] = (v2246_data + (v2228_data * v2244_data));
            float v2249_data = s1[57];
            float v2251_data = ir5[4];
            ir5[4] = (v2251_data + (v2228_data * v2249_data));
            float v2254_data = s1[69];
            float v2256_data = ir5[5];
            ir5[5] = (v2256_data + (v2228_data * v2254_data));
            float v2259_data = s1[81];
            float v2261_data = ir5[6];
            ir5[6] = (v2261_data + (v2228_data * v2259_data));
            float v2264_data = s1[93];
            float v2266_data = ir5[7];
            ir5[7] = (v2266_data + (v2228_data * v2264_data));
            float v2269_data = s1[105];
            float v2271_data = ir5[8];
            ir5[8] = (v2271_data + (v2228_data * v2269_data));
            float v2274_data = s1[117];
            float v2276_data = ir5[9];
            ir5[9] = (v2276_data + (v2228_data * v2274_data));
            float v2279_data = s1[129];
            float v2281_data = ir5[10];
            ir5[10] = (v2281_data + (v2228_data * v2279_data));
            float v2284_data = s1[141];
            float v2286_data = ir5[11];
            ir5[11] = (v2286_data + (v2228_data * v2284_data));
          }
          if (v3_lead < 12) {
            float v2292_data = r4[10];
            float v2293_data = s1[10];
            float v2295_data = ir5[0];
            ir5[0] = (v2295_data + (v2292_data * v2293_data));
            float v2298_data = s1[22];
            float v2300_data = ir5[1];
            ir5[1] = (v2300_data + (v2292_data * v2298_data));
            float v2303_data = s1[34];
            float v2305_data = ir5[2];
            ir5[2] = (v2305_data + (v2292_data * v2303_data));
            float v2308_data = s1[46];
            float v2310_data = ir5[3];
            ir5[3] = (v2310_data + (v2292_data * v2308_data));
            float v2313_data = s1[58];
            float v2315_data = ir5[4];
            ir5[4] = (v2315_data + (v2292_data * v2313_data));
            float v2318_data = s1[70];
            float v2320_data = ir5[5];
            ir5[5] = (v2320_data + (v2292_data * v2318_data));
            float v2323_data = s1[82];
            float v2325_data = ir5[6];
            ir5[6] = (v2325_data + (v2292_data * v2323_data));
            float v2328_data = s1[94];
            float v2330_data = ir5[7];
            ir5[7] = (v2330_data + (v2292_data * v2328_data));
            float v2333_data = s1[106];
            float v2335_data = ir5[8];
            ir5[8] = (v2335_data + (v2292_data * v2333_data));
            float v2338_data = s1[118];
            float v2340_data = ir5[9];
            ir5[9] = (v2340_data + (v2292_data * v2338_data));
            float v2343_data = s1[130];
            float v2345_data = ir5[10];
            ir5[10] = (v2345_data + (v2292_data * v2343_data));
            float v2348_data = s1[142];
            float v2350_data = ir5[11];
            ir5[11] = (v2350_data + (v2292_data * v2348_data));
          }
          if (v3_lead < 12) {
            float v2356_data = r4[11];
            float v2357_data = s1[11];
            float v2359_data = ir5[0];
            ir5[0] = (v2359_data + (v2356_data * v2357_data));
            float v2362_data = s1[23];
            float v2364_data = ir5[1];
            ir5[1] = (v2364_data + (v2356_data * v2362_data));
            float v2367_data = s1[35];
            float v2369_data = ir5[2];
            ir5[2] = (v2369_data + (v2356_data * v2367_data));
            float v2372_data = s1[47];
            float v2374_data = ir5[3];
            ir5[3] = (v2374_data + (v2356_data * v2372_data));
            float v2377_data = s1[59];
            float v2379_data = ir5[4];
            ir5[4] = (v2379_data + (v2356_data * v2377_data));
            float v2382_data = s1[71];
            float v2384_data = ir5[5];
            ir5[5] = (v2384_data + (v2356_data * v2382_data));
            float v2387_data = s1[83];
            float v2389_data = ir5[6];
            ir5[6] = (v2389_data + (v2356_data * v2387_data));
            float v2392_data = s1[95];
            float v2394_data = ir5[7];
            ir5[7] = (v2394_data + (v2356_data * v2392_data));
            float v2397_data = s1[107];
            float v2399_data = ir5[8];
            ir5[8] = (v2399_data + (v2356_data * v2397_data));
            float v2402_data = s1[119];
            float v2404_data = ir5[9];
            ir5[9] = (v2404_data + (v2356_data * v2402_data));
            float v2407_data = s1[131];
            float v2409_data = ir5[10];
            ir5[10] = (v2409_data + (v2356_data * v2407_data));
            float v2412_data = s1[143];
            float v2414_data = ir5[11];
            ir5[11] = (v2414_data + (v2356_data * v2412_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v2420_n1 = 0; v2420_n1 < 12; ++v2420_n1) {
              int32_t v2421_a = 0 + v2420_n1;
              float v2423_data = ir5[v2420_n1];
              int32_t v2424_a = 0 + v2420_n1;
              r5[v2420_n1] = v2423_data;
            }
          }
          // glb_m3 = store{r>g}(r5);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v2430_i1 = 0; v2430_i1 < 12; ++v2430_i1) {
              int32_t v2431_a = 0 + v2430_i1;
              float v2433_data = r5[v2430_i1];
              int32_t v2440_a = v3_lead + (v2430_i1 * 12);
              glb_m3[v2440_a] = v2433_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

