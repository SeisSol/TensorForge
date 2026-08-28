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
            int32_t v18_lead = v13_i0 * 32;
            int32_t v19_lead = v12_lead + v18_lead;
            int32_t v26_lead = v12_lead + v18_lead;
            #pragma unroll
            for (int32_t v14_i1 = 10; v14_i1 < 13; ++v14_i1) {
              int32_t v20_a = v14_i1 * 32;
              int32_t v21_a = v19_lead + v20_a;
              float v29_data = __ldcg(&glb_m1[(v26_lead + v20_a)]);
              r0[(v13_i0 + (v14_i1 - 10))] = v29_data;
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
          float v40_data = r0[0];
          float v41_data = s0[114];
          float v43_data = ir1[0];
          ir1[0] = (v43_data + (v40_data * v41_data));
          float v48_data = r0[1];
          float v49_data = s0[115];
          float v51_data = ir1[0];
          ir1[0] = (v51_data + (v48_data * v49_data));
          float v56_data = r0[2];
          float v57_data = s0[116];
          float v59_data = ir1[0];
          ir1[0] = (v59_data + (v56_data * v57_data));
          #pragma unroll
          for (int32_t v64_n0 = 0; v64_n0 < 1; ++v64_n0) {
            #pragma unroll
            for (int32_t v65_n1 = 0; v65_n1 < 1; ++v65_n1) {
              int32_t v66_a = v64_n0 + v65_n1;
              int32_t v67_a = v64_n0 + v65_n1;
              float v68_data = ir1[v67_a];
              r1[v67_a] = v68_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v73_i0 = 0; v73_i0 < 1; ++v73_i0) {
            int32_t v82_lead = v12_lead + (v73_i0 * 32);
            #pragma unroll
            for (int32_t v74_i1 = 0; v74_i1 < 1; ++v74_i1) {
              int32_t v75_a = v73_i0 + v74_i1;
              float v77_data = r1[(v73_i0 + v74_i1)];
              glb_m0[(v82_lead + ((v74_i1 + 8) * 32))] = v77_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v90_i0 = 0; v90_i0 < 1; ++v90_i0) {
            int32_t v95_lead = v90_i0 * 32;
            int32_t v96_lead = v12_lead + v95_lead;
            int32_t v103_lead = v12_lead + v95_lead;
            #pragma unroll
            for (int32_t v91_i1 = 0; v91_i1 < 13; ++v91_i1) {
              int32_t v97_a = v91_i1 * 32;
              int32_t v98_a = v96_lead + v97_a;
              float v106_data = glb_m0[(v103_lead + v97_a)];
              r2[(v90_i0 + v91_i1)] = v106_data;
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
          float v116_data = r2[0];
          float v117_data = s1[0];
          float v119_data = ir3[0];
          ir3[0] = (v119_data + (v116_data * v117_data));
          float v122_data = s1[13];
          float v124_data = ir3[1];
          ir3[1] = (v124_data + (v116_data * v122_data));
          float v127_data = s1[26];
          float v129_data = ir3[2];
          ir3[2] = (v129_data + (v116_data * v127_data));
          float v132_data = s1[39];
          float v134_data = ir3[3];
          ir3[3] = (v134_data + (v116_data * v132_data));
          float v137_data = s1[52];
          float v139_data = ir3[4];
          ir3[4] = (v139_data + (v116_data * v137_data));
          float v142_data = s1[65];
          float v144_data = ir3[5];
          ir3[5] = (v144_data + (v116_data * v142_data));
          float v147_data = s1[78];
          float v149_data = ir3[6];
          ir3[6] = (v149_data + (v116_data * v147_data));
          float v152_data = s1[91];
          float v154_data = ir3[7];
          ir3[7] = (v154_data + (v116_data * v152_data));
          float v157_data = s1[104];
          float v159_data = ir3[8];
          ir3[8] = (v159_data + (v116_data * v157_data));
          float v162_data = s1[117];
          float v164_data = ir3[9];
          ir3[9] = (v164_data + (v116_data * v162_data));
          float v167_data = s1[130];
          float v169_data = ir3[10];
          ir3[10] = (v169_data + (v116_data * v167_data));
          float v172_data = s1[143];
          float v174_data = ir3[11];
          ir3[11] = (v174_data + (v116_data * v172_data));
          float v177_data = s1[156];
          float v179_data = ir3[12];
          ir3[12] = (v179_data + (v116_data * v177_data));
          float v184_data = r2[1];
          float v185_data = s1[1];
          float v187_data = ir3[0];
          ir3[0] = (v187_data + (v184_data * v185_data));
          float v190_data = s1[14];
          float v192_data = ir3[1];
          ir3[1] = (v192_data + (v184_data * v190_data));
          float v195_data = s1[27];
          float v197_data = ir3[2];
          ir3[2] = (v197_data + (v184_data * v195_data));
          float v200_data = s1[40];
          float v202_data = ir3[3];
          ir3[3] = (v202_data + (v184_data * v200_data));
          float v205_data = s1[53];
          float v207_data = ir3[4];
          ir3[4] = (v207_data + (v184_data * v205_data));
          float v210_data = s1[66];
          float v212_data = ir3[5];
          ir3[5] = (v212_data + (v184_data * v210_data));
          float v215_data = s1[79];
          float v217_data = ir3[6];
          ir3[6] = (v217_data + (v184_data * v215_data));
          float v220_data = s1[92];
          float v222_data = ir3[7];
          ir3[7] = (v222_data + (v184_data * v220_data));
          float v225_data = s1[105];
          float v227_data = ir3[8];
          ir3[8] = (v227_data + (v184_data * v225_data));
          float v230_data = s1[118];
          float v232_data = ir3[9];
          ir3[9] = (v232_data + (v184_data * v230_data));
          float v235_data = s1[131];
          float v237_data = ir3[10];
          ir3[10] = (v237_data + (v184_data * v235_data));
          float v240_data = s1[144];
          float v242_data = ir3[11];
          ir3[11] = (v242_data + (v184_data * v240_data));
          float v245_data = s1[157];
          float v247_data = ir3[12];
          ir3[12] = (v247_data + (v184_data * v245_data));
          float v252_data = r2[2];
          float v253_data = s1[2];
          float v255_data = ir3[0];
          ir3[0] = (v255_data + (v252_data * v253_data));
          float v258_data = s1[15];
          float v260_data = ir3[1];
          ir3[1] = (v260_data + (v252_data * v258_data));
          float v263_data = s1[28];
          float v265_data = ir3[2];
          ir3[2] = (v265_data + (v252_data * v263_data));
          float v268_data = s1[41];
          float v270_data = ir3[3];
          ir3[3] = (v270_data + (v252_data * v268_data));
          float v273_data = s1[54];
          float v275_data = ir3[4];
          ir3[4] = (v275_data + (v252_data * v273_data));
          float v278_data = s1[67];
          float v280_data = ir3[5];
          ir3[5] = (v280_data + (v252_data * v278_data));
          float v283_data = s1[80];
          float v285_data = ir3[6];
          ir3[6] = (v285_data + (v252_data * v283_data));
          float v288_data = s1[93];
          float v290_data = ir3[7];
          ir3[7] = (v290_data + (v252_data * v288_data));
          float v293_data = s1[106];
          float v295_data = ir3[8];
          ir3[8] = (v295_data + (v252_data * v293_data));
          float v298_data = s1[119];
          float v300_data = ir3[9];
          ir3[9] = (v300_data + (v252_data * v298_data));
          float v303_data = s1[132];
          float v305_data = ir3[10];
          ir3[10] = (v305_data + (v252_data * v303_data));
          float v308_data = s1[145];
          float v310_data = ir3[11];
          ir3[11] = (v310_data + (v252_data * v308_data));
          float v313_data = s1[158];
          float v315_data = ir3[12];
          ir3[12] = (v315_data + (v252_data * v313_data));
          float v320_data = r2[3];
          float v321_data = s1[3];
          float v323_data = ir3[0];
          ir3[0] = (v323_data + (v320_data * v321_data));
          float v326_data = s1[16];
          float v328_data = ir3[1];
          ir3[1] = (v328_data + (v320_data * v326_data));
          float v331_data = s1[29];
          float v333_data = ir3[2];
          ir3[2] = (v333_data + (v320_data * v331_data));
          float v336_data = s1[42];
          float v338_data = ir3[3];
          ir3[3] = (v338_data + (v320_data * v336_data));
          float v341_data = s1[55];
          float v343_data = ir3[4];
          ir3[4] = (v343_data + (v320_data * v341_data));
          float v346_data = s1[68];
          float v348_data = ir3[5];
          ir3[5] = (v348_data + (v320_data * v346_data));
          float v351_data = s1[81];
          float v353_data = ir3[6];
          ir3[6] = (v353_data + (v320_data * v351_data));
          float v356_data = s1[94];
          float v358_data = ir3[7];
          ir3[7] = (v358_data + (v320_data * v356_data));
          float v361_data = s1[107];
          float v363_data = ir3[8];
          ir3[8] = (v363_data + (v320_data * v361_data));
          float v366_data = s1[120];
          float v368_data = ir3[9];
          ir3[9] = (v368_data + (v320_data * v366_data));
          float v371_data = s1[133];
          float v373_data = ir3[10];
          ir3[10] = (v373_data + (v320_data * v371_data));
          float v376_data = s1[146];
          float v378_data = ir3[11];
          ir3[11] = (v378_data + (v320_data * v376_data));
          float v381_data = s1[159];
          float v383_data = ir3[12];
          ir3[12] = (v383_data + (v320_data * v381_data));
          float v388_data = r2[4];
          float v389_data = s1[4];
          float v391_data = ir3[0];
          ir3[0] = (v391_data + (v388_data * v389_data));
          float v394_data = s1[17];
          float v396_data = ir3[1];
          ir3[1] = (v396_data + (v388_data * v394_data));
          float v399_data = s1[30];
          float v401_data = ir3[2];
          ir3[2] = (v401_data + (v388_data * v399_data));
          float v404_data = s1[43];
          float v406_data = ir3[3];
          ir3[3] = (v406_data + (v388_data * v404_data));
          float v409_data = s1[56];
          float v411_data = ir3[4];
          ir3[4] = (v411_data + (v388_data * v409_data));
          float v414_data = s1[69];
          float v416_data = ir3[5];
          ir3[5] = (v416_data + (v388_data * v414_data));
          float v419_data = s1[82];
          float v421_data = ir3[6];
          ir3[6] = (v421_data + (v388_data * v419_data));
          float v424_data = s1[95];
          float v426_data = ir3[7];
          ir3[7] = (v426_data + (v388_data * v424_data));
          float v429_data = s1[108];
          float v431_data = ir3[8];
          ir3[8] = (v431_data + (v388_data * v429_data));
          float v434_data = s1[121];
          float v436_data = ir3[9];
          ir3[9] = (v436_data + (v388_data * v434_data));
          float v439_data = s1[134];
          float v441_data = ir3[10];
          ir3[10] = (v441_data + (v388_data * v439_data));
          float v444_data = s1[147];
          float v446_data = ir3[11];
          ir3[11] = (v446_data + (v388_data * v444_data));
          float v449_data = s1[160];
          float v451_data = ir3[12];
          ir3[12] = (v451_data + (v388_data * v449_data));
          float v456_data = r2[5];
          float v457_data = s1[5];
          float v459_data = ir3[0];
          ir3[0] = (v459_data + (v456_data * v457_data));
          float v462_data = s1[18];
          float v464_data = ir3[1];
          ir3[1] = (v464_data + (v456_data * v462_data));
          float v467_data = s1[31];
          float v469_data = ir3[2];
          ir3[2] = (v469_data + (v456_data * v467_data));
          float v472_data = s1[44];
          float v474_data = ir3[3];
          ir3[3] = (v474_data + (v456_data * v472_data));
          float v477_data = s1[57];
          float v479_data = ir3[4];
          ir3[4] = (v479_data + (v456_data * v477_data));
          float v482_data = s1[70];
          float v484_data = ir3[5];
          ir3[5] = (v484_data + (v456_data * v482_data));
          float v487_data = s1[83];
          float v489_data = ir3[6];
          ir3[6] = (v489_data + (v456_data * v487_data));
          float v492_data = s1[96];
          float v494_data = ir3[7];
          ir3[7] = (v494_data + (v456_data * v492_data));
          float v497_data = s1[109];
          float v499_data = ir3[8];
          ir3[8] = (v499_data + (v456_data * v497_data));
          float v502_data = s1[122];
          float v504_data = ir3[9];
          ir3[9] = (v504_data + (v456_data * v502_data));
          float v507_data = s1[135];
          float v509_data = ir3[10];
          ir3[10] = (v509_data + (v456_data * v507_data));
          float v512_data = s1[148];
          float v514_data = ir3[11];
          ir3[11] = (v514_data + (v456_data * v512_data));
          float v517_data = s1[161];
          float v519_data = ir3[12];
          ir3[12] = (v519_data + (v456_data * v517_data));
          float v524_data = r2[6];
          float v525_data = s1[6];
          float v527_data = ir3[0];
          ir3[0] = (v527_data + (v524_data * v525_data));
          float v530_data = s1[19];
          float v532_data = ir3[1];
          ir3[1] = (v532_data + (v524_data * v530_data));
          float v535_data = s1[32];
          float v537_data = ir3[2];
          ir3[2] = (v537_data + (v524_data * v535_data));
          float v540_data = s1[45];
          float v542_data = ir3[3];
          ir3[3] = (v542_data + (v524_data * v540_data));
          float v545_data = s1[58];
          float v547_data = ir3[4];
          ir3[4] = (v547_data + (v524_data * v545_data));
          float v550_data = s1[71];
          float v552_data = ir3[5];
          ir3[5] = (v552_data + (v524_data * v550_data));
          float v555_data = s1[84];
          float v557_data = ir3[6];
          ir3[6] = (v557_data + (v524_data * v555_data));
          float v560_data = s1[97];
          float v562_data = ir3[7];
          ir3[7] = (v562_data + (v524_data * v560_data));
          float v565_data = s1[110];
          float v567_data = ir3[8];
          ir3[8] = (v567_data + (v524_data * v565_data));
          float v570_data = s1[123];
          float v572_data = ir3[9];
          ir3[9] = (v572_data + (v524_data * v570_data));
          float v575_data = s1[136];
          float v577_data = ir3[10];
          ir3[10] = (v577_data + (v524_data * v575_data));
          float v580_data = s1[149];
          float v582_data = ir3[11];
          ir3[11] = (v582_data + (v524_data * v580_data));
          float v585_data = s1[162];
          float v587_data = ir3[12];
          ir3[12] = (v587_data + (v524_data * v585_data));
          float v592_data = r2[7];
          float v593_data = s1[7];
          float v595_data = ir3[0];
          ir3[0] = (v595_data + (v592_data * v593_data));
          float v598_data = s1[20];
          float v600_data = ir3[1];
          ir3[1] = (v600_data + (v592_data * v598_data));
          float v603_data = s1[33];
          float v605_data = ir3[2];
          ir3[2] = (v605_data + (v592_data * v603_data));
          float v608_data = s1[46];
          float v610_data = ir3[3];
          ir3[3] = (v610_data + (v592_data * v608_data));
          float v613_data = s1[59];
          float v615_data = ir3[4];
          ir3[4] = (v615_data + (v592_data * v613_data));
          float v618_data = s1[72];
          float v620_data = ir3[5];
          ir3[5] = (v620_data + (v592_data * v618_data));
          float v623_data = s1[85];
          float v625_data = ir3[6];
          ir3[6] = (v625_data + (v592_data * v623_data));
          float v628_data = s1[98];
          float v630_data = ir3[7];
          ir3[7] = (v630_data + (v592_data * v628_data));
          float v633_data = s1[111];
          float v635_data = ir3[8];
          ir3[8] = (v635_data + (v592_data * v633_data));
          float v638_data = s1[124];
          float v640_data = ir3[9];
          ir3[9] = (v640_data + (v592_data * v638_data));
          float v643_data = s1[137];
          float v645_data = ir3[10];
          ir3[10] = (v645_data + (v592_data * v643_data));
          float v648_data = s1[150];
          float v650_data = ir3[11];
          ir3[11] = (v650_data + (v592_data * v648_data));
          float v653_data = s1[163];
          float v655_data = ir3[12];
          ir3[12] = (v655_data + (v592_data * v653_data));
          float v660_data = r2[8];
          float v661_data = s1[8];
          float v663_data = ir3[0];
          ir3[0] = (v663_data + (v660_data * v661_data));
          float v666_data = s1[21];
          float v668_data = ir3[1];
          ir3[1] = (v668_data + (v660_data * v666_data));
          float v671_data = s1[34];
          float v673_data = ir3[2];
          ir3[2] = (v673_data + (v660_data * v671_data));
          float v676_data = s1[47];
          float v678_data = ir3[3];
          ir3[3] = (v678_data + (v660_data * v676_data));
          float v681_data = s1[60];
          float v683_data = ir3[4];
          ir3[4] = (v683_data + (v660_data * v681_data));
          float v686_data = s1[73];
          float v688_data = ir3[5];
          ir3[5] = (v688_data + (v660_data * v686_data));
          float v691_data = s1[86];
          float v693_data = ir3[6];
          ir3[6] = (v693_data + (v660_data * v691_data));
          float v696_data = s1[99];
          float v698_data = ir3[7];
          ir3[7] = (v698_data + (v660_data * v696_data));
          float v701_data = s1[112];
          float v703_data = ir3[8];
          ir3[8] = (v703_data + (v660_data * v701_data));
          float v706_data = s1[125];
          float v708_data = ir3[9];
          ir3[9] = (v708_data + (v660_data * v706_data));
          float v711_data = s1[138];
          float v713_data = ir3[10];
          ir3[10] = (v713_data + (v660_data * v711_data));
          float v716_data = s1[151];
          float v718_data = ir3[11];
          ir3[11] = (v718_data + (v660_data * v716_data));
          float v721_data = s1[164];
          float v723_data = ir3[12];
          ir3[12] = (v723_data + (v660_data * v721_data));
          float v728_data = r2[9];
          float v729_data = s1[9];
          float v731_data = ir3[0];
          ir3[0] = (v731_data + (v728_data * v729_data));
          float v734_data = s1[22];
          float v736_data = ir3[1];
          ir3[1] = (v736_data + (v728_data * v734_data));
          float v739_data = s1[35];
          float v741_data = ir3[2];
          ir3[2] = (v741_data + (v728_data * v739_data));
          float v744_data = s1[48];
          float v746_data = ir3[3];
          ir3[3] = (v746_data + (v728_data * v744_data));
          float v749_data = s1[61];
          float v751_data = ir3[4];
          ir3[4] = (v751_data + (v728_data * v749_data));
          float v754_data = s1[74];
          float v756_data = ir3[5];
          ir3[5] = (v756_data + (v728_data * v754_data));
          float v759_data = s1[87];
          float v761_data = ir3[6];
          ir3[6] = (v761_data + (v728_data * v759_data));
          float v764_data = s1[100];
          float v766_data = ir3[7];
          ir3[7] = (v766_data + (v728_data * v764_data));
          float v769_data = s1[113];
          float v771_data = ir3[8];
          ir3[8] = (v771_data + (v728_data * v769_data));
          float v774_data = s1[126];
          float v776_data = ir3[9];
          ir3[9] = (v776_data + (v728_data * v774_data));
          float v779_data = s1[139];
          float v781_data = ir3[10];
          ir3[10] = (v781_data + (v728_data * v779_data));
          float v784_data = s1[152];
          float v786_data = ir3[11];
          ir3[11] = (v786_data + (v728_data * v784_data));
          float v789_data = s1[165];
          float v791_data = ir3[12];
          ir3[12] = (v791_data + (v728_data * v789_data));
          float v796_data = r2[10];
          float v797_data = s1[10];
          float v799_data = ir3[0];
          ir3[0] = (v799_data + (v796_data * v797_data));
          float v802_data = s1[23];
          float v804_data = ir3[1];
          ir3[1] = (v804_data + (v796_data * v802_data));
          float v807_data = s1[36];
          float v809_data = ir3[2];
          ir3[2] = (v809_data + (v796_data * v807_data));
          float v812_data = s1[49];
          float v814_data = ir3[3];
          ir3[3] = (v814_data + (v796_data * v812_data));
          float v817_data = s1[62];
          float v819_data = ir3[4];
          ir3[4] = (v819_data + (v796_data * v817_data));
          float v822_data = s1[75];
          float v824_data = ir3[5];
          ir3[5] = (v824_data + (v796_data * v822_data));
          float v827_data = s1[88];
          float v829_data = ir3[6];
          ir3[6] = (v829_data + (v796_data * v827_data));
          float v832_data = s1[101];
          float v834_data = ir3[7];
          ir3[7] = (v834_data + (v796_data * v832_data));
          float v837_data = s1[114];
          float v839_data = ir3[8];
          ir3[8] = (v839_data + (v796_data * v837_data));
          float v842_data = s1[127];
          float v844_data = ir3[9];
          ir3[9] = (v844_data + (v796_data * v842_data));
          float v847_data = s1[140];
          float v849_data = ir3[10];
          ir3[10] = (v849_data + (v796_data * v847_data));
          float v852_data = s1[153];
          float v854_data = ir3[11];
          ir3[11] = (v854_data + (v796_data * v852_data));
          float v857_data = s1[166];
          float v859_data = ir3[12];
          ir3[12] = (v859_data + (v796_data * v857_data));
          float v864_data = r2[11];
          float v865_data = s1[11];
          float v867_data = ir3[0];
          ir3[0] = (v867_data + (v864_data * v865_data));
          float v870_data = s1[24];
          float v872_data = ir3[1];
          ir3[1] = (v872_data + (v864_data * v870_data));
          float v875_data = s1[37];
          float v877_data = ir3[2];
          ir3[2] = (v877_data + (v864_data * v875_data));
          float v880_data = s1[50];
          float v882_data = ir3[3];
          ir3[3] = (v882_data + (v864_data * v880_data));
          float v885_data = s1[63];
          float v887_data = ir3[4];
          ir3[4] = (v887_data + (v864_data * v885_data));
          float v890_data = s1[76];
          float v892_data = ir3[5];
          ir3[5] = (v892_data + (v864_data * v890_data));
          float v895_data = s1[89];
          float v897_data = ir3[6];
          ir3[6] = (v897_data + (v864_data * v895_data));
          float v900_data = s1[102];
          float v902_data = ir3[7];
          ir3[7] = (v902_data + (v864_data * v900_data));
          float v905_data = s1[115];
          float v907_data = ir3[8];
          ir3[8] = (v907_data + (v864_data * v905_data));
          float v910_data = s1[128];
          float v912_data = ir3[9];
          ir3[9] = (v912_data + (v864_data * v910_data));
          float v915_data = s1[141];
          float v917_data = ir3[10];
          ir3[10] = (v917_data + (v864_data * v915_data));
          float v920_data = s1[154];
          float v922_data = ir3[11];
          ir3[11] = (v922_data + (v864_data * v920_data));
          float v925_data = s1[167];
          float v927_data = ir3[12];
          ir3[12] = (v927_data + (v864_data * v925_data));
          float v932_data = r2[12];
          float v933_data = s1[12];
          float v935_data = ir3[0];
          ir3[0] = (v935_data + (v932_data * v933_data));
          float v938_data = s1[25];
          float v940_data = ir3[1];
          ir3[1] = (v940_data + (v932_data * v938_data));
          float v943_data = s1[38];
          float v945_data = ir3[2];
          ir3[2] = (v945_data + (v932_data * v943_data));
          float v948_data = s1[51];
          float v950_data = ir3[3];
          ir3[3] = (v950_data + (v932_data * v948_data));
          float v953_data = s1[64];
          float v955_data = ir3[4];
          ir3[4] = (v955_data + (v932_data * v953_data));
          float v958_data = s1[77];
          float v960_data = ir3[5];
          ir3[5] = (v960_data + (v932_data * v958_data));
          float v963_data = s1[90];
          float v965_data = ir3[6];
          ir3[6] = (v965_data + (v932_data * v963_data));
          float v968_data = s1[103];
          float v970_data = ir3[7];
          ir3[7] = (v970_data + (v932_data * v968_data));
          float v973_data = s1[116];
          float v975_data = ir3[8];
          ir3[8] = (v975_data + (v932_data * v973_data));
          float v978_data = s1[129];
          float v980_data = ir3[9];
          ir3[9] = (v980_data + (v932_data * v978_data));
          float v983_data = s1[142];
          float v985_data = ir3[10];
          ir3[10] = (v985_data + (v932_data * v983_data));
          float v988_data = s1[155];
          float v990_data = ir3[11];
          ir3[11] = (v990_data + (v932_data * v988_data));
          float v993_data = s1[168];
          float v995_data = ir3[12];
          ir3[12] = (v995_data + (v932_data * v993_data));
          #pragma unroll
          for (int32_t v1000_n0 = 0; v1000_n0 < 1; ++v1000_n0) {
            #pragma unroll
            for (int32_t v1001_n1 = 0; v1001_n1 < 13; ++v1001_n1) {
              int32_t v1002_a = v1000_n0 + v1001_n1;
              int32_t v1003_a = v1000_n0 + v1001_n1;
              float v1004_data = ir3[v1003_a];
              r3[v1003_a] = v1004_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v1009_i0 = 0; v1009_i0 < 1; ++v1009_i0) {
            int32_t v1018_lead = v12_lead + (v1009_i0 * 32);
            #pragma unroll
            for (int32_t v1010_i1 = 0; v1010_i1 < 13; ++v1010_i1) {
              int32_t v1011_a = v1009_i0 + v1010_i1;
              float v1013_data = r3[(v1009_i0 + v1010_i1)];
              glb_m3[(v1018_lead + (v1010_i1 * 32))] = v1013_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

