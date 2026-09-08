// === base name ===
kernel_609dd06e89

// === header ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_609dd06e89, block.x * block.y * block.z, 512 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_609dd06e89, cudaFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_609dd06e89<<<grid,block,512 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 8×8(8×8) {0..8}×{0..8} strided
    // m1 8×8(8×8) {0..8}×{0..8} strided
    // m2 8×8(8×8) {0..8}×{0..8} strided
    // m3 8×8(8×8) {0..8}×{0..8} strided
    // m4 8×8(8×8) {0..8}×{0..8} strided
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
    // C = abs(TMP)
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float* __restrict__ s0 = &localShrMem0[0];
      float* __restrict__ s2 = &localShrMem0[0];
      float* __restrict__ s1 = &localShrMem0[0];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v18_lead = threadIdx.x % 32;
          if (v18_lead < 8) {
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
              float v28_data = __ldcg(&glb_m0[(v18_lead + (v20_i1 * 8))]);
              r0[v20_i1] = v28_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[8]{};
          // r2 = load{g>r}(glb_m2);
          if (v18_lead < 8) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 8; ++v37_i1) {
              float v45_data = __ldcg(&glb_m2[(v18_lead + (v37_i1 * 8))]);
              r2[v37_i1] = v45_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          if (v18_lead < 8) {
            float v52_data = r0[0];
            float v53_data = s0[0];
            float v55_data = r1[0];
            r1[0] = (v55_data + (v52_data * v53_data));
            float v58_data = s0[8];
            float v60_data = r1[1];
            r1[1] = (v60_data + (v52_data * v58_data));
            float v63_data = s0[16];
            float v65_data = r1[2];
            r1[2] = (v65_data + (v52_data * v63_data));
            float v68_data = s0[24];
            float v70_data = r1[3];
            r1[3] = (v70_data + (v52_data * v68_data));
            float v73_data = s0[32];
            float v75_data = r1[4];
            r1[4] = (v75_data + (v52_data * v73_data));
            float v78_data = s0[40];
            float v80_data = r1[5];
            r1[5] = (v80_data + (v52_data * v78_data));
            float v83_data = s0[48];
            float v85_data = r1[6];
            r1[6] = (v85_data + (v52_data * v83_data));
            float v88_data = s0[56];
            float v90_data = r1[7];
            r1[7] = (v90_data + (v52_data * v88_data));
          }
          if (v18_lead < 8) {
            float v96_data = r0[1];
            float v97_data = s0[1];
            float v99_data = r1[0];
            r1[0] = (v99_data + (v96_data * v97_data));
            float v102_data = s0[9];
            float v104_data = r1[1];
            r1[1] = (v104_data + (v96_data * v102_data));
            float v107_data = s0[17];
            float v109_data = r1[2];
            r1[2] = (v109_data + (v96_data * v107_data));
            float v112_data = s0[25];
            float v114_data = r1[3];
            r1[3] = (v114_data + (v96_data * v112_data));
            float v117_data = s0[33];
            float v119_data = r1[4];
            r1[4] = (v119_data + (v96_data * v117_data));
            float v122_data = s0[41];
            float v124_data = r1[5];
            r1[5] = (v124_data + (v96_data * v122_data));
            float v127_data = s0[49];
            float v129_data = r1[6];
            r1[6] = (v129_data + (v96_data * v127_data));
            float v132_data = s0[57];
            float v134_data = r1[7];
            r1[7] = (v134_data + (v96_data * v132_data));
          }
          if (v18_lead < 8) {
            float v140_data = r0[2];
            float v141_data = s0[2];
            float v143_data = r1[0];
            r1[0] = (v143_data + (v140_data * v141_data));
            float v146_data = s0[10];
            float v148_data = r1[1];
            r1[1] = (v148_data + (v140_data * v146_data));
            float v151_data = s0[18];
            float v153_data = r1[2];
            r1[2] = (v153_data + (v140_data * v151_data));
            float v156_data = s0[26];
            float v158_data = r1[3];
            r1[3] = (v158_data + (v140_data * v156_data));
            float v161_data = s0[34];
            float v163_data = r1[4];
            r1[4] = (v163_data + (v140_data * v161_data));
            float v166_data = s0[42];
            float v168_data = r1[5];
            r1[5] = (v168_data + (v140_data * v166_data));
            float v171_data = s0[50];
            float v173_data = r1[6];
            r1[6] = (v173_data + (v140_data * v171_data));
            float v176_data = s0[58];
            float v178_data = r1[7];
            r1[7] = (v178_data + (v140_data * v176_data));
          }
          if (v18_lead < 8) {
            float v184_data = r0[3];
            float v185_data = s0[3];
            float v187_data = r1[0];
            r1[0] = (v187_data + (v184_data * v185_data));
            float v190_data = s0[11];
            float v192_data = r1[1];
            r1[1] = (v192_data + (v184_data * v190_data));
            float v195_data = s0[19];
            float v197_data = r1[2];
            r1[2] = (v197_data + (v184_data * v195_data));
            float v200_data = s0[27];
            float v202_data = r1[3];
            r1[3] = (v202_data + (v184_data * v200_data));
            float v205_data = s0[35];
            float v207_data = r1[4];
            r1[4] = (v207_data + (v184_data * v205_data));
            float v210_data = s0[43];
            float v212_data = r1[5];
            r1[5] = (v212_data + (v184_data * v210_data));
            float v215_data = s0[51];
            float v217_data = r1[6];
            r1[6] = (v217_data + (v184_data * v215_data));
            float v220_data = s0[59];
            float v222_data = r1[7];
            r1[7] = (v222_data + (v184_data * v220_data));
          }
          if (v18_lead < 8) {
            float v228_data = r0[4];
            float v229_data = s0[4];
            float v231_data = r1[0];
            r1[0] = (v231_data + (v228_data * v229_data));
            float v234_data = s0[12];
            float v236_data = r1[1];
            r1[1] = (v236_data + (v228_data * v234_data));
            float v239_data = s0[20];
            float v241_data = r1[2];
            r1[2] = (v241_data + (v228_data * v239_data));
            float v244_data = s0[28];
            float v246_data = r1[3];
            r1[3] = (v246_data + (v228_data * v244_data));
            float v249_data = s0[36];
            float v251_data = r1[4];
            r1[4] = (v251_data + (v228_data * v249_data));
            float v254_data = s0[44];
            float v256_data = r1[5];
            r1[5] = (v256_data + (v228_data * v254_data));
            float v259_data = s0[52];
            float v261_data = r1[6];
            r1[6] = (v261_data + (v228_data * v259_data));
            float v264_data = s0[60];
            float v266_data = r1[7];
            r1[7] = (v266_data + (v228_data * v264_data));
          }
          if (v18_lead < 8) {
            float v272_data = r0[5];
            float v273_data = s0[5];
            float v275_data = r1[0];
            r1[0] = (v275_data + (v272_data * v273_data));
            float v278_data = s0[13];
            float v280_data = r1[1];
            r1[1] = (v280_data + (v272_data * v278_data));
            float v283_data = s0[21];
            float v285_data = r1[2];
            r1[2] = (v285_data + (v272_data * v283_data));
            float v288_data = s0[29];
            float v290_data = r1[3];
            r1[3] = (v290_data + (v272_data * v288_data));
            float v293_data = s0[37];
            float v295_data = r1[4];
            r1[4] = (v295_data + (v272_data * v293_data));
            float v298_data = s0[45];
            float v300_data = r1[5];
            r1[5] = (v300_data + (v272_data * v298_data));
            float v303_data = s0[53];
            float v305_data = r1[6];
            r1[6] = (v305_data + (v272_data * v303_data));
            float v308_data = s0[61];
            float v310_data = r1[7];
            r1[7] = (v310_data + (v272_data * v308_data));
          }
          if (v18_lead < 8) {
            float v316_data = r0[6];
            float v317_data = s0[6];
            float v319_data = r1[0];
            r1[0] = (v319_data + (v316_data * v317_data));
            float v322_data = s0[14];
            float v324_data = r1[1];
            r1[1] = (v324_data + (v316_data * v322_data));
            float v327_data = s0[22];
            float v329_data = r1[2];
            r1[2] = (v329_data + (v316_data * v327_data));
            float v332_data = s0[30];
            float v334_data = r1[3];
            r1[3] = (v334_data + (v316_data * v332_data));
            float v337_data = s0[38];
            float v339_data = r1[4];
            r1[4] = (v339_data + (v316_data * v337_data));
            float v342_data = s0[46];
            float v344_data = r1[5];
            r1[5] = (v344_data + (v316_data * v342_data));
            float v347_data = s0[54];
            float v349_data = r1[6];
            r1[6] = (v349_data + (v316_data * v347_data));
            float v352_data = s0[62];
            float v354_data = r1[7];
            r1[7] = (v354_data + (v316_data * v352_data));
          }
          if (v18_lead < 8) {
            float v360_data = r0[7];
            float v361_data = s0[7];
            float v363_data = r1[0];
            r1[0] = (v363_data + (v360_data * v361_data));
            float v366_data = s0[15];
            float v368_data = r1[1];
            r1[1] = (v368_data + (v360_data * v366_data));
            float v371_data = s0[23];
            float v373_data = r1[2];
            r1[2] = (v373_data + (v360_data * v371_data));
            float v376_data = s0[31];
            float v378_data = r1[3];
            r1[3] = (v378_data + (v360_data * v376_data));
            float v381_data = s0[39];
            float v383_data = r1[4];
            r1[4] = (v383_data + (v360_data * v381_data));
            float v386_data = s0[47];
            float v388_data = r1[5];
            r1[5] = (v388_data + (v360_data * v386_data));
            float v391_data = s0[55];
            float v393_data = r1[6];
            r1[6] = (v393_data + (v360_data * v391_data));
            float v396_data = s0[63];
            float v398_data = r1[7];
            r1[7] = (v398_data + (v360_data * v396_data));
          }
          __syncwarp();
          // s2 = load{g>s}(glb_m3[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m3[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 32], &glb_m3[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m2););
          // wait(s2 = load{g>s}(glb_m3[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir3[8]{};
          if (v18_lead < 8) {
            float v408_data = r2[0];
            float v409_data = s2[0];
            float v411_data = ir3[0];
            ir3[0] = (v411_data + (v408_data * v409_data));
            float v414_data = s2[8];
            float v416_data = ir3[1];
            ir3[1] = (v416_data + (v408_data * v414_data));
            float v419_data = s2[16];
            float v421_data = ir3[2];
            ir3[2] = (v421_data + (v408_data * v419_data));
            float v424_data = s2[24];
            float v426_data = ir3[3];
            ir3[3] = (v426_data + (v408_data * v424_data));
            float v429_data = s2[32];
            float v431_data = ir3[4];
            ir3[4] = (v431_data + (v408_data * v429_data));
            float v434_data = s2[40];
            float v436_data = ir3[5];
            ir3[5] = (v436_data + (v408_data * v434_data));
            float v439_data = s2[48];
            float v441_data = ir3[6];
            ir3[6] = (v441_data + (v408_data * v439_data));
            float v444_data = s2[56];
            float v446_data = ir3[7];
            ir3[7] = (v446_data + (v408_data * v444_data));
          }
          if (v18_lead < 8) {
            float v452_data = r2[1];
            float v453_data = s2[1];
            float v455_data = ir3[0];
            ir3[0] = (v455_data + (v452_data * v453_data));
            float v458_data = s2[9];
            float v460_data = ir3[1];
            ir3[1] = (v460_data + (v452_data * v458_data));
            float v463_data = s2[17];
            float v465_data = ir3[2];
            ir3[2] = (v465_data + (v452_data * v463_data));
            float v468_data = s2[25];
            float v470_data = ir3[3];
            ir3[3] = (v470_data + (v452_data * v468_data));
            float v473_data = s2[33];
            float v475_data = ir3[4];
            ir3[4] = (v475_data + (v452_data * v473_data));
            float v478_data = s2[41];
            float v480_data = ir3[5];
            ir3[5] = (v480_data + (v452_data * v478_data));
            float v483_data = s2[49];
            float v485_data = ir3[6];
            ir3[6] = (v485_data + (v452_data * v483_data));
            float v488_data = s2[57];
            float v490_data = ir3[7];
            ir3[7] = (v490_data + (v452_data * v488_data));
          }
          if (v18_lead < 8) {
            float v496_data = r2[2];
            float v497_data = s2[2];
            float v499_data = ir3[0];
            ir3[0] = (v499_data + (v496_data * v497_data));
            float v502_data = s2[10];
            float v504_data = ir3[1];
            ir3[1] = (v504_data + (v496_data * v502_data));
            float v507_data = s2[18];
            float v509_data = ir3[2];
            ir3[2] = (v509_data + (v496_data * v507_data));
            float v512_data = s2[26];
            float v514_data = ir3[3];
            ir3[3] = (v514_data + (v496_data * v512_data));
            float v517_data = s2[34];
            float v519_data = ir3[4];
            ir3[4] = (v519_data + (v496_data * v517_data));
            float v522_data = s2[42];
            float v524_data = ir3[5];
            ir3[5] = (v524_data + (v496_data * v522_data));
            float v527_data = s2[50];
            float v529_data = ir3[6];
            ir3[6] = (v529_data + (v496_data * v527_data));
            float v532_data = s2[58];
            float v534_data = ir3[7];
            ir3[7] = (v534_data + (v496_data * v532_data));
          }
          if (v18_lead < 8) {
            float v540_data = r2[3];
            float v541_data = s2[3];
            float v543_data = ir3[0];
            ir3[0] = (v543_data + (v540_data * v541_data));
            float v546_data = s2[11];
            float v548_data = ir3[1];
            ir3[1] = (v548_data + (v540_data * v546_data));
            float v551_data = s2[19];
            float v553_data = ir3[2];
            ir3[2] = (v553_data + (v540_data * v551_data));
            float v556_data = s2[27];
            float v558_data = ir3[3];
            ir3[3] = (v558_data + (v540_data * v556_data));
            float v561_data = s2[35];
            float v563_data = ir3[4];
            ir3[4] = (v563_data + (v540_data * v561_data));
            float v566_data = s2[43];
            float v568_data = ir3[5];
            ir3[5] = (v568_data + (v540_data * v566_data));
            float v571_data = s2[51];
            float v573_data = ir3[6];
            ir3[6] = (v573_data + (v540_data * v571_data));
            float v576_data = s2[59];
            float v578_data = ir3[7];
            ir3[7] = (v578_data + (v540_data * v576_data));
          }
          if (v18_lead < 8) {
            float v584_data = r2[4];
            float v585_data = s2[4];
            float v587_data = ir3[0];
            ir3[0] = (v587_data + (v584_data * v585_data));
            float v590_data = s2[12];
            float v592_data = ir3[1];
            ir3[1] = (v592_data + (v584_data * v590_data));
            float v595_data = s2[20];
            float v597_data = ir3[2];
            ir3[2] = (v597_data + (v584_data * v595_data));
            float v600_data = s2[28];
            float v602_data = ir3[3];
            ir3[3] = (v602_data + (v584_data * v600_data));
            float v605_data = s2[36];
            float v607_data = ir3[4];
            ir3[4] = (v607_data + (v584_data * v605_data));
            float v610_data = s2[44];
            float v612_data = ir3[5];
            ir3[5] = (v612_data + (v584_data * v610_data));
            float v615_data = s2[52];
            float v617_data = ir3[6];
            ir3[6] = (v617_data + (v584_data * v615_data));
            float v620_data = s2[60];
            float v622_data = ir3[7];
            ir3[7] = (v622_data + (v584_data * v620_data));
          }
          if (v18_lead < 8) {
            float v628_data = r2[5];
            float v629_data = s2[5];
            float v631_data = ir3[0];
            ir3[0] = (v631_data + (v628_data * v629_data));
            float v634_data = s2[13];
            float v636_data = ir3[1];
            ir3[1] = (v636_data + (v628_data * v634_data));
            float v639_data = s2[21];
            float v641_data = ir3[2];
            ir3[2] = (v641_data + (v628_data * v639_data));
            float v644_data = s2[29];
            float v646_data = ir3[3];
            ir3[3] = (v646_data + (v628_data * v644_data));
            float v649_data = s2[37];
            float v651_data = ir3[4];
            ir3[4] = (v651_data + (v628_data * v649_data));
            float v654_data = s2[45];
            float v656_data = ir3[5];
            ir3[5] = (v656_data + (v628_data * v654_data));
            float v659_data = s2[53];
            float v661_data = ir3[6];
            ir3[6] = (v661_data + (v628_data * v659_data));
            float v664_data = s2[61];
            float v666_data = ir3[7];
            ir3[7] = (v666_data + (v628_data * v664_data));
          }
          if (v18_lead < 8) {
            float v672_data = r2[6];
            float v673_data = s2[6];
            float v675_data = ir3[0];
            ir3[0] = (v675_data + (v672_data * v673_data));
            float v678_data = s2[14];
            float v680_data = ir3[1];
            ir3[1] = (v680_data + (v672_data * v678_data));
            float v683_data = s2[22];
            float v685_data = ir3[2];
            ir3[2] = (v685_data + (v672_data * v683_data));
            float v688_data = s2[30];
            float v690_data = ir3[3];
            ir3[3] = (v690_data + (v672_data * v688_data));
            float v693_data = s2[38];
            float v695_data = ir3[4];
            ir3[4] = (v695_data + (v672_data * v693_data));
            float v698_data = s2[46];
            float v700_data = ir3[5];
            ir3[5] = (v700_data + (v672_data * v698_data));
            float v703_data = s2[54];
            float v705_data = ir3[6];
            ir3[6] = (v705_data + (v672_data * v703_data));
            float v708_data = s2[62];
            float v710_data = ir3[7];
            ir3[7] = (v710_data + (v672_data * v708_data));
          }
          if (v18_lead < 8) {
            float v716_data = r2[7];
            float v717_data = s2[7];
            float v719_data = ir3[0];
            ir3[0] = (v719_data + (v716_data * v717_data));
            float v722_data = s2[15];
            float v724_data = ir3[1];
            ir3[1] = (v724_data + (v716_data * v722_data));
            float v727_data = s2[23];
            float v729_data = ir3[2];
            ir3[2] = (v729_data + (v716_data * v727_data));
            float v732_data = s2[31];
            float v734_data = ir3[3];
            ir3[3] = (v734_data + (v716_data * v732_data));
            float v737_data = s2[39];
            float v739_data = ir3[4];
            ir3[4] = (v739_data + (v716_data * v737_data));
            float v742_data = s2[47];
            float v744_data = ir3[5];
            ir3[5] = (v744_data + (v716_data * v742_data));
            float v747_data = s2[55];
            float v749_data = ir3[6];
            ir3[6] = (v749_data + (v716_data * v747_data));
            float v752_data = s2[63];
            float v754_data = ir3[7];
            ir3[7] = (v754_data + (v716_data * v752_data));
          }
          if (v18_lead < 8) {
            #pragma unroll
            for (int32_t v760_n1 = 0; v760_n1 < 8; ++v760_n1) {
              float v762_data = ir3[v760_n1];
              float v764_data = r1[v760_n1];
              r3[v760_n1] = (v764_data + v762_data);
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v18_lead < 8) {
            #pragma unroll
            for (int32_t v771_i1 = 0; v771_i1 < 8; ++v771_i1) {
              float v773_data = r3[v771_i1];
              int32_t v780_a = v18_lead + (v771_i1 * 8);
              s1[(v780_a ^ ((v780_a >> 5) & 31))] = v773_data;
            }
          }
          __syncwarp();
          // glb_m4 = abs(s1)
          if (v18_lead < 8) {
            #pragma unroll
            for (int32_t v788_k1 = 0; v788_k1 < 8; ++v788_k1) {
              int32_t v794_a = v788_k1 * 8;
              int32_t v795_a = v18_lead + v794_a;
              float v799_data = s1[(v795_a ^ ((v795_a >> 5) & 31))];
              glb_m4[(v18_lead + v794_a)] = (fabsf(v799_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

