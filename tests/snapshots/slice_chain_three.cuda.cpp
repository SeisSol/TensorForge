// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_08703cce1d, block.x * block.y * block.z, 1792 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_08703cce1d, cudaFuncAttributeMaxDynamicSharedMemorySize, 1792 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_08703cce1d<<<grid,block,1792 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×32(12×6) {0..12}×{0..6} strided
    // m1 32×32(6×6) {0..6}×{0..6} strided
    // m2 32×32(12×6) {0..12}×{0..6} strided
    // m3 32×32(12×12) {0..12}×{0..12} strided
    // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
    // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v11_lead = threadIdx.x % 16;
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v13_i1 = 0; v13_i1 < 6; ++v13_i1) {
              float v21_data = __ldcg(&glb_m0[(v11_lead + (v13_i1 * 12))]);
              r0[v13_i1] = v21_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_commit();
          if (threadIdx.x < 4) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
              float v40_data = __ldcg(&glb_m3[(v11_lead + (v32_i1 * 12))]);
              r2[v32_i1] = v40_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          if (v11_lead < 12) {
            float v47_data = r0[0];
            float v48_data = s0[0];
            float v50_data = r1[0];
            r1[0] = (v50_data + (v47_data * v48_data));
            float v53_data = s0[6];
            float v55_data = r1[1];
            r1[1] = (v55_data + (v47_data * v53_data));
            float v58_data = s0[12];
            float v60_data = r1[2];
            r1[2] = (v60_data + (v47_data * v58_data));
            float v63_data = s0[18];
            float v65_data = r1[3];
            r1[3] = (v65_data + (v47_data * v63_data));
            float v68_data = s0[24];
            float v70_data = r1[4];
            r1[4] = (v70_data + (v47_data * v68_data));
            float v73_data = s0[30];
            float v75_data = r1[5];
            r1[5] = (v75_data + (v47_data * v73_data));
          }
          if (v11_lead < 12) {
            float v81_data = r0[1];
            float v82_data = s0[1];
            float v84_data = r1[0];
            r1[0] = (v84_data + (v81_data * v82_data));
            float v87_data = s0[7];
            float v89_data = r1[1];
            r1[1] = (v89_data + (v81_data * v87_data));
            float v92_data = s0[13];
            float v94_data = r1[2];
            r1[2] = (v94_data + (v81_data * v92_data));
            float v97_data = s0[19];
            float v99_data = r1[3];
            r1[3] = (v99_data + (v81_data * v97_data));
            float v102_data = s0[25];
            float v104_data = r1[4];
            r1[4] = (v104_data + (v81_data * v102_data));
            float v107_data = s0[31];
            float v109_data = r1[5];
            r1[5] = (v109_data + (v81_data * v107_data));
          }
          if (v11_lead < 12) {
            float v115_data = r0[2];
            float v116_data = s0[2];
            float v118_data = r1[0];
            r1[0] = (v118_data + (v115_data * v116_data));
            float v121_data = s0[8];
            float v123_data = r1[1];
            r1[1] = (v123_data + (v115_data * v121_data));
            float v126_data = s0[14];
            float v128_data = r1[2];
            r1[2] = (v128_data + (v115_data * v126_data));
            float v131_data = s0[20];
            float v133_data = r1[3];
            r1[3] = (v133_data + (v115_data * v131_data));
            float v136_data = s0[26];
            float v138_data = r1[4];
            r1[4] = (v138_data + (v115_data * v136_data));
            float v141_data = s0[32];
            float v143_data = r1[5];
            r1[5] = (v143_data + (v115_data * v141_data));
          }
          if (v11_lead < 12) {
            float v149_data = r0[3];
            float v150_data = s0[3];
            float v152_data = r1[0];
            r1[0] = (v152_data + (v149_data * v150_data));
            float v155_data = s0[9];
            float v157_data = r1[1];
            r1[1] = (v157_data + (v149_data * v155_data));
            float v160_data = s0[15];
            float v162_data = r1[2];
            r1[2] = (v162_data + (v149_data * v160_data));
            float v165_data = s0[21];
            float v167_data = r1[3];
            r1[3] = (v167_data + (v149_data * v165_data));
            float v170_data = s0[27];
            float v172_data = r1[4];
            r1[4] = (v172_data + (v149_data * v170_data));
            float v175_data = s0[33];
            float v177_data = r1[5];
            r1[5] = (v177_data + (v149_data * v175_data));
          }
          if (v11_lead < 12) {
            float v183_data = r0[4];
            float v184_data = s0[4];
            float v186_data = r1[0];
            r1[0] = (v186_data + (v183_data * v184_data));
            float v189_data = s0[10];
            float v191_data = r1[1];
            r1[1] = (v191_data + (v183_data * v189_data));
            float v194_data = s0[16];
            float v196_data = r1[2];
            r1[2] = (v196_data + (v183_data * v194_data));
            float v199_data = s0[22];
            float v201_data = r1[3];
            r1[3] = (v201_data + (v183_data * v199_data));
            float v204_data = s0[28];
            float v206_data = r1[4];
            r1[4] = (v206_data + (v183_data * v204_data));
            float v209_data = s0[34];
            float v211_data = r1[5];
            r1[5] = (v211_data + (v183_data * v209_data));
          }
          if (v11_lead < 12) {
            float v217_data = r0[5];
            float v218_data = s0[5];
            float v220_data = r1[0];
            r1[0] = (v220_data + (v217_data * v218_data));
            float v223_data = s0[11];
            float v225_data = r1[1];
            r1[1] = (v225_data + (v217_data * v223_data));
            float v228_data = s0[17];
            float v230_data = r1[2];
            r1[2] = (v230_data + (v217_data * v228_data));
            float v233_data = s0[23];
            float v235_data = r1[3];
            r1[3] = (v235_data + (v217_data * v233_data));
            float v238_data = s0[29];
            float v240_data = r1[4];
            r1[4] = (v240_data + (v217_data * v238_data));
            float v243_data = s0[35];
            float v245_data = r1[5];
            r1[5] = (v245_data + (v217_data * v243_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = store{r>s}(localShrMem0, r1);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v252_i1 = 0; v252_i1 < 6; ++v252_i1) {
              float v254_data = r1[v252_i1];
              s1[(v11_lead + (v252_i1 * 12))] = v254_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v11_lead < 12) {
            float v268_data = r2[0];
            float v269_data = s1[0];
            float v271_data = ir3[0];
            ir3[0] = (v271_data + (v268_data * v269_data));
            float v274_data = s1[12];
            float v276_data = ir3[1];
            ir3[1] = (v276_data + (v268_data * v274_data));
            float v279_data = s1[24];
            float v281_data = ir3[2];
            ir3[2] = (v281_data + (v268_data * v279_data));
            float v284_data = s1[36];
            float v286_data = ir3[3];
            ir3[3] = (v286_data + (v268_data * v284_data));
            float v289_data = s1[48];
            float v291_data = ir3[4];
            ir3[4] = (v291_data + (v268_data * v289_data));
            float v294_data = s1[60];
            float v296_data = ir3[5];
            ir3[5] = (v296_data + (v268_data * v294_data));
          }
          if (v11_lead < 12) {
            float v302_data = r2[1];
            float v303_data = s1[1];
            float v305_data = ir3[0];
            ir3[0] = (v305_data + (v302_data * v303_data));
            float v308_data = s1[13];
            float v310_data = ir3[1];
            ir3[1] = (v310_data + (v302_data * v308_data));
            float v313_data = s1[25];
            float v315_data = ir3[2];
            ir3[2] = (v315_data + (v302_data * v313_data));
            float v318_data = s1[37];
            float v320_data = ir3[3];
            ir3[3] = (v320_data + (v302_data * v318_data));
            float v323_data = s1[49];
            float v325_data = ir3[4];
            ir3[4] = (v325_data + (v302_data * v323_data));
            float v328_data = s1[61];
            float v330_data = ir3[5];
            ir3[5] = (v330_data + (v302_data * v328_data));
          }
          if (v11_lead < 12) {
            float v336_data = r2[2];
            float v337_data = s1[2];
            float v339_data = ir3[0];
            ir3[0] = (v339_data + (v336_data * v337_data));
            float v342_data = s1[14];
            float v344_data = ir3[1];
            ir3[1] = (v344_data + (v336_data * v342_data));
            float v347_data = s1[26];
            float v349_data = ir3[2];
            ir3[2] = (v349_data + (v336_data * v347_data));
            float v352_data = s1[38];
            float v354_data = ir3[3];
            ir3[3] = (v354_data + (v336_data * v352_data));
            float v357_data = s1[50];
            float v359_data = ir3[4];
            ir3[4] = (v359_data + (v336_data * v357_data));
            float v362_data = s1[62];
            float v364_data = ir3[5];
            ir3[5] = (v364_data + (v336_data * v362_data));
          }
          if (v11_lead < 12) {
            float v370_data = r2[3];
            float v371_data = s1[3];
            float v373_data = ir3[0];
            ir3[0] = (v373_data + (v370_data * v371_data));
            float v376_data = s1[15];
            float v378_data = ir3[1];
            ir3[1] = (v378_data + (v370_data * v376_data));
            float v381_data = s1[27];
            float v383_data = ir3[2];
            ir3[2] = (v383_data + (v370_data * v381_data));
            float v386_data = s1[39];
            float v388_data = ir3[3];
            ir3[3] = (v388_data + (v370_data * v386_data));
            float v391_data = s1[51];
            float v393_data = ir3[4];
            ir3[4] = (v393_data + (v370_data * v391_data));
            float v396_data = s1[63];
            float v398_data = ir3[5];
            ir3[5] = (v398_data + (v370_data * v396_data));
          }
          if (v11_lead < 12) {
            float v404_data = r2[4];
            float v405_data = s1[4];
            float v407_data = ir3[0];
            ir3[0] = (v407_data + (v404_data * v405_data));
            float v410_data = s1[16];
            float v412_data = ir3[1];
            ir3[1] = (v412_data + (v404_data * v410_data));
            float v415_data = s1[28];
            float v417_data = ir3[2];
            ir3[2] = (v417_data + (v404_data * v415_data));
            float v420_data = s1[40];
            float v422_data = ir3[3];
            ir3[3] = (v422_data + (v404_data * v420_data));
            float v425_data = s1[52];
            float v427_data = ir3[4];
            ir3[4] = (v427_data + (v404_data * v425_data));
            float v430_data = s1[64];
            float v432_data = ir3[5];
            ir3[5] = (v432_data + (v404_data * v430_data));
          }
          if (v11_lead < 12) {
            float v438_data = r2[5];
            float v439_data = s1[5];
            float v441_data = ir3[0];
            ir3[0] = (v441_data + (v438_data * v439_data));
            float v444_data = s1[17];
            float v446_data = ir3[1];
            ir3[1] = (v446_data + (v438_data * v444_data));
            float v449_data = s1[29];
            float v451_data = ir3[2];
            ir3[2] = (v451_data + (v438_data * v449_data));
            float v454_data = s1[41];
            float v456_data = ir3[3];
            ir3[3] = (v456_data + (v438_data * v454_data));
            float v459_data = s1[53];
            float v461_data = ir3[4];
            ir3[4] = (v461_data + (v438_data * v459_data));
            float v464_data = s1[65];
            float v466_data = ir3[5];
            ir3[5] = (v466_data + (v438_data * v464_data));
          }
          if (v11_lead < 12) {
            float v472_data = r2[6];
            float v473_data = s1[6];
            float v475_data = ir3[0];
            ir3[0] = (v475_data + (v472_data * v473_data));
            float v478_data = s1[18];
            float v480_data = ir3[1];
            ir3[1] = (v480_data + (v472_data * v478_data));
            float v483_data = s1[30];
            float v485_data = ir3[2];
            ir3[2] = (v485_data + (v472_data * v483_data));
            float v488_data = s1[42];
            float v490_data = ir3[3];
            ir3[3] = (v490_data + (v472_data * v488_data));
            float v493_data = s1[54];
            float v495_data = ir3[4];
            ir3[4] = (v495_data + (v472_data * v493_data));
            float v498_data = s1[66];
            float v500_data = ir3[5];
            ir3[5] = (v500_data + (v472_data * v498_data));
          }
          if (v11_lead < 12) {
            float v506_data = r2[7];
            float v507_data = s1[7];
            float v509_data = ir3[0];
            ir3[0] = (v509_data + (v506_data * v507_data));
            float v512_data = s1[19];
            float v514_data = ir3[1];
            ir3[1] = (v514_data + (v506_data * v512_data));
            float v517_data = s1[31];
            float v519_data = ir3[2];
            ir3[2] = (v519_data + (v506_data * v517_data));
            float v522_data = s1[43];
            float v524_data = ir3[3];
            ir3[3] = (v524_data + (v506_data * v522_data));
            float v527_data = s1[55];
            float v529_data = ir3[4];
            ir3[4] = (v529_data + (v506_data * v527_data));
            float v532_data = s1[67];
            float v534_data = ir3[5];
            ir3[5] = (v534_data + (v506_data * v532_data));
          }
          if (v11_lead < 12) {
            float v540_data = r2[8];
            float v541_data = s1[8];
            float v543_data = ir3[0];
            ir3[0] = (v543_data + (v540_data * v541_data));
            float v546_data = s1[20];
            float v548_data = ir3[1];
            ir3[1] = (v548_data + (v540_data * v546_data));
            float v551_data = s1[32];
            float v553_data = ir3[2];
            ir3[2] = (v553_data + (v540_data * v551_data));
            float v556_data = s1[44];
            float v558_data = ir3[3];
            ir3[3] = (v558_data + (v540_data * v556_data));
            float v561_data = s1[56];
            float v563_data = ir3[4];
            ir3[4] = (v563_data + (v540_data * v561_data));
            float v566_data = s1[68];
            float v568_data = ir3[5];
            ir3[5] = (v568_data + (v540_data * v566_data));
          }
          if (v11_lead < 12) {
            float v574_data = r2[9];
            float v575_data = s1[9];
            float v577_data = ir3[0];
            ir3[0] = (v577_data + (v574_data * v575_data));
            float v580_data = s1[21];
            float v582_data = ir3[1];
            ir3[1] = (v582_data + (v574_data * v580_data));
            float v585_data = s1[33];
            float v587_data = ir3[2];
            ir3[2] = (v587_data + (v574_data * v585_data));
            float v590_data = s1[45];
            float v592_data = ir3[3];
            ir3[3] = (v592_data + (v574_data * v590_data));
            float v595_data = s1[57];
            float v597_data = ir3[4];
            ir3[4] = (v597_data + (v574_data * v595_data));
            float v600_data = s1[69];
            float v602_data = ir3[5];
            ir3[5] = (v602_data + (v574_data * v600_data));
          }
          if (v11_lead < 12) {
            float v608_data = r2[10];
            float v609_data = s1[10];
            float v611_data = ir3[0];
            ir3[0] = (v611_data + (v608_data * v609_data));
            float v614_data = s1[22];
            float v616_data = ir3[1];
            ir3[1] = (v616_data + (v608_data * v614_data));
            float v619_data = s1[34];
            float v621_data = ir3[2];
            ir3[2] = (v621_data + (v608_data * v619_data));
            float v624_data = s1[46];
            float v626_data = ir3[3];
            ir3[3] = (v626_data + (v608_data * v624_data));
            float v629_data = s1[58];
            float v631_data = ir3[4];
            ir3[4] = (v631_data + (v608_data * v629_data));
            float v634_data = s1[70];
            float v636_data = ir3[5];
            ir3[5] = (v636_data + (v608_data * v634_data));
          }
          if (v11_lead < 12) {
            float v642_data = r2[11];
            float v643_data = s1[11];
            float v645_data = ir3[0];
            ir3[0] = (v645_data + (v642_data * v643_data));
            float v648_data = s1[23];
            float v650_data = ir3[1];
            ir3[1] = (v650_data + (v642_data * v648_data));
            float v653_data = s1[35];
            float v655_data = ir3[2];
            ir3[2] = (v655_data + (v642_data * v653_data));
            float v658_data = s1[47];
            float v660_data = ir3[3];
            ir3[3] = (v660_data + (v642_data * v658_data));
            float v663_data = s1[59];
            float v665_data = ir3[4];
            ir3[4] = (v665_data + (v642_data * v663_data));
            float v668_data = s1[71];
            float v670_data = ir3[5];
            ir3[5] = (v670_data + (v642_data * v668_data));
          }
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v676_n1 = 0; v676_n1 < 6; ++v676_n1) {
              float v678_data = ir3[v676_n1];
              r3[v676_n1] = v678_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v11_lead < 12) {
            #pragma unroll
            for (int32_t v684_i1 = 0; v684_i1 < 6; ++v684_i1) {
              float v686_data = r3[v684_i1];
              glb_m2[(v11_lead + (v684_i1 * 12))] = v686_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

