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
      float* __restrict__ s0 = &localShrMem0[0];
      float* __restrict__ s1 = &localShrMem0[0];
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
          int32_t v16_lead = threadIdx.x % 16;
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v18_i1 = 0; v18_i1 < 6; ++v18_i1) {
              float v26_data = __ldcg(&glb_m0[(v16_lead + (v18_i1 * 12))]);
              r0[v18_i1] = v26_data;
            }
          }
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
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              float v44_data = __ldcg(&glb_m3[(v16_lead + (v36_i1 * 12))]);
              r2[v36_i1] = v44_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          if (v16_lead < 12) {
            float v51_data = r0[0];
            float v52_data = s0[0];
            float v54_data = r1[0];
            r1[0] = (v54_data + (v51_data * v52_data));
            float v57_data = s0[6];
            float v59_data = r1[1];
            r1[1] = (v59_data + (v51_data * v57_data));
            float v62_data = s0[12];
            float v64_data = r1[2];
            r1[2] = (v64_data + (v51_data * v62_data));
            float v67_data = s0[18];
            float v69_data = r1[3];
            r1[3] = (v69_data + (v51_data * v67_data));
            float v72_data = s0[24];
            float v74_data = r1[4];
            r1[4] = (v74_data + (v51_data * v72_data));
            float v77_data = s0[30];
            float v79_data = r1[5];
            r1[5] = (v79_data + (v51_data * v77_data));
          }
          if (v16_lead < 12) {
            float v85_data = r0[1];
            float v86_data = s0[1];
            float v88_data = r1[0];
            r1[0] = (v88_data + (v85_data * v86_data));
            float v91_data = s0[7];
            float v93_data = r1[1];
            r1[1] = (v93_data + (v85_data * v91_data));
            float v96_data = s0[13];
            float v98_data = r1[2];
            r1[2] = (v98_data + (v85_data * v96_data));
            float v101_data = s0[19];
            float v103_data = r1[3];
            r1[3] = (v103_data + (v85_data * v101_data));
            float v106_data = s0[25];
            float v108_data = r1[4];
            r1[4] = (v108_data + (v85_data * v106_data));
            float v111_data = s0[31];
            float v113_data = r1[5];
            r1[5] = (v113_data + (v85_data * v111_data));
          }
          if (v16_lead < 12) {
            float v119_data = r0[2];
            float v120_data = s0[2];
            float v122_data = r1[0];
            r1[0] = (v122_data + (v119_data * v120_data));
            float v125_data = s0[8];
            float v127_data = r1[1];
            r1[1] = (v127_data + (v119_data * v125_data));
            float v130_data = s0[14];
            float v132_data = r1[2];
            r1[2] = (v132_data + (v119_data * v130_data));
            float v135_data = s0[20];
            float v137_data = r1[3];
            r1[3] = (v137_data + (v119_data * v135_data));
            float v140_data = s0[26];
            float v142_data = r1[4];
            r1[4] = (v142_data + (v119_data * v140_data));
            float v145_data = s0[32];
            float v147_data = r1[5];
            r1[5] = (v147_data + (v119_data * v145_data));
          }
          if (v16_lead < 12) {
            float v153_data = r0[3];
            float v154_data = s0[3];
            float v156_data = r1[0];
            r1[0] = (v156_data + (v153_data * v154_data));
            float v159_data = s0[9];
            float v161_data = r1[1];
            r1[1] = (v161_data + (v153_data * v159_data));
            float v164_data = s0[15];
            float v166_data = r1[2];
            r1[2] = (v166_data + (v153_data * v164_data));
            float v169_data = s0[21];
            float v171_data = r1[3];
            r1[3] = (v171_data + (v153_data * v169_data));
            float v174_data = s0[27];
            float v176_data = r1[4];
            r1[4] = (v176_data + (v153_data * v174_data));
            float v179_data = s0[33];
            float v181_data = r1[5];
            r1[5] = (v181_data + (v153_data * v179_data));
          }
          if (v16_lead < 12) {
            float v187_data = r0[4];
            float v188_data = s0[4];
            float v190_data = r1[0];
            r1[0] = (v190_data + (v187_data * v188_data));
            float v193_data = s0[10];
            float v195_data = r1[1];
            r1[1] = (v195_data + (v187_data * v193_data));
            float v198_data = s0[16];
            float v200_data = r1[2];
            r1[2] = (v200_data + (v187_data * v198_data));
            float v203_data = s0[22];
            float v205_data = r1[3];
            r1[3] = (v205_data + (v187_data * v203_data));
            float v208_data = s0[28];
            float v210_data = r1[4];
            r1[4] = (v210_data + (v187_data * v208_data));
            float v213_data = s0[34];
            float v215_data = r1[5];
            r1[5] = (v215_data + (v187_data * v213_data));
          }
          if (v16_lead < 12) {
            float v221_data = r0[5];
            float v222_data = s0[5];
            float v224_data = r1[0];
            r1[0] = (v224_data + (v221_data * v222_data));
            float v227_data = s0[11];
            float v229_data = r1[1];
            r1[1] = (v229_data + (v221_data * v227_data));
            float v232_data = s0[17];
            float v234_data = r1[2];
            r1[2] = (v234_data + (v221_data * v232_data));
            float v237_data = s0[23];
            float v239_data = r1[3];
            r1[3] = (v239_data + (v221_data * v237_data));
            float v242_data = s0[29];
            float v244_data = r1[4];
            r1[4] = (v244_data + (v221_data * v242_data));
            float v247_data = s0[35];
            float v249_data = r1[5];
            r1[5] = (v249_data + (v221_data * v247_data));
          }
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v255_i1 = 0; v255_i1 < 6; ++v255_i1) {
              float v257_data = r1[v255_i1];
              s1[(v16_lead + (v255_i1 * 12))] = v257_data;
            }
          }
          float r3[6]{};
          __syncwarp();
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          if (v16_lead < 12) {
            float v271_data = r2[0];
            float v272_data = s1[0];
            float v274_data = ir3[0];
            ir3[0] = (v274_data + (v271_data * v272_data));
            float v277_data = s1[12];
            float v279_data = ir3[1];
            ir3[1] = (v279_data + (v271_data * v277_data));
            float v282_data = s1[24];
            float v284_data = ir3[2];
            ir3[2] = (v284_data + (v271_data * v282_data));
            float v287_data = s1[36];
            float v289_data = ir3[3];
            ir3[3] = (v289_data + (v271_data * v287_data));
            float v292_data = s1[48];
            float v294_data = ir3[4];
            ir3[4] = (v294_data + (v271_data * v292_data));
            float v297_data = s1[60];
            float v299_data = ir3[5];
            ir3[5] = (v299_data + (v271_data * v297_data));
          }
          if (v16_lead < 12) {
            float v305_data = r2[1];
            float v306_data = s1[1];
            float v308_data = ir3[0];
            ir3[0] = (v308_data + (v305_data * v306_data));
            float v311_data = s1[13];
            float v313_data = ir3[1];
            ir3[1] = (v313_data + (v305_data * v311_data));
            float v316_data = s1[25];
            float v318_data = ir3[2];
            ir3[2] = (v318_data + (v305_data * v316_data));
            float v321_data = s1[37];
            float v323_data = ir3[3];
            ir3[3] = (v323_data + (v305_data * v321_data));
            float v326_data = s1[49];
            float v328_data = ir3[4];
            ir3[4] = (v328_data + (v305_data * v326_data));
            float v331_data = s1[61];
            float v333_data = ir3[5];
            ir3[5] = (v333_data + (v305_data * v331_data));
          }
          if (v16_lead < 12) {
            float v339_data = r2[2];
            float v340_data = s1[2];
            float v342_data = ir3[0];
            ir3[0] = (v342_data + (v339_data * v340_data));
            float v345_data = s1[14];
            float v347_data = ir3[1];
            ir3[1] = (v347_data + (v339_data * v345_data));
            float v350_data = s1[26];
            float v352_data = ir3[2];
            ir3[2] = (v352_data + (v339_data * v350_data));
            float v355_data = s1[38];
            float v357_data = ir3[3];
            ir3[3] = (v357_data + (v339_data * v355_data));
            float v360_data = s1[50];
            float v362_data = ir3[4];
            ir3[4] = (v362_data + (v339_data * v360_data));
            float v365_data = s1[62];
            float v367_data = ir3[5];
            ir3[5] = (v367_data + (v339_data * v365_data));
          }
          if (v16_lead < 12) {
            float v373_data = r2[3];
            float v374_data = s1[3];
            float v376_data = ir3[0];
            ir3[0] = (v376_data + (v373_data * v374_data));
            float v379_data = s1[15];
            float v381_data = ir3[1];
            ir3[1] = (v381_data + (v373_data * v379_data));
            float v384_data = s1[27];
            float v386_data = ir3[2];
            ir3[2] = (v386_data + (v373_data * v384_data));
            float v389_data = s1[39];
            float v391_data = ir3[3];
            ir3[3] = (v391_data + (v373_data * v389_data));
            float v394_data = s1[51];
            float v396_data = ir3[4];
            ir3[4] = (v396_data + (v373_data * v394_data));
            float v399_data = s1[63];
            float v401_data = ir3[5];
            ir3[5] = (v401_data + (v373_data * v399_data));
          }
          if (v16_lead < 12) {
            float v407_data = r2[4];
            float v408_data = s1[4];
            float v410_data = ir3[0];
            ir3[0] = (v410_data + (v407_data * v408_data));
            float v413_data = s1[16];
            float v415_data = ir3[1];
            ir3[1] = (v415_data + (v407_data * v413_data));
            float v418_data = s1[28];
            float v420_data = ir3[2];
            ir3[2] = (v420_data + (v407_data * v418_data));
            float v423_data = s1[40];
            float v425_data = ir3[3];
            ir3[3] = (v425_data + (v407_data * v423_data));
            float v428_data = s1[52];
            float v430_data = ir3[4];
            ir3[4] = (v430_data + (v407_data * v428_data));
            float v433_data = s1[64];
            float v435_data = ir3[5];
            ir3[5] = (v435_data + (v407_data * v433_data));
          }
          if (v16_lead < 12) {
            float v441_data = r2[5];
            float v442_data = s1[5];
            float v444_data = ir3[0];
            ir3[0] = (v444_data + (v441_data * v442_data));
            float v447_data = s1[17];
            float v449_data = ir3[1];
            ir3[1] = (v449_data + (v441_data * v447_data));
            float v452_data = s1[29];
            float v454_data = ir3[2];
            ir3[2] = (v454_data + (v441_data * v452_data));
            float v457_data = s1[41];
            float v459_data = ir3[3];
            ir3[3] = (v459_data + (v441_data * v457_data));
            float v462_data = s1[53];
            float v464_data = ir3[4];
            ir3[4] = (v464_data + (v441_data * v462_data));
            float v467_data = s1[65];
            float v469_data = ir3[5];
            ir3[5] = (v469_data + (v441_data * v467_data));
          }
          if (v16_lead < 12) {
            float v475_data = r2[6];
            float v476_data = s1[6];
            float v478_data = ir3[0];
            ir3[0] = (v478_data + (v475_data * v476_data));
            float v481_data = s1[18];
            float v483_data = ir3[1];
            ir3[1] = (v483_data + (v475_data * v481_data));
            float v486_data = s1[30];
            float v488_data = ir3[2];
            ir3[2] = (v488_data + (v475_data * v486_data));
            float v491_data = s1[42];
            float v493_data = ir3[3];
            ir3[3] = (v493_data + (v475_data * v491_data));
            float v496_data = s1[54];
            float v498_data = ir3[4];
            ir3[4] = (v498_data + (v475_data * v496_data));
            float v501_data = s1[66];
            float v503_data = ir3[5];
            ir3[5] = (v503_data + (v475_data * v501_data));
          }
          if (v16_lead < 12) {
            float v509_data = r2[7];
            float v510_data = s1[7];
            float v512_data = ir3[0];
            ir3[0] = (v512_data + (v509_data * v510_data));
            float v515_data = s1[19];
            float v517_data = ir3[1];
            ir3[1] = (v517_data + (v509_data * v515_data));
            float v520_data = s1[31];
            float v522_data = ir3[2];
            ir3[2] = (v522_data + (v509_data * v520_data));
            float v525_data = s1[43];
            float v527_data = ir3[3];
            ir3[3] = (v527_data + (v509_data * v525_data));
            float v530_data = s1[55];
            float v532_data = ir3[4];
            ir3[4] = (v532_data + (v509_data * v530_data));
            float v535_data = s1[67];
            float v537_data = ir3[5];
            ir3[5] = (v537_data + (v509_data * v535_data));
          }
          if (v16_lead < 12) {
            float v543_data = r2[8];
            float v544_data = s1[8];
            float v546_data = ir3[0];
            ir3[0] = (v546_data + (v543_data * v544_data));
            float v549_data = s1[20];
            float v551_data = ir3[1];
            ir3[1] = (v551_data + (v543_data * v549_data));
            float v554_data = s1[32];
            float v556_data = ir3[2];
            ir3[2] = (v556_data + (v543_data * v554_data));
            float v559_data = s1[44];
            float v561_data = ir3[3];
            ir3[3] = (v561_data + (v543_data * v559_data));
            float v564_data = s1[56];
            float v566_data = ir3[4];
            ir3[4] = (v566_data + (v543_data * v564_data));
            float v569_data = s1[68];
            float v571_data = ir3[5];
            ir3[5] = (v571_data + (v543_data * v569_data));
          }
          if (v16_lead < 12) {
            float v577_data = r2[9];
            float v578_data = s1[9];
            float v580_data = ir3[0];
            ir3[0] = (v580_data + (v577_data * v578_data));
            float v583_data = s1[21];
            float v585_data = ir3[1];
            ir3[1] = (v585_data + (v577_data * v583_data));
            float v588_data = s1[33];
            float v590_data = ir3[2];
            ir3[2] = (v590_data + (v577_data * v588_data));
            float v593_data = s1[45];
            float v595_data = ir3[3];
            ir3[3] = (v595_data + (v577_data * v593_data));
            float v598_data = s1[57];
            float v600_data = ir3[4];
            ir3[4] = (v600_data + (v577_data * v598_data));
            float v603_data = s1[69];
            float v605_data = ir3[5];
            ir3[5] = (v605_data + (v577_data * v603_data));
          }
          if (v16_lead < 12) {
            float v611_data = r2[10];
            float v612_data = s1[10];
            float v614_data = ir3[0];
            ir3[0] = (v614_data + (v611_data * v612_data));
            float v617_data = s1[22];
            float v619_data = ir3[1];
            ir3[1] = (v619_data + (v611_data * v617_data));
            float v622_data = s1[34];
            float v624_data = ir3[2];
            ir3[2] = (v624_data + (v611_data * v622_data));
            float v627_data = s1[46];
            float v629_data = ir3[3];
            ir3[3] = (v629_data + (v611_data * v627_data));
            float v632_data = s1[58];
            float v634_data = ir3[4];
            ir3[4] = (v634_data + (v611_data * v632_data));
            float v637_data = s1[70];
            float v639_data = ir3[5];
            ir3[5] = (v639_data + (v611_data * v637_data));
          }
          if (v16_lead < 12) {
            float v645_data = r2[11];
            float v646_data = s1[11];
            float v648_data = ir3[0];
            ir3[0] = (v648_data + (v645_data * v646_data));
            float v651_data = s1[23];
            float v653_data = ir3[1];
            ir3[1] = (v653_data + (v645_data * v651_data));
            float v656_data = s1[35];
            float v658_data = ir3[2];
            ir3[2] = (v658_data + (v645_data * v656_data));
            float v661_data = s1[47];
            float v663_data = ir3[3];
            ir3[3] = (v663_data + (v645_data * v661_data));
            float v666_data = s1[59];
            float v668_data = ir3[4];
            ir3[4] = (v668_data + (v645_data * v666_data));
            float v671_data = s1[71];
            float v673_data = ir3[5];
            ir3[5] = (v673_data + (v645_data * v671_data));
          }
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v679_n1 = 0; v679_n1 < 6; ++v679_n1) {
              float v681_data = ir3[v679_n1];
              r3[v679_n1] = v681_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v16_lead < 12) {
            #pragma unroll
            for (int32_t v687_i1 = 0; v687_i1 < 6; ++v687_i1) {
              float v689_data = r3[v687_i1];
              glb_m2[(v16_lead + (v687_i1 * 12))] = v689_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

