// === base name ===
kernel_f61651fe59

// === header ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f61651fe59, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_f61651fe59, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f61651fe59<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 32×16(12×16) {4..16}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(12×16) {4..16}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            int32_t v11_a = (v2_lead + 4) - 4;
            int32_t v20_a = (v2_lead + 4) - 4;
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v12_a = v4_i1 * 12;
              int32_t v13_a = v11_a + v12_a;
              float v23_data = __ldcg(&glb_m1[(v20_a + v12_a)]);
              int32_t v24_a = 0 + v4_i1;
              r0[v24_a] = v23_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<4>(4), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          float r1[8]{};
          __syncwarp();
          {
            // r1 = +(r0 * s0) + None
            // [(0, 12), (0, 8)] [(0, 16)]
            float ir1[8]{};
            if (v2_lead < 12) {
              float v29_data = r0[0];
              float v30_data = s0[0];
              float v32_data = ir1[0];
              ir1[0] = (v32_data + (v29_data * v30_data));
              float v35_data = s0[16];
              float v37_data = ir1[1];
              ir1[1] = (v37_data + (v29_data * v35_data));
              float v40_data = s0[32];
              float v42_data = ir1[2];
              ir1[2] = (v42_data + (v29_data * v40_data));
              float v45_data = s0[48];
              float v47_data = ir1[3];
              ir1[3] = (v47_data + (v29_data * v45_data));
              float v50_data = s0[64];
              float v52_data = ir1[4];
              ir1[4] = (v52_data + (v29_data * v50_data));
              float v55_data = s0[80];
              float v57_data = ir1[5];
              ir1[5] = (v57_data + (v29_data * v55_data));
              float v60_data = s0[96];
              float v62_data = ir1[6];
              ir1[6] = (v62_data + (v29_data * v60_data));
              float v65_data = s0[112];
              float v67_data = ir1[7];
              ir1[7] = (v67_data + (v29_data * v65_data));
            }
            if (v2_lead < 12) {
              float v73_data = r0[1];
              float v74_data = s0[1];
              float v76_data = ir1[0];
              ir1[0] = (v76_data + (v73_data * v74_data));
              float v79_data = s0[17];
              float v81_data = ir1[1];
              ir1[1] = (v81_data + (v73_data * v79_data));
              float v84_data = s0[33];
              float v86_data = ir1[2];
              ir1[2] = (v86_data + (v73_data * v84_data));
              float v89_data = s0[49];
              float v91_data = ir1[3];
              ir1[3] = (v91_data + (v73_data * v89_data));
              float v94_data = s0[65];
              float v96_data = ir1[4];
              ir1[4] = (v96_data + (v73_data * v94_data));
              float v99_data = s0[81];
              float v101_data = ir1[5];
              ir1[5] = (v101_data + (v73_data * v99_data));
              float v104_data = s0[97];
              float v106_data = ir1[6];
              ir1[6] = (v106_data + (v73_data * v104_data));
              float v109_data = s0[113];
              float v111_data = ir1[7];
              ir1[7] = (v111_data + (v73_data * v109_data));
            }
            if (v2_lead < 12) {
              float v117_data = r0[2];
              float v118_data = s0[2];
              float v120_data = ir1[0];
              ir1[0] = (v120_data + (v117_data * v118_data));
              float v123_data = s0[18];
              float v125_data = ir1[1];
              ir1[1] = (v125_data + (v117_data * v123_data));
              float v128_data = s0[34];
              float v130_data = ir1[2];
              ir1[2] = (v130_data + (v117_data * v128_data));
              float v133_data = s0[50];
              float v135_data = ir1[3];
              ir1[3] = (v135_data + (v117_data * v133_data));
              float v138_data = s0[66];
              float v140_data = ir1[4];
              ir1[4] = (v140_data + (v117_data * v138_data));
              float v143_data = s0[82];
              float v145_data = ir1[5];
              ir1[5] = (v145_data + (v117_data * v143_data));
              float v148_data = s0[98];
              float v150_data = ir1[6];
              ir1[6] = (v150_data + (v117_data * v148_data));
              float v153_data = s0[114];
              float v155_data = ir1[7];
              ir1[7] = (v155_data + (v117_data * v153_data));
            }
            if (v2_lead < 12) {
              float v161_data = r0[3];
              float v162_data = s0[3];
              float v164_data = ir1[0];
              ir1[0] = (v164_data + (v161_data * v162_data));
              float v167_data = s0[19];
              float v169_data = ir1[1];
              ir1[1] = (v169_data + (v161_data * v167_data));
              float v172_data = s0[35];
              float v174_data = ir1[2];
              ir1[2] = (v174_data + (v161_data * v172_data));
              float v177_data = s0[51];
              float v179_data = ir1[3];
              ir1[3] = (v179_data + (v161_data * v177_data));
              float v182_data = s0[67];
              float v184_data = ir1[4];
              ir1[4] = (v184_data + (v161_data * v182_data));
              float v187_data = s0[83];
              float v189_data = ir1[5];
              ir1[5] = (v189_data + (v161_data * v187_data));
              float v192_data = s0[99];
              float v194_data = ir1[6];
              ir1[6] = (v194_data + (v161_data * v192_data));
              float v197_data = s0[115];
              float v199_data = ir1[7];
              ir1[7] = (v199_data + (v161_data * v197_data));
            }
            if (v2_lead < 12) {
              float v205_data = r0[4];
              float v206_data = s0[4];
              float v208_data = ir1[0];
              ir1[0] = (v208_data + (v205_data * v206_data));
              float v211_data = s0[20];
              float v213_data = ir1[1];
              ir1[1] = (v213_data + (v205_data * v211_data));
              float v216_data = s0[36];
              float v218_data = ir1[2];
              ir1[2] = (v218_data + (v205_data * v216_data));
              float v221_data = s0[52];
              float v223_data = ir1[3];
              ir1[3] = (v223_data + (v205_data * v221_data));
              float v226_data = s0[68];
              float v228_data = ir1[4];
              ir1[4] = (v228_data + (v205_data * v226_data));
              float v231_data = s0[84];
              float v233_data = ir1[5];
              ir1[5] = (v233_data + (v205_data * v231_data));
              float v236_data = s0[100];
              float v238_data = ir1[6];
              ir1[6] = (v238_data + (v205_data * v236_data));
              float v241_data = s0[116];
              float v243_data = ir1[7];
              ir1[7] = (v243_data + (v205_data * v241_data));
            }
            if (v2_lead < 12) {
              float v249_data = r0[5];
              float v250_data = s0[5];
              float v252_data = ir1[0];
              ir1[0] = (v252_data + (v249_data * v250_data));
              float v255_data = s0[21];
              float v257_data = ir1[1];
              ir1[1] = (v257_data + (v249_data * v255_data));
              float v260_data = s0[37];
              float v262_data = ir1[2];
              ir1[2] = (v262_data + (v249_data * v260_data));
              float v265_data = s0[53];
              float v267_data = ir1[3];
              ir1[3] = (v267_data + (v249_data * v265_data));
              float v270_data = s0[69];
              float v272_data = ir1[4];
              ir1[4] = (v272_data + (v249_data * v270_data));
              float v275_data = s0[85];
              float v277_data = ir1[5];
              ir1[5] = (v277_data + (v249_data * v275_data));
              float v280_data = s0[101];
              float v282_data = ir1[6];
              ir1[6] = (v282_data + (v249_data * v280_data));
              float v285_data = s0[117];
              float v287_data = ir1[7];
              ir1[7] = (v287_data + (v249_data * v285_data));
            }
            if (v2_lead < 12) {
              float v293_data = r0[6];
              float v294_data = s0[6];
              float v296_data = ir1[0];
              ir1[0] = (v296_data + (v293_data * v294_data));
              float v299_data = s0[22];
              float v301_data = ir1[1];
              ir1[1] = (v301_data + (v293_data * v299_data));
              float v304_data = s0[38];
              float v306_data = ir1[2];
              ir1[2] = (v306_data + (v293_data * v304_data));
              float v309_data = s0[54];
              float v311_data = ir1[3];
              ir1[3] = (v311_data + (v293_data * v309_data));
              float v314_data = s0[70];
              float v316_data = ir1[4];
              ir1[4] = (v316_data + (v293_data * v314_data));
              float v319_data = s0[86];
              float v321_data = ir1[5];
              ir1[5] = (v321_data + (v293_data * v319_data));
              float v324_data = s0[102];
              float v326_data = ir1[6];
              ir1[6] = (v326_data + (v293_data * v324_data));
              float v329_data = s0[118];
              float v331_data = ir1[7];
              ir1[7] = (v331_data + (v293_data * v329_data));
            }
            if (v2_lead < 12) {
              float v337_data = r0[7];
              float v338_data = s0[7];
              float v340_data = ir1[0];
              ir1[0] = (v340_data + (v337_data * v338_data));
              float v343_data = s0[23];
              float v345_data = ir1[1];
              ir1[1] = (v345_data + (v337_data * v343_data));
              float v348_data = s0[39];
              float v350_data = ir1[2];
              ir1[2] = (v350_data + (v337_data * v348_data));
              float v353_data = s0[55];
              float v355_data = ir1[3];
              ir1[3] = (v355_data + (v337_data * v353_data));
              float v358_data = s0[71];
              float v360_data = ir1[4];
              ir1[4] = (v360_data + (v337_data * v358_data));
              float v363_data = s0[87];
              float v365_data = ir1[5];
              ir1[5] = (v365_data + (v337_data * v363_data));
              float v368_data = s0[103];
              float v370_data = ir1[6];
              ir1[6] = (v370_data + (v337_data * v368_data));
              float v373_data = s0[119];
              float v375_data = ir1[7];
              ir1[7] = (v375_data + (v337_data * v373_data));
            }
            if (v2_lead < 12) {
              float v381_data = r0[8];
              float v382_data = s0[8];
              float v384_data = ir1[0];
              ir1[0] = (v384_data + (v381_data * v382_data));
              float v387_data = s0[24];
              float v389_data = ir1[1];
              ir1[1] = (v389_data + (v381_data * v387_data));
              float v392_data = s0[40];
              float v394_data = ir1[2];
              ir1[2] = (v394_data + (v381_data * v392_data));
              float v397_data = s0[56];
              float v399_data = ir1[3];
              ir1[3] = (v399_data + (v381_data * v397_data));
              float v402_data = s0[72];
              float v404_data = ir1[4];
              ir1[4] = (v404_data + (v381_data * v402_data));
              float v407_data = s0[88];
              float v409_data = ir1[5];
              ir1[5] = (v409_data + (v381_data * v407_data));
              float v412_data = s0[104];
              float v414_data = ir1[6];
              ir1[6] = (v414_data + (v381_data * v412_data));
              float v417_data = s0[120];
              float v419_data = ir1[7];
              ir1[7] = (v419_data + (v381_data * v417_data));
            }
            if (v2_lead < 12) {
              float v425_data = r0[9];
              float v426_data = s0[9];
              float v428_data = ir1[0];
              ir1[0] = (v428_data + (v425_data * v426_data));
              float v431_data = s0[25];
              float v433_data = ir1[1];
              ir1[1] = (v433_data + (v425_data * v431_data));
              float v436_data = s0[41];
              float v438_data = ir1[2];
              ir1[2] = (v438_data + (v425_data * v436_data));
              float v441_data = s0[57];
              float v443_data = ir1[3];
              ir1[3] = (v443_data + (v425_data * v441_data));
              float v446_data = s0[73];
              float v448_data = ir1[4];
              ir1[4] = (v448_data + (v425_data * v446_data));
              float v451_data = s0[89];
              float v453_data = ir1[5];
              ir1[5] = (v453_data + (v425_data * v451_data));
              float v456_data = s0[105];
              float v458_data = ir1[6];
              ir1[6] = (v458_data + (v425_data * v456_data));
              float v461_data = s0[121];
              float v463_data = ir1[7];
              ir1[7] = (v463_data + (v425_data * v461_data));
            }
            if (v2_lead < 12) {
              float v469_data = r0[10];
              float v470_data = s0[10];
              float v472_data = ir1[0];
              ir1[0] = (v472_data + (v469_data * v470_data));
              float v475_data = s0[26];
              float v477_data = ir1[1];
              ir1[1] = (v477_data + (v469_data * v475_data));
              float v480_data = s0[42];
              float v482_data = ir1[2];
              ir1[2] = (v482_data + (v469_data * v480_data));
              float v485_data = s0[58];
              float v487_data = ir1[3];
              ir1[3] = (v487_data + (v469_data * v485_data));
              float v490_data = s0[74];
              float v492_data = ir1[4];
              ir1[4] = (v492_data + (v469_data * v490_data));
              float v495_data = s0[90];
              float v497_data = ir1[5];
              ir1[5] = (v497_data + (v469_data * v495_data));
              float v500_data = s0[106];
              float v502_data = ir1[6];
              ir1[6] = (v502_data + (v469_data * v500_data));
              float v505_data = s0[122];
              float v507_data = ir1[7];
              ir1[7] = (v507_data + (v469_data * v505_data));
            }
            if (v2_lead < 12) {
              float v513_data = r0[11];
              float v514_data = s0[11];
              float v516_data = ir1[0];
              ir1[0] = (v516_data + (v513_data * v514_data));
              float v519_data = s0[27];
              float v521_data = ir1[1];
              ir1[1] = (v521_data + (v513_data * v519_data));
              float v524_data = s0[43];
              float v526_data = ir1[2];
              ir1[2] = (v526_data + (v513_data * v524_data));
              float v529_data = s0[59];
              float v531_data = ir1[3];
              ir1[3] = (v531_data + (v513_data * v529_data));
              float v534_data = s0[75];
              float v536_data = ir1[4];
              ir1[4] = (v536_data + (v513_data * v534_data));
              float v539_data = s0[91];
              float v541_data = ir1[5];
              ir1[5] = (v541_data + (v513_data * v539_data));
              float v544_data = s0[107];
              float v546_data = ir1[6];
              ir1[6] = (v546_data + (v513_data * v544_data));
              float v549_data = s0[123];
              float v551_data = ir1[7];
              ir1[7] = (v551_data + (v513_data * v549_data));
            }
            if (v2_lead < 12) {
              float v557_data = r0[12];
              float v558_data = s0[12];
              float v560_data = ir1[0];
              ir1[0] = (v560_data + (v557_data * v558_data));
              float v563_data = s0[28];
              float v565_data = ir1[1];
              ir1[1] = (v565_data + (v557_data * v563_data));
              float v568_data = s0[44];
              float v570_data = ir1[2];
              ir1[2] = (v570_data + (v557_data * v568_data));
              float v573_data = s0[60];
              float v575_data = ir1[3];
              ir1[3] = (v575_data + (v557_data * v573_data));
              float v578_data = s0[76];
              float v580_data = ir1[4];
              ir1[4] = (v580_data + (v557_data * v578_data));
              float v583_data = s0[92];
              float v585_data = ir1[5];
              ir1[5] = (v585_data + (v557_data * v583_data));
              float v588_data = s0[108];
              float v590_data = ir1[6];
              ir1[6] = (v590_data + (v557_data * v588_data));
              float v593_data = s0[124];
              float v595_data = ir1[7];
              ir1[7] = (v595_data + (v557_data * v593_data));
            }
            if (v2_lead < 12) {
              float v601_data = r0[13];
              float v602_data = s0[13];
              float v604_data = ir1[0];
              ir1[0] = (v604_data + (v601_data * v602_data));
              float v607_data = s0[29];
              float v609_data = ir1[1];
              ir1[1] = (v609_data + (v601_data * v607_data));
              float v612_data = s0[45];
              float v614_data = ir1[2];
              ir1[2] = (v614_data + (v601_data * v612_data));
              float v617_data = s0[61];
              float v619_data = ir1[3];
              ir1[3] = (v619_data + (v601_data * v617_data));
              float v622_data = s0[77];
              float v624_data = ir1[4];
              ir1[4] = (v624_data + (v601_data * v622_data));
              float v627_data = s0[93];
              float v629_data = ir1[5];
              ir1[5] = (v629_data + (v601_data * v627_data));
              float v632_data = s0[109];
              float v634_data = ir1[6];
              ir1[6] = (v634_data + (v601_data * v632_data));
              float v637_data = s0[125];
              float v639_data = ir1[7];
              ir1[7] = (v639_data + (v601_data * v637_data));
            }
            if (v2_lead < 12) {
              float v645_data = r0[14];
              float v646_data = s0[14];
              float v648_data = ir1[0];
              ir1[0] = (v648_data + (v645_data * v646_data));
              float v651_data = s0[30];
              float v653_data = ir1[1];
              ir1[1] = (v653_data + (v645_data * v651_data));
              float v656_data = s0[46];
              float v658_data = ir1[2];
              ir1[2] = (v658_data + (v645_data * v656_data));
              float v661_data = s0[62];
              float v663_data = ir1[3];
              ir1[3] = (v663_data + (v645_data * v661_data));
              float v666_data = s0[78];
              float v668_data = ir1[4];
              ir1[4] = (v668_data + (v645_data * v666_data));
              float v671_data = s0[94];
              float v673_data = ir1[5];
              ir1[5] = (v673_data + (v645_data * v671_data));
              float v676_data = s0[110];
              float v678_data = ir1[6];
              ir1[6] = (v678_data + (v645_data * v676_data));
              float v681_data = s0[126];
              float v683_data = ir1[7];
              ir1[7] = (v683_data + (v645_data * v681_data));
            }
            if (v2_lead < 12) {
              float v689_data = r0[15];
              float v690_data = s0[15];
              float v692_data = ir1[0];
              ir1[0] = (v692_data + (v689_data * v690_data));
              float v695_data = s0[31];
              float v697_data = ir1[1];
              ir1[1] = (v697_data + (v689_data * v695_data));
              float v700_data = s0[47];
              float v702_data = ir1[2];
              ir1[2] = (v702_data + (v689_data * v700_data));
              float v705_data = s0[63];
              float v707_data = ir1[3];
              ir1[3] = (v707_data + (v689_data * v705_data));
              float v710_data = s0[79];
              float v712_data = ir1[4];
              ir1[4] = (v712_data + (v689_data * v710_data));
              float v715_data = s0[95];
              float v717_data = ir1[5];
              ir1[5] = (v717_data + (v689_data * v715_data));
              float v720_data = s0[111];
              float v722_data = ir1[6];
              ir1[6] = (v722_data + (v689_data * v720_data));
              float v725_data = s0[127];
              float v727_data = ir1[7];
              ir1[7] = (v727_data + (v689_data * v725_data));
            }
            if (v2_lead < 12) {
              #pragma unroll
              for (int32_t v733_n1 = 0; v733_n1 < 8; ++v733_n1) {
                int32_t v734_a = 0 + v733_n1;
                float v736_data = ir1[v733_n1];
                int32_t v737_a = 0 + v733_n1;
                r1[v733_n1] = v736_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v743_i1 = 0; v743_i1 < 8; ++v743_i1) {
              int32_t v744_a = 0 + v743_i1;
              float v746_data = r1[v743_i1];
              int32_t v753_a = v2_lead + (v743_i1 * 12);
              glb_m0[v753_a] = v746_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

