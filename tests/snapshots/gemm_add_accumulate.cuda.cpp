// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5e7da3148f, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_5e7da3148f, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_5e7da3148f<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 12×8(12×8) {0..12}×{0..8} strided
    // m1 12×16(12×16) {0..12}×{0..16} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              float v20_data = __ldcg(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
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
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
              int32_t v34_a = v28_i1 * 12;
              int32_t v35_a = v3_lead + v34_a;
              float v43_data = glb_m0[(v3_lead + v34_a)];
              int32_t v44_a = 0 + v28_i1;
              r1[v44_a] = v43_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          // wait(r1 = load{g>r}(glb_m0););
          float r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir2[8]{};
          if (v3_lead < 12) {
            float v51_data = r0[0];
            float v52_data = s0[0];
            float v54_data = ir2[0];
            ir2[0] = (v54_data + (v51_data * v52_data));
            float v57_data = s0[16];
            float v59_data = ir2[1];
            ir2[1] = (v59_data + (v51_data * v57_data));
            float v62_data = s0[32];
            float v64_data = ir2[2];
            ir2[2] = (v64_data + (v51_data * v62_data));
            float v67_data = s0[48];
            float v69_data = ir2[3];
            ir2[3] = (v69_data + (v51_data * v67_data));
            float v72_data = s0[64];
            float v74_data = ir2[4];
            ir2[4] = (v74_data + (v51_data * v72_data));
            float v77_data = s0[80];
            float v79_data = ir2[5];
            ir2[5] = (v79_data + (v51_data * v77_data));
            float v82_data = s0[96];
            float v84_data = ir2[6];
            ir2[6] = (v84_data + (v51_data * v82_data));
            float v87_data = s0[112];
            float v89_data = ir2[7];
            ir2[7] = (v89_data + (v51_data * v87_data));
          }
          if (v3_lead < 12) {
            float v95_data = r0[1];
            float v96_data = s0[1];
            float v98_data = ir2[0];
            ir2[0] = (v98_data + (v95_data * v96_data));
            float v101_data = s0[17];
            float v103_data = ir2[1];
            ir2[1] = (v103_data + (v95_data * v101_data));
            float v106_data = s0[33];
            float v108_data = ir2[2];
            ir2[2] = (v108_data + (v95_data * v106_data));
            float v111_data = s0[49];
            float v113_data = ir2[3];
            ir2[3] = (v113_data + (v95_data * v111_data));
            float v116_data = s0[65];
            float v118_data = ir2[4];
            ir2[4] = (v118_data + (v95_data * v116_data));
            float v121_data = s0[81];
            float v123_data = ir2[5];
            ir2[5] = (v123_data + (v95_data * v121_data));
            float v126_data = s0[97];
            float v128_data = ir2[6];
            ir2[6] = (v128_data + (v95_data * v126_data));
            float v131_data = s0[113];
            float v133_data = ir2[7];
            ir2[7] = (v133_data + (v95_data * v131_data));
          }
          if (v3_lead < 12) {
            float v139_data = r0[2];
            float v140_data = s0[2];
            float v142_data = ir2[0];
            ir2[0] = (v142_data + (v139_data * v140_data));
            float v145_data = s0[18];
            float v147_data = ir2[1];
            ir2[1] = (v147_data + (v139_data * v145_data));
            float v150_data = s0[34];
            float v152_data = ir2[2];
            ir2[2] = (v152_data + (v139_data * v150_data));
            float v155_data = s0[50];
            float v157_data = ir2[3];
            ir2[3] = (v157_data + (v139_data * v155_data));
            float v160_data = s0[66];
            float v162_data = ir2[4];
            ir2[4] = (v162_data + (v139_data * v160_data));
            float v165_data = s0[82];
            float v167_data = ir2[5];
            ir2[5] = (v167_data + (v139_data * v165_data));
            float v170_data = s0[98];
            float v172_data = ir2[6];
            ir2[6] = (v172_data + (v139_data * v170_data));
            float v175_data = s0[114];
            float v177_data = ir2[7];
            ir2[7] = (v177_data + (v139_data * v175_data));
          }
          if (v3_lead < 12) {
            float v183_data = r0[3];
            float v184_data = s0[3];
            float v186_data = ir2[0];
            ir2[0] = (v186_data + (v183_data * v184_data));
            float v189_data = s0[19];
            float v191_data = ir2[1];
            ir2[1] = (v191_data + (v183_data * v189_data));
            float v194_data = s0[35];
            float v196_data = ir2[2];
            ir2[2] = (v196_data + (v183_data * v194_data));
            float v199_data = s0[51];
            float v201_data = ir2[3];
            ir2[3] = (v201_data + (v183_data * v199_data));
            float v204_data = s0[67];
            float v206_data = ir2[4];
            ir2[4] = (v206_data + (v183_data * v204_data));
            float v209_data = s0[83];
            float v211_data = ir2[5];
            ir2[5] = (v211_data + (v183_data * v209_data));
            float v214_data = s0[99];
            float v216_data = ir2[6];
            ir2[6] = (v216_data + (v183_data * v214_data));
            float v219_data = s0[115];
            float v221_data = ir2[7];
            ir2[7] = (v221_data + (v183_data * v219_data));
          }
          if (v3_lead < 12) {
            float v227_data = r0[4];
            float v228_data = s0[4];
            float v230_data = ir2[0];
            ir2[0] = (v230_data + (v227_data * v228_data));
            float v233_data = s0[20];
            float v235_data = ir2[1];
            ir2[1] = (v235_data + (v227_data * v233_data));
            float v238_data = s0[36];
            float v240_data = ir2[2];
            ir2[2] = (v240_data + (v227_data * v238_data));
            float v243_data = s0[52];
            float v245_data = ir2[3];
            ir2[3] = (v245_data + (v227_data * v243_data));
            float v248_data = s0[68];
            float v250_data = ir2[4];
            ir2[4] = (v250_data + (v227_data * v248_data));
            float v253_data = s0[84];
            float v255_data = ir2[5];
            ir2[5] = (v255_data + (v227_data * v253_data));
            float v258_data = s0[100];
            float v260_data = ir2[6];
            ir2[6] = (v260_data + (v227_data * v258_data));
            float v263_data = s0[116];
            float v265_data = ir2[7];
            ir2[7] = (v265_data + (v227_data * v263_data));
          }
          if (v3_lead < 12) {
            float v271_data = r0[5];
            float v272_data = s0[5];
            float v274_data = ir2[0];
            ir2[0] = (v274_data + (v271_data * v272_data));
            float v277_data = s0[21];
            float v279_data = ir2[1];
            ir2[1] = (v279_data + (v271_data * v277_data));
            float v282_data = s0[37];
            float v284_data = ir2[2];
            ir2[2] = (v284_data + (v271_data * v282_data));
            float v287_data = s0[53];
            float v289_data = ir2[3];
            ir2[3] = (v289_data + (v271_data * v287_data));
            float v292_data = s0[69];
            float v294_data = ir2[4];
            ir2[4] = (v294_data + (v271_data * v292_data));
            float v297_data = s0[85];
            float v299_data = ir2[5];
            ir2[5] = (v299_data + (v271_data * v297_data));
            float v302_data = s0[101];
            float v304_data = ir2[6];
            ir2[6] = (v304_data + (v271_data * v302_data));
            float v307_data = s0[117];
            float v309_data = ir2[7];
            ir2[7] = (v309_data + (v271_data * v307_data));
          }
          if (v3_lead < 12) {
            float v315_data = r0[6];
            float v316_data = s0[6];
            float v318_data = ir2[0];
            ir2[0] = (v318_data + (v315_data * v316_data));
            float v321_data = s0[22];
            float v323_data = ir2[1];
            ir2[1] = (v323_data + (v315_data * v321_data));
            float v326_data = s0[38];
            float v328_data = ir2[2];
            ir2[2] = (v328_data + (v315_data * v326_data));
            float v331_data = s0[54];
            float v333_data = ir2[3];
            ir2[3] = (v333_data + (v315_data * v331_data));
            float v336_data = s0[70];
            float v338_data = ir2[4];
            ir2[4] = (v338_data + (v315_data * v336_data));
            float v341_data = s0[86];
            float v343_data = ir2[5];
            ir2[5] = (v343_data + (v315_data * v341_data));
            float v346_data = s0[102];
            float v348_data = ir2[6];
            ir2[6] = (v348_data + (v315_data * v346_data));
            float v351_data = s0[118];
            float v353_data = ir2[7];
            ir2[7] = (v353_data + (v315_data * v351_data));
          }
          if (v3_lead < 12) {
            float v359_data = r0[7];
            float v360_data = s0[7];
            float v362_data = ir2[0];
            ir2[0] = (v362_data + (v359_data * v360_data));
            float v365_data = s0[23];
            float v367_data = ir2[1];
            ir2[1] = (v367_data + (v359_data * v365_data));
            float v370_data = s0[39];
            float v372_data = ir2[2];
            ir2[2] = (v372_data + (v359_data * v370_data));
            float v375_data = s0[55];
            float v377_data = ir2[3];
            ir2[3] = (v377_data + (v359_data * v375_data));
            float v380_data = s0[71];
            float v382_data = ir2[4];
            ir2[4] = (v382_data + (v359_data * v380_data));
            float v385_data = s0[87];
            float v387_data = ir2[5];
            ir2[5] = (v387_data + (v359_data * v385_data));
            float v390_data = s0[103];
            float v392_data = ir2[6];
            ir2[6] = (v392_data + (v359_data * v390_data));
            float v395_data = s0[119];
            float v397_data = ir2[7];
            ir2[7] = (v397_data + (v359_data * v395_data));
          }
          if (v3_lead < 12) {
            float v403_data = r0[8];
            float v404_data = s0[8];
            float v406_data = ir2[0];
            ir2[0] = (v406_data + (v403_data * v404_data));
            float v409_data = s0[24];
            float v411_data = ir2[1];
            ir2[1] = (v411_data + (v403_data * v409_data));
            float v414_data = s0[40];
            float v416_data = ir2[2];
            ir2[2] = (v416_data + (v403_data * v414_data));
            float v419_data = s0[56];
            float v421_data = ir2[3];
            ir2[3] = (v421_data + (v403_data * v419_data));
            float v424_data = s0[72];
            float v426_data = ir2[4];
            ir2[4] = (v426_data + (v403_data * v424_data));
            float v429_data = s0[88];
            float v431_data = ir2[5];
            ir2[5] = (v431_data + (v403_data * v429_data));
            float v434_data = s0[104];
            float v436_data = ir2[6];
            ir2[6] = (v436_data + (v403_data * v434_data));
            float v439_data = s0[120];
            float v441_data = ir2[7];
            ir2[7] = (v441_data + (v403_data * v439_data));
          }
          if (v3_lead < 12) {
            float v447_data = r0[9];
            float v448_data = s0[9];
            float v450_data = ir2[0];
            ir2[0] = (v450_data + (v447_data * v448_data));
            float v453_data = s0[25];
            float v455_data = ir2[1];
            ir2[1] = (v455_data + (v447_data * v453_data));
            float v458_data = s0[41];
            float v460_data = ir2[2];
            ir2[2] = (v460_data + (v447_data * v458_data));
            float v463_data = s0[57];
            float v465_data = ir2[3];
            ir2[3] = (v465_data + (v447_data * v463_data));
            float v468_data = s0[73];
            float v470_data = ir2[4];
            ir2[4] = (v470_data + (v447_data * v468_data));
            float v473_data = s0[89];
            float v475_data = ir2[5];
            ir2[5] = (v475_data + (v447_data * v473_data));
            float v478_data = s0[105];
            float v480_data = ir2[6];
            ir2[6] = (v480_data + (v447_data * v478_data));
            float v483_data = s0[121];
            float v485_data = ir2[7];
            ir2[7] = (v485_data + (v447_data * v483_data));
          }
          if (v3_lead < 12) {
            float v491_data = r0[10];
            float v492_data = s0[10];
            float v494_data = ir2[0];
            ir2[0] = (v494_data + (v491_data * v492_data));
            float v497_data = s0[26];
            float v499_data = ir2[1];
            ir2[1] = (v499_data + (v491_data * v497_data));
            float v502_data = s0[42];
            float v504_data = ir2[2];
            ir2[2] = (v504_data + (v491_data * v502_data));
            float v507_data = s0[58];
            float v509_data = ir2[3];
            ir2[3] = (v509_data + (v491_data * v507_data));
            float v512_data = s0[74];
            float v514_data = ir2[4];
            ir2[4] = (v514_data + (v491_data * v512_data));
            float v517_data = s0[90];
            float v519_data = ir2[5];
            ir2[5] = (v519_data + (v491_data * v517_data));
            float v522_data = s0[106];
            float v524_data = ir2[6];
            ir2[6] = (v524_data + (v491_data * v522_data));
            float v527_data = s0[122];
            float v529_data = ir2[7];
            ir2[7] = (v529_data + (v491_data * v527_data));
          }
          if (v3_lead < 12) {
            float v535_data = r0[11];
            float v536_data = s0[11];
            float v538_data = ir2[0];
            ir2[0] = (v538_data + (v535_data * v536_data));
            float v541_data = s0[27];
            float v543_data = ir2[1];
            ir2[1] = (v543_data + (v535_data * v541_data));
            float v546_data = s0[43];
            float v548_data = ir2[2];
            ir2[2] = (v548_data + (v535_data * v546_data));
            float v551_data = s0[59];
            float v553_data = ir2[3];
            ir2[3] = (v553_data + (v535_data * v551_data));
            float v556_data = s0[75];
            float v558_data = ir2[4];
            ir2[4] = (v558_data + (v535_data * v556_data));
            float v561_data = s0[91];
            float v563_data = ir2[5];
            ir2[5] = (v563_data + (v535_data * v561_data));
            float v566_data = s0[107];
            float v568_data = ir2[6];
            ir2[6] = (v568_data + (v535_data * v566_data));
            float v571_data = s0[123];
            float v573_data = ir2[7];
            ir2[7] = (v573_data + (v535_data * v571_data));
          }
          if (v3_lead < 12) {
            float v579_data = r0[12];
            float v580_data = s0[12];
            float v582_data = ir2[0];
            ir2[0] = (v582_data + (v579_data * v580_data));
            float v585_data = s0[28];
            float v587_data = ir2[1];
            ir2[1] = (v587_data + (v579_data * v585_data));
            float v590_data = s0[44];
            float v592_data = ir2[2];
            ir2[2] = (v592_data + (v579_data * v590_data));
            float v595_data = s0[60];
            float v597_data = ir2[3];
            ir2[3] = (v597_data + (v579_data * v595_data));
            float v600_data = s0[76];
            float v602_data = ir2[4];
            ir2[4] = (v602_data + (v579_data * v600_data));
            float v605_data = s0[92];
            float v607_data = ir2[5];
            ir2[5] = (v607_data + (v579_data * v605_data));
            float v610_data = s0[108];
            float v612_data = ir2[6];
            ir2[6] = (v612_data + (v579_data * v610_data));
            float v615_data = s0[124];
            float v617_data = ir2[7];
            ir2[7] = (v617_data + (v579_data * v615_data));
          }
          if (v3_lead < 12) {
            float v623_data = r0[13];
            float v624_data = s0[13];
            float v626_data = ir2[0];
            ir2[0] = (v626_data + (v623_data * v624_data));
            float v629_data = s0[29];
            float v631_data = ir2[1];
            ir2[1] = (v631_data + (v623_data * v629_data));
            float v634_data = s0[45];
            float v636_data = ir2[2];
            ir2[2] = (v636_data + (v623_data * v634_data));
            float v639_data = s0[61];
            float v641_data = ir2[3];
            ir2[3] = (v641_data + (v623_data * v639_data));
            float v644_data = s0[77];
            float v646_data = ir2[4];
            ir2[4] = (v646_data + (v623_data * v644_data));
            float v649_data = s0[93];
            float v651_data = ir2[5];
            ir2[5] = (v651_data + (v623_data * v649_data));
            float v654_data = s0[109];
            float v656_data = ir2[6];
            ir2[6] = (v656_data + (v623_data * v654_data));
            float v659_data = s0[125];
            float v661_data = ir2[7];
            ir2[7] = (v661_data + (v623_data * v659_data));
          }
          if (v3_lead < 12) {
            float v667_data = r0[14];
            float v668_data = s0[14];
            float v670_data = ir2[0];
            ir2[0] = (v670_data + (v667_data * v668_data));
            float v673_data = s0[30];
            float v675_data = ir2[1];
            ir2[1] = (v675_data + (v667_data * v673_data));
            float v678_data = s0[46];
            float v680_data = ir2[2];
            ir2[2] = (v680_data + (v667_data * v678_data));
            float v683_data = s0[62];
            float v685_data = ir2[3];
            ir2[3] = (v685_data + (v667_data * v683_data));
            float v688_data = s0[78];
            float v690_data = ir2[4];
            ir2[4] = (v690_data + (v667_data * v688_data));
            float v693_data = s0[94];
            float v695_data = ir2[5];
            ir2[5] = (v695_data + (v667_data * v693_data));
            float v698_data = s0[110];
            float v700_data = ir2[6];
            ir2[6] = (v700_data + (v667_data * v698_data));
            float v703_data = s0[126];
            float v705_data = ir2[7];
            ir2[7] = (v705_data + (v667_data * v703_data));
          }
          if (v3_lead < 12) {
            float v711_data = r0[15];
            float v712_data = s0[15];
            float v714_data = ir2[0];
            ir2[0] = (v714_data + (v711_data * v712_data));
            float v717_data = s0[31];
            float v719_data = ir2[1];
            ir2[1] = (v719_data + (v711_data * v717_data));
            float v722_data = s0[47];
            float v724_data = ir2[2];
            ir2[2] = (v724_data + (v711_data * v722_data));
            float v727_data = s0[63];
            float v729_data = ir2[3];
            ir2[3] = (v729_data + (v711_data * v727_data));
            float v732_data = s0[79];
            float v734_data = ir2[4];
            ir2[4] = (v734_data + (v711_data * v732_data));
            float v737_data = s0[95];
            float v739_data = ir2[5];
            ir2[5] = (v739_data + (v711_data * v737_data));
            float v742_data = s0[111];
            float v744_data = ir2[6];
            ir2[6] = (v744_data + (v711_data * v742_data));
            float v747_data = s0[127];
            float v749_data = ir2[7];
            ir2[7] = (v749_data + (v711_data * v747_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v755_n1 = 0; v755_n1 < 8; ++v755_n1) {
              int32_t v756_a = 0 + v755_n1;
              float v758_data = ir2[v755_n1];
              int32_t v759_a = 0 + v755_n1;
              float v761_data = r1[v755_n1];
              int32_t v763_a = 0 + v755_n1;
              r2[v755_n1] = (v761_data + v758_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v769_i1 = 0; v769_i1 < 8; ++v769_i1) {
              int32_t v770_a = 0 + v769_i1;
              float v772_data = r2[v769_i1];
              int32_t v779_a = v3_lead + (v769_i1 * 12);
              glb_m0[v779_a] = v772_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

