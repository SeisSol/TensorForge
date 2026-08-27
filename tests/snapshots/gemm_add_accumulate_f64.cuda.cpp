// === base name ===
kernel_16c847f49d

// === header ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_16c847f49d, block.x * block.y * block.z, 2304 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_16c847f49d, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_16c847f49d<<<grid,block,2304 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_16c847f49d(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v3_lead = threadIdx.x % 16;
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v5_i1 = 0; v5_i1 < 16; ++v5_i1) {
              int32_t v11_a = v5_i1 * 12;
              int32_t v12_a = v3_lead + v11_a;
              double v20_data = __ldcg(&glb_m1[(v3_lead + v11_a)]);
              int32_t v21_a = 0 + v5_i1;
              r0[v21_a] = v20_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            pipeline.producer_acquire();
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              cuda::memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], cuda::aligned_size_t<8>(8), pipeline);
            }
            __syncwarp();
            pipeline.producer_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
              int32_t v34_a = v28_i1 * 12;
              int32_t v35_a = v3_lead + v34_a;
              double v43_data = glb_m0[(v3_lead + v34_a)];
              int32_t v44_a = 0 + v28_i1;
              r1[v44_a] = v43_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          if (v3_lead < 12) {
            double v51_data = r0[0];
            double v52_data = s0[0];
            double v54_data = ir2[0];
            ir2[0] = (v54_data + (v51_data * v52_data));
            double v57_data = s0[16];
            double v59_data = ir2[1];
            ir2[1] = (v59_data + (v51_data * v57_data));
            double v62_data = s0[32];
            double v64_data = ir2[2];
            ir2[2] = (v64_data + (v51_data * v62_data));
            double v67_data = s0[48];
            double v69_data = ir2[3];
            ir2[3] = (v69_data + (v51_data * v67_data));
            double v72_data = s0[64];
            double v74_data = ir2[4];
            ir2[4] = (v74_data + (v51_data * v72_data));
            double v77_data = s0[80];
            double v79_data = ir2[5];
            ir2[5] = (v79_data + (v51_data * v77_data));
            double v82_data = s0[96];
            double v84_data = ir2[6];
            ir2[6] = (v84_data + (v51_data * v82_data));
            double v87_data = s0[112];
            double v89_data = ir2[7];
            ir2[7] = (v89_data + (v51_data * v87_data));
          }
          if (v3_lead < 12) {
            double v95_data = r0[1];
            double v96_data = s0[1];
            double v98_data = ir2[0];
            ir2[0] = (v98_data + (v95_data * v96_data));
            double v101_data = s0[17];
            double v103_data = ir2[1];
            ir2[1] = (v103_data + (v95_data * v101_data));
            double v106_data = s0[33];
            double v108_data = ir2[2];
            ir2[2] = (v108_data + (v95_data * v106_data));
            double v111_data = s0[49];
            double v113_data = ir2[3];
            ir2[3] = (v113_data + (v95_data * v111_data));
            double v116_data = s0[65];
            double v118_data = ir2[4];
            ir2[4] = (v118_data + (v95_data * v116_data));
            double v121_data = s0[81];
            double v123_data = ir2[5];
            ir2[5] = (v123_data + (v95_data * v121_data));
            double v126_data = s0[97];
            double v128_data = ir2[6];
            ir2[6] = (v128_data + (v95_data * v126_data));
            double v131_data = s0[113];
            double v133_data = ir2[7];
            ir2[7] = (v133_data + (v95_data * v131_data));
          }
          if (v3_lead < 12) {
            double v139_data = r0[2];
            double v140_data = s0[2];
            double v142_data = ir2[0];
            ir2[0] = (v142_data + (v139_data * v140_data));
            double v145_data = s0[18];
            double v147_data = ir2[1];
            ir2[1] = (v147_data + (v139_data * v145_data));
            double v150_data = s0[34];
            double v152_data = ir2[2];
            ir2[2] = (v152_data + (v139_data * v150_data));
            double v155_data = s0[50];
            double v157_data = ir2[3];
            ir2[3] = (v157_data + (v139_data * v155_data));
            double v160_data = s0[66];
            double v162_data = ir2[4];
            ir2[4] = (v162_data + (v139_data * v160_data));
            double v165_data = s0[82];
            double v167_data = ir2[5];
            ir2[5] = (v167_data + (v139_data * v165_data));
            double v170_data = s0[98];
            double v172_data = ir2[6];
            ir2[6] = (v172_data + (v139_data * v170_data));
            double v175_data = s0[114];
            double v177_data = ir2[7];
            ir2[7] = (v177_data + (v139_data * v175_data));
          }
          if (v3_lead < 12) {
            double v183_data = r0[3];
            double v184_data = s0[3];
            double v186_data = ir2[0];
            ir2[0] = (v186_data + (v183_data * v184_data));
            double v189_data = s0[19];
            double v191_data = ir2[1];
            ir2[1] = (v191_data + (v183_data * v189_data));
            double v194_data = s0[35];
            double v196_data = ir2[2];
            ir2[2] = (v196_data + (v183_data * v194_data));
            double v199_data = s0[51];
            double v201_data = ir2[3];
            ir2[3] = (v201_data + (v183_data * v199_data));
            double v204_data = s0[67];
            double v206_data = ir2[4];
            ir2[4] = (v206_data + (v183_data * v204_data));
            double v209_data = s0[83];
            double v211_data = ir2[5];
            ir2[5] = (v211_data + (v183_data * v209_data));
            double v214_data = s0[99];
            double v216_data = ir2[6];
            ir2[6] = (v216_data + (v183_data * v214_data));
            double v219_data = s0[115];
            double v221_data = ir2[7];
            ir2[7] = (v221_data + (v183_data * v219_data));
          }
          if (v3_lead < 12) {
            double v227_data = r0[4];
            double v228_data = s0[4];
            double v230_data = ir2[0];
            ir2[0] = (v230_data + (v227_data * v228_data));
            double v233_data = s0[20];
            double v235_data = ir2[1];
            ir2[1] = (v235_data + (v227_data * v233_data));
            double v238_data = s0[36];
            double v240_data = ir2[2];
            ir2[2] = (v240_data + (v227_data * v238_data));
            double v243_data = s0[52];
            double v245_data = ir2[3];
            ir2[3] = (v245_data + (v227_data * v243_data));
            double v248_data = s0[68];
            double v250_data = ir2[4];
            ir2[4] = (v250_data + (v227_data * v248_data));
            double v253_data = s0[84];
            double v255_data = ir2[5];
            ir2[5] = (v255_data + (v227_data * v253_data));
            double v258_data = s0[100];
            double v260_data = ir2[6];
            ir2[6] = (v260_data + (v227_data * v258_data));
            double v263_data = s0[116];
            double v265_data = ir2[7];
            ir2[7] = (v265_data + (v227_data * v263_data));
          }
          if (v3_lead < 12) {
            double v271_data = r0[5];
            double v272_data = s0[5];
            double v274_data = ir2[0];
            ir2[0] = (v274_data + (v271_data * v272_data));
            double v277_data = s0[21];
            double v279_data = ir2[1];
            ir2[1] = (v279_data + (v271_data * v277_data));
            double v282_data = s0[37];
            double v284_data = ir2[2];
            ir2[2] = (v284_data + (v271_data * v282_data));
            double v287_data = s0[53];
            double v289_data = ir2[3];
            ir2[3] = (v289_data + (v271_data * v287_data));
            double v292_data = s0[69];
            double v294_data = ir2[4];
            ir2[4] = (v294_data + (v271_data * v292_data));
            double v297_data = s0[85];
            double v299_data = ir2[5];
            ir2[5] = (v299_data + (v271_data * v297_data));
            double v302_data = s0[101];
            double v304_data = ir2[6];
            ir2[6] = (v304_data + (v271_data * v302_data));
            double v307_data = s0[117];
            double v309_data = ir2[7];
            ir2[7] = (v309_data + (v271_data * v307_data));
          }
          if (v3_lead < 12) {
            double v315_data = r0[6];
            double v316_data = s0[6];
            double v318_data = ir2[0];
            ir2[0] = (v318_data + (v315_data * v316_data));
            double v321_data = s0[22];
            double v323_data = ir2[1];
            ir2[1] = (v323_data + (v315_data * v321_data));
            double v326_data = s0[38];
            double v328_data = ir2[2];
            ir2[2] = (v328_data + (v315_data * v326_data));
            double v331_data = s0[54];
            double v333_data = ir2[3];
            ir2[3] = (v333_data + (v315_data * v331_data));
            double v336_data = s0[70];
            double v338_data = ir2[4];
            ir2[4] = (v338_data + (v315_data * v336_data));
            double v341_data = s0[86];
            double v343_data = ir2[5];
            ir2[5] = (v343_data + (v315_data * v341_data));
            double v346_data = s0[102];
            double v348_data = ir2[6];
            ir2[6] = (v348_data + (v315_data * v346_data));
            double v351_data = s0[118];
            double v353_data = ir2[7];
            ir2[7] = (v353_data + (v315_data * v351_data));
          }
          if (v3_lead < 12) {
            double v359_data = r0[7];
            double v360_data = s0[7];
            double v362_data = ir2[0];
            ir2[0] = (v362_data + (v359_data * v360_data));
            double v365_data = s0[23];
            double v367_data = ir2[1];
            ir2[1] = (v367_data + (v359_data * v365_data));
            double v370_data = s0[39];
            double v372_data = ir2[2];
            ir2[2] = (v372_data + (v359_data * v370_data));
            double v375_data = s0[55];
            double v377_data = ir2[3];
            ir2[3] = (v377_data + (v359_data * v375_data));
            double v380_data = s0[71];
            double v382_data = ir2[4];
            ir2[4] = (v382_data + (v359_data * v380_data));
            double v385_data = s0[87];
            double v387_data = ir2[5];
            ir2[5] = (v387_data + (v359_data * v385_data));
            double v390_data = s0[103];
            double v392_data = ir2[6];
            ir2[6] = (v392_data + (v359_data * v390_data));
            double v395_data = s0[119];
            double v397_data = ir2[7];
            ir2[7] = (v397_data + (v359_data * v395_data));
          }
          if (v3_lead < 12) {
            double v403_data = r0[8];
            double v404_data = s0[8];
            double v406_data = ir2[0];
            ir2[0] = (v406_data + (v403_data * v404_data));
            double v409_data = s0[24];
            double v411_data = ir2[1];
            ir2[1] = (v411_data + (v403_data * v409_data));
            double v414_data = s0[40];
            double v416_data = ir2[2];
            ir2[2] = (v416_data + (v403_data * v414_data));
            double v419_data = s0[56];
            double v421_data = ir2[3];
            ir2[3] = (v421_data + (v403_data * v419_data));
            double v424_data = s0[72];
            double v426_data = ir2[4];
            ir2[4] = (v426_data + (v403_data * v424_data));
            double v429_data = s0[88];
            double v431_data = ir2[5];
            ir2[5] = (v431_data + (v403_data * v429_data));
            double v434_data = s0[104];
            double v436_data = ir2[6];
            ir2[6] = (v436_data + (v403_data * v434_data));
            double v439_data = s0[120];
            double v441_data = ir2[7];
            ir2[7] = (v441_data + (v403_data * v439_data));
          }
          if (v3_lead < 12) {
            double v447_data = r0[9];
            double v448_data = s0[9];
            double v450_data = ir2[0];
            ir2[0] = (v450_data + (v447_data * v448_data));
            double v453_data = s0[25];
            double v455_data = ir2[1];
            ir2[1] = (v455_data + (v447_data * v453_data));
            double v458_data = s0[41];
            double v460_data = ir2[2];
            ir2[2] = (v460_data + (v447_data * v458_data));
            double v463_data = s0[57];
            double v465_data = ir2[3];
            ir2[3] = (v465_data + (v447_data * v463_data));
            double v468_data = s0[73];
            double v470_data = ir2[4];
            ir2[4] = (v470_data + (v447_data * v468_data));
            double v473_data = s0[89];
            double v475_data = ir2[5];
            ir2[5] = (v475_data + (v447_data * v473_data));
            double v478_data = s0[105];
            double v480_data = ir2[6];
            ir2[6] = (v480_data + (v447_data * v478_data));
            double v483_data = s0[121];
            double v485_data = ir2[7];
            ir2[7] = (v485_data + (v447_data * v483_data));
          }
          if (v3_lead < 12) {
            double v491_data = r0[10];
            double v492_data = s0[10];
            double v494_data = ir2[0];
            ir2[0] = (v494_data + (v491_data * v492_data));
            double v497_data = s0[26];
            double v499_data = ir2[1];
            ir2[1] = (v499_data + (v491_data * v497_data));
            double v502_data = s0[42];
            double v504_data = ir2[2];
            ir2[2] = (v504_data + (v491_data * v502_data));
            double v507_data = s0[58];
            double v509_data = ir2[3];
            ir2[3] = (v509_data + (v491_data * v507_data));
            double v512_data = s0[74];
            double v514_data = ir2[4];
            ir2[4] = (v514_data + (v491_data * v512_data));
            double v517_data = s0[90];
            double v519_data = ir2[5];
            ir2[5] = (v519_data + (v491_data * v517_data));
            double v522_data = s0[106];
            double v524_data = ir2[6];
            ir2[6] = (v524_data + (v491_data * v522_data));
            double v527_data = s0[122];
            double v529_data = ir2[7];
            ir2[7] = (v529_data + (v491_data * v527_data));
          }
          if (v3_lead < 12) {
            double v535_data = r0[11];
            double v536_data = s0[11];
            double v538_data = ir2[0];
            ir2[0] = (v538_data + (v535_data * v536_data));
            double v541_data = s0[27];
            double v543_data = ir2[1];
            ir2[1] = (v543_data + (v535_data * v541_data));
            double v546_data = s0[43];
            double v548_data = ir2[2];
            ir2[2] = (v548_data + (v535_data * v546_data));
            double v551_data = s0[59];
            double v553_data = ir2[3];
            ir2[3] = (v553_data + (v535_data * v551_data));
            double v556_data = s0[75];
            double v558_data = ir2[4];
            ir2[4] = (v558_data + (v535_data * v556_data));
            double v561_data = s0[91];
            double v563_data = ir2[5];
            ir2[5] = (v563_data + (v535_data * v561_data));
            double v566_data = s0[107];
            double v568_data = ir2[6];
            ir2[6] = (v568_data + (v535_data * v566_data));
            double v571_data = s0[123];
            double v573_data = ir2[7];
            ir2[7] = (v573_data + (v535_data * v571_data));
          }
          if (v3_lead < 12) {
            double v579_data = r0[12];
            double v580_data = s0[12];
            double v582_data = ir2[0];
            ir2[0] = (v582_data + (v579_data * v580_data));
            double v585_data = s0[28];
            double v587_data = ir2[1];
            ir2[1] = (v587_data + (v579_data * v585_data));
            double v590_data = s0[44];
            double v592_data = ir2[2];
            ir2[2] = (v592_data + (v579_data * v590_data));
            double v595_data = s0[60];
            double v597_data = ir2[3];
            ir2[3] = (v597_data + (v579_data * v595_data));
            double v600_data = s0[76];
            double v602_data = ir2[4];
            ir2[4] = (v602_data + (v579_data * v600_data));
            double v605_data = s0[92];
            double v607_data = ir2[5];
            ir2[5] = (v607_data + (v579_data * v605_data));
            double v610_data = s0[108];
            double v612_data = ir2[6];
            ir2[6] = (v612_data + (v579_data * v610_data));
            double v615_data = s0[124];
            double v617_data = ir2[7];
            ir2[7] = (v617_data + (v579_data * v615_data));
          }
          if (v3_lead < 12) {
            double v623_data = r0[13];
            double v624_data = s0[13];
            double v626_data = ir2[0];
            ir2[0] = (v626_data + (v623_data * v624_data));
            double v629_data = s0[29];
            double v631_data = ir2[1];
            ir2[1] = (v631_data + (v623_data * v629_data));
            double v634_data = s0[45];
            double v636_data = ir2[2];
            ir2[2] = (v636_data + (v623_data * v634_data));
            double v639_data = s0[61];
            double v641_data = ir2[3];
            ir2[3] = (v641_data + (v623_data * v639_data));
            double v644_data = s0[77];
            double v646_data = ir2[4];
            ir2[4] = (v646_data + (v623_data * v644_data));
            double v649_data = s0[93];
            double v651_data = ir2[5];
            ir2[5] = (v651_data + (v623_data * v649_data));
            double v654_data = s0[109];
            double v656_data = ir2[6];
            ir2[6] = (v656_data + (v623_data * v654_data));
            double v659_data = s0[125];
            double v661_data = ir2[7];
            ir2[7] = (v661_data + (v623_data * v659_data));
          }
          if (v3_lead < 12) {
            double v667_data = r0[14];
            double v668_data = s0[14];
            double v670_data = ir2[0];
            ir2[0] = (v670_data + (v667_data * v668_data));
            double v673_data = s0[30];
            double v675_data = ir2[1];
            ir2[1] = (v675_data + (v667_data * v673_data));
            double v678_data = s0[46];
            double v680_data = ir2[2];
            ir2[2] = (v680_data + (v667_data * v678_data));
            double v683_data = s0[62];
            double v685_data = ir2[3];
            ir2[3] = (v685_data + (v667_data * v683_data));
            double v688_data = s0[78];
            double v690_data = ir2[4];
            ir2[4] = (v690_data + (v667_data * v688_data));
            double v693_data = s0[94];
            double v695_data = ir2[5];
            ir2[5] = (v695_data + (v667_data * v693_data));
            double v698_data = s0[110];
            double v700_data = ir2[6];
            ir2[6] = (v700_data + (v667_data * v698_data));
            double v703_data = s0[126];
            double v705_data = ir2[7];
            ir2[7] = (v705_data + (v667_data * v703_data));
          }
          if (v3_lead < 12) {
            double v711_data = r0[15];
            double v712_data = s0[15];
            double v714_data = ir2[0];
            ir2[0] = (v714_data + (v711_data * v712_data));
            double v717_data = s0[31];
            double v719_data = ir2[1];
            ir2[1] = (v719_data + (v711_data * v717_data));
            double v722_data = s0[47];
            double v724_data = ir2[2];
            ir2[2] = (v724_data + (v711_data * v722_data));
            double v727_data = s0[63];
            double v729_data = ir2[3];
            ir2[3] = (v729_data + (v711_data * v727_data));
            double v732_data = s0[79];
            double v734_data = ir2[4];
            ir2[4] = (v734_data + (v711_data * v732_data));
            double v737_data = s0[95];
            double v739_data = ir2[5];
            ir2[5] = (v739_data + (v711_data * v737_data));
            double v742_data = s0[111];
            double v744_data = ir2[6];
            ir2[6] = (v744_data + (v711_data * v742_data));
            double v747_data = s0[127];
            double v749_data = ir2[7];
            ir2[7] = (v749_data + (v711_data * v747_data));
          }
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v755_n1 = 0; v755_n1 < 8; ++v755_n1) {
              int32_t v756_a = 0 + v755_n1;
              double v758_data = ir2[v755_n1];
              int32_t v759_a = 0 + v755_n1;
              double v761_data = r1[v755_n1];
              int32_t v763_a = 0 + v755_n1;
              r2[v755_n1] = (v761_data + v758_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v3_lead < 12) {
            #pragma unroll
            for (int32_t v769_i1 = 0; v769_i1 < 8; ++v769_i1) {
              int32_t v770_a = 0 + v769_i1;
              double v772_data = r2[v769_i1];
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

