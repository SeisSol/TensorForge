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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v10_a = v4_i1 * 12;
              int32_t v11_a = v2_lead + v10_a;
              double v19_data = __ldcg(&glb_m1[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
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
          int32_t v23_lead = threadIdx.x % 16;
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
              int32_t v31_a = v25_i1 * 12;
              int32_t v32_a = v23_lead + v31_a;
              double v40_data = glb_m0[(v23_lead + v31_a)];
              int32_t v41_a = 0 + v25_i1;
              r1[v41_a] = v40_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          __syncwarp();
          {
            // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 16)]
            double ir2[8]{};
            int32_t v44_lead = threadIdx.x % 16;
            if (v44_lead < 12) {
              double v46_data = r0[0];
              double v47_data = s0[0];
              double v49_data = ir2[0];
              ir2[0] = (v49_data + (v46_data * v47_data));
              double v52_data = s0[16];
              double v54_data = ir2[1];
              ir2[1] = (v54_data + (v46_data * v52_data));
              double v57_data = s0[32];
              double v59_data = ir2[2];
              ir2[2] = (v59_data + (v46_data * v57_data));
              double v62_data = s0[48];
              double v64_data = ir2[3];
              ir2[3] = (v64_data + (v46_data * v62_data));
              double v67_data = s0[64];
              double v69_data = ir2[4];
              ir2[4] = (v69_data + (v46_data * v67_data));
              double v72_data = s0[80];
              double v74_data = ir2[5];
              ir2[5] = (v74_data + (v46_data * v72_data));
              double v77_data = s0[96];
              double v79_data = ir2[6];
              ir2[6] = (v79_data + (v46_data * v77_data));
              double v82_data = s0[112];
              double v84_data = ir2[7];
              ir2[7] = (v84_data + (v46_data * v82_data));
            }
            if (v44_lead < 12) {
              double v90_data = r0[1];
              double v91_data = s0[1];
              double v93_data = ir2[0];
              ir2[0] = (v93_data + (v90_data * v91_data));
              double v96_data = s0[17];
              double v98_data = ir2[1];
              ir2[1] = (v98_data + (v90_data * v96_data));
              double v101_data = s0[33];
              double v103_data = ir2[2];
              ir2[2] = (v103_data + (v90_data * v101_data));
              double v106_data = s0[49];
              double v108_data = ir2[3];
              ir2[3] = (v108_data + (v90_data * v106_data));
              double v111_data = s0[65];
              double v113_data = ir2[4];
              ir2[4] = (v113_data + (v90_data * v111_data));
              double v116_data = s0[81];
              double v118_data = ir2[5];
              ir2[5] = (v118_data + (v90_data * v116_data));
              double v121_data = s0[97];
              double v123_data = ir2[6];
              ir2[6] = (v123_data + (v90_data * v121_data));
              double v126_data = s0[113];
              double v128_data = ir2[7];
              ir2[7] = (v128_data + (v90_data * v126_data));
            }
            if (v44_lead < 12) {
              double v134_data = r0[2];
              double v135_data = s0[2];
              double v137_data = ir2[0];
              ir2[0] = (v137_data + (v134_data * v135_data));
              double v140_data = s0[18];
              double v142_data = ir2[1];
              ir2[1] = (v142_data + (v134_data * v140_data));
              double v145_data = s0[34];
              double v147_data = ir2[2];
              ir2[2] = (v147_data + (v134_data * v145_data));
              double v150_data = s0[50];
              double v152_data = ir2[3];
              ir2[3] = (v152_data + (v134_data * v150_data));
              double v155_data = s0[66];
              double v157_data = ir2[4];
              ir2[4] = (v157_data + (v134_data * v155_data));
              double v160_data = s0[82];
              double v162_data = ir2[5];
              ir2[5] = (v162_data + (v134_data * v160_data));
              double v165_data = s0[98];
              double v167_data = ir2[6];
              ir2[6] = (v167_data + (v134_data * v165_data));
              double v170_data = s0[114];
              double v172_data = ir2[7];
              ir2[7] = (v172_data + (v134_data * v170_data));
            }
            if (v44_lead < 12) {
              double v178_data = r0[3];
              double v179_data = s0[3];
              double v181_data = ir2[0];
              ir2[0] = (v181_data + (v178_data * v179_data));
              double v184_data = s0[19];
              double v186_data = ir2[1];
              ir2[1] = (v186_data + (v178_data * v184_data));
              double v189_data = s0[35];
              double v191_data = ir2[2];
              ir2[2] = (v191_data + (v178_data * v189_data));
              double v194_data = s0[51];
              double v196_data = ir2[3];
              ir2[3] = (v196_data + (v178_data * v194_data));
              double v199_data = s0[67];
              double v201_data = ir2[4];
              ir2[4] = (v201_data + (v178_data * v199_data));
              double v204_data = s0[83];
              double v206_data = ir2[5];
              ir2[5] = (v206_data + (v178_data * v204_data));
              double v209_data = s0[99];
              double v211_data = ir2[6];
              ir2[6] = (v211_data + (v178_data * v209_data));
              double v214_data = s0[115];
              double v216_data = ir2[7];
              ir2[7] = (v216_data + (v178_data * v214_data));
            }
            if (v44_lead < 12) {
              double v222_data = r0[4];
              double v223_data = s0[4];
              double v225_data = ir2[0];
              ir2[0] = (v225_data + (v222_data * v223_data));
              double v228_data = s0[20];
              double v230_data = ir2[1];
              ir2[1] = (v230_data + (v222_data * v228_data));
              double v233_data = s0[36];
              double v235_data = ir2[2];
              ir2[2] = (v235_data + (v222_data * v233_data));
              double v238_data = s0[52];
              double v240_data = ir2[3];
              ir2[3] = (v240_data + (v222_data * v238_data));
              double v243_data = s0[68];
              double v245_data = ir2[4];
              ir2[4] = (v245_data + (v222_data * v243_data));
              double v248_data = s0[84];
              double v250_data = ir2[5];
              ir2[5] = (v250_data + (v222_data * v248_data));
              double v253_data = s0[100];
              double v255_data = ir2[6];
              ir2[6] = (v255_data + (v222_data * v253_data));
              double v258_data = s0[116];
              double v260_data = ir2[7];
              ir2[7] = (v260_data + (v222_data * v258_data));
            }
            if (v44_lead < 12) {
              double v266_data = r0[5];
              double v267_data = s0[5];
              double v269_data = ir2[0];
              ir2[0] = (v269_data + (v266_data * v267_data));
              double v272_data = s0[21];
              double v274_data = ir2[1];
              ir2[1] = (v274_data + (v266_data * v272_data));
              double v277_data = s0[37];
              double v279_data = ir2[2];
              ir2[2] = (v279_data + (v266_data * v277_data));
              double v282_data = s0[53];
              double v284_data = ir2[3];
              ir2[3] = (v284_data + (v266_data * v282_data));
              double v287_data = s0[69];
              double v289_data = ir2[4];
              ir2[4] = (v289_data + (v266_data * v287_data));
              double v292_data = s0[85];
              double v294_data = ir2[5];
              ir2[5] = (v294_data + (v266_data * v292_data));
              double v297_data = s0[101];
              double v299_data = ir2[6];
              ir2[6] = (v299_data + (v266_data * v297_data));
              double v302_data = s0[117];
              double v304_data = ir2[7];
              ir2[7] = (v304_data + (v266_data * v302_data));
            }
            if (v44_lead < 12) {
              double v310_data = r0[6];
              double v311_data = s0[6];
              double v313_data = ir2[0];
              ir2[0] = (v313_data + (v310_data * v311_data));
              double v316_data = s0[22];
              double v318_data = ir2[1];
              ir2[1] = (v318_data + (v310_data * v316_data));
              double v321_data = s0[38];
              double v323_data = ir2[2];
              ir2[2] = (v323_data + (v310_data * v321_data));
              double v326_data = s0[54];
              double v328_data = ir2[3];
              ir2[3] = (v328_data + (v310_data * v326_data));
              double v331_data = s0[70];
              double v333_data = ir2[4];
              ir2[4] = (v333_data + (v310_data * v331_data));
              double v336_data = s0[86];
              double v338_data = ir2[5];
              ir2[5] = (v338_data + (v310_data * v336_data));
              double v341_data = s0[102];
              double v343_data = ir2[6];
              ir2[6] = (v343_data + (v310_data * v341_data));
              double v346_data = s0[118];
              double v348_data = ir2[7];
              ir2[7] = (v348_data + (v310_data * v346_data));
            }
            if (v44_lead < 12) {
              double v354_data = r0[7];
              double v355_data = s0[7];
              double v357_data = ir2[0];
              ir2[0] = (v357_data + (v354_data * v355_data));
              double v360_data = s0[23];
              double v362_data = ir2[1];
              ir2[1] = (v362_data + (v354_data * v360_data));
              double v365_data = s0[39];
              double v367_data = ir2[2];
              ir2[2] = (v367_data + (v354_data * v365_data));
              double v370_data = s0[55];
              double v372_data = ir2[3];
              ir2[3] = (v372_data + (v354_data * v370_data));
              double v375_data = s0[71];
              double v377_data = ir2[4];
              ir2[4] = (v377_data + (v354_data * v375_data));
              double v380_data = s0[87];
              double v382_data = ir2[5];
              ir2[5] = (v382_data + (v354_data * v380_data));
              double v385_data = s0[103];
              double v387_data = ir2[6];
              ir2[6] = (v387_data + (v354_data * v385_data));
              double v390_data = s0[119];
              double v392_data = ir2[7];
              ir2[7] = (v392_data + (v354_data * v390_data));
            }
            if (v44_lead < 12) {
              double v398_data = r0[8];
              double v399_data = s0[8];
              double v401_data = ir2[0];
              ir2[0] = (v401_data + (v398_data * v399_data));
              double v404_data = s0[24];
              double v406_data = ir2[1];
              ir2[1] = (v406_data + (v398_data * v404_data));
              double v409_data = s0[40];
              double v411_data = ir2[2];
              ir2[2] = (v411_data + (v398_data * v409_data));
              double v414_data = s0[56];
              double v416_data = ir2[3];
              ir2[3] = (v416_data + (v398_data * v414_data));
              double v419_data = s0[72];
              double v421_data = ir2[4];
              ir2[4] = (v421_data + (v398_data * v419_data));
              double v424_data = s0[88];
              double v426_data = ir2[5];
              ir2[5] = (v426_data + (v398_data * v424_data));
              double v429_data = s0[104];
              double v431_data = ir2[6];
              ir2[6] = (v431_data + (v398_data * v429_data));
              double v434_data = s0[120];
              double v436_data = ir2[7];
              ir2[7] = (v436_data + (v398_data * v434_data));
            }
            if (v44_lead < 12) {
              double v442_data = r0[9];
              double v443_data = s0[9];
              double v445_data = ir2[0];
              ir2[0] = (v445_data + (v442_data * v443_data));
              double v448_data = s0[25];
              double v450_data = ir2[1];
              ir2[1] = (v450_data + (v442_data * v448_data));
              double v453_data = s0[41];
              double v455_data = ir2[2];
              ir2[2] = (v455_data + (v442_data * v453_data));
              double v458_data = s0[57];
              double v460_data = ir2[3];
              ir2[3] = (v460_data + (v442_data * v458_data));
              double v463_data = s0[73];
              double v465_data = ir2[4];
              ir2[4] = (v465_data + (v442_data * v463_data));
              double v468_data = s0[89];
              double v470_data = ir2[5];
              ir2[5] = (v470_data + (v442_data * v468_data));
              double v473_data = s0[105];
              double v475_data = ir2[6];
              ir2[6] = (v475_data + (v442_data * v473_data));
              double v478_data = s0[121];
              double v480_data = ir2[7];
              ir2[7] = (v480_data + (v442_data * v478_data));
            }
            if (v44_lead < 12) {
              double v486_data = r0[10];
              double v487_data = s0[10];
              double v489_data = ir2[0];
              ir2[0] = (v489_data + (v486_data * v487_data));
              double v492_data = s0[26];
              double v494_data = ir2[1];
              ir2[1] = (v494_data + (v486_data * v492_data));
              double v497_data = s0[42];
              double v499_data = ir2[2];
              ir2[2] = (v499_data + (v486_data * v497_data));
              double v502_data = s0[58];
              double v504_data = ir2[3];
              ir2[3] = (v504_data + (v486_data * v502_data));
              double v507_data = s0[74];
              double v509_data = ir2[4];
              ir2[4] = (v509_data + (v486_data * v507_data));
              double v512_data = s0[90];
              double v514_data = ir2[5];
              ir2[5] = (v514_data + (v486_data * v512_data));
              double v517_data = s0[106];
              double v519_data = ir2[6];
              ir2[6] = (v519_data + (v486_data * v517_data));
              double v522_data = s0[122];
              double v524_data = ir2[7];
              ir2[7] = (v524_data + (v486_data * v522_data));
            }
            if (v44_lead < 12) {
              double v530_data = r0[11];
              double v531_data = s0[11];
              double v533_data = ir2[0];
              ir2[0] = (v533_data + (v530_data * v531_data));
              double v536_data = s0[27];
              double v538_data = ir2[1];
              ir2[1] = (v538_data + (v530_data * v536_data));
              double v541_data = s0[43];
              double v543_data = ir2[2];
              ir2[2] = (v543_data + (v530_data * v541_data));
              double v546_data = s0[59];
              double v548_data = ir2[3];
              ir2[3] = (v548_data + (v530_data * v546_data));
              double v551_data = s0[75];
              double v553_data = ir2[4];
              ir2[4] = (v553_data + (v530_data * v551_data));
              double v556_data = s0[91];
              double v558_data = ir2[5];
              ir2[5] = (v558_data + (v530_data * v556_data));
              double v561_data = s0[107];
              double v563_data = ir2[6];
              ir2[6] = (v563_data + (v530_data * v561_data));
              double v566_data = s0[123];
              double v568_data = ir2[7];
              ir2[7] = (v568_data + (v530_data * v566_data));
            }
            if (v44_lead < 12) {
              double v574_data = r0[12];
              double v575_data = s0[12];
              double v577_data = ir2[0];
              ir2[0] = (v577_data + (v574_data * v575_data));
              double v580_data = s0[28];
              double v582_data = ir2[1];
              ir2[1] = (v582_data + (v574_data * v580_data));
              double v585_data = s0[44];
              double v587_data = ir2[2];
              ir2[2] = (v587_data + (v574_data * v585_data));
              double v590_data = s0[60];
              double v592_data = ir2[3];
              ir2[3] = (v592_data + (v574_data * v590_data));
              double v595_data = s0[76];
              double v597_data = ir2[4];
              ir2[4] = (v597_data + (v574_data * v595_data));
              double v600_data = s0[92];
              double v602_data = ir2[5];
              ir2[5] = (v602_data + (v574_data * v600_data));
              double v605_data = s0[108];
              double v607_data = ir2[6];
              ir2[6] = (v607_data + (v574_data * v605_data));
              double v610_data = s0[124];
              double v612_data = ir2[7];
              ir2[7] = (v612_data + (v574_data * v610_data));
            }
            if (v44_lead < 12) {
              double v618_data = r0[13];
              double v619_data = s0[13];
              double v621_data = ir2[0];
              ir2[0] = (v621_data + (v618_data * v619_data));
              double v624_data = s0[29];
              double v626_data = ir2[1];
              ir2[1] = (v626_data + (v618_data * v624_data));
              double v629_data = s0[45];
              double v631_data = ir2[2];
              ir2[2] = (v631_data + (v618_data * v629_data));
              double v634_data = s0[61];
              double v636_data = ir2[3];
              ir2[3] = (v636_data + (v618_data * v634_data));
              double v639_data = s0[77];
              double v641_data = ir2[4];
              ir2[4] = (v641_data + (v618_data * v639_data));
              double v644_data = s0[93];
              double v646_data = ir2[5];
              ir2[5] = (v646_data + (v618_data * v644_data));
              double v649_data = s0[109];
              double v651_data = ir2[6];
              ir2[6] = (v651_data + (v618_data * v649_data));
              double v654_data = s0[125];
              double v656_data = ir2[7];
              ir2[7] = (v656_data + (v618_data * v654_data));
            }
            if (v44_lead < 12) {
              double v662_data = r0[14];
              double v663_data = s0[14];
              double v665_data = ir2[0];
              ir2[0] = (v665_data + (v662_data * v663_data));
              double v668_data = s0[30];
              double v670_data = ir2[1];
              ir2[1] = (v670_data + (v662_data * v668_data));
              double v673_data = s0[46];
              double v675_data = ir2[2];
              ir2[2] = (v675_data + (v662_data * v673_data));
              double v678_data = s0[62];
              double v680_data = ir2[3];
              ir2[3] = (v680_data + (v662_data * v678_data));
              double v683_data = s0[78];
              double v685_data = ir2[4];
              ir2[4] = (v685_data + (v662_data * v683_data));
              double v688_data = s0[94];
              double v690_data = ir2[5];
              ir2[5] = (v690_data + (v662_data * v688_data));
              double v693_data = s0[110];
              double v695_data = ir2[6];
              ir2[6] = (v695_data + (v662_data * v693_data));
              double v698_data = s0[126];
              double v700_data = ir2[7];
              ir2[7] = (v700_data + (v662_data * v698_data));
            }
            if (v44_lead < 12) {
              double v706_data = r0[15];
              double v707_data = s0[15];
              double v709_data = ir2[0];
              ir2[0] = (v709_data + (v706_data * v707_data));
              double v712_data = s0[31];
              double v714_data = ir2[1];
              ir2[1] = (v714_data + (v706_data * v712_data));
              double v717_data = s0[47];
              double v719_data = ir2[2];
              ir2[2] = (v719_data + (v706_data * v717_data));
              double v722_data = s0[63];
              double v724_data = ir2[3];
              ir2[3] = (v724_data + (v706_data * v722_data));
              double v727_data = s0[79];
              double v729_data = ir2[4];
              ir2[4] = (v729_data + (v706_data * v727_data));
              double v732_data = s0[95];
              double v734_data = ir2[5];
              ir2[5] = (v734_data + (v706_data * v732_data));
              double v737_data = s0[111];
              double v739_data = ir2[6];
              ir2[6] = (v739_data + (v706_data * v737_data));
              double v742_data = s0[127];
              double v744_data = ir2[7];
              ir2[7] = (v744_data + (v706_data * v742_data));
            }
            if (v44_lead < 12) {
              #pragma unroll
              for (int32_t v750_n1 = 0; v750_n1 < 8; ++v750_n1) {
                int32_t v751_a = 0 + v750_n1;
                double v753_data = ir2[v750_n1];
                int32_t v754_a = 0 + v750_n1;
                double v756_data = r1[v750_n1];
                int32_t v758_a = 0 + v750_n1;
                r2[v750_n1] = (v756_data + v753_data);
              }
            }
          }
          // glb_m0 = store{r>g}(r2);
          int32_t v762_lead = threadIdx.x % 16;
          if (v762_lead < 12) {
            #pragma unroll
            for (int32_t v764_i1 = 0; v764_i1 < 8; ++v764_i1) {
              int32_t v765_a = 0 + v764_i1;
              double v767_data = r2[v764_i1];
              int32_t v774_a = v762_lead + (v764_i1 * 12);
              glb_m0[v774_a] = v767_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

