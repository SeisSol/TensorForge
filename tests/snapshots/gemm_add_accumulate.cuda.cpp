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
          int32_t v2_lead = threadIdx.x % 16;
          if (v2_lead < 12) {
            #pragma unroll
            for (int32_t v4_i1 = 0; v4_i1 < 16; ++v4_i1) {
              int32_t v10_a = v4_i1 * 12;
              int32_t v11_a = v2_lead + v10_a;
              float v19_data = __ldcg(&glb_m1[(v2_lead + v10_a)]);
              int32_t v20_a = 0 + v4_i1;
              r0[v20_a] = v19_data;
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
          int32_t v23_lead = threadIdx.x % 16;
          if (v23_lead < 12) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
              int32_t v31_a = v25_i1 * 12;
              int32_t v32_a = v23_lead + v31_a;
              float v40_data = glb_m0[(v23_lead + v31_a)];
              int32_t v41_a = 0 + v25_i1;
              r1[v41_a] = v40_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          pipeline.consumer_wait();
          pipeline.consumer_release();
          // wait(r1 = load{g>r}(glb_m0););
          float r2[8]{};
          __syncwarp();
          {
            // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
            // [(0, 12), (0, 8)] [(0, 16)]
            float ir2[8]{};
            int32_t v44_lead = threadIdx.x % 16;
            if (v44_lead < 12) {
              float v46_data = r0[0];
              float v47_data = s0[0];
              float v49_data = ir2[0];
              ir2[0] = (v49_data + (v46_data * v47_data));
              float v52_data = s0[16];
              float v54_data = ir2[1];
              ir2[1] = (v54_data + (v46_data * v52_data));
              float v57_data = s0[32];
              float v59_data = ir2[2];
              ir2[2] = (v59_data + (v46_data * v57_data));
              float v62_data = s0[48];
              float v64_data = ir2[3];
              ir2[3] = (v64_data + (v46_data * v62_data));
              float v67_data = s0[64];
              float v69_data = ir2[4];
              ir2[4] = (v69_data + (v46_data * v67_data));
              float v72_data = s0[80];
              float v74_data = ir2[5];
              ir2[5] = (v74_data + (v46_data * v72_data));
              float v77_data = s0[96];
              float v79_data = ir2[6];
              ir2[6] = (v79_data + (v46_data * v77_data));
              float v82_data = s0[112];
              float v84_data = ir2[7];
              ir2[7] = (v84_data + (v46_data * v82_data));
            }
            if (v44_lead < 12) {
              float v90_data = r0[1];
              float v91_data = s0[1];
              float v93_data = ir2[0];
              ir2[0] = (v93_data + (v90_data * v91_data));
              float v96_data = s0[17];
              float v98_data = ir2[1];
              ir2[1] = (v98_data + (v90_data * v96_data));
              float v101_data = s0[33];
              float v103_data = ir2[2];
              ir2[2] = (v103_data + (v90_data * v101_data));
              float v106_data = s0[49];
              float v108_data = ir2[3];
              ir2[3] = (v108_data + (v90_data * v106_data));
              float v111_data = s0[65];
              float v113_data = ir2[4];
              ir2[4] = (v113_data + (v90_data * v111_data));
              float v116_data = s0[81];
              float v118_data = ir2[5];
              ir2[5] = (v118_data + (v90_data * v116_data));
              float v121_data = s0[97];
              float v123_data = ir2[6];
              ir2[6] = (v123_data + (v90_data * v121_data));
              float v126_data = s0[113];
              float v128_data = ir2[7];
              ir2[7] = (v128_data + (v90_data * v126_data));
            }
            if (v44_lead < 12) {
              float v134_data = r0[2];
              float v135_data = s0[2];
              float v137_data = ir2[0];
              ir2[0] = (v137_data + (v134_data * v135_data));
              float v140_data = s0[18];
              float v142_data = ir2[1];
              ir2[1] = (v142_data + (v134_data * v140_data));
              float v145_data = s0[34];
              float v147_data = ir2[2];
              ir2[2] = (v147_data + (v134_data * v145_data));
              float v150_data = s0[50];
              float v152_data = ir2[3];
              ir2[3] = (v152_data + (v134_data * v150_data));
              float v155_data = s0[66];
              float v157_data = ir2[4];
              ir2[4] = (v157_data + (v134_data * v155_data));
              float v160_data = s0[82];
              float v162_data = ir2[5];
              ir2[5] = (v162_data + (v134_data * v160_data));
              float v165_data = s0[98];
              float v167_data = ir2[6];
              ir2[6] = (v167_data + (v134_data * v165_data));
              float v170_data = s0[114];
              float v172_data = ir2[7];
              ir2[7] = (v172_data + (v134_data * v170_data));
            }
            if (v44_lead < 12) {
              float v178_data = r0[3];
              float v179_data = s0[3];
              float v181_data = ir2[0];
              ir2[0] = (v181_data + (v178_data * v179_data));
              float v184_data = s0[19];
              float v186_data = ir2[1];
              ir2[1] = (v186_data + (v178_data * v184_data));
              float v189_data = s0[35];
              float v191_data = ir2[2];
              ir2[2] = (v191_data + (v178_data * v189_data));
              float v194_data = s0[51];
              float v196_data = ir2[3];
              ir2[3] = (v196_data + (v178_data * v194_data));
              float v199_data = s0[67];
              float v201_data = ir2[4];
              ir2[4] = (v201_data + (v178_data * v199_data));
              float v204_data = s0[83];
              float v206_data = ir2[5];
              ir2[5] = (v206_data + (v178_data * v204_data));
              float v209_data = s0[99];
              float v211_data = ir2[6];
              ir2[6] = (v211_data + (v178_data * v209_data));
              float v214_data = s0[115];
              float v216_data = ir2[7];
              ir2[7] = (v216_data + (v178_data * v214_data));
            }
            if (v44_lead < 12) {
              float v222_data = r0[4];
              float v223_data = s0[4];
              float v225_data = ir2[0];
              ir2[0] = (v225_data + (v222_data * v223_data));
              float v228_data = s0[20];
              float v230_data = ir2[1];
              ir2[1] = (v230_data + (v222_data * v228_data));
              float v233_data = s0[36];
              float v235_data = ir2[2];
              ir2[2] = (v235_data + (v222_data * v233_data));
              float v238_data = s0[52];
              float v240_data = ir2[3];
              ir2[3] = (v240_data + (v222_data * v238_data));
              float v243_data = s0[68];
              float v245_data = ir2[4];
              ir2[4] = (v245_data + (v222_data * v243_data));
              float v248_data = s0[84];
              float v250_data = ir2[5];
              ir2[5] = (v250_data + (v222_data * v248_data));
              float v253_data = s0[100];
              float v255_data = ir2[6];
              ir2[6] = (v255_data + (v222_data * v253_data));
              float v258_data = s0[116];
              float v260_data = ir2[7];
              ir2[7] = (v260_data + (v222_data * v258_data));
            }
            if (v44_lead < 12) {
              float v266_data = r0[5];
              float v267_data = s0[5];
              float v269_data = ir2[0];
              ir2[0] = (v269_data + (v266_data * v267_data));
              float v272_data = s0[21];
              float v274_data = ir2[1];
              ir2[1] = (v274_data + (v266_data * v272_data));
              float v277_data = s0[37];
              float v279_data = ir2[2];
              ir2[2] = (v279_data + (v266_data * v277_data));
              float v282_data = s0[53];
              float v284_data = ir2[3];
              ir2[3] = (v284_data + (v266_data * v282_data));
              float v287_data = s0[69];
              float v289_data = ir2[4];
              ir2[4] = (v289_data + (v266_data * v287_data));
              float v292_data = s0[85];
              float v294_data = ir2[5];
              ir2[5] = (v294_data + (v266_data * v292_data));
              float v297_data = s0[101];
              float v299_data = ir2[6];
              ir2[6] = (v299_data + (v266_data * v297_data));
              float v302_data = s0[117];
              float v304_data = ir2[7];
              ir2[7] = (v304_data + (v266_data * v302_data));
            }
            if (v44_lead < 12) {
              float v310_data = r0[6];
              float v311_data = s0[6];
              float v313_data = ir2[0];
              ir2[0] = (v313_data + (v310_data * v311_data));
              float v316_data = s0[22];
              float v318_data = ir2[1];
              ir2[1] = (v318_data + (v310_data * v316_data));
              float v321_data = s0[38];
              float v323_data = ir2[2];
              ir2[2] = (v323_data + (v310_data * v321_data));
              float v326_data = s0[54];
              float v328_data = ir2[3];
              ir2[3] = (v328_data + (v310_data * v326_data));
              float v331_data = s0[70];
              float v333_data = ir2[4];
              ir2[4] = (v333_data + (v310_data * v331_data));
              float v336_data = s0[86];
              float v338_data = ir2[5];
              ir2[5] = (v338_data + (v310_data * v336_data));
              float v341_data = s0[102];
              float v343_data = ir2[6];
              ir2[6] = (v343_data + (v310_data * v341_data));
              float v346_data = s0[118];
              float v348_data = ir2[7];
              ir2[7] = (v348_data + (v310_data * v346_data));
            }
            if (v44_lead < 12) {
              float v354_data = r0[7];
              float v355_data = s0[7];
              float v357_data = ir2[0];
              ir2[0] = (v357_data + (v354_data * v355_data));
              float v360_data = s0[23];
              float v362_data = ir2[1];
              ir2[1] = (v362_data + (v354_data * v360_data));
              float v365_data = s0[39];
              float v367_data = ir2[2];
              ir2[2] = (v367_data + (v354_data * v365_data));
              float v370_data = s0[55];
              float v372_data = ir2[3];
              ir2[3] = (v372_data + (v354_data * v370_data));
              float v375_data = s0[71];
              float v377_data = ir2[4];
              ir2[4] = (v377_data + (v354_data * v375_data));
              float v380_data = s0[87];
              float v382_data = ir2[5];
              ir2[5] = (v382_data + (v354_data * v380_data));
              float v385_data = s0[103];
              float v387_data = ir2[6];
              ir2[6] = (v387_data + (v354_data * v385_data));
              float v390_data = s0[119];
              float v392_data = ir2[7];
              ir2[7] = (v392_data + (v354_data * v390_data));
            }
            if (v44_lead < 12) {
              float v398_data = r0[8];
              float v399_data = s0[8];
              float v401_data = ir2[0];
              ir2[0] = (v401_data + (v398_data * v399_data));
              float v404_data = s0[24];
              float v406_data = ir2[1];
              ir2[1] = (v406_data + (v398_data * v404_data));
              float v409_data = s0[40];
              float v411_data = ir2[2];
              ir2[2] = (v411_data + (v398_data * v409_data));
              float v414_data = s0[56];
              float v416_data = ir2[3];
              ir2[3] = (v416_data + (v398_data * v414_data));
              float v419_data = s0[72];
              float v421_data = ir2[4];
              ir2[4] = (v421_data + (v398_data * v419_data));
              float v424_data = s0[88];
              float v426_data = ir2[5];
              ir2[5] = (v426_data + (v398_data * v424_data));
              float v429_data = s0[104];
              float v431_data = ir2[6];
              ir2[6] = (v431_data + (v398_data * v429_data));
              float v434_data = s0[120];
              float v436_data = ir2[7];
              ir2[7] = (v436_data + (v398_data * v434_data));
            }
            if (v44_lead < 12) {
              float v442_data = r0[9];
              float v443_data = s0[9];
              float v445_data = ir2[0];
              ir2[0] = (v445_data + (v442_data * v443_data));
              float v448_data = s0[25];
              float v450_data = ir2[1];
              ir2[1] = (v450_data + (v442_data * v448_data));
              float v453_data = s0[41];
              float v455_data = ir2[2];
              ir2[2] = (v455_data + (v442_data * v453_data));
              float v458_data = s0[57];
              float v460_data = ir2[3];
              ir2[3] = (v460_data + (v442_data * v458_data));
              float v463_data = s0[73];
              float v465_data = ir2[4];
              ir2[4] = (v465_data + (v442_data * v463_data));
              float v468_data = s0[89];
              float v470_data = ir2[5];
              ir2[5] = (v470_data + (v442_data * v468_data));
              float v473_data = s0[105];
              float v475_data = ir2[6];
              ir2[6] = (v475_data + (v442_data * v473_data));
              float v478_data = s0[121];
              float v480_data = ir2[7];
              ir2[7] = (v480_data + (v442_data * v478_data));
            }
            if (v44_lead < 12) {
              float v486_data = r0[10];
              float v487_data = s0[10];
              float v489_data = ir2[0];
              ir2[0] = (v489_data + (v486_data * v487_data));
              float v492_data = s0[26];
              float v494_data = ir2[1];
              ir2[1] = (v494_data + (v486_data * v492_data));
              float v497_data = s0[42];
              float v499_data = ir2[2];
              ir2[2] = (v499_data + (v486_data * v497_data));
              float v502_data = s0[58];
              float v504_data = ir2[3];
              ir2[3] = (v504_data + (v486_data * v502_data));
              float v507_data = s0[74];
              float v509_data = ir2[4];
              ir2[4] = (v509_data + (v486_data * v507_data));
              float v512_data = s0[90];
              float v514_data = ir2[5];
              ir2[5] = (v514_data + (v486_data * v512_data));
              float v517_data = s0[106];
              float v519_data = ir2[6];
              ir2[6] = (v519_data + (v486_data * v517_data));
              float v522_data = s0[122];
              float v524_data = ir2[7];
              ir2[7] = (v524_data + (v486_data * v522_data));
            }
            if (v44_lead < 12) {
              float v530_data = r0[11];
              float v531_data = s0[11];
              float v533_data = ir2[0];
              ir2[0] = (v533_data + (v530_data * v531_data));
              float v536_data = s0[27];
              float v538_data = ir2[1];
              ir2[1] = (v538_data + (v530_data * v536_data));
              float v541_data = s0[43];
              float v543_data = ir2[2];
              ir2[2] = (v543_data + (v530_data * v541_data));
              float v546_data = s0[59];
              float v548_data = ir2[3];
              ir2[3] = (v548_data + (v530_data * v546_data));
              float v551_data = s0[75];
              float v553_data = ir2[4];
              ir2[4] = (v553_data + (v530_data * v551_data));
              float v556_data = s0[91];
              float v558_data = ir2[5];
              ir2[5] = (v558_data + (v530_data * v556_data));
              float v561_data = s0[107];
              float v563_data = ir2[6];
              ir2[6] = (v563_data + (v530_data * v561_data));
              float v566_data = s0[123];
              float v568_data = ir2[7];
              ir2[7] = (v568_data + (v530_data * v566_data));
            }
            if (v44_lead < 12) {
              float v574_data = r0[12];
              float v575_data = s0[12];
              float v577_data = ir2[0];
              ir2[0] = (v577_data + (v574_data * v575_data));
              float v580_data = s0[28];
              float v582_data = ir2[1];
              ir2[1] = (v582_data + (v574_data * v580_data));
              float v585_data = s0[44];
              float v587_data = ir2[2];
              ir2[2] = (v587_data + (v574_data * v585_data));
              float v590_data = s0[60];
              float v592_data = ir2[3];
              ir2[3] = (v592_data + (v574_data * v590_data));
              float v595_data = s0[76];
              float v597_data = ir2[4];
              ir2[4] = (v597_data + (v574_data * v595_data));
              float v600_data = s0[92];
              float v602_data = ir2[5];
              ir2[5] = (v602_data + (v574_data * v600_data));
              float v605_data = s0[108];
              float v607_data = ir2[6];
              ir2[6] = (v607_data + (v574_data * v605_data));
              float v610_data = s0[124];
              float v612_data = ir2[7];
              ir2[7] = (v612_data + (v574_data * v610_data));
            }
            if (v44_lead < 12) {
              float v618_data = r0[13];
              float v619_data = s0[13];
              float v621_data = ir2[0];
              ir2[0] = (v621_data + (v618_data * v619_data));
              float v624_data = s0[29];
              float v626_data = ir2[1];
              ir2[1] = (v626_data + (v618_data * v624_data));
              float v629_data = s0[45];
              float v631_data = ir2[2];
              ir2[2] = (v631_data + (v618_data * v629_data));
              float v634_data = s0[61];
              float v636_data = ir2[3];
              ir2[3] = (v636_data + (v618_data * v634_data));
              float v639_data = s0[77];
              float v641_data = ir2[4];
              ir2[4] = (v641_data + (v618_data * v639_data));
              float v644_data = s0[93];
              float v646_data = ir2[5];
              ir2[5] = (v646_data + (v618_data * v644_data));
              float v649_data = s0[109];
              float v651_data = ir2[6];
              ir2[6] = (v651_data + (v618_data * v649_data));
              float v654_data = s0[125];
              float v656_data = ir2[7];
              ir2[7] = (v656_data + (v618_data * v654_data));
            }
            if (v44_lead < 12) {
              float v662_data = r0[14];
              float v663_data = s0[14];
              float v665_data = ir2[0];
              ir2[0] = (v665_data + (v662_data * v663_data));
              float v668_data = s0[30];
              float v670_data = ir2[1];
              ir2[1] = (v670_data + (v662_data * v668_data));
              float v673_data = s0[46];
              float v675_data = ir2[2];
              ir2[2] = (v675_data + (v662_data * v673_data));
              float v678_data = s0[62];
              float v680_data = ir2[3];
              ir2[3] = (v680_data + (v662_data * v678_data));
              float v683_data = s0[78];
              float v685_data = ir2[4];
              ir2[4] = (v685_data + (v662_data * v683_data));
              float v688_data = s0[94];
              float v690_data = ir2[5];
              ir2[5] = (v690_data + (v662_data * v688_data));
              float v693_data = s0[110];
              float v695_data = ir2[6];
              ir2[6] = (v695_data + (v662_data * v693_data));
              float v698_data = s0[126];
              float v700_data = ir2[7];
              ir2[7] = (v700_data + (v662_data * v698_data));
            }
            if (v44_lead < 12) {
              float v706_data = r0[15];
              float v707_data = s0[15];
              float v709_data = ir2[0];
              ir2[0] = (v709_data + (v706_data * v707_data));
              float v712_data = s0[31];
              float v714_data = ir2[1];
              ir2[1] = (v714_data + (v706_data * v712_data));
              float v717_data = s0[47];
              float v719_data = ir2[2];
              ir2[2] = (v719_data + (v706_data * v717_data));
              float v722_data = s0[63];
              float v724_data = ir2[3];
              ir2[3] = (v724_data + (v706_data * v722_data));
              float v727_data = s0[79];
              float v729_data = ir2[4];
              ir2[4] = (v729_data + (v706_data * v727_data));
              float v732_data = s0[95];
              float v734_data = ir2[5];
              ir2[5] = (v734_data + (v706_data * v732_data));
              float v737_data = s0[111];
              float v739_data = ir2[6];
              ir2[6] = (v739_data + (v706_data * v737_data));
              float v742_data = s0[127];
              float v744_data = ir2[7];
              ir2[7] = (v744_data + (v706_data * v742_data));
            }
            if (v44_lead < 12) {
              #pragma unroll
              for (int32_t v750_n1 = 0; v750_n1 < 8; ++v750_n1) {
                int32_t v751_a = 0 + v750_n1;
                float v753_data = ir2[v750_n1];
                int32_t v754_a = 0 + v750_n1;
                float v756_data = r1[v750_n1];
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
              float v767_data = r2[v764_i1];
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

