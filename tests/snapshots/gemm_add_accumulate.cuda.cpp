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
          int32_t v7_lead = threadIdx.x % 16;
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
              int32_t v15_a = v9_i1 * 12;
              int32_t v16_a = v7_lead + v15_a;
              float v24_data = __ldcg(&glb_m1[(v7_lead + v15_a)]);
              r0[v9_i1] = v24_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
              int32_t v39_a = v33_i1 * 12;
              int32_t v40_a = v7_lead + v39_a;
              float v48_data = glb_m0[(v7_lead + v39_a)];
              r1[v33_i1] = v48_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          float r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir2[8]{};
          if (v7_lead < 12) {
            float v56_data = r0[0];
            float v57_data = s0[0];
            float v59_data = ir2[0];
            ir2[0] = (v59_data + (v56_data * v57_data));
            float v62_data = s0[16];
            float v64_data = ir2[1];
            ir2[1] = (v64_data + (v56_data * v62_data));
            float v67_data = s0[32];
            float v69_data = ir2[2];
            ir2[2] = (v69_data + (v56_data * v67_data));
            float v72_data = s0[48];
            float v74_data = ir2[3];
            ir2[3] = (v74_data + (v56_data * v72_data));
            float v77_data = s0[64];
            float v79_data = ir2[4];
            ir2[4] = (v79_data + (v56_data * v77_data));
            float v82_data = s0[80];
            float v84_data = ir2[5];
            ir2[5] = (v84_data + (v56_data * v82_data));
            float v87_data = s0[96];
            float v89_data = ir2[6];
            ir2[6] = (v89_data + (v56_data * v87_data));
            float v92_data = s0[112];
            float v94_data = ir2[7];
            ir2[7] = (v94_data + (v56_data * v92_data));
          }
          if (v7_lead < 12) {
            float v100_data = r0[1];
            float v101_data = s0[1];
            float v103_data = ir2[0];
            ir2[0] = (v103_data + (v100_data * v101_data));
            float v106_data = s0[17];
            float v108_data = ir2[1];
            ir2[1] = (v108_data + (v100_data * v106_data));
            float v111_data = s0[33];
            float v113_data = ir2[2];
            ir2[2] = (v113_data + (v100_data * v111_data));
            float v116_data = s0[49];
            float v118_data = ir2[3];
            ir2[3] = (v118_data + (v100_data * v116_data));
            float v121_data = s0[65];
            float v123_data = ir2[4];
            ir2[4] = (v123_data + (v100_data * v121_data));
            float v126_data = s0[81];
            float v128_data = ir2[5];
            ir2[5] = (v128_data + (v100_data * v126_data));
            float v131_data = s0[97];
            float v133_data = ir2[6];
            ir2[6] = (v133_data + (v100_data * v131_data));
            float v136_data = s0[113];
            float v138_data = ir2[7];
            ir2[7] = (v138_data + (v100_data * v136_data));
          }
          if (v7_lead < 12) {
            float v144_data = r0[2];
            float v145_data = s0[2];
            float v147_data = ir2[0];
            ir2[0] = (v147_data + (v144_data * v145_data));
            float v150_data = s0[18];
            float v152_data = ir2[1];
            ir2[1] = (v152_data + (v144_data * v150_data));
            float v155_data = s0[34];
            float v157_data = ir2[2];
            ir2[2] = (v157_data + (v144_data * v155_data));
            float v160_data = s0[50];
            float v162_data = ir2[3];
            ir2[3] = (v162_data + (v144_data * v160_data));
            float v165_data = s0[66];
            float v167_data = ir2[4];
            ir2[4] = (v167_data + (v144_data * v165_data));
            float v170_data = s0[82];
            float v172_data = ir2[5];
            ir2[5] = (v172_data + (v144_data * v170_data));
            float v175_data = s0[98];
            float v177_data = ir2[6];
            ir2[6] = (v177_data + (v144_data * v175_data));
            float v180_data = s0[114];
            float v182_data = ir2[7];
            ir2[7] = (v182_data + (v144_data * v180_data));
          }
          if (v7_lead < 12) {
            float v188_data = r0[3];
            float v189_data = s0[3];
            float v191_data = ir2[0];
            ir2[0] = (v191_data + (v188_data * v189_data));
            float v194_data = s0[19];
            float v196_data = ir2[1];
            ir2[1] = (v196_data + (v188_data * v194_data));
            float v199_data = s0[35];
            float v201_data = ir2[2];
            ir2[2] = (v201_data + (v188_data * v199_data));
            float v204_data = s0[51];
            float v206_data = ir2[3];
            ir2[3] = (v206_data + (v188_data * v204_data));
            float v209_data = s0[67];
            float v211_data = ir2[4];
            ir2[4] = (v211_data + (v188_data * v209_data));
            float v214_data = s0[83];
            float v216_data = ir2[5];
            ir2[5] = (v216_data + (v188_data * v214_data));
            float v219_data = s0[99];
            float v221_data = ir2[6];
            ir2[6] = (v221_data + (v188_data * v219_data));
            float v224_data = s0[115];
            float v226_data = ir2[7];
            ir2[7] = (v226_data + (v188_data * v224_data));
          }
          if (v7_lead < 12) {
            float v232_data = r0[4];
            float v233_data = s0[4];
            float v235_data = ir2[0];
            ir2[0] = (v235_data + (v232_data * v233_data));
            float v238_data = s0[20];
            float v240_data = ir2[1];
            ir2[1] = (v240_data + (v232_data * v238_data));
            float v243_data = s0[36];
            float v245_data = ir2[2];
            ir2[2] = (v245_data + (v232_data * v243_data));
            float v248_data = s0[52];
            float v250_data = ir2[3];
            ir2[3] = (v250_data + (v232_data * v248_data));
            float v253_data = s0[68];
            float v255_data = ir2[4];
            ir2[4] = (v255_data + (v232_data * v253_data));
            float v258_data = s0[84];
            float v260_data = ir2[5];
            ir2[5] = (v260_data + (v232_data * v258_data));
            float v263_data = s0[100];
            float v265_data = ir2[6];
            ir2[6] = (v265_data + (v232_data * v263_data));
            float v268_data = s0[116];
            float v270_data = ir2[7];
            ir2[7] = (v270_data + (v232_data * v268_data));
          }
          if (v7_lead < 12) {
            float v276_data = r0[5];
            float v277_data = s0[5];
            float v279_data = ir2[0];
            ir2[0] = (v279_data + (v276_data * v277_data));
            float v282_data = s0[21];
            float v284_data = ir2[1];
            ir2[1] = (v284_data + (v276_data * v282_data));
            float v287_data = s0[37];
            float v289_data = ir2[2];
            ir2[2] = (v289_data + (v276_data * v287_data));
            float v292_data = s0[53];
            float v294_data = ir2[3];
            ir2[3] = (v294_data + (v276_data * v292_data));
            float v297_data = s0[69];
            float v299_data = ir2[4];
            ir2[4] = (v299_data + (v276_data * v297_data));
            float v302_data = s0[85];
            float v304_data = ir2[5];
            ir2[5] = (v304_data + (v276_data * v302_data));
            float v307_data = s0[101];
            float v309_data = ir2[6];
            ir2[6] = (v309_data + (v276_data * v307_data));
            float v312_data = s0[117];
            float v314_data = ir2[7];
            ir2[7] = (v314_data + (v276_data * v312_data));
          }
          if (v7_lead < 12) {
            float v320_data = r0[6];
            float v321_data = s0[6];
            float v323_data = ir2[0];
            ir2[0] = (v323_data + (v320_data * v321_data));
            float v326_data = s0[22];
            float v328_data = ir2[1];
            ir2[1] = (v328_data + (v320_data * v326_data));
            float v331_data = s0[38];
            float v333_data = ir2[2];
            ir2[2] = (v333_data + (v320_data * v331_data));
            float v336_data = s0[54];
            float v338_data = ir2[3];
            ir2[3] = (v338_data + (v320_data * v336_data));
            float v341_data = s0[70];
            float v343_data = ir2[4];
            ir2[4] = (v343_data + (v320_data * v341_data));
            float v346_data = s0[86];
            float v348_data = ir2[5];
            ir2[5] = (v348_data + (v320_data * v346_data));
            float v351_data = s0[102];
            float v353_data = ir2[6];
            ir2[6] = (v353_data + (v320_data * v351_data));
            float v356_data = s0[118];
            float v358_data = ir2[7];
            ir2[7] = (v358_data + (v320_data * v356_data));
          }
          if (v7_lead < 12) {
            float v364_data = r0[7];
            float v365_data = s0[7];
            float v367_data = ir2[0];
            ir2[0] = (v367_data + (v364_data * v365_data));
            float v370_data = s0[23];
            float v372_data = ir2[1];
            ir2[1] = (v372_data + (v364_data * v370_data));
            float v375_data = s0[39];
            float v377_data = ir2[2];
            ir2[2] = (v377_data + (v364_data * v375_data));
            float v380_data = s0[55];
            float v382_data = ir2[3];
            ir2[3] = (v382_data + (v364_data * v380_data));
            float v385_data = s0[71];
            float v387_data = ir2[4];
            ir2[4] = (v387_data + (v364_data * v385_data));
            float v390_data = s0[87];
            float v392_data = ir2[5];
            ir2[5] = (v392_data + (v364_data * v390_data));
            float v395_data = s0[103];
            float v397_data = ir2[6];
            ir2[6] = (v397_data + (v364_data * v395_data));
            float v400_data = s0[119];
            float v402_data = ir2[7];
            ir2[7] = (v402_data + (v364_data * v400_data));
          }
          if (v7_lead < 12) {
            float v408_data = r0[8];
            float v409_data = s0[8];
            float v411_data = ir2[0];
            ir2[0] = (v411_data + (v408_data * v409_data));
            float v414_data = s0[24];
            float v416_data = ir2[1];
            ir2[1] = (v416_data + (v408_data * v414_data));
            float v419_data = s0[40];
            float v421_data = ir2[2];
            ir2[2] = (v421_data + (v408_data * v419_data));
            float v424_data = s0[56];
            float v426_data = ir2[3];
            ir2[3] = (v426_data + (v408_data * v424_data));
            float v429_data = s0[72];
            float v431_data = ir2[4];
            ir2[4] = (v431_data + (v408_data * v429_data));
            float v434_data = s0[88];
            float v436_data = ir2[5];
            ir2[5] = (v436_data + (v408_data * v434_data));
            float v439_data = s0[104];
            float v441_data = ir2[6];
            ir2[6] = (v441_data + (v408_data * v439_data));
            float v444_data = s0[120];
            float v446_data = ir2[7];
            ir2[7] = (v446_data + (v408_data * v444_data));
          }
          if (v7_lead < 12) {
            float v452_data = r0[9];
            float v453_data = s0[9];
            float v455_data = ir2[0];
            ir2[0] = (v455_data + (v452_data * v453_data));
            float v458_data = s0[25];
            float v460_data = ir2[1];
            ir2[1] = (v460_data + (v452_data * v458_data));
            float v463_data = s0[41];
            float v465_data = ir2[2];
            ir2[2] = (v465_data + (v452_data * v463_data));
            float v468_data = s0[57];
            float v470_data = ir2[3];
            ir2[3] = (v470_data + (v452_data * v468_data));
            float v473_data = s0[73];
            float v475_data = ir2[4];
            ir2[4] = (v475_data + (v452_data * v473_data));
            float v478_data = s0[89];
            float v480_data = ir2[5];
            ir2[5] = (v480_data + (v452_data * v478_data));
            float v483_data = s0[105];
            float v485_data = ir2[6];
            ir2[6] = (v485_data + (v452_data * v483_data));
            float v488_data = s0[121];
            float v490_data = ir2[7];
            ir2[7] = (v490_data + (v452_data * v488_data));
          }
          if (v7_lead < 12) {
            float v496_data = r0[10];
            float v497_data = s0[10];
            float v499_data = ir2[0];
            ir2[0] = (v499_data + (v496_data * v497_data));
            float v502_data = s0[26];
            float v504_data = ir2[1];
            ir2[1] = (v504_data + (v496_data * v502_data));
            float v507_data = s0[42];
            float v509_data = ir2[2];
            ir2[2] = (v509_data + (v496_data * v507_data));
            float v512_data = s0[58];
            float v514_data = ir2[3];
            ir2[3] = (v514_data + (v496_data * v512_data));
            float v517_data = s0[74];
            float v519_data = ir2[4];
            ir2[4] = (v519_data + (v496_data * v517_data));
            float v522_data = s0[90];
            float v524_data = ir2[5];
            ir2[5] = (v524_data + (v496_data * v522_data));
            float v527_data = s0[106];
            float v529_data = ir2[6];
            ir2[6] = (v529_data + (v496_data * v527_data));
            float v532_data = s0[122];
            float v534_data = ir2[7];
            ir2[7] = (v534_data + (v496_data * v532_data));
          }
          if (v7_lead < 12) {
            float v540_data = r0[11];
            float v541_data = s0[11];
            float v543_data = ir2[0];
            ir2[0] = (v543_data + (v540_data * v541_data));
            float v546_data = s0[27];
            float v548_data = ir2[1];
            ir2[1] = (v548_data + (v540_data * v546_data));
            float v551_data = s0[43];
            float v553_data = ir2[2];
            ir2[2] = (v553_data + (v540_data * v551_data));
            float v556_data = s0[59];
            float v558_data = ir2[3];
            ir2[3] = (v558_data + (v540_data * v556_data));
            float v561_data = s0[75];
            float v563_data = ir2[4];
            ir2[4] = (v563_data + (v540_data * v561_data));
            float v566_data = s0[91];
            float v568_data = ir2[5];
            ir2[5] = (v568_data + (v540_data * v566_data));
            float v571_data = s0[107];
            float v573_data = ir2[6];
            ir2[6] = (v573_data + (v540_data * v571_data));
            float v576_data = s0[123];
            float v578_data = ir2[7];
            ir2[7] = (v578_data + (v540_data * v576_data));
          }
          if (v7_lead < 12) {
            float v584_data = r0[12];
            float v585_data = s0[12];
            float v587_data = ir2[0];
            ir2[0] = (v587_data + (v584_data * v585_data));
            float v590_data = s0[28];
            float v592_data = ir2[1];
            ir2[1] = (v592_data + (v584_data * v590_data));
            float v595_data = s0[44];
            float v597_data = ir2[2];
            ir2[2] = (v597_data + (v584_data * v595_data));
            float v600_data = s0[60];
            float v602_data = ir2[3];
            ir2[3] = (v602_data + (v584_data * v600_data));
            float v605_data = s0[76];
            float v607_data = ir2[4];
            ir2[4] = (v607_data + (v584_data * v605_data));
            float v610_data = s0[92];
            float v612_data = ir2[5];
            ir2[5] = (v612_data + (v584_data * v610_data));
            float v615_data = s0[108];
            float v617_data = ir2[6];
            ir2[6] = (v617_data + (v584_data * v615_data));
            float v620_data = s0[124];
            float v622_data = ir2[7];
            ir2[7] = (v622_data + (v584_data * v620_data));
          }
          if (v7_lead < 12) {
            float v628_data = r0[13];
            float v629_data = s0[13];
            float v631_data = ir2[0];
            ir2[0] = (v631_data + (v628_data * v629_data));
            float v634_data = s0[29];
            float v636_data = ir2[1];
            ir2[1] = (v636_data + (v628_data * v634_data));
            float v639_data = s0[45];
            float v641_data = ir2[2];
            ir2[2] = (v641_data + (v628_data * v639_data));
            float v644_data = s0[61];
            float v646_data = ir2[3];
            ir2[3] = (v646_data + (v628_data * v644_data));
            float v649_data = s0[77];
            float v651_data = ir2[4];
            ir2[4] = (v651_data + (v628_data * v649_data));
            float v654_data = s0[93];
            float v656_data = ir2[5];
            ir2[5] = (v656_data + (v628_data * v654_data));
            float v659_data = s0[109];
            float v661_data = ir2[6];
            ir2[6] = (v661_data + (v628_data * v659_data));
            float v664_data = s0[125];
            float v666_data = ir2[7];
            ir2[7] = (v666_data + (v628_data * v664_data));
          }
          if (v7_lead < 12) {
            float v672_data = r0[14];
            float v673_data = s0[14];
            float v675_data = ir2[0];
            ir2[0] = (v675_data + (v672_data * v673_data));
            float v678_data = s0[30];
            float v680_data = ir2[1];
            ir2[1] = (v680_data + (v672_data * v678_data));
            float v683_data = s0[46];
            float v685_data = ir2[2];
            ir2[2] = (v685_data + (v672_data * v683_data));
            float v688_data = s0[62];
            float v690_data = ir2[3];
            ir2[3] = (v690_data + (v672_data * v688_data));
            float v693_data = s0[78];
            float v695_data = ir2[4];
            ir2[4] = (v695_data + (v672_data * v693_data));
            float v698_data = s0[94];
            float v700_data = ir2[5];
            ir2[5] = (v700_data + (v672_data * v698_data));
            float v703_data = s0[110];
            float v705_data = ir2[6];
            ir2[6] = (v705_data + (v672_data * v703_data));
            float v708_data = s0[126];
            float v710_data = ir2[7];
            ir2[7] = (v710_data + (v672_data * v708_data));
          }
          if (v7_lead < 12) {
            float v716_data = r0[15];
            float v717_data = s0[15];
            float v719_data = ir2[0];
            ir2[0] = (v719_data + (v716_data * v717_data));
            float v722_data = s0[31];
            float v724_data = ir2[1];
            ir2[1] = (v724_data + (v716_data * v722_data));
            float v727_data = s0[47];
            float v729_data = ir2[2];
            ir2[2] = (v729_data + (v716_data * v727_data));
            float v732_data = s0[63];
            float v734_data = ir2[3];
            ir2[3] = (v734_data + (v716_data * v732_data));
            float v737_data = s0[79];
            float v739_data = ir2[4];
            ir2[4] = (v739_data + (v716_data * v737_data));
            float v742_data = s0[95];
            float v744_data = ir2[5];
            ir2[5] = (v744_data + (v716_data * v742_data));
            float v747_data = s0[111];
            float v749_data = ir2[6];
            ir2[6] = (v749_data + (v716_data * v747_data));
            float v752_data = s0[127];
            float v754_data = ir2[7];
            ir2[7] = (v754_data + (v716_data * v752_data));
          }
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v760_n1 = 0; v760_n1 < 8; ++v760_n1) {
              int32_t v761_a = 0 + v760_n1;
              float v763_data = ir2[v760_n1];
              int32_t v764_a = 0 + v760_n1;
              float v766_data = r1[v760_n1];
              r2[v760_n1] = (v766_data + v763_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v773_i1 = 0; v773_i1 < 8; ++v773_i1) {
              int32_t v774_a = 0 + v773_i1;
              float v776_data = r2[v773_i1];
              glb_m0[(v7_lead + (v773_i1 * 12))] = v776_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

