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
          int32_t v7_lead = threadIdx.x % 16;
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
              int32_t v15_a = v9_i1 * 12;
              int32_t v16_a = v7_lead + v15_a;
              double v24_data = __ldcg(&glb_m1[(v7_lead + v15_a)]);
              r0[v9_i1] = v24_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          {
            // s0 = load{g>s}(glb_m2[0, 1])
            #pragma unroll
            for (int32_t i = 0; i < 8; i += 1) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
              __pipeline_commit();
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
              int32_t v39_a = v33_i1 * 12;
              int32_t v40_a = v7_lead + v39_a;
              double v48_data = glb_m0[(v7_lead + v39_a)];
              r1[v33_i1] = v48_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          __syncwarp();
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          if (v7_lead < 12) {
            double v56_data = r0[0];
            double v57_data = s0[0];
            double v59_data = ir2[0];
            ir2[0] = (v59_data + (v56_data * v57_data));
            double v62_data = s0[16];
            double v64_data = ir2[1];
            ir2[1] = (v64_data + (v56_data * v62_data));
            double v67_data = s0[32];
            double v69_data = ir2[2];
            ir2[2] = (v69_data + (v56_data * v67_data));
            double v72_data = s0[48];
            double v74_data = ir2[3];
            ir2[3] = (v74_data + (v56_data * v72_data));
            double v77_data = s0[64];
            double v79_data = ir2[4];
            ir2[4] = (v79_data + (v56_data * v77_data));
            double v82_data = s0[80];
            double v84_data = ir2[5];
            ir2[5] = (v84_data + (v56_data * v82_data));
            double v87_data = s0[96];
            double v89_data = ir2[6];
            ir2[6] = (v89_data + (v56_data * v87_data));
            double v92_data = s0[112];
            double v94_data = ir2[7];
            ir2[7] = (v94_data + (v56_data * v92_data));
          }
          if (v7_lead < 12) {
            double v100_data = r0[1];
            double v101_data = s0[1];
            double v103_data = ir2[0];
            ir2[0] = (v103_data + (v100_data * v101_data));
            double v106_data = s0[17];
            double v108_data = ir2[1];
            ir2[1] = (v108_data + (v100_data * v106_data));
            double v111_data = s0[33];
            double v113_data = ir2[2];
            ir2[2] = (v113_data + (v100_data * v111_data));
            double v116_data = s0[49];
            double v118_data = ir2[3];
            ir2[3] = (v118_data + (v100_data * v116_data));
            double v121_data = s0[65];
            double v123_data = ir2[4];
            ir2[4] = (v123_data + (v100_data * v121_data));
            double v126_data = s0[81];
            double v128_data = ir2[5];
            ir2[5] = (v128_data + (v100_data * v126_data));
            double v131_data = s0[97];
            double v133_data = ir2[6];
            ir2[6] = (v133_data + (v100_data * v131_data));
            double v136_data = s0[113];
            double v138_data = ir2[7];
            ir2[7] = (v138_data + (v100_data * v136_data));
          }
          if (v7_lead < 12) {
            double v144_data = r0[2];
            double v145_data = s0[2];
            double v147_data = ir2[0];
            ir2[0] = (v147_data + (v144_data * v145_data));
            double v150_data = s0[18];
            double v152_data = ir2[1];
            ir2[1] = (v152_data + (v144_data * v150_data));
            double v155_data = s0[34];
            double v157_data = ir2[2];
            ir2[2] = (v157_data + (v144_data * v155_data));
            double v160_data = s0[50];
            double v162_data = ir2[3];
            ir2[3] = (v162_data + (v144_data * v160_data));
            double v165_data = s0[66];
            double v167_data = ir2[4];
            ir2[4] = (v167_data + (v144_data * v165_data));
            double v170_data = s0[82];
            double v172_data = ir2[5];
            ir2[5] = (v172_data + (v144_data * v170_data));
            double v175_data = s0[98];
            double v177_data = ir2[6];
            ir2[6] = (v177_data + (v144_data * v175_data));
            double v180_data = s0[114];
            double v182_data = ir2[7];
            ir2[7] = (v182_data + (v144_data * v180_data));
          }
          if (v7_lead < 12) {
            double v188_data = r0[3];
            double v189_data = s0[3];
            double v191_data = ir2[0];
            ir2[0] = (v191_data + (v188_data * v189_data));
            double v194_data = s0[19];
            double v196_data = ir2[1];
            ir2[1] = (v196_data + (v188_data * v194_data));
            double v199_data = s0[35];
            double v201_data = ir2[2];
            ir2[2] = (v201_data + (v188_data * v199_data));
            double v204_data = s0[51];
            double v206_data = ir2[3];
            ir2[3] = (v206_data + (v188_data * v204_data));
            double v209_data = s0[67];
            double v211_data = ir2[4];
            ir2[4] = (v211_data + (v188_data * v209_data));
            double v214_data = s0[83];
            double v216_data = ir2[5];
            ir2[5] = (v216_data + (v188_data * v214_data));
            double v219_data = s0[99];
            double v221_data = ir2[6];
            ir2[6] = (v221_data + (v188_data * v219_data));
            double v224_data = s0[115];
            double v226_data = ir2[7];
            ir2[7] = (v226_data + (v188_data * v224_data));
          }
          if (v7_lead < 12) {
            double v232_data = r0[4];
            double v233_data = s0[4];
            double v235_data = ir2[0];
            ir2[0] = (v235_data + (v232_data * v233_data));
            double v238_data = s0[20];
            double v240_data = ir2[1];
            ir2[1] = (v240_data + (v232_data * v238_data));
            double v243_data = s0[36];
            double v245_data = ir2[2];
            ir2[2] = (v245_data + (v232_data * v243_data));
            double v248_data = s0[52];
            double v250_data = ir2[3];
            ir2[3] = (v250_data + (v232_data * v248_data));
            double v253_data = s0[68];
            double v255_data = ir2[4];
            ir2[4] = (v255_data + (v232_data * v253_data));
            double v258_data = s0[84];
            double v260_data = ir2[5];
            ir2[5] = (v260_data + (v232_data * v258_data));
            double v263_data = s0[100];
            double v265_data = ir2[6];
            ir2[6] = (v265_data + (v232_data * v263_data));
            double v268_data = s0[116];
            double v270_data = ir2[7];
            ir2[7] = (v270_data + (v232_data * v268_data));
          }
          if (v7_lead < 12) {
            double v276_data = r0[5];
            double v277_data = s0[5];
            double v279_data = ir2[0];
            ir2[0] = (v279_data + (v276_data * v277_data));
            double v282_data = s0[21];
            double v284_data = ir2[1];
            ir2[1] = (v284_data + (v276_data * v282_data));
            double v287_data = s0[37];
            double v289_data = ir2[2];
            ir2[2] = (v289_data + (v276_data * v287_data));
            double v292_data = s0[53];
            double v294_data = ir2[3];
            ir2[3] = (v294_data + (v276_data * v292_data));
            double v297_data = s0[69];
            double v299_data = ir2[4];
            ir2[4] = (v299_data + (v276_data * v297_data));
            double v302_data = s0[85];
            double v304_data = ir2[5];
            ir2[5] = (v304_data + (v276_data * v302_data));
            double v307_data = s0[101];
            double v309_data = ir2[6];
            ir2[6] = (v309_data + (v276_data * v307_data));
            double v312_data = s0[117];
            double v314_data = ir2[7];
            ir2[7] = (v314_data + (v276_data * v312_data));
          }
          if (v7_lead < 12) {
            double v320_data = r0[6];
            double v321_data = s0[6];
            double v323_data = ir2[0];
            ir2[0] = (v323_data + (v320_data * v321_data));
            double v326_data = s0[22];
            double v328_data = ir2[1];
            ir2[1] = (v328_data + (v320_data * v326_data));
            double v331_data = s0[38];
            double v333_data = ir2[2];
            ir2[2] = (v333_data + (v320_data * v331_data));
            double v336_data = s0[54];
            double v338_data = ir2[3];
            ir2[3] = (v338_data + (v320_data * v336_data));
            double v341_data = s0[70];
            double v343_data = ir2[4];
            ir2[4] = (v343_data + (v320_data * v341_data));
            double v346_data = s0[86];
            double v348_data = ir2[5];
            ir2[5] = (v348_data + (v320_data * v346_data));
            double v351_data = s0[102];
            double v353_data = ir2[6];
            ir2[6] = (v353_data + (v320_data * v351_data));
            double v356_data = s0[118];
            double v358_data = ir2[7];
            ir2[7] = (v358_data + (v320_data * v356_data));
          }
          if (v7_lead < 12) {
            double v364_data = r0[7];
            double v365_data = s0[7];
            double v367_data = ir2[0];
            ir2[0] = (v367_data + (v364_data * v365_data));
            double v370_data = s0[23];
            double v372_data = ir2[1];
            ir2[1] = (v372_data + (v364_data * v370_data));
            double v375_data = s0[39];
            double v377_data = ir2[2];
            ir2[2] = (v377_data + (v364_data * v375_data));
            double v380_data = s0[55];
            double v382_data = ir2[3];
            ir2[3] = (v382_data + (v364_data * v380_data));
            double v385_data = s0[71];
            double v387_data = ir2[4];
            ir2[4] = (v387_data + (v364_data * v385_data));
            double v390_data = s0[87];
            double v392_data = ir2[5];
            ir2[5] = (v392_data + (v364_data * v390_data));
            double v395_data = s0[103];
            double v397_data = ir2[6];
            ir2[6] = (v397_data + (v364_data * v395_data));
            double v400_data = s0[119];
            double v402_data = ir2[7];
            ir2[7] = (v402_data + (v364_data * v400_data));
          }
          if (v7_lead < 12) {
            double v408_data = r0[8];
            double v409_data = s0[8];
            double v411_data = ir2[0];
            ir2[0] = (v411_data + (v408_data * v409_data));
            double v414_data = s0[24];
            double v416_data = ir2[1];
            ir2[1] = (v416_data + (v408_data * v414_data));
            double v419_data = s0[40];
            double v421_data = ir2[2];
            ir2[2] = (v421_data + (v408_data * v419_data));
            double v424_data = s0[56];
            double v426_data = ir2[3];
            ir2[3] = (v426_data + (v408_data * v424_data));
            double v429_data = s0[72];
            double v431_data = ir2[4];
            ir2[4] = (v431_data + (v408_data * v429_data));
            double v434_data = s0[88];
            double v436_data = ir2[5];
            ir2[5] = (v436_data + (v408_data * v434_data));
            double v439_data = s0[104];
            double v441_data = ir2[6];
            ir2[6] = (v441_data + (v408_data * v439_data));
            double v444_data = s0[120];
            double v446_data = ir2[7];
            ir2[7] = (v446_data + (v408_data * v444_data));
          }
          if (v7_lead < 12) {
            double v452_data = r0[9];
            double v453_data = s0[9];
            double v455_data = ir2[0];
            ir2[0] = (v455_data + (v452_data * v453_data));
            double v458_data = s0[25];
            double v460_data = ir2[1];
            ir2[1] = (v460_data + (v452_data * v458_data));
            double v463_data = s0[41];
            double v465_data = ir2[2];
            ir2[2] = (v465_data + (v452_data * v463_data));
            double v468_data = s0[57];
            double v470_data = ir2[3];
            ir2[3] = (v470_data + (v452_data * v468_data));
            double v473_data = s0[73];
            double v475_data = ir2[4];
            ir2[4] = (v475_data + (v452_data * v473_data));
            double v478_data = s0[89];
            double v480_data = ir2[5];
            ir2[5] = (v480_data + (v452_data * v478_data));
            double v483_data = s0[105];
            double v485_data = ir2[6];
            ir2[6] = (v485_data + (v452_data * v483_data));
            double v488_data = s0[121];
            double v490_data = ir2[7];
            ir2[7] = (v490_data + (v452_data * v488_data));
          }
          if (v7_lead < 12) {
            double v496_data = r0[10];
            double v497_data = s0[10];
            double v499_data = ir2[0];
            ir2[0] = (v499_data + (v496_data * v497_data));
            double v502_data = s0[26];
            double v504_data = ir2[1];
            ir2[1] = (v504_data + (v496_data * v502_data));
            double v507_data = s0[42];
            double v509_data = ir2[2];
            ir2[2] = (v509_data + (v496_data * v507_data));
            double v512_data = s0[58];
            double v514_data = ir2[3];
            ir2[3] = (v514_data + (v496_data * v512_data));
            double v517_data = s0[74];
            double v519_data = ir2[4];
            ir2[4] = (v519_data + (v496_data * v517_data));
            double v522_data = s0[90];
            double v524_data = ir2[5];
            ir2[5] = (v524_data + (v496_data * v522_data));
            double v527_data = s0[106];
            double v529_data = ir2[6];
            ir2[6] = (v529_data + (v496_data * v527_data));
            double v532_data = s0[122];
            double v534_data = ir2[7];
            ir2[7] = (v534_data + (v496_data * v532_data));
          }
          if (v7_lead < 12) {
            double v540_data = r0[11];
            double v541_data = s0[11];
            double v543_data = ir2[0];
            ir2[0] = (v543_data + (v540_data * v541_data));
            double v546_data = s0[27];
            double v548_data = ir2[1];
            ir2[1] = (v548_data + (v540_data * v546_data));
            double v551_data = s0[43];
            double v553_data = ir2[2];
            ir2[2] = (v553_data + (v540_data * v551_data));
            double v556_data = s0[59];
            double v558_data = ir2[3];
            ir2[3] = (v558_data + (v540_data * v556_data));
            double v561_data = s0[75];
            double v563_data = ir2[4];
            ir2[4] = (v563_data + (v540_data * v561_data));
            double v566_data = s0[91];
            double v568_data = ir2[5];
            ir2[5] = (v568_data + (v540_data * v566_data));
            double v571_data = s0[107];
            double v573_data = ir2[6];
            ir2[6] = (v573_data + (v540_data * v571_data));
            double v576_data = s0[123];
            double v578_data = ir2[7];
            ir2[7] = (v578_data + (v540_data * v576_data));
          }
          if (v7_lead < 12) {
            double v584_data = r0[12];
            double v585_data = s0[12];
            double v587_data = ir2[0];
            ir2[0] = (v587_data + (v584_data * v585_data));
            double v590_data = s0[28];
            double v592_data = ir2[1];
            ir2[1] = (v592_data + (v584_data * v590_data));
            double v595_data = s0[44];
            double v597_data = ir2[2];
            ir2[2] = (v597_data + (v584_data * v595_data));
            double v600_data = s0[60];
            double v602_data = ir2[3];
            ir2[3] = (v602_data + (v584_data * v600_data));
            double v605_data = s0[76];
            double v607_data = ir2[4];
            ir2[4] = (v607_data + (v584_data * v605_data));
            double v610_data = s0[92];
            double v612_data = ir2[5];
            ir2[5] = (v612_data + (v584_data * v610_data));
            double v615_data = s0[108];
            double v617_data = ir2[6];
            ir2[6] = (v617_data + (v584_data * v615_data));
            double v620_data = s0[124];
            double v622_data = ir2[7];
            ir2[7] = (v622_data + (v584_data * v620_data));
          }
          if (v7_lead < 12) {
            double v628_data = r0[13];
            double v629_data = s0[13];
            double v631_data = ir2[0];
            ir2[0] = (v631_data + (v628_data * v629_data));
            double v634_data = s0[29];
            double v636_data = ir2[1];
            ir2[1] = (v636_data + (v628_data * v634_data));
            double v639_data = s0[45];
            double v641_data = ir2[2];
            ir2[2] = (v641_data + (v628_data * v639_data));
            double v644_data = s0[61];
            double v646_data = ir2[3];
            ir2[3] = (v646_data + (v628_data * v644_data));
            double v649_data = s0[77];
            double v651_data = ir2[4];
            ir2[4] = (v651_data + (v628_data * v649_data));
            double v654_data = s0[93];
            double v656_data = ir2[5];
            ir2[5] = (v656_data + (v628_data * v654_data));
            double v659_data = s0[109];
            double v661_data = ir2[6];
            ir2[6] = (v661_data + (v628_data * v659_data));
            double v664_data = s0[125];
            double v666_data = ir2[7];
            ir2[7] = (v666_data + (v628_data * v664_data));
          }
          if (v7_lead < 12) {
            double v672_data = r0[14];
            double v673_data = s0[14];
            double v675_data = ir2[0];
            ir2[0] = (v675_data + (v672_data * v673_data));
            double v678_data = s0[30];
            double v680_data = ir2[1];
            ir2[1] = (v680_data + (v672_data * v678_data));
            double v683_data = s0[46];
            double v685_data = ir2[2];
            ir2[2] = (v685_data + (v672_data * v683_data));
            double v688_data = s0[62];
            double v690_data = ir2[3];
            ir2[3] = (v690_data + (v672_data * v688_data));
            double v693_data = s0[78];
            double v695_data = ir2[4];
            ir2[4] = (v695_data + (v672_data * v693_data));
            double v698_data = s0[94];
            double v700_data = ir2[5];
            ir2[5] = (v700_data + (v672_data * v698_data));
            double v703_data = s0[110];
            double v705_data = ir2[6];
            ir2[6] = (v705_data + (v672_data * v703_data));
            double v708_data = s0[126];
            double v710_data = ir2[7];
            ir2[7] = (v710_data + (v672_data * v708_data));
          }
          if (v7_lead < 12) {
            double v716_data = r0[15];
            double v717_data = s0[15];
            double v719_data = ir2[0];
            ir2[0] = (v719_data + (v716_data * v717_data));
            double v722_data = s0[31];
            double v724_data = ir2[1];
            ir2[1] = (v724_data + (v716_data * v722_data));
            double v727_data = s0[47];
            double v729_data = ir2[2];
            ir2[2] = (v729_data + (v716_data * v727_data));
            double v732_data = s0[63];
            double v734_data = ir2[3];
            ir2[3] = (v734_data + (v716_data * v732_data));
            double v737_data = s0[79];
            double v739_data = ir2[4];
            ir2[4] = (v739_data + (v716_data * v737_data));
            double v742_data = s0[95];
            double v744_data = ir2[5];
            ir2[5] = (v744_data + (v716_data * v742_data));
            double v747_data = s0[111];
            double v749_data = ir2[6];
            ir2[6] = (v749_data + (v716_data * v747_data));
            double v752_data = s0[127];
            double v754_data = ir2[7];
            ir2[7] = (v754_data + (v716_data * v752_data));
          }
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v760_n1 = 0; v760_n1 < 8; ++v760_n1) {
              int32_t v761_a = 0 + v760_n1;
              double v763_data = ir2[v760_n1];
              int32_t v764_a = 0 + v760_n1;
              double v766_data = r1[v760_n1];
              r2[v760_n1] = (v766_data + v763_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v7_lead < 12) {
            #pragma unroll
            for (int32_t v773_i1 = 0; v773_i1 < 8; ++v773_i1) {
              int32_t v774_a = 0 + v773_i1;
              double v776_data = r2[v773_i1];
              glb_m0[(v7_lead + (v773_i1 * 12))] = v776_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

