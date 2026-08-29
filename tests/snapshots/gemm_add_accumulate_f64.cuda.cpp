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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 16;
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              double v23_data = __ldcg(&glb_m1[(v13_lead + (v15_i1 * 12))]);
              r0[v15_i1] = v23_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
              double v40_data = glb_m0[(v13_lead + (v32_i1 * 12))];
              r1[v32_i1] = v40_data;
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
          if (v13_lead < 12) {
            double v48_data = r0[0];
            double v49_data = s0[0];
            double v51_data = ir2[0];
            ir2[0] = (v51_data + (v48_data * v49_data));
            double v54_data = s0[16];
            double v56_data = ir2[1];
            ir2[1] = (v56_data + (v48_data * v54_data));
            double v59_data = s0[33];
            double v61_data = ir2[2];
            ir2[2] = (v61_data + (v48_data * v59_data));
            double v64_data = s0[49];
            double v66_data = ir2[3];
            ir2[3] = (v66_data + (v48_data * v64_data));
            double v69_data = s0[66];
            double v71_data = ir2[4];
            ir2[4] = (v71_data + (v48_data * v69_data));
            double v74_data = s0[82];
            double v76_data = ir2[5];
            ir2[5] = (v76_data + (v48_data * v74_data));
            double v79_data = s0[99];
            double v81_data = ir2[6];
            ir2[6] = (v81_data + (v48_data * v79_data));
            double v84_data = s0[115];
            double v86_data = ir2[7];
            ir2[7] = (v86_data + (v48_data * v84_data));
          }
          if (v13_lead < 12) {
            double v92_data = r0[1];
            double v93_data = s0[1];
            double v95_data = ir2[0];
            ir2[0] = (v95_data + (v92_data * v93_data));
            double v98_data = s0[17];
            double v100_data = ir2[1];
            ir2[1] = (v100_data + (v92_data * v98_data));
            double v103_data = s0[32];
            double v105_data = ir2[2];
            ir2[2] = (v105_data + (v92_data * v103_data));
            double v108_data = s0[48];
            double v110_data = ir2[3];
            ir2[3] = (v110_data + (v92_data * v108_data));
            double v113_data = s0[67];
            double v115_data = ir2[4];
            ir2[4] = (v115_data + (v92_data * v113_data));
            double v118_data = s0[83];
            double v120_data = ir2[5];
            ir2[5] = (v120_data + (v92_data * v118_data));
            double v123_data = s0[98];
            double v125_data = ir2[6];
            ir2[6] = (v125_data + (v92_data * v123_data));
            double v128_data = s0[114];
            double v130_data = ir2[7];
            ir2[7] = (v130_data + (v92_data * v128_data));
          }
          if (v13_lead < 12) {
            double v136_data = r0[2];
            double v137_data = s0[2];
            double v139_data = ir2[0];
            ir2[0] = (v139_data + (v136_data * v137_data));
            double v142_data = s0[18];
            double v144_data = ir2[1];
            ir2[1] = (v144_data + (v136_data * v142_data));
            double v147_data = s0[35];
            double v149_data = ir2[2];
            ir2[2] = (v149_data + (v136_data * v147_data));
            double v152_data = s0[51];
            double v154_data = ir2[3];
            ir2[3] = (v154_data + (v136_data * v152_data));
            double v157_data = s0[64];
            double v159_data = ir2[4];
            ir2[4] = (v159_data + (v136_data * v157_data));
            double v162_data = s0[80];
            double v164_data = ir2[5];
            ir2[5] = (v164_data + (v136_data * v162_data));
            double v167_data = s0[97];
            double v169_data = ir2[6];
            ir2[6] = (v169_data + (v136_data * v167_data));
            double v172_data = s0[113];
            double v174_data = ir2[7];
            ir2[7] = (v174_data + (v136_data * v172_data));
          }
          if (v13_lead < 12) {
            double v180_data = r0[3];
            double v181_data = s0[3];
            double v183_data = ir2[0];
            ir2[0] = (v183_data + (v180_data * v181_data));
            double v186_data = s0[19];
            double v188_data = ir2[1];
            ir2[1] = (v188_data + (v180_data * v186_data));
            double v191_data = s0[34];
            double v193_data = ir2[2];
            ir2[2] = (v193_data + (v180_data * v191_data));
            double v196_data = s0[50];
            double v198_data = ir2[3];
            ir2[3] = (v198_data + (v180_data * v196_data));
            double v201_data = s0[65];
            double v203_data = ir2[4];
            ir2[4] = (v203_data + (v180_data * v201_data));
            double v206_data = s0[81];
            double v208_data = ir2[5];
            ir2[5] = (v208_data + (v180_data * v206_data));
            double v211_data = s0[96];
            double v213_data = ir2[6];
            ir2[6] = (v213_data + (v180_data * v211_data));
            double v216_data = s0[112];
            double v218_data = ir2[7];
            ir2[7] = (v218_data + (v180_data * v216_data));
          }
          if (v13_lead < 12) {
            double v224_data = r0[4];
            double v225_data = s0[4];
            double v227_data = ir2[0];
            ir2[0] = (v227_data + (v224_data * v225_data));
            double v230_data = s0[20];
            double v232_data = ir2[1];
            ir2[1] = (v232_data + (v224_data * v230_data));
            double v235_data = s0[37];
            double v237_data = ir2[2];
            ir2[2] = (v237_data + (v224_data * v235_data));
            double v240_data = s0[53];
            double v242_data = ir2[3];
            ir2[3] = (v242_data + (v224_data * v240_data));
            double v245_data = s0[70];
            double v247_data = ir2[4];
            ir2[4] = (v247_data + (v224_data * v245_data));
            double v250_data = s0[86];
            double v252_data = ir2[5];
            ir2[5] = (v252_data + (v224_data * v250_data));
            double v255_data = s0[103];
            double v257_data = ir2[6];
            ir2[6] = (v257_data + (v224_data * v255_data));
            double v260_data = s0[119];
            double v262_data = ir2[7];
            ir2[7] = (v262_data + (v224_data * v260_data));
          }
          if (v13_lead < 12) {
            double v268_data = r0[5];
            double v269_data = s0[5];
            double v271_data = ir2[0];
            ir2[0] = (v271_data + (v268_data * v269_data));
            double v274_data = s0[21];
            double v276_data = ir2[1];
            ir2[1] = (v276_data + (v268_data * v274_data));
            double v279_data = s0[36];
            double v281_data = ir2[2];
            ir2[2] = (v281_data + (v268_data * v279_data));
            double v284_data = s0[52];
            double v286_data = ir2[3];
            ir2[3] = (v286_data + (v268_data * v284_data));
            double v289_data = s0[71];
            double v291_data = ir2[4];
            ir2[4] = (v291_data + (v268_data * v289_data));
            double v294_data = s0[87];
            double v296_data = ir2[5];
            ir2[5] = (v296_data + (v268_data * v294_data));
            double v299_data = s0[102];
            double v301_data = ir2[6];
            ir2[6] = (v301_data + (v268_data * v299_data));
            double v304_data = s0[118];
            double v306_data = ir2[7];
            ir2[7] = (v306_data + (v268_data * v304_data));
          }
          if (v13_lead < 12) {
            double v312_data = r0[6];
            double v313_data = s0[6];
            double v315_data = ir2[0];
            ir2[0] = (v315_data + (v312_data * v313_data));
            double v318_data = s0[22];
            double v320_data = ir2[1];
            ir2[1] = (v320_data + (v312_data * v318_data));
            double v323_data = s0[39];
            double v325_data = ir2[2];
            ir2[2] = (v325_data + (v312_data * v323_data));
            double v328_data = s0[55];
            double v330_data = ir2[3];
            ir2[3] = (v330_data + (v312_data * v328_data));
            double v333_data = s0[68];
            double v335_data = ir2[4];
            ir2[4] = (v335_data + (v312_data * v333_data));
            double v338_data = s0[84];
            double v340_data = ir2[5];
            ir2[5] = (v340_data + (v312_data * v338_data));
            double v343_data = s0[101];
            double v345_data = ir2[6];
            ir2[6] = (v345_data + (v312_data * v343_data));
            double v348_data = s0[117];
            double v350_data = ir2[7];
            ir2[7] = (v350_data + (v312_data * v348_data));
          }
          if (v13_lead < 12) {
            double v356_data = r0[7];
            double v357_data = s0[7];
            double v359_data = ir2[0];
            ir2[0] = (v359_data + (v356_data * v357_data));
            double v362_data = s0[23];
            double v364_data = ir2[1];
            ir2[1] = (v364_data + (v356_data * v362_data));
            double v367_data = s0[38];
            double v369_data = ir2[2];
            ir2[2] = (v369_data + (v356_data * v367_data));
            double v372_data = s0[54];
            double v374_data = ir2[3];
            ir2[3] = (v374_data + (v356_data * v372_data));
            double v377_data = s0[69];
            double v379_data = ir2[4];
            ir2[4] = (v379_data + (v356_data * v377_data));
            double v382_data = s0[85];
            double v384_data = ir2[5];
            ir2[5] = (v384_data + (v356_data * v382_data));
            double v387_data = s0[100];
            double v389_data = ir2[6];
            ir2[6] = (v389_data + (v356_data * v387_data));
            double v392_data = s0[116];
            double v394_data = ir2[7];
            ir2[7] = (v394_data + (v356_data * v392_data));
          }
          if (v13_lead < 12) {
            double v400_data = r0[8];
            double v401_data = s0[8];
            double v403_data = ir2[0];
            ir2[0] = (v403_data + (v400_data * v401_data));
            double v406_data = s0[24];
            double v408_data = ir2[1];
            ir2[1] = (v408_data + (v400_data * v406_data));
            double v411_data = s0[41];
            double v413_data = ir2[2];
            ir2[2] = (v413_data + (v400_data * v411_data));
            double v416_data = s0[57];
            double v418_data = ir2[3];
            ir2[3] = (v418_data + (v400_data * v416_data));
            double v421_data = s0[74];
            double v423_data = ir2[4];
            ir2[4] = (v423_data + (v400_data * v421_data));
            double v426_data = s0[90];
            double v428_data = ir2[5];
            ir2[5] = (v428_data + (v400_data * v426_data));
            double v431_data = s0[107];
            double v433_data = ir2[6];
            ir2[6] = (v433_data + (v400_data * v431_data));
            double v436_data = s0[123];
            double v438_data = ir2[7];
            ir2[7] = (v438_data + (v400_data * v436_data));
          }
          if (v13_lead < 12) {
            double v444_data = r0[9];
            double v445_data = s0[9];
            double v447_data = ir2[0];
            ir2[0] = (v447_data + (v444_data * v445_data));
            double v450_data = s0[25];
            double v452_data = ir2[1];
            ir2[1] = (v452_data + (v444_data * v450_data));
            double v455_data = s0[40];
            double v457_data = ir2[2];
            ir2[2] = (v457_data + (v444_data * v455_data));
            double v460_data = s0[56];
            double v462_data = ir2[3];
            ir2[3] = (v462_data + (v444_data * v460_data));
            double v465_data = s0[75];
            double v467_data = ir2[4];
            ir2[4] = (v467_data + (v444_data * v465_data));
            double v470_data = s0[91];
            double v472_data = ir2[5];
            ir2[5] = (v472_data + (v444_data * v470_data));
            double v475_data = s0[106];
            double v477_data = ir2[6];
            ir2[6] = (v477_data + (v444_data * v475_data));
            double v480_data = s0[122];
            double v482_data = ir2[7];
            ir2[7] = (v482_data + (v444_data * v480_data));
          }
          if (v13_lead < 12) {
            double v488_data = r0[10];
            double v489_data = s0[10];
            double v491_data = ir2[0];
            ir2[0] = (v491_data + (v488_data * v489_data));
            double v494_data = s0[26];
            double v496_data = ir2[1];
            ir2[1] = (v496_data + (v488_data * v494_data));
            double v499_data = s0[43];
            double v501_data = ir2[2];
            ir2[2] = (v501_data + (v488_data * v499_data));
            double v504_data = s0[59];
            double v506_data = ir2[3];
            ir2[3] = (v506_data + (v488_data * v504_data));
            double v509_data = s0[72];
            double v511_data = ir2[4];
            ir2[4] = (v511_data + (v488_data * v509_data));
            double v514_data = s0[88];
            double v516_data = ir2[5];
            ir2[5] = (v516_data + (v488_data * v514_data));
            double v519_data = s0[105];
            double v521_data = ir2[6];
            ir2[6] = (v521_data + (v488_data * v519_data));
            double v524_data = s0[121];
            double v526_data = ir2[7];
            ir2[7] = (v526_data + (v488_data * v524_data));
          }
          if (v13_lead < 12) {
            double v532_data = r0[11];
            double v533_data = s0[11];
            double v535_data = ir2[0];
            ir2[0] = (v535_data + (v532_data * v533_data));
            double v538_data = s0[27];
            double v540_data = ir2[1];
            ir2[1] = (v540_data + (v532_data * v538_data));
            double v543_data = s0[42];
            double v545_data = ir2[2];
            ir2[2] = (v545_data + (v532_data * v543_data));
            double v548_data = s0[58];
            double v550_data = ir2[3];
            ir2[3] = (v550_data + (v532_data * v548_data));
            double v553_data = s0[73];
            double v555_data = ir2[4];
            ir2[4] = (v555_data + (v532_data * v553_data));
            double v558_data = s0[89];
            double v560_data = ir2[5];
            ir2[5] = (v560_data + (v532_data * v558_data));
            double v563_data = s0[104];
            double v565_data = ir2[6];
            ir2[6] = (v565_data + (v532_data * v563_data));
            double v568_data = s0[120];
            double v570_data = ir2[7];
            ir2[7] = (v570_data + (v532_data * v568_data));
          }
          if (v13_lead < 12) {
            double v576_data = r0[12];
            double v577_data = s0[12];
            double v579_data = ir2[0];
            ir2[0] = (v579_data + (v576_data * v577_data));
            double v582_data = s0[28];
            double v584_data = ir2[1];
            ir2[1] = (v584_data + (v576_data * v582_data));
            double v587_data = s0[45];
            double v589_data = ir2[2];
            ir2[2] = (v589_data + (v576_data * v587_data));
            double v592_data = s0[61];
            double v594_data = ir2[3];
            ir2[3] = (v594_data + (v576_data * v592_data));
            double v597_data = s0[78];
            double v599_data = ir2[4];
            ir2[4] = (v599_data + (v576_data * v597_data));
            double v602_data = s0[94];
            double v604_data = ir2[5];
            ir2[5] = (v604_data + (v576_data * v602_data));
            double v607_data = s0[111];
            double v609_data = ir2[6];
            ir2[6] = (v609_data + (v576_data * v607_data));
            double v612_data = s0[127];
            double v614_data = ir2[7];
            ir2[7] = (v614_data + (v576_data * v612_data));
          }
          if (v13_lead < 12) {
            double v620_data = r0[13];
            double v621_data = s0[13];
            double v623_data = ir2[0];
            ir2[0] = (v623_data + (v620_data * v621_data));
            double v626_data = s0[29];
            double v628_data = ir2[1];
            ir2[1] = (v628_data + (v620_data * v626_data));
            double v631_data = s0[44];
            double v633_data = ir2[2];
            ir2[2] = (v633_data + (v620_data * v631_data));
            double v636_data = s0[60];
            double v638_data = ir2[3];
            ir2[3] = (v638_data + (v620_data * v636_data));
            double v641_data = s0[79];
            double v643_data = ir2[4];
            ir2[4] = (v643_data + (v620_data * v641_data));
            double v646_data = s0[95];
            double v648_data = ir2[5];
            ir2[5] = (v648_data + (v620_data * v646_data));
            double v651_data = s0[110];
            double v653_data = ir2[6];
            ir2[6] = (v653_data + (v620_data * v651_data));
            double v656_data = s0[126];
            double v658_data = ir2[7];
            ir2[7] = (v658_data + (v620_data * v656_data));
          }
          if (v13_lead < 12) {
            double v664_data = r0[14];
            double v665_data = s0[14];
            double v667_data = ir2[0];
            ir2[0] = (v667_data + (v664_data * v665_data));
            double v670_data = s0[30];
            double v672_data = ir2[1];
            ir2[1] = (v672_data + (v664_data * v670_data));
            double v675_data = s0[47];
            double v677_data = ir2[2];
            ir2[2] = (v677_data + (v664_data * v675_data));
            double v680_data = s0[63];
            double v682_data = ir2[3];
            ir2[3] = (v682_data + (v664_data * v680_data));
            double v685_data = s0[76];
            double v687_data = ir2[4];
            ir2[4] = (v687_data + (v664_data * v685_data));
            double v690_data = s0[92];
            double v692_data = ir2[5];
            ir2[5] = (v692_data + (v664_data * v690_data));
            double v695_data = s0[109];
            double v697_data = ir2[6];
            ir2[6] = (v697_data + (v664_data * v695_data));
            double v700_data = s0[125];
            double v702_data = ir2[7];
            ir2[7] = (v702_data + (v664_data * v700_data));
          }
          if (v13_lead < 12) {
            double v708_data = r0[15];
            double v709_data = s0[15];
            double v711_data = ir2[0];
            ir2[0] = (v711_data + (v708_data * v709_data));
            double v714_data = s0[31];
            double v716_data = ir2[1];
            ir2[1] = (v716_data + (v708_data * v714_data));
            double v719_data = s0[46];
            double v721_data = ir2[2];
            ir2[2] = (v721_data + (v708_data * v719_data));
            double v724_data = s0[62];
            double v726_data = ir2[3];
            ir2[3] = (v726_data + (v708_data * v724_data));
            double v729_data = s0[77];
            double v731_data = ir2[4];
            ir2[4] = (v731_data + (v708_data * v729_data));
            double v734_data = s0[93];
            double v736_data = ir2[5];
            ir2[5] = (v736_data + (v708_data * v734_data));
            double v739_data = s0[108];
            double v741_data = ir2[6];
            ir2[6] = (v741_data + (v708_data * v739_data));
            double v744_data = s0[124];
            double v746_data = ir2[7];
            ir2[7] = (v746_data + (v708_data * v744_data));
          }
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v752_n1 = 0; v752_n1 < 8; ++v752_n1) {
              double v754_data = ir2[v752_n1];
              double v756_data = r1[v752_n1];
              r2[v752_n1] = (v756_data + v754_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v763_i1 = 0; v763_i1 < 8; ++v763_i1) {
              double v765_data = r2[v763_i1];
              glb_m0[(v13_lead + (v763_i1 * 12))] = v765_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

