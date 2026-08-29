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
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 16;
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
              float v23_data = __ldcg(&glb_m1[(v13_lead + (v15_i1 * 12))]);
              r0[v15_i1] = v23_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
              float v40_data = glb_m0[(v13_lead + (v32_i1 * 12))];
              r1[v32_i1] = v40_data;
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
          if (v13_lead < 12) {
            float v48_data = r0[0];
            float v49_data = s0[0];
            float v51_data = ir2[0];
            ir2[0] = (v51_data + (v48_data * v49_data));
            float v54_data = s0[16];
            float v56_data = ir2[1];
            ir2[1] = (v56_data + (v48_data * v54_data));
            float v59_data = s0[33];
            float v61_data = ir2[2];
            ir2[2] = (v61_data + (v48_data * v59_data));
            float v64_data = s0[49];
            float v66_data = ir2[3];
            ir2[3] = (v66_data + (v48_data * v64_data));
            float v69_data = s0[66];
            float v71_data = ir2[4];
            ir2[4] = (v71_data + (v48_data * v69_data));
            float v74_data = s0[82];
            float v76_data = ir2[5];
            ir2[5] = (v76_data + (v48_data * v74_data));
            float v79_data = s0[99];
            float v81_data = ir2[6];
            ir2[6] = (v81_data + (v48_data * v79_data));
            float v84_data = s0[115];
            float v86_data = ir2[7];
            ir2[7] = (v86_data + (v48_data * v84_data));
          }
          if (v13_lead < 12) {
            float v92_data = r0[1];
            float v93_data = s0[1];
            float v95_data = ir2[0];
            ir2[0] = (v95_data + (v92_data * v93_data));
            float v98_data = s0[17];
            float v100_data = ir2[1];
            ir2[1] = (v100_data + (v92_data * v98_data));
            float v103_data = s0[32];
            float v105_data = ir2[2];
            ir2[2] = (v105_data + (v92_data * v103_data));
            float v108_data = s0[48];
            float v110_data = ir2[3];
            ir2[3] = (v110_data + (v92_data * v108_data));
            float v113_data = s0[67];
            float v115_data = ir2[4];
            ir2[4] = (v115_data + (v92_data * v113_data));
            float v118_data = s0[83];
            float v120_data = ir2[5];
            ir2[5] = (v120_data + (v92_data * v118_data));
            float v123_data = s0[98];
            float v125_data = ir2[6];
            ir2[6] = (v125_data + (v92_data * v123_data));
            float v128_data = s0[114];
            float v130_data = ir2[7];
            ir2[7] = (v130_data + (v92_data * v128_data));
          }
          if (v13_lead < 12) {
            float v136_data = r0[2];
            float v137_data = s0[2];
            float v139_data = ir2[0];
            ir2[0] = (v139_data + (v136_data * v137_data));
            float v142_data = s0[18];
            float v144_data = ir2[1];
            ir2[1] = (v144_data + (v136_data * v142_data));
            float v147_data = s0[35];
            float v149_data = ir2[2];
            ir2[2] = (v149_data + (v136_data * v147_data));
            float v152_data = s0[51];
            float v154_data = ir2[3];
            ir2[3] = (v154_data + (v136_data * v152_data));
            float v157_data = s0[64];
            float v159_data = ir2[4];
            ir2[4] = (v159_data + (v136_data * v157_data));
            float v162_data = s0[80];
            float v164_data = ir2[5];
            ir2[5] = (v164_data + (v136_data * v162_data));
            float v167_data = s0[97];
            float v169_data = ir2[6];
            ir2[6] = (v169_data + (v136_data * v167_data));
            float v172_data = s0[113];
            float v174_data = ir2[7];
            ir2[7] = (v174_data + (v136_data * v172_data));
          }
          if (v13_lead < 12) {
            float v180_data = r0[3];
            float v181_data = s0[3];
            float v183_data = ir2[0];
            ir2[0] = (v183_data + (v180_data * v181_data));
            float v186_data = s0[19];
            float v188_data = ir2[1];
            ir2[1] = (v188_data + (v180_data * v186_data));
            float v191_data = s0[34];
            float v193_data = ir2[2];
            ir2[2] = (v193_data + (v180_data * v191_data));
            float v196_data = s0[50];
            float v198_data = ir2[3];
            ir2[3] = (v198_data + (v180_data * v196_data));
            float v201_data = s0[65];
            float v203_data = ir2[4];
            ir2[4] = (v203_data + (v180_data * v201_data));
            float v206_data = s0[81];
            float v208_data = ir2[5];
            ir2[5] = (v208_data + (v180_data * v206_data));
            float v211_data = s0[96];
            float v213_data = ir2[6];
            ir2[6] = (v213_data + (v180_data * v211_data));
            float v216_data = s0[112];
            float v218_data = ir2[7];
            ir2[7] = (v218_data + (v180_data * v216_data));
          }
          if (v13_lead < 12) {
            float v224_data = r0[4];
            float v225_data = s0[4];
            float v227_data = ir2[0];
            ir2[0] = (v227_data + (v224_data * v225_data));
            float v230_data = s0[20];
            float v232_data = ir2[1];
            ir2[1] = (v232_data + (v224_data * v230_data));
            float v235_data = s0[37];
            float v237_data = ir2[2];
            ir2[2] = (v237_data + (v224_data * v235_data));
            float v240_data = s0[53];
            float v242_data = ir2[3];
            ir2[3] = (v242_data + (v224_data * v240_data));
            float v245_data = s0[70];
            float v247_data = ir2[4];
            ir2[4] = (v247_data + (v224_data * v245_data));
            float v250_data = s0[86];
            float v252_data = ir2[5];
            ir2[5] = (v252_data + (v224_data * v250_data));
            float v255_data = s0[103];
            float v257_data = ir2[6];
            ir2[6] = (v257_data + (v224_data * v255_data));
            float v260_data = s0[119];
            float v262_data = ir2[7];
            ir2[7] = (v262_data + (v224_data * v260_data));
          }
          if (v13_lead < 12) {
            float v268_data = r0[5];
            float v269_data = s0[5];
            float v271_data = ir2[0];
            ir2[0] = (v271_data + (v268_data * v269_data));
            float v274_data = s0[21];
            float v276_data = ir2[1];
            ir2[1] = (v276_data + (v268_data * v274_data));
            float v279_data = s0[36];
            float v281_data = ir2[2];
            ir2[2] = (v281_data + (v268_data * v279_data));
            float v284_data = s0[52];
            float v286_data = ir2[3];
            ir2[3] = (v286_data + (v268_data * v284_data));
            float v289_data = s0[71];
            float v291_data = ir2[4];
            ir2[4] = (v291_data + (v268_data * v289_data));
            float v294_data = s0[87];
            float v296_data = ir2[5];
            ir2[5] = (v296_data + (v268_data * v294_data));
            float v299_data = s0[102];
            float v301_data = ir2[6];
            ir2[6] = (v301_data + (v268_data * v299_data));
            float v304_data = s0[118];
            float v306_data = ir2[7];
            ir2[7] = (v306_data + (v268_data * v304_data));
          }
          if (v13_lead < 12) {
            float v312_data = r0[6];
            float v313_data = s0[6];
            float v315_data = ir2[0];
            ir2[0] = (v315_data + (v312_data * v313_data));
            float v318_data = s0[22];
            float v320_data = ir2[1];
            ir2[1] = (v320_data + (v312_data * v318_data));
            float v323_data = s0[39];
            float v325_data = ir2[2];
            ir2[2] = (v325_data + (v312_data * v323_data));
            float v328_data = s0[55];
            float v330_data = ir2[3];
            ir2[3] = (v330_data + (v312_data * v328_data));
            float v333_data = s0[68];
            float v335_data = ir2[4];
            ir2[4] = (v335_data + (v312_data * v333_data));
            float v338_data = s0[84];
            float v340_data = ir2[5];
            ir2[5] = (v340_data + (v312_data * v338_data));
            float v343_data = s0[101];
            float v345_data = ir2[6];
            ir2[6] = (v345_data + (v312_data * v343_data));
            float v348_data = s0[117];
            float v350_data = ir2[7];
            ir2[7] = (v350_data + (v312_data * v348_data));
          }
          if (v13_lead < 12) {
            float v356_data = r0[7];
            float v357_data = s0[7];
            float v359_data = ir2[0];
            ir2[0] = (v359_data + (v356_data * v357_data));
            float v362_data = s0[23];
            float v364_data = ir2[1];
            ir2[1] = (v364_data + (v356_data * v362_data));
            float v367_data = s0[38];
            float v369_data = ir2[2];
            ir2[2] = (v369_data + (v356_data * v367_data));
            float v372_data = s0[54];
            float v374_data = ir2[3];
            ir2[3] = (v374_data + (v356_data * v372_data));
            float v377_data = s0[69];
            float v379_data = ir2[4];
            ir2[4] = (v379_data + (v356_data * v377_data));
            float v382_data = s0[85];
            float v384_data = ir2[5];
            ir2[5] = (v384_data + (v356_data * v382_data));
            float v387_data = s0[100];
            float v389_data = ir2[6];
            ir2[6] = (v389_data + (v356_data * v387_data));
            float v392_data = s0[116];
            float v394_data = ir2[7];
            ir2[7] = (v394_data + (v356_data * v392_data));
          }
          if (v13_lead < 12) {
            float v400_data = r0[8];
            float v401_data = s0[8];
            float v403_data = ir2[0];
            ir2[0] = (v403_data + (v400_data * v401_data));
            float v406_data = s0[24];
            float v408_data = ir2[1];
            ir2[1] = (v408_data + (v400_data * v406_data));
            float v411_data = s0[41];
            float v413_data = ir2[2];
            ir2[2] = (v413_data + (v400_data * v411_data));
            float v416_data = s0[57];
            float v418_data = ir2[3];
            ir2[3] = (v418_data + (v400_data * v416_data));
            float v421_data = s0[74];
            float v423_data = ir2[4];
            ir2[4] = (v423_data + (v400_data * v421_data));
            float v426_data = s0[90];
            float v428_data = ir2[5];
            ir2[5] = (v428_data + (v400_data * v426_data));
            float v431_data = s0[107];
            float v433_data = ir2[6];
            ir2[6] = (v433_data + (v400_data * v431_data));
            float v436_data = s0[123];
            float v438_data = ir2[7];
            ir2[7] = (v438_data + (v400_data * v436_data));
          }
          if (v13_lead < 12) {
            float v444_data = r0[9];
            float v445_data = s0[9];
            float v447_data = ir2[0];
            ir2[0] = (v447_data + (v444_data * v445_data));
            float v450_data = s0[25];
            float v452_data = ir2[1];
            ir2[1] = (v452_data + (v444_data * v450_data));
            float v455_data = s0[40];
            float v457_data = ir2[2];
            ir2[2] = (v457_data + (v444_data * v455_data));
            float v460_data = s0[56];
            float v462_data = ir2[3];
            ir2[3] = (v462_data + (v444_data * v460_data));
            float v465_data = s0[75];
            float v467_data = ir2[4];
            ir2[4] = (v467_data + (v444_data * v465_data));
            float v470_data = s0[91];
            float v472_data = ir2[5];
            ir2[5] = (v472_data + (v444_data * v470_data));
            float v475_data = s0[106];
            float v477_data = ir2[6];
            ir2[6] = (v477_data + (v444_data * v475_data));
            float v480_data = s0[122];
            float v482_data = ir2[7];
            ir2[7] = (v482_data + (v444_data * v480_data));
          }
          if (v13_lead < 12) {
            float v488_data = r0[10];
            float v489_data = s0[10];
            float v491_data = ir2[0];
            ir2[0] = (v491_data + (v488_data * v489_data));
            float v494_data = s0[26];
            float v496_data = ir2[1];
            ir2[1] = (v496_data + (v488_data * v494_data));
            float v499_data = s0[43];
            float v501_data = ir2[2];
            ir2[2] = (v501_data + (v488_data * v499_data));
            float v504_data = s0[59];
            float v506_data = ir2[3];
            ir2[3] = (v506_data + (v488_data * v504_data));
            float v509_data = s0[72];
            float v511_data = ir2[4];
            ir2[4] = (v511_data + (v488_data * v509_data));
            float v514_data = s0[88];
            float v516_data = ir2[5];
            ir2[5] = (v516_data + (v488_data * v514_data));
            float v519_data = s0[105];
            float v521_data = ir2[6];
            ir2[6] = (v521_data + (v488_data * v519_data));
            float v524_data = s0[121];
            float v526_data = ir2[7];
            ir2[7] = (v526_data + (v488_data * v524_data));
          }
          if (v13_lead < 12) {
            float v532_data = r0[11];
            float v533_data = s0[11];
            float v535_data = ir2[0];
            ir2[0] = (v535_data + (v532_data * v533_data));
            float v538_data = s0[27];
            float v540_data = ir2[1];
            ir2[1] = (v540_data + (v532_data * v538_data));
            float v543_data = s0[42];
            float v545_data = ir2[2];
            ir2[2] = (v545_data + (v532_data * v543_data));
            float v548_data = s0[58];
            float v550_data = ir2[3];
            ir2[3] = (v550_data + (v532_data * v548_data));
            float v553_data = s0[73];
            float v555_data = ir2[4];
            ir2[4] = (v555_data + (v532_data * v553_data));
            float v558_data = s0[89];
            float v560_data = ir2[5];
            ir2[5] = (v560_data + (v532_data * v558_data));
            float v563_data = s0[104];
            float v565_data = ir2[6];
            ir2[6] = (v565_data + (v532_data * v563_data));
            float v568_data = s0[120];
            float v570_data = ir2[7];
            ir2[7] = (v570_data + (v532_data * v568_data));
          }
          if (v13_lead < 12) {
            float v576_data = r0[12];
            float v577_data = s0[12];
            float v579_data = ir2[0];
            ir2[0] = (v579_data + (v576_data * v577_data));
            float v582_data = s0[28];
            float v584_data = ir2[1];
            ir2[1] = (v584_data + (v576_data * v582_data));
            float v587_data = s0[45];
            float v589_data = ir2[2];
            ir2[2] = (v589_data + (v576_data * v587_data));
            float v592_data = s0[61];
            float v594_data = ir2[3];
            ir2[3] = (v594_data + (v576_data * v592_data));
            float v597_data = s0[78];
            float v599_data = ir2[4];
            ir2[4] = (v599_data + (v576_data * v597_data));
            float v602_data = s0[94];
            float v604_data = ir2[5];
            ir2[5] = (v604_data + (v576_data * v602_data));
            float v607_data = s0[111];
            float v609_data = ir2[6];
            ir2[6] = (v609_data + (v576_data * v607_data));
            float v612_data = s0[127];
            float v614_data = ir2[7];
            ir2[7] = (v614_data + (v576_data * v612_data));
          }
          if (v13_lead < 12) {
            float v620_data = r0[13];
            float v621_data = s0[13];
            float v623_data = ir2[0];
            ir2[0] = (v623_data + (v620_data * v621_data));
            float v626_data = s0[29];
            float v628_data = ir2[1];
            ir2[1] = (v628_data + (v620_data * v626_data));
            float v631_data = s0[44];
            float v633_data = ir2[2];
            ir2[2] = (v633_data + (v620_data * v631_data));
            float v636_data = s0[60];
            float v638_data = ir2[3];
            ir2[3] = (v638_data + (v620_data * v636_data));
            float v641_data = s0[79];
            float v643_data = ir2[4];
            ir2[4] = (v643_data + (v620_data * v641_data));
            float v646_data = s0[95];
            float v648_data = ir2[5];
            ir2[5] = (v648_data + (v620_data * v646_data));
            float v651_data = s0[110];
            float v653_data = ir2[6];
            ir2[6] = (v653_data + (v620_data * v651_data));
            float v656_data = s0[126];
            float v658_data = ir2[7];
            ir2[7] = (v658_data + (v620_data * v656_data));
          }
          if (v13_lead < 12) {
            float v664_data = r0[14];
            float v665_data = s0[14];
            float v667_data = ir2[0];
            ir2[0] = (v667_data + (v664_data * v665_data));
            float v670_data = s0[30];
            float v672_data = ir2[1];
            ir2[1] = (v672_data + (v664_data * v670_data));
            float v675_data = s0[47];
            float v677_data = ir2[2];
            ir2[2] = (v677_data + (v664_data * v675_data));
            float v680_data = s0[63];
            float v682_data = ir2[3];
            ir2[3] = (v682_data + (v664_data * v680_data));
            float v685_data = s0[76];
            float v687_data = ir2[4];
            ir2[4] = (v687_data + (v664_data * v685_data));
            float v690_data = s0[92];
            float v692_data = ir2[5];
            ir2[5] = (v692_data + (v664_data * v690_data));
            float v695_data = s0[109];
            float v697_data = ir2[6];
            ir2[6] = (v697_data + (v664_data * v695_data));
            float v700_data = s0[125];
            float v702_data = ir2[7];
            ir2[7] = (v702_data + (v664_data * v700_data));
          }
          if (v13_lead < 12) {
            float v708_data = r0[15];
            float v709_data = s0[15];
            float v711_data = ir2[0];
            ir2[0] = (v711_data + (v708_data * v709_data));
            float v714_data = s0[31];
            float v716_data = ir2[1];
            ir2[1] = (v716_data + (v708_data * v714_data));
            float v719_data = s0[46];
            float v721_data = ir2[2];
            ir2[2] = (v721_data + (v708_data * v719_data));
            float v724_data = s0[62];
            float v726_data = ir2[3];
            ir2[3] = (v726_data + (v708_data * v724_data));
            float v729_data = s0[77];
            float v731_data = ir2[4];
            ir2[4] = (v731_data + (v708_data * v729_data));
            float v734_data = s0[93];
            float v736_data = ir2[5];
            ir2[5] = (v736_data + (v708_data * v734_data));
            float v739_data = s0[108];
            float v741_data = ir2[6];
            ir2[6] = (v741_data + (v708_data * v739_data));
            float v744_data = s0[124];
            float v746_data = ir2[7];
            ir2[7] = (v746_data + (v708_data * v744_data));
          }
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v752_n1 = 0; v752_n1 < 8; ++v752_n1) {
              float v754_data = ir2[v752_n1];
              float v756_data = r1[v752_n1];
              r2[v752_n1] = (v756_data + v754_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v763_i1 = 0; v763_i1 < 8; ++v763_i1) {
              float v765_data = r2[v763_i1];
              glb_m0[(v13_lead + (v763_i1 * 12))] = v765_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

