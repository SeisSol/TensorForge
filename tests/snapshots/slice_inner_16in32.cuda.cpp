// === base name ===
kernel_87f2838a59

// === header ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_87f2838a59, block.x * block.y * block.z, 2304 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_87f2838a59, cudaFuncAttributeMaxDynamicSharedMemorySize, 2304 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_87f2838a59<<<grid,block,2304 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×8(16×8) {0..16}×{0..8} strided
    // m1 32×32(32×32) {0..32}×{0..32} strided
    // m2 16×8(16×8) {0..16}×{0..8} strided
    // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
          float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v2_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v3_i0 = 0; v3_i0 < 1; ++v3_i0) {
            int32_t v8_lead = v3_i0 * 16;
            int32_t v10_off = (v2_lead + v8_lead) + 8;
            int32_t v18_off = (v2_lead + v8_lead) + 8;
            #pragma unroll
            for (int32_t v4_i1 = 8; v4_i1 < 24; ++v4_i1) {
              int32_t v11_a = v4_i1 * 32;
              int32_t v12_a = v10_off + v11_a;
              float v21_data = __ldcg(&glb_m1[(v18_off + v11_a)]);
              int32_t v23_a = v3_i0 + (v4_i1 - 8);
              r0[v23_a] = v21_data;
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
            // [(0, 16), (0, 8)] [(0, 16)]
            float ir1[8]{};
            float v27_data = r0[0];
            float v28_data = s0[0];
            float v30_data = ir1[0];
            ir1[0] = (v30_data + (v27_data * v28_data));
            float v33_data = s0[16];
            float v35_data = ir1[1];
            ir1[1] = (v35_data + (v27_data * v33_data));
            float v38_data = s0[32];
            float v40_data = ir1[2];
            ir1[2] = (v40_data + (v27_data * v38_data));
            float v43_data = s0[48];
            float v45_data = ir1[3];
            ir1[3] = (v45_data + (v27_data * v43_data));
            float v48_data = s0[64];
            float v50_data = ir1[4];
            ir1[4] = (v50_data + (v27_data * v48_data));
            float v53_data = s0[80];
            float v55_data = ir1[5];
            ir1[5] = (v55_data + (v27_data * v53_data));
            float v58_data = s0[96];
            float v60_data = ir1[6];
            ir1[6] = (v60_data + (v27_data * v58_data));
            float v63_data = s0[112];
            float v65_data = ir1[7];
            ir1[7] = (v65_data + (v27_data * v63_data));
            float v70_data = r0[1];
            float v71_data = s0[1];
            float v73_data = ir1[0];
            ir1[0] = (v73_data + (v70_data * v71_data));
            float v76_data = s0[17];
            float v78_data = ir1[1];
            ir1[1] = (v78_data + (v70_data * v76_data));
            float v81_data = s0[33];
            float v83_data = ir1[2];
            ir1[2] = (v83_data + (v70_data * v81_data));
            float v86_data = s0[49];
            float v88_data = ir1[3];
            ir1[3] = (v88_data + (v70_data * v86_data));
            float v91_data = s0[65];
            float v93_data = ir1[4];
            ir1[4] = (v93_data + (v70_data * v91_data));
            float v96_data = s0[81];
            float v98_data = ir1[5];
            ir1[5] = (v98_data + (v70_data * v96_data));
            float v101_data = s0[97];
            float v103_data = ir1[6];
            ir1[6] = (v103_data + (v70_data * v101_data));
            float v106_data = s0[113];
            float v108_data = ir1[7];
            ir1[7] = (v108_data + (v70_data * v106_data));
            float v113_data = r0[2];
            float v114_data = s0[2];
            float v116_data = ir1[0];
            ir1[0] = (v116_data + (v113_data * v114_data));
            float v119_data = s0[18];
            float v121_data = ir1[1];
            ir1[1] = (v121_data + (v113_data * v119_data));
            float v124_data = s0[34];
            float v126_data = ir1[2];
            ir1[2] = (v126_data + (v113_data * v124_data));
            float v129_data = s0[50];
            float v131_data = ir1[3];
            ir1[3] = (v131_data + (v113_data * v129_data));
            float v134_data = s0[66];
            float v136_data = ir1[4];
            ir1[4] = (v136_data + (v113_data * v134_data));
            float v139_data = s0[82];
            float v141_data = ir1[5];
            ir1[5] = (v141_data + (v113_data * v139_data));
            float v144_data = s0[98];
            float v146_data = ir1[6];
            ir1[6] = (v146_data + (v113_data * v144_data));
            float v149_data = s0[114];
            float v151_data = ir1[7];
            ir1[7] = (v151_data + (v113_data * v149_data));
            float v156_data = r0[3];
            float v157_data = s0[3];
            float v159_data = ir1[0];
            ir1[0] = (v159_data + (v156_data * v157_data));
            float v162_data = s0[19];
            float v164_data = ir1[1];
            ir1[1] = (v164_data + (v156_data * v162_data));
            float v167_data = s0[35];
            float v169_data = ir1[2];
            ir1[2] = (v169_data + (v156_data * v167_data));
            float v172_data = s0[51];
            float v174_data = ir1[3];
            ir1[3] = (v174_data + (v156_data * v172_data));
            float v177_data = s0[67];
            float v179_data = ir1[4];
            ir1[4] = (v179_data + (v156_data * v177_data));
            float v182_data = s0[83];
            float v184_data = ir1[5];
            ir1[5] = (v184_data + (v156_data * v182_data));
            float v187_data = s0[99];
            float v189_data = ir1[6];
            ir1[6] = (v189_data + (v156_data * v187_data));
            float v192_data = s0[115];
            float v194_data = ir1[7];
            ir1[7] = (v194_data + (v156_data * v192_data));
            float v199_data = r0[4];
            float v200_data = s0[4];
            float v202_data = ir1[0];
            ir1[0] = (v202_data + (v199_data * v200_data));
            float v205_data = s0[20];
            float v207_data = ir1[1];
            ir1[1] = (v207_data + (v199_data * v205_data));
            float v210_data = s0[36];
            float v212_data = ir1[2];
            ir1[2] = (v212_data + (v199_data * v210_data));
            float v215_data = s0[52];
            float v217_data = ir1[3];
            ir1[3] = (v217_data + (v199_data * v215_data));
            float v220_data = s0[68];
            float v222_data = ir1[4];
            ir1[4] = (v222_data + (v199_data * v220_data));
            float v225_data = s0[84];
            float v227_data = ir1[5];
            ir1[5] = (v227_data + (v199_data * v225_data));
            float v230_data = s0[100];
            float v232_data = ir1[6];
            ir1[6] = (v232_data + (v199_data * v230_data));
            float v235_data = s0[116];
            float v237_data = ir1[7];
            ir1[7] = (v237_data + (v199_data * v235_data));
            float v242_data = r0[5];
            float v243_data = s0[5];
            float v245_data = ir1[0];
            ir1[0] = (v245_data + (v242_data * v243_data));
            float v248_data = s0[21];
            float v250_data = ir1[1];
            ir1[1] = (v250_data + (v242_data * v248_data));
            float v253_data = s0[37];
            float v255_data = ir1[2];
            ir1[2] = (v255_data + (v242_data * v253_data));
            float v258_data = s0[53];
            float v260_data = ir1[3];
            ir1[3] = (v260_data + (v242_data * v258_data));
            float v263_data = s0[69];
            float v265_data = ir1[4];
            ir1[4] = (v265_data + (v242_data * v263_data));
            float v268_data = s0[85];
            float v270_data = ir1[5];
            ir1[5] = (v270_data + (v242_data * v268_data));
            float v273_data = s0[101];
            float v275_data = ir1[6];
            ir1[6] = (v275_data + (v242_data * v273_data));
            float v278_data = s0[117];
            float v280_data = ir1[7];
            ir1[7] = (v280_data + (v242_data * v278_data));
            float v285_data = r0[6];
            float v286_data = s0[6];
            float v288_data = ir1[0];
            ir1[0] = (v288_data + (v285_data * v286_data));
            float v291_data = s0[22];
            float v293_data = ir1[1];
            ir1[1] = (v293_data + (v285_data * v291_data));
            float v296_data = s0[38];
            float v298_data = ir1[2];
            ir1[2] = (v298_data + (v285_data * v296_data));
            float v301_data = s0[54];
            float v303_data = ir1[3];
            ir1[3] = (v303_data + (v285_data * v301_data));
            float v306_data = s0[70];
            float v308_data = ir1[4];
            ir1[4] = (v308_data + (v285_data * v306_data));
            float v311_data = s0[86];
            float v313_data = ir1[5];
            ir1[5] = (v313_data + (v285_data * v311_data));
            float v316_data = s0[102];
            float v318_data = ir1[6];
            ir1[6] = (v318_data + (v285_data * v316_data));
            float v321_data = s0[118];
            float v323_data = ir1[7];
            ir1[7] = (v323_data + (v285_data * v321_data));
            float v328_data = r0[7];
            float v329_data = s0[7];
            float v331_data = ir1[0];
            ir1[0] = (v331_data + (v328_data * v329_data));
            float v334_data = s0[23];
            float v336_data = ir1[1];
            ir1[1] = (v336_data + (v328_data * v334_data));
            float v339_data = s0[39];
            float v341_data = ir1[2];
            ir1[2] = (v341_data + (v328_data * v339_data));
            float v344_data = s0[55];
            float v346_data = ir1[3];
            ir1[3] = (v346_data + (v328_data * v344_data));
            float v349_data = s0[71];
            float v351_data = ir1[4];
            ir1[4] = (v351_data + (v328_data * v349_data));
            float v354_data = s0[87];
            float v356_data = ir1[5];
            ir1[5] = (v356_data + (v328_data * v354_data));
            float v359_data = s0[103];
            float v361_data = ir1[6];
            ir1[6] = (v361_data + (v328_data * v359_data));
            float v364_data = s0[119];
            float v366_data = ir1[7];
            ir1[7] = (v366_data + (v328_data * v364_data));
            float v371_data = r0[8];
            float v372_data = s0[8];
            float v374_data = ir1[0];
            ir1[0] = (v374_data + (v371_data * v372_data));
            float v377_data = s0[24];
            float v379_data = ir1[1];
            ir1[1] = (v379_data + (v371_data * v377_data));
            float v382_data = s0[40];
            float v384_data = ir1[2];
            ir1[2] = (v384_data + (v371_data * v382_data));
            float v387_data = s0[56];
            float v389_data = ir1[3];
            ir1[3] = (v389_data + (v371_data * v387_data));
            float v392_data = s0[72];
            float v394_data = ir1[4];
            ir1[4] = (v394_data + (v371_data * v392_data));
            float v397_data = s0[88];
            float v399_data = ir1[5];
            ir1[5] = (v399_data + (v371_data * v397_data));
            float v402_data = s0[104];
            float v404_data = ir1[6];
            ir1[6] = (v404_data + (v371_data * v402_data));
            float v407_data = s0[120];
            float v409_data = ir1[7];
            ir1[7] = (v409_data + (v371_data * v407_data));
            float v414_data = r0[9];
            float v415_data = s0[9];
            float v417_data = ir1[0];
            ir1[0] = (v417_data + (v414_data * v415_data));
            float v420_data = s0[25];
            float v422_data = ir1[1];
            ir1[1] = (v422_data + (v414_data * v420_data));
            float v425_data = s0[41];
            float v427_data = ir1[2];
            ir1[2] = (v427_data + (v414_data * v425_data));
            float v430_data = s0[57];
            float v432_data = ir1[3];
            ir1[3] = (v432_data + (v414_data * v430_data));
            float v435_data = s0[73];
            float v437_data = ir1[4];
            ir1[4] = (v437_data + (v414_data * v435_data));
            float v440_data = s0[89];
            float v442_data = ir1[5];
            ir1[5] = (v442_data + (v414_data * v440_data));
            float v445_data = s0[105];
            float v447_data = ir1[6];
            ir1[6] = (v447_data + (v414_data * v445_data));
            float v450_data = s0[121];
            float v452_data = ir1[7];
            ir1[7] = (v452_data + (v414_data * v450_data));
            float v457_data = r0[10];
            float v458_data = s0[10];
            float v460_data = ir1[0];
            ir1[0] = (v460_data + (v457_data * v458_data));
            float v463_data = s0[26];
            float v465_data = ir1[1];
            ir1[1] = (v465_data + (v457_data * v463_data));
            float v468_data = s0[42];
            float v470_data = ir1[2];
            ir1[2] = (v470_data + (v457_data * v468_data));
            float v473_data = s0[58];
            float v475_data = ir1[3];
            ir1[3] = (v475_data + (v457_data * v473_data));
            float v478_data = s0[74];
            float v480_data = ir1[4];
            ir1[4] = (v480_data + (v457_data * v478_data));
            float v483_data = s0[90];
            float v485_data = ir1[5];
            ir1[5] = (v485_data + (v457_data * v483_data));
            float v488_data = s0[106];
            float v490_data = ir1[6];
            ir1[6] = (v490_data + (v457_data * v488_data));
            float v493_data = s0[122];
            float v495_data = ir1[7];
            ir1[7] = (v495_data + (v457_data * v493_data));
            float v500_data = r0[11];
            float v501_data = s0[11];
            float v503_data = ir1[0];
            ir1[0] = (v503_data + (v500_data * v501_data));
            float v506_data = s0[27];
            float v508_data = ir1[1];
            ir1[1] = (v508_data + (v500_data * v506_data));
            float v511_data = s0[43];
            float v513_data = ir1[2];
            ir1[2] = (v513_data + (v500_data * v511_data));
            float v516_data = s0[59];
            float v518_data = ir1[3];
            ir1[3] = (v518_data + (v500_data * v516_data));
            float v521_data = s0[75];
            float v523_data = ir1[4];
            ir1[4] = (v523_data + (v500_data * v521_data));
            float v526_data = s0[91];
            float v528_data = ir1[5];
            ir1[5] = (v528_data + (v500_data * v526_data));
            float v531_data = s0[107];
            float v533_data = ir1[6];
            ir1[6] = (v533_data + (v500_data * v531_data));
            float v536_data = s0[123];
            float v538_data = ir1[7];
            ir1[7] = (v538_data + (v500_data * v536_data));
            float v543_data = r0[12];
            float v544_data = s0[12];
            float v546_data = ir1[0];
            ir1[0] = (v546_data + (v543_data * v544_data));
            float v549_data = s0[28];
            float v551_data = ir1[1];
            ir1[1] = (v551_data + (v543_data * v549_data));
            float v554_data = s0[44];
            float v556_data = ir1[2];
            ir1[2] = (v556_data + (v543_data * v554_data));
            float v559_data = s0[60];
            float v561_data = ir1[3];
            ir1[3] = (v561_data + (v543_data * v559_data));
            float v564_data = s0[76];
            float v566_data = ir1[4];
            ir1[4] = (v566_data + (v543_data * v564_data));
            float v569_data = s0[92];
            float v571_data = ir1[5];
            ir1[5] = (v571_data + (v543_data * v569_data));
            float v574_data = s0[108];
            float v576_data = ir1[6];
            ir1[6] = (v576_data + (v543_data * v574_data));
            float v579_data = s0[124];
            float v581_data = ir1[7];
            ir1[7] = (v581_data + (v543_data * v579_data));
            float v586_data = r0[13];
            float v587_data = s0[13];
            float v589_data = ir1[0];
            ir1[0] = (v589_data + (v586_data * v587_data));
            float v592_data = s0[29];
            float v594_data = ir1[1];
            ir1[1] = (v594_data + (v586_data * v592_data));
            float v597_data = s0[45];
            float v599_data = ir1[2];
            ir1[2] = (v599_data + (v586_data * v597_data));
            float v602_data = s0[61];
            float v604_data = ir1[3];
            ir1[3] = (v604_data + (v586_data * v602_data));
            float v607_data = s0[77];
            float v609_data = ir1[4];
            ir1[4] = (v609_data + (v586_data * v607_data));
            float v612_data = s0[93];
            float v614_data = ir1[5];
            ir1[5] = (v614_data + (v586_data * v612_data));
            float v617_data = s0[109];
            float v619_data = ir1[6];
            ir1[6] = (v619_data + (v586_data * v617_data));
            float v622_data = s0[125];
            float v624_data = ir1[7];
            ir1[7] = (v624_data + (v586_data * v622_data));
            float v629_data = r0[14];
            float v630_data = s0[14];
            float v632_data = ir1[0];
            ir1[0] = (v632_data + (v629_data * v630_data));
            float v635_data = s0[30];
            float v637_data = ir1[1];
            ir1[1] = (v637_data + (v629_data * v635_data));
            float v640_data = s0[46];
            float v642_data = ir1[2];
            ir1[2] = (v642_data + (v629_data * v640_data));
            float v645_data = s0[62];
            float v647_data = ir1[3];
            ir1[3] = (v647_data + (v629_data * v645_data));
            float v650_data = s0[78];
            float v652_data = ir1[4];
            ir1[4] = (v652_data + (v629_data * v650_data));
            float v655_data = s0[94];
            float v657_data = ir1[5];
            ir1[5] = (v657_data + (v629_data * v655_data));
            float v660_data = s0[110];
            float v662_data = ir1[6];
            ir1[6] = (v662_data + (v629_data * v660_data));
            float v665_data = s0[126];
            float v667_data = ir1[7];
            ir1[7] = (v667_data + (v629_data * v665_data));
            float v672_data = r0[15];
            float v673_data = s0[15];
            float v675_data = ir1[0];
            ir1[0] = (v675_data + (v672_data * v673_data));
            float v678_data = s0[31];
            float v680_data = ir1[1];
            ir1[1] = (v680_data + (v672_data * v678_data));
            float v683_data = s0[47];
            float v685_data = ir1[2];
            ir1[2] = (v685_data + (v672_data * v683_data));
            float v688_data = s0[63];
            float v690_data = ir1[3];
            ir1[3] = (v690_data + (v672_data * v688_data));
            float v693_data = s0[79];
            float v695_data = ir1[4];
            ir1[4] = (v695_data + (v672_data * v693_data));
            float v698_data = s0[95];
            float v700_data = ir1[5];
            ir1[5] = (v700_data + (v672_data * v698_data));
            float v703_data = s0[111];
            float v705_data = ir1[6];
            ir1[6] = (v705_data + (v672_data * v703_data));
            float v708_data = s0[127];
            float v710_data = ir1[7];
            ir1[7] = (v710_data + (v672_data * v708_data));
            #pragma unroll
            for (int32_t v715_n0 = 0; v715_n0 < 1; ++v715_n0) {
              #pragma unroll
              for (int32_t v716_n1 = 0; v716_n1 < 8; ++v716_n1) {
                int32_t v717_a = v715_n0 + v716_n1;
                int32_t v718_a = v715_n0 + v716_n1;
                float v719_data = ir1[v718_a];
                int32_t v720_a = v715_n0 + v716_n1;
                r1[v718_a] = v719_data;
              }
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v725_i0 = 0; v725_i0 < 1; ++v725_i0) {
            int32_t v734_lead = v2_lead + (v725_i0 * 16);
            #pragma unroll
            for (int32_t v726_i1 = 0; v726_i1 < 8; ++v726_i1) {
              int32_t v727_a = v725_i0 + v726_i1;
              float v729_data = r1[(v725_i0 + v726_i1)];
              int32_t v736_a = v734_lead + (v726_i1 * 16);
              glb_m0[v736_a] = v729_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

